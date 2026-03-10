import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch_geometric.nn import DenseGCNConv
from torch_geometric.nn.conv import MessagePassing

def _rbf(D, D_min=0., D_max=6., D_count=9, device='cpu'):
    D_mu = torch.linspace(D_min, D_max, D_count).to(device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)
    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF

def gnn_norm(x, norm):
    batch_size, num_nodes, num_channels = x.size()
    x = x.view(-1, num_channels)
    x = norm(x)
    x = x.view(batch_size, num_nodes, num_channels)
    return x

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, drop_rate):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.Mish(),
            nn.Dropout(drop_rate),
        )

    def forward(self, x):
        return self.mlp(x)

class FC(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layer, drop_rate, output_dim):
        super(FC, self).__init__()
        self.predict = nn.ModuleList()
        self.predict.append(MLP(input_dim, hidden_dim, drop_rate))
        for _ in range(num_layer - 2):
            self.predict.append(MLP(hidden_dim, hidden_dim, drop_rate))
        self.predict.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, h):
        for layer in self.predict:
            h = layer(h)
        return h

class HIL(MessagePassing):
    def __init__(self, input_dim, output_dim, drop_rate, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(HIL, self).__init__(**kwargs)
        self.mlp_coord = MLP(9, input_dim, 0.0)
        self.out = MLP(input_dim, output_dim, drop_rate)

    def message(self, x_j, x_i, radial, index):
        return x_j * radial

    def forward(self, x, data, edge_index):
        res = x
        pos, size = data.pos, None
        row, col = edge_index
        coord_diff = pos[row] - pos[col]
        dist = torch.norm(coord_diff, p=2, dim=-1)
        radial = self.mlp_coord(_rbf(dist, device=x.device))
        x = self.propagate(edge_index=edge_index, x=x, radial=radial, size=size)
        x = self.out(x) + res
        return x

class GeoBlock(nn.Module):
    def __init__(self, input_dim, output_dim, drop_rate):
        super(GeoBlock, self).__init__()
        self.gconv_intra = HIL(input_dim, output_dim, drop_rate)
        self.gconv_inter = HIL(input_dim, output_dim, drop_rate)

    def forward(self, x, data):
        x_intra = self.gconv_intra(x, data, data.edge_index_intra)
        x_inter = self.gconv_inter(x, data, data.edge_index_inter)
        x = (x_intra + x_inter) / 2
        return x


class DiffPool(nn.Module):
    def __init__(self, input_dim, output_dim, max_num, red_node, edge, drop_rate):
        super().__init__()
        self.max_num = max_num
        self.red_node = red_node
        self.edge = edge
        self.gnn_p = DenseGCNConv(input_dim, red_node, improved=True, bias=True)
        self.gnn_p_norm = nn.Sequential(nn.BatchNorm1d(red_node), nn.Mish())
        self.gnn_e = DenseGCNConv(input_dim, output_dim, improved=True, bias=True)
        self.gnn_e_norm = nn.Sequential(nn.BatchNorm1d(output_dim), nn.Mish())
        self.out = nn.Linear(output_dim, output_dim)
        self.out_norm = nn.Sequential(nn.BatchNorm1d(output_dim))

    def pooling(self, x, adj, s, mask=None):
        batch_size, num_nodes, _ = x.size()
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        s = s.unsqueeze(0) if s.dim() == 2 else s
        s = F.softmax(s, dim=-1)
        if mask is not None:
            mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
            x, s = x * mask, s * mask
        x = torch.matmul(s.transpose(1, 2), x)
        adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)
        return x, adj, s

    def set_edge_index(self, data, edge):
        switch = {
            "intra": data.edge_index_intra,
            "inter": data.edge_index_inter,
            "intra_lig": data.edge_index_intra_lig,
            "intra_pro": data.edge_index_intra_pro,
        }
        data.edge_index = switch.get(edge, None)

    def forward(self, x, data):
        self.set_edge_index(data, self.edge)
        adj = to_dense_adj(data.edge_index, data.batch, max_num_nodes=self.max_num)
        x, mask = to_dense_batch(x, data.batch, fill_value=0, max_num_nodes=self.max_num)
        s = gnn_norm(self.gnn_p(x, adj, mask), self.gnn_p_norm)
        x, adj, s = self.pooling(x, adj, s, mask)
        x = gnn_norm(self.gnn_e(x, adj), self.gnn_e_norm)
        x = gnn_norm(self.out(x), self.out_norm)
        return x, s

class AttentionBlock(nn.Module):
    def __init__(self, hidden_dim, heads, drop_rate):
        super().__init__()
        self.heads = heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // heads
        self.W_Q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_K = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_V = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_O = MLP(hidden_dim, hidden_dim, drop_rate)

    def forward(self, q, k, v):
        res = q.sum(dim=1)
        batch_size, seqlen_q, _ = q.shape
        _, seqlen_k, _ = k.shape
        Q = self.W_Q(q)
        K = self.W_K(k)
        V = self.W_V(v)
        Q = Q.view(batch_size, seqlen_q, self.heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seqlen_k, self.heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seqlen_k, self.heads, self.head_dim).transpose(1, 2)
        energy = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention = torch.softmax(energy, dim=-1)
        x = torch.matmul(attention, V)
        x = x.transpose(1, 2).contiguous().view(batch_size, seqlen_q, self.hidden_dim)
        x = x.sum(dim=1)
        x = self.W_O(x) + res
        return x, attention

class CrossAttentionAtomChannel(nn.Module):

    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.ligand_preprocessor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.protein_preprocessor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.lig2pro_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.pro2lig_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.ligand_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.protein_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.ligand_norm = nn.LayerNorm(hidden_dim)
        self.protein_norm = nn.LayerNorm(hidden_dim)

        self.ligand_aggregator = nn.Linear(hidden_dim, 1)
        self.protein_aggregator = nn.Linear(hidden_dim, 1)

    def forward(self, atom_embeddings, data):
        batch_representations = []
        interaction_info = {
            'ligand_aggregation_weights': [],
            'protein_aggregation_weights': [],
        }

        for batch_idx in range(data.batch.max().item() + 1):
            batch_mask = (data.batch == batch_idx)
            batch_atoms = atom_embeddings[batch_mask]
            batch_split = data.split[batch_mask]

            ligand_mask = (batch_split == 0)
            protein_mask = (batch_split == 1)

            ligand_atoms = batch_atoms[ligand_mask]
            protein_atoms = batch_atoms[protein_mask]

            if ligand_atoms.size(0) == 0 or protein_atoms.size(0) == 0:
                graph_representation = torch.zeros(self.hidden_dim, device=atom_embeddings.device)
                batch_representations.append(graph_representation)
                continue

            ligand_features = self.ligand_preprocessor(ligand_atoms)
            protein_features = self.protein_preprocessor(protein_atoms)

            ligand_features = ligand_features.unsqueeze(0)
            protein_features = protein_features.unsqueeze(0)

            ligand_enhanced, _ = self.lig2pro_attention(
                query=ligand_features,
                key=protein_features,
                value=protein_features
            )

            protein_enhanced, _ = self.pro2lig_attention(
                query=protein_features,
                key=ligand_features,
                value=ligand_features
            )

            ligand_enhanced = ligand_enhanced.squeeze(0)
            protein_enhanced = protein_enhanced.squeeze(0)

            ligand_fused = self.ligand_fusion(torch.cat([ligand_atoms, ligand_enhanced], dim=-1))
            protein_fused = self.protein_fusion(torch.cat([protein_atoms, protein_enhanced], dim=-1))

            ligand_fused = self.ligand_norm(ligand_fused)
            protein_fused = self.protein_norm(protein_fused)

            ligand_attn_weights = torch.softmax(self.ligand_aggregator(ligand_fused), dim=0)
            protein_attn_weights = torch.softmax(self.protein_aggregator(protein_fused), dim=0)

            ligand_representation = torch.sum(ligand_fused * ligand_attn_weights, dim=0)
            protein_representation = torch.sum(protein_fused * protein_attn_weights, dim=0)

            combined_representation = torch.cat([ligand_representation, protein_representation], dim=-1)
            graph_representation = torch.mean(combined_representation.view(2, -1), dim=0)

            batch_representations.append(graph_representation)

            interaction_info['ligand_aggregation_weights'].append(ligand_attn_weights.detach().cpu())
            interaction_info['protein_aggregation_weights'].append(protein_attn_weights.detach().cpu())

        z_atoms = torch.stack(batch_representations, dim=0)

        return z_atoms, interaction_info

class RobustGatingFusionModule(nn.Module):

    def __init__(self, hidden_dim, temperature=2.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.temperature = temperature

        self.cluster_norm = nn.LayerNorm(hidden_dim)
        self.atom_norm = nn.LayerNorm(hidden_dim)

        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.gate_network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        if hasattr(self.gate_network[-1], 'bias') and self.gate_network[-1].bias is not None:
            nn.init.constant_(self.gate_network[-1].bias, 0.0)

    def forward(self, z_cluster, z_atoms, epoch=None):
        z_cluster_norm = self.cluster_norm(z_cluster)
        z_atoms_norm = self.atom_norm(z_atoms)

        combined_features = torch.cat([z_cluster_norm, z_atoms_norm], dim=-1)
        gate_logits = self.gate_network(combined_features)

        if epoch is not None and epoch < 30:
            gate_weights = torch.full_like(gate_logits, 0.5)
        elif epoch is not None and epoch < 80:
            alpha = (epoch - 30) / 50
            forced_weight = 0.5
            learned_weight = torch.sigmoid(gate_logits / self.temperature)
            gate_weights = (1 - alpha) * forced_weight + alpha * learned_weight
        else:
            gate_weights = torch.sigmoid(gate_logits / self.temperature)

        z_fused = gate_weights * z_cluster_norm + (1 - gate_weights) * z_atoms_norm

        return z_fused, gate_weights

    def compute_gating_balance_loss(self, gate_weights, epoch=None):
        deviation_loss = torch.mean((gate_weights - 0.5) ** 2)

        extreme_penalty = torch.mean(
            torch.exp(5 * torch.clamp(gate_weights - 0.8, min=0)) +
            torch.exp(5 * torch.clamp(0.2 - gate_weights, min=0))
        )

        gate_std = torch.std(gate_weights)
        std_loss = torch.clamp(0.1 - gate_std, min=0)

        if epoch is not None:
            if epoch < 30:
                weight = 0.05
            elif epoch < 80:
                weight = 0.15
            elif epoch < 130:
                weight = 0.25
            else:
                weight = 0.2
        else:
            weight = 0.2

        balance_loss = weight * (deviation_loss + extreme_penalty + std_loss)

        return balance_loss

    def get_training_stage_info(self, epoch):
        if epoch < 30:
            stage = "forced"
            description = f"forced balance ({epoch}/30) - weight: 0.05"
        elif epoch < 80:
            stage = "progressive"
            progress = (epoch - 30) / 50 * 100
            description = f"progressive release ({progress:.1f}%) - weight: 0.15"
        elif epoch < 130:
            stage = "free_early"
            progress = (epoch - 80) / 50 * 100
            description = f"free gating early ({progress:.1f}%) - weight: 0.25"
        else:
            stage = "free_late"
            description = f"free gating late (epoch {epoch}) - weight: 0.2"

        return {
            'stage': stage,
            'description': description,
            'epoch': epoch
        }

    def get_gating_info(self):
        return {
            'module_type': 'RobustGatingFusionModule',
            'temperature': self.temperature,
            'has_layer_norm': True,
            'gate_network_layers': len([m for m in self.gate_network if isinstance(m, nn.Linear)]),
        }

class DualStreamGeoHDS(nn.Module):

    def __init__(self, node_dim, hidden_dim, num_clusters=[25, 372],
                 heads=1, drop_rate=0.1, enable_dual_stream=True):
        super().__init__()

        self.enable_dual_stream = enable_dual_stream

        self.embedding = MLP(node_dim, hidden_dim, 0.0)
        self.GeoBlock1 = GeoBlock(hidden_dim, hidden_dim, drop_rate)
        self.GeoBlock2 = GeoBlock(hidden_dim, hidden_dim, drop_rate)
        self.GeoBlock3 = GeoBlock(hidden_dim, hidden_dim, drop_rate)

        self.diffpool1 = DiffPool(hidden_dim, hidden_dim, 1100, num_clusters[0], "intra_lig", drop_rate)
        self.diffpool2 = DiffPool(hidden_dim, hidden_dim, 1100, num_clusters[1], "intra_pro", drop_rate)
        self.attblock1 = AttentionBlock(hidden_dim, heads, drop_rate)
        self.attblock2 = AttentionBlock(hidden_dim, heads, drop_rate)

        if enable_dual_stream:
            self.cross_attention_channel = CrossAttentionAtomChannel(
                hidden_dim=hidden_dim,
                num_heads=4,
                dropout=drop_rate
            )
            self.gating_fusion = RobustGatingFusionModule(
                hidden_dim=hidden_dim,
                temperature=2.0
            )

    def make_edge_index(self, data):
        data.edge_index_intra_lig = data.edge_index_intra[:, data.split[data.edge_index_intra[0, :]] == 0]
        data.edge_index_intra_pro = data.edge_index_intra[:, data.split[data.edge_index_intra[0, :]] == 1]

    def forward(self, data, return_attention=False, return_selection_info=False, epoch=None):
        x = data.x
        x = self.embedding(x)

        self.make_edge_index(data)
        x = self.GeoBlock1(x, data)
        x = self.GeoBlock2(x, data)
        x = self.GeoBlock3(x, data)

        x_lig, _ = self.diffpool1(x, data)
        x_pro, _ = self.diffpool2(x, data)

        l2p, attention1 = self.attblock1(x_lig, x_pro, x_pro)
        p2l, attention2 = self.attblock2(x_pro, x_lig, x_lig)
        z_cluster = l2p + p2l

        if self.enable_dual_stream:
            z_atoms, interaction_info = self.cross_attention_channel(x, data)
            z_final, gate_weights = self.gating_fusion(z_cluster, z_atoms, epoch=epoch)
        else:
            z_final = z_cluster
            gate_weights = None
            interaction_info = None

        result = z_final

        extra_info = {}
        if return_attention:
            extra_info['attention_weights'] = [attention1, attention2]
        if return_selection_info and self.enable_dual_stream:
            extra_info['interaction_info'] = interaction_info
            extra_info['gate_weights'] = gate_weights

        if extra_info:
            return result, extra_info
        else:
            return result

    def get_model_info(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        info = {
            'model_type': 'DualStreamGeoHDS' if self.enable_dual_stream else 'GeoHDS',
            'total_parameters': total_params,
            'dual_stream_enabled': self.enable_dual_stream,
        }

        if self.enable_dual_stream:
            cross_attention_params = sum(p.numel() for p in self.cross_attention_channel.parameters() if p.requires_grad)
            fusion_params = sum(p.numel() for p in self.gating_fusion.parameters() if p.requires_grad)

            info.update({
                'cross_attention_parameters': cross_attention_params,
                'fusion_module_parameters': fusion_params,
                'num_attention_heads': self.cross_attention_channel.num_heads,
                'atom_channel_type': 'CrossAttentionAtomChannel',
                'gating_architecture': self.gating_fusion.get_gating_info()['module_type'],
                'gating_temperature': self.gating_fusion.temperature,
                'gating_has_layer_norm': self.gating_fusion.get_gating_info()['has_layer_norm'],
            })

        return info


class GeoHDS(DualStreamGeoHDS):
    def __init__(self, node_dim, hidden_dim, num_clusters, heads=1, drop_rate=0.1):
        super().__init__(node_dim, hidden_dim, num_clusters, heads, drop_rate, enable_dual_stream=False)

class MLP_LEP(torch.nn.Module):
    def __init__(self, hidden_dim):
        super(MLP_LEP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Mish(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, input1, input2):
        x = torch.cat((input1, input2), dim=1)
        x = self.mlp(x)
        return x.view(-1)
