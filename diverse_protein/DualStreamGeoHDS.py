import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch_geometric.nn import DenseGCNConv
from torch_geometric.nn.conv import MessagePassing

from model import _rbf, gnn_norm, MLP, FC, HIL, GeoBlock, DiffPool, AttentionBlock

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
            'supports_staged_training': True
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

        self.fc = FC(hidden_dim, hidden_dim, 2, drop_rate, 1)

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

        prediction = self.fc(z_final)
        result = prediction.view(-1)

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


def create_geohds_model(node_dim, hidden_dim, num_clusters, heads=1, drop_rate=0.1, dual_stream=True):
    return DualStreamGeoHDS(
        node_dim=node_dim,
        hidden_dim=hidden_dim,
        num_clusters=num_clusters,
        heads=heads,
        drop_rate=drop_rate,
        enable_dual_stream=dual_stream
    )
