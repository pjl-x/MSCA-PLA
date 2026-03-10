import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch_geometric.nn import DenseGCNConv
from torch_geometric.nn.conv import MessagePassing

from GeoHDS import _rbf, gnn_norm, MLP, FC, HIL, GeoBlock, DiffPool, AttentionBlock

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
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.protein_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.ligand_aggregator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.protein_aggregator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.final_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.ligand_norm = nn.LayerNorm(hidden_dim)
        self.protein_norm = nn.LayerNorm(hidden_dim)

    def forward(self, atom_embeddings, data):
        device = atom_embeddings.device
        batch_size = data.batch.max().item() + 1

        batch_representations = []
        interaction_info = {
            'lig2pro_attention_weights': [],
            'pro2lig_attention_weights': [],
            'ligand_aggregation_weights': [],
            'protein_aggregation_weights': []
        }

        for graph_idx in range(batch_size):
            graph_mask = (data.batch == graph_idx)

            if graph_mask.sum() == 0:
                batch_representations.append(torch.zeros(self.hidden_dim, device=device))
                for key in interaction_info.keys():
                    interaction_info[key].append(None)
                continue

            graph_atoms = atom_embeddings[graph_mask]
            graph_split = data.split[graph_mask]

            ligand_mask = (graph_split == 0)
            protein_mask = (graph_split == 1)

            ligand_atoms = graph_atoms[ligand_mask]
            protein_atoms = graph_atoms[protein_mask]

            if ligand_atoms.size(0) == 0 or protein_atoms.size(0) == 0:
                graph_representation = torch.mean(graph_atoms, dim=0)
                batch_representations.append(graph_representation)
                for key in interaction_info.keys():
                    interaction_info[key].append(None)
                continue

            ligand_features = self.ligand_preprocessor(ligand_atoms)
            protein_features = self.protein_preprocessor(protein_atoms)

            ligand_features_batch = ligand_features.unsqueeze(0)
            protein_features_batch = protein_features.unsqueeze(0)

            ligand_enhanced, lig2pro_attn = self.lig2pro_attention(
                query=ligand_features_batch,
                key=protein_features_batch,
                value=protein_features_batch,
                need_weights=True
            )
            ligand_enhanced = ligand_enhanced.squeeze(0)

            protein_enhanced, pro2lig_attn = self.pro2lig_attention(
                query=protein_features_batch,
                key=ligand_features_batch,
                value=ligand_features_batch,
                need_weights=True
            )
            protein_enhanced = protein_enhanced.squeeze(0)

            ligand_fused = self.ligand_fusion(torch.cat([ligand_features, ligand_enhanced], dim=-1))
            protein_fused = self.protein_fusion(torch.cat([protein_features, protein_enhanced], dim=-1))

            ligand_fused = self.ligand_norm(ligand_fused)
            protein_fused = self.protein_norm(protein_fused)

            ligand_attn_weights = F.softmax(self.ligand_aggregator(ligand_fused).squeeze(-1), dim=0)
            protein_attn_weights = F.softmax(self.protein_aggregator(protein_fused).squeeze(-1), dim=0)

            ligand_representation = torch.sum(ligand_fused * ligand_attn_weights.unsqueeze(-1), dim=0)
            protein_representation = torch.sum(protein_fused * protein_attn_weights.unsqueeze(-1), dim=0)

            combined_representation = torch.cat([ligand_representation, protein_representation], dim=-1)
            final_representation = self.final_fusion(combined_representation)

            batch_representations.append(final_representation)

            interaction_info['lig2pro_attention_weights'].append(lig2pro_attn.squeeze(0).cpu())
            interaction_info['pro2lig_attention_weights'].append(pro2lig_attn.squeeze(0).cpu())
            interaction_info['ligand_aggregation_weights'].append(ligand_attn_weights.cpu())
            interaction_info['protein_aggregation_weights'].append(protein_attn_weights.cpu())

        z_atoms = torch.stack(batch_representations, dim=0)

        return z_atoms, interaction_info


class RobustGatingFusionModule(nn.Module):

    def __init__(self, hidden_dim, fusion_dropout=0.1):
        super().__init__()

        self.norm_cluster = nn.LayerNorm(hidden_dim)
        self.norm_atoms = nn.LayerNorm(hidden_dim)

        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.temperature = nn.Parameter(torch.tensor(1.0))

        self.fusion_enhancement = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(fusion_dropout)
        )

        self._init_weights()

    def _init_weights(self):
        for layer in self.gate_network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.5)
                nn.init.zeros_(layer.bias)
        with torch.no_grad():
            self.gate_network[-1].weight.mul_(0.1)
            self.gate_network[-1].bias.fill_(0.0)

    def forward(self, z_cluster, z_atoms, epoch=None, training_stage=None):
        batch_size = z_cluster.size(0)
        device = z_cluster.device

        z_cluster_norm = self.norm_cluster(z_cluster)
        z_atoms_norm = self.norm_atoms(z_atoms)

        concatenated = torch.cat([z_cluster_norm, z_atoms_norm], dim=-1)
        gate_logits = self.gate_network(concatenated)

        raw_gate_weight = torch.sigmoid(gate_logits / (self.temperature + 1e-8))

        gate_weight = self._apply_staged_gating(raw_gate_weight, epoch, training_stage, device)

        z_fused = gate_weight * z_cluster_norm + (1 - gate_weight) * z_atoms_norm

        z_enhanced = self.fusion_enhancement(z_fused)
        z_final = z_enhanced + z_fused

        return z_final, gate_weight

    def _apply_staged_gating(self, raw_gate_weight, epoch, training_stage, device):
        batch_size = raw_gate_weight.size(0)

        if epoch is None:
            return raw_gate_weight

        if training_stage is None:
            if epoch < 50:
                training_stage = "forced"
            elif epoch < 100:
                training_stage = "progressive"
            else:
                training_stage = "free"

        if training_stage == "forced":
            gate_weight = torch.full((batch_size, 1), 0.5, device=device)

        elif training_stage == "progressive":
            progress = (epoch - 50) / 50
            progress = torch.clamp(torch.tensor(progress), 0.0, 1.0)
            forced_weight = torch.full((batch_size, 1), 0.5, device=device)
            gate_weight = (1 - progress) * forced_weight + progress * raw_gate_weight

        else:
            gate_weight = raw_gate_weight

        return gate_weight

    def get_gating_info(self):
        return {
            'temperature': self.temperature.item(),
            'gate_network_layers': len(self.gate_network),
            'has_layer_norm': True,
            'architecture_type': 'robust_anti_collapse',
            'supports_staged_training': True
        }

    def compute_gating_balance_loss(self, gate_weights, epoch=None):
        deviation_loss = torch.mean((gate_weights - 0.5) ** 2)

        extreme_penalty = torch.mean(
            torch.exp(5 * torch.clamp(gate_weights - 0.8, min=0)) +
            torch.exp(5 * torch.clamp(0.2 - gate_weights, min=0))
        )

        gate_std = torch.std(gate_weights)
        std_loss = torch.clamp(0.1 - gate_std, min=0)

        if epoch is not None:
            if epoch < 50:
                weight = 0.05
            elif epoch < 100:
                weight = 0.15
            elif epoch < 150:
                weight = 0.25
            else:
                weight = 0.2
        else:
            weight = 0.2

        balance_loss = weight * (deviation_loss + extreme_penalty + std_loss)

        return balance_loss

    def get_training_stage_info(self, epoch):
        if epoch < 50:
            stage = "forced"
            description = f"forced balance ({epoch}/50) - weight: 0.05"
        elif epoch < 100:
            stage = "progressive"
            progress = (epoch - 50) / 50 * 100
            description = f"progressive release ({progress:.1f}%) - weight: 0.15"
        elif epoch < 150:
            stage = "free_early"
            progress = (epoch - 100) / 50 * 100
            description = f"free gating early ({progress:.1f}%) - weight: 0.25"
        else:
            stage = "free_late"
            description = f"free gating late (epoch {epoch}) - weight: 0.2"

        return {
            'stage': stage,
            'description': description,
            'epoch': epoch
        }


class DualStreamGeoHDS(nn.Module):

    def __init__(self, node_dim, hidden_dim, num_clusters=[28, 156],
                 heads=1, drop_rate=0.1, enable_dual_stream=True):
        super().__init__()

        self.enable_dual_stream = enable_dual_stream

        self.embedding = MLP(node_dim, hidden_dim, 0.0)
        self.GeoBlock1 = GeoBlock(hidden_dim, hidden_dim, drop_rate)
        self.GeoBlock2 = GeoBlock(hidden_dim, hidden_dim, drop_rate)
        self.GeoBlock3 = GeoBlock(hidden_dim, hidden_dim, drop_rate)

        self.diffpool1 = DiffPool(hidden_dim, hidden_dim, 600, num_clusters[0], "intra_lig", drop_rate)
        self.diffpool2 = DiffPool(hidden_dim, hidden_dim, 600, num_clusters[1], "intra_pro", drop_rate)
        self.attblock1 = AttentionBlock(hidden_dim, heads, drop_rate)
        self.attblock2 = AttentionBlock(hidden_dim, heads, drop_rate)

        if self.enable_dual_stream:
            self.cross_attention_channel = CrossAttentionAtomChannel(
                hidden_dim=hidden_dim,
                num_heads=4,
                dropout=drop_rate
            )
            self.gating_fusion = RobustGatingFusionModule(hidden_dim, drop_rate)

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
            extra_info['attention_weights'] = {
                'ligand_to_protein': attention1,
                'protein_to_ligand': attention2
            }

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
            gating_info = self.gating_fusion.get_gating_info()

            info.update({
                'cross_attention_parameters': cross_attention_params,
                'fusion_module_parameters': fusion_params,
                'num_attention_heads': self.cross_attention_channel.num_heads,
                'atom_channel_type': 'CrossAttention',
                'gating_architecture': gating_info['architecture_type'],
                'gating_temperature': gating_info['temperature'],
                'gating_has_layer_norm': gating_info['has_layer_norm'],
            })

        return info


def create_geohds_model(node_dim, hidden_dim, num_clusters=[28, 156],
                        heads=1, drop_rate=0.1, dual_stream=False, **kwargs):
    if dual_stream:
        return DualStreamGeoHDS(
            node_dim=node_dim,
            hidden_dim=hidden_dim,
            num_clusters=num_clusters,
            heads=heads,
            drop_rate=drop_rate,
            enable_dual_stream=True
        )
    else:
        return DualStreamGeoHDS(
            node_dim=node_dim,
            hidden_dim=hidden_dim,
            num_clusters=num_clusters,
            heads=heads,
            drop_rate=drop_rate,
            enable_dual_stream=False
        )
