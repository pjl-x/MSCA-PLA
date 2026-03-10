import os
import sys
sys.path.append(os.path.abspath('../atom3d'))
import argparse
import logging
import time
import datetime
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from model import GeoHDS
from DualStreamGeoHDS import create_geohds_model
from data import GNNTransformLBA
from atom3d.datasets import LMDBDataset, PTGDataset
from scipy.stats import spearmanr
import warnings
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
warnings.filterwarnings("ignore")

def train_loop(args, model, loader, optimizer, scheduler, epoch, device):
    model.train()

    loss_all = 0
    total = 0
    for it, data in enumerate(loader):
        data = data.to(device)
        optimizer.zero_grad()

        if args.enable_dual_stream:
            pred, extra_info = model(data, return_selection_info=True, epoch=epoch)
            gate_weights = extra_info.get('gate_weights')
        else:
            pred = model(data)
            gate_weights = None

        mse_loss = F.mse_loss(pred, data.y)

        if args.enable_dual_stream and gate_weights is not None:
            balance_loss = model.gating_fusion.compute_gating_balance_loss(gate_weights, epoch)
            loss = mse_loss + balance_loss
            if it % 100 == 0:
                print(f"Loss (Epoch {epoch}, Batch {it}): MSE={mse_loss.item():.4f}, Balance={balance_loss.item():.4f}, Total={loss.item():.4f}")
        else:
            loss = mse_loss

        loss.backward()
        loss_all += loss.item() * data.num_graphs
        total += data.num_graphs
        optimizer.step()

    return np.sqrt(loss_all / total)


@torch.no_grad()
def test(model, loader, device, collect_dual_stream_info=False, epoch=None):
    model.eval()

    loss_all = 0
    total = 0
    gate_weights_list = []

    y_true = []
    y_pred = []

    for data in loader:
        data = data.to(device)

        if collect_dual_stream_info and hasattr(model, 'enable_dual_stream') and model.enable_dual_stream:
            pred, extra_info = model(data, return_selection_info=True, epoch=epoch)
            gate_weights = extra_info.get('gate_weights')
            if gate_weights is not None:
                gate_weights_list.append(gate_weights.detach().cpu().numpy())
        else:
            pred = model(data, epoch=epoch) if hasattr(model, 'enable_dual_stream') and model.enable_dual_stream else model(data)

        loss = F.mse_loss(pred, data.y)
        loss_all += loss.item() * data.num_graphs
        total += data.num_graphs
        y_true.extend(data.y.tolist())
        y_pred.extend(pred.tolist())

    r_p = np.corrcoef(y_true, y_pred)[0, 1]
    r_s = spearmanr(y_true, y_pred)[0]

    dual_stream_info = None
    if collect_dual_stream_info and gate_weights_list:
        gate_weights = np.concatenate(gate_weights_list, axis=0)
        dual_stream_info = {
            'gate_weights_mean': np.mean(gate_weights),
            'gate_weights_std': np.std(gate_weights),
            'cluster_dominance_ratio': np.mean(gate_weights > 0.5),
            'atom_dominance_ratio': np.mean(gate_weights < 0.5),
        }

    if dual_stream_info is not None:
        return np.sqrt(loss_all / total), r_p, r_s, y_true, y_pred, dual_stream_info
    else:
        return np.sqrt(loss_all / total), r_p, r_s, y_true, y_pred

def save_weights(model, weight_dir):
    torch.save(model.state_dict(), weight_dir)

def train(args, device, log_dir, rep=None, test_mode=False):

    if args.precomputed:
        train_dataset = PTGDataset(os.path.join(args.data_dir, 'train'))
        val_dataset = PTGDataset(os.path.join(args.data_dir, 'val'))
        test_dataset = PTGDataset(os.path.join(args.data_dir, 'test'))
    else:
        transform = GNNTransformLBA()
        train_dataset = LMDBDataset(os.path.join(args.data_dir, 'train'), transform=transform)
        val_dataset = LMDBDataset(os.path.join(args.data_dir, 'val'), transform=transform)
        test_dataset = LMDBDataset(os.path.join(args.data_dir, 'test'), transform=transform)
    print(f'Total samples : {len(train_dataset) + len(val_dataset) + len(test_dataset)}')

    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=4)

    for data in train_loader:
        num_features = data.num_features
        break

    num_clusters = [25, 372] if args.seqid == 30 else [24, 362]

    if args.enable_dual_stream:
        model = create_geohds_model(
            node_dim=num_features,
            hidden_dim=args.hidden_dim,
            num_clusters=num_clusters,
            heads=1,
            drop_rate=0.1,
            dual_stream=True
        ).to(device)
    else:
        model = GeoHDS(num_features, hidden_dim=args.hidden_dim, num_clusters=num_clusters).to(device)

    model.to(device)

    if hasattr(model, 'get_model_info'):
        model_info = model.get_model_info()
        print(f"Model type: {model_info['model_type']}")
        print(f"Total parameters: {model_info['total_parameters']:,}")
        logger.info(f"Model type: {model_info['model_type']}")
        logger.info(f"Total parameters: {model_info['total_parameters']:,}")

        if args.enable_dual_stream:
            print(f"Dual-stream mode enabled")
            print(f"Cross-attention parameters: {model_info['cross_attention_parameters']:,}")
            print(f"Fusion module parameters: {model_info['fusion_module_parameters']:,}")
            print(f"Attention heads: {model_info['num_attention_heads']}")
            print(f"Gating architecture: {model_info['gating_architecture']}")
            print(f"Gating temperature: {model_info['gating_temperature']:.3f}")
            print(f"LayerNorm: {'enabled' if model_info['gating_has_layer_norm'] else 'disabled'}")

            logger.info(f"Dual-stream mode enabled")
            logger.info(f"Cross-attention parameters: {model_info['cross_attention_parameters']:,}")
            logger.info(f"Fusion module parameters: {model_info['fusion_module_parameters']:,}")
            logger.info(f"Attention heads: {model_info['num_attention_heads']}")
            logger.info(f"Gating architecture: {model_info['gating_architecture']}")
            logger.info(f"Gating temperature: {model_info['gating_temperature']:.3f}")
            logger.info(f"LayerNorm: {'enabled' if model_info['gating_has_layer_norm'] else 'disabled'}")
        else:
            print(f"Single-stream mode")
            logger.info(f"Single-stream mode")
    else:
        print(f'GeoHDS params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
        logger.info(f"GeoHDS params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    best_val_loss = 999
    best_rp = 0
    best_rs = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-6)
    if args.use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, verbose=True)
    else:
        scheduler = None
    count = 0
    for epoch in range(1, args.num_epochs + 1):
        if args.enable_dual_stream and epoch % 10 == 0:
            stage_info = model.gating_fusion.get_training_stage_info(epoch)
            print(f"Training stage: {stage_info['description']}")
            logger.info(f"Training stage: {stage_info['description']}")

        start = time.time()
        train_loss = train_loop(args, model, train_loader, optimizer, scheduler, epoch, device)

        collect_info = (epoch % 10 == 0) and args.enable_dual_stream

        if collect_info:
            val_loss, r_p, r_s, y_true, y_pred, dual_info = test(model, val_loader, device, collect_dual_stream_info=True, epoch=epoch)
            print(f"Dual-stream info (Epoch {epoch}): gate_mean={dual_info['gate_weights_mean']:.4f}, gate_std={dual_info['gate_weights_std']:.4f}, cluster_ratio={dual_info['cluster_dominance_ratio']:.1%}, atom_ratio={dual_info['atom_dominance_ratio']:.1%}")
            logger.info(f"Dual-stream info (Epoch {epoch}): gate_mean={dual_info['gate_weights_mean']:.4f}, gate_std={dual_info['gate_weights_std']:.4f}, cluster_ratio={dual_info['cluster_dominance_ratio']:.1%}, atom_ratio={dual_info['atom_dominance_ratio']:.1%}")
        else:
            val_loss, r_p, r_s, y_true, y_pred = test(model, val_loader, device, epoch=epoch)

        if args.use_scheduler:
            scheduler.step(val_loss)
        if val_loss < best_val_loss:
            count = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }, os.path.join(log_dir, f'best_weights_rep{rep}.pt'))
            print(f'Best model saved at epoch {epoch} with val rmse: {val_loss:.7f}')
            logger.info(f'Best model saved at epoch {epoch} with val rmse: {val_loss:.7f}')
            best_val_loss = val_loss
            best_rp = r_p
            best_rs = r_s
            if test_mode:
                cpt = torch.load(os.path.join(log_dir, f'best_weights_rep{rep}.pt'))
                model.load_state_dict(cpt['model_state_dict'])
                rmse, pearson, spearman, _, _ = test(model, test_loader, device)
                print(f'\tTest RMSE {rmse:.7f}, Test Pearson {pearson:.7f}, Test Spearman {spearman:.7f}')
        else:
            count += 1
            if count > args.early_stop_patience and epoch > 200:
                print("Early stopping")
                logger.info("Early stopping")
                break
        elapsed = (time.time() - start)
        print('Epoch: {:03d}, Time: {:.3f}s'.format(epoch, elapsed), end=', ')
        print('Train RMSE: {:.7f}, Val RMSE: {:.7f}, Pearson R: {:.7f}, Spearman R: {:.7f}'.format(train_loss, val_loss, r_p, r_s))
        logger.info('Epoch: {:03d}, Train RMSE: {:.7f}, Val RMSE: {:.7f}, Pearson R: {:.7f}, Spearman R: {:.7f}'.format(epoch, train_loss, val_loss, r_p, r_s))

    if test_mode:
        cpt = torch.load(os.path.join(log_dir, f'best_weights_rep{rep}.pt'))
        model.load_state_dict(cpt['model_state_dict'])
        _, _, _, y_true_train, y_pred_train = test(model, train_loader, device)
        torch.save({'targets': y_true_train, 'predictions': y_pred_train},
                   os.path.join(log_dir, f'lba-rep{rep}.best.train.pt'))
        _, _, _, y_true_val, y_pred_val = test(model, val_loader, device)
        torch.save({'targets': y_true_val, 'predictions': y_pred_val},
                   os.path.join(log_dir, f'lba-rep{rep}.best.val.pt'))
        rmse, pearson, spearman, y_true_test, y_pred_test = test(model, test_loader, device)
        print(f'\tTest RMSE {rmse:.7f}, Test Pearson {pearson:.7f}, Test Spearman {spearman:.7f}')
        logger.info(f'Test RMSE {rmse:.7f}, Test Pearson {pearson:.7f}, Test Spearman {spearman:.7f}')
        torch.save({'targets': y_true_test, 'predictions': y_pred_test},
                   os.path.join(log_dir, f'lba-rep{rep}.best.test.pt'))

    return best_val_loss, best_rp, best_rs

def check_quantiles(train_loader, val_loader, test_loader):
    lig_list = []
    pro_list = []
    total_list = []
    for j in [train_loader, val_loader, test_loader]:
        for i, data in enumerate(j):
            for i in range(data.batch.max().item() + 1):
                mask = data.batch[data.edge_index_intra[0, :]] == i
                mask_lig = data.split[data.edge_index_intra[0, :]] == 0
                mask_pro = data.split[data.edge_index_intra[0, :]] == 1
                edge_index_lig = data.edge_index_intra[:, mask & mask_lig]
                edge_index_pro = data.edge_index_intra[:, mask & mask_pro]
                unique_nodes_lig = torch.unique(edge_index_lig)
                unique_nodes_pro = torch.unique(edge_index_pro)

                lig_list.append(unique_nodes_lig.size(0))
                pro_list.append(unique_nodes_pro.size(0))
                total_list.append(unique_nodes_lig.size(0) + unique_nodes_pro.size(0))
    lig_list = np.array(lig_list)
    q1 = np.percentile(lig_list, 25)
    q2 = np.percentile(lig_list, 50)
    q3 = np.percentile(lig_list, 75)
    q4 = np.percentile(lig_list, 100)
    avg = np.mean(lig_list)
    std = np.std(lig_list)
    print(f'LIG: {q1}, {q2}, {q3}, {q4}, {avg:.2f}, {std:.2f}')

    pro_list = np.array(pro_list)
    q1 = np.percentile(pro_list, 25)
    q2 = np.percentile(pro_list, 50)
    q3 = np.percentile(pro_list, 75)
    q4 = np.percentile(pro_list, 100)
    avg = np.mean(pro_list)
    std = np.std(pro_list)
    print(f'PRO: {q1}, {q2}, {q3}, {q4}, {avg:.2f}, {std:.2f}')

    total_list = np.array(total_list)
    q1 = np.percentile(total_list, 25)
    q2 = np.percentile(total_list, 50)
    q3 = np.percentile(total_list, 75)
    q4 = np.percentile(total_list, 100)
    avg = np.mean(total_list)
    std = np.std(total_list)
    print(f'TOTAL: {q1}, {q2}, {q3}, {q4}, {avg:.2f}, {std:.2f}')
    print(f'------------------------')

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--learning_rate', type=float, default=10e-4)
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--seqid', type=int, default=30)
    parser.add_argument('--precomputed', type=bool, default=True)
    parser.add_argument('--early_stop_patience', type=int, default=100)
    parser.add_argument('--GPU_NUM', type=int, default=None)
    parser.add_argument('--use_scheduler', type=int, default=1)
    parser.add_argument('--seed_set', type=int, default=0)
    parser.add_argument('--rep', type=int, default=None)
    parser.add_argument('--enable_dual_stream', type=int, default=1,
                        help='Enable dual-stream architecture (1: enabled, 0: disabled)')
    args = parser.parse_args()

    args.enable_dual_stream = bool(args.enable_dual_stream)

    if args.data_dir is None:
        args.data_dir = f'dataset/split-by-sequence-identity-{args.seqid}/data'
    if args.GPU_NUM is None:
        args.GPU_NUM = 0 if args.seqid == 30 else 1

    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    log_dir = args.log_dir

    if args.mode == 'train':
        if log_dir is None:
            now = datetime.datetime.now().strftime(f"%Y-%m-%d-%H-%M-%S-{args.seqid}")
            log_dir = os.path.join('logs', now)
        else:
            log_dir = os.path.join('logs', log_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        train(args, device, log_dir)

    elif args.mode == 'test':
        for repeat in range(100):
            if args.seed_set:
                if args.seqid == 30:
                    seed_always = [370, 679, 261]
                elif args.seqid == 60:
                    seed_always = [437, 245, 927]
                iter_list = seed_always
            else:
                iter_list = np.random.randint(0, 1000, size=3)
            for rep, seed in enumerate(iter_list):
                if args.rep is not None and rep != args.rep:
                    continue
                log_dir = os.path.join('logs', f'lba_test_{args.seqid}_{repeat}_{args.GPU_NUM}')
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                logger = logging.getLogger('lba')
                logger.setLevel(logging.INFO)
                fh = logging.FileHandler(os.path.join(log_dir, f'log_rep{rep}.txt'))
                logger.addHandler(fh)
                print(f'seed: {seed}')
                logger.info(f'seed: {seed}')
                seed_everything(seed)
                train(args, device, log_dir, rep, test_mode=True)
                logger.removeHandler(fh)
