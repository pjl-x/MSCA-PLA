import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import random
from utils import AverageMeter
from GeoHDS import GeoHDS
from DualStreamGeoHDS import create_geohds_model
from dataset_GeoHDS import GraphDataset, PLIDataLoader
from config.config_dict import Config
from log.train_logger import TrainLogger
import numpy as np
from utils import *
from sklearn.metrics import mean_squared_error

def val(model, dataloader, device, collect_dual_stream_info=False, epoch=None):
    model.eval()

    pred_list = []
    label_list = []
    gate_weights_list = []

    for data in dataloader:
        data = data.to(device)
        with torch.no_grad():
            if collect_dual_stream_info and hasattr(model, 'enable_dual_stream') and model.enable_dual_stream:
                pred, extra_info = model(data, return_selection_info=True, epoch=epoch)
                gate_weights = extra_info.get('gate_weights')
                if gate_weights is not None:
                    gate_weights_list.append(gate_weights.detach().cpu().numpy())
            else:
                pred = model(data, epoch=epoch) if hasattr(model, 'enable_dual_stream') and model.enable_dual_stream else model(data)

            label = data.y
            pred_list.append(pred.detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())

    pred = np.concatenate(pred_list, axis=0)
    label = np.concatenate(label_list, axis=0)
    coff = np.corrcoef(pred, label)[0, 1]
    rmse = np.sqrt(mean_squared_error(label, pred))

    model.train()

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
        return rmse, coff, dual_stream_info
    else:
        return rmse, coff

def seed_everything(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def check_quantiles(train_loader, valid_loader, test2013_loader, test2016_loader, test2019_loader):
    lig_list = []
    pro_list = []
    total_list = []
    for idx, j in [('train_loader', train_loader), ('valid_loader', valid_loader), ('test2013_loader', test2013_loader), \
                ('test2016_loader', test2016_loader), ('test2019_loader', test2019_loader)]:
        print(f'=' * 20)
        print(f'Processing {idx}')
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

if __name__ == '__main__':
    cfg = 'TrainConfig_GeoHDS'
    config = Config(cfg)
    args = config.get_config()
    graph_type = args.get("graph_type")
    save_model = args.get("save_model")
    batch_size = args.get("batch_size")
    data_root = args.get('data_root')
    epochs = args.get('epochs')
    repeats = args.get('repeat')
    early_stop_epoch = args.get("early_stop_epoch")
    lr = args.get("lr")

    for rep in range(10):
        for repeat, seed in enumerate(np.random.randint(0, 1000, size=3)):
            seed_everything(seed)

            save_dir = f"./model/cross_dataset_test_{rep}"
            msg_info = f"lr={lr}, repeat={repeat}, seed={seed}"
            args['repeat'] = repeat

            train_dir = os.path.join(data_root, 'train')
            valid_dir = os.path.join(data_root, 'valid')
            test2013_dir = os.path.join(data_root, 'test2013')
            test2016_dir = os.path.join(data_root, 'test2016')
            test2019_dir = os.path.join(data_root, 'test2019')

            train_df = pd.read_csv(os.path.join(data_root, 'train.csv'))
            valid_df = pd.read_csv(os.path.join(data_root, 'valid.csv'))
            test2013_df = pd.read_csv(os.path.join(data_root, 'test2013.csv'))
            test2016_df = pd.read_csv(os.path.join(data_root, 'test2016.csv'))
            test2019_df = pd.read_csv(os.path.join(data_root, 'test2019.csv'))

            train_set = GraphDataset(train_dir, train_df, graph_type=graph_type, create=False)
            valid_set = GraphDataset(valid_dir, valid_df, graph_type=graph_type, create=False)
            test2013_set = GraphDataset(test2013_dir, test2013_df, graph_type=graph_type, create=False)
            test2016_set = GraphDataset(test2016_dir, test2016_df, graph_type=graph_type, create=False)
            test2019_set = GraphDataset(test2019_dir, test2019_df, graph_type=graph_type, create=False)

            train_loader = PLIDataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
            valid_loader = PLIDataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=4)
            test2013_loader = PLIDataLoader(test2013_set, batch_size=batch_size, shuffle=False, num_workers=4)
            test2016_loader = PLIDataLoader(test2016_set, batch_size=batch_size, shuffle=False, num_workers=4)
            test2019_loader = PLIDataLoader(test2019_set, batch_size=batch_size, shuffle=False, num_workers=4)

            print(f'Train/Valid samples : {len(train_set) + len(valid_set)}')
            print(f'Test2013 samples : {len(test2013_set)}')
            print(f'Test2016 samples : {len(test2016_set)}')
            print(f'Test2019 samples : {len(test2019_set)}')

            iters = len(train_loader)
            logger = TrainLogger(args, cfg, save_dir, create=True)
            logger.info(msg_info)
            logger.info(__file__)
            logger.info(f"train data: {len(train_set)}")
            logger.info(f"valid data: {len(valid_set)}")
            logger.info(f"test2013 data: {len(test2013_set)}")
            logger.info(f"test2016 data: {len(test2016_set)}")
            logger.info(f"test2019 data: {len(test2019_set)}")

            q_lig = [0, 20, 28, 37, 177]
            q_pro = [0, 130, 156, 186, 500]
            q_i_lig = 2
            q_i_pro = 2
            num_clusters = [q_lig[q_i_lig], q_pro[q_i_pro]]

            device = "cuda" if torch.cuda.is_available() else "cpu"

            enable_dual_stream = args.get("enable_dual_stream", True)

            model = create_geohds_model(
                node_dim=35,
                hidden_dim=256,
                num_clusters=num_clusters,
                heads=1,
                drop_rate=0.1,
                dual_stream=enable_dual_stream
            ).to(device)

            model.train()

            model_info = model.get_model_info()
            logger.info(f"Model type: {model_info['model_type']}")
            logger.info(f"Total parameters: {model_info['total_parameters']:,}")

            if enable_dual_stream:
                logger.info(f"Dual-stream mode enabled")
                logger.info(f"Cross-attention parameters: {model_info['cross_attention_parameters']:,}")
                logger.info(f"Fusion module parameters: {model_info['fusion_module_parameters']:,}")
                logger.info(f"Attention heads: {model_info['num_attention_heads']}")
                logger.info(f"Gating architecture: {model_info['gating_architecture']}")
                logger.info(f"Gating temperature: {model_info['gating_temperature']:.3f}")
                logger.info(f"LayerNorm: {'enabled' if model_info['gating_has_layer_norm'] else 'disabled'}")
            else:
                logger.info(f"Single-stream mode")

            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, verbose=True)
            criterion = nn.MSELoss()

            running_loss = AverageMeter()
            running_acc = AverageMeter()
            running_best_mse = BestMeter("min")
            best_model_list = []

            for epoch in range(epochs):
                if enable_dual_stream and epoch % 10 == 0:
                    stage_info = model.gating_fusion.get_training_stage_info(epoch)
                    logger.info(f"Training stage: {stage_info['description']}")

                for batch_idx, data in enumerate(train_loader):
                    data = data.to(device)

                    if enable_dual_stream:
                        pred, extra_info = model(data, return_selection_info=True, epoch=epoch)
                        gate_weights = extra_info.get('gate_weights')
                    else:
                        pred = model(data)
                        gate_weights = None

                    label = data.y
                    MSE_loss = criterion(pred, label)

                    if enable_dual_stream and gate_weights is not None:
                        balance_loss = model.gating_fusion.compute_gating_balance_loss(gate_weights, epoch)
                        loss = MSE_loss + balance_loss
                        if batch_idx % 100 == 0:
                            logger.info(f"Loss (Epoch {epoch}, Batch {batch_idx}): MSE={MSE_loss.item():.4f}, Balance={balance_loss.item():.4f}, Total={loss.item():.4f}")
                    else:
                        loss = MSE_loss
                        balance_loss = torch.tensor(0.0)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    running_loss.update(loss.item(), label.size(0))

                epoch_loss = running_loss.get_average()
                epoch_rmse = np.sqrt(epoch_loss)
                running_loss.reset()

                collect_info = (epoch % 10 == 0) and enable_dual_stream

                if collect_info:
                    valid_rmse, valid_pr, dual_info = val(model, valid_loader, device, collect_dual_stream_info=True, epoch=epoch)
                    logger.info(f"Dual-stream info (Epoch {epoch}): gate_mean={dual_info['gate_weights_mean']:.4f}, gate_std={dual_info['gate_weights_std']:.4f}, cluster_ratio={dual_info['cluster_dominance_ratio']:.1%}, atom_ratio={dual_info['atom_dominance_ratio']:.1%}")
                else:
                    valid_rmse, valid_pr = val(model, valid_loader, device, epoch=epoch)

                scheduler.step(valid_rmse)

                msg = "epoch-%d, train_loss-%.4f, train_rmse-%.4f, valid_rmse-%.4f, valid_pr-%.4f" \
                        % (epoch, epoch_loss, epoch_rmse, valid_rmse, valid_pr)
                logger.info(msg)

                if valid_rmse < running_best_mse.get_best():
                    running_best_mse.update(valid_rmse)
                    if save_model:
                        test2013_rmse, test2013_pr = val(model, test2013_loader, device)
                        test2016_rmse, test2016_pr = val(model, test2016_loader, device)
                        test2019_rmse, test2019_pr = val(model, test2019_loader, device)
                        msg_train = f"Validation : valid_rmse-{valid_rmse:.4f}, valid_pr-{valid_pr:.4f}, \ntest2013_rmse-{test2013_rmse:.4f}, test2013_pr-{test2013_pr:.4f}, test2016_rmse-{test2016_rmse:.4f}, test2016_pr-{test2016_pr:.4f}, test2019_rmse-{test2019_rmse:.4f}, test2019_pr-{test2019_pr:.4f}"
                        logger.info(msg_train)
                        msg = "epoch-%d, train_loss-%.4f, train_rmse-%.4f, valid_rmse-%.4f, valid_pr-%.4f" \
                        % (epoch, epoch_loss, epoch_rmse, valid_rmse, valid_pr)
                        model_path = os.path.join(logger.get_model_dir(), msg + '.pt')
                        best_model_list.append(model_path)
                        save_model_dict(model, logger.get_model_dir(), msg)

                else:
                    count = running_best_mse.counter()
                    if count > early_stop_epoch:
                        best_mse = running_best_mse.get_best()
                        msg = "best_rmse: %.4f" % best_mse
                        logger.info(f"early stop in epoch {epoch}")
                        logger.info(msg)
                        break_flag = True
                        break

            load_model_dict(model, best_model_list[-1])
            valid_rmse, valid_pr = val(model, valid_loader, device)
            test2013_rmse, test2013_pr = val(model, test2013_loader, device)
            test2016_rmse, test2016_pr = val(model, test2016_loader, device)
            test2019_rmse, test2019_pr = val(model, test2019_loader, device)
            msg_test = f"valid_rmse-{valid_rmse:.4f}, valid_pr-{valid_pr:.4f}, test2013_rmse-{test2013_rmse:.4f}, test2013_pr-{test2013_pr:.4f}, test2016_rmse-{test2016_rmse:.4f}, test2016_pr-{test2016_pr:.4f}, test2019_rmse-{test2019_rmse:.4f}, test2019_pr-{test2019_pr:.4f}"
            logger.info(msg_test)
