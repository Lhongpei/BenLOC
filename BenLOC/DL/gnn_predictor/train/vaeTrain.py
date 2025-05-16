import os
import random
import sys

sys.path.append('..')
sys.path.append('../..')
sys.path.append('.')
import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from sklearn import metrics
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree
from tqdm import tqdm

import wandb
from BenLOC.DL.gnn_predictor.config.setConfig import create_config
from BenLOC.DL.gnn_predictor.dataset_gen.graphDataset import mipGraphDataset, mipGraphN2N
from BenLOC.DL.gnn_predictor.models.vae_models import Decoder, Encoder


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def kl_loss(mu, log_sigma):
    return -0.5 * torch.mean(
        torch.sum(1 + 2 * log_sigma - mu ** 2 - log_sigma.exp() ** 2, dim=1))


def forward_loop(encoder, decoder, dataloader, device, del_ratio, use_wandb, ep, mode: str = 'train'):
    assert mode in ['train', 'valid', 'test']
    if mode == 'train':
        encoder.train()
        decoder.train()
    else:
        encoder.eval()
        decoder.eval()

    loss_counts = {key: 0 for key in ['degree', 'logits', 'weights', 'hlb_s', 'hub_s', 'lb_s', 'ub_s', 'hlb_t', 'hub_t', 'c', 'lb_t', 'ub_t', 't']}

    for batch_data in dataloader:
        batch_data = batch_data.to(device)

        with torch.set_grad_enabled(mode == 'train'):
            xs_mu, xs_logsigma, xs_z, xt_mu, xt_logsigma, xt_z = encoder(batch_data.x_s, batch_data.x_t, batch_data.edge_index, batch_data.edge_attr)
            masked_x_s = batch_data.x_s[:, 0].reshape(-1, 1)
            masked_x_t = batch_data.x_t[:, 0].reshape(-1, 1)

            num_constraint_node = batch_data.x_s.shape[0]
            num_variable_node = batch_data.x_t.shape[0]

            n_del = int(np.ceil(del_ratio * num_constraint_node).item())
            # random select n_del constraint nodes
            selected_constraint_indices = torch.tensor(random.sample(range(num_constraint_node), n_del), device=device)
            edge_to_delete = torch.where(torch.isin(batch_data.edge_index[0], selected_constraint_indices))[0]
            deleted_edge_index = batch_data.edge_index[:, edge_to_delete]
            
            # Create a mask for the edges to delete
            mask = torch.ones(batch_data.edge_index.shape[1], dtype=torch.bool, device=device)
            mask[edge_to_delete] = False

            # Apply the mask to the edge_index and edge_attr tensors
            masked_edge_index = batch_data.edge_index[:, mask]
            masked_edge_attr = batch_data.edge_attr[mask]

            # edge_attr_label: the label of the masked edge attrs
            edge_attr_label_values = batch_data.edge_attr[edge_to_delete].reshape(-1)

            edge_attr_label = torch.zeros(batch_data.x_t.shape[0]).to(device)
            edge_attr_mask = edge_attr_label.clone()
            edge_attr_label[deleted_edge_index[1]] = edge_attr_label_values
            edge_attr_label = edge_attr_label.reshape(-1, 1)
            edge_attr_mask[deleted_edge_index[1]] = 1.

            # degrees of each deleted constraint node
            degree_label = degree(batch_data.edge_index[0], batch_data.x_s.shape[0])
            degree_label = degree_label.reshape(-1, 1)
            min_degree = degree_label.min()
            max_degree = degree_label.max()
            if min_degree == max_degree:
                normalized_degree_label = degree_label / max_degree
            else:
                normalized_degree_label = (degree_label - min_degree) / (max_degree - min_degree)

            # the logits that should be 1
            logits_label = torch.zeros(num_variable_node).to(device)
            logits_label[deleted_edge_index[1]] = 1.
            logits_label = logits_label.reshape(-1, 1)

            # hasLB,hasUB,varLB,varUB,obj,varType
            # the x, y, s, r labels
            hlb_s_label = batch_data.x_s[:, 1].reshape(-1, 1)
            lb_s_label = batch_data.x_s[:, 2].reshape(-1, 1)
            hub_s_label = batch_data.x_s[:, 3].reshape(-1, 1)
            ub_s_label = batch_data.x_s[:, 4].reshape(-1, 1)

            hlb_label = batch_data.x_t[:, 1].reshape(-1, 1)
            hub_label = batch_data.x_t[:, 2].reshape(-1, 1)
            lb_label = batch_data.x_t[:, 3].reshape(-1, 1)
            ub_label = batch_data.x_t[:, 4].reshape(-1, 1)
            c_label = batch_data.x_t[:, 5].reshape(-1, 1)

            t_label = batch_data.x_t[:, 6].reshape(-1, 1)
            predict_degree, predict_weights, predict_hlb_s, predict_hub_s, predict_lb_s, predict_ub_s, predict_hlb, predict_hub, predict_c, predict_lb, predict_ub, predict_t, predict_logits = decoder(masked_x_s, masked_x_t, masked_edge_index, masked_edge_attr, xs_z, xt_z)
            bin_predict_t = (predict_t > 0.).float()
            bin_predict_hlb_s = (predict_hlb_s > 0.).float()
            bin_predict_hub_s = (predict_hub_s > 0.).float()
            bin_predict_hlb = (predict_hlb > 0.).float()
            bin_predict_hub = (predict_hub > 0.).float()
            bin_predict_logits = (predict_logits > 0.).float()

            # Losses
            losses = {
                'c': regression_loss(predict_c, c_label),
                'hlb_s': 10*bce_loss(predict_hlb_s, hlb_s_label),
                'hub_s': 10*bce_loss(predict_hub_s, hub_s_label),
                'lb_s': regression_loss(predict_lb_s, lb_s_label),
                'ub_s': regression_loss(predict_ub_s, ub_s_label),
                'logits': 10*bce_loss(predict_logits, logits_label),
                'degree': regression_loss(predict_degree, normalized_degree_label),
                'hlb_t': 10*bce_loss(predict_hlb, hlb_label),
                'hub_t': 10*bce_loss(predict_hub, hub_label),
                'lb_t': regression_loss(predict_lb, lb_label),
                'ub_t': regression_loss(predict_ub, ub_label),
                't': 10*bce_loss(predict_t, t_label),
                'weights': regression_loss(predict_weights, edge_attr_label),
            }
            pres = {
                'f1_score_logits': metrics.f1_score(logits_label.cpu(), bin_predict_logits.cpu(), zero_division=0),
                'precision_logits': metrics.precision_score(logits_label.cpu(), bin_predict_logits.cpu(), zero_division=0),
                'recall_logits': metrics.recall_score(logits_label.cpu(), bin_predict_logits.cpu(), zero_division=0),
                'f1_score_t': metrics.f1_score(t_label.cpu(), bin_predict_t.cpu(), zero_division=0),
                'f1_score_hlb_s': metrics.f1_score(hlb_s_label.cpu(), bin_predict_hlb_s.cpu(), zero_division=0),
                'f1_score_hub_s': metrics.f1_score(hub_s_label.cpu(), bin_predict_hub_s.cpu(), zero_division=0),
                'f1_score_hlb_t': metrics.f1_score(hlb_label.cpu(), bin_predict_hlb.cpu(), zero_division=0),
                'f1_score_hub_t': metrics.f1_score(hub_label.cpu(), bin_predict_hub.cpu(), zero_division=0)
            }

            loss_graph = sum(losses.values())
            loss_kl = kl_loss(xs_mu, xs_logsigma) + kl_loss(xt_mu, xt_logsigma)
            #loss_graph += 3*(losses['logits']+losses['weights'])
            loss = alpha * loss_graph + loss_kl

            if ep % 10 == 0:
                if not os.path.isdir('record_hlb_s_label'):
                    os.mkdir('record_hlb_s_label')
                if not os.path.isdir('record_predict_hlb_s'):
                    os.mkdir('record_predict_hlb_s')
                torch.save(hlb_s_label, os.path.join('record_hlb_s_label', str(ep)))
                torch.save(predict_hlb_s, os.path.join('record_predict_hlb_s', str(ep)))

            if mode == 'train':
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(encoder.parameters(), 1e8)
                clip_grad_norm_(decoder.parameters(), 1e8)
                optimizer.step()

                hasnan = False
                for name, par in list(decoder.named_parameters()) + list(encoder.named_parameters()):
                    if par.grad is None:
                        continue
                    if torch.isnan(par.grad).any():
                        print(name)
                        hasnan = True
                if hasnan:
                    raise Exception("NaN in gradient")

            for key in losses.keys():
                loss_counts[key] += losses[key].detach().item()

    n_data = len(dataloader)
    loss_means = {key: val / n_data for key, val in loss_counts.items()}
    loss_tot_mean = sum(loss_means.values())

    if mode == 'train':
        scheduler.step()

    if use_wandb:
        log_dict = {f"{mode}_loss_{key}": val for key, val in loss_means.items()} | {f"{mode}_loss": loss_tot_mean} | {f"{mode}_pres": pres}
        log_dict |= {f"lr": scheduler.get_last_lr()[0]}
        wandb.log(log_dict)

    return loss_tot_mean


if __name__ == '__main__':
    pathConfig = create_config()
    config = OmegaConf.load(pathConfig)
    retrain = True
    alpha = 5
    lr = 1e-2
    epoch = 200
    test_epoch = 1
    batchsize = 256
    seed = 20
    step_size = 1
    gamma=0.85
    use_wandb = True
    assert torch.cuda.is_available()
    num_gpus = torch.cuda.device_count()
    device = torch.device('cuda:0')
    benchmark = 'setcover'
    file_path = 'setcover-flat'
    encoder_input_dim_xs = config.encoder.input_dim_xs  # 替换为实际值，x_s特征的维度
    encoder_input_dim_xt = config.encoder.input_dim_xt  # 替换为实际值，x_t特征的维度
    encoder_input_dim_edge = config.encoder.input_dim_edge  # 替换为实际值，边特征的维度
    encoder_num_layers = config.encoder.num_layers  # 根据需求设置，编码器的数量
    encoder_hidden_dim = config.encoder.hidden_dim  # 根据需求设置，编码器的隐藏层维度
    encoder_mlp_hidden_dim = config.encoder.mlp_hidden_dim  # 根据需求设置，编码器的MLP隐藏层维度
    encoder_enc_dim = config.encoder.enc_dim  # 根据需求设置，编码器的输出维度
    decoder_input_dim_xs = 1  # 根据需求设置，解码器输入的维度
    decoder_input_dim_xt = 1  # 根据需求设置，解码器输入的维度
    decoder_num_layers = 1  # 根据需求设置，解码器的数量
    decoder_input_dim_edge = 1  # 根据需求设置，解码器输入的维度
    decoder_hidden_dim = 16
    decoder_mlp_hidden_dim = 16
    decoder_mlp_out_dim = 1  # 根据需求设置，解码器输出的维度
    del_ratio = 0.1
    assert 0 <= del_ratio <= 1

    if use_wandb:
        wandb.init(
            # set the wandb project where this run will be logged
            project="learning-to-features",
            name='setcover-VAE ',
            group='specific dataset',
            
            # track hyperparameters and run metadata
            config={
                "initial_learning_rate": lr,
                "architecture": "VAE",
                "epochs": epoch,
                "batch_size": batchsize,
                "alpha": alpha,
                "seed": seed,
                "encoder_input_dim_xs": encoder_input_dim_xs,
                "encoder_input_dim_xt": encoder_input_dim_xt,
                "encoder_input_dim_edge": encoder_input_dim_edge,
                "encoder_num_layers": encoder_num_layers,
                "encoder_hidden_dim": encoder_hidden_dim,
                "encoder_mlp_hidden_dim": encoder_mlp_hidden_dim,
                "encoder_enc_dim": encoder_enc_dim,
                "decoder_input_dim_xs": decoder_input_dim_xs,
                "decoder_input_dim_xt": decoder_input_dim_xt,
                "decoder_num_layers": decoder_num_layers,
                "decoder_input_dim_edge": decoder_input_dim_edge,
                "decoder_hidden_dim": decoder_hidden_dim,
                "decoder_mlp_hidden_dim": decoder_mlp_hidden_dim,
                "decoder_mlp_out_dim": decoder_mlp_out_dim,
                "del_ratio": del_ratio,
                'scheduler_step_size': step_size,
                'scheduler_gamma': gamma,
                'benchmark': benchmark
            }
        )

    train_folder = f'{benchmark}_train'
    if not os.path.isdir(train_folder):
        os.mkdir(train_folder)
    setup_seed(seed)

    lowest_loss = 100000

    # existing code
    # dataset = mipGraphDataset(root=file_path, norm=True)
    dataset = mipGraphN2N(root=file_path)
    # calculate lengths of splits
    total_len = len(dataset)
    train_len = int(total_len * 0.7)
    valid_len = int(total_len * 0.15)
    test_len = total_len - train_len - valid_len

    # create datasets
    train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_len, valid_len, test_len])
    print(len(train_dataset), len(valid_dataset), len(test_dataset))

    # create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, follow_batch=['x_s', 'x_t'])
    valid_dataloader = DataLoader(valid_dataset, batch_size=batchsize, shuffle=False, follow_batch=['x_s', 'x_t'])
    test_dataloader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False, follow_batch=['x_s', 'x_t'])

    # create model
    encoder = Encoder(input_dim_xs=encoder_input_dim_xs, input_dim_xt=encoder_input_dim_xt,
                      input_dim_edge=encoder_input_dim_edge, num_layers=encoder_num_layers,
                      hidden_dim=encoder_hidden_dim,
                      mlp_hidden_dim=encoder_mlp_hidden_dim, enc_dim=encoder_enc_dim)
    decoder = Decoder(input_dim_xs=decoder_input_dim_xs, input_dim_xt=decoder_input_dim_xt,
                      input_dim_edge=decoder_input_dim_edge, num_layers=decoder_num_layers,
                      hidden_dim=decoder_hidden_dim,
                      mlp_hidden_dim=decoder_mlp_hidden_dim, mlp_out_dim=decoder_mlp_out_dim, enc_dim=encoder_enc_dim)
    if retrain is False and os.path.exists(os.path.join(train_folder, 'encoder.pth')):
        encoder.load_state_dict(torch.load(os.path.join(train_folder, 'encoder.pth')))
    if retrain is False and os.path.exists(os.path.join(train_folder, 'decoder.pth')):
        decoder.load_state_dict(torch.load(os.path.join(train_folder, 'decoder.pth')))

    encoder.to(device)
    decoder.to(device)

    # Define the loss criterion
    regression_loss = nn.SmoothL1Loss()
    # regression_loss = nn.L1Loss()
    bce_loss = nn.BCEWithLogitsLoss()

    # Define the optimizer
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    # Define the learning rate scheduler
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    # training loop
    for ep in tqdm(range(epoch)):
        # with (autograd.detect_anomaly(check_nan=False)):
        train_loss_tot = forward_loop(encoder, decoder, train_dataloader, device, del_ratio, use_wandb, ep=ep, mode='train')
        print('Epoch: {:03d}, Training Loss: {:.5f}'.format(ep, train_loss_tot))

        # validation loop
        valid_loss_tot = forward_loop(encoder, decoder, valid_dataloader, device, del_ratio, use_wandb, ep=ep, mode='valid')
        print('Epoch: {:03d}, Validation Loss: {:.5f}'.format(ep, valid_loss_tot))

        if ep > 0 and ep % 20 == 0:
            torch.save(encoder.state_dict(), os.path.join(train_folder, 'encoder' + str(ep) + '.pth'))
            torch.save(decoder.state_dict(), os.path.join(train_folder, 'decoder' + str(ep) + '.pth'))
        if valid_loss_tot < lowest_loss:
            lowest_loss = valid_loss_tot
            torch.save(encoder.state_dict(), os.path.join(train_folder, 'encoder' + str(ep) + '.pth'))
            torch.save(decoder.state_dict(), os.path.join(train_folder, 'decoder' + str(ep) + '.pth'))

    # testing loop
    total_test_loss = 0
    for ep in tqdm(range(test_epoch)):
        test_loss_tot = forward_loop(encoder, decoder, test_dataloader, device, del_ratio, use_wandb, ep=ep, mode='test')
        print('Epoch: {:03d}, Testing Loss: {:.5f}'.format(ep, test_loss_tot))
        total_test_loss += test_loss_tot

    print('Average Testing Loss: {:.5f}'.format(total_test_loss / test_epoch))
    # Save the trained models
    torch.save(encoder.state_dict(), os.path.join(train_folder, 'encoder.pth'))
    torch.save(decoder.state_dict(), os.path.join(train_folder, 'decoder.pth'))

    if use_wandb:
        wandb.finish()
