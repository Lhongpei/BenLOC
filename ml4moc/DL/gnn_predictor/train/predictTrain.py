import sys

sys.path.append('..')
sys.path.append('../..')
sys.path.append('.')
import os
import random
import traceback
import warnings

import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import wandb
from ml4moc.DL.gnn_predictor.config.setConfig import create_config
from ml4moc.DL.gnn_predictor.dataset_gen.predictDataset import (RankListSolTimeDataset,
                                            RegrSolTimeDataset)
from ml4moc.DL.gnn_predictor.models.tasks import RankListNet, RegrNet
from ml4moc.DL.gnn_predictor.models.vae_models import Encoder
from ml4moc.DL.gnn_predictor.train.trainer import RankListTrainer, RegrTrainer
from ml4moc.DL.gnn_predictor.utils.loss import *
from ml4moc.DL.gnn_predictor.utils.utils import *

if __name__ == '__main__':
    assert torch.cuda.is_available()
    device = torch.device('cuda:3')
    # if not os.path.exists('Config/config.yaml'):
    #     create_config()
    create_config()
    config = OmegaConf.load('src/config/config.yaml')
    #--------------------------------------
    # 1. Set the mode
    #--------------------------------------
    ListWise = True #FIXME change to True if you want to use listwise mode, default is False(Regr mode)
    taskMode = 'RankList' if ListWise else 'Regr'
    pretrain = False #FIXME change to True if you want to pre-train the model using the pre-trained encoder, default is True
    reparam = False #FIXME change to True if you want to use reparameterization, default is False
    fineTune = False #FIXME change to True if you want to fine-tune the model, default is True
    weighted = False #FIXME change to True if you want to use weighted loss, default is True
    use_wandb = True #FIXME change to True if you want to use wandb to record result, default is True
    scale = None #FIXME: change to the scale of the dataset you want to use, use None if you don't want to scale the dataset
    modeGNN = 'GCN' #FIXME: change to 'GCN' if you want to use GCN, change to 'GIN' if you want to use GIN
    fold = 1 #FIXME: change to the number of folds you want to use
    reproData = False #FIXME: change to True if you want to reprocess the dataset, default is False
    if not pretrain:
        reparam = False
        fineTune = False
        
    #--------------------------------------
    # 2. Set the hyperparameters
    #--------------------------------------
    alpha = config.hyper_L2R.alpha
    lr = config.hyper_L2R.lr
    epoch = config.hyper_L2R.epoch
    test_epoch = config.hyper_L2R.test_epoch
    batchsize = 8#config.hyper_L2R.batchsize
    stepsize = config.hyper_L2R.stepsize
    gamma = config.hyper_L2R.gamma
    pooling = config.hyper_L2R.pooling
    config_dim = config.hyper_L2R.listwise.config_dim if ListWise else config.hyper_L2R.pairwise.config_dim
    tot_config_dim = config.hyper_L2R.tot_config_dim
    assert pooling in ['add', 'mean', 'max']
    
    encoder_input_dim_xs = config.encoder.input_dim_xs
    encoder_input_dim_xt = config.encoder.input_dim_xt
    encoder_input_dim_edge = config.encoder.input_dim_edge
    encoder_num_layers = config.encoder.num_layers
    encoder_hidden_dim = config.encoder.hidden_dim
    encoder_mlp_hidden_dim = config.encoder.mlp_hidden_dim
    encoder_enc_dim = config.encoder.enc_dim
    
    predict_hidden_dim = config.predict_L2R.hidden_dim
    predict_mlp_hidden_dim = config.predict_L2R.mlp_hidden_dim
    predict_num_layers = config.predict_L2R.num_layers
    
    setup_seed(config.seed)
    torch.set_num_threads(config.cpu_threads)
    
    #--------------------------------------
    # 3. Set the file path
    #--------------------------------------
    predict_config_folder = 'setcover_config'
    if not os.path.isdir(config.paths.predict_config_folder):
        os.mkdir(config.paths.predict_config_folder)
        
    file_path=config.paths.setcover_folder
    report_file_root = config.paths.setcover_report_root
    train_folder = config.paths.train_folder
    if not os.path.isdir(train_folder):
        os.mkdir(train_folder)
    result_folder = config.paths.result_folder
    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)
    #--------------------------------------
    # 4. Set the items of wandb
    #--------------------------------------
    if use_wandb:
        wandb.init(
            group='setcover config_embeded',
            name = 'self-imple ListNet',
            # set the wandb project where this run will be logged
            project="learning-to-mapping",
            # track hyperparameters and run metadata
            config={
                "learning_rate": config.hyper_L2R.lr,
                "epochs": config.hyper_L2R.epoch,
                "batch_size": config.hyper_L2R.batchsize,
                "weight": weighted,
                "layer": config.predict_L2R.num_layers,
                "dim": config.predict_L2R.hidden_dim,
                "stepsize": config.hyper_L2R.stepsize,
                "gamma": config.hyper_L2R.gamma,
                "fineTune": fineTune,
                'pretrain': pretrain,
                'reparam': reparam,
                'pooling': config.hyper_L2R.pooling,
            }
        )
    
    #--------------------------------------
    # 5. train
    #--------------------------------------
    print(f'Start training {taskMode} model')
    report_Dict = reportDict(report_file_root)
    forward_loop = RegrTrainer if not ListWise else RankListTrainer
    predictNet = RegrNet if not ListWise else RankListNet
    dataset = RegrSolTimeDataset if not ListWise else RankListSolTimeDataset
    
    test_dataset = dataset(root=file_path, fold=fold, report_file_root=report_file_root, type='test', report_norm=None, reprocess=reproData)
    test_time_sum = torch.zeros(tot_config_dim)
    for i in test_dataset:
        test_time_sum += report_Dict['time_Dict'][i.name]
    result = pd.DataFrame()
    
    for fold_step in range(1, fold + 1):
        try:
            train_time_sum = torch.zeros(tot_config_dim)
            valid_time_sum = torch.zeros(tot_config_dim)
            print('Processing fold:', fold_step)
            
            train_dataset = dataset(root=file_path, fold=fold_step, report_file_root=report_file_root, type='train', report_norm=None, reprocess=reproData)
            valid_dataset = dataset(root=file_path, fold=fold_step, report_file_root=report_file_root, type='val', report_norm=None, reprocess=reproData)
            
            if scale is not None:
                # test_dataset,_ = random_split(test_dataset, [int(len(test_dataset)*scale), len(test_dataset)-int(len(test_dataset)*scale)])
                # valid_dataset,_ = random_split(valid_dataset, [int(len(valid_dataset)*scale), len(valid_dataset)-int(len(valid_dataset)*scale)])
                train_dataset,_ = random_split(train_dataset, [int(len(train_dataset)*scale), len(train_dataset)-int(len(train_dataset)*scale)])
                
            for i in train_dataset:
                train_time_sum += report_Dict['time_Dict'][i.name]
            for i in valid_dataset:
                valid_time_sum += report_Dict['time_Dict'][i.name]
                
            choose = torch.argmin(train_time_sum).item()
            train_static_improve = (train_time_sum[0] - train_time_sum[choose])/train_time_sum[0]#FIXME
            valid_static_improve = (valid_time_sum[0] - valid_time_sum[choose])/valid_time_sum[0]#FIXME
            test_static_improve = (test_time_sum[0] - test_time_sum[choose])/test_time_sum[0]#FIXME
            
            train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, follow_batch=['x_s', 'x_t'])
            valid_dataloader = DataLoader(valid_dataset, batch_size=batchsize, shuffle=True, follow_batch=['x_s', 'x_t'])
            test_dataloader = DataLoader(test_dataset, batch_size=batchsize, shuffle=True, follow_batch=['x_s', 'x_t'])
            
            #--------------------------------------
            # 6. Set the model and optimizer
            #--------------------------------------
            if pretrain:
                encoder=Encoder(input_dim_xs=encoder_input_dim_xs,input_dim_xt=encoder_input_dim_xt,
                                input_dim_edge=encoder_input_dim_edge,num_layers=encoder_num_layers,
                                hidden_dim=encoder_hidden_dim,mlp_hidden_dim=encoder_mlp_hidden_dim,enc_dim=encoder_enc_dim)
                
                model = predictNet(input_dim_xs=encoder_enc_dim, input_dim_xt=encoder_enc_dim, input_dim_edge=1, 
                                num_layers=predict_num_layers, hidden_dim=predict_hidden_dim, mlp_hidden_dim=predict_mlp_hidden_dim, 
                                task_dim=config_dim, pooling=pooling, GNN = modeGNN)
                
                encoder.load_state_dict(torch.load(config.paths.savedEncoder))
                encoder.to(device)
                if not fineTune:
                    encoder.eval()
            else:
                encoder = None
                reparam = False
                model = predictNet(input_dim_xs=encoder_input_dim_xs, input_dim_xt=encoder_input_dim_xt, input_dim_edge=1, 
                                   num_layers=predict_num_layers, hidden_dim=predict_hidden_dim, mlp_hidden_dim=predict_mlp_hidden_dim, 
                                   task_dim=config_dim, pooling=pooling, GNN = modeGNN)
            model.to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            scheduler = StepLR(optimizer, step_size=stepsize, gamma=gamma)
            
            #--------------------------------------
            # 7. Train the model and test the model
            #--------------------------------------
            max_improve = -1000
            for ep in tqdm(range(epoch),desc=f'Fold_{fold_step} Train Epoch:'):
                train_result = forward_loop(model, encoder, train_dataloader, report_Dict, optimizer, scheduler, device, use_wandb, ep, fold=fold_step, 
                                            train_folder=train_folder, weighted=weighted, reparam=reparam, pretrain=pretrain, mode='train')
                print(f'fold_{fold_step}_train_loss:', train_result[f'fold_{fold_step}_train_loss'])
                valid_result = forward_loop(model, encoder, valid_dataloader, report_Dict, optimizer, scheduler, device, use_wandb, ep, fold=fold_step, 
                                            train_folder=train_folder, weighted=weighted, reparam=reparam, pretrain=pretrain, mode='valid')
                print(f'fold_{fold_step}_valid_loss:', valid_result[f'fold_{fold_step}_valid_loss'])
                
                if valid_result[f'fold_{fold_step}_valid_improve'] > max_improve:
                    max_improve = valid_result[f'fold_{fold_step}_valid_improve']
                    valid_stop = valid_result
                    print('max_improve:', max_improve)
                    torch.save(model.state_dict(), os.path.join(train_folder,f'fold_{fold_step}_best_Model_' + taskMode + '.pth'))
                
            for ep in tqdm(range(test_epoch)):
                test_result = forward_loop(model, encoder, test_dataloader, report_Dict, optimizer, scheduler, device, use_wandb, ep, fold=fold_step, 
                                            train_folder=train_folder, weighted=weighted, reparam=reparam, pretrain=pretrain, mode='test')
                print(f'fold_{fold_step}_test_loss:', test_result[f'fold_{fold_step}_test_loss'])

            model.load_state_dict(torch.load(os.path.join(train_folder,f'fold_{fold_step}_best_Model_' + taskMode + '.pth')))
            test_stop = forward_loop(model, encoder, test_dataloader, report_Dict, optimizer, scheduler, device, use_wandb, ep, fold=fold_step, 
                                            train_folder=train_folder, weighted=weighted, reparam=reparam, pretrain=pretrain, mode='test')

            result[f'fold_{fold_step}'] = [
                train_result[f'fold_{fold_step}_train_improve'].item(), train_static_improve.item(), 
                valid_result[f'fold_{fold_step}_valid_improve'].item(), valid_static_improve.item(), valid_stop[f'fold_{fold_step}_valid_improve'].item(), 
                test_result[f'fold_{fold_step}_test_improve'].item(), test_static_improve.item(), test_stop[f'fold_{fold_step}_test_improve'].item()
                                        ] 
        except Exception as e:
            print(f'fold_{fold_step} failed:')
            traceback.print_exc()
            

                
    #--------------------------------------
    # 8. Save the result
    #--------------------------------------
    result = result.transpose()
    columns = pd.MultiIndex.from_tuples([
    ('train', 'improve'), ('train', 'static'),
    ('valid', 'improve'), ('valid', 'static'), ('valid', 'stop'),
    ('test', 'improve'), ('test', 'static'), ('test', 'stop')
    ])

    result.columns = columns
    result.to_csv(os.path.join(result_folder, 'rank_result.csv'))
    print(result)
    latex = df_to_custom_latex(result, 'Learning To Rank Result', os.path.join(result_folder, 'rank_result_latex.tex'))