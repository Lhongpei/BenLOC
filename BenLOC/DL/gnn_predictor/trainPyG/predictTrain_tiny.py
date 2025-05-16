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
import torch_geometric.transforms as T
from omegaconf import OmegaConf
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import wandb
from BenLOC.DL.gnn_predictor.config.setConfig import create_config
from BenLOC.DL.gnn_predictor.dataset_gen.heteroDataset import (RegrSolTimeHeteroDataset,
                                           RegrSolTimeHeteroDataset_tiny)
from BenLOC.DL.gnn_predictor.PyGmodel.tasks_pyg import RegrNet
from BenLOC.DL.gnn_predictor.trainPyG.trainer import RankListTrainer, RegrTrainer
from BenLOC.DL.gnn_predictor.utils.loss import *
from BenLOC.DL.gnn_predictor.utils.utils import *

if __name__ == '__main__':
    assert torch.cuda.is_available()
    device = torch.device('cuda:2')
    # if not os.path.exists('Config/config.yaml'):
    #     create_config()
    create_config()
    config = OmegaConf.load('src/config/config.yaml')
    #--------------------------------------
    # 1. Set the mode
    #--------------------------------------
    ListWise = False #FIXME change to True if you want to use listwise mode, default is False(Regr mode)
    taskMode = 'RankList' if ListWise else 'Regr'
    weighted = False #FIXME change to True if you want to use weighted loss, default is True
    use_wandb = True #FIXME change to True if you want to use wandb to record result, default is True
    scale = None #FIXME: change to the scale of the dataset you want to use, use None if you don't want to scale the dataset
    modeGNN = 'GAT' #FIXME: change to 'GCN' if you want to use GCN, change to 'GIN' if you want to use GIN
    fold = 1 #FIXME: change to the number of folds you want to use
    reproData = True #FIXME: change to True if you want to reprocess the dataset, default is False
    prob_num =10
    config_num = 24
    #--------------------------------------
    # 2. Set the hyperparameters
    #--------------------------------------
    alpha = config.hyper_L2R.alpha
    lr = config.hyper_L2R.lr
    epoch = config.hyper_L2R.epoch
    test_epoch = config.hyper_L2R.test_epoch
    batchsize = 2 #config.hyper_L2R.batchsize
    stepsize = config.hyper_L2R.stepsize
    gamma = config.hyper_L2R.gamma
    pooling = config.hyper_L2R.pooling
    config_dim = config.hyper_L2R.listwise.config_dim if ListWise else config.hyper_L2R.pairwise.config_dim
    tot_config_dim = config.hyper_L2R.tot_config_dim
    assert pooling in ['add', 'mean', 'max']
    
    predict_task_config = config.pyg_regression
    
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
            name = 'Regr Tiny',
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
                'pooling': config.hyper_L2R.pooling,
            }
        )
    
    #--------------------------------------
    # 5. train
    #--------------------------------------
    print(f'Start training {taskMode} model')
    report_Dict = reportDict(report_file_root)
    forward_loop = RegrTrainer if not ListWise else RankListTrainer
    predictNet = RegrNet
    dataset = RegrSolTimeHeteroDataset_tiny
    
    for fold_step in range(1, fold + 1):
        try:
            train_time_sum = torch.zeros(tot_config_dim)
            valid_time_sum = torch.zeros(tot_config_dim)
            print('Processing fold:', fold_step)
            
            train_dataset = dataset(root=file_path, fold=fold_step, report_file_root=report_file_root, type='train', report_norm=None, reprocess=reproData, transform=T.ToUndirected(), prob_num=prob_num, config_num=config_num)
            metadata = train_dataset[0].metadata()
            #valid_dataset = dataset(root=file_path, fold=fold_step, report_file_root=report_file_root, type='val', report_norm=None, reprocess=reproData, transform=T.ToUndirected(), prob_num=prob_num, config_num=config_num)

            if scale is not None:
                # test_dataset,_ = random_split(test_dataset, [int(len(test_dataset)*scale), len(test_dataset)-int(len(test_dataset)*scale)])
                # valid_dataset,_ = random_split(valid_dataset, [int(len(valid_dataset)*scale), len(valid_dataset)-int(len(valid_dataset)*scale)])
                train_dataset,_ = random_split(train_dataset, [int(len(train_dataset)*scale), len(train_dataset)-int(len(train_dataset)*scale)])
                
            for i in train_dataset:
                train_time_sum += report_Dict['time_Dict'][i.name]
                
            
            train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, follow_batch=['x_s', 'x_t'])

            
            #--------------------------------------
            # 6. Set the model and optimizer
            #--------------------------------------
            model = predictNet(predict_task_config, metadata)
            model.to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            scheduler = StepLR(optimizer, step_size=stepsize, gamma=gamma)
            
            #--------------------------------------
            # 7. Train the model and test the model
            #--------------------------------------
            max_improve = -1000
            for ep in tqdm(range(epoch),desc=f'Fold_{fold_step} Train Epoch:'):
                train_result = forward_loop(model=model, dataloader=train_dataloader, report_Dict=report_Dict, optimizer=optimizer, scheduler=scheduler, device=device, use_wandb=use_wandb, ep=ep, fold=fold_step, 
                                            train_folder=train_folder, weighted=weighted, mode='train')
                print(f'fold_{fold_step}_train_loss:', train_result[f'fold_{fold_step}_train_loss'])
                
        except Exception as e:
            print(f'fold_{fold_step} failed:')
            traceback.print_exc()
            

                
    #--------------------------------------
    # 8. Save the result
    #--------------------------------------
    