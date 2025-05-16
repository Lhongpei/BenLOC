import sys

sys.path.append('..')
sys.path.append('../..')
sys.path.append('.')
sys.path.append('../../..')
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
from ml4moc.DL.gnn_predictor.config.setConfig import create_config
from ml4moc.DL.gnn_predictor.dataset_gen.heteroDataset import (RegrSolTimeHeteroDataset,
                                           RegrSolTimeHeteroDataset_tiny,
                                           RankListSolTimeDataset)
from ml4moc.DL.gnn_predictor.PyGmodel.tasks_pyg import (RegrNetPool, RankListNetPool)
from ml4moc.DL.gnn_predictor.trainPyG.trainer import RankListTrainer, RegrTrainer
from ml4moc.DL.gnn_predictor.utils.loss import *
from ml4moc.DL.gnn_predictor.utils.utils import *

def gnn_predictor():
    assert torch.cuda.is_available()
    device = torch.device('cuda:1')
    # if not os.path.exists('Config/config.yaml'):
    #     create_config()
    create_config()
    config = OmegaConf.load('src/config/config.yaml')
    #--------------------------------------
    # 1. Set the mode
    #--------------------------------------
    ListWise = True #FIXME change to True if you want to use listwise mode, default is False(Regr mode)
    taskMode = 'RankList' if ListWise else 'Regr'
    weighted = False #FIXME change to True if you want to use weighted loss, default is True
    use_wandb = True #FIXME change to True if you want to use wandb to record result, default is True
    scale = None #FIXME: change to the scale of the dataset you want to use, use None if you don't want to scale the dataset
    modeGNN = 'GAT' #FIXME: change to 'GCN' if you want to use GCN, change to 'GIN' if you want to use GIN
    fold = 1 #FIXME: change to the number of folds you want to use
    reproData = True #FIXME: change to True if you want to reprocess the dataset, default is False
    report_norm = 'std' 
    dataset = 'indset' 
    assert dataset in ['indset', 'setcover']
    #--------------------------------------
    # 2. Set the hyperparameters
    #--------------------------------------
    alpha = config.hyper_L2R.alpha
    lr = config.hyper_L2R.lr
    epoch = config.hyper_L2R.epoch
    test_epoch = config.hyper_L2R.test_epoch
    batchsize = config.hyper_L2R.batchsize
    stepsize = config.hyper_L2R.stepsize
    gamma = config.hyper_L2R.gamma
    pooling = config.hyper_L2R.pooling
    config_dim = config.hyper_L2R.ranklist.config_dim if ListWise else config.hyper_L2R.regression.config_dim
    tot_config_dim = config.hyper_L2R.tot_config_dim
    assert pooling in ['add', 'mean', 'max']
    
    predict_task_config = config.pyg_regression if not ListWise else config.pyg_ranklist
    
    setup_seed(config.seed)
    # torch.set_num_threads(config.cpu_threads)
    
    #--------------------------------------
    # 3. Set the file path
    #--------------------------------------
    predict_config_folder = 'setcover_config' if dataset == 'setcover' else 'indset_config'
    if not os.path.isdir(config.paths.predict_config_folder):
        os.mkdir(config.paths.predict_config_folder)
    
    if dataset == 'indset':
        file_path=config.paths.indset_folder
        report_file_root = config.paths.indset_report_root
    elif dataset == 'setcover':
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
            name = 'PyG Regr',
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
    predictNet = RegrNetPool if not ListWise else RankListNetPool
    dataset = RegrSolTimeHeteroDataset if not ListWise else RankListSolTimeDataset
    
    test_dataset = dataset(root=file_path, fold=fold, report_file_root=report_file_root, type='test', report_norm=report_norm, reprocess=reproData, transform=None)
    metadata = test_dataset[0].metadata()
    print('metadata:', metadata)
    test_time_sum = torch.zeros(tot_config_dim)
    for i in test_dataset:
        test_time_sum += report_Dict['time_Dict'][i.name]
    result = pd.DataFrame()
    
    for fold_step in range(1, fold + 1):
        try:
            train_time_sum = torch.zeros(tot_config_dim)
            valid_time_sum = torch.zeros(tot_config_dim)
            print('Processing fold:', fold_step)
            
            train_dataset = dataset(root=file_path, fold=fold_step, report_file_root=report_file_root, type='train', report_norm=report_norm, reprocess=reproData, transform=None)
            valid_dataset = dataset(root=file_path, fold=fold_step, report_file_root=report_file_root, type='val', report_norm=report_norm, reprocess=reproData, transform=None)

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
            model = predictNet(predict_task_config, metadata)
            model.to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            scheduler = StepLR(optimizer, step_size=stepsize, gamma=gamma)
            
            #--------------------------------------
            # 7. Train the model and test the model
            #--------------------------------------
            max_improve = -1000
            for ep in tqdm(range(epoch),desc=f'Fold_{fold_step} Train Epoch:'):
                cal_improve = True
                train_result = forward_loop(Loss = listnet_loss, model=model, dataloader=train_dataloader, report_Dict=report_Dict, optimizer=optimizer, scheduler=scheduler, device=device, use_wandb=use_wandb, ep=ep, fold=fold_step, 
                                            train_folder=train_folder, weighted=weighted, mode='train', cal_improve=cal_improve)
                print(f'fold_{fold_step}_train_loss:', train_result[f'fold_{fold_step}_train_loss'])
                valid_result = forward_loop(Loss = listnet_loss, model=model, dataloader=valid_dataloader, report_Dict=report_Dict, optimizer=optimizer, scheduler=scheduler, device=device, use_wandb=use_wandb, ep=ep, fold=fold_step, 
                                            train_folder=train_folder, weighted=weighted, mode='valid',cal_improve=cal_improve)
                print(f'fold_{fold_step}_valid_loss:', valid_result[f'fold_{fold_step}_valid_loss'])
                test_result = forward_loop(Loss = listnet_loss, model=model, dataloader=test_dataloader, report_Dict=report_Dict, optimizer=optimizer, scheduler=scheduler, device=device, use_wandb=use_wandb, ep=ep, fold=fold_step, 
                                            train_folder=train_folder, weighted=weighted, mode='test',cal_improve=True)
                print(f'fold_{fold_step}_test_loss:', test_result[f'fold_{fold_step}_test_loss'])
                if valid_result.get(f'fold_{fold_step}_valid_improve') is not None and valid_result[f'fold_{fold_step}_valid_improve'] > max_improve:
                    max_improve = valid_result[f'fold_{fold_step}_valid_improve']
                    valid_stop = valid_result
                    print('max_improve:', max_improve)
                    torch.save(model.state_dict(), os.path.join(train_folder,f'fold_{fold_step}_best_Model_' + taskMode + '.pth'))
                
            # for ep in tqdm(range(test_epoch)):
            #     test_result = forward_loop(model=model, dataloader=test_dataloader, report_Dict=report_Dict, optimizer=optimizer, scheduler=scheduler, device=device, use_wandb=use_wandb, ep=ep, fold=fold_step, 
            #                                 train_folder=train_folder, weighted=weighted, mode='test',cal_improve=True)
            #     print(f'fold_{fold_step}_test_loss:', test_result[f'fold_{fold_step}_test_loss'])

            # model.load_state_dict(torch.load(os.path.join(train_folder,f'fold_{fold_step}_best_Model_' + taskMode + '.pth')))
            # test_stop = forward_loop(model=model, dataloader=test_dataloader, report_Dict=report_Dict, optimizer=optimizer, scheduler=scheduler, device=device, use_wandb=use_wandb, ep=ep, fold=fold_step, 
            #                                 train_folder=train_folder, weighted=weighted, mode='test')

            result[f'fold_{fold_step}'] = [
                train_result[f'fold_{fold_step}_train_improve'].item(), train_static_improve.item(), 
                valid_result[f'fold_{fold_step}_valid_improve'].item(), valid_static_improve.item(), #valid_stop[f'fold_{fold_step}_valid_improve'].item(), 
                test_result[f'fold_{fold_step}_test_improve'].item(), test_static_improve.item()#, test_stop[f'fold_{fold_step}_test_improve'].item()
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