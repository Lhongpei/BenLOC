import sys
import os
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)

current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
src_path = os.path.dirname(parent_dir)
repo_path = os.path.dirname(src_path) #src

sys.path.append(parent_dir)
sys.path.append(src_path)
sys.path.append(repo_path)


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
from ml4moc.DL.gnn_predictor.dataset_gen.heteroDataset_temp import FoldManager
from ml4moc.DL.gnn_predictor.dataset_gen.heteroDataset import (RegrSolTimeHeteroDataset,
                                           RegrSolTimeHeteroDataset_tiny,
                                           RankListSolTimeDataset,
                                           RankListConfigNodesDataset)
from ml4moc.DL.gnn_predictor.PyGmodel.tasks_pyg import (RegrNetPool, RankListNetPool, RankListNetNode)
from ml4moc.DL.gnn_predictor.trainPyG.trainer import RankListTrainer, RegrTrainer, Predictor
from ml4moc.DL.gnn_predictor.utils.loss import *
from ml4moc.DL.gnn_predictor.utils.utils import *

if __name__ == '__main__':
    assert torch.cuda.is_available()
    device = torch.device('cuda:1')
    # if not os.path.exists('Config/config.yaml'):
    #     create_config()
    config_path = create_config(repo_path)
    config = OmegaConf.load(config_path)
    
    ListWise = True #FIXME change to True if you want to use listwise mode, default is False(Regr mode)
    taskMode = 'RankList' if ListWise else 'Regr'
    modeGNN = 'GAT' #FIXME: change to 'GCN' if you want to use GCN, change to 'GIN' if you want to use GIN
    fold = 5 #FIXME: change to the number of folds you want to use
    reproData = False #FIXME: change to True if you want to reprocess the dataset, default is False
    report_norm = None 
    #--------------------------------------
    # 2. Set the hyperparameters
    #--------------------------------------
    alpha = config.hyper_L2R.alpha
    lr = config.hyper_L2R.lr
    epoch = 1
    test_epoch = config.hyper_L2R.test_epoch
    batchsize = config.hyper_L2R.batchsize
    stepsize = config.hyper_L2R.stepsize
    gamma = config.hyper_L2R.gamma
    pooling = config.hyper_L2R.pooling
    config_dim = config.hyper_L2R.ranklist.config_dim if ListWise else config.hyper_L2R.regression.config_dim
    tot_config_dim = config.hyper_L2R.tot_config_dim
    assert pooling in ['add', 'mean', 'max']
    
    predict_task_config = config.pyg_regression if not ListWise else config.pyg_ranklist_node
    
    setup_seed(config.seed)
    # torch.set_num_threads(config.cpu_threads)
    
    #--------------------------------------
    # 3. Set the file path
    #--------------------------------------
    file_path=config.paths.mixed_folder
    report_file_root = config.paths.mixed_report_root
    train_folder = config.paths.train_folder
    print(file_path, report_file_root, train_folder)
    if not os.path.isdir(train_folder):
        os.mkdir(train_folder)
    result_folder = config.paths.result_folder
    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)
    
    predictNet = RankListNetNode
    fold_manager = FoldManager(root=file_path, report_file_root=report_file_root,store_root=file_path, reprocess=reproData, report_norm=None)
    
    test_dataset = fold_manager.get_dataset_for_fold(fold=1 ,type='test', dataset_type='ranklist_config_nodes') #(root=file_path, fold=fold, report_file_root=report_file_root, type='test', report_norm=report_norm, reprocess=reproData, transform=None)
    metadata = test_dataset[0].metadata()
    print('metadata:', metadata)
    result = pd.DataFrame()
    config_result={}
    for fold_step in range(1, fold + 1):
        try:
            print('Processing fold:', fold_step)
            
            train_dataset = fold_manager.get_dataset_for_fold(fold=fold_step ,type='train', dataset_type='ranklist_config_nodes')
            valid_dataset = fold_manager.get_dataset_for_fold(fold=fold_step ,type='val', dataset_type='ranklist_config_nodes')
            test_dataset = fold_manager.get_dataset_for_fold(fold=fold_step ,type='test', dataset_type='ranklist_config_nodes')
            train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
            valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
            test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
            
            #--------------------------------------
            # 6. Set the model and optimizer
            #--------------------------------------
            model = predictNet(predict_task_config, metadata)
            model.to(device)
            model.eval()
            #saved_state_dict = torch.load(os.path.join(train_folder, f'fold_{fold_step}_Model_RankList.pth'))
            # for key, value in saved_state_dict.items():
            #     print(key, value.shape)
            # print("Model's state_dict:")
            # for param_tensor in model.state_dict():
            #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())

            

            config_result[f'fold_{fold_step}_train'] = Predictor(model,train_dataloader,device)
            config_result[f'fold_{fold_step}_val'] = Predictor(model,valid_dataloader,device)
            config_result[f'fold_{fold_step}_test'] = Predictor(model,test_dataloader,device)
                

        except Exception as e:
            print(f'fold_{fold_step} failed:')
            traceback.print_exc()
            
config_result_df = pd.DataFrame(config_result)
config_result_df.to_csv(os.path.join(result_folder, 'config_result.csv'))
   