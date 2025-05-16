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
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

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
from ml4moc.DL.gnn_pairwise.config.setConfig import create_config
from ml4moc.DL.gnn_pairwise.dataset_gen.heteroDataset import HeteroDataset
from ml4moc.DL.gnn_pairwise.models.gnn import BiparGAT
from ml4moc.DL.gnn_pairwise.train.trainer import train_loop, predict_loop
from ml4moc.DL.gnn_pairwise.utils.loss import *
from ml4moc.DL.gnn_pairwise.utils.utils import *
import argparse
def parse_arguments():
    parser = argparse.ArgumentParser(description="Configuration for training the GNN model.")

    # Configuration for the main script
    parser.add_argument('--use_wandb', type=bool, default=True, help="Set to True to use wandb for result recording.")
    parser.add_argument('--modeGNN', type=str, choices=['GAT', 'GCN', 'GIN'], default='GAT', help="Select GNN mode (GAT, GCN, or GIN).")
    parser.add_argument('--fold', type=int, default=2, help="Number of folds to use.")
    parser.add_argument('--reproData', type=bool, default=False, help="Set to True to reprocess the dataset.")
    parser.add_argument('--default_index', type=int, default=7, help="Index of the default configuration.")
    parser.add_argument('--report_root_path', type=str, default='repo/fold_0822/indset', help="Path to the report root directory.")
    parser.add_argument('--problem_root_path', type=str, default='./ml_presolved_prob_0720', help="Path to the problem root directory.")
    parser.add_argument('--result_root_path', type=str, default='./indset_all_result', help="Path to the result root directory.")
    parser.add_argument('--save_model_path', type=str, default='./best_model_indset', help="Path to save the best model.")
    
    # Hyperparameters
    parser.add_argument('--alpha', type=float, required=True, help="Alpha value for L2R.")
    parser.add_argument('--lr', type=float, required=True, help="Learning rate for training.")
    parser.add_argument('--epoch', type=int, required=True, help="Number of epochs.")
    parser.add_argument('--batchsize', type=int, required=True, help="Batch size for training.")
    parser.add_argument('--stepsize', type=int, required=True, help="Step size for StepLR scheduler.")
    parser.add_argument('--gamma', type=float, required=True, help="Gamma value for StepLR scheduler.")


    return parser.parse_args()

def gnn_pairwise():
    assert torch.cuda.is_available()
    device = torch.device('cuda:1')
    config_path = create_config(repo_path)
    config = OmegaConf.load(config_path)
    #--------------------------------------v
    # Set the mode
    #--------------------------------------
    Loss = soft_l1_regr_loss

    parse = parse_arguments()

    use_wandb = parse.use_wandb
    modeGNN = parse.modeGNN
    fold = parse.fold
    reproData = parse.reproData
    default_index = parse.default_index
    report_root_path = parse.report_root_path
    problem_root_path = parse.problem_root_path
    train_folder = parse.save_model_path
    result_folder = parse.result_root_path

    nn_config = config.nn_config
    setup_seed(config.seed)


    if use_wandb:
        wandb.init(
            group='predict_loop',
            name = 'indset-all_minimax',

            project="learning-to-mapping",
        )

    #--------------------------------------
    # train
    #--------------------------------------

    forward_loop = train_loop
    predictNet = BiparGAT
    result = pd.DataFrame()

    for fold_step in range(1, fold + 1):
        config_result = {}

        print('Processing fold:', fold_step)

        train_dataset = HeteroDataset(root = problem_root_path, report_file_root=report_root_path, fold=fold_step, type='train')
        print(train_dataset.report_file)
        if reproData:
            train_dataset.process()
            train_dataset = HeteroDataset(root = problem_root_path, report_file_root=report_root_path, fold=fold_step, type='train')
        metadata = train_dataset[0]

        test_dataset = HeteroDataset(root = problem_root_path, report_file_root=report_root_path, fold=fold_step, type='test')
        if reproData:
            test_dataset.process()
            test_dataset = HeteroDataset(root = problem_root_path, report_file_root=report_root_path, fold=fold_step, type='test')

        train_dataset, valid_dataset = random_split(train_dataset, [int(len(train_dataset)*0.8), len(train_dataset)-int(len(train_dataset)*0.8)])
        train_dataloader = DataLoader(train_dataset, batch_size=config.predict_task.batchsize, shuffle=True, follow_batch=['x_s', 'x_t'])
        valid_dataloader = DataLoader(valid_dataset, batch_size=config.predict_task.batchsize, shuffle=True, follow_batch=['x_s', 'x_t'])
        test_dataloader = DataLoader(test_dataset, batch_size=config.predict_task.batchsize, shuffle=True, follow_batch=['x_s', 'x_t'])

        #--------------------------------------
        # Set the model and optimizer
        #--------------------------------------
        model = predictNet(nn_config, metadata)
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=config._predict_task.steplr.lr)
        scheduler = StepLR(optimizer, step_size=config.predict_task.steplr.stepsize, gamma=config.predict_task.steplr.gamma )

        #--------------------------------------
        # Train the model and test the model
        #--------------------------------------
        max_improve = -1000
        for ep in tqdm(range(config.predict_task.epoch),desc=f'Fold_{fold_step} Train Epoch:'):
            cal_improve = True
            train_result = forward_loop(Loss = Loss, model=model, dataloader=train_dataloader, optimizer=optimizer, scheduler=scheduler, device=device, use_wandb=use_wandb, ep=ep, fold=fold_step,
                                        train_folder=train_folder, mode='train', cal_improve=cal_improve, default_index=default_index)
            print(f'fold_{fold_step}_train_loss:', train_result[f'fold_{fold_step}_train_loss'])
            valid_result = forward_loop(Loss = Loss, model=model, dataloader=valid_dataloader, optimizer=optimizer, scheduler=scheduler, device=device, use_wandb=use_wandb, ep=ep, fold=fold_step,
                                        train_folder=train_folder, mode='valid',cal_improve=cal_improve, default_index=default_index)
            print(f'fold_{fold_step}_valid_loss:', valid_result[f'fold_{fold_step}_valid_loss'])
            test_result = forward_loop(Loss = Loss, model=model, dataloader=test_dataloader, optimizer=optimizer, scheduler=scheduler, device=device, use_wandb=use_wandb, ep=ep, fold=fold_step,
                                        train_folder=train_folder, mode='test',cal_improve=True, default_index=default_index)
            print(f'fold_{fold_step}_test_loss:', test_result[f'fold_{fold_step}_test_loss'])
            if valid_result.get(f'fold_{fold_step}_valid_improve') is not None and valid_result[f'fold_{fold_step}_valid_improve'] > max_improve:
                max_improve = valid_result[f'fold_{fold_step}_valid_improve']
                valid_stop = valid_result
                print('max_improve:', max_improve)
                torch.save(model.state_dict(), os.path.join('best_model_indset',f'fold_{fold_step}_best_Model' + '.pth'))

        torch.save(model.state_dict(), os.path.join(train_folder,f'fold_{fold_step}_Model' + '.pth'))

        model.load_state_dict(torch.load(os.path.join(train_folder,f'fold_{fold_step}_best_Model' + '.pth')))
        train_choice, train_time = predict_loop(model, train_dataloader, device)
        config_result[f'fold_{fold_step}_train_choice'] = train_choice
        config_result[f'fold_{fold_step}_train_time'] = train_time
        valid_choice, valid_time = predict_loop(model, valid_dataloader, device)
        config_result[f'fold_{fold_step}_valid_choice'] = valid_choice
        config_result[f'fold_{fold_step}_valid_time'] = valid_time
        test_choice, test_time = predict_loop(model, test_dataloader, device)
        config_result[f'fold_{fold_step}_test_choice'] = test_choice
        config_result[f'fold_{fold_step}_test_time'] = test_time
        config_result_df = pd.DataFrame(config_result)
        config_result_df.to_csv(os.path.join(result_folder, f'config_result_fold_{fold_step}.csv'))
        result[f'fold_{fold_step}'] = [
            train_result[f'fold_{fold_step}_train_improve'].item(),
            valid_result[f'fold_{fold_step}_valid_improve'].item(),
            test_result[f'fold_{fold_step}_test_improve'].item()
                                    ]




    # #--------------------------------------
    # # 8. Save the result
    # #--------------------------------------
    # result = result.transpose()
    # columns = pd.MultiIndex.from_tuples([
    # ('train', 'improve'), #('train', 'static'),
    # ('valid', 'improve'), #('valid', 'static'), ('valid', 'stop'),
    # ('test', 'improve')#, ('test', 'static'), ('test', 'stop')
    # ])

    # result.columns = columns
    # result.to_csv(os.path.join(result_folder, 'rank_result.csv'))
    # print(result)
    # latex = df_to_custom_latex(result, 'Learning To Rank Result', os.path.join(result_folder, 'rank_result_latex.tex'))