import os
import torch
import pandas as pd
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from BenLOC.DL.gnn_pairwise.dataset_gen.heteroDataset import HeteroDataset
from BenLOC.DL.gnn_pairwise.models.gnn import BiparGAT
from BenLOC.extract_feature.vae_extractor.trainer import train_loop, predict_loop
from BenLOC.DL.gnn_pairwise.utils.loss import soft_l1_regr_loss
from tqdm import tqdm
import wandb
from BenLOC.DL.gnn_pairwise.config.setConfig import create_config
from BenLOC.DL.gnn_pairwise.utils.utils import setup_seed


class GNNTrainer:
    def __init__(self, params):
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        self.params = params
        self.train_folder = params.save_model_path
        self.result_folder = params.result_root_path
        self.problem_root_path = params.problem_root_path
        self.report_root_path = params.report_root_path
        self.loss_function = soft_l1_regr_loss
        self.fold = params.fold
        self.repro_data = params.reproData
        self.default_index = params.default_index
        self.nn_config = None
        self.metadata = None
        self.use_wandb = params.use_wandb
        self.mode_gnn = params.modeGNN

        if self.use_wandb:
            wandb.init(
                group='predict_loop',
                name='indset-all_minimax',
                project="learning-to-mapping",
            )

    def initialize(self):
        config_path = create_config(self.params.repo_path)
        config = OmegaConf.load(config_path)
        self.nn_config = config.nn_config
        setup_seed(config.seed)

    def load_data(self, fold_step):
        train_dataset = HeteroDataset(
            root=self.problem_root_path, 
            report_file_root=self.report_root_path, 
            fold=fold_step, 
            type='train'
        )
        if self.repro_data:
            train_dataset.process()
            train_dataset = HeteroDataset(
                root=self.problem_root_path, 
                report_file_root=self.report_root_path, 
                fold=fold_step, 
                type='train'
            )
        self.metadata = train_dataset[0]
        test_dataset = HeteroDataset(
            root=self.problem_root_path, 
            report_file_root=self.report_root_path, 
            fold=fold_step, 
            type='test'
        )
        if self.repro_data:
            test_dataset.process()
            test_dataset = HeteroDataset(
                root=self.problem_root_path, 
                report_file_root=self.report_root_path, 
                fold=fold_step, 
                type='test'
            )

        train_dataset, valid_dataset = random_split(
            train_dataset, [int(len(train_dataset) * 0.8), len(train_dataset) - int(len(train_dataset) * 0.8)]
        )
        return train_dataset, valid_dataset, test_dataset

    def create_data_loaders(self, train_dataset, valid_dataset, test_dataset):
        train_loader = DataLoader(train_dataset, batch_size=self.params.batchsize, shuffle=True, follow_batch=['x_s', 'x_t'])
        valid_loader = DataLoader(valid_dataset, batch_size=self.params.batchsize, shuffle=True, follow_batch=['x_s', 'x_t'])
        test_loader = DataLoader(test_dataset, batch_size=self.params.batchsize, shuffle=True, follow_batch=['x_s', 'x_t'])
        return train_loader, valid_loader, test_loader

    def train_and_evaluate(self, fold_step):
        train_dataset, valid_dataset, test_dataset = self.load_data(fold_step)
        train_loader, valid_loader, test_loader = self.create_data_loaders(train_dataset, valid_dataset, test_dataset)

        model = BiparGAT(self.nn_config, self.metadata)
        model.to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.params.lr)
        scheduler = StepLR(optimizer, step_size=self.params.stepsize, gamma=self.params.gamma)

        max_improve = -1000
        for epoch in tqdm(range(self.params.epoch), desc=f'Fold_{fold_step} Train Epoch:'):
            train_result = train_loop(
                Loss=self.loss_function, model=model, dataloader=train_loader,
                optimizer=optimizer, scheduler=scheduler, device=self.device,
                use_wandb=self.use_wandb, ep=epoch, fold=fold_step,
                train_folder=self.train_folder, mode='train',
                cal_improve=True, default_index=self.default_index
            )
            valid_result = train_loop(
                Loss=self.loss_function, model=model, dataloader=valid_loader,
                optimizer=optimizer, scheduler=scheduler, device=self.device,
                use_wandb=self.use_wandb, ep=epoch, fold=fold_step,
                train_folder=self.train_folder, mode='valid',
                cal_improve=True, default_index=self.default_index
            )
            if valid_result.get(f'fold_{fold_step}_valid_improve', 0) > max_improve:
                max_improve = valid_result[f'fold_{fold_step}_valid_improve']
                torch.save(model.state_dict(), os.path.join(self.train_folder, f'fold_{fold_step}_best_Model.pth'))

        return model

    def save_results(self, model, fold_step, train_loader, valid_loader, test_loader):
        model.load_state_dict(torch.load(os.path.join(self.train_folder, f'fold_{fold_step}_best_Model.pth')))
        result_dict = {}
        for mode, loader in zip(['train', 'valid', 'test'], [train_loader, valid_loader, test_loader]):
            choice, time = predict_loop(model, loader, self.device)
            result_dict[f'fold_{fold_step}_{mode}_choice'] = choice
            result_dict[f'fold_{fold_step}_{mode}_time'] = time

        result_df = pd.DataFrame(result_dict)
        result_df.to_csv(os.path.join(self.result_folder, f'config_result_fold_{fold_step}.csv'))

    def run(self):
        self.initialize()
        for fold_step in range(1, self.fold + 1):
            print(f'Processing fold: {fold_step}')
            model = self.train_and_evaluate(fold_step)
            train_loader, valid_loader, test_loader = self.load_data(fold_step)
            self.save_results(model, fold_step, train_loader, valid_loader, test_loader)
