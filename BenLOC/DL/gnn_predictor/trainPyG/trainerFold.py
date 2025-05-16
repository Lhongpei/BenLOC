import torch
from torch_geometric.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import os
class ModelTrainer:
    def __init__(self, dataset, predictNet, Loss, lr, stepsize, gamma, batchsize, epoch, device, train_folder, weighted, use_wandb, report_Dict, report_norm, reproData, scale=None):
        self.dataset = dataset
        self.predictNet = predictNet
        self.Loss = Loss
        self.lr = lr
        self.stepsize = stepsize
        self.gamma = gamma
        self.batchsize = batchsize
        self.epoch = epoch
        self.device = device
        self.train_folder = train_folder
        self.weighted = weighted
        self.use_wandb = use_wandb
        self.report_Dict = report_Dict
        self.report_norm = report_norm
        self.reproData = reproData
        self.scale = scale

    def train_model(self, fold):
        for fold_step in range(1, fold + 1):
            print('Processing fold:', fold_step)
            train_dataset, valid_dataset, test_dataset = self.setup_datasets(fold_step)
            train_dataloader, valid_dataloader, test_dataloader = self.setup_dataloaders(train_dataset, valid_dataset, test_dataset)
            model, optimizer, scheduler = self.setup_model_optim_scheduler(fold_step)

            max_improve = -1000
            for ep in tqdm(range(self.epoch), desc=f'Fold_{fold_step} Train Epoch:'):
                train_result = self.forward_loop(model, train_dataloader, optimizer, scheduler, 'train', ep, fold_step)
                valid_result = self.forward_loop(model, valid_dataloader, optimizer, scheduler, 'valid', ep, fold_step)
                test_result = self.forward_loop(model, test_dataloader, optimizer, scheduler, 'test', ep, fold_step)

                if valid_result > max_improve:
                    max_improve = valid_result
                    torch.save(model.state_dict(), os.path.join(self.train_folder, f'fold_{fold_step}_best_Model.pth'))

    def setup_datasets(self, fold_step):
        # Setup your train, valid, and test datasets here
        pass

    def setup_dataloaders(self, train_dataset, valid_dataset, test_dataset):
        # Setup your data loaders here
        pass

    def setup_model_optim_scheduler(self, fold_step):
        # Setup your model, optimizer, and scheduler here
        pass

    def forward_loop(self, model, dataloader, optimizer, scheduler, mode, ep, fold_step):
        # Your forward loop logic here
        pass
