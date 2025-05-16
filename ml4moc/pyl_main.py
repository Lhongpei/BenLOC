import pytorch_lightning as pl
import pandas as pd
from torch.utils.data import TensorDataset, random_split, DataLoader
import torch

import copy
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.utils.data
from pytorch_lightning.utilities import rank_zero_info
import os
from ml4moc.DL.gnn_pairwise.utils.lr_schedulers import get_schedule_fn

class TabModelPyL(pl.LightningModule):
    def __init__(self, model, param_args):
        super(TabModelPyL, self).__init__()
        self.model = model
        self.args = param_args
        self.loss_fn = F.mse_loss
        self.num_training_steps_cached = None 
    def load_train_dataset(self, X: torch.Tensor, y: torch.Tensor, val_size=0.0):
        if type(X) == np.ndarray:
            X = torch.from_numpy(X).float()
        if type(y) == np.ndarray:
            y = torch.from_numpy(y).float()
        self.train_dataset = TensorDataset(X, y)
        if val_size > 0:
            self.train_dataset, self.val_dataset = random_split(
                self.train_dataset,
                [
                    int((1 - val_size) * len(self.train_dataset)),
                    len(self.train_dataset) - int((1 - val_size) * len(self.train_dataset)),
                ],
            )

    def load_test_dataset(self, X: torch.Tensor, y: torch.Tensor):
        self.test_dataset = TensorDataset(X, y)

    def forward(self, x):
        """Please implement the forward method for your model.
        """
        return self.model(x)
    
    def set_loss_func(self, loss_func):
        self.loss_func = loss_func
    
    def training_step(self, batch, batch_idx):
        pred = self.forward(batch[0])
        loss = self.loss_fn(pred, batch[1])
        print('Training loss: %f' % loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        pred = self.forward(batch[0])
        loss = self.loss_fn(pred, batch[1])
        print('Validation loss: %f' % loss)
        return loss
    
        
    
    def get_total_num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if self.num_training_steps_cached is not None:
            return self.num_training_steps_cached
        dataset = self.train_dataloader()
        if self.trainer.max_steps and self.trainer.max_steps > 0:
            return self.trainer.max_steps

        dataset_size = (
            self.trainer.limit_train_batches * len(dataset)
            if self.trainer.limit_train_batches != 0
            else len(dataset)
        )

        num_devices = max(1, self.trainer.num_devices)
        effective_batch_size = self.trainer.accumulate_grad_batches * num_devices
        self.num_training_steps_cached = (
            dataset_size // effective_batch_size
        ) * self.trainer.max_epochs
        return self.num_training_steps_cached
    
    def configure_optimizers(self):
        rank_zero_info('Parameters: %d' % sum([p.numel() for p in self.model.parameters()]))
        rank_zero_info('Training steps: %d' % self.get_total_num_training_steps())

        if self.args.lr_scheduler == "constant":
            return torch.optim.AdamW(
                self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)

        else:
            optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
            scheduler = get_schedule_fn(self.args.lr_scheduler, self.get_total_num_training_steps())(optimizer)

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }

    def train_dataloader(self):
        batch_size = self.args.batch_size
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        return train_dataloader
    
    def val_dataloader(self):
        batch_size = self.args.batch_size
        val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        return val_dataloader
    
    def test_dataloader(self):
        batch_size = self.args.batch_size
        test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        return test_dataloader
