import pytorch_lightning as pl
import pandas as pd
from torch.utils.data import TensorDataset, random_split, DataLoader
import torch

"""A meta PyTorch Lightning model for training and evaluating DIFUSCO models."""
import copy
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.utils.data
from pytorch_lightning.utilities import rank_zero_info
import os
from ml4moc.DL.utils.lr_schedulers import get_schedule_fn

class FeatModel(pl.LightningModule):
    def __init__(self, model_class, param_args):
        super(FeatModel, self).__init__()
        self.model_class = model_class
        self.args = param_args

    def load_train_dataset(self, X: torch.Tensor, y: torch.Tensor, val_size=0.0):
        self.train_dataset = TensorDataset(X, y)
        if val_size > 0:
            self.train_dataset, self.val_dataset = random_split(
                self.train_dataset,
                [
                    int((1 - val_size) * len(self.train_dataset)),
                    int(val_size * len(self.train_dataset)),
                ],
            )

    def load_test_dataset(self, X: torch.Tensor, y: torch.Tensor):
        self.test_dataset = TensorDataset(X, y)

    def load_train_dataset_from_df(self, X: pd.DataFrame, y: pd.Series, val_size=0.0):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)
        self.train_dataset = TensorDataset(self.X, self.y, val_size)

    def load_test_dataset_from_df(self, X: pd.DataFrame, y: pd.Series):
        self.X_test = torch.tensor(X.values, dtype=torch.float32)
        self.y_test = torch.tensor(y.values, dtype=torch.float32)
        self.test_dataset = TensorDataset(self.X_test, self.y_test)

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