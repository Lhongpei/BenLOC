import pytorch_lightning as pl
import pandas as pd
from torch.utils.data import TensorDataset
import torch
class FeatModel(pl.LightningModule):
    def __init__(self, model_class, param_args):
        super(FeatModel, self).__init__()
        self.model_class = model_class
        self.args = param_args
        
    def load_dataset_from_df(self, X: pd.DataFrame, y: pd.Series):
        
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)
        self.train_dataset = TensorDataset()