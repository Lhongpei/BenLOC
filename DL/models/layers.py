import numpy as np
import torch
import torch.nn as nn


class FourierEncoder(torch.nn.Module):
    """Node encoder using Fourier features.
    """
    def __init__(self, level, include_self=True):
        super(FourierEncoder, self).__init__()
        self.level = level
        self.include_self = include_self

    def multiscale(self, x, scales):
        return torch.hstack([x / i for i in scales])

    def forward(self, x):
        device, dtype, orig_x = x.device, x.dtype, x
        scales = 2 ** torch.arange(-self.level / 2, self.level / 2, device=device, dtype=dtype)
        lifted_feature = torch.cat((torch.sin(self.multiscale(x, scales)), torch.cos(self.multiscale(x, scales))), 1)
        return lifted_feature

class LinearEncoder(nn.Module):
    """Node encoder using linear layers
    """
    def __init__(self, input_dim, hidden_dim, layer = 1, BN = False):
        super(LinearEncoder, self).__init__()
        self.layer = nn.ModuleList()
        self.layer.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(layer):
            self.layer.append(nn.Linear(hidden_dim, hidden_dim))
            self.layer.append(nn.ReLU())
        if BN:
            self.layer.append(nn.BatchNorm1d(hidden_dim))
        

    def forward(self, x):
        x = self.layer(x)
        return x
    
class PreNormException(Exception):
    pass

class PreNormLayer(torch.nn.Module):
    def __init__(self, n_units, shift=True, scale=True, name=None):
        super(PreNormLayer, self).__init__()
        assert shift or scale
        self.register_buffer('shift', torch.zeros(n_units) if shift else None)
        self.register_buffer('scale', torch.ones(n_units) if scale else None)
        self.n_units = n_units
        self.waiting_updates = False
        self.received_updates = False

    def forward(self, input_):
        if self.waiting_updates:
            self.update_stats(input_)
            self.received_updates = True
            raise PreNormException
        if self.shift is not None:
            input_ = input_ + self.shift
        if self.scale is not None:
            input_ = input_ * self.scale
        return input_

    def start_updates(self):
        self.avg = 0
        self.var = 0
        self.m2 = 0
        self.count = 0
        self.waiting_updates = True
        self.received_updates = False

    def update_stats(self, input_):
        """
        Online mean and variance estimation. See: Chan et al. (1979) Updating
        Formulae and a Pairwise Algorithm for Computing Sample Variances.
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
        """
        assert self.n_units == 1 or input_.shape[-1] == self.n_units, f"Expected input dimension of size {self.n_units}, got {input_.shape[-1]}."
        input_ = input_.reshape(-1, self.n_units)
        sample_avg = input_.mean(dim=0)
        sample_var = (input_ - sample_avg).pow(2).mean(dim=0)
        sample_count = np.prod(input_.size()) / self.n_units
        delta = sample_avg - self.avg
        self.m2 = self.var * self.count + sample_var * sample_count + delta ** 2 * self.count * sample_count / (
                self.count + sample_count)
        self.count += sample_count
        self.avg += delta * sample_count / self.count
        self.var = self.m2 / self.count if self.count > 0 else 1

    def stop_updates(self):
        """
        Ends pre-training for that layer, and fixes the layers's parameters.
        """
        assert self.count > 0
        if self.shift is not None:
            self.shift = -self.avg
        if self.scale is not None:
            self.var[self.var < 1e-8] = 1
            self.scale = 1 / torch.sqrt(self.var)
        del self.avg, self.var, self.m2, self.count
        self.waiting_updates = False
        self.trainable = False
    
class predictMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim_list, num_layers=1, dropout_prob=0.5, norm=False):
        if type(hidden_dim_list) == int:
            hidden_dim_list = [hidden_dim_list] * num_layers
        else:
            num_layers = len(hidden_dim_list) - 1
        super(predictMLP, self).__init__()
        self.hidden_dim = hidden_dim_list
        self.num_layers = num_layers
        self.mlp = nn.ModuleList()
        self.in_layer = nn.Linear(input_dim, hidden_dim_list[0])

        if norm:
            for i in range(num_layers):
                if i == num_layers - 1:
                    self.mlp.append(nn.Sequential(
                        nn.Linear(hidden_dim_list[i], hidden_dim_list[i+1]),
                        nn.LayerNorm(hidden_dim_list[i+1]),
                        #nn.BatchNorm1d(hidden_dim_list[i+1]),
                        nn.LeakyReLU(),
                        nn.Dropout(p=dropout_prob)
                    ))
                else:
                    self.mlp.append(nn.Sequential(
                        nn.Linear(hidden_dim_list[i], hidden_dim_list[i+1]),
                        nn.LayerNorm(hidden_dim_list[i+1]),
                        #nn.BatchNorm1d(hidden_dim_list[i+1]),
                        nn.LeakyReLU(),
                    ))
        else:
            for i in range(num_layers):
                if i == num_layers - 1:
                    self.mlp.append(nn.Sequential(
                        nn.Linear(hidden_dim_list[i], hidden_dim_list[i+1]),
                        nn.LeakyReLU(),
                        nn.Dropout(p=dropout_prob)
                    ))
                else:
                    self.mlp.append(nn.Sequential(
                        nn.Linear(hidden_dim_list[i], hidden_dim_list[i+1]),
                        nn.LeakyReLU(),
                    ))
        
        self.out_layer = nn.Linear(hidden_dim_list[-1], output_dim)
        
    def forward(self, x):
        x = self.in_layer(x)
        x = torch.nn.LeakyReLU()(x)
        for i in range(self.num_layers):
            x = self.mlp[i](x)
        x = self.out_layer(x)
        return x
