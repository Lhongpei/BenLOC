import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import (global_add_pool, global_max_pool,
                                global_mean_pool)

from ml4moc.DL.gnn_predictor.models.gcn_models import BipartiteGCN
from ml4moc.DL.gnn_predictor.models.gin_models import BipartiteGIN
from ml4moc.DL.gnn_predictor.models.layers import predictMLP


class  RankListNet(nn.Module):
    
    def __init__(self, input_dim_xs, input_dim_xt, input_dim_edge, num_layers, hidden_dim, 
                 mlp_hidden_dim, task_dim, pooling='add', jk_mode='cat', GNN='GCN'):
        super(RankListNet, self).__init__()
        if GNN == 'GCN':
            self.node_embedding = BipartiteGCN(input_dim_xs, input_dim_xt, input_dim_edge, hidden_dim, num_layers, jk_mode = jk_mode)
        elif GNN == 'GIN':
            self.node_embedding = BipartiteGIN(input_dim_xs, input_dim_xt, input_dim_edge, hidden_dim, num_layers, jk_mode = jk_mode, aggr='add')
        
        # Pooling
        if pooling == 'add':
            self.pooling = global_add_pool
        elif pooling == 'mean':
            self.pooling = global_mean_pool
        elif pooling == 'max':
            self.pooling = global_max_pool

        if jk_mode == 'cat':
            jk_dim = hidden_dim * num_layers
        else:
            jk_dim = hidden_dim
            
        self.mlp_pred_config1 = predictMLP(2 * jk_dim, 5, mlp_hidden_dim, num_layers=1)
        self.mlp_pred_config2 = predictMLP(2 * jk_dim, 5, mlp_hidden_dim, num_layers=1)
        self.mlp_tot = predictMLP(10, task_dim, mlp_hidden_dim, num_layers=1)
        
    def forward(self, x_s, x_t, edge_attr, edge_index, x_s_batch, x_t_batch):
        
        x_s, x_t = self.node_embedding(x_s, x_t, edge_attr, edge_index)
        
        # Pooling
        x_s = self.pooling(x_s, x_s_batch)
        x_t = self.pooling(x_t, x_t_batch)
        
        # MLP Predictor
        x = torch.cat((x_s, x_t), dim=1)
        x1 = self.mlp_pred_config1(x)
        x2 = self.mlp_pred_config2(x)
        x = torch.cat((x1, x2), dim=1)
        x = self.mlp_tot(x)
        return x
    
class RegrNet(nn.Module):
    def __init__(self, input_dim_xs, input_dim_xt, input_dim_edge, num_layers, hidden_dim, 
                 mlp_hidden_dim, task_dim, pooling = 'add', jk_mode = 'cat', GNN = 'GCN'):
        super(RegrNet, self).__init__()
        if GNN == 'GCN':
            self.node_embedding = BipartiteGCN(input_dim_xs, input_dim_xt, input_dim_edge, hidden_dim, num_layers, jk_mode = jk_mode)
        elif GNN == 'GIN':
            self.node_embedding = BipartiteGIN(input_dim_xs, input_dim_xt, input_dim_edge, hidden_dim, num_layers, jk_mode = jk_mode, aggr='add')
        if pooling == 'add':
            self.pooling = global_add_pool
        elif pooling == 'mean':
            self.pooling = global_mean_pool
        elif pooling == 'max':
            self.pooling = global_max_pool

        if jk_mode == 'cat':
            jk_dim = hidden_dim * num_layers
        else:
            jk_dim = hidden_dim
                
        self.mlp_config = predictMLP(task_dim, task_dim, mlp_hidden_dim, num_layers=0)
        self.mlp_pred = predictMLP(2 * jk_dim + task_dim, 1, mlp_hidden_dim, num_layers=0)
        
        
    def forward(self, x_s, x_t, edge_attr, edge_index, x_s_batch, x_t_batch, y):
        x_s, x_t = self.node_embedding(x_s, x_t, edge_attr, edge_index)
        
        x_s = self.pooling(x_s, x_s_batch)
        x_t = self.pooling(x_t, x_t_batch)
        
        x = torch.cat((x_s, x_t), dim=1)
        y = self.mlp_config(y)
        xy = torch.cat((x, y), dim=1)
        x_pre = self.mlp_pred(xy)

        return x_pre
         