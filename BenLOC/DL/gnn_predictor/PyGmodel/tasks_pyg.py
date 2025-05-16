import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import (GAT, GIN, GraphSAGE, global_add_pool,SAGPooling,
                                global_max_pool, global_mean_pool, to_hetero)
from torch_geometric.utils import to_dense_batch

from BenLOC.DL.gnn_predictor.models.layers import predictMLP


class  RankListNetPool(nn.Module):
    
    def __init__(self, config, metadata):
        super(RankListNetPool, self).__init__()
        if config.GNN == 'GAT':
            self.graph_embedding = GAT(in_channels=(-1,-1), hidden_channels=config.gnn_hidden_dim, out_channels = config.gnn_out_dim, num_layers = config.num_layers, jk=config.jk_mode, add_self_loops = False )
        elif config.GNN == 'GIN':
            self.graph_embedding = GIN(in_channels=(-1,-1), hidden_channels=config.gnn_hidden_dim, out_channels = config.gnn_out_dim, num_layers = config.num_layers, jk=config.jk_mode, add_self_loops = False )
        elif config.GNN == 'GraphSAGE':
            self.graph_embedding = GraphSAGE(in_channels=(-1,-1), hidden_channels=config.gnn_hidden_dim, out_channels = config.gnn_out_dim, num_layers = config.num_layers, jk=config.jk_mode, add_self_loops = False )
        if config.pooling == 'add':
            self.pooling = global_add_pool
        elif config.pooling == 'mean':
            self.pooling = global_mean_pool
        elif config.pooling == 'max':
            self.pooling = global_max_pool
        self.graph_embedding = to_hetero(self.graph_embedding, metadata)
        # if config.jk_mode == 'cat':
        #     jk_dim = config.gnn_out_dim * config.num_layers
        # else:
        #     jk_dim = config.gnn_out_dim
        
        self.pooling_embedding = predictMLP(2 * config.gnn_out_dim, config.pooling_mlp_out_dim, config.pooling_mlp_hidden_dim, num_layers=config.pooling_mlp_layers)
        # self.mlp_pred = predictMLP(2 * jk_dim, config.task_dim, config.mlp_hidden_dim, num_layers=config.pred_mlp_layers)
        self.mlps = nn.ModuleList([
                predictMLP(config.pooling_mlp_out_dim, 1, config.mlp_hidden_dim, num_layers=config.pred_mlp_layers) for _ in range(config.task_dim)
        ])
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def forward(self, heteroData):
        
        x = self.graph_embedding(heteroData.x_dict, heteroData.edge_index_dict)
        x_vars = x['vars']
        x_cons = x['cons']
        x_vars = self.pooling(x_vars, heteroData.batch_dict['vars'])
        x_cons = self.pooling(x_cons, heteroData.batch_dict['cons'])
        
        x = torch.cat((x_vars, x_cons), dim=1)
        x = self.pooling_embedding(x)
        #x_pre = self.mlp_pred(x)
        x_pre = torch.cat([mlp(x) for mlp in self.mlps], dim=1)

        return x_pre
    
class  RankListNetNode(nn.Module):
    
    def __init__(self, config, metadata):
        super(RankListNetNode, self).__init__()
        if config.GNN == 'GAT':
            self.graph_embedding = GAT(in_channels=(-1,-1), hidden_channels=config.gnn_hidden_dim, out_channels = config.gnn_out_dim, num_layers = config.num_layers, jk=config.jk_mode, add_self_loops = False )
        elif config.GNN == 'GIN':
            self.graph_embedding = GIN(in_channels=(-1,-1), hidden_channels=config.gnn_hidden_dim, out_channels = config.gnn_out_dim, num_layers = config.num_layers, jk=config.jk_mode, add_self_loops = False )
        elif config.GNN == 'GraphSAGE':
            self.graph_embedding = GraphSAGE(in_channels=(-1,-1), hidden_channels=config.gnn_hidden_dim, out_channels = config.gnn_out_dim, num_layers = config.num_layers, jk=config.jk_mode, add_self_loops = False )
        
        self.graph_embedding = to_hetero(self.graph_embedding, metadata)
        # self.attention = nn.MultiheadAttention(config.gnn_out_dim, config.num_heads)
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def forward(self, heteroData):
        
        x = self.graph_embedding(heteroData.x_dict, heteroData.edge_index_dict, heteroData.edge_attr_dict)
        x_configs = to_dense_batch(x['config'], batch=heteroData.batch_dict['config'])[0]
        #x_pre = self.mlp_pred(x)
        x_pre = x_configs.squeeze()

        return x_pre
    
class RegrNetPool(nn.Module):
    def __init__(self, config, metadata):
        super(RegrNetPool, self).__init__()
        if config.GNN == 'GAT':
            self.graph_embedding = GAT(in_channels=(-1,-1), hidden_channels=config.gnn_hidden_dim, out_channels = config.gnn_out_dim, num_layers = config.num_layers, jk=config.jk_mode, add_self_loops = False )
        elif config.GNN == 'GIN':
            self.graph_embedding = GIN(in_channels=(-1,-1), hidden_channels=config.gnn_hidden_dim, out_channels = config.gnn_out_dim, num_layers = config.num_layers, jk=config.jk_mode, add_self_loops = False )
        if config.pooling == 'add':
            self.pooling = global_add_pool
        elif config.pooling == 'mean':
            self.pooling = global_mean_pool
        elif config.pooling == 'max':
            self.pooling = global_max_pool
        self.graph_embedding = to_hetero(self.graph_embedding, metadata)
        if config.jk_mode == 'cat':
            jk_dim = config.gnn_out_dim
        else:
            jk_dim = config.gnn_out_dim
                
        self.mlp_config = predictMLP(config.task_dim, config.task_dim, config.config_mlp_hidden_dim, num_layers=config.config_mlp_layers)
        self.mlp_pred = predictMLP(2 * jk_dim + config.task_dim, 1, config.pred_mlp_hidden_dim, num_layers=config.pred_mlp_layers)
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def forward(self, heteroData):
        x = self.graph_embedding(heteroData.x_dict, heteroData.edge_index_dict)
        x_vars = x['vars']
        x_cons = x['cons']
        x_vars = self.pooling(x_vars, heteroData.batch_dict['vars'])
        x_cons = self.pooling(x_cons, heteroData.batch_dict['cons'])
        
        x = torch.cat((x_vars, x_cons), dim=1)
        y = self.mlp_config(heteroData.config)
        xy = torch.cat((x, y), dim=1)
        x_pre = self.mlp_pred(xy)

        return x_pre
         