import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import (GAT, GIN, GraphSAGE, global_add_pool, SAGPooling,
                                global_max_pool, global_mean_pool, to_hetero)
from torch_geometric.utils import to_dense_batch
from torch.nn import ModuleList
from torch_geometric.nn import GATConv, GATv2Conv, HeteroConv
from ml4moc.DL.gnn_pairwise.models.layers import predictMLP, FourierEncoder
import torch.nn.functional as F
import copy
import math
from torch_geometric.nn import GraphNorm

class BiparGAT(nn.Module):
    
    def __init__(self, config, data):
        super(BiparGAT, self).__init__()
        self.hidden_dim = config.gnn_hidden_dim
        config.gnn_out_dim = config.gnn_hidden_dim
        self.num_heads = config.heads
        self.convs = ModuleList()
        self.feature_conv_var = FourierEncoder(math.ceil(self.hidden_dim/(2*data['vars'].x.size(-1))))
        self.feature_conv_cons = FourierEncoder(math.ceil(self.hidden_dim/(2*data['cons'].x.size(-1))))
        var_enc_dim = math.ceil(self.hidden_dim/(2*data['vars'].x.size(-1)))*2*data['vars'].x.size(-1)
        cons_enc_dim = math.ceil(self.hidden_dim/(2*data['cons'].x.size(-1)))*2*data['cons'].x.size(-1)
        self.cons_linear = nn.Linear(cons_enc_dim, self.hidden_dim)
        self.var_linear = nn.Linear(var_enc_dim, self.hidden_dim)   
        # cons_enc_dim = data['cons'].x.size(-1)
        # var_enc_dim = data['vars'].x.size(-1)
        self.convs.append(HeteroConv({
            ('cons', 'to', 'vars'): GATv2Conv((self.hidden_dim, self.hidden_dim), self.hidden_dim//self.num_heads, add_self_loops=False, heads=self.num_heads, edge_dim=1),
            ('vars', 'to', 'cons'): GATv2Conv((self.hidden_dim, self.hidden_dim), self.hidden_dim//self.num_heads, add_self_loops=False, heads=self.num_heads, edge_dim=1),
        }))
        for _ in range(config.num_layers - 1):
            self.convs.append(HeteroConv({
                ('cons', 'to', 'vars'): GATv2Conv((self.hidden_dim, self.hidden_dim), self.hidden_dim//self.num_heads, add_self_loops=False, heads=self.num_heads, edge_dim=1),
                ('vars', 'to', 'cons'): GATv2Conv((self.hidden_dim, self.hidden_dim), self.hidden_dim//self.num_heads, add_self_loops=False, heads=self.num_heads, edge_dim=1),
            }))
        
        self.graph_norm = GraphNorm(self.hidden_dim) 
        if config.pooling == 'add':
            self.pooling = global_add_pool
        elif config.pooling == 'mean':
            self.pooling = global_mean_pool
        elif config.pooling == 'max':
            self.pooling = global_max_pool

        # if config.jk_mode == 'cat':
        #     jk_dim = config.gnn_out_dim * config.num_layers
        # else:
        #     jk_dim = config.gnn_out_dim
        self.use_feat = False
        self.layer_norm = nn.LayerNorm(config.gnn_out_dim)
        pred_input_dim = config.pooling_mlp_out_dim
        if getattr(data, 'feat', None) is not None:
            print('Using feature embedding')
            self.use_feat = True
            self.feat_embedding = predictMLP(data.feat.size(-1), config.feat_mlp_out_dim, config.feat_mlp_hidden_dim, num_layers=config.feat_mlp_layers, dropout_prob=0.3, norm = True)
            pred_input_dim = config.feat_mlp_out_dim + config.pooling_mlp_out_dim
        self.pooling_embedding = predictMLP(2 * config.gnn_out_dim, config.pooling_mlp_out_dim, config.pooling_mlp_hidden_dim, num_layers=config.pred_mlp_layers, dropout_prob=0.3, norm = True)
        # self.predict_mlp = nn.ModuleList([
        #         predictMLP(pred_input_dim, 1, config.pred_mlp_hidden_dim, num_layers=config.pred_mlp_layers) for _ in range(data.y.size(-1))
        # ])
        self.predict_mlp = predictMLP(pred_input_dim, data.y.size(-1), config.pred_mlp_hidden_dim, num_layers=config.pred_mlp_layers, dropout_prob=0.3, norm = True)
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def forward(self, heteroData):
        x_dict = copy.deepcopy(heteroData.x_dict)
        x_dict['vars'] = self.var_linear(self.feature_conv_var(x_dict['vars']))
        x_dict['cons'] = self.cons_linear(self.feature_conv_cons(x_dict['cons']))

        if hasattr(heteroData, 'batch_dict'):
            batch_dict = heteroData.batch_dict
        else:
            batch_dict = None
        edge_index_dict = heteroData.edge_index_dict
        edge_attr_dict = heteroData.edge_attr_dict
        
        for conv in self.convs:
            x_dict_new = conv(x_dict, edge_index_dict, edge_attr_dict)
            # Apply residual connection
            x_dict_new = {key: F.leaky_relu(x + x_dict[key]) for key, x in x_dict.items()}
            if batch_dict is not None:
                x_dict_new = {key: self.graph_norm(x, batch_dict[key]) for key, x in x_dict.items()}
            else:
                x_dict_new = {key: self.graph_norm(x) for key, x in x_dict.items()}
            x_dict = x_dict_new
        
        x_vars = x_dict['vars']
        x_cons = x_dict['cons']
        
        x_vars = self.pooling(x_vars, batch_dict['vars']) if batch_dict is not None else self.pooling(x_vars, torch.zeros(x_vars.size(0), dtype=torch.long, device=x_vars.device))
        x_cons = self.pooling(x_cons, batch_dict['cons']) if batch_dict is not None else self.pooling(x_cons, torch.zeros(x_cons.size(0), dtype=torch.long, device=x_cons.device))
        
        x_graph = torch.cat((x_vars, x_cons), dim=1)
        
        pool_embedding = self.pooling_embedding(x_graph)
        if not self.use_feat:
            x = pool_embedding
        else:
            x = torch.cat((self.feat_embedding(heteroData.feat), pool_embedding), dim=1)
            if torch.isnan(x).any():
                print('NAN')
                print(heteroData.feat)
                print(x_graph)
                print(self.feat_embedding(heteroData.feat))
                print(self.pooling_embedding(x_graph))
                exit()
        x_pre = self.predict_mlp(x)

        return x_pre
    
    def graph_embedding(self, heteroData):
        x_dict = copy.deepcopy(heteroData.x_dict)
        x_dict['vars'] = self.feature_conv_var(x_dict['vars'])
        x_dict['cons'] = self.feature_conv_cons(x_dict['cons'])
        
        residual_x_dict = copy.deepcopy(x_dict)
        
        if hasattr(heteroData, 'batch'):
            batch_dict = heteroData.batch_dict
        else:
            batch_dict = None
        edge_index_dict = heteroData.edge_index_dict
        edge_attr_dict = heteroData.edge_attr_dict
        
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict, edge_attr_dict)
            x_dict = {key: F.leaky_relu(x + residual_x_dict[key]) for key, x in x_dict.items()}
            if batch_dict is not None:
                x_dict = {key: self.graph_norm(x, batch_dict[key]) for key, x in x_dict.items()}
            else:
                x_dict = {key: self.graph_norm(x) for key, x in x_dict.items()}
            residual_x_dict = copy.deepcopy(x_dict)
        x_vars = x_dict['vars']
        x_cons = x_dict['cons']
        
        x_vars = self.pooling(x_vars, batch_dict['vars']) if batch_dict is not None else self.pooling(x_vars, torch.zeros(x_vars.size(0), dtype=torch.long, device=x_vars.device))
        x_cons = self.pooling(x_cons, batch_dict['cons']) if batch_dict is not None else self.pooling(x_cons, torch.zeros(x_cons.size(0), dtype=torch.long, device=x_cons.device))
        
        x_graph = torch.cat((x_vars, x_cons), dim=1)
        x = self.pooling_embedding(x_graph) 
        return x
