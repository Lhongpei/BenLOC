import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import (JumpingKnowledge, MessagePassing,
                                global_add_pool, global_max_pool,
                                global_mean_pool)

from ml4moc.DL.gnn_predictor.models.layers import FourierEncoder, LinearEncoder, PreNormLayer


class BipartiteGCNConv(MessagePassing):
    def __init__(self, emb_size):
        super(BipartiteGCNConv, self).__init__('add')
        self.feature_module_left = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size))
        self.feature_module_edge = torch.nn.Sequential(
            torch.nn.Linear(1, emb_size, bias=False))
        self.feature_module_right = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size, bias=False))
        self.feature_module_final = torch.nn.Sequential(
            PreNormLayer(1, shift=False),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size))
        self.post_conv_module = torch.nn.Sequential(
            PreNormLayer(1, shift=False))
        # output_layers
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(2 * emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LayerNorm(emb_size),
        )

    def forward(self, left_features, edge_indices, edge_features, right_features):
        output = self.propagate(edge_indices, size=(left_features.shape[0], right_features.shape[0]),
                                node_features=(left_features, right_features), edge_features=edge_features)
        return self.output_module(torch.cat([self.post_conv_module(output), right_features], dim=-1))

    def message(self, node_features_i, node_features_j, edge_features):
        output = self.feature_module_final(self.feature_module_left(node_features_i)
                                           + self.feature_module_edge(edge_features)
                                           + self.feature_module_right(node_features_j))
        return output
    
class BipartiteGCN(nn.Module):
    def __init__(self, input_dim_xs, input_dim_xt, input_dim_edge, hidden_dim, num_layers, jk_mode='cat'):
        super(BipartiteGCN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.jk_mode = jk_mode
        
        # Embedding layers
        self.xs_embedding = FourierEncoder(hidden_dim // (2 * input_dim_xs))
        self.xt_embedding = FourierEncoder(hidden_dim // (2 * input_dim_xt))
        # self.xs_embedding = LinearEncoder(input_dim_xs, hidden_dim)
        # self.xt_embedding = LinearEncoder(input_dim_xt, hidden_dim)

        # Convolution layers
        self.conv_s_t = nn.ModuleList([BipartiteGCNConv(hidden_dim) for _ in range(num_layers)])
        self.conv_t_s = nn.ModuleList([BipartiteGCNConv(hidden_dim) for _ in range(num_layers)])
        
        # Jumping Knowledge
        self.jk_x_s = JumpingKnowledge(mode=jk_mode, channels=hidden_dim, num_layers=num_layers)
        self.jk_x_t = JumpingKnowledge(mode=jk_mode, channels=hidden_dim, num_layers=num_layers)
        
    def forward(self, x_s, x_t, edge_attr, edge_index):
        x_s = self.xs_embedding(x_s)
        x_t = self.xt_embedding(x_t)

        # Collect outputs from each layer
        x_s_outs, x_t_outs = [], []
        inverse_edge_index = edge_index.clone()
        inverse_edge_index[[0, 1]] = edge_index[[1, 0]]
        
        for conv_s_t, conv_t_s in zip(self.conv_s_t, self.conv_t_s):
            x_t = conv_s_t(x_s, edge_index, edge_attr, x_t)
            x_s = conv_t_s(x_t, inverse_edge_index, edge_attr, x_s)
            x_s_outs.append(x_s)
            x_t_outs.append(x_t)
        
        # Use Jumping Knowledge to aggregate outputs
        x_s = self.jk_x_s(x_s_outs)
        x_t = self.jk_x_t(x_t_outs)
        
        return x_s, x_t
