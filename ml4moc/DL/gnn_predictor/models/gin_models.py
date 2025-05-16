import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d as BN
from torch.nn import Linear, ReLU, Sequential, Sigmoid
from torch_geometric.nn import JumpingKnowledge, MessagePassing


# class BipartiteGINconv(MessagePassing):
#     def __init__(self, edge_dim, dim, aggr):
#         super(BipartiteGINconv, self).__init__(aggr=aggr, flow="source_to_target")
#         self.edge_encoder = Sequential(Linear(edge_dim, dim), ReLU(), Linear(dim, dim), ReLU(), BN(dim))
# Update constraint embeddings based on variable embeddings.
class VarConBipartiteLayer(MessagePassing):
    def __init__(self, edge_dim, dim, var_assignment, aggr):
        super(VarConBipartiteLayer, self).__init__(aggr=aggr, flow="source_to_target")
        self.edge_encoder = Sequential(Linear(edge_dim, dim), ReLU(), Linear(dim, dim), ReLU(), BN(dim))
        self.joint_var = Sequential(Linear(dim + 1, dim), ReLU(), Linear(dim, dim), ReLU(), BN(dim))
        self.var_assignment = var_assignment
        self.mlp = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim), ReLU(), BN(dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        self.initial_eps = 0

    def forward(self, source, target, edge_index, edge_attr, size):
        var_assignment = self.var_assignment(source)
        source = self.joint_var(torch.cat([source, var_assignment], dim=-1))
        edge_embedding = self.edge_encoder(edge_attr)
        assert edge_index[0].min()>= 0 and edge_index[0].max() < source.size(0)
        assert edge_index[1].min()>= 0 and edge_index[1].max() < target.size(0)
        tmp = self.propagate(edge_index, x=source, edge_attr=edge_embedding, size=size)
        out = self.mlp((1 + self.eps) * target + tmp)
        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

# Update variable embeddings based on constraint embeddings.
class ConVarBipartiteLayer(MessagePassing):
    def __init__(self, edge_dim, dim, aggr):
        super(ConVarBipartiteLayer, self).__init__(aggr=aggr, flow="source_to_target")
        self.edge_encoder = Sequential(Linear(edge_dim, dim), ReLU(), Linear(dim, dim), ReLU(), BN(dim))
        self.joint_var = Sequential(Linear(dim * 2, dim), ReLU(), Linear(dim, dim), ReLU(), BN(dim))
        self.mlp = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim), ReLU(), BN(dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        self.initial_eps = 0

    def forward(self, source, target, edge_index, edge_attr, size):
        edge_embedding = self.edge_encoder(edge_attr)
        source = self.joint_var(torch.cat([source, source], dim=-1))  # Adjusted to duplicate source instead of concatenating with error since error is removed
        tmp = self.propagate(edge_index, x=source, edge_attr=edge_embedding, size=size)
        out = self.mlp((1 + self.eps) * target + tmp)
        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

class BipartiteGIN(torch.nn.Module):
    def __init__(self, input_dim_xs, input_dim_xt, input_dim_edge, hidden_dim, num_layers, aggr='add', regression=False, jk_mode='cat'):
        super(BipartiteGIN, self).__init__()
        self.num_layers = num_layers-1
        self.var_node_encoder = Sequential(Linear(input_dim_xt, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
        self.con_node_encoder = Sequential(Linear(input_dim_xs, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
        self.layers_ass = [Sequential(Linear(hidden_dim, hidden_dim), ReLU(), Linear(hidden_dim, 1), Sigmoid()) for _ in range(self.num_layers)]
        self.layers_con = torch.nn.ModuleList([ConVarBipartiteLayer(1, hidden_dim, aggr=aggr) for _ in range(self.num_layers)])  # 将这里的赋值改为直接添加到ModuleList中
        self.layers_var = torch.nn.ModuleList([VarConBipartiteLayer(1, hidden_dim, self.layers_ass[i], aggr=aggr) for i in range(self.num_layers)])  # 将这里的赋值改为直接添加到ModuleList中
        self.jk_x_s = JumpingKnowledge(mode=jk_mode, channels=hidden_dim, num_layers=num_layers)
        self.jk_x_t = JumpingKnowledge(mode=jk_mode, channels=hidden_dim, num_layers=num_layers)

    def forward(self, x_s, x_t, edge_attr, edge_index):
        assert edge_index[0].min()>= 0 and edge_index[0].max() < x_s.size(0)
        assert edge_index[1].min()>= 0 and edge_index[1].max() < x_t.size(0)
        var_node_features_0 = self.var_node_encoder(x_t)
        con_node_features_0 = self.con_node_encoder(x_s)

        x_var = [var_node_features_0]
        x_con = [con_node_features_0]

        inverse_edge_index = edge_index.clone()
        inverse_edge_index[[0, 1]] = edge_index[[1, 0]]
        
        for i in range(self.num_layers):
            x_con.append(F.relu(self.layers_var[i](x_var[i], x_con[i], inverse_edge_index, edge_attr, (var_node_features_0.size(0), con_node_features_0.size(0)))))
            x_var.append(F.relu(self.layers_con[i](x_con[i+1], x_var[i], edge_index, edge_attr, (con_node_features_0.size(0), var_node_features_0.size(0)))))
        x_s = self.jk_x_s(x_con)
        x_t = self.jk_x_t(x_var)
        
        return x_s, x_t
