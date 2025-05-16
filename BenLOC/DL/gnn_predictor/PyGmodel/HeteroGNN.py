import torch
import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')
import torch_geometric.transforms as T
from omegaconf import OmegaConf
from torch_geometric.datasets import OGB_MAG
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (GAT, GCN, GATConv, GCNConv, HeteroConv, Linear, SAGPooling,TopKPooling,
                                SAGEConv, to_hetero, to_hetero_with_bases,GraphSAGE)

from BenLOC.DL.gnn_predictor.dataset_gen.heteroDataset import RegrSolTimeHeteroDataset,RankListConfigNodesDataset
from BenLOC.DL.gnn_predictor.PyGmodel.tasks_pyg import RegrNetPool, RankListNetNode
from torch_geometric.loader import DataLoader

# class GAT(torch.nn.Module):
#     def __init__(self, hidden_channels, out_channels):
#         super().__init__()
#         self.conv1 = GATConv((-1, -1), hidden_channels, add_self_loops=False)
#         self.lin1 = Linear(-1, hidden_channels)
#         self.conv2 = GATConv((-1, -1), out_channels, add_self_loops=False)
#         self.lin2 = Linear(-1, out_channels)

#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index) + self.lin1(x)
#         x = x.relu()
#         x = self.conv2(x, edge_index) + self.lin2(x)
#         return x
    
if __name__ == '__main__':
    root = 'setcover-flat'
    report = 'labels/setcover_fixed_5fold'
    reprocess = False
    config = OmegaConf.load('src/config/config.yaml')
    dataset = RankListConfigNodesDataset(root, report,fold=1, type='train', reprocess = reprocess)
    model = GAT(in_channels=(-1,-1), hidden_channels=64, out_channels = 4,num_layers = 1, jk='cat', add_self_loops = False )
    model = to_hetero(model, dataset[0].metadata())
    ranklist = RankListNetNode(config.pyg_ranklist_node, dataset[0].metadata())
    dataset1 = DataLoader(dataset, batch_size=5)
    data = next(iter(dataset1))
    oyt = ranklist(data)
    out = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
    for edge_type, edge_index in data.edge_index_dict.items():
        max_source_index = edge_index[0].max()
        max_target_index = edge_index[1].max()
        print(f"{edge_type} source max index: {max_source_index}")
        print(f"{edge_type} target max index: {max_target_index}")
        # 这里可以进一步比较 max_source_index 和 max_target_index 与相应节点类型的 num_nodes
