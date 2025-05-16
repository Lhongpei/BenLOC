from torch_geometric.data import HeteroData
import torch
from ml4moc.DL.gnn_predictor.utils.utils import configsEncode
def toHeteroData(x_s, x_t, edge_attr, edge_index, undirected = True):
    data = HeteroData()
    data['cons'].x = x_s
    data['vars'].x = x_t
    data['cons', 'vars'].edge_attr = edge_attr
    data['cons', 'vars'].edge_index = edge_index
    if undirected:
        inverse_edge_index = edge_index.clone()
        inverse_edge_index[[0, 1]] = edge_index[[1, 0]]
        assert inverse_edge_index[0].max() == edge_index[1].max() and inverse_edge_index[1].max() == edge_index[0].max()
        data['vars', 'cons'].edge_index = inverse_edge_index
        data['vars', 'cons'].edge_attr = edge_attr
    return data

def add_config_node_link(data: HeteroData, config, add_self = True, undirected = False):
    """Add a config node to the hetero graph, and link it to all other nodes.

    Args:
        data (HeteroData): A hetero graph
        config (Any): _description_
        add_self (bool, optional): Whether add self-loop for each config_node. Defaults to True.
        undirected (bool, optional): Wheter link config nodes 2 other nodes. Defaults to False.

    Returns:
        HeteroData: A hetero graph with config node linked to all other nodes.
    """
    data['config'].x = torch.tensor(config, dtype=torch.float32)
    num_config = data['config'].num_nodes
    
    # if add_self:
    #     data['config', 'config'].edge_index = torch.stack([torch.arange(num_config), torch.arange(num_config)])
        
    for i in data.x_dict.keys():
        if i != 'config':
            num_i = data[i].num_nodes
            if undirected:
                data['config',i].edge_index = torch.cartesian_prod(torch.arange(num_config), torch.arange(num_i)).t()
            data[i,'config'].edge_index = torch.cartesian_prod(torch.arange(num_i), torch.arange(num_config)).t()
    return data

def addingConfigNode(data: HeteroData, config, add_self = True, undirected = False, encMethod:str = 'totOnehot'):
    """Add a config node to the hetero graph, and link it to all other nodes.

    Args:
        data (HeteroData): A hetero graph
        config (Any): a list of config with type of tuple
        add_self (bool, optional): Whether add self-loop for each config_node. Defaults to True.
        undirected (bool, optional): Wheter link config nodes 2 other nodes. Defaults to False.
        encMethod (str, optional): Encoding method for config node in ['eachOneHot','totOneHot']. Defaults to 'forone'.
            eachOneHot: each config is encoded as a one-hot vector and concatenated together.
            totOneHot: all config is encoded as a one-hot together.

    Returns:
        HeteroData: A hetero graph with config node linked to all other nodes.
    """
    assert encMethod in ['eachOnehot', 'totOnehot', 'continue'], "encMethod should be one of ['eachOnehot', 'totOnehot', 'continue]"
    return add_config_node_link(data, configsEncode(config,mode=encMethod), add_self, undirected)