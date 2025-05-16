import os
import pickle

import torch
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

from ml4moc.DL.gnn_predictor.utils.MIPmodel import MIPmodel


class BipartiteData(Data):
    def __init__(self, edge_index=None, x_s=None, x_t=None, edge_attr=None):
        super().__init__()
        self.edge_index = edge_index
        self.x_s = x_s
        self.x_t = x_t
        self.edge_attr = edge_attr
        self.name = None
        self.label = None
        self.y = None
        self.num_nodes = 0  # FIXME: disable warning since num_nodes is not used

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index':
            return torch.tensor([[self.x_s.size(0)], [self.x_t.size(0)]])
        else:
            return super().__inc__(key, value, *args, **kwargs)
    

class mipGraphDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, type='origin', norm=False):
        self.model = MIPmodel()
        self.type = type
        self.norm = norm
        self.root = root
        self.dict = None
        if self.type == 'origin':
            path = os.path.join(self.root, 'origin_processed')
        elif self.type == 'reformulate':
            path = os.path.join(self.root, 'reformulate_processed')
        self.dict_path = os.path.join(path, 'dict.pkl')
        super(mipGraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        if os.path.exists(self.dict_path):
            with open(self.dict_path, 'rb') as f:
                self.dict = pickle.load(f)
        else:
            self.dict = None
        # 初始化一个空的数据列表

    @property
    def processed_dir(self):
        if self.type == 'origin':
            return os.path.join(self.root, 'origin_processed')
        elif self.type == 'reformulate':
            return os.path.join(self.root, 'reformulate_processed')
        else:
            raise ResourceWarning('Wrong type')

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        # load normalize statistics
        data_list = []
        allFile = os.listdir(self.root)
        allPro = [i for i in allFile if i.endswith('.mps.gz') or i.endswith('.mps') or i.endswith('.lp')]
        for pro in tqdm(allPro, desc='Adding data to dataset'):
            edge_index, edge_attr, x_s, x_t, ddict = (
                self.model.generBiparNormal(os.path.join(self.root, pro), norm_f=self.norm))
            if self.dict is None:
                self.dict = ddict

            else:
                for key in ddict.keys():  # FIXME: what is doing here?
                    if 'max' in key:
                        self.dict[key] = max(self.dict[key], ddict[key])
                    elif 'min' in key:
                        self.dict[key] = min(self.dict[key], ddict[key])
            data = BipartiteData(edge_index=edge_index, x_s=x_s, x_t=x_t, edge_attr=edge_attr)
            data.name = pro
            data_list.append(data)
        if len(data_list) == 0:
            raise ValueError('No data in the dataset')
        if self.norm:
            for data in tqdm(data_list, desc='Normalizing data'):
                data.x_s[:, 2] = (data.x_s[:, 2] - self.dict['minLB_s']) / (self.dict['maxLB_s'] - self.dict['minLB_s'])
                data.x_s[:, 4] = (data.x_s[:, 4] - self.dict['minUB_s']) / (self.dict['maxUB_s'] - self.dict['minUB_s'])
                data.x_t[:, 3] = (data.x_t[:, 3] - self.dict['minLB_t']) / (self.dict['maxLB_t'] - self.dict['minLB_t'])  # FIXME: bug?
                data.x_t[:, 4] = (data.x_t[:, 4] - self.dict['minUB_t']) / (self.dict['maxUB_t'] - self.dict['minUB_t'])
                data.x_t[:, 5] = (data.x_t[:, 5] - self.dict['minObj']) / (self.dict['maxObj'] - self.dict['minObj'])
                data.edge_attr[:] = (data.edge_attr - self.dict['min_edge']) / (self.dict['max_edge'] - self.dict['min_edge'])
        # torch.save(self.dict,self.dict_path)
        with open(self.dict_path, 'wb') as f:
            pickle.dump(self.dict, f)
        # Concatenate the list of `Data` objects into a single `Data` object
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def getDict(self):
        return self.dict
    
class mipGraphN2N(InMemoryDataset):
    def __init__(self, root, reprocess=False, transform=None, pre_transform=None):
        assert root is not None, 'root path is None'
        assert os.path.exists(root), 'root path does not exist'
        self.model = MIPmodel()
        self.root = root
        self.dict = None
        path = os.path.join(self.root, 'N2N_processed')
        # if not os.path.exists(path):
        #     os.mkdir(path)
        self.dict_path = os.path.join(path, 'dictN2N.pkl')
        if reprocess:
            self.clear_processed_files()
        super(mipGraphN2N, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        if os.path.exists(self.dict_path):
            with open(self.dict_path, 'rb') as f:
                self.dict = pickle.load(f)
        else:
            self.dict = None
        # 初始化一个空的数据列表

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'N2N_processed')

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        # load normalize statistics
        data_list = []
        allFile = os.listdir(self.root)
        allPro = [i for i in allFile if i.endswith('.mps.gz') or i.endswith('.mps') or i.endswith('.lp')]
        
        for pro in tqdm(allPro, desc='Adding data to dataset'):
            edge_index, edge_attr, x_s, x_t, ddict = (
                self.model.generBiparN2N(os.path.join(self.root, pro)))
            if x_s.size(0) == 0 or x_t.size(0) == 0:
                continue
            if self.dict is None:
                self.dict = ddict
            else:
                for key in ddict.keys(): 
                    self.dict[key] = max(self.dict[key], ddict[key])

            data = BipartiteData(edge_index=edge_index, x_s=x_s, x_t=x_t, edge_attr=edge_attr)
            data.name = pro
            data_list.append(data)
            
        if len(data_list) == 0:
            raise ValueError('No data in the dataset')
        for data in tqdm(data_list, desc='Normalizing data'):
            data.x_t[:, -1] = data.x_t[:, -1]/ self.dict['varMaxScale']
            data.x_t[:, 5] = data.x_t[:, 5]/ self.dict['maxObj']

        with open(self.dict_path, 'wb') as f:
            pickle.dump(self.dict, f)
        # Concatenate the list of `Data` objects into a single `Data` object
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def getDict(self):
        return self.dict

    def clear_processed_files(self):
        """删除处理后的数据文件"""
        for file_name in os.listdir(self.processed_dir):
            file_path = os.path.join(self.processed_dir, file_name)
            if os.path.exists(file_path):
                os.remove(file_path)
        print("Processed files have been deleted.")