import sys

sys.path.append('..')
sys.path.append('../..')
sys.path.append('.')
import os
import pickle

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData, InMemoryDataset
from tqdm import tqdm

from ml4moc.DL.gnn_predictor.dataset_gen.graphDataset import BipartiteData
from ml4moc.DL.gnn_predictor.utils.MIPmodel import MIPmodel
from ml4moc.DL.gnn_predictor.utils.utils import *
from ml4moc.DL.gnn_predictor.dataset_gen.heteroGraph import *

class RankListConfigNodesDataset(InMemoryDataset):

    def __init__(self, root, report_file_root, fold, reprocess=False, type = None, transform=None, pre_transform=None, report_norm=None):
        """Create a dataset from a folder of files. This dataset is designed for ranking task.

        Args:
            root (str): the root path of the dataset
            report_file_root (str): report file path
            reprocess (bool, optional): whether to reprocess the dataset. Defaults to False.
            type (str, optional): type should be one of the following: ["","train","test","valid"]. Defaults to None.
            transform (_type_, optional): _description_. Defaults to None.
            pre_transform (_type_, optional): _description_. Defaults to None.
        """
        assert root is not None, 'root path is None'
        assert os.path.exists(root), 'root path does not exist'
        assert type in ['train','test','val'], 'type should be one of the following: ["","train","test","valid"]'
        assert report_norm in ['minmax','std',None], 'report_norm should be one of the following: ["minmax","std"]'
        self.model = MIPmodel()
        self.root = root
        self.dict = None
        self.report_file_root = report_file_root
        self.fold = fold
        
        if report_norm == 'minmax':
            self.report_norm = minmaxNorm
        elif report_norm == 'std':
            self.report_norm = stdNorm
        else:
            self.report_norm = lambda x: x
            
        self.norm_mode = 'ori' if report_norm is None else report_norm
            
        if type == 'test':
            self.report_file = os.path.join(report_file_root, 'test.csv')
        else:
            self.report_file = os.path.join(report_file_root, 'fold_'+str(fold)+'_'+type+'.csv')
        
        if type != 'test':
            assert fold is not None, 'fold should not be None for train and valid dataset'
        if type is not None:
            self.type = type
            
        path = os.path.join(self.root, ('Hetero_confignodes_fold_' + str(self.fold))*(self.type != 'test') + self.type + '_ranklist_'+self.norm_mode)
        self.dict_path = os.path.join(path, 'dict.pkl')
        if reprocess:
            self.clear_processed_files()
            print('Reprocessing the dataset')
        print(f'Initializing {self.type} dataset ' + (self.type != 'test') * f'of fold {self.fold}')
        super(RankListConfigNodesDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
        if os.path.exists(self.dict_path):
            with open(self.dict_path, 'rb') as f:
                self.dict = pickle.load(f)
        else:
            self.dict = None
        # 初始化一个空的数据列表

    @property
    def processed_dir(self):
        return os.path.join(self.root, ('Hetero_confignodes_fold_' + str(self.fold))*(self.type != 'test') + self.type + '_ranklist_' + self.norm_mode)

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
        report = pd.read_csv(self.report_file)
        timeValue = torch.tensor(report.loc[:,report.columns.str.contains("\(")].values, dtype=torch.float32)
        configs = [eval(i) for i in report.columns if '\(' in i or '(' in i]
        weight = torch.var(timeValue, dim=-1)
        timeValue = self.report_norm(timeValue)
        #allFile = os.listdir(self.root)
        allPro = [report['File Name'][i] for i in range(len(report['File Name']))]
        for i,pro in enumerate(tqdm(allPro, desc='Adding data to dataset')):
            file_path = os.path.join(self.root, pro)
            if not os.path.exists(file_path):
                file_path = os.path.join('indset-flat', pro)
            #print(file_path)
            edge_index, edge_attr, x_s, x_t, ddict = (
                self.model.generBiparN2N(file_path))
            if self.dict is None:
                self.dict = ddict
            else:
                for key in ddict.keys(): 
                    self.dict[key] = max(self.dict[key], ddict[key])
            data = toHeteroData(edge_index=edge_index, x_s=x_s, x_t=x_t, edge_attr=edge_attr,undirected=True)
            data = addingConfigNode(data, configs, add_self = False, undirected = False, encMethod='totOnehot')
            data['y'] = timeValue[i].unsqueeze(0)
            data['name'] = pro
            data['weight'] = weight[i].item()
            #data.default = report['ref_Time'][i]
            #data.label = report['Solve time'][i]
            data_list.append(data)
        if len(data_list) == 0:
            raise ValueError('No data in the dataset')
        for data in tqdm(data_list, desc='Normalizing data'):
            data['vars'].x[:, -1] = data['vars'].x[:, -1]/ self.dict['varMaxScale']
            data['vars'].x[:, 5] = data['vars'].x[:, 5]/ self.dict['maxObj']

        with open(self.dict_path, 'wb') as f:
            pickle.dump(self.dict, f)
        # Concatenate the list of `Data` objects into a single `Data` object
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def getDict(self):
        return self.dict
    
    def clear_processed_files(self):
        """删除处理后的数据文件"""
        if not os.path.exists(self.processed_dir):
            print('No processed files to delete')
            return
        for file_name in os.listdir(self.processed_dir):
            file_path = os.path.join(self.processed_dir, file_name)
            if os.path.exists(file_path):
                os.remove(file_path)
        print("Processed files have been deleted.")

class RankListSolTimeDataset(InMemoryDataset):

    def __init__(self, root, report_file_root, fold, reprocess=False, type = None, transform=None, pre_transform=None, report_norm=None):
        """Create a dataset from a folder of files. This dataset is designed for ranking task.

        Args:
            root (str): the root path of the dataset
            report_file_root (str): report file path
            reprocess (bool, optional): whether to reprocess the dataset. Defaults to False.
            type (str, optional): type should be one of the following: ["","train","test","valid"]. Defaults to None.
            transform (_type_, optional): _description_. Defaults to None.
            pre_transform (_type_, optional): _description_. Defaults to None.
        """
        assert root is not None, 'root path is None'
        assert os.path.exists(root), 'root path does not exist'
        assert type in ['train','test','val'], 'type should be one of the following: ["","train","test","valid"]'
        assert report_norm in ['minmax','std',None], 'report_norm should be one of the following: ["minmax","std"]'
        self.model = MIPmodel()
        self.root = root
        self.dict = None
        self.report_file_root = report_file_root
        self.fold = fold
        
        if report_norm == 'minmax':
            self.report_norm = minmaxNorm
        elif report_norm == 'std':
            self.report_norm = stdNorm
        else:
            self.report_norm = lambda x: x
            
        self.norm_mode = 'ori' if report_norm is None else report_norm
            
        if type == 'test':
            self.report_file = os.path.join(report_file_root, 'test.csv')
        else:
            self.report_file = os.path.join(report_file_root, 'fold_'+str(fold)+'_'+type+'.csv')
        
        if type != 'test':
            assert fold is not None, 'fold should not be None for train and valid dataset'
        if type is not None:
            self.type = type
            
        path = os.path.join(self.root, ('Hetero_fold_' + str(self.fold))*(self.type != 'test') + self.type + '_ranklist_'+self.norm_mode)
        self.dict_path = os.path.join(path, 'dict.pkl')
        if reprocess:
            self.clear_processed_files()
            print('Reprocessing the dataset')
        print(f'Initializing {self.type} dataset ' + (self.type != 'test') * f'of fold {self.fold}')
        super(RankListSolTimeDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
        if os.path.exists(self.dict_path):
            with open(self.dict_path, 'rb') as f:
                self.dict = pickle.load(f)
        else:
            self.dict = None
        # 初始化一个空的数据列表

    @property
    def processed_dir(self):
        return os.path.join(self.root, ('Hetero_fold_' + str(self.fold))*(self.type != 'test') + self.type + '_ranklist_' + self.norm_mode)

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
        report = pd.read_csv(self.report_file)
        timeValue = torch.tensor(report.loc[:,report.columns.str.contains("\(")].values, dtype=torch.float32)
        weight = torch.var(timeValue, dim=-1)
        timeValue = self.report_norm(timeValue)
        #allFile = os.listdir(self.root)
        allPro = [report['File Name'][i] for i in range(len(report['File Name']))]
        for i,pro in enumerate(tqdm(allPro, desc='Adding data to dataset')):
            edge_index, edge_attr, x_s, x_t, ddict = (
                self.model.generBiparN2N(os.path.join(self.root, pro)))
            if self.dict is None:
                self.dict = ddict
            else:
                for key in ddict.keys(): 
                    self.dict[key] = max(self.dict[key], ddict[key])
            data = toHeteroData(edge_index=edge_index, x_s=x_s, x_t=x_t, edge_attr=edge_attr,undirected=True)
            data['y'] = timeValue[i].unsqueeze(0)
            data['name'] = pro
            data['weight'] = weight[i].item()
            #data.default = report['ref_Time'][i]
            #data.label = report['Solve time'][i]
            data_list.append(data)
        if len(data_list) == 0:
            raise ValueError('No data in the dataset')
        for data in tqdm(data_list, desc='Normalizing data'):
            data['vars'].x[:, -1] = data['vars'].x[:, -1]/ self.dict['varMaxScale']
            data['vars'].x[:, 5] = data['vars'].x[:, 5]/ self.dict['maxObj']

        with open(self.dict_path, 'wb') as f:
            pickle.dump(self.dict, f)
        # Concatenate the list of `Data` objects into a single `Data` object
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def getDict(self):
        return self.dict
    
    def clear_processed_files(self):
        """删除处理后的数据文件"""
        if not os.path.exists(self.processed_dir):
            print('No processed files to delete')
            return
        for file_name in os.listdir(self.processed_dir):
            file_path = os.path.join(self.processed_dir, file_name)
            if os.path.exists(file_path):
                os.remove(file_path)
        print("Processed files have been deleted.")

class RegrSolTimeHeteroDataset(InMemoryDataset):
    
    def __init__(self, root, report_file_root, fold, reprocess=False, type = None, transform=None, pre_transform=None, report_norm=None):
        """Create a dataset from a folder of files. This dataset is designed for ranking task.

        Args:
            root (str): the root path of the dataset
            report_file (str): report file path
            reprocess (bool, optional): whether to reprocess the dataset. Defaults to False.
            type (str, optional): type should be one of the following: ["","train","test","valid"]. Defaults to None.
            transform (_type_, optional): _description_. Defaults to None.
            pre_transform (_type_, optional): _description_. Defaults to None.
        """
        assert root is not None, 'root path is None'
        assert os.path.exists(root), 'root path does not exist'
        assert type in ['train','test','val'], 'type should be one of the following: ["","train","test","valid"]'
        self.model = MIPmodel()
        self.root = root
        self.dict = None
        self.report_file_root = report_file_root
        
        if report_norm == 'minmax':
            self.report_norm = minmaxNorm
        elif report_norm == 'std':
            self.report_norm = stdNorm
        else:
            self.report_norm = lambda x: x
            
        self.norm_mode = 'ori' if report_norm is None else report_norm
        
        if type == 'test':
            self.report_file = os.path.join(report_file_root, 'test.csv')
        else:
            self.report_file = os.path.join(report_file_root, 'fold_'+str(fold)+'_'+type+'.csv')
        self.fold = fold
        if type != 'test':
            assert fold is not None, 'fold should not be None for train and valid dataset'
        if type is not None:
            self.type = type
        path = os.path.join(self.root, ('Hetero_fold_' + str(self.fold))*(self.type != 'test')+ '-' + self.type + '_regr_' + self.norm_mode)
        self.dict_path = os.path.join(path, 'dict.pkl')
        if reprocess:
            self.clear_processed_files()
            
        print(f'Initializing {self.type} dataset ' + (self.type != 'test') * f'of fold {self.fold}')
        super(RegrSolTimeHeteroDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
        if os.path.exists(self.dict_path):
            with open(self.dict_path, 'rb') as f:
                self.dict = pickle.load(f)
        else:
            self.dict = None
        # 初始化一个空的数据列表

    @property
    def processed_dir(self):
        return os.path.join(self.root, ('Hetero_fold_' + str(self.fold))*(self.type != 'test')+ '-' + self.type + '_regr_' + self.norm_mode)

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
        report = pd.read_csv(self.report_file)
        timeValue = torch.tensor(report.loc[:,report.columns.str.contains("\(")].values, dtype=torch.float32)
        timeValue = self.report_norm(timeValue)
        config = [configEncode(eval(i)) for i in report.columns if i not in ['File Name','mean_time_category', 'mean_time','type']]
        #allFile = os.listdir(self.root)
        allPro = [report['File Name'][i] for i in range(len(report['File Name']))]
        
        for i,pro in enumerate(tqdm(allPro, desc='Adding data to dataset')):
            edge_index, edge_attr, x_s, x_t, ddict = (
                self.model.generBiparN2N(os.path.join(self.root, pro)))
            if self.dict is None:
                self.dict = ddict
            else:
                for key in ddict.keys(): 
                    self.dict[key] = max(self.dict[key], ddict[key])
            for j in range(np.size(timeValue,-1)):
                data = toHeteroData(edge_index=edge_index, x_s=x_s, x_t=x_t, edge_attr=edge_attr,undirected=True)
                data['config'] = torch.tensor(config[j], dtype=torch.float).unsqueeze(0)
                data['name'] = pro
                data['default'] = timeValue[i][0]
                data['y'] = timeValue[i][j].unsqueeze(0)
                data_list.append(data)
        if len(data_list) == 0:
            raise ValueError('No data in the dataset')
        for data in tqdm(data_list, desc='Normalizing data'):
            data['vars'].x[:, -1] = data['vars'].x[:, -1]/ self.dict['varMaxScale']
            data['vars'].x[:, 5] = data['vars'].x[:, 5]/ self.dict['maxObj']

        with open(self.dict_path, 'wb') as f:
            pickle.dump(self.dict, f)
        # Concatenate the list of `Data` objects into a single `Data` object
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def getDict(self):
        return self.dict
    
    def clear_processed_files(self):
        """删除处理后的数据文件"""
        if not os.path.exists(self.processed_dir):
            print('No processed files to delete')
            return
        for file_name in os.listdir(self.processed_dir):
            file_path = os.path.join(self.processed_dir, file_name)
            if os.path.exists(file_path):
                os.remove(file_path)
        print("Processed files have been deleted.")


class RegrSolTimeHeteroDataset_tiny(InMemoryDataset):
    
    def __init__(self, root, report_file_root, fold, reprocess=False, type = None, transform=None, pre_transform=None, report_norm=None, config_num=1, prob_num=10):
        """Create a dataset from a folder of files. This dataset is designed for ranking task.

        Args:
            root (str): the root path of the dataset
            report_file (str): report file path
            reprocess (bool, optional): whether to reprocess the dataset. Defaults to False.
            type (str, optional): type should be one of the following: ["","train","test","valid"]. Defaults to None.
            transform (_type_, optional): _description_. Defaults to None.
            pre_transform (_type_, optional): _description_. Defaults to None.
        """
        assert root is not None, 'root path is None'
        assert os.path.exists(root), 'root path does not exist'
        assert type in ['train','test','val'], 'type should be one of the following: ["","train","test","valid"]'
        self.model = MIPmodel()
        self.root = root
        self.dict = None
        self.report_file_root = report_file_root
        self.prob_num = prob_num
        self.config_num = config_num
        if report_norm == 'minmax':
            self.report_norm = minmaxNorm
        elif report_norm == 'std':
            self.report_norm = stdNorm
        else:
            self.report_norm = lambda x: x
            
        self.norm_mode = 'ori' if report_norm is None else report_norm
        
        if type == 'test':
            self.report_file = os.path.join(report_file_root, 'test.csv')
        else:
            self.report_file = os.path.join(report_file_root, 'fold_'+str(fold)+'_'+type+'.csv')
        self.fold = fold
        if type != 'test':
            assert fold is not None, 'fold should not be None for train and valid dataset'
        if type is not None:
            self.type = type
        path = os.path.join(self.root, ('Hetero_fold_' + str(self.fold))*(self.type != 'test')+ '-' + self.type + '_regr_' + self.norm_mode)
        self.dict_path = os.path.join(path, 'dict.pkl')
        if reprocess:
            self.clear_processed_files()
            
        print(f'Initializing {self.type} dataset ' + (self.type != 'test') * f'of fold {self.fold}')
        super(RegrSolTimeHeteroDataset_tiny, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
        if os.path.exists(self.dict_path):
            with open(self.dict_path, 'rb') as f:
                self.dict = pickle.load(f)
        else:
            self.dict = None
        # 初始化一个空的数据列表

    @property
    def processed_dir(self):
        return os.path.join(self.root, ('Hetero_fold_' + str(self.fold))*(self.type != 'test')+ '-' + self.type + '_regr_' + self.norm_mode)

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
        report = pd.read_csv(self.report_file)
        allPro = random.sample([report['File Name'][i] for i in range(len(report['File Name']))], self.prob_num)
        report=report[report['File Name'].isin(allPro)]
        
        timeValue = torch.tensor(report.loc[:,report.columns.str.contains("\(")].values, dtype=torch.float32)
        timeValue = self.report_norm(timeValue)
        config = [configEncode(eval(i)) for i in report.columns if i not in ['File Name','mean_time_category', 'mean_time','type']]
        #allFile = os.listdir(self.root)
        
        for i,pro in enumerate(tqdm(allPro, desc='Adding data to dataset')):
            edge_index, edge_attr, x_s, x_t, ddict = (
                self.model.generBiparN2N(os.path.join(self.root, pro)))
            if self.dict is None:
                self.dict = ddict
            else:
                for key in ddict.keys(): 
                    self.dict[key] = max(self.dict[key], ddict[key])
            for j in range(self.config_num):
                data = toHeteroData(edge_index=edge_index, x_s=x_s, x_t=x_t, edge_attr=edge_attr)
                data['config'] = torch.tensor(config[j], dtype=torch.float).unsqueeze(0)
                data['name'] = pro
                data['default'] = timeValue[i][0]
                data['y'] = timeValue[i][j].unsqueeze(0)
                data_list.append(data)
        if len(data_list) == 0:
            raise ValueError('No data in the dataset')
        for data in tqdm(data_list, desc='Normalizing data'):
            data['vars'].x[:, -1] = data['vars'].x[:, -1]/ self.dict['varMaxScale']
            data['vars'].x[:, 5] = data['vars'].x[:, 5]/ self.dict['maxObj']

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
  