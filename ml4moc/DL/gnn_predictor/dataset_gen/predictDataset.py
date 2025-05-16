import sys

sys.path.append('..')
sys.path.append('../..')
sys.path.append('.')
import os
import pickle

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm

from ml4moc.DL.gnn_predictor.dataset_gen.graphDataset import BipartiteData
from ml4moc.DL.gnn_predictor.utils.MIPmodel import MIPmodel
from ml4moc.DL.gnn_predictor.utils.utils import *


def pivot_resplit_report(path = 'parsed_log.csv', store_path = 'datasetOfSetcover.csv', seed=42, resplit_num=650):
    """To re-split the dataset into train, valid and test set, pivoting the result in matrix shape and store the result to a new csv file.

    Args:
        path (str, optional): . Defaults to 'parsed_log.csv'.
        store_path (str, optional): . Defaults to 'datasetOfSetcover.csv'.
        seed (int, optional): . Defaults to 42.
        resplit_num (int, optional): . Defaults to 650. design to assign this number of samples from train set to valid set.
    """
    df = pd.read_csv(path)
    df.dropna(subset=['Solution Status'], inplace=True)
    df = df[df['Solution Status'].str.contains('optimal', case=False)]
    df['Config'] = df.apply(lambda row: (row['RootCutLevel'], row['TreeCutLevel']), axis=1)
    resultDF = df.pivot_table(index='File Name', columns='Config', values='Solve time').dropna()
    resultDF.head()

    df_sample = resultDF[resultDF.index.str.contains('train')].sample(n=resplit_num, random_state=seed)
    resultDF = resultDF.drop(df_sample.index)
    resultDF['type'] = resultDF.index.str.extract(r'([a-z]+)', expand=False)
    df_sample['type'] = 'valid'
    df_new = pd.concat([resultDF, df_sample]) 
    print('Dataset Splitting result is:\n', df_new['type'].value_counts())

    df_new.to_csv(store_path)
    print(f'New dataset solving time report is saved to {store_path}')

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
            
        path = os.path.join(self.root, ('fold_' + str(self.fold))*(self.type != 'test') + self.type + '_ranklist_'+self.norm_mode)
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
        return os.path.join(self.root, ('fold_' + str(self.fold))*(self.type != 'test') + self.type + '_ranklist_' + self.norm_mode)

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
            data = BipartiteData(edge_index=edge_index, x_s=x_s, x_t=x_t, edge_attr=edge_attr)
            data.y = timeValue[i].unsqueeze(0)
            data.name = pro
            data.weight = weight[i].item()
            #data.default = report['ref_Time'][i]
            #data.label = report['Solve time'][i]
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

class RegrSolTimeDataset(InMemoryDataset):
    
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
        path = os.path.join(self.root, ('fold_' + str(self.fold))*(self.type != 'test') + self.type + '_regr_' + self.norm_mode)
        self.dict_path = os.path.join(path, 'dict.pkl')
        if reprocess:
            self.clear_processed_files()
            
        print(f'Initializing {self.type} dataset ' + (self.type != 'test') * f'of fold {self.fold}')
        super(RegrSolTimeDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
        if os.path.exists(self.dict_path):
            with open(self.dict_path, 'rb') as f:
                self.dict = pickle.load(f)
        else:
            self.dict = None
        # 初始化一个空的数据列表

    @property
    def processed_dir(self):
        return os.path.join(self.root, ('fold_' + str(self.fold))*(self.type != 'test') + self.type + '_regr_' + self.norm_mode)

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
                data = BipartiteData(edge_index=edge_index, x_s=x_s, x_t=x_t, edge_attr=edge_attr)
                data.config = torch.tensor(config[j], dtype=torch.float).unsqueeze(0)
                data.name = pro
                data.default = timeValue[i][0]
                data.y = timeValue[i][j].unsqueeze(0)
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
        
class RegrSolTimeDatasetTiny(InMemoryDataset):
    
    def __init__(self, root, report_file_root, fold, reprocess=False, type = None, transform=None, pre_transform=None, report_norm=None, config_num=1, pro_num=10):
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
        self.config_num = config_num
        self.pro_num = pro_num
        if type == 'test':
            self.report_file = os.path.join(report_file_root, 'test.csv')
        else:
            self.report_file = os.path.join(report_file_root, 'fold_'+str(fold)+'_'+type+'.csv')
        self.fold = fold
        if type != 'test':
            assert fold is not None, 'fold should not be None for train and valid dataset'
        if type is not None:
            self.type = type
        path = os.path.join(self.root, ('fold_' + str(self.fold))*(self.type != 'test') + self.type + '_regr_' + self.norm_mode+'_'+str(self.config_num)+'_'+str(self.pro_num))
        self.dict_path = os.path.join(path, 'dict.pkl')
        if reprocess:
            self.clear_processed_files()
            
        print(f'Initializing {self.type} dataset ' + (self.type != 'test') * f'of fold {self.fold}')
        super(RegrSolTimeDatasetTiny, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
        if os.path.exists(self.dict_path):
            with open(self.dict_path, 'rb') as f:
                self.dict = pickle.load(f)
        else:
            self.dict = None
        # 初始化一个空的数据列表

    @property
    def processed_dir(self):
        return os.path.join(self.root, ('fold_' + str(self.fold))*(self.type != 'test') + self.type + '_regr_' + self.norm_mode+'_'+str(self.config_num)+'_'+str(self.pro_num))

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
        allPro = random.sample([report['File Name'][i] for i in range(len(report['File Name']))], self.pro_num)
        
        for i,pro in enumerate(tqdm(allPro, desc='Adding data to dataset')):
            edge_index, edge_attr, x_s, x_t, ddict = (
                self.model.generBiparN2N(os.path.join(self.root, pro)))
            if self.dict is None:
                self.dict = ddict
            else:
                for key in ddict.keys(): 
                    self.dict[key] = max(self.dict[key], ddict[key])
            for j in range(self.config_num):
                data = BipartiteData(edge_index=edge_index, x_s=x_s, x_t=x_t, edge_attr=edge_attr)
                data.config = torch.tensor(config[j], dtype=torch.float).unsqueeze(0)
                data.name = pro
                data.default = timeValue[i][0]
                data.y = timeValue[i][j].unsqueeze(0)
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
        
if __name__ == '__main__':
    root = 'setcover-flat'
    report_root = 'setcover_fixed_5fold'
    fold = 1
    reprocess = False
    type = 'train'
    p = RegrSolTimeDatasetTiny(root, report_root, fold, reprocess, type, config_num=1, pro_num=10)
    for i in p:
        print(i.y)