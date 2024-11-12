import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import argparse
import os
def splitfold(name,time_path,feat_path):
    time=pd.read_csv(f'{time_path}/time_{name}.csv')
    feat=pd.read_csv(f'{feat_path}/feat_{name}.csv')
    # time=time[['File Name','SubMipHeurLevel-0','MipLogLevel-2']]
    df=pd.merge(feat,time,on='File Name',how='inner')
    time = time[time['File Name'].isin(df['File Name'].tolist())]

    print(df.shape)
    min_ratios = time.iloc[:, 1:].apply(lambda x: (x == x.min()).mean(), axis=1)
    time['min_ratio'] = min_ratios
    
    single_class_samples = time['min_ratio'].value_counts().loc[lambda x: x == 1].index
    single_class_indices = time[time['min_ratio'].isin(single_class_samples)].index
    # print(len(single_class_samples))

    single_class_test_mask = np.random.choice([True, False], size=len(single_class_indices))
    single_class_train_indices = single_class_indices[~single_class_test_mask]
    single_class_test_indices = single_class_indices[single_class_test_mask]
    
    time = time[~time['min_ratio'].isin(single_class_samples)]

    strat_split = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=100)
    
    for fold, (train_idx, test_idx) in enumerate(strat_split.split(time.iloc[:, 1:], time['min_ratio']), 1):
        train_idx = np.concatenate([train_idx, single_class_train_indices])
        test_idx = np.concatenate([test_idx, single_class_test_indices])

        train_data = df.loc[train_idx]
        test_data = df.loc[test_idx]

        os.makedirs(f'./data/{name}')
        train_data.to_csv(f'./data/{name}/fold_{fold}_train.csv', index=False)
        test_data.to_csv(f'./data/{name}/fold_{fold}_test.csv', index=False)

def main():
    parser = argparse.ArgumentParser(description='Feature combination parser')
    parser.add_argument('--dataset_name', dest = 'dataset_name', type = str, default = 'miplib', help = 'name of the dataset')
    parser.add_argument('--time_path', dest = 'time_path', type = str, default = './data', help = 'which folder to get the solving time')
    parser.add_argument('--feat_path', dest = 'feat_path', type = str, default = './data', help = 'which folder to get the features')

    args = parser.parse_args()

    splitfold(args.dataset_name,args.time_path,args.feat_path)

if __name__ == '__main__':
    main()