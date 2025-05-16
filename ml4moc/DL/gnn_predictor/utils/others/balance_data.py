import os
import sys
import pandas as pd 
import numpy as np
import random
import sklearn
from torch.utils.data import random_split
path = ['repo/labels/indset_table_dataset.csv', 'repo/labels/setcover_table_dataset.csv']
print(os.getcwd())
df_list = [pd.read_csv(p) for p in path]

tot_data = pd.concat(df_list, ignore_index=True)
tot_data.set_index('File Name', inplace=True)
tot_data = tot_data.loc[:, tot_data.columns.str.contains('\(')]
cat_dict = {i:[] for i in tot_data.columns if '(' in i}
col = tot_data.columns
for i in range(len(tot_data)):
    data = tot_data.iloc[i]
    cat_dict[col[data.argmin()]].append(data.name)
    
print([len(cat_dict[i]) for i in cat_dict])   
print([i for i in cat_dict if len(cat_dict[i]) > 252])

if not os.path.exists('labels/balanced_setcover'):
    os.makedirs('labels/balanced_setcover')

fold = 5
for t in range(1,fold+1):
    print(f'fold {t}')
    train_name = []
    valid_name = []
    test_name = []
    for i in cat_dict:
        if len(cat_dict[i]) > 252:
            train_len, valid_len,test_len = 180, 20, 52
            others = len(cat_dict[i]) - train_len - valid_len - test_len
            train,val,test,_ = random_split(cat_dict[i], [train_len, valid_len, test_len, others])
            train_name.extend(train)
            valid_name.extend(val)
            test_name.extend(test)
    large_categories = [col_name for col_name, items in cat_dict.items() if len(items) > 252]

    # 筛选tot_data中，列名在large_categories中的列
    balance_data = tot_data
    train_df = balance_data.loc[train_name]
    val_df = balance_data.loc[valid_name]
    test_df = balance_data.loc[test_name]
    train_df['File Name'] = train_df.index
    val_df['File Name'] = val_df.index
    test_df['File Name'] = test_df.index

        
    train_df.to_csv(f'repo/labels/balanced_setcover/fold_{t}_train.csv')
    val_df.to_csv(f'repo/labels/balanced_setcover/fold_{t}_val.csv')
    test_df.to_csv(f'repo/labels/balanced_setcover/fold_{t}_test.csv')