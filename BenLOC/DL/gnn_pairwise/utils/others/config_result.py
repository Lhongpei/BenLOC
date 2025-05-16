import pandas as pd
import os
def findRank(ranks_df, target):
    """
    Find the rank of the configurations in the target based on the ranks dataframe.

    Args:
        ranks_df (pd.DataFrame): DataFrame where each element denotes the rank of a configuration for a given instance.
        target (pd.Series): Series where each element is the configuration number for a given instance.

    Returns:
        pd.DataFrame: A DataFrame with the original target configurations and their corresponding ranks.
    """
    rank_list = []
    for i in range(len(target.index)):
        rank_list.append(ranks_df.iloc[i, int(target[i])])
    rank_series = pd.Series(rank_list, index=target.index)
    df = pd.concat([target, rank_series], axis=1)
    df.columns = ['config', 'rank']
    df['config'] = df['config'].astype(int)  # 将config列转换为整数类型
    df['rank'] = df['rank'].astype(int) 
    return df
        
    
dataset_list = ['repo/labels/setcover_table_dataset.csv', 'repo/labels/indset_table_dataset.csv']
tot_data_df = pd.concat([pd.read_csv(f,index_col=0) for f in dataset_list]).loc[:, lambda df: df.columns.str.contains('\(')]
rank_df = tot_data_df.apply(lambda row: row.argsort(), axis=1)

result_file = 'repo/fold_rank_result/config_result_fold_1.csv'
config_result_df = pd.read_csv(result_file,index_col=0)

train_result = config_result_df['fold_1_train'].dropna()
val_result = config_result_df['fold_1_valid'].dropna()
test_result = config_result_df['fold_1_test'].dropna()

print(findRank(rank_df, train_result))