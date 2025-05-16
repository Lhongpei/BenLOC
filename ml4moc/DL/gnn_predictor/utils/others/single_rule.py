import pandas as pd 
import os 
def cal_mean_time(file_path: str):
    df = pd.read_csv(file_path)
    df = df.loc[:, df.columns.str.contains("\(")]
    mean = df.mean(axis=0)
    improvement = (mean[0]-mean)/mean[0]
    best_improvemrnt = improvement.max()
    return mean, improvement, best_improvemrnt
def cal_single_rule_fold(root_path:str):
    _, _, test_best = cal_mean_time(os.path.join(root_path, 'test.csv'))
    print(f"Test improvement: {test_best}")
    for i in range(5):
        train_report = os.path.join(root_path, 'fold_'+str(i+1)+'_train.csv')
        valid_report = os.path.join(root_path, 'fold_'+str(i+1)+'_val.csv')
        _, _, train_best = cal_mean_time(train_report)
        _, _, valid_best = cal_mean_time(valid_report)
        print(f"Fold {i+1} improvement: {train_best}, {valid_best}")
def cal_info_dataset(file_path: str):
    df = pd.read_csv(file_path)
    df = df.loc[:, df.columns.str.contains("\(")]
    mean = df.mean(axis=0)
    improvement = (mean[0]-mean)/mean[0]
    best_config = improvement.idxmax()
    best_improvemrnt = improvement.max()
    default = df.iloc[:,0].sum()
    min_time = df.min(axis=1).sum()
    global_min = (default-df.min(axis=1).sum())/default
    return pd.Series({df.columns[i]:improvement[i] for i in range(len(improvement))}|{'single_best_improvemrnt':best_improvemrnt, 'single_best_config':best_config,'global_best_improvement':global_min})
if __name__ == '__main__':
    # root_path = 'labels/setcover_fixed_5fold'
    # cal_single_rule_fold(root_path)
    # file_path = ["indset_table_dataset.csv", "setcover_table_dataset.csv"]
    # print(pd.DataFrame([cal_info_dataset(i) for i in file_path], index = ['indset', 'setcover']).T)
    fold_root = 'repo/labels/balanced_setcover'
    full_df = pd.DataFrame()
    fold_mean_dict = pd.DataFrame({i : {j : 0 for j in ['train', 'val', 'test']} for i in ['single_best_improvemrnt', 'global_best_improvement']})
    # 遍历所有的 fold
    for i in range(5):
        fold_number = i + 1
        print(f"Fold {fold_number}")
        df_fold = pd.concat([
            cal_info_dataset(os.path.join(fold_root, f'fold_{fold_number}_train.csv')),
            cal_info_dataset(os.path.join(fold_root, f'fold_{fold_number}_val.csv')),
            cal_info_dataset(os.path.join(fold_root, f'fold_{fold_number}_test.csv'))
        ], axis=1)
        df_fold.columns = ['train', 'val', 'test']
        fold_mean_dict['single_best_improvemrnt'][f'train'] += df_fold.loc['single_best_improvemrnt', 'train']
        fold_mean_dict['single_best_improvemrnt'][f'val'] += df_fold.loc['single_best_improvemrnt', 'val']
        fold_mean_dict['single_best_improvemrnt'][f'test'] += df_fold.loc['single_best_improvemrnt', 'test']
        fold_mean_dict['global_best_improvement'][f'train'] += df_fold.loc['global_best_improvement', 'train']
        fold_mean_dict['global_best_improvement'][f'val'] += df_fold.loc['global_best_improvement', 'val']
        fold_mean_dict['global_best_improvement'][f'test'] += df_fold.loc['global_best_improvement', 'test']

        # 为每个fold创建多层次列
        df_fold.columns = pd.MultiIndex.from_product([[f'Fold {fold_number}'], ['train', 'val', 'test']])
        
        # 将当前的 fold DataFrame 拼接到全局 DataFrame 中
        if full_df.empty:
            full_df = df_fold
        else:
            full_df = pd.concat([full_df, df_fold], axis=1)
    fold_mean_dict = fold_mean_dict.apply(lambda x: x/5)
    print(full_df)
    print(fold_mean_dict)
    