import pandas as pd
import numpy as np
import os
def split_k_folds(data:pd.DataFrame, k:int = 5, seed:int = 42):
    """
    This function is used to split the data into k folds.

    Args:
        data (pandas.DataFrame): The data to be split.
        k (int): The number of folds.
        seed (int, optional): The seed of the random state. Defaults to 0.

    Returns:
        list: The list of the k folds.
    """
    np.random.seed(seed)
    data = data.sample(frac=1).reset_index(drop=True)
    n = len(data)

    fold_size = n // k
    folds = []
    for i in range(k):
        if i == k - 1:
            fold = data[i * fold_size:].reset_index(drop=True)
        else:
            fold = data[i * fold_size: (i + 1) * fold_size].reset_index(drop=True)
        folds.append(fold)

    return folds


def cross_validation_split(data:pd.DataFrame,  k:int = 5, seed:int = 42, use_val=False):
    """
    This function is used to perform the cross validation.

    Args:
        data (pandas.DataFrame): The data to be split.
        k (int, optional): The number of folds. Defaults to 5.
        seed (int, optional): The seed of the random state. Defaults to 0.

    Returns:
        list: The list of the k folds.
    """
    folds = split_k_folds(data, k, seed)
    trains = []
    tests = []
    vals = [] if use_val else None
    for i in range(k):
        if use_val:
            val = folds[i].sample(frac=0.3)
            test = folds[i].drop(val.index)
            vals.append(val)
            tests.append(test)
        else:
            test = folds[i]
            tests.append(test) 
        train = pd.concat([folds[j] for j in range(k) if j != i]).reset_index(drop=True)
        trains.append(train)

    return trains, vals, tests



if __name__ == '__main__':
    data = pd.read_csv('time_1064_new.csv')
    trains, vals, tests = cross_validation_split(data,  k=5, seed=42, use_val=True)
    # if not os.path.exists('repo/labels/mipcut_bin_5fold'):
    #     os.mkdir('repo/labels/mipcut_bin_5fold')
    for i in range(5):
        print(f'Fold {i+1} train shape: {trains[i].shape}, val shape: {vals[i].shape}, test shape: {tests[i].shape}')
        trains[i].to_csv(f'repo/datasets/miplib/labels/kfold/fold_{i+1}_train.csv', index=False)
        vals[i].to_csv(f'repo/datasets/miplib/labels/kfold/fold_{i+1}_val.csv', index=False)
        tests[i].to_csv(f'repo/datasets/miplib/labels/kfold/fold_{i+1}_test.csv', index=False)