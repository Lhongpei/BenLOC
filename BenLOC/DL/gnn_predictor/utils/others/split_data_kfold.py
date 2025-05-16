import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
import os
# Read the dataset from CSV


def split_k_fold(dataset_path: str, label_folder: str, k: int = 5, test_size: float = 0.2, random_state: int = 42, accord: str = "mean_time"):
    dataset = pd.read_csv(dataset_path)
    if dataset.isna().any().any():
        print(f"Warning: {dataset_path} has NaN values.")
        dataset = dataset.dropna()
    # mean time is the mean of columns except the File Name column
    dataset["mean_time"] = dataset.iloc[:, 1:].mean(axis=1)

    # Convert "mean_time" into categories
    dataset['mean_time_category'] = pd.qcut(dataset['mean_time'], q=10, labels=False)

    # Split the dataset into train and test sets
    train_data, test_data = train_test_split(dataset, test_size=test_size, random_state=random_state, stratify=dataset['mean_time_category'])
    # print value counts of "mean_time_category" of train and test sets to check if they are stratified
    print(train_data['mean_time_category'].value_counts())
    print(test_data['mean_time_category'].value_counts())

    # Split the train data into 5-fold train-validation sets
    folds = 5
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    train_val_data = []

    for train_index, val_index in skf.split(train_data, train_data['mean_time_category']):
        fold_train_data = train_data.iloc[train_index]
        fold_val_data = train_data.iloc[val_index]
        train_val_data.append((fold_train_data, fold_val_data))
        
        # print value counts of "mean_time_category" of each fold to check if they are stratified
        print(f"Fold {len(train_val_data)}")
        print(fold_train_data['mean_time_category'].value_counts())
        print(fold_val_data['mean_time_category'].value_counts()) 

    # Save each fold to a CSV file
    label_folder = label_folder
    if not os.path.exists(label_folder):
        os.makedirs(label_folder)
    for i, (fold_train_data, fold_val_data) in enumerate(train_val_data):
        fold_train_data.to_csv(os.path.join(label_folder, f"fold_{i+1}_train.csv"), index=False)
        fold_val_data.to_csv(os.path.join(label_folder, f"fold_{i+1}_val.csv"), index=False)

    test_data.to_csv(os.path.join(label_folder, "test.csv"), index=False)

if __name__ == '__main__':
    dataset = 'setcover'
    dataset_path = dataset+"_table_dataset.csv"
    label_folder = "labels/"+dataset+"_fixed_5fold"
    split_k_fold(dataset_path, label_folder)
    for i in range(5):
        train_report = pd.read_csv(os.path.join(label_folder, 'fold_'+str(i+1)+'_train.csv'))
        valid_report = pd.read_csv(os.path.join(label_folder, 'fold_'+str(i+1)+'_val.csv'))
        
