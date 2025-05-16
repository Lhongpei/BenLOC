import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split


def melt_data(res_path: str):
    # Define the column names for the 25 different configurations
    config_cols = ["(-1, -1)","(-1, 0)","(-1, 1)","(-1, 2)","(-1, 3)","(0, -1)","(0, 0)","(0, 1)","(0, 2)","(0, 3)","(1, -1)","(1, 0)","(1, 1)","(1, 2)","(1, 3)","(2, -1)","(2, 0)","(2, 1)","(2, 2)","(2, 3)","(3, -1)","(3, 0)","(3, 1)","(3, 2)","(3, 3)"]

    # Reshape the data
    df = pd.read_csv(res_path)
    
    df = df.rename(columns={"Unnamed: 0": "File Name"})  # FIXME: bug in data

    # Unpivot the DataFrame from wide format to long format
    df_melt = pd.melt(df, id_vars='File Name', value_vars=config_cols, var_name='config', value_name='time')
    return df_melt


def df_preprocess(res_path: str, ref_df, config_cols):
    df = melt_data(res_path)
    
    # Merge the data
    df = pd.merge(ref_df, df, left_on='instance_name', right_on='File Name')
    
    instance_name = df['instance_name']
    # Drop the 'File Name' column as it's no longer needed
    df = df.drop(columns=['File Name', 'instance_name'])
    
    # Transform 'config' column into 2 features
    df[config_cols] = df['config'].str.strip('()').str.split(',', expand=True).astype(int)
    
    # drop the 'config' column
    configs = df["config"]
    df = df.drop(columns=['config'])

    # Perform one-hot encoding for the new columns
    df = pd.get_dummies(df, columns=config_cols)

    # assert no NaN values
    assert df.isnull().sum().sum() == 0
    # assert all columns are numeric or boolean
    assert all(np.issubdtype(df[col].dtype, np.number) or df[col].dtype == bool for col in df.columns)

    # Perform LightGBM regression to learn the time
    X = df.drop(columns=['time'])
    y = df['time']
    
    return X, y, (instance_name, configs)

if __name__ == '__main__':
    ## Data Preprocessing
    # Read the data
    ref_df = pd.read_csv('data/setcover-flat.csv')

    # Add '.mps.gz' to the end of 'instance_name'
    ref_df['instance_name'] = ref_df['instance_name'] + '.mps.gz'

    config_cols = ['RootCutLevel', 'TreeCutLevel']

    X_test, y_test, _ = df_preprocess('setcover_fixed_5fold/test.csv', ref_df, config_cols)

    # Initialize lists to store the training and validation data
    X_train_list = []
    y_train_list = []
    X_valid_list = []
    y_valid_list = []

    # Load each data from setcover_fixed_5fold and append the data to the lists
    for i in range(1, 2):
        X_train, y_train, _ = df_preprocess(f'setcover_fixed_5fold/fold_{i}_train.csv', ref_df, config_cols)
        X_valid, y_valid, _ = df_preprocess(f'setcover_fixed_5fold/fold_{i}_val.csv', ref_df, config_cols)

        X_train_list.append(X_train)
        y_train_list.append(y_train)
        X_valid_list.append(X_valid)
        y_valid_list.append(y_valid)

    # Concatenate the data from the 5 folds
    X_train = pd.concat(X_train_list)
    y_train = pd.concat(y_train_list)
    X_valid = pd.concat(X_valid_list)
    y_valid = pd.concat(y_valid_list)
    
    # hyperparameters tuning
    # Define the parameter grid
    parameters = {
    'task' : ['predict'],
    'boosting': ['gbdt' ],
    'objective': ['mean_squared_error'],
    'num_iterations': [  1500, 2000,5000  ],
    'learning_rate':[  0.05, 0.005 ],
    'num_leaves':[ 7, 15, 31  ],
    'max_depth' :[ 10,15,25],
    'min_data_in_leaf':[15,25 ],
    'feature_fraction': [ 0.6, 0.8,  0.9],
    'bagging_fraction': [  0.6, 0.8 ],
    'bagging_freq': [   100, 200, 400  ],
    }
    parameters = {
    'task' : ['predict'],
    'boosting': ['gbdt' ],
    'objective': ['mean_squared_error'],
    # 'num_iterations': [  1500],
    # 'learning_rate':[  0.05],
    'lambda_l1': [ 0, 0.001, 0.01 ],
    'num_leaves':[ 3, 5, 10  ],
    'max_depth' :[ 2, 4, 8 ],
    # 'min_data_in_leaf':[ 20 ],
    'feature_fraction': [ 1.0, 0.9, 0.8],
    'bagging_fraction': [ 1.0, 0.9, 0.8 ],
    # 'bagging_freq': [    400  ],
    'n_estimators': [ 41, 51, 61],
    }

    # Initialize the LightGBM regressor
    gbm = lgb.LGBMRegressor(boosting_type='gbdt', objective='mean_squared_error', metric=['l2'], verbose=0)

    # Initialize the grid search
    grid = GridSearchCV(gbm, parameters, cv=5, scoring='neg_mean_squared_error', verbose=1, n_jobs=8)
    
    grid.fit(X_train, y_train)
    print('Best parameters found by grid search are:', grid.best_params_)
    
    gbm = lgb.LGBMRegressor(**grid.best_params_)

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_valid = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

    gbm.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
    
    # save the model to disk
    gbm.booster_.save_model(f'model_{str(grid.best_params_)}.txt')

    y_train_pred = gbm.predict(X_train, num_iteration=gbm.best_iteration_)
    y_valid_pred = gbm.predict(X_valid, num_iteration=gbm.best_iteration_)
    y_test_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)

    # Print the MSE
    print(f'The mse of prediction train is:', mean_squared_error(y_train, y_train_pred))
    print(f'The mse of prediction valid is:', mean_squared_error(y_valid, y_valid_pred))
    print(f'The mse of prediction test is:', mean_squared_error(y_test, y_test_pred))
