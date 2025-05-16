from main import df_preprocess
import pandas as pd
from sklearn.metrics import mean_squared_error


def unmelt_data(df_melt):
    # cols = ["RootCutLevel_{i}" for i in range(-1, 4)] + ["TreeCutLevel_{i}" for i in range(-1, 4)]
    # # for example if RootCutLevel_i = True, and TreeCutLevel_j = True, then the config is (i, j)
    # # it's an one-hot encoding, I want to transform it back
    # df_melt["RootCutLevel"] = df_melt[["RootCutLevel_{i}".format(i=i) for i in range(-1, 4)]].idxmax(axis=1).str.extract(r'(\d+)').astype(int)
    # df_melt["TreeCutLevel"] = df_melt[["TreeCutLevel_{i}".format(i=i) for i in range(-1, 4)]].idxmax(axis=1).str.extract(r'(\d+)').astype(int)
    # df_melt["config"] = df_melt[["RootCutLevel", "TreeCutLevel"]].apply(tuple, axis=1).astype(str)
    # Pivot the DataFrame from long format to wide format
    df = df_melt.pivot(index='instance_name', columns='config', values='time')
    
    # Reset the index
    df.reset_index(inplace=True)
    
    return df


def get_best_config(df):
    # Get the best configuration for each instance
    df['best_config'] = df.iloc[:, 1:].idxmin(axis=1)

    return df


def cal_improvement(df, df_pred):
    assert 'best_config' in df_pred.columns
    default_config = "(-1, -1)"
    
    # Assuming df_pred['best_config'] contains the column names as they appear in df
    time_pred = df.apply(lambda x: x[df_pred.loc[x.name, 'best_config']], axis=1)
    
    improve_ratio = 1 - time_pred.mean() / df[default_config].mean()
    
    return improve_ratio


if __name__ == "__main__":
    import lightgbm as lgb

    # Read the model from the file
    with open('model.txt', 'r') as f:
        model_str = f.read()

    # Load the model from the string
    model = lgb.Booster(model_str=model_str)
    
    # Read the data
    ref_df = pd.read_csv('data/setcover-flat.csv')
    
    config_cols = ['RootCutLevel', 'TreeCutLevel']

    # Add '.mps.gz' to the end of 'instance_name'
    ref_df['instance_name'] = ref_df['instance_name'] + '.mps.gz'
    X_test, y_test, info_test = df_preprocess('setcover_fixed_5fold/test.csv', ref_df, config_cols)
    X_train, y_train, info_train = df_preprocess('setcover_fixed_5fold/fold_1_train.csv', ref_df, config_cols)
    X_valid, y_valid, info_valid = df_preprocess('setcover_fixed_5fold/fold_1_val.csv', ref_df, config_cols)
    
    y_test_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    y_valid_pred = model.predict(X_valid)
    
    print(f'Test MSE: {mean_squared_error(y_test, y_test_pred)}')
    print(f'Train MSE: {mean_squared_error(y_train, y_train_pred)}')
    print(f'Validation MSE: {mean_squared_error(y_valid, y_valid_pred)}')
    
    # unmelt the data
    df_test = unmelt_data(pd.concat([*info_test, X_test, y_test], axis=1))
    df_test_pred = get_best_config(unmelt_data(pd.concat([*info_test, X_test, pd.DataFrame(y_test_pred, columns=["time"])], axis=1)))
    test_improve_ratio = cal_improvement(df_test, df_test_pred)
    
    df_train = unmelt_data(pd.concat([*info_train, X_train, y_train], axis=1))
    df_test_pred = get_best_config(unmelt_data(pd.concat([*info_train, X_train, pd.DataFrame(y_train_pred, columns=["time"])], axis=1)))
    train_improve_ratio = cal_improvement(df_train, df_test_pred)
    
    df_valid = unmelt_data(pd.concat([*info_valid, X_valid, y_valid], axis=1))
    df_test_pred = get_best_config(unmelt_data(pd.concat([*info_valid, X_valid, pd.DataFrame(y_valid_pred, columns=["time"])], axis=1)))
    valid_improve_ratio = cal_improvement(df_valid, df_test_pred)
    
    print(f'Test improvement ratio: {test_improve_ratio}')
    print(f'Train improvement ratio: {train_improve_ratio}')
    print(f'Validation improvement ratio: {valid_improve_ratio}')
