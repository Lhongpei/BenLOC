from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
import pickle
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import logging
import os
from sklearn.metrics import mean_squared_error, accuracy_score
def select_common_rows(df1, df2):
    common_file_names = set(df1["File Name"]).intersection(set(df2["File Name"]))

    df1 = df1[df1["File Name"].isin(common_file_names)].reset_index(drop=True)
    df2 = df2[df2["File Name"].isin(common_file_names)].reset_index(drop=True)
    return df1, df2


def preprocess_data(data):
    column = [col for col in data.columns if "feat" not in col and "Name" not in col]

    data["min_time"] = data[column].min(axis=1)

    id_var = [col for col in data.columns if col not in column]

    data = data.melt(
        id_vars=id_var, value_vars=column, var_name="configuration", value_name="time"
    )

    config = pd.DataFrame(
        0, index=data.index, columns=[f"feat_{col}" for col in column]
    )

    for index, row in data.iterrows():
        col = row["configuration"]
        config.at[index, f"feat_{col}"] = 1

    data = pd.concat([data, config], axis=1)

    return data


def baseline(df:pd.DataFrame, default, col=None):
    if col == None:
        data_backup = df.copy()
    data = df
    default_time = shifted_geometric_mean(data[default], 10)
    column = [
        coln for coln in data.columns if "feat" not in coln and "Name" not in coln
    ]
    data["min_time"] = data[column].min(axis=1)
    oracle = shifted_geometric_mean(data["min_time"], 10)
    if col == None:
        default_time = shifted_geometric_mean(data[default], 10)
        data = data_backup

        min_stf = 100000000000
        best_stf = 0
        best_col = None
        for col in column:
            sft_time = shifted_geometric_mean(data[col], 10)

            if sft_time < min_stf:
                best_stf = sft_time
                best_col = col
                min_stf = sft_time

        return best_col, best_stf, default_time, oracle
    else:
        return shifted_geometric_mean(data[col], 10), default_time, oracle


def process(data):
    columns = data.columns.tolist()
    data.columns = columns
    time_col = ["File Name", "time"]
    feature_col = ["File Name"] + [col for col in data.columns if "feat" in col]
    time = data[time_col]
    feature = data[feature_col]
    return feature, time





def replace_inf_with_max(arr):
    arr[arr == -np.inf] = -1000
    arr[arr == np.inf] = 3
    return arr


def get_X(feature):
    data = feature.drop(["File Name"], axis=1)
    X = np.array(data)
    X = replace_inf_with_max(X)
    return X


def get_log_scale_y(data):
    y = np.log2((data["default"] + 1) / (data["time"] + 1))
    y = np.array(y)
    return y


def get_origin_y(data):
    y = np.array(data["time"])
    return y


def rfr_solver_obj(params, X, y):
    length = params.shape[0]
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    mse_scores = []
    for i in range(length):
        mse_fold = []
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            rfr = RandomForestRegressor(
                n_estimators=int(params.iloc[i, 0]),
                max_depth=int(params.iloc[i, 1]),
                min_samples_leaf=int(params.iloc[i, 2]),
                min_samples_split=int(params.iloc[i, 3]),
                max_features=params.iloc[i, 4],
                random_state=42,
                n_jobs=32,
            )
            rfr.fit(X_train, y_train)
            y_pred = rfr.predict(X_val)
            mse = mean_squared_error(y_val, y_pred)
            mse_fold.append(mse)

        avg_mse = np.mean(mse_fold)
        mse_scores.append(avg_mse)
    return np.array(mse_scores)


def setup_logger(index, folder="time_label/log"):
    if not os.path.exists(folder):
        os.makedirs(folder)

    log_file_path = os.path.join(folder, f"progress_{index}.log")

    logger = logging.getLogger(f"Process_{index}")
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    if os.path.exists(log_file_path):
        os.remove(log_file_path)

    file_handler = logging.FileHandler(log_file_path)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def run_cross_validation(X_train, y_train, X_test, y_test, index, space):
    logger = setup_logger(index)
    try:
        logger.info(f"Process_{index} start optimization")
        opt = HEBO(space)
        for i in range(400):
            logger.info(f"Process_{index} iteration {i}")
            rec = opt.suggest(n_suggestions=1)
            mse_scores = rfr_solver_obj(rec, X_train, y_train)
            opt.observe(rec, mse_scores)

        best_params = pd.DataFrame(opt.X.iloc[[opt.y.argmin()]])

        rfr = RandomForestRegressor(
            n_estimators=int(best_params.iloc[0, 0]),
            max_depth=int(best_params.iloc[0, 1]),
            min_samples_leaf=int(best_params.iloc[0, 2]),
            min_samples_split=int(best_params.iloc[0, 3]),
            max_features=best_params.iloc[0, 4],
            random_state=42,
            # n_jobs=32,
        )
        logger.info(f"Process_{index} start training")
        rfr.fit(X_train, y_train)
        y_predict_test = rfr.predict(X_test)
        y_predict_train = rfr.predict(X_train)
        logger.info(f"Process_{index} finished")
        return y_predict_train, y_predict_test, rfr, best_params

    except Exception as e:
        logger.error(f"An error occurred in process {index}: {e}", exc_info=True)
        return None, None, None, None


def shifted_geometric_mean(metrics, shift):
    product_of_shifted_metrics = 1
    for i in metrics:
        product_of_shifted_metrics = product_of_shifted_metrics * np.power(
            i + shift, 1 / len(metrics)
        )
        geom_mean_shifted = product_of_shifted_metrics - shift
    return geom_mean_shifted


def time_train_f(time_train, y_predict, mode="shift"):
    rf_y_train_time = np.where(
        np.array(y_predict) > 0, time_train["config_0"], time_train["config_1"]
    )
    rf_time = (
        shifted_geometric_mean(rf_y_train_time, 10)
        if mode == "shift"
        else rf_y_train_time.sum()
    )
    time_0 = (
        shifted_geometric_mean(time_train["config_0"], 10)
        if mode == "shift"
        else time_train["config_0"].sum()
    )
    time_1 = (
        shifted_geometric_mean(time_train["config_1"], 10)
        if mode == "shift"
        else time_train["config_1"].sum()
    )
    time = time_1
    result = [
        np.min([time_train["config_0"].iloc[i], time_train["config_1"].iloc[i]])
        for i in range(len(y_predict))
    ]
    Oracle = shifted_geometric_mean(result, 10)
    para = 1
    if time_0 <= time_1:
        time = time_0
        para = 0
    Imp_stime = (time - rf_time) / time
    Imp_ub = (time - Oracle) / time
    return rf_time, time_1, time_0, Imp_stime, Oracle, Imp_ub, para


def time_test_f(time_test, y_predict, para, mode="shift"):
    rf_y_train_time = np.where(
        np.array(y_predict) > 0, time_test["config_0"], time_test["config_1"]
    )
    rf_time = (
        shifted_geometric_mean(rf_y_train_time, 10)
        if mode == "shift"
        else rf_y_train_time.sum()
    )
    time_0 = (
        shifted_geometric_mean(time_test["config_0"], 10)
        if mode == "shift"
        else time_test["config_0"].sum()
    )
    time_1 = (
        shifted_geometric_mean(time_test["config_1"], 10)
        if mode == "shift"
        else time_test["config_0"].sum()
    )
    time = (
        shifted_geometric_mean(time_test[f"config_{para}"], 10)
        if mode == "shift"
        else time_test[f"config_{para}"].sum()
    )
    Imp_stime = (time - rf_time) / time

    result = [
        np.min([time_test["config_0"].iloc[i], time_test["config_1"].iloc[i]])
        for i in range(len(y_predict))
    ]
    Oracle = shifted_geometric_mean(result, 10)
    Imp_ub = (time - Oracle) / time
    return rf_time, time_1, time_0, Imp_stime, Oracle, Imp_ub


def accuracy_f(time, y_predict):
    y_label = [0 if a < b else 1 for a, b in zip(time["config_0"], time["config_1"])]
    y_label = np.array(y_label)
    rf_y_label = np.where(np.array(y_predict) > 0, 0, 1)
    pertrue0 = list(y_label).count(0) / len(y_label)
    pertrue1 = list(y_label).count(1) / len(y_label)
    per0 = list(rf_y_label).count(0) / len(rf_y_label)
    per1 = list(rf_y_label).count(1) / len(rf_y_label)
    accuracy = accuracy_score(y_label, rf_y_label)
    return pertrue0, pertrue1, per0, per1, accuracy


def get_result(oracle, default, local_min, predict, baseline_col):
    baseline_name = baseline_col
    default_time = default
    baseline = local_min
    ipv_default = (default - predict) / default
    ipv_baseline = (baseline - predict) / baseline
    ipv_oracle = (default - oracle) / default
    return [
        baseline_name,
        default_time,
        baseline,
        predict,
        ipv_default,
        ipv_baseline,
        oracle,
        ipv_oracle,
    ]

def combine(name):
    df = pd.read_csv(f"./data/parsed_log_{name}_new.csv")
    df_cidp = pd.read_csv(f"./data/parsed_log_cidp_{name}.csv")
    
    df["pre_row"] = np.log(df["rows"].astype(float))
    df["pre_columns"] = np.log(df["columns"].astype(float))
    df["preinteger"] = df["integers"].astype(float) / df["columns"].astype(float)
    merged_df = df.drop(["rows", "columns", "integers"], axis=1)
    merged_df = pd.merge(
        merged_df, df_cidp, on=["Log Name", "File Name", "c_i"], how="inner"
    )

    dynamic["DualInitialGap"] = dynamic.apply(
        lambda row: (
            0
            if (abs(row["c_i"]) == 0 or abs(row["BestBound"]) == 0)
            else abs(row["c_i"] - row["BestBound"])
            / max(
                abs(row["c_i"]),
                abs(row["BestBound"]),
                abs(row["c_i"] - row["BestBound"]),
            )
        ),
        axis=1,
    )
    dynamic["PrimalDualGap"] = dynamic.apply(
        lambda row: (
            0
            if (abs(row["BestSolution"]) == 0 or abs(row["BestBound"]) == 0)
            else abs(row["BestSolution"] - row["BestBound"])
            / max(
                abs(row["BestSolution"]),
                abs(row["BestBound"] - row["BestSolution"]),
                abs(row["BestBound"]),
            )
        ),
        axis=1,
    )
    dynamic["PrimalInitialGap"] = dynamic.apply(
        lambda row: (
            0
            if (abs(row["c_i"]) == 0 or abs(row["BestSolution"]) == 0)
            else abs(row["c_i"] - row["BestSolution"])
            / max(
                abs(row["c_i"]),
                abs(row["BestSolution"]),
                abs(row["BestSolution"] - row["c_i"]),
            )
        ),
        axis=1,
    )

    def calculate_gap_closed(row):
        if row["PrimalDualGap"] == 0 and row["DualInitialGap"] == 0:
            return 0
        elif row["DualInitialGap"] == 0:
            return float("-inf")
        else:
            return 1 - row["PrimalDualGap"] / row["DualInitialGap"]

    dynamic["GapClosed"] = dynamic.apply(calculate_gap_closed, axis=1)
    dynamic = dynamic.drop(["c_i", "BestBound", "BestSolution", "c_d", "c_p"], axis=1)
    print(dynamic.shape)

    feature_ori = pd.read_csv(f"./data/features_{name}_ori.csv")
    feature_ori = feature_ori.rename(columns={"File Name": "Name"})
    if "nn" in name:
        feature_ori["File Name"] = (
            feature_ori["Name"].astype(str).apply(lambda x: x[:-3])
        )
    else:
        feature_ori["File Name"] = (
            feature_ori["Name"].astype(str).apply(lambda x: x[:-7])
        )
    feature_ori = feature_ori.drop(
        ["Name", "RHS_dynamic", "obj_dynamic", "Coe_dynamic"], axis=1
    )
    merge = pd.merge(dynamic, feature_ori, on="File Name", how="left")

    # merge = merge.replace([np.nan], 0)
    # merge['RHS_dynamic']= merge['RHS_dynamic'].replace([np.inf], 0)
    new_column_names = [
        "feat_" + col if i >= 2 else col for i, col in enumerate(merge.columns)
    ]
    merge.columns = new_column_names

    # symmetry = pd.DataFrame()
    # df_list = []
    # for file in log_files_def:
    #     tmp_df = parse_log_symmetry(file)
    #     df_list.append(tmp_df)
    # symmetry = pd.concat(df_list, ignore_index=True, axis=0)
    # symmetry=symmetry.fillna(0)

    # final_df = pd.merge(merge, symmetry, on="Log Name", how="left")
    final_df = merge
    print(final_df.shape)
    final_df.to_csv(f"./data/feat_{name}.csv", index=False)