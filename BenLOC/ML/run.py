from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO
import pandas as pd
import pickle
import pickle
import pandas as pd
import argparse
import os

from ML.utils import (
    preprocess,
    baseline,
    shifted_geometric_mean,
    process,
    run_cross_validation,
    get_result,
)


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label_type",
        type=str,
        default="log_scaled",
        choices=["log_scaled", "original"],
        help="label type, log_scaled or original",
    )
    parser.add_argument(
        "--dataset", type=str, help="The dataset need to be processed", required=True
    )
    parser.add_argument("--fold", type=int, default=5, help="The number of fold")
    parser.add_argument(
        "--report_root_path",
        type=str,
        required=True,
        help="The root path of the report",
    )
    parser.add_argument(
        "--result_root_path",
        type=str,
        required=True,
        help="The root path of the result",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = set_args()
    label_type = args.label_type
    dataset = args.dataset
    report_root_path_dataset = args.report_root_path
    result_root_path = args.result_root_path

    space = DesignSpace().parse(
        [
            {"name": "n_estimators", "type": "int", "lb": 300, "ub": 500},
            {"name": "max_depth", "type": "int", "lb": 3, "ub": 10},
            {"name": "min_samples_leaf", "type": "int", "lb": 3, "ub": 20},
            {"name": "min_samples_split", "type": "int", "lb": 4, "ub": 20},
            {
                "name": "max_features",
                "type": "cat",
                "categories": ["sqrt", "log2", None],
            },
        ]
    )

    default = "default"
    index = dataset
    ind_range = range(1, args.fold + 1)

    if not os.path.exists(os.path.join(result_root_path, "stat_result")):
        os.makedirs(os.path.join(result_root_path, "stat_result"))
    if not os.path.exists(os.path.join(result_root_path, "pred_result")):
        os.makedirs(os.path.join(result_root_path, "pred_result"))
    if not os.path.exists(os.path.join(result_root_path, "model_result")):
        os.makedirs(os.path.join(result_root_path, "model_result"))
    predict_folder = os.path.join(result_root_path, "pred_result")
    stat_folder = os.path.join(result_root_path, "stat_result")
    model_folder = os.path.join(result_root_path, "model_result")
    for i in ind_range:
        train_file = os.path.join(report_root_path_dataset, f"fold_{i}_train.csv")
        test_file = os.path.join(report_root_path_dataset, f"fold_{i}_test.csv")
        data_train = preprocess(train_file)
        data_test = preprocess(test_file)
        feature_train, time_train, X_train, y_train = process(data_train, label_type)
        feature_test, time_test, X_test, y_test = process(data_test, label_type)
        results = run_cross_validation(
            X_train, y_train, X_test, y_test, f"{index}_fold{i}", space
        )

        with open(os.join(model_folder, f"{index}_fold{i}.pkl"), "wb") as f:
            pickle.dump(results, f)

        y_predict_train, y_predict_test, rfr, best_params = results

        if y_predict_train is None:
            continue
        data_train["predict_y"] = y_predict_train
        data_test["predict_y"] = y_predict_test
        min_idx_train = data_train.groupby("File Name")["predict_y"].idxmin()

        result_train = data_train.loc[min_idx_train].reset_index(drop=True)
        min_idx_test = data_test.groupby("File Name")["predict_y"].idxmin()
        result_test = data_test.loc[min_idx_test].reset_index(drop=True)
        best_col, best_stf, default_time, oracle = baseline(train_file, default)
        row_train = get_result(
            oracle,
            default_time,
            best_stf,
            shifted_geometric_mean(result_train["time"], 10),
            best_col,
        )
        best_stf, default_time, oracle = baseline(test_file, default, best_col)
        row_test = get_result(
            oracle,
            default_time,
            best_stf,
            shifted_geometric_mean(result_test["time"], 10),
            best_col,
        )
        col = [
            "baseline_name",
            "default_time",
            "baseline_time",
            "rf_time",
            "ipv_default",
            "ipv_baseline",
            "oracle",
            "ipv_oracle",
        ]
        df = pd.DataFrame(columns=col)
        df.loc["train"] = row_train
        df.loc["test"] = row_test

        df.to_csv(os.path.join(stat_folder, f"{index}_fold{i}.csv"))

        data_test.to_csv(os.path.join(predict_folder, f"{index}_fold{i}_test.csv"))

        data_train.to_csv(os.path.join(predict_folder, f"{index}_fold{i}_train.csv"))
