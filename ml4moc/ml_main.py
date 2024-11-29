import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import logging
import sklearn.model_selection
from ml4moc.ML.utils import (
    setup_logger,
    process,
    preprocess_data,
    shifted_geometric_mean,
    get_X,
    get_log_scale_y,
    get_origin_y,
    select_common_rows,
    get_result,
)
from ml4moc.params import Params

class ML4MOC:
    def __init__(self, params = Params()):
        self.trainner = None
        self.imple_dataset = [
            "setcover",
            "indset",
            "nn_verification",
            "load_balance",
            "miplib",
        ]
        self.label, self.feat = None, None
        self.test_label, self.test_feat = None, None
        self.label_processed, self.feat_processed = None, None
        self.test_label_processed, self.test_feat_processed = None, None
        self.label_type = params.label_type
        self.default = params.default
        self.shift_scale = params.shift_scale
        self.train_test_split_flag = False
        self.best_col = None
        self.has_processed = False

    def set_label_type(self, label_type: str):
        assert label_type in [
            "log_scaled",
            "original",
        ], "label_type should be one of the following: ['log_scaled', 'original']"
        self.label_type = label_type

    def input_features(self, feat: pd.DataFrame):
        # Check if the input is as expected
        self.feat = feat

    def input_labels(self, label: pd.DataFrame):
        # Check if the input is as expected
        self.label = label

    def set_train_data(self, feat: pd.DataFrame, label: pd.DataFrame):
        self.has_processed = False
        self.feat, self.label = select_common_rows(feat, label)

    def set_test_data(self, test_feat: pd.DataFrame, test_label: pd.DataFrame):
        self.has_processed = False
        self.test_feat, self.test_label = select_common_rows(test_feat, test_label)
        self.train_test_split_flag = True

    def process(self):
        self.feat_processed, self.label_processed = self._preprocess(
            self.feat, self.label, mode="train"
        )
        if self.train_test_split_flag:
            self.test_feat_processed, self.test_label_processed = self._preprocess(
                self.test_feat, self.test_label, mode="test"
            )
        self.has_processed = True

    def _preprocess(self, feat, label, mode="train"):
        if mode == "train":
            best_col, best_stf, default_time, oracle = self.baseline(
                label, self.default
            )
            self.best_col = best_col
            self.train_best_stf = best_stf
            self.train_default_time = default_time
            self.train_oracle = oracle
        else:
            best_stf, default_time, oracle = self.baseline(
                label, self.default, self.best_col
            )
            self.test_best_stf = best_stf
            self.test_default_time = default_time
            self.test_oracle = oracle

        train_data = pd.merge(feat, label, on="File Name", how="inner")
        train_data = preprocess_data(train_data)
        feat, label = process(train_data)
        return feat, label

    def train_test_split(self, test_size=0.2):
        feat_train, feat_test, label_train, label_test = train_test_split(
            self.feat, self.label, test_size=test_size, random_state=42, shuffle=True
        )
        self.train_test_split_flag = True
        self.set_train_data(feat_train, label_train)
        self.set_test_data(feat_test, label_test)

    def load_features(self, dataset: str, processed: bool = False):
        assert (
            dataset in self.imple_dataset
        ), f"dataset should be one of the following: {self.imple_dataset}"
        self.feat = (
            pd.read_csv(f"ml4moc/data/feat/feat_{dataset}.csv")
            if not processed
            else pd.read_csv(f"ml4moc/data/feat/feat_{dataset}_processed.csv")
        )

    def load_labels(self, dataset: str):
        assert (
            dataset in self.imple_dataset
        ), f"dataset should be one of the following: {self.imple_dataset}"
        self.label = pd.read_csv(f"ml4moc/data/time/time_{dataset}.csv")

    def load_dataset(self, dataset: str, processed: bool = True, verbose: bool = False):
        self.load_features(dataset, processed)
        self.load_labels(dataset)
        self.feat, self.label = select_common_rows(self.feat, self.label)
        if verbose:
            print("Finished loading dataset")
            print(f"Features info: {self.feat.info()}")
            print(f"Labels info: {self.label.info()}")
        if verbose:
            print("Finished preprocessing dataset")

    def obtain_dataset(self, dataset: str, verbose: bool = True):
        if self.feat is None or self.label is None:
            self.load_dataset(dataset, verbose)
        return self.feat, self.label

    def set_trainner(self, model: sklearn.base.BaseEstimator):
        self.trainner = model

    def fit(self):
        self.trainner.fit(self.get_X, self.get_Y)

    @property
    def rfr_parameter_space(self):
        return {
            "n_estimators": [300, 500],
            "max_depth": [3, 10],
            "min_samples_leaf": [3, 20],
            "min_samples_split": [4, 20],
            "max_features": ["sqrt", "log2", None],
        }

    def rfr_grid_search(self, cv=3):
        return GridSearchCV(
            estimator=self.trainner,
            param_grid=self.rfr_grid_search_space,
            scoring="neg_mean_squared_error",
            cv=cv,
            verbose=1,
            n_jobs=-1,
        )

    def cross_validation(
        self,
        CVer: sklearn.base.BaseEstimator,
        space=None,
        log_name="cross_validation",
    ):
        """
        Perform cross-validation using sklearn's GridSearchCV.
        """
        logger = setup_logger(log_name)

        try:
            if space is None and isinstance(self.trainner, RandomForestRegressor):
                space = {
                    "n_estimators": [300, 500],
                    "max_depth": [3, 10],
                    "min_samples_leaf": [3, 20],
                    "min_samples_split": [4, 20],
                    "max_features": ["sqrt", "log2", None],
                }
            # Define parameter grid based on the provided space

            logger.info(f"Start Model Selection")

            X_train, y_train = self.get_X, self.get_Y

            # Fit GridSearchCV on training data
            CVer.fit(X_train, y_train)

            # Get the best model and parameters
            best_params = CVer.best_params_
            best_model = CVer.best_estimator_

            logger.info(f"Best parameters: {best_params}")

            # Train the best model on the full training set
            logger.info(f"Start training")
            best_model.fit(X_train, y_train)

            # Make predictions
            y_predict_train = best_model.predict(X_train)

            # Log performance
            train_mse = mean_squared_error(y_train, y_predict_train)
            logger.info(f"Train MSE: {train_mse:.4f}")
            return {
                "best_model": best_model,
                "best_params": best_params,
                "train_mse": train_mse,
            }

        except Exception as e:
            logger.error(f"An error occurred in process: {e}", exc_info=True)
            return None

    def shifted_geometric_mean(self, data):
        return shifted_geometric_mean(data, self.shift_scale)

    def baseline(self, df: pd.DataFrame, default, col=None):
        if col == None:
            data_backup = df.copy()
        data = df

        default_time = self.shifted_geometric_mean(data[default])
        column = [
            coln for coln in data.columns if "feat" not in coln and "Name" not in coln
        ]
        data["min_time"] = data[column].min(axis=1)
        oracle = self.shifted_geometric_mean(data["min_time"])
        if col == None:
            default_time = self.shifted_geometric_mean(data[default])
            data = data_backup

            min_stf = 100000000000
            best_stf = 0
            best_col = None
            for col in column:
                sft_time = self.shifted_geometric_mean(data[col])

                if sft_time < min_stf:
                    best_stf = sft_time
                    best_col = col
                    min_stf = sft_time

            return best_col, best_stf, default_time, oracle
        else:
            return self.shifted_geometric_mean(data[col]), default_time, oracle

    @property
    def get_processed_X(self):
        if not self.has_processed:
            print("Processing data")
            self.process()
        return self.feat_processed

    @property
    def get_processed_Y(self):
        if not self.has_processed:
            print("Processing data")
            self.process()
        return self.label_processed

    @property
    def get_X(self):
        return get_X(self.get_processed_X)

    @property
    def get_Y(self):
        if self.label_type == "log_scaled":
            return get_log_scale_y(self.get_processed_Y)
        else:
            return get_origin_y(self.get_processed_Y)

    @property
    def get_test_X(self):
        return get_X(self.test_feat_processed)

    @property
    def get_test_Y(self):
        if self.label_type == "log_scaled":
            return get_log_scale_y(self.test_label_processed)
        else:
            return get_origin_y(self.test_label_processed)

    def evaluate(self, test_feat=None, test_label=None):
        if test_feat is None or test_label is None:
            assert (
                self.train_test_split_flag
            ), "Haven't set test data, please set test data or use `train_test_split` method to split data."
        else:
            self.set_test_data(test_feat, test_label)
        predict_train = self.trainner.predict(self.get_X)
        predict_test = self.trainner.predict(self.get_test_X)
        result_feat_label = self.label_processed
        result_test_feat_label = self.test_label_processed
        result_feat_label["predict_y"] = predict_train
        result_test_feat_label["predict_y"] = predict_test
        min_idx_train = result_feat_label.groupby("File Name")["predict_y"].idxmin()
        result_train = result_feat_label.loc[min_idx_train].reset_index(drop=True)
        min_idx_test = result_test_feat_label.groupby("File Name")["predict_y"].idxmin()
        result_test = result_test_feat_label.loc[min_idx_test].reset_index(drop=True)
        row_train = get_result(
            self.train_oracle,
            self.train_default_time,
            self.train_best_stf,
            self.shifted_geometric_mean(result_train["time"]),
            self.best_col,
        )
        row_test = get_result(
            self.test_oracle,
            self.test_default_time,
            self.test_best_stf,
            self.shifted_geometric_mean(result_test["time"]),
            self.best_col,
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
        return df
