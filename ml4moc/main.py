import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import logging
import sklearn.model_selection
from ml4moc.ML.utils import setup_logger, process, preprocess, shifted_geometric_mean   
class ML4MOC:
    def __init__(self):
        self.trainner = None
        self.imple_dataset = ['setcover', 'indset', 'nn_verification', 'load_balance', 'miplib']
        self.label, self.feat = None, None
        self.label_type = 'original'
        self.default = "MipLogLevel-2"
        self.shift_scale = 10

    def set_label_type(self, label_type: str):
        assert label_type in ['log_scaled', 'original'], "label_type should be one of the following: ['log_scaled', 'original']"
        self.label_type = label_type
    
    def input_features(self, feat: pd.DataFrame):
        # Check if the input is as expected
        self.feat = feat
        
    def input_labels(self, label: pd.DataFrame):
        # Check if the input is as expected
        self.label = label
    
    def load_features(self, dataset: str, processed: bool = False):
        assert dataset in self.imple_dataset, f"dataset should be one of the following: {self.imple_dataset}"
        self.feat = pd.read_csv(f"ml4moc/data/feat/feat_{dataset}.csv") if not processed \
            else pd.read_csv(f"ml4moc/data/feat/feat_{dataset}_processed.csv")
    
    def load_labels(self, dataset: str):
        assert dataset in self.imple_dataset, f"dataset should be one of the following: {self.imple_dataset}"
        self.label = pd.read_csv(f"ml4moc/data/time/time_{dataset}.csv")
    
    def load_dataset(self, dataset: str, verbose: bool = False):
        self.load_features(dataset)
        self.load_labels(dataset)
        if verbose:
            print("Finished loading dataset")
            print(f"Features info: {self.feat.info()}")
            print(f"Labels info: {self.label.info()}")
            
    def obtain_dataset(self, dataset: str, verbose: bool = True):
        if self.feat is None or self.label is None:
            self.load_dataset(dataset, verbose)
        return self.feat, self.label
    
    def set_trainner(self, model: sklearn.base.BaseEstimator):
        self.trainner = model
        
    def fit(self):
        self.trainner.fit(self.feat, self.label)
    
    @property   
    def rfr_parameter_space(self):
        return {
            "n_estimators": [300, 500],
            "max_depth": [3, 10],
            "min_samples_leaf": [3, 20],
            "min_samples_split": [4, 20],
            "max_features": ["sqrt", "log2", None]
        }
    
    def rfr_grid_search(self, cv = 3):
        return GridSearchCV(
            estimator=self.trainner,
            param_grid=self.rfr_grid_search_space,
            scoring="neg_mean_squared_error",
            cv=cv,
            verbose=1,
            n_jobs=-1
        )
    
    def cross_validation(self, CVer:sklearn.base.BaseEstimator, X_train, y_train,
                        space = None, log_name = "cross_validation", 
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
                    "max_features": ["sqrt", "log2", None]
                }
            # Define parameter grid based on the provided space

            
            logger.info(f"Start Model Selection")

            
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
                "train_mse": train_mse
            }

        except Exception as e:
            logger.error(f"An error occurred in process: {e}", exc_info=True)
            return None

    def shifted_geometric_mean(self, data):
        return shifted_geometric_mean(data, self.shift_scale)
    
    def baseline(self, df:pd.DataFrame, default, col=None):
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
                print(col, sft_time)
                if sft_time < min_stf:
                    best_stf = sft_time
                    best_col = col
                    min_stf = sft_time
                    print(min_stf, best_col)
            return best_col, best_stf, default_time, oracle
        else:
            return self.shifted_geometric_mean(data[col]), default_time, oracle
        
    def evaluate(self):
        pass
                