import pandas as pd
import sklearn
class ML4MOC:
    def __init__(self):
        self.trainner = None
        self.imple_dataset = ['setcover', 'indset', 'nn_verification', 'load_balance', 'miplib']
        self.label, self.feat = None, None

    def load_features(self, dataset: str, processed: bool = False):
        assert dataset in self.imple_dataset, f"dataset should be one of the following: {self.imple_dataset}"
        self.feat = pd.read_csv(f"ml4moc/data/feat/feat_{dataset}_ori.csv") if not processed \
            else pd.read_csv(f"ml4moc/data/feat/feat_{dataset}.csv")
    
    def load_labels(self, dataset: str):
        assert dataset in self.imple_dataset, f"dataset should be one of the following: {self.imple_dataset}"
        self.label = pd.read_csv(f"ml4moc/data/time/time_{dataset}.csv")
    
    def load_dataset(self, dataset: str, verbose: bool = True):
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
    
    def set_trainner(self, model: sklearn.base.BaseEstimator, ):
        self.trainner = model

    
    
        
        