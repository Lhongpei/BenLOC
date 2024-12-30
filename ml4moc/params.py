class Params:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def set_params(self, **kwargs):
        self.__dict__.update(kwargs)
    
    def __repr__(self):
        return str(self.__dict__)

    def __str__(self):
        return str(self.__dict__)
    
    def __getitem__(self, key):
        return self.__dict__[key]
    
    def __setattr__(self, key, value):
        self.__dict__[key] = value

    def __getattr__(self, key):
        try:
            return self.__dict__[key]
        except KeyError:
            raise AttributeError(f"'Params' object has no attribute '{key}'")
        
    def return_dict(self):
        return self.__dict__
        
class GraphParams(Params):
    def __init__(self, **kwargs):
        self.label_type: str = "log_scaled", 
        self.default: int = 0, 
        self.shift_scale: float = 1.0, 
        self.train_test_split_ratio: float = 0.8, 
        self.random_seed: int = 42,
        self.hyperparams: dict = None
        super().__init__(**kwargs)
        
class TabParams(Params):
    def __init__(self, **kwargs):
        self.shift_scale = 10
        self.label_type = 'log_scaled'
        self.default = "MipLogLevel-2"
        super().__init__(**kwargs)
        
