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
        self.valid_train_ratio = 0.2
        #About PyTorch Lightning
        self._init_pyl_params()
        
        super().__init__(**kwargs)
        
    def _init_pyl_params(self):
        # Training parameters
        self.batch_size = 64
        self.num_epochs = 50
        self.learning_rate = 2e-4
        self.weight_decay = 1e-4
        self.lr_scheduler = 'cosine-decay'

        # System settings
        self.num_workers = 16
        self.fp16 = False
        self.use_activation_checkpoint = False

        # Logging and checkpointing
        self.project_name = 'BenLOC_DEFAULT_NAME'
        self.wandb_entity = None
        self.wandb_logger_name = None
        self.resume_id = None
        self.ckpt_path = None
        self.resume_weight_only = False