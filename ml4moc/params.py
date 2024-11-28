class Params:
    def __init__(self, **kwargs):
        self.shift_scale = 10
        self.label_type = 'log_scaled'
        self.default = "MipLogLevel-2"
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