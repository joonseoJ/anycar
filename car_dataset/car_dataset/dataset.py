import numpy as np

class CarDataset:
    def __init__(self) -> None:
        self.reset_logs()  
        self.car_params = {"wheelbase": None, 
                           "mass": None, 
                           "com": None, 
                           "friction": None,  
                           "delay": 0, 
                           'max_throttle': None, 
                           'max_steer': None,
                           'steer_bias': None,
                           "sim": None,
                           "static_features": None}

    
    def reset_logs(self):
        self.data_logs = {
            "state": [],
            "action": [],
        } 
        
    def __len__(self):
        assert len(self.data_logs["state"]) == len(self.data_logs["action"])
        return len(self.data_logs["state"])