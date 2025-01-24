from abc import ABC, abstractmethod
from bikeshare.utils.config import Config

class BaseModel(ABC):
    """ Base class for all models """
    
    def __init__(self, cfg):
        self.config = Config.from_json(cfg)
        
    @abstractmethod
    def load_data(self):
        pass
    
    @abstractmethod
    def build(self):
        pass
    
    @abstractmethod
    def train(self):
        pass    
    
    @abstractmethod
    def evaluate(self):
        pass    
    
    @abstractmethod
    def evaluate_new_data(self):
        pass    