from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Callable, Union
from core.factory import Factory
from components.model import Model


@Factory.register_component_type
class Optimizer(ABC):
    """Base class for optimizers."""
    
    def __init__(self, model: Model, **kwargs):
        self.model = model
    
    @abstractmethod
    def step(self, gradients: List[np.ndarray]) -> None:
        """Apply gradients and update model parameters."""
        pass
    
    @classmethod
    def create(cls, config: Union[str, Dict[str, Any]]):
        """Factory method to create an Optimizer instance."""
        if isinstance(config, str):
            return Factory.create_from_config('optimizer', config_path=config)
        else:
            return Factory.create_from_config('optimizer', config_dict=config)


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer."""
    
    def __init__(self, model: Model, learning_rate: float = 0.01, **kwargs):
        super().__init__(model)
        self.learning_rate = learning_rate
    
    def step(self, gradients: List[np.ndarray]) -> None:
        params = self.model.get_params()
        
        for i, (param, grad) in enumerate(zip(params, gradients)):
            params[i] = param - self.learning_rate * grad
        
        self.model.set_params(params)


class Adam(Optimizer):
    """Adam optimizer."""
    
    def __init__(self, model: Model, learning_rate: float = 0.001, 
                 beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8, **kwargs):
        super().__init__(model)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        self.m = None
        self.v = None
        self.t = 0
    
    def step(self, gradients: List[np.ndarray]) -> None:
        params = self.model.get_params()
        
        # Initialize moment estimates if this is the first step
        if self.m is None:
            self.m = [np.zeros_like(grad) for grad in gradients]
            self.v = [np.zeros_like(grad) for grad in gradients]
        
        self.t += 1
        
        # Update parameters
        for i, (param, grad) in enumerate(zip(params, gradients)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * np.square(grad)
            
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            params[i] = param - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        self.model.set_params(params) 