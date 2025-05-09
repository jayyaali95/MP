from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, List, Tuple, Union
from core.factory import Factory


@Factory.register_component_type
class Model(ABC):
    """Base class for ML models."""
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions for input data."""
        pass
    
    @abstractmethod
    def compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute loss for the given inputs and targets."""
        pass
    
    @abstractmethod
    def compute_gradients(self, X: np.ndarray, y: np.ndarray) -> List[np.ndarray]:
        """Compute gradients of the loss with respect to model parameters."""
        pass
    
    @abstractmethod
    def get_params(self) -> List[np.ndarray]:
        """Get model parameters."""
        pass
    
    @abstractmethod
    def set_params(self, params: List[np.ndarray]) -> None:
        """Set model parameters."""
        pass
    
    @classmethod
    def create(cls, config: Union[str, Dict[str, Any]]):
        """Factory method to create a Model instance."""
        if isinstance(config, str):
            return Factory.create_from_config('model', config_path=config)
        else:
            return Factory.create_from_config('model', config_dict=config)


class LinearModel(Model):
    """Simple linear regression model."""
    
    def __init__(self, input_dim: int = 1, **kwargs):
        self.input_dim = input_dim
        # Initialize weights and bias
        self.weights = np.random.randn(input_dim) * 0.01
        self.bias = np.zeros(1)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.weights) + self.bias
    
    def compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Mean squared error loss."""
        predictions = self.predict(X)
        return np.mean(np.square(predictions - y))
    
    def compute_gradients(self, X: np.ndarray, y: np.ndarray) -> List[np.ndarray]:
        """Compute gradients of MSE loss w.r.t. weights and bias."""
        predictions = self.predict(X)
        error = predictions - y
        
        # Gradient of loss w.r.t. weights
        dw = (2.0 / len(y)) * np.dot(X.T, error)
        
        # Gradient of loss w.r.t. bias
        db = (2.0 / len(y)) * np.sum(error)
        
        return [dw, np.array([db])]
    
    def get_params(self) -> List[np.ndarray]:
        return [self.weights, self.bias]
    
    def set_params(self, params: List[np.ndarray]) -> None:
        self.weights = params[0]
        self.bias = params[1] 