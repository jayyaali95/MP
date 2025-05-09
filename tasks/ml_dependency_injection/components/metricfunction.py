from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Union, List
from core.factory import Factory


@Factory.register_component_type
class MetricFunction(ABC):
    """Base class for metric functions."""
    
    @abstractmethod
    def calculate(self, y_true: Union[np.ndarray, List], y_pred: Union[np.ndarray, List]) -> float:
        """Calculate the metric between true values and predictions."""
        pass
    
    @classmethod
    def create(cls, config: Union[str, Dict[str, Any]]):
        """Factory method to create a MetricFunction instance."""
        if isinstance(config, str):
            return Factory.create_from_config('metricfunction', config_path=config)
        else:
            return Factory.create_from_config('metricfunction', config_dict=config)


class MAE(MetricFunction):
    """Mean Absolute Error metric."""
    
    def __init__(self, **kwargs):
        pass
    
    def calculate(self, y_true: Union[np.ndarray, List], y_pred: Union[np.ndarray, List]) -> float:
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return np.mean(np.abs(y_true - y_pred))


class MSE(MetricFunction):
    """Mean Squared Error metric."""
    
    def __init__(self, **kwargs):
        pass
    
    def calculate(self, y_true: Union[np.ndarray, List], y_pred: Union[np.ndarray, List]) -> float:
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return np.mean(np.square(y_true - y_pred)) 