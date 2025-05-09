from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Dict, Any, Union, Tuple
from core.factory import Factory


@Factory.register_component_type
class Preprocessor(ABC):
    """Base class for data preprocessors."""
    
    @abstractmethod
    def fit(self, data: Union[np.ndarray, pd.DataFrame]) -> None:
        """Fit the preprocessor to the data."""
        pass
    
    @abstractmethod
    def transform(self, data: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """Transform the data."""
        pass
    
    def fit_transform(self, data: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """Fit and transform the data."""
        self.fit(data)
        return self.transform(data)
    
    @classmethod
    def create(cls, config: Union[str, Dict[str, Any]]):
        """Factory method to create a Preprocessor instance."""
        if isinstance(config, str):
            return Factory.create_from_config('preprocessor', config_path=config)
        else:
            return Factory.create_from_config('preprocessor', config_dict=config)


class MinMaxNormalizer(Preprocessor):
    """Normalize features to [0, 1] range."""
    
    def __init__(self, **kwargs):
        self.min_vals = None
        self.max_vals = None
    
    def fit(self, data: Union[np.ndarray, pd.DataFrame]) -> None:
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        self.min_vals = np.min(data, axis=0)
        self.max_vals = np.max(data, axis=0)
    
    def transform(self, data: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        is_df = isinstance(data, pd.DataFrame)
        if is_df:
            columns = data.columns
            index = data.index
            data = data.values
        
        # Avoid division by zero
        range_vals = self.max_vals - self.min_vals
        range_vals[range_vals == 0] = 1
        
        normalized = (data - self.min_vals) / range_vals
        
        if is_df:
            return pd.DataFrame(normalized, columns=columns, index=index)
        return normalized


class MeanVarNormalizer(Preprocessor):
    """Normalize features to mean=0 and variance=1."""
    
    def __init__(self, **kwargs):
        self.mean_vals = None
        self.std_vals = None
    
    def fit(self, data: Union[np.ndarray, pd.DataFrame]) -> None:
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        self.mean_vals = np.mean(data, axis=0)
        self.std_vals = np.std(data, axis=0)
    
    def transform(self, data: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        is_df = isinstance(data, pd.DataFrame)
        if is_df:
            columns = data.columns
            index = data.index
            data = data.values
        
        # Avoid division by zero
        std_vals = self.std_vals.copy()
        std_vals[std_vals == 0] = 1
        
        normalized = (data - self.mean_vals) / std_vals
        
        if is_df:
            return pd.DataFrame(normalized, columns=columns, index=index)
        return normalized 