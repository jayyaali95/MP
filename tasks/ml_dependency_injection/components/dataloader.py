from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any, Union
from core.factory import Factory


@Factory.register_component_type
class DataLoader(ABC):
    """Base class for data loaders."""
    
    @abstractmethod
    def load_data(self):
        pass
    
    @classmethod
    def create(cls, config: Union[str, Dict[str, Any]]):
        if isinstance(config, str):
            return Factory.create_from_config('dataloader', config_path=config)
        else:
            return Factory.create_from_config('dataloader', config_dict=config)


class ParquetFileDataLoader(DataLoader):
    """DataLoader for Parquet files."""
    
    def __init__(self, filename: str, **kwargs):
        self.filename = filename
    
    def load_data(self):
        return pd.read_parquet(self.filename)


class CSVDataLoader(DataLoader):
    """DataLoader for CSV files."""
    
    def __init__(self, filename: str, **kwargs):
        self.filename = filename
    
    def load_data(self):
        return pd.read_csv(self.filename) 