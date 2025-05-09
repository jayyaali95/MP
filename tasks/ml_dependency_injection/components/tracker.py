from abc import ABC, abstractmethod
from typing import Dict, Any, Union, List
import os
from core.factory import Factory


@Factory.register_component_type
class Tracker(ABC):
    """Base class for experiment trackers."""
    
    @abstractmethod
    def log_metric(self, name: str, value: Union[float, int]) -> None:
        """Log a numerical metric."""
        pass
    
    @abstractmethod
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters."""
        pass
    
    @classmethod
    def create(cls, config: Union[str, Dict[str, Any]]):
        """Factory method to create a Tracker instance."""
        if isinstance(config, str):
            return Factory.create_from_config('tracker', config_path=config)
        else:
            return Factory.create_from_config('tracker', config_dict=config)


class StdoutTracker(Tracker):
    """Simple tracker that prints to stdout."""
    
    def __init__(self, **kwargs):
        pass
    
    def log_metric(self, name: str, value: Union[float, int]) -> None:
        print(f"METRIC: {name} = {value}")
    
    def log_params(self, params: Dict[str, Any]) -> None:
        print("PARAMS:")
        for key, value in params.items():
            print(f"  {key} = {value}")


class MLFlowTracker(Tracker):
    """Tracker that logs to MLFlow."""
    
    def __init__(self, experiment_name: str = "default", **kwargs):
        self.experiment_name = experiment_name
        print(f"Initializing MLFlow with experiment: {experiment_name}")
    
    def log_metric(self, name: str, value: Union[float, int]) -> None:
        print(f"MLFLOW METRIC: {name} = {value}")
    
    def log_params(self, params: Dict[str, Any]) -> None:
        print("MLFLOW PARAMS:")
        for key, value in params.items():
            print(f"  {key} = {value}")


class LogfileTracker(Tracker):
    """Tracker that logs to a file."""
    
    def __init__(self, logfile: str = "experiment.log", **kwargs):
        self.logfile = logfile
        with open(self.logfile, 'w') as f:
            f.write("=== Experiment Log ===\n")
    
    def log_metric(self, name: str, value: Union[float, int]) -> None:
        with open(self.logfile, 'a') as f:
            f.write(f"METRIC: {name} = {value}\n")
    
    def log_params(self, params: Dict[str, Any]) -> None:
        with open(self.logfile, 'a') as f:
            f.write("PARAMS:\n")
            for key, value in params.items():
                f.write(f"  {key} = {value}\n") 