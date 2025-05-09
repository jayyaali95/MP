#!/usr/bin/env python3
"""
Test script to demonstrate the dependency injection system.
"""

import os
import yaml
from components.trainloop import TrainLoop
from components.dataloader import DataLoader
from components.metricfunction import MetricFunction
from components.model import Model
from components.optimizer import Optimizer
from components.preprocessor import Preprocessor
from components.tracker import Tracker


def test_with_yaml_file():
    """Test creating components from a YAML file."""
    print("\n=== Testing with YAML file ===")
    
    # Generate sample data
    from data.generate_sample_data import main as generate_data
    generate_data()
    
    # Create train loop from config file
    config_path = os.path.join(os.path.dirname(__file__), 'configs/sample_config.yaml')
    print(f"Using config from: {config_path}")
    
    train_loop = TrainLoop.create(config_path)
    print(f"Created TrainLoop: {train_loop.__class__.__name__}")
    print(f"DataLoader: {train_loop.dataloader.__class__.__name__}")
    print(f"Preprocessor: {train_loop.preprocessor.__class__.__name__}")
    print(f"Model: {train_loop.model.__class__.__name__}")
    print(f"Optimizer: {train_loop.optimizer.__class__.__name__}")
    print(f"Metric: {train_loop.metric_function.__class__.__name__}")
    print(f"Tracker: {train_loop.tracker.__class__.__name__}")
    
    train_loop.epochs = 2
    train_loop.execute()


def test_with_dict_config():
    """Test creating components from a dictionary config."""
    print("\n=== Testing with Dictionary Config ===")
    
    config = {
        'dataloader': {
            'class': 'CSVDataLoader',
            'filename': 'data/sample_data.csv'
        },
        'metricfunction': {
            'class': 'MAE'
        },
        'tracker': {
            'class': 'LogfileTracker',
            'logfile': 'test_run.log'
        },
        'preprocessor': {
            'class': 'MeanVarNormalizer'
        },
        'model': {
            'class': 'LinearModel',
            'input_dim': 4
        },
        'optimizer': {
            'class': 'SGD',
            'learning_rate': 0.05
        },
        'trainloop': {
            'class': 'OnlineLearningTrainLoop',
            'epochs': 1
        }
    }
    
    print("Using dictionary config with different component choices:")
    print(yaml.dump(config))
    
    train_loop = TrainLoop.create(config)
    print(f"Created TrainLoop: {train_loop.__class__.__name__}")
    print(f"DataLoader: {train_loop.dataloader.__class__.__name__}")
    print(f"Preprocessor: {train_loop.preprocessor.__class__.__name__}")
    print(f"Model: {train_loop.model.__class__.__name__}")
    print(f"Optimizer: {train_loop.optimizer.__class__.__name__}")
    print(f"Metric: {train_loop.metric_function.__class__.__name__}")
    print(f"Tracker: {train_loop.tracker.__class__.__name__}")
    
    # Execute training
    train_loop.execute()

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    print("=== ML Dependency Injection System Test ===")
    
    # Run tests
    test_with_yaml_file()
    test_with_dict_config()
    
    print("\nAll tests completed.") 