from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from core.factory import Factory
from components.dataloader import DataLoader
from components.preprocessor import Preprocessor
from components.model import Model
from components.optimizer import Optimizer
from components.metricfunction import MetricFunction
from components.tracker import Tracker


@Factory.register_component_type
class TrainLoop(ABC):
    """Base class for training loops."""
    
    def __init__(self, 
                 dataloader: DataLoader,
                 model: Model,
                 optimizer: Optimizer,
                 tracker: Tracker,
                 metricfunction: MetricFunction,
                 preprocessor: Optional[Preprocessor] = None,
                 **kwargs):
        self.dataloader = dataloader
        self.model = model
        self.optimizer = optimizer
        self.tracker = tracker
        self.metric_function = metricfunction
        self.preprocessor = preprocessor
    
    @abstractmethod
    def execute(self) -> None:
        """Execute the training loop."""
        pass
    
    @classmethod
    def create(cls, config: Union[str, Dict[str, Any]]):
        """Factory method to create a TrainLoop instance."""
        if isinstance(config, str):
            return Factory.create_from_config('trainloop', config_path=config)
        else:
            return Factory.create_from_config('trainloop', config_dict=config)


class StandardTrainLoop(TrainLoop):
    """Standard batch training loop."""
    
    def __init__(self, 
                 dataloader: DataLoader,
                 model: Model,
                 optimizer: Optimizer,
                 tracker: Tracker,
                 metricfunction: MetricFunction,
                 preprocessor: Optional[Preprocessor] = None,
                 epochs: int = 100,
                 batch_size: int = 32,
                 **kwargs):
        super().__init__(dataloader, model, optimizer, tracker, metricfunction, preprocessor)
        self.epochs = epochs
        self.batch_size = batch_size
    
    def execute(self) -> None:
        data = self.dataloader.load_data()
        
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        
        if self.preprocessor:
            X = self.preprocessor.fit_transform(X)
        
        self.tracker.log_params({
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'input_dim': X.shape[1]
        })
        
        n_samples = X.shape[0]
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size
        
        for epoch in range(self.epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0.0
            
            for batch in range(n_batches):
                start_idx = batch * self.batch_size
                end_idx = min((batch + 1) * self.batch_size, n_samples)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                loss = self.model.compute_loss(X_batch, y_batch)
                gradients = self.model.compute_gradients(X_batch, y_batch)
                
                self.optimizer.step(gradients)
                
                epoch_loss += loss * (end_idx - start_idx)
            
            epoch_loss /= n_samples
            
            y_pred = self.model.predict(X)
            metric_value = self.metric_function.calculate(y, y_pred)
            
            self.tracker.log_metric(f'loss_epoch_{epoch}', epoch_loss)
            self.tracker.log_metric(f'metric_epoch_{epoch}', metric_value)
            
            if epoch % 10 == 0:
                self.tracker.log_metric('epoch', epoch)
                print(f"Epoch {epoch}/{self.epochs}: Loss = {epoch_loss:.4f}, Metric = {metric_value:.4f}")


class OnlineLearningTrainLoop(TrainLoop):
    """Online learning training loop that processes one sample at a time."""
    
    def __init__(self, 
                 dataloader: DataLoader,
                 model: Model,
                 optimizer: Optimizer,
                 tracker: Tracker,
                 metricfunction: MetricFunction,
                 preprocessor: Optional[Preprocessor] = None,
                 epochs: int = 1,
                 **kwargs):
        super().__init__(dataloader, model, optimizer, tracker, metricfunction, preprocessor)
        self.epochs = epochs
    
    def execute(self) -> None:
        data = self.dataloader.load_data()
        
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        
        if self.preprocessor:
            X = self.preprocessor.fit_transform(X)
        
        self.tracker.log_params({
            'epochs': self.epochs,
            'input_dim': X.shape[1],
            'training_mode': 'online'
        })
        
        n_samples = X.shape[0]
        
        for epoch in range(self.epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0.0
            
            for i in range(n_samples):
                X_sample = X_shuffled[i:i+1]
                y_sample = y_shuffled[i:i+1]
                
                loss = self.model.compute_loss(X_sample, y_sample)
                gradients = self.model.compute_gradients(X_sample, y_sample)
                
                self.optimizer.step(gradients)
                
                epoch_loss += loss
                
                if i % 1000 == 0 and i > 0:
                    self.tracker.log_metric(f'sample_{i}_loss', loss)
            
            epoch_loss /= n_samples
            
            y_pred = self.model.predict(X)
            metric_value = self.metric_function.calculate(y, y_pred)
            
            self.tracker.log_metric(f'loss_epoch_{epoch}', epoch_loss)
            self.tracker.log_metric(f'metric_epoch_{epoch}', metric_value)
            
            self.tracker.log_metric('epoch', epoch)
            print(f"Epoch {epoch}/{self.epochs}: Loss = {epoch_loss:.4f}, Metric = {metric_value:.4f}") 