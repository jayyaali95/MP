import yaml
import importlib
import inspect
from typing import Dict, Any, Type, Set, List, Optional, Union


class DependencyInjector:
    """
    Manages the creation and injection of dependencies based on configuration.
    """
    
    def __init__(self, config_path: Optional[str] = None, config_dict: Optional[Dict[str, Any]] = None):
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        elif config_dict:
            self.config = config_dict
        else:
            self.config = {}
        
        self.instances = {}
        self.dependency_graph = {}
        self._build_dependency_graph()
    
    def _build_dependency_graph(self):
        """
        Analyzes the component classes to build a dependency graph.
        """
        from components import (
            DataLoader, MetricFunction, Tracker, Preprocessor, 
            TrainLoop, Optimizer, Model
        )
        
        base_classes = [
            DataLoader, MetricFunction, Tracker, Preprocessor, 
            TrainLoop, Optimizer, Model
        ]
        
        # Build dependency graph based on constructor parameters
        for base_class in base_classes:
            class_name = base_class.__name__.lower()
            self.dependency_graph[class_name] = []
            
            # Get all subclasses of the base class
            for subclass in base_class.__subclasses__():
                # Get constructor parameters
                params = inspect.signature(subclass.__init__).parameters
                
                # Skip self parameter
                for param_name, param in list(params.items())[1:]:
                    param_type = param.annotation
                    
                    # Check if the parameter type is one of our base classes
                    for bc in base_classes:
                        if (param_type == bc or 
                            (hasattr(param_type, "__origin__") and 
                             param_type.__origin__ is Optional and 
                             param_type.__args__[0] == bc)):
                            if bc.__name__.lower() not in self.dependency_graph[class_name]:
                                self.dependency_graph[class_name].append(bc.__name__.lower())
    
    def get_instance(self, component_type: str) -> Any:
        """
        Gets or creates an instance of the specified component type.
        
        Args:
            component_type: The type of component to get/create (e.g. 'dataloader')
            
        Returns:
            An instance of the specified component type
        """
        # Return existing instance if already created
        if component_type in self.instances:
            return self.instances[component_type]
        
        # Check if component is in config
        if component_type not in self.config:
            raise ValueError(f"No configuration found for component type: {component_type}")
        
        # Get component configuration
        component_config = self.config[component_type]
        
        # Handle case where value is already an instance
        if isinstance(component_config, object) and hasattr(component_config, '__class__') and not isinstance(component_config, dict):
            self.instances[component_type] = component_config
            return component_config
        
        # Get the class to instantiate
        class_name = component_config.get('class')
        if not class_name:
            raise ValueError(f"No class specified for component type: {component_type}")
        
        # Create dependencies first (recursive call)
        dependencies = {}
        for dep_type in self.dependency_graph.get(component_type, []):
            if dep_type in self.config:
                dependencies[dep_type] = self.get_instance(dep_type)
        
        # Import the class
        module_path = f"components.{component_type.lower()}"
        module = importlib.import_module(module_path)
        component_class = getattr(module, class_name)
        
        kwargs = {k: v for k, v in component_config.items() if k != 'class'}
        
        kwargs.update(dependencies)
        
        instance = component_class(**kwargs)
        
        self.instances[component_type] = instance
        
        return instance 