from typing import Dict, Any, TypeVar, Type
from core.dependency_injection import DependencyInjector

T = TypeVar('T')

class Factory:
    """
    Factory class to create components using dependency injection.
    """
    
    @staticmethod
    def create_from_config(component_type: str, config_path: str = None, config_dict: Dict[str, Any] = None) -> Any:
        """
        Creates a component instance from configuration.
        
        Args:
            component_type: The type of component to create (e.g., 'trainloop')
            config_path: Path to YAML configuration file
            config_dict: Dictionary containing configuration
            
        Returns:
            An instance of the specified component
        """
        injector = DependencyInjector(config_path=config_path, config_dict=config_dict)
        return injector.get_instance(component_type)
    
    @staticmethod
    def register_component_type(base_class: Type[T]) -> Type[T]:
        """
        Decorator to register a component type for dependency injection.
        
        Args:
            base_class: The base class to register
            
        Returns:
            The original base class
        """
        # This is just a marker decorator for now
        # We could add more functionality here later
        return base_class 