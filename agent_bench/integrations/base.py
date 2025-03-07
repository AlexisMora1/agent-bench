from typing import Callable, Any

class BaseModelProvider:
    """Base class for model providers integrations"""
    
    @staticmethod
    def get_model(*args, **kwargs) -> Callable:
        raise NotImplementedError("Each provider must implement its get_model method")