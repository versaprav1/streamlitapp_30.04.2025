from functools import wraps
from typing import List, Dict, Any

def validate_state(required_keys: List[str]):
    """
    Decorator to validate required state keys before executing agent methods.
    
    Args:
        required_keys (List[str]): List of required state keys
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            state = self.state
            if not state:
                raise ValueError("State object is None")
                
            missing_keys = [key for key in required_keys if key not in state]
            if missing_keys:
                raise ValueError(f"Missing required state keys: {missing_keys}")
            return func(self, *args, **kwargs)
        return wrapper
    return decorator