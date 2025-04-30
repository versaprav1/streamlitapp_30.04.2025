from functools import lru_cache
import hashlib
import json
from typing import Dict, Any, Optional

class LLMCache:
    def __init__(self, cache_size: int = 1000):
        self.cache_size = cache_size
        
    @lru_cache(maxsize=1000)
    def _generate_cache_key(self, prompt: str, temperature: float, model: str) -> str:
        """Generate a unique cache key for the LLM request"""
        params = f"{prompt}:{temperature}:{model}"
        return hashlib.md5(params.encode()).hexdigest()
    
    @lru_cache(maxsize=1000)
    def get_cached_response(self, prompt: str, temperature: float, model: str) -> Optional[Dict[str, Any]]:
        """Get cached LLM response if available"""
        cache_key = self._generate_cache_key(prompt, temperature, model)
        return self.get_cached_response_by_key(cache_key)
    
    def cache_response(self, prompt: str, temperature: float, model: str, response: Dict[str, Any]) -> None:
        """Cache an LLM response"""
        cache_key = self._generate_cache_key(prompt, temperature, model)
        self.cache_response_by_key(cache_key, response)