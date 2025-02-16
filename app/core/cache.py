from functools import lru_cache
from typing import Dict, Any
import time

class SearchCache:
    def __init__(self, maxsize=100, ttl=3600):  # 1 hour TTL
        self.cache = {}
        self.maxsize = maxsize
        self.ttl = ttl

    def get(self, key: str) -> Dict[str, Any]:
        if key in self.cache:
            item = self.cache[key]
            if time.time() - item['timestamp'] < self.ttl:
                return item['data']
            else:
                del self.cache[key]
        return None

    def set(self, key: str, value: Dict[str, Any]):
        if len(self.cache) >= self.maxsize:
            # Remove oldest item
            oldest = min(self.cache.items(), key=lambda x: x[1]['timestamp'])
            del self.cache[oldest[0]]
        
        self.cache[key] = {
            'data': value,
            'timestamp': time.time()
        } 