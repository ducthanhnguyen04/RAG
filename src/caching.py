"""
Caching System - Query and result caching for performance optimization
Supports: in-memory cache, file-based cache, TTL-based expiration
"""
import json
import hashlib
import time
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timedelta
from utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class CachedResult:
    """Cached result with metadata"""
    query: str
    result: Any
    timestamp: float
    ttl: int
    hits: int = 0
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired"""
        if self.ttl == 0:  # No expiration
            return False
        return time.time() - self.timestamp > self.ttl
    
    def touch(self):
        """Update hit count and timestamp"""
        self.hits += 1
        self.timestamp = time.time()


class MemoryCache:
    """
    In-memory cache with TTL support
    Fast but limited by RAM
    """
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize memory cache
        
        Args:
            max_size: Maximum number of cached items
        """
        self.max_size = max_size
        self.cache: Dict[str, CachedResult] = {}
        self.logger = logger
    
    def _get_key(self, query: str) -> str:
        """Generate cache key from query"""
        return hashlib.md5(query.encode()).hexdigest()
    
    def get(self, query: str) -> Optional[Any]:
        """
        Get cached result
        
        Args:
            query: Query string
            
        Returns:
            Cached result or None
        """
        key = self._get_key(query)
        
        if key not in self.cache:
            return None
        
        cached = self.cache[key]
        
        if cached.is_expired():
            del self.cache[key]
            return None
        
        cached.touch()
        self.logger.debug(f"Cache HIT for: {query[:30]}...")
        return cached.result
    
    def set(self, query: str, result: Any, ttl: int = 3600) -> None:
        """
        Cache result
        
        Args:
            query: Query string
            result: Result to cache
            ttl: Time to live in seconds (0 = no expiration)
        """
        # Evict oldest if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = min(
                self.cache.keys(),
                key=lambda k: self.cache[k].timestamp
            )
            del self.cache[oldest_key]
            self.logger.debug("Cache evicted oldest entry")
        
        key = self._get_key(query)
        self.cache[key] = CachedResult(
            query=query,
            result=result,
            timestamp=time.time(),
            ttl=ttl
        )
        
        self.logger.debug(f"Cache SET for: {query[:30]}...")
    
    def clear(self) -> None:
        """Clear all cache"""
        self.cache.clear()
        self.logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_hits = sum(item.hits for item in self.cache.values())
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "total_hits": total_hits,
            "avg_hits_per_item": total_hits / len(self.cache) if self.cache else 0
        }


class FileCache:
    """
    File-based cache for persistence
    Slower but survives process restarts
    """
    
    def __init__(self, cache_dir: str = "./cache", ttl: int = 3600):
        """
        Initialize file cache
        
        Args:
            cache_dir: Directory for cache files
            ttl: Time to live in seconds
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.ttl = ttl
        self.logger = logger
    
    def _get_filename(self, query: str) -> Path:
        """Generate cache filename from query"""
        key = hashlib.md5(query.encode()).hexdigest()
        return self.cache_dir / f"{key}.json"
    
    def get(self, query: str) -> Optional[Any]:
        """
        Get cached result from file
        
        Args:
            query: Query string
            
        Returns:
            Cached result or None
        """
        filepath = self._get_filename(query)
        
        if not filepath.exists():
            return None
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Check expiration
            timestamp = data.get('timestamp', 0)
            if time.time() - timestamp > self.ttl:
                filepath.unlink()
                return None
            
            self.logger.debug(f"File cache HIT for: {query[:30]}...")
            return data.get('result')
            
        except Exception as e:
            self.logger.error(f"File cache read error: {e}")
            return None
    
    def set(self, query: str, result: Any) -> None:
        """
        Cache result to file
        
        Args:
            query: Query string
            result: Result to cache
        """
        filepath = self._get_filename(query)
        
        try:
            data = {
                'query': query,
                'result': result,
                'timestamp': time.time()
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f)
            
            self.logger.debug(f"File cache SET for: {query[:30]}...")
            
        except Exception as e:
            self.logger.error(f"File cache write error: {e}")
    
    def clear(self) -> None:
        """Clear all cache files"""
        for f in self.cache_dir.glob('*.json'):
            f.unlink()
        self.logger.info("File cache cleared")


class HybridCache:
    """
    Hybrid cache combining memory (fast) and file (persistent)
    """
    
    def __init__(self, memory_size: int = 1000, cache_dir: str = "./cache"):
        """
        Initialize hybrid cache
        
        Args:
            memory_size: Max items in memory
            cache_dir: Directory for file cache
        """
        self.memory = MemoryCache(max_size=memory_size)
        self.file = FileCache(cache_dir=cache_dir)
        self.logger = logger
    
    def get(self, query: str) -> Optional[Any]:
        """
        Get from memory first, then file
        
        Args:
            query: Query string
            
        Returns:
            Cached result or None
        """
        # Try memory first (faster)
        result = self.memory.get(query)
        if result is not None:
            return result
        
        # Try file cache
        result = self.file.get(query)
        if result is not None:
            # Restore to memory
            self.memory.set(query, result)
            return result
        
        return None
    
    def set(self, query: str, result: Any, ttl: int = 3600) -> None:
        """
        Cache in both memory and file
        
        Args:
            query: Query string
            result: Result to cache
            ttl: Time to live in seconds
        """
        self.memory.set(query, result, ttl=ttl)
        self.file.set(query, result)
        self.logger.debug(f"Hybrid cache SET for: {query[:30]}...")
    
    def clear(self) -> None:
        """Clear both caches"""
        self.memory.clear()
        self.file.clear()


class CacheManager:
    """
    Centralized cache management
    Handles cache strategy selection and statistics
    """
    
    def __init__(self, strategy: str = "hybrid"):
        """
        Initialize cache manager
        
        Args:
            strategy: "memory", "file", or "hybrid"
        """
        self.strategy = strategy
        self.logger = logger
        
        if strategy == "memory":
            self.cache = MemoryCache()
        elif strategy == "file":
            self.cache = FileCache()
        else:  # hybrid
            self.cache = HybridCache()
        
        self.logger.info(f"Cache manager initialized with {strategy} strategy")
    
    def get(self, query: str) -> Tuple[Optional[Any], bool]:
        """
        Get cached result
        
        Args:
            query: Query string
            
        Returns:
            Tuple of (result, is_cached)
        """
        result = self.cache.get(query)
        return result, result is not None
    
    def set(self, query: str, result: Any, ttl: int = 3600) -> None:
        """
        Cache result
        
        Args:
            query: Query string
            result: Result to cache
            ttl: Time to live in seconds
        """
        self.cache.set(query, result, ttl=ttl)
    
    def clear(self) -> None:
        """Clear cache"""
        self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if hasattr(self.cache, 'get_stats'):
            return self.cache.get_stats()
        return {}
