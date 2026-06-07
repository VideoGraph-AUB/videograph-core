"""
OpenAI API call caching by hashing model + prompt + input + params.
"""

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)


class OpenAICache:
    """Cache for OpenAI API calls to avoid redundant requests."""
    
    def __init__(self, cache_dir: str = ".cache", expiry: int = 0):
        """
        Initialize the cache.
        
        Args:
            cache_dir: Directory to store cache files
            expiry: Cache expiry in seconds (0 = never expire)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.expiry = expiry
        self._enabled = True
    
    @property
    def enabled(self) -> bool:
        return self._enabled
    
    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value
    
    def _compute_hash(self, model: str, messages: list, params: dict) -> str:
        """Compute a hash for the API call parameters."""
        # Create a deterministic string from the parameters
        cache_key = {
            "model": model,
            "messages": messages,
            "params": {k: v for k, v in sorted(params.items())}
        }
        key_str = json.dumps(cache_key, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def _get_cache_path(self, cache_hash: str) -> Path:
        """Get the file path for a cache entry."""
        # Use subdirectories to avoid too many files in one directory
        subdir = cache_hash[:2]
        return self.cache_dir / subdir / f"{cache_hash}.json"
    
    def get(self, model: str, messages: list, params: dict) -> Optional[dict]:
        """
        Get a cached response if available and not expired.
        
        Returns:
            The cached response dict, or None if not found/expired
        """
        if not self._enabled:
            return None
        
        cache_hash = self._compute_hash(model, messages, params)
        cache_path = self._get_cache_path(cache_hash)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cached = json.load(f)
            
            # Check expiry
            if self.expiry > 0:
                cached_time = cached.get("_cached_at", 0)
                if time.time() - cached_time > self.expiry:
                    logger.debug(f"Cache expired for {cache_hash[:8]}...")
                    cache_path.unlink()
                    return None
            
            logger.debug(f"Cache hit for {cache_hash[:8]}...")
            return cached.get("response")
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Invalid cache entry {cache_hash[:8]}: {e}")
            return None
    
    def set(self, model: str, messages: list, params: dict, response: dict):
        """Cache a response."""
        if not self._enabled:
            return
        
        cache_hash = self._compute_hash(model, messages, params)
        cache_path = self._get_cache_path(cache_hash)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        cached = {
            "_cached_at": time.time(),
            "_model": model,
            "response": response
        }
        
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cached, f, ensure_ascii=False)
        
        logger.debug(f"Cached response for {cache_hash[:8]}...")
    
    def clear(self):
        """Clear all cache entries."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Cache cleared")
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        total_files = 0
        total_size = 0
        
        for cache_file in self.cache_dir.rglob("*.json"):
            total_files += 1
            total_size += cache_file.stat().st_size
        
        return {
            "entries": total_files,
            "size_bytes": total_size,
            "size_mb": round(total_size / (1024 * 1024), 2)
        }


# Global cache instance
_cache: Optional[OpenAICache] = None


def get_cache(cache_dir: str = ".cache", expiry: int = 0) -> OpenAICache:
    """Get or create the global cache instance."""
    global _cache
    if _cache is None:
        _cache = OpenAICache(cache_dir, expiry)
    return _cache


