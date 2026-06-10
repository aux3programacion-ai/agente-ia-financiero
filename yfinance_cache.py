#!/usr/bin/env python3
"""
yfinance_cache.py - Cache persistente para llamadas a yfinance usando diskcache.
Reduce drásticamente el tiempo de descarga de datos.
"""
import os
import time
import hashlib
from pathlib import Path
from functools import wraps

DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')
CACHE_DIR = Path(DATA_DIR) / 'Datos' / 'cache_yf'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

try:
    from diskcache import Cache
    CACHE = Cache(str(CACHE_DIR))
    DISKCACHE_AVAILABLE = True
except ImportError:
    DISKCACHE_AVAILABLE = False
    CACHE = None

CACHE_TTL = int(os.environ.get('YF_CACHE_TTL', '3600'))  # 1h default


def cache_yfinance(ttl: int = CACHE_TTL):
    """
    Decorador que cachea resultados de yfinance.
    Usa hash de (ticker, period, interval, start, end) como key.
    
    Args:
        ttl: Time-to-live en segundos (default 1h)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not DISKCACHE_AVAILABLE or not CACHE:
                return func(*args, **kwargs)
            
            kwargs.pop('progress', None)
            kwargs.pop('auto_adjust', None)
            kwargs['progress'] = False
            
            key_parts = [func.__name__] + [str(a) for a in args]
            for k, v in sorted(kwargs.items()):
                if k not in ('progress', 'auto_adjust'):
                    key_parts.append(f'{k}:{v}')
            
            cache_key = hashlib.md5('|'.join(key_parts).encode()).hexdigest()
            
            cached = CACHE.get(cache_key)
            if cached is not None:
                return cached
            
            result = func(*args, **kwargs)
            
            if result is not None and not (hasattr(result, 'empty') and result.empty):
                CACHE.set(cache_key, result, expire=ttl)
            
            return result
        return wrapper
    return decorator


def clear_cache(pattern: str = None):
    """Limpia cache. Si pattern=None, limpia todo."""
    if not CACHE:
        return
    if pattern:
        count = 0
        for key in CACHE.iterkeys():
            if pattern in key:
                CACHE.delete(key)
                count += 1
        print(f'[Cache] Limpiados {count} items con pattern={pattern}')
    else:
        CACHE.clear()
        print('[Cache] Cache completamente limpiada')


def cache_stats() -> dict:
    if not CACHE:
        return {'available': False}
    return {
        'available': True,
        'size': CACHE.volume(),
        'size_mb': round(CACHE.volume() / (1024 * 1024), 2),
        'count': len(CACHE),
        'hits': getattr(CACHE, 'hits', 0),
        'misses': getattr(CACHE, 'misses', 0),
        'location': str(CACHE.directory)
    }


def monkey_patch_yfinance():
    """Parchea yfinance.download para usar cache automáticamente."""
    import yfinance as yf
    original_download = yf.download
    
    @wraps(original_download)
    def cached_download(*args, **kwargs):
        if not DISKCACHE_AVAILABLE:
            return original_download(*args, **kwargs)
        
        kwargs.pop('progress', None)
        kwargs.pop('auto_adjust', None)
        
        key_parts = ['yf_download'] + [str(a) for a in args]
        for k, v in sorted(kwargs.items()):
            key_parts.append(f'{k}:{v}')
        
        cache_key = hashlib.md5('|'.join(key_parts).encode()).hexdigest()
        
        ttl = CACHE_TTL
        cached = CACHE.get(cache_key)
        if cached is not None:
            return cached
        
        result = original_download(*args, **kwargs)
        if result is not None and not result.empty:
            CACHE.set(cache_key, result, expire=ttl)
        
        return result
    
    yf.download = cached_download
    print(f'[Cache] yfinance monkey-patched con cache (TTL={ttl}s)')


if __name__ == '__main__':
    print(f'[Cache] diskcache available: {DISKCACHE_AVAILABLE}')
    if DISKCACHE_AVAILABLE:
        print(json.dumps(cache_stats(), indent=2))
    
    if not DISKCACHE_AVAILABLE:
        print('[Cache] Instalar: pip install diskcache')