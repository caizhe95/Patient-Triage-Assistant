from __future__ import annotations 
 
from functools import lru_cache 
 
from qdrant_client import QdrantClient 
 
@lru_cache(maxsize=8) 
def _create_cached_client(url: str, api_key: str, storage_dir: str): 
    if url: 
        if api_key: 
            return QdrantClient(url=url, api_key=api_key) 
        return QdrantClient(url=url) 
    return QdrantClient(path=storage_dir) 
 
def create_qdrant_client(settings): 
    return _create_cached_client(str(settings.qdrant_url or ''), str(settings.qdrant_api_key or ''), str(settings.qdrant_storage_dir))
