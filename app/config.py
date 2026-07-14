from functools import lru_cache  
from pathlib import Path  
  
from pydantic import Field  
from pydantic_settings import BaseSettings, SettingsConfigDict  
  
BASE_DIR = Path(__file__).resolve().parent.parent  
  
class Settings(BaseSettings):  
    model_config = SettingsConfigDict(env_file=BASE_DIR / '.env', extra='ignore')  
  
    app_name: str = 'Patient Triage Assistant'  
    app_host: str = '0.0.0.0'  
    app_port: int = 8000  
    log_level: str = 'INFO'  
  
    deepseek_api_key: str = Field(default='', alias='DEEPSEEK_API_KEY')  
    deepseek_base_url: str = Field(default='https://api.deepseek.com', alias='DEEPSEEK_BASE_URL')  
    deepseek_model: str = Field(default='deepseek-chat', alias='DEEPSEEK_MODEL')  
  
    embedding_backend: str = Field(default='bge', alias='EMBEDDING_BACKEND')  
    embedding_model: str = Field(default='BAAI/bge-small-zh-v1.5', alias='EMBEDDING_MODEL')  
    rerank_model: str = Field(default='BAAI/bge-reranker-base', alias='RERANK_MODEL')  
  
    raw_data_dir: Path = BASE_DIR / 'data' / 'raw'  
    processed_data_dir: Path = BASE_DIR / 'data' / 'processed'  
    qdrant_url: str = Field(default='http://127.0.0.1:6333', alias='QDRANT_URL')  
    qdrant_api_key: str = Field(default='', alias='QDRANT_API_KEY')  
    qdrant_storage_dir: Path = Field(default=BASE_DIR / 'indexes' / 'qdrant', alias='QDRANT_STORAGE_DIR')  
    bm25_index_dir: Path = BASE_DIR / 'indexes' / 'bm25'  
    qdrant_collection_prefix: str = Field(default='patient_triage', alias='QDRANT_COLLECTION_PREFIX')  
  
    retrieval_top_k: int = 4  
    route_top_k: int = 1  
    rrf_k: int = 60  
    rerank_top_k: int = 4  
    chunk_size: int = 320  
    chunk_overlap: int = 50  
    api_timeout_seconds: int = 45  
    enable_llm_route: bool = True  
    enable_llm_answer: bool = True  
    enable_bge_reranker: bool = Field(default=False, alias='ENABLE_BGE_RERANKER')
  
@lru_cache(maxsize=1)  
def get_settings():  
    return Settings() 
