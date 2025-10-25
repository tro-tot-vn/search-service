from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 5555
    
    # Milvus
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    
    # Model
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    EMBEDDING_DIM: int = 128
    DEVICE: str = "cuda"
    
    # Search
    DEFAULT_SEARCH_LIMIT: int = 100
    MAX_SEARCH_LIMIT: int = 500
    
    class Config:
        env_file = ".env"

settings = Settings()

