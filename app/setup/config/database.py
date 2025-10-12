from pydantic_settings import BaseSettings

class DatabaseSettings(BaseSettings):
    """Database configuration"""
    url: str = ""
    pool_size: int = 20
    max_overflow: int = 0
    echo: bool = False
    
    class Config:
        env_prefix = "DATABASE_"
        env_file = ".env"
        extra = "ignore"  

