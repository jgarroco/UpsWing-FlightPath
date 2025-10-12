from pydantic_settings import BaseSettings
from typing import List

class APISettings(BaseSettings):
    """API and HTTP configuration"""
    v1_prefix: str = ""
    cors_origins: List[str] = ["*"]
    
    class Config:
        env_prefix = "API_"
        env_file = ".env"
        extra = "ignore"  
