from pydantic_settings import BaseSettings

class SecuritySettings(BaseSettings):
    """Security and authentication configuration"""
    secret_key: str = ""
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 120
    
    class Config:
        env_prefix = "AUTH_"
        env_file = ".env"
        extra = "ignore"  
