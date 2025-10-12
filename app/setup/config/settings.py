from pydantic_settings import BaseSettings
from .database import DatabaseSettings
from .security import SecuritySettings
from .assessment import AssessmentSettings
from .api import APISettings

class Settings(BaseSettings):
    """Main settings aggregator"""
    environment: str = "development"
    debug: bool = True
    log_level: str = "INFO"
    
    database: DatabaseSettings = DatabaseSettings()
    security: SecuritySettings = SecuritySettings()
    assessment: AssessmentSettings = AssessmentSettings()
    api: APISettings = APISettings()
    
    class Config:
        env_file = ".env"
        extra = "ignore"  


settings = Settings()


