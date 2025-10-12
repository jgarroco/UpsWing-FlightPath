from pydantic_settings import BaseSettings
from typing import Dict, List

class AssessmentSettings(BaseSettings):
    """Assessment and CAT algorithm defaults"""
    default_max_questions: int = 25
    default_min_questions: int = 10
    default_skill_areas: List[str] = ["grammar", "vocabulary", "reading"]
    default_stopping_criterion: Dict[str, float] = {"standard_error": 0.3}
    session_timeout_minutes: int = 120
    
    cefr_thresholds: Dict[str, float] = {
        "A1": -2.0,
        "A2": -1.0,
        "B1": 0.0,
        "B2": 1.0,
        "C1": 2.0,
        "C2": 3.0,
    }
    
    class Config:
        env_prefix = "ASSESSMENT_"
        env_file = ".env"   
        extra = "ignore"  


