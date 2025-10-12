from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
from enum import Enum
from app.application.dto import PlacementTestItemPublicDTO, ProgressDTO, PlacementResultDTO, SpeakingResultDTO, WritingResultDTO, RecommendationItemDTO


class AssessmentType(str, Enum):
    ADAPTIVE = "adaptive"
    DIAGNOSTIC_WRITING = "diagnostic_writing"
    DIAGNOSTIC_SPEAKING = "diagnostic_speaking"


class LearningPathwayType(str, Enum):
    DEFAULT = "Default"
    GENERAL = "General"
    ACADEMIC = "Academic"
    LIFESOCIAL = "Life & Social"
    CAREER = "Career"


class PlacementStartRequest(BaseModel):
    test_taker_id: str = Field(..., min_length=1, max_length=50)
    assessment_type: AssessmentType
    learning_pathway: LearningPathwayType

    class Config:
        from_attributes = True


class PlacementSubmitAnswerRequest(BaseModel):
    test_taker_id: str = Field(..., min_length=1, max_length=50)
    response_data: Dict[str, Any] = Field(..., description="Student's answer data")
    time_taken: Optional[int] = Field(None, description="Time taken in seconds")

    class Config:
        from_attributes = True


class PlacementTestStartResponse(BaseModel):
    session_id: str
    first_question: PlacementTestItemPublicDTO
    progress: ProgressDTO

    class Config:
        from_attributes = True



class PlacementTestSubmitAnswerResponse(BaseModel):
    next_question: Optional[PlacementTestItemPublicDTO] = None
    progress: ProgressDTO
    assessment_complete: bool = False

    class Config:
        from_attributes = True

class AssessmentCompleteResponse(BaseModel):
    session_id: str
    placement_result: Optional[PlacementResultDTO] = None
    speaking_result: Optional[SpeakingResultDTO] = None
    writing_result: Optional[WritingResultDTO] = None
    recommendations: List[RecommendationItemDTO] = []

