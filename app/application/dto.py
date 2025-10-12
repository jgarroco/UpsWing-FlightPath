from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from decimal import Decimal
from enum import Enum

from datetime import datetime


class ItemType(str, Enum):
    MULTIPLE_CHOICE = "multiple_choice"


class ResultType(str, Enum):
    PLACEMENT = "P"
    SPEAKING = "S"
    WRITING = "W"


class AssessmentConfigDTO(BaseModel):
    """DTO representing the configuration for an assessment."""
    id: str
    learning_pathway_id: str
    name: str = Field(..., max_length=100)
    assessment_type: str = Field(..., max_length=100)
    starting_ability: Optional[Decimal] = None
    max_questions: Optional[int] = None
    min_questions: Optional[int] = None
    stopping_criterion: Optional[Dict[str, Any]] = None
    skill_areas: Optional[List[str]] = None
    proficiency_range: Optional[Dict[str, Any]] = None
    is_active: bool = True

    class Config:
        from_attributes = True

class LearningPathwayDTO(BaseModel):
    """DTO representing a learning pathway."""
    id: str
    name: str
    description: Optional[str]
    is_active: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class ResultDTO(BaseModel):
    """DTO representing a generic result for an assessment session."""
    id: str  
    session_id: str
    proficiency_level: Optional[str] = None
    feedback: Optional[str] = None
    validated: bool
    skill_scores: Optional[Dict[str, Any]] = None
    result_type: ResultType
    standard_error: Optional[Decimal] = None
    created_at: datetime

    class Config:
        from_attributes = True

class PlacementResultDTO(BaseModel):
    """DTO representing a placement result with additional placement-specific fields."""
    id: str  
    session_id: str
    proficiency_level: Optional[str] = None
    feedback: Optional[str] = None
    validated: bool
    skill_scores: Optional[Dict[str, Any]] = None
    result_type: ResultType
    standard_error: Optional[Decimal] = None
    created_at: datetime

    result_id: str
    final_ability: Dict[str, Any]
    completion_time: datetime

    class Config:
        from_attributes = True

class SpeakingResultDTO(BaseModel):
    """DTO representing a speaking result with transcript and criteria scores."""
    id: str  
    session_id: str
    proficiency_level: Optional[str] = None
    feedback: Optional[str] = None
    validated: bool
    skill_scores: Optional[Dict[str, Any]] = None
    result_type: ResultType
    standard_error: Optional[Decimal] = None
    created_at: datetime

    result_id: str
    transcript: Optional[str]
    criteria_scores: Optional[Dict[str, Any]]
    overall_score: Optional[Decimal]

    class Config:
        from_attributes = True

class WritingResultDTO(BaseModel):
    """DTO representing a writing result with essay text and criteria scores."""
    id: str
    session_id: str
    proficiency_level: Optional[str] = None
    feedback: Optional[str] = None
    validated: bool
    skill_scores: Optional[Dict[str, Any]] = None
    result_type: ResultType
    standard_error: Optional[Decimal] = None
    created_at: datetime

    result_id: str
    essay_text: Optional[str]
    criteria_scores: Optional[Dict[str, Any]]
    overall_score: Optional[Decimal]

    class Config:
        from_attributes = True

class CourseDTO(BaseModel):
    """DTO representing a course within a learning pathway."""
    id: str
    pathway_id: str
    title: str
    description: Optional[str]
    course_code: Optional[str]
    target_proficiency_level: Optional[str]
    primary_skill: Optional[str]
    skill_scores: Optional[Dict[str, Any]]
    estimated_duration_hours: Optional[Decimal]
    difficulty_order: Optional[int]
    prerequisites: Optional[List[str]]
    is_active: Optional[datetime]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class LessonDTO(BaseModel):
    """DTO representing a lesson within a course."""
    id: str
    course_id: str
    title: str
    description: Optional[str]
    lesson_order: Optional[int]
    target_skills: Optional[List[str]]
    learning_objectives: Optional[List[str]]
    content_type: Optional[str]
    relative_difficulty: Optional[Decimal]
    estimated_duration_minutes: Optional[Decimal]
    is_active: bool

    class Config:
        from_attributes = True

class RecommendationItemDTO(BaseModel):
    """DTO representing a recommended content item for a user."""
    id: str
    result_id: str
    content_id: str
    content_type: str
    target_skill: str
    skill_gap_size: Optional[Decimal]
    priority_order: Optional[int]
    created_at: datetime

    class Config:
        from_attributes = True

class QuestionContentBase(BaseModel):
    """Base class for different question content types"""
    pass

class MultipleChoiceContent(QuestionContentBase):
    """DTO representing content for multiple choice items."""
    item: str  
    options: List[str]
    instruction: Optional[str] = ""

    class Config:
        from_attributes = True

PlacementItemContentDTO = MultipleChoiceContent

class PlacementTestItemPublicDTO(BaseModel):
    """DTO for placement test items sent to the client (no sensitive data)."""
    id: str
    content: PlacementItemContentDTO
    item_type: ItemType
    skill_area: List[str]
    target_proficiency_level: str

    class Config:
        from_attributes = True

class PlacementTestItemPrivateDTO(BaseModel):
    """DTO for placement test items used in business logic (with sensitive data)."""
    id: str
    content: PlacementItemContentDTO
    item_type: ItemType
    skill_area: List[str]
    target_proficiency_level: str
    parameters: Dict[str, float]
    correct_answer: Optional[str] = None

    class Config:
        from_attributes = True

class ProgressDTO(BaseModel):
    """DTO representing the progress of a placement test session."""
    questions_completed: int
    max_questions: Optional[int] = None
    estimated_remaining: Optional[int] = None
    current_ability: Optional[float] = None
    standard_error: Optional[Decimal] = None

    class Config:
        from_attributes = True

class StartPlacementTestCommand(BaseModel):
    """Command DTO to start a placement test session."""
    test_taker_id: str
    assessment_type: str
    learning_pathway: str

    class Config:
        from_attributes = True


class StartPlacementTestResult(BaseModel):
    """DTO representing the result of starting a placement test session."""
    session_id: str
    first_question: PlacementTestItemPublicDTO
    progress: ProgressDTO

    class Config:
        from_attributes = True

class SubmitAnswerCommand(BaseModel):
    """Command DTO to submit an answer for a placement test item."""
    session_id: str
    response_data: Dict
    time_taken: Optional[int] = None

    class Config:
        from_attributes = True

class SubmitAnswerResult(BaseModel):
    """DTO representing the result from submitting an answer."""
    next_question: Optional[PlacementTestItemPublicDTO]
    progress: ProgressDTO
    is_complete: bool
    is_correct: bool

    class Config:
        from_attributes = True
