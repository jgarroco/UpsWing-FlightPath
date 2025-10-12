import uuid
from enum import Enum
from decimal import Decimal
from datetime import datetime
from typing import Optional, List,TYPE_CHECKING
from sqlalchemy import (
    String,
    Integer,
    Boolean,
    Text,
    DECIMAL,
    TIMESTAMP,
    ForeignKey,
    JSON,
    Enum as SQLEnum,
    CHAR,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.infrastructure.persistence.base import Base
if TYPE_CHECKING:
    from app.infrastructure.persistence.models.learning import LearningPathway


class TestTakerType(str, Enum):
    STUDENT = "student"
    TEACHER = "teacher"

class ResultType(str, Enum):
    PLACEMENT = "P"
    SPEAKING = "S"
    WRITING = "W"

def new_uuid() -> str:
    return str(uuid.uuid4())

class AssessmentSession(Base):
    __tablename__ = "assessment_session"

    id: Mapped[str] = mapped_column(CHAR(36), primary_key=True, default=new_uuid)
    config_id: Mapped[str] = mapped_column(CHAR(36), ForeignKey("assessment_config.id"))
    test_taker_id: Mapped[str] = mapped_column(String(50), nullable=False)
    test_taker_type: Mapped[TestTakerType] = mapped_column(SQLEnum(TestTakerType), default=TestTakerType.STUDENT)
    current_ability: Mapped[Decimal] = mapped_column(DECIMAL(8, 4), default=0.0)
    standard_error: Mapped[Optional[Decimal]] = mapped_column(DECIMAL(8, 4))
    questions_answered: Mapped[int] = mapped_column(Integer, default=0)
    is_complete: Mapped[bool] = mapped_column(Boolean, default=False)
    started_at: Mapped[datetime] = mapped_column(TIMESTAMP, default=func.now())
    completed_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP)
    expires_at: Mapped[datetime] = mapped_column(TIMESTAMP)

    config: Mapped["AssessmentConfig"] = relationship("AssessmentConfig", back_populates="sessions")
    responses: Mapped[List["AssessmentResponse"]] = relationship(
        "AssessmentResponse", back_populates="session", cascade="all, delete-orphan"
    )
    results: Mapped[List["Result"]] = relationship(
        "Result", back_populates="session", cascade="all, delete-orphan"
    )

class AssessmentResponse(Base):
    __tablename__ = "assessment_response"

    id: Mapped[str] = mapped_column(CHAR(36), primary_key=True, default=new_uuid)
    session_id: Mapped[str] = mapped_column(CHAR(36), ForeignKey("assessment_session.id"))
    item_id: Mapped[str] = mapped_column(CHAR(36), ForeignKey("assessment_item.id"))
    response_data: Mapped[dict] = mapped_column(JSON, nullable=False)
    is_correct: Mapped[Optional[bool]] = mapped_column(Boolean)
    raw_score: Mapped[Optional[Decimal]] = mapped_column(DECIMAL(5, 2))
    presented_at: Mapped[datetime] = mapped_column(TIMESTAMP, default=func.now())
    submitted_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP)
    time_taken: Mapped[Optional[int]] = mapped_column(Integer)

    session: Mapped["AssessmentSession"] = relationship("AssessmentSession", back_populates="responses")
    item: Mapped["AssessmentItem"] = relationship("AssessmentItem", back_populates="responses")

class AssessmentItem(Base):
    __tablename__ = "assessment_item"

    id: Mapped[str] = mapped_column(CHAR(36), primary_key=True, default=new_uuid)
    content: Mapped[dict] = mapped_column(JSON, nullable=False)
    item_type: Mapped[str] = mapped_column(String(50))
    skill_area: Mapped[list] = mapped_column(JSON)
    target_proficiency_level: Mapped[str] = mapped_column(String(15))
    parameters: Mapped[dict] = mapped_column(JSON)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    responses: Mapped[List["AssessmentResponse"]] = relationship("AssessmentResponse", back_populates="item")

class AssessmentConfig(Base):
    __tablename__ = "assessment_config"

    id: Mapped[str] = mapped_column(CHAR(36), primary_key=True, default=new_uuid)
    learning_pathway_id: Mapped[str] = mapped_column(CHAR(36), ForeignKey("learning_pathway.id"))
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    assessment_type: Mapped[str] = mapped_column(String(100), nullable=False)
    starting_ability: Mapped[Decimal] = mapped_column(DECIMAL)
    max_questions: Mapped[Optional[int]] = mapped_column(Integer)
    min_questions: Mapped[Optional[int]] = mapped_column(Integer)
    stopping_criterion: Mapped[dict] = mapped_column(JSON)
    skill_areas: Mapped[list] = mapped_column(JSON)
    proficiency_range: Mapped[Optional[dict]] = mapped_column(JSON)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    learning_pathway: Mapped["LearningPathway"] = relationship(
        "LearningPathway", back_populates="assessment_configs"
    )
    sessions: Mapped[List["AssessmentSession"]] = relationship("AssessmentSession", back_populates="config")

class Result(Base):
    __tablename__ = "result"

    id: Mapped[str] = mapped_column(CHAR(36), primary_key=True, default=new_uuid)
    session_id: Mapped[str] = mapped_column(CHAR(36), ForeignKey("assessment_session.id"))
    proficiency_level: Mapped[Optional[str]] = mapped_column(String(10))
    feedback: Mapped[Optional[str]] = mapped_column(Text)
    validated: Mapped[bool] = mapped_column(Boolean, default=False)
    skill_scores: Mapped[Optional[dict]] = mapped_column(JSON)
    result_type: Mapped[ResultType] = mapped_column(SQLEnum(ResultType), nullable=False)
    standard_error: Mapped[Optional[Decimal]] = mapped_column(DECIMAL(8, 4))
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP, default=func.now())

    session: Mapped["AssessmentSession"] = relationship("AssessmentSession", back_populates="results")

class PlacementResult(Base):
    __tablename__ = "placement_result"

    id: Mapped[str] = mapped_column(CHAR(36), primary_key=True, default=new_uuid)
    result_id: Mapped[str] = mapped_column(CHAR(36), ForeignKey("result.id"))
    final_ability: Mapped[Optional[dict]] = mapped_column(JSON)
    completion_time: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP)

    result: Mapped["Result"] = relationship("Result")

class SpeakingResult(Base):
    __tablename__ = "speaking_result"

    id: Mapped[str] = mapped_column(CHAR(36), primary_key=True, default=new_uuid)
    result_id: Mapped[str] = mapped_column(CHAR(36), ForeignKey("result.id"))
    transcript: Mapped[Optional[str]] = mapped_column(Text)
    criteria_scores: Mapped[Optional[dict]] = mapped_column(JSON)
    overall_score: Mapped[Optional[Decimal]] = mapped_column(DECIMAL)

    result: Mapped["Result"] = relationship("Result")

class WritingResult(Base):
    __tablename__ = "writing_result"

    id: Mapped[str] = mapped_column(CHAR(36), primary_key=True, default=new_uuid)
    result_id: Mapped[str] = mapped_column(CHAR(36), ForeignKey("result.id"))
    essay_text: Mapped[Optional[str]] = mapped_column(Text)
    criteria_scores: Mapped[Optional[dict]] = mapped_column(JSON)
    overall_score: Mapped[Optional[Decimal]] = mapped_column(DECIMAL)

    result: Mapped["Result"] = relationship("Result")
