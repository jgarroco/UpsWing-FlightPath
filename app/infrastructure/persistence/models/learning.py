import uuid
from decimal import Decimal
from datetime import datetime
from typing import Optional, List , TYPE_CHECKING
from sqlalchemy import (
    String,
    Integer,
    Boolean,
    TIMESTAMP,
    DECIMAL,
    ForeignKey,
    JSON,
    CHAR,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.infrastructure.persistence.base import Base

if TYPE_CHECKING:
    from app.infrastructure.persistence.models.assessment import AssessmentConfig


def new_uuid() -> str:
    return str(uuid.uuid4())


class LearningPathway(Base):
    __tablename__ = "learning_pathway"

    id: Mapped[str] = mapped_column(CHAR(36), primary_key=True, default=new_uuid)
    name: Mapped[str] = mapped_column(String(50), nullable=False, unique=True)
    description: Mapped[Optional[str]] = mapped_column(String(100))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP, default=func.now(), onupdate=func.now())

    assessment_configs: Mapped[List["AssessmentConfig"]] = relationship(
        "AssessmentConfig", back_populates="learning_pathway"
    )
    courses: Mapped[List["Course"]] = relationship("Course", back_populates="pathway")


class Course(Base):
    __tablename__ = "course"

    id: Mapped[str] = mapped_column(CHAR(36), primary_key=True, default=new_uuid)
    pathway_id: Mapped[str] = mapped_column(CHAR(36), ForeignKey("learning_pathway.id"))
    title: Mapped[str] = mapped_column(String(50), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(String(255))
    course_code: Mapped[Optional[str]] = mapped_column(String(50))
    target_proficiency_level: Mapped[Optional[str]] = mapped_column(String(20))
    primary_skill: Mapped[Optional[str]] = mapped_column(String(50))
    skill_scores: Mapped[Optional[dict]] = mapped_column(JSON)
    estimated_duration_hours: Mapped[Optional[Decimal]] = mapped_column(DECIMAL)
    difficulty_order: Mapped[Optional[int]] = mapped_column(Integer)
    prerequisites: Mapped[Optional[dict]] = mapped_column(JSON)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP, default=func.now(), onupdate=func.now())

    pathway: Mapped["LearningPathway"] = relationship("LearningPathway", back_populates="courses")
    lessons: Mapped[List["Lesson"]] = relationship(
        "Lesson", back_populates="course", cascade="all, delete-orphan"
    )


class Lesson(Base):
    __tablename__ = "lesson"

    id: Mapped[str] = mapped_column(CHAR(36), primary_key=True, default=new_uuid)
    course_id: Mapped[str] = mapped_column(CHAR(36), ForeignKey("course.id"))
    title: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(String(255))
    lesson_order: Mapped[Optional[int]] = mapped_column(Integer)
    target_skills: Mapped[Optional[list]] = mapped_column(JSON)
    learning_objectives: Mapped[Optional[list]] = mapped_column(JSON)
    content_type: Mapped[Optional[str]] = mapped_column(String(50))
    relative_difficulty: Mapped[Optional[Decimal]] = mapped_column(DECIMAL)
    estimated_duration_minutes: Mapped[Optional[Decimal]] = mapped_column(DECIMAL)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    course: Mapped["Course"] = relationship("Course", back_populates="lessons")
