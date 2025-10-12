import uuid
from decimal import Decimal
from datetime import datetime
from typing import Optional, TYPE_CHECKING
from sqlalchemy import (
    String,
    Integer,
    TIMESTAMP,
    ForeignKey,
    DECIMAL,
    CHAR,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.infrastructure.persistence.base import Base
if TYPE_CHECKING:
    from app.infrastructure.persistence.models.assessment import Result


def new_uuid() -> str:
    return str(uuid.uuid4())


class RecommendationItem(Base):
    __tablename__ = "recommendation_item"

    id: Mapped[str] = mapped_column(CHAR(36), primary_key=True, default=new_uuid)
    result_id: Mapped[str] = mapped_column(CHAR(36), ForeignKey("result.id"))
    content_id: Mapped[str] = mapped_column(CHAR(36))
    content_type: Mapped[str] = mapped_column(String(20), nullable=False)
    target_skill: Mapped[str] = mapped_column(String(50), nullable=False)
    skill_gap_size: Mapped[Optional[Decimal]] = mapped_column(DECIMAL)
    priority_order: Mapped[Optional[int]] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP, default=func.now())

    result: Mapped["Result"] = relationship("Result")
