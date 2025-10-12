from typing import Annotated
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.application.interactors import StartPlacementTestInteractor, SubmitAnswerInteractor

from app.domain.services.cat_service import CATService

from app.infrastructure.persistence.repositories.sql_repositories import (
    SQLPlacementTestSessionRepository,
    SQLAssessmentConfigRepository,
)
from app.infrastructure.persistence.uow.sql_uow import SQLAlchemyUnitOfWork
from app.infrastructure.adapters import (
    IRTServiceAdapter,
    SystemClockService,
)
from app.infrastructure.persistence.connection import get_session

def get_irt_service_adapter() -> IRTServiceAdapter:
    """Provides the concrete IRT service adapter."""
    return IRTServiceAdapter()

def get_system_clock_service() -> SystemClockService:
    """Provides the concrete system clock service."""
    return SystemClockService()

def get_cat_service(
    psychometric_model: Annotated[IRTServiceAdapter, Depends(get_irt_service_adapter)],
) -> CATService:
    """Builds the CATService using the concrete IRT adapter."""
    return CATService(psychometric_model)


def get_start_placement_test_interactor(
    db_session: Annotated[AsyncSession, Depends(get_session)],
    cat_service: Annotated[CATService, Depends(get_cat_service)],
    clock_service: Annotated[SystemClockService, Depends(get_system_clock_service)],
) -> StartPlacementTestInteractor:
    """Constructs the interactor for starting a placement test."""
    session_repo = SQLPlacementTestSessionRepository(db_session)
    config_repo = SQLAssessmentConfigRepository(db_session)
    uow = SQLAlchemyUnitOfWork(db_session)

    return StartPlacementTestInteractor(
        session_repo, config_repo, cat_service, clock_service, uow
    )

def get_submit_answer_interactor(
    db_session: Annotated[AsyncSession, Depends(get_session)],
    cat_service: Annotated[CATService, Depends(get_cat_service)],
    clock_service: Annotated[SystemClockService, Depends(get_system_clock_service)],
) -> SubmitAnswerInteractor:
    """Constructs the interactor for submitting an answer."""
    session_repo = SQLPlacementTestSessionRepository(db_session)
    config_repo = SQLAssessmentConfigRepository(db_session)
    uow = SQLAlchemyUnitOfWork(db_session)

    return SubmitAnswerInteractor(
        session_repo, config_repo, cat_service, clock_service, uow
    )
