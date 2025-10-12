from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from sqlalchemy.sql import func
from sqlalchemy import func as sql_func
import uuid

from app.application.ports import (
    PlacementTestSessionRepositoryPort,
    AssessmentConfigRepositoryPort
)
from app.domain.entities import (
    PlacementTestSession,  PlacementTestResponse, AssessmentConfig
)

from app.domain.value_objects import PlacementTestItem
from app.infrastructure.persistence.models.assessment import (
    AssessmentSession as SQLAssessmentSession,
    AssessmentItem as SQLAssessmentItem,
    AssessmentResponse as SQLAssessmentResponse,
    AssessmentConfig as SQLAssessmentConfig,
)
from app.infrastructure.utils import naive_to_utc_aware

class SQLPlacementTestSessionRepository(PlacementTestSessionRepositoryPort):
    """SQL implementation of the placement test session repository."""
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_session(self, session_id: str) -> Optional[PlacementTestSession]:
        """Get a placement test session by ID."""
        result = await self.session.execute(
            select(SQLAssessmentSession).where(SQLAssessmentSession.id == session_id)
        )
        sql_session = result.scalar_one_or_none()
        
        if not sql_session:
            return None
        
        return PlacementTestSession(
            id=sql_session.id,
            test_taker_id=sql_session.test_taker_id,
            config_id=sql_session.config_id,
            current_ability=sql_session.current_ability,
            standard_error=sql_session.standard_error,
            questions_answered=sql_session.questions_answered,
            is_complete=sql_session.is_complete,
            started_at=naive_to_utc_aware(sql_session.started_at),
            completed_at=naive_to_utc_aware(sql_session.completed_at),
            expires_at=naive_to_utc_aware(sql_session.expires_at)
        )

    async def get_active_session_by_test_taker_id(self, test_taker_id: str) -> Optional[PlacementTestSession]:
        """Get an active placement test session by test taker ID."""
        result = await self.session.execute(
            select(SQLAssessmentSession).where(
                and_(
                    SQLAssessmentSession.test_taker_id == test_taker_id,
                    SQLAssessmentSession.is_complete == False,
                    SQLAssessmentSession.expires_at > func.now(),
                )
            )
        )
        sql_session = result.scalar_one_or_none()

        if not sql_session:
            return None

        return PlacementTestSession(
            id=sql_session.id,
            test_taker_id=sql_session.test_taker_id,
            config_id=sql_session.config_id,
            current_ability=sql_session.current_ability,
            standard_error=sql_session.standard_error,
            questions_answered=sql_session.questions_answered,
            is_complete=sql_session.is_complete,
            started_at=naive_to_utc_aware(sql_session.started_at),
            completed_at=naive_to_utc_aware(sql_session.completed_at),
            expires_at=naive_to_utc_aware(sql_session.expires_at)
        )
    
    async def save_session(self, session: PlacementTestSession) -> None:
        """Save a placement test session."""
        result = await self.session.execute(
            select(SQLAssessmentSession).where(SQLAssessmentSession.id == session.id)
        )
        existing = result.scalar_one_or_none()
        
        if existing:
            existing.current_ability = session.current_ability
            existing.standard_error = session.standard_error
            existing.questions_answered = session.questions_answered
            existing.is_complete = session.is_complete
            existing.completed_at = session.completed_at
 

    async def create_session(self, session: PlacementTestSession) -> PlacementTestSession:
        """Create a new placement test session."""
        session.id = str(uuid.uuid4())
        sql_session = SQLAssessmentSession(
            id=session.id,
            test_taker_id=session.test_taker_id,
            config_id=session.config_id,
            current_ability=session.current_ability,
            standard_error=session.standard_error,
            questions_answered=session.questions_answered,
            is_complete=session.is_complete,
            started_at=session.started_at,
            completed_at=session.completed_at,
            expires_at=session.expires_at
        )
        self.session.add(sql_session)
        return session
    

    async def get_item(self, item_id: str) -> Optional[PlacementTestItem]:
        """Get a placement test item by ID."""
        result = await self.session.execute(
            select(SQLAssessmentItem).where(SQLAssessmentItem.id == item_id)
        )
        sql_item = result.scalar_one_or_none()
        
        if not sql_item:
            return None
        
        return PlacementTestItem(
            id=sql_item.id,
            content=sql_item.content,
            item_type=sql_item.item_type,
            skill_area=sql_item.skill_area,
            target_proficiency_level=sql_item.target_proficiency_level,
            parameters=sql_item.parameters
        )
    
    async def save_response(self, response: PlacementTestResponse) -> None:
        """Save a placement test response."""
        result = await self.session.execute(
            select(SQLAssessmentResponse).where(SQLAssessmentResponse.id == response.id)
        )
        existing = result.scalar_one_or_none()
  
        if existing:
            existing.response_data = response.response_data
            existing.is_correct = response.is_correct
            existing.raw_score = response.raw_score
            existing.submitted_at = response.submitted_at
            existing.time_taken = response.time_taken
        

    async def create_response(self, response: PlacementTestResponse) -> PlacementTestResponse:
        """Create a placement test response."""
        response.id = str(uuid.uuid4())
        sql_response = SQLAssessmentResponse(
            id=response.id,
            session_id=response.session_id,
            item_id=response.item_id,
            response_data=response.response_data,
            is_correct=response.is_correct,
            raw_score=response.raw_score,
            presented_at=response.presented_at,
            submitted_at=response.submitted_at,
            time_taken=response.time_taken
        )
        self.session.add(sql_response)
        return response
    
    async def get_items_by_skill_areas(
        self, 
        skill_areas: List[str], 
        used_item_ids: List[str]
    ) -> List[PlacementTestItem]:
        """Get available items for specific skill areas that haven't been used."""
        skill_areas_json = sql_func.JSON_ARRAY(*skill_areas)
        
        conditions = [
            sql_func.JSON_OVERLAPS(SQLAssessmentItem.skill_area, skill_areas_json) == True,
            SQLAssessmentItem.is_active == True,
        ]
        if used_item_ids:
            conditions.append(~SQLAssessmentItem.id.in_(used_item_ids))

        query = select(SQLAssessmentItem).where(and_(*conditions))
        result = await self.session.execute(query)
        sql_items = result.scalars().all()
        
        items = []
        for sql_item in sql_items:
            items.append(PlacementTestItem(
                id=sql_item.id,
                content=sql_item.content,
                item_type=sql_item.item_type,
                skill_area=sql_item.skill_area,
                target_proficiency_level=sql_item.target_proficiency_level,
                parameters=sql_item.parameters
            ))
        
        return items
    
    async def get_session_responses(self, session_id: str) -> List[PlacementTestResponse]:
        """Get all responses for a session."""
        result = await self.session.execute(
            select(SQLAssessmentResponse)
            .where(
                and_(
                    SQLAssessmentResponse.session_id == session_id,
                    SQLAssessmentResponse.submitted_at.isnot(None)
                )
            )
            .order_by(SQLAssessmentResponse.presented_at)
        )
        sql_responses = result.scalars().all()
        
        responses = []
        for sql_response in sql_responses:
            responses.append(PlacementTestResponse(
                id=sql_response.id,
                session_id=sql_response.session_id,
                item_id=sql_response.item_id,
                response_data=sql_response.response_data,
                is_correct=sql_response.is_correct,
                raw_score=sql_response.raw_score,
                presented_at=naive_to_utc_aware(sql_response.presented_at),
                submitted_at=naive_to_utc_aware(sql_response.submitted_at),
                time_taken=sql_response.time_taken
            ))
        
        return responses

    async def get_pending_response(self, session_id: str) -> Optional[PlacementTestResponse]:
        """Get the pending response for a session."""
        result = await self.session.execute(
            select(SQLAssessmentResponse)
            .where(
                and_(
                    SQLAssessmentResponse.session_id == session_id,
                    SQLAssessmentResponse.submitted_at.is_(None),
                )
            )
            .order_by(SQLAssessmentResponse.presented_at.asc())
        )
        sql_response = result.scalar_one_or_none()

        if not sql_response:
            return None

        return PlacementTestResponse(
            id=sql_response.id,
            session_id=sql_response.session_id,
            item_id=sql_response.item_id,
            response_data=sql_response.response_data,
            is_correct=sql_response.is_correct,
            raw_score=sql_response.raw_score,
            presented_at=naive_to_utc_aware(sql_response.presented_at),
            submitted_at=naive_to_utc_aware(sql_response.submitted_at),
            time_taken=sql_response.time_taken
        )


class SQLAssessmentConfigRepository(AssessmentConfigRepositoryPort):
    """SQL implementation of the assessment config repository."""
    def __init__(self, session: AsyncSession):
        self.session = session
    async def get_config(self, config_id: str) -> Optional[AssessmentConfig]:
        """Get assessment configuration by ID."""
        result = await self.session.execute(
            select(SQLAssessmentConfig).where(SQLAssessmentConfig.id == config_id)
        )
        sql_config = result.scalar_one_or_none()
        
        if not sql_config:
            return None
        
        return AssessmentConfig(
            id=sql_config.id,
            learning_pathway_id=sql_config.learning_pathway_id,
            name=sql_config.name,
            assessment_type=sql_config.assessment_type,
            starting_ability=float( sql_config.starting_ability ),
            max_questions=sql_config.max_questions,
            min_questions=sql_config.min_questions,
            stopping_criterion=sql_config.stopping_criterion,
            skill_areas=sql_config.skill_areas,
            proficiency_range=sql_config.proficiency_range,
            is_active=sql_config.is_active
        )
    
    async def get_config_by_type_and_pathway(
        self, 
        assessment_type: str, 
        learning_pathway_id: str
    ) -> Optional[AssessmentConfig]:
        """Get assessment configuration by type and learning pathway."""
        result = await self.session.execute(
            select(SQLAssessmentConfig).where(
                and_(
                    SQLAssessmentConfig.assessment_type == assessment_type,
                    SQLAssessmentConfig.learning_pathway_id == learning_pathway_id,
                    SQLAssessmentConfig.is_active == True
                )
            )
        )
        sql_config = result.scalar_one_or_none()
        
        if not sql_config:
            return None
        
        return AssessmentConfig(
            id=sql_config.id,
            learning_pathway_id=sql_config.learning_pathway_id,
            name=sql_config.name,
            assessment_type=sql_config.assessment_type,
            starting_ability=float(sql_config.starting_ability),
            max_questions=sql_config.max_questions,
            min_questions=sql_config.min_questions,
            stopping_criterion=sql_config.stopping_criterion,
            skill_areas=sql_config.skill_areas,
            proficiency_range=sql_config.proficiency_range,
            is_active=sql_config.is_active
        )

    async def get_default_config(self) -> Optional[AssessmentConfig]:
        """Get the default assessment configuration."""
        result = await self.session.execute(
            select(SQLAssessmentConfig).where(SQLAssessmentConfig.name == "Default")
        )
        sql_config = result.scalar_one_or_none()

        if not sql_config:
            return None

        return AssessmentConfig(
            id=sql_config.id,
            learning_pathway_id=sql_config.learning_pathway_id,
            name=sql_config.name,
            assessment_type=sql_config.assessment_type,
            starting_ability=float(sql_config.starting_ability),
            max_questions=sql_config.max_questions,
            min_questions=sql_config.min_questions,
            stopping_criterion=sql_config.stopping_criterion,
            skill_areas=sql_config.skill_areas,
            proficiency_range=sql_config.proficiency_range,
            is_active=sql_config.is_active
        )
