from datetime import datetime
from typing import List, Protocol, Optional, Tuple 
from app.domain.entities import PlacementTestSession, PlacementTestResponse, AssessmentConfig
from app.domain.value_objects import PlacementTestItem

class PlacementTestSessionRepositoryPort(Protocol):
    """Port for placement test session repository operations."""
    
    async def get_session(self, session_id: str) -> Optional[PlacementTestSession]:
        """Get a placement test session by ID."""
        ...

    async def get_active_session_by_test_taker_id(self, test_taker_id: str) -> Optional[PlacementTestSession]:
        """Get an active placement test session by test taker ID."""
        ...
    
    async def save_session(self, session: PlacementTestSession) -> None:
        """Save a placement test session."""
        ...

    async def create_session(self, session: PlacementTestSession) -> PlacementTestSession:
        """Create a placement test session."""
        ...
    
    async def get_item(self, item_id: str) -> Optional[PlacementTestItem]:
        """Get a placement test item by ID."""
        ...
    
    async def save_response(self, response: PlacementTestResponse) -> None:
        """Save a placement test response."""
        ...

    async def create_response(self, response: PlacementTestResponse) -> PlacementTestResponse:
        """Save a placement test response."""
        ...
    
    async def get_items_by_skill_areas(
        self, 
        skill_areas: List[str], 
        used_item_ids: List[str]
    ) -> List[PlacementTestItem]:
        """Get available items for specific skill areas that haven't been used."""
        ...
    
    async def get_session_responses(self, session_id: str) -> List[PlacementTestResponse]:
        """Get all responses for a session."""
        ...

    async def get_pending_response(self, session_id: str) -> Optional[PlacementTestResponse]:
        """Get the pending response for a session."""
        ...


class AssessmentConfigRepositoryPort(Protocol):
    """Port for assessment configuration operations."""
    
    async def get_config(self, config_id: str) -> Optional[AssessmentConfig]:
        """Get assessment configuration by ID."""
        ...
    
    async def get_config_by_type_and_pathway(
        self, 
        assessment_type: str, 
        learning_pathway_id: str
    ) -> Optional[AssessmentConfig]:
        """Get assessment configuration by type and learning pathway."""
        ...

    async def get_default_config(self) -> Optional[AssessmentConfig]:
        """Get the default assessment configuration."""
        ...



class ClockService(Protocol):
    """Port: Time provider (for testability)"""
    def now(self) -> datetime:
        ...




class UnitOfWork(Protocol):
    """Unit of Work protocol for transaction management."""
    
    async def __aenter__(self):
        """Enter async context manager."""
        ...
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context manager, committing or rolling back as appropriate."""
        ...
    
    async def commit(self):
        """Commit the current transaction."""
        ...
    
    async def rollback(self):
        """Rollback the current transaction."""
        ...