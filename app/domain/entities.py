"""Domain entities for the placement assessment system with business rules and invariants."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional


@dataclass
class PlacementTestSession:
    """Domain entity representing an active placement test session with business rules."""
    
    id: str
    test_taker_id: str
    config_id: str
    current_ability: Decimal
    standard_error: Optional[Decimal]
    questions_answered: int
    is_complete: bool
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    expires_at: Optional[datetime]
    
    def can_accept_answer(self, now: datetime) -> bool:
        """Business rule: session must be active and not expired to accept answers."""
        if self.is_complete:
            return False
        if self.is_expired(now):
            return False
        return True

    def is_expired(self, now: datetime) -> bool:
        """Business rule: check if session has expired."""
        if self.started_at is None or self.expires_at is None:
            return False
        return now > self.expires_at
    
    def is_terminated(self, now: datetime) -> bool:
        """Business rule: session is terminated if complete or expired."""
        return self.is_complete or self.is_expired(now)
    
    def increment_questions_answered(self) -> None:
        """Business rule: update questions answered count."""
        self.questions_answered += 1
    
    def update_ability_estimate(self, new_ability: Decimal, new_standard_error: Optional[Decimal]) -> None:
        """Business rule: update ability and standard error after processing response."""
        self.current_ability = new_ability
        self.standard_error = new_standard_error
    
    def mark_complete(self, completed_at: datetime) -> None:
        """Business rule: mark session as complete."""
        self.is_complete = True
        self.completed_at = completed_at
    
    def has_reached_max_questions(self, max_questions: int) -> bool:
        """Business rule: check if session has reached maximum number of questions."""
        return self.questions_answered >= max_questions
    
    def has_sufficient_precision(self, min_standard_error: float) -> bool:
        """Business rule: check if standard error is below threshold (sufficient precision)."""
        if self.standard_error is None:
            return False
        return float(self.standard_error) <= min_standard_error
    
    def has_reached_min_questions(self, min_questions: int) -> bool:
        """Business rule: check if session has reached minimum number of questions."""
        return self.questions_answered >= min_questions




@dataclass
class PlacementTestResponse:
    """Domain entity representing a test-taker's response to a placement test item."""
    
    id: str
    session_id: str
    item_id: str
    response_data: Dict
    is_correct: Optional[bool]
    raw_score: Optional[Decimal]
    presented_at: Optional[datetime]
    submitted_at: Optional[datetime]
    time_taken: Optional[int]
    
    def has_valid_response(self, response) -> bool:
        """Business rule: response must have appropriate data."""
        return bool(response.get("selected_option"))
    
    def calculate_score(self) -> float:
        """Business rule: calculate the score for this response."""
        if self.raw_score is not None:
            return float(self.raw_score)
        return 1.0 if self.is_correct else 0.0
    
    def mark_as_submitted(self, submitted_at: datetime) -> None:
        """Business rule: mark response as submitted with provided timestamp."""
        self.submitted_at = submitted_at




@dataclass
class AssessmentConfig:
    """Domain entity for assessment configuration with business rules."""
    
    id: str
    learning_pathway_id: str
    name: str
    assessment_type: str
    starting_ability: float
    max_questions: Optional[int]
    min_questions: Optional[int]
    stopping_criterion: Dict[str, float] 
    skill_areas: List[str]
    proficiency_range: Optional[Dict[str, float]]
    is_active: bool
    
    def has_valid_question_limits(self) -> bool:
        """Business rule: config should have valid question limits."""
        if self.min_questions is not None and self.max_questions is not None:
            return self.min_questions <= self.max_questions
        return True
    
    def get_stopping_standard_error(self) -> float:
        """Business rule: get the standard error threshold for stopping."""
        return self.stopping_criterion.get("standard_error", 0.3)


