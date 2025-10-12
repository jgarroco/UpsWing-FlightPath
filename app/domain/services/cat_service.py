"""
CAT (Computerized Adaptive Testing) Domain Service

Implements the core business logic for adaptive testing using IRT (Item Response Theory).
This service orchestrates the adaptive testing flow but depends on the PsychometricModelInterface
for the actual psychometric calculations, following the dependency inversion principle.
"""
from typing import List, Optional, Tuple

from app.domain.entities import PlacementTestSession,  PlacementTestResponse, AssessmentConfig
from app.domain.value_objects import PlacementTestItem
from app.domain.ports import PsychometricModelInterface


class CATService:
    """
    Domain service implementing Computerized Adaptive Testing business logic.
    
    This service handles the core CAT algorithm workflow while depending on
    the PsychometricModelInterface for the actual IRT calculations.
    """
    
    def __init__(self, psychometric_model: PsychometricModelInterface):
        self.psychometric_model = psychometric_model

    async def select_next_question(
        self,
        ability: float,
        skill_areas: List[str],
        used_item_ids: List[str],
        available_items: List[PlacementTestItem]
    ) -> Optional[PlacementTestItem]:
        """
        Select the next most informative question based on current ability estimate.
        
        Args:
            ability: Current ability estimate
            skill_areas: Skill areas to focus on
            used_item_ids: IDs of items already presented
            available_items: Available items to choose from
            
        Returns:
            The next question to present, or None if no suitable items available
        """
        filtered_items = [
            item for item in available_items
            if any(skill in item.skill_area for skill in skill_areas)
            and item.id not in used_item_ids
            and item.is_active
        ]
        
        if not filtered_items:
            return None
        
        best_item = None
        max_information = -1
        
        for item in filtered_items:
            info = await self.psychometric_model.calculate_information(
                ability=ability,
                item=item
            )
            
            if info > max_information:
                max_information = info
                best_item = item
        
        return best_item

    async def calculate_ability_estimate(
        self,
        responses: List[PlacementTestResponse],
        items: List[PlacementTestItem]
    ) -> Tuple[float, float]:
        """
        Calculate the current ability estimate based on all responses.
        
        Args:
            responses: List of all the test-taker's responses
            items: List of all items that were answered
            
        Returns:
            Tuple of (ability_estimate, standard_error)
        """
        return await self.psychometric_model.calculate_ability(responses, items)

    async def process_response(
        self,
        responses: List[PlacementTestResponse],
        items: List[PlacementTestItem],
        current_item: PlacementTestItem,
        current_response_score: float
    ) -> Tuple[float, float]:
        """
        Process a new response and return updated ability estimate.
        
        Args:
            responses: Previous responses
            items: Items corresponding to previous responses
            current_item: The item that was just answered
            current_response_score: Score for the current response (0.0 or 1.0)
            
        Returns:
            Tuple of (new_ability_estimate, new_standard_error)
        """
        return await self.psychometric_model.estimate_ability_with_item(
            responses=responses,
            items=items,
            current_item=current_item,
            current_response_score=current_response_score
        )

    def check_termination_criteria(
        self,
        session: PlacementTestSession,
        config: AssessmentConfig,
    ) -> bool:
        """
        Check if the assessment should terminate based on configured criteria.
        
        Args:
            session: Current test session
            config: Assessment configuration
            current_standard_error: Current standard error of ability estimate
            
        Returns:
            True if assessment should terminate, False otherwise
        """
        min_questions = config.min_questions or 5
        max_questions = config.max_questions or 30
        stopping_se = config.get_stopping_standard_error()
        
        if not session.has_reached_min_questions(min_questions):
            return False
            
        if session.has_reached_max_questions(max_questions):
            return True
            
        if session.has_sufficient_precision(stopping_se):
            return True
            
        return False

    async def score_response(
        self,
        item: PlacementTestItem,
        response_data: dict
    ) -> Tuple[float, bool]:
        """
        Score a response to a test item.
        
        Args:
            item: The item being responded to
            response_data: The response data
            
        Returns:
            Tuple of (score, is_correct)
        """
        correct_answer = item.content.get("correct_answer")
        student_answer = response_data.get("selected_option")
        
        if correct_answer is None or student_answer is None:
            raise ValueError("Missing correct or student answer for scoring.")
        
        correct_str = str(correct_answer).strip().lower()
        student_str = str(student_answer).strip().lower()
        
        is_correct = correct_str == student_str
        score = 1.0 if is_correct else 0.0
        
        return score, is_correct
