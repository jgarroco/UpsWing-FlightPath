"""Mappers for converting between domain entities and DTOs"""
from typing import Dict, Any
from app.domain.value_objects import PlacementTestItem
from app.application.dto import (
    PlacementTestItemPublicDTO, PlacementTestItemPrivateDTO, 
    MultipleChoiceContent , ItemType, PlacementItemContentDTO
)


def map_placement_test_item_to_public_dto(item: PlacementTestItem) -> PlacementTestItemPublicDTO:
    """Map domain PlacementTestItem to public DTO (for client use, no sensitive data like correct answers)"""
    content_dto = _create_item_content_dto(item.content, item.item_type)
    
    return PlacementTestItemPublicDTO(
        id=item.id,
        content=content_dto,
        item_type=ItemType(item.item_type),
        skill_area=item.skill_area,
        target_proficiency_level=item.target_proficiency_level,
    )


def map_placement_test_item_to_private_dto(item: PlacementTestItem) -> PlacementTestItemPrivateDTO:
    """Map domain PlacementTestItem to private DTO (for evaluation, with sensitive data like correct answers)"""
    content_dto = _create_item_content_dto(item.content, item.item_type)
    
    correct_answer = item.content.get('correct_answer') if isinstance(item.content, dict) else None
        

    return PlacementTestItemPrivateDTO(
        id=item.id,
        content=content_dto,
        item_type=ItemType(item.item_type),
        skill_area=item.skill_area,
        target_proficiency_level=item.target_proficiency_level,
        parameters=item.parameters,
        correct_answer=correct_answer
    )


def _create_item_content_dto(content: Dict[str, Any], item_type: str) -> PlacementItemContentDTO:
    """Create appropriate content DTO based on item type, filtering out sensitive data"""
    safe_content = content.copy()
    
    item_type_enum = ItemType(item_type)
    
    if item_type_enum == ItemType.MULTIPLE_CHOICE:
        return MultipleChoiceContent(
            item=safe_content.get('item', ''),
            options=safe_content.get('options', []),
            instruction=safe_content.get('instruction', '')
        )
    else:
        return MultipleChoiceContent(
            item=safe_content.get('item', ''),
            options=safe_content.get('options', []),
            instruction=safe_content.get('instruction', '')
        )


