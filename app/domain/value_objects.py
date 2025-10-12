from dataclasses import dataclass
from typing import Dict, List

@dataclass
class PlacementTestItem:
    """Value Object representing a placement test item/question with validation."""
    
    id: str
    content: Dict
    item_type: str
    skill_area: List[str]
    target_proficiency_level: str
    parameters: Dict[str, float] 
    is_active: bool = True
     
    @property
    def difficulty(self) -> float:
        """IRT parameter: difficulty of the item."""
        return self.parameters.get('difficulty', 0.0)
    
    @property
    def discrimination(self) -> float:
        """IRT parameter: discrimination of the item."""
        return self.parameters.get('discrimination', 1.0)
    
    @property
    def guessing(self) -> float:
        """IRT parameter: guessing parameter of the item."""
        return self.parameters.get('guessing', 0.25)



