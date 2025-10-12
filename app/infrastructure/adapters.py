import numpy as np
from datetime import datetime, timezone
from typing import List, Tuple

from app.domain.ports import PsychometricModelInterface
from app.application.ports import ClockService
from app.domain.entities import PlacementTestResponse

from app.domain.value_objects import PlacementTestItem


class IRTServiceAdapter(PsychometricModelInterface):
    """
    Infrastructure adapter implementing PsychometricModelInterface using IRT calculations.
    This adapter contains the actual mathematical implementations of IRT functions.
    """
    
    def __init__(self):
        pass
    
    async def calculate_information(
        self, 
        ability: float, 
        item: PlacementTestItem
    ) -> float:
        """
        Calculate Fisher Information for 2PL model.
        """
        try:
            discrimination = float(item.discrimination)
            difficulty = float(item.difficulty)
            
            z = discrimination * (ability - difficulty)
            z = np.clip(z, -30, 30)
            
            if z >= 0:
                exp_neg_z = np.exp(-z)
                prob = 1 / (1 + exp_neg_z)
            else:
                exp_z = np.exp(z)
                prob = exp_z / (1 + exp_z)
            
            p_times_q = prob * (1 - prob)
            information = (discrimination ** 2) * p_times_q
            
            return max(0.0, float(information))
        except (OverflowError, ValueError) as e:
            raise ValueError(
                f"Information calculation failed for item with difficulty={item.difficulty}, "
                f"discrimination={item.discrimination}, ability={ability}. "
                f"Error: {str(e)}"
            )
    
    async def calculate_ability(
        self, 
        responses: List[PlacementTestResponse],
        items: List[PlacementTestItem]
    ) -> Tuple[float, float]:
        """
        Calculate ability estimate and standard error from responses and items using MAP estimation.
        """
        if len(responses) != len(items):
            raise ValueError("Number of responses must match number of items")
        
        if not responses or not items:
            return 0.0, 2.0  
        
        response_data = []
        item_params = []
        
        for response, item in zip(responses, items):
            score = response.calculate_score()
            response_data.append(float(score))
            item_params.append([float(item.discrimination), float(item.difficulty)])
        
        responses_array = np.array(response_data, dtype=np.float64)
        params_array = np.array(item_params, dtype=np.float64)
        
        return self._map_estimation(responses_array, params_array)
    
    async def estimate_ability_with_item(
        self,
        responses: List[PlacementTestResponse],
        items: List[PlacementTestItem],
        current_item: PlacementTestItem,
        current_response_score: float
    ) -> Tuple[float, float]:
        """
        Calculate new ability estimate including the current response.
        """
        response_data = []
        item_params = []
        
        for response, item in zip(responses, items):
            score = response.calculate_score()
            response_data.append(float(score))
            item_params.append([float(item.discrimination), float(item.difficulty)])
        
        response_data.append(current_response_score)
        item_params.append([
            float(current_item.discrimination),
            float(current_item.difficulty)
        ])
        
        responses_array = np.array(response_data, dtype=np.float64)
        params_array = np.array(item_params, dtype=np.float64)
        
        return self._map_estimation(responses_array, params_array)
    
    def _map_estimation(
        self, 
        responses: np.ndarray, 
        item_params: np.ndarray
    ) -> Tuple[float, float]:
        """
        Maximum A Posteriori (MAP) estimation using Newton-Raphson method for 2PL IRT.
        Uses a normal prior N(0, 1) to regularize ability estimates.
        """
        prior_mean = 0.0
        prior_variance = 1.0
        
        theta = prior_mean
        
        for iteration in range(50):
            first_deriv, second_deriv = 0.0, 0.0
            
            for response, params in zip(responses, item_params):
                a, b = params[0], params[1]
                z = a * (theta - b)
                z = np.clip(z, -30, 30)
                
                if z >= 0:
                    exp_neg_z = np.exp(-z)
                    p = 1 / (1 + exp_neg_z)
                else:
                    exp_z = np.exp(z)
                    p = exp_z / (1 + exp_z)
                
                q = 1 - p
                
                first_deriv += a * (response - p)
                
                second_deriv -= (a ** 2) * p * q
            
            prior_first_deriv = -(theta - prior_mean) / prior_variance
            prior_second_deriv = -1.0 / prior_variance
            
            total_first_deriv = first_deriv + prior_first_deriv
            total_second_deriv = second_deriv + prior_second_deriv

            if total_second_deriv >= 0:
                break

            theta_new = theta - (total_first_deriv / total_second_deriv)
            theta_new = np.clip(theta_new, -10, 10)  

            if abs(theta_new - theta) < 1e-6:
                theta = theta_new
                break
            
            theta = theta_new

        final_second_deriv = 0.0
        for response, params in zip(responses, item_params):
            a, b = params[0], params[1]
            z = a * (theta - b)
            z = np.clip(z, -30, 30)
            
            if z >= 0:
                exp_neg_z = np.exp(-z)
                p = 1 / (1 + exp_neg_z)
            else:
                exp_z = np.exp(z)
                p = exp_z / (1 + exp_z)
            
            q = 1 - p
            final_second_deriv -= (a ** 2) * p * q
        
        final_second_deriv += prior_second_deriv
        
        information = -final_second_deriv

        if information > 0:
            standard_error = 1.0 / np.sqrt(information)
            standard_error = np.clip(standard_error, 0.01, 2.0)
        else:
            standard_error = 2.0

        return float(theta), float(standard_error)




class SystemClockService(ClockService):
    """Adapter: Provides real system time"""
    def now(self) -> datetime:
        return datetime.now(timezone.utc);

