"""
Static spectrum allocation policy.

Allocates resources based on fixed rules without adaptation.
"""

import numpy as np
from typing import List, Dict, Optional


class StaticPolicy:
    """
    Static allocation policy with fixed allocation rules.
    """
    
    def __init__(self, allocation_strategy: str = 'equal'):
        """
        Initialize static policy.
        
        Args:
            allocation_strategy: 'equal', 'proportional', or 'weighted'
        """
        self.strategy = allocation_strategy
    
    def allocate(self, demands: np.ndarray, 
                available_resources: float,
                weights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Allocate resources based on static policy.
        
        Args:
            demands: Array of user demands
            available_resources: Total available resources
            weights: Optional weights for weighted allocation
            
        Returns:
            Array of allocations
        """
        demands = np.array(demands)
        n = len(demands)
        
        if self.strategy == 'equal':
            # Equal allocation
            allocation = np.ones(n) * (available_resources / n)
        
        elif self.strategy == 'proportional':
            # Proportional to demand
            total_demand = np.sum(demands)
            if total_demand > 0:
                allocation = demands * (available_resources / total_demand)
            else:
                allocation = np.ones(n) * (available_resources / n)
        
        elif self.strategy == 'weighted':
            # Weighted allocation
            if weights is None:
                weights = np.ones(n)
            weights = np.array(weights)
            total_weight = np.sum(weights)
            if total_weight > 0:
                allocation = weights * (available_resources / total_weight)
            else:
                allocation = np.ones(n) * (available_resources / n)
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        # Ensure allocations don't exceed demands
        allocation = np.minimum(allocation, demands)
        
        return allocation

