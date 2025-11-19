"""
Load-adaptive spectrum allocation policy.

Adapts allocation strategy based on network load conditions.
"""

import numpy as np
from typing import List, Dict, Optional


class LoadAdaptivePolicy:
    """
    Load-adaptive allocation policy.
    
    Adjusts allocation strategy based on network load:
    - Light load: Maximize fairness
    - Moderate load: Balance fairness and efficiency
    - Heavy load: Prioritize efficiency
    """
    
    def __init__(self, light_load_threshold: float = 0.3,
                 heavy_load_threshold: float = 0.7):
        """
        Initialize load-adaptive policy.
        
        Args:
            light_load_threshold: Load below this is considered light
            heavy_load_threshold: Load above this is considered heavy
        """
        self.light_threshold = light_load_threshold
        self.heavy_threshold = heavy_load_threshold
    
    def allocate(self, demands: np.ndarray,
                available_resources: float,
                current_load: float,
                priorities: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Allocate resources adaptively based on load.
        
        Args:
            demands: Array of user demands
            available_resources: Total available resources
            current_load: Current network load (0-1)
            priorities: Optional user priorities
            
        Returns:
            Array of allocations
        """
        demands = np.array(demands)
        n = len(demands)
        
        if current_load < self.light_threshold:
            # Light load: maximize fairness (equal allocation)
            allocation = self._fair_allocation(demands, available_resources)
        
        elif current_load > self.heavy_threshold:
            # Heavy load: maximize efficiency (proportional to demand)
            allocation = self._efficient_allocation(demands, available_resources)
        
        else:
            # Moderate load: balance fairness and efficiency
            allocation = self._balanced_allocation(
                demands, available_resources, current_load, priorities
            )
        
        return allocation
    
    def _fair_allocation(self, demands: np.ndarray,
                        available_resources: float) -> np.ndarray:
        """Equal allocation for fairness."""
        n = len(demands)
        allocation = np.ones(n) * (available_resources / n)
        return np.minimum(allocation, demands)
    
    def _efficient_allocation(self, demands: np.ndarray,
                             available_resources: float) -> np.ndarray:
        """Proportional allocation for efficiency."""
        total_demand = np.sum(demands)
        if total_demand > 0:
            allocation = demands * (available_resources / total_demand)
        else:
            allocation = np.ones(len(demands)) * (available_resources / len(demands))
        return np.minimum(allocation, demands)
    
    def _balanced_allocation(self, demands: np.ndarray,
                            available_resources: float,
                            load: float,
                            priorities: Optional[np.ndarray]) -> np.ndarray:
        """Balance fairness and efficiency."""
        n = len(demands)
        
        # Blend factor: more efficient as load increases
        efficiency_weight = (load - self.light_threshold) / (
            self.heavy_threshold - self.light_threshold
        )
        
        # Fair allocation
        fair_alloc = np.ones(n) * (available_resources / n)
        
        # Efficient allocation
        total_demand = np.sum(demands)
        if total_demand > 0:
            efficient_alloc = demands * (available_resources / total_demand)
        else:
            efficient_alloc = fair_alloc
        
        # Blend
        allocation = (1 - efficiency_weight) * fair_alloc + efficiency_weight * efficient_alloc
        
        # Apply priorities if provided
        if priorities is not None:
            priorities = np.array(priorities)
            priority_weights = 1.0 + 0.5 * priorities  # Moderate boost
            total_weight = np.sum(priority_weights)
            allocation = allocation * priority_weights * (n / total_weight)
        
        # Ensure bounds
        allocation = np.minimum(allocation, demands)
        allocation = np.maximum(allocation, 0)
        
        # Normalize
        total_allocated = np.sum(allocation)
        if total_allocated > 0:
            allocation = allocation * (available_resources / total_allocated)
        
        return allocation

