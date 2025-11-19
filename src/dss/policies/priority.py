"""
Priority-based spectrum allocation policy.

Allocates resources based on user priorities with fairness constraints.
"""

import numpy as np
from typing import List, Dict, Optional


class PriorityPolicy:
    """
    Priority-based allocation policy.
    
    Allocates resources to high-priority users first, then lower priorities,
    while maintaining minimum fairness guarantees.
    """
    
    def __init__(self, min_fairness: float = 0.3, max_priority_boost: float = 2.0):
        """
        Initialize priority policy.
        
        Args:
            min_fairness: Minimum fairness threshold (0-1)
            max_priority_boost: Maximum allocation boost for high priority (multiplier)
        """
        self.min_fairness = min_fairness
        self.max_priority_boost = max_priority_boost
    
    def allocate(self, demands: np.ndarray,
                available_resources: float,
                priorities: np.ndarray) -> np.ndarray:
        """
        Allocate resources based on priorities.
        
        Args:
            demands: Array of user demands
            available_resources: Total available resources
            priorities: Array of user priorities (0-1, higher is better)
            
        Returns:
            Array of allocations
        """
        demands = np.array(demands)
        priorities = np.array(priorities)
        n = len(demands)
        
        # Normalize priorities to [0, 1]
        if np.max(priorities) > np.min(priorities):
            priorities_norm = (priorities - np.min(priorities)) / (
                np.max(priorities) - np.min(priorities)
            )
        else:
            priorities_norm = np.ones(n)
        
        # Compute priority weights
        # Higher priority gets more resources, but bounded by max_priority_boost
        priority_weights = 1.0 + (self.max_priority_boost - 1.0) * priorities_norm
        
        # Initial allocation based on priority weights
        total_weight = np.sum(priority_weights)
        allocation = priority_weights * (available_resources / total_weight)
        
        # Ensure allocations don't exceed demands
        allocation = np.minimum(allocation, demands)
        
        # Check fairness constraint
        fairness = self._compute_fairness(allocation)
        
        if fairness < self.min_fairness:
            # Redistribute to improve fairness while respecting priorities
            allocation = self._balance_fairness_priority(
                allocation, demands, priorities, available_resources
            )
        
        return allocation
    
    def _compute_fairness(self, allocations: np.ndarray) -> float:
        """
        Compute fairness metric (Jain's index).
        
        Args:
            allocations: Array of allocations
            
        Returns:
            Fairness score [0, 1]
        """
        allocations = allocations[allocations >= 0]
        if len(allocations) == 0:
            return 0.0
        
        sum_alloc = np.sum(allocations)
        if sum_alloc == 0:
            return 0.0
        
        n = len(allocations)
        jain = (sum_alloc ** 2) / (n * np.sum(allocations ** 2))
        return jain
    
    def _balance_fairness_priority(self, allocation: np.ndarray,
                                  demands: np.ndarray,
                                  priorities: np.ndarray,
                                  available_resources: float) -> np.ndarray:
        """
        Balance fairness and priority constraints.
        
        Args:
            allocation: Current allocation
            demands: User demands
            priorities: User priorities
            available_resources: Total available resources
            
        Returns:
            Balanced allocation
        """
        n = len(allocation)
        
        # Start with equal allocation
        equal_alloc = np.ones(n) * (available_resources / n)
        equal_alloc = np.minimum(equal_alloc, demands)
        
        # Blend with priority-based allocation
        alpha = self.min_fairness  # Blend factor
        balanced = alpha * equal_alloc + (1 - alpha) * allocation
        
        # Normalize to use all available resources
        total_allocated = np.sum(balanced)
        if total_allocated > 0:
            balanced = balanced * (available_resources / total_allocated)
        
        # Ensure within bounds
        balanced = np.minimum(balanced, demands)
        balanced = np.maximum(balanced, 0)
        
        return balanced

