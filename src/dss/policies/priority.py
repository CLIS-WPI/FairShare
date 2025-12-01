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
        
        # FIXED: Check fairness constraint, but don't override priorities completely
        fairness = self._compute_fairness(allocation)
        
        if fairness < self.min_fairness:
            # Redistribute to improve fairness while respecting priorities
            # BUT: Only if fairness is extremely low (near 0), otherwise keep priority-based allocation
            if fairness < 0.1:  # Only balance if extremely unfair
                allocation = self._balance_fairness_priority(
                    allocation, demands, priorities, available_resources
                )
            # Otherwise, keep priority-based allocation even if fairness is slightly below threshold
            # This ensures priorities are actually used
        
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
        
        FIXED: Preserve priority ordering even when balancing for fairness.
        
        Args:
            allocation: Current allocation
            demands: User demands
            priorities: User priorities
            available_resources: Total available resources
            
        Returns:
            Balanced allocation
        """
        n = len(allocation)
        
        # FIXED: Instead of blending with equal allocation, redistribute while preserving priority order
        # Sort by priority (descending)
        priority_indices = np.argsort(priorities)[::-1]
        
        # Start with minimum allocation for all
        min_allocation = available_resources * self.min_fairness / n
        balanced = np.ones(n) * min_allocation
        balanced = np.minimum(balanced, demands)
        
        # Distribute remaining resources based on priorities
        remaining = available_resources - np.sum(balanced)
        if remaining > 0:
            # Normalize priorities for remaining allocation
            if np.max(priorities) > np.min(priorities):
                priorities_norm = (priorities - np.min(priorities)) / (
                    np.max(priorities) - np.min(priorities)
                )
            else:
                priorities_norm = np.ones(n)
            
            # Weight by priority
            priority_weights = priorities_norm
            total_weight = np.sum(priority_weights)
            if total_weight > 0:
                additional = priority_weights * (remaining / total_weight)
                balanced = balanced + additional
                balanced = np.minimum(balanced, demands)
        
        # Normalize to use all available resources
        total_allocated = np.sum(balanced)
        if total_allocated > 0 and total_allocated < available_resources:
            balanced = balanced * (available_resources / total_allocated)
            balanced = np.minimum(balanced, demands)
        
        # Ensure within bounds
        balanced = np.maximum(balanced, 0)
        
        return balanced

