"""
Fairness metrics for spectrum allocation evaluation.

Implements:
- Jain's fairness index
- α-Fairness (proportional, max-min)
- Gini coefficient
- Weighted fairness (multi-dimensional)
"""

import numpy as np
from typing import List, Dict, Optional


def jain_fairness_index(allocations: np.ndarray) -> float:
    """
    Compute Jain's fairness index.
    
    J = (Σx_i)² / (n * Σx_i²)
    
    Range: [1/n, 1] where 1 is perfectly fair.
    
    Args:
        allocations: Array of resource allocations
        
    Returns:
        Jain's fairness index
    """
    if len(allocations) == 0:
        return 0.0
    
    allocations = np.array(allocations)
    allocations = allocations[allocations >= 0]  # Remove negative
    
    if len(allocations) == 0:
        return 0.0
    
    sum_alloc = np.sum(allocations)
    if sum_alloc == 0:
        return 0.0
    
    n = len(allocations)
    jain = (sum_alloc ** 2) / (n * np.sum(allocations ** 2))
    
    return jain


def alpha_fairness(allocations: np.ndarray, alpha: float = 1.0) -> float:
    """
    Compute α-fairness utility.
    
    For α = 1: proportional fairness (log utility)
    For α → ∞: max-min fairness
    
    Args:
        allocations: Array of resource allocations
        alpha: Fairness parameter (α = 1 for proportional, α → ∞ for max-min)
        
    Returns:
        α-fairness utility value
    """
    allocations = np.array(allocations)
    allocations = allocations[allocations > 0]  # Only positive
    
    if len(allocations) == 0:
        return 0.0
    
    if alpha == 1.0:
        # Proportional fairness: sum of log
        return np.sum(np.log(allocations))
    elif alpha == 0:
        # Sum utility
        return np.sum(allocations)
    else:
        # General α-fairness
        return np.sum((allocations ** (1 - alpha)) / (1 - alpha))


def gini_coefficient(allocations: np.ndarray) -> float:
    """
    Compute Gini coefficient of inequality.
    
    Range: [0, 1] where 0 is perfect equality, 1 is maximum inequality.
    
    Args:
        allocations: Array of resource allocations
        
    Returns:
        Gini coefficient
    """
    allocations = np.array(allocations)
    allocations = allocations[allocations >= 0]
    
    if len(allocations) == 0:
        return 1.0
    
    n = len(allocations)
    if n == 1:
        return 0.0
    
    # Sort allocations
    sorted_alloc = np.sort(allocations)
    
    # Compute Gini (handle zero sum case)
    total_alloc = np.sum(sorted_alloc)
    if total_alloc == 0 or n == 0:
        return 1.0  # Maximum inequality when all zeros
    
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * sorted_alloc)) / (n * total_alloc) - (n + 1) / n
    
    return np.clip(gini, 0.0, 1.0)  # Ensure valid range


def max_min_fairness(allocations: np.ndarray) -> float:
    """
    Compute max-min fairness (minimum allocation).
    
    Args:
        allocations: Array of resource allocations
        
    Returns:
        Minimum allocation value
    """
    allocations = np.array(allocations)
    allocations = allocations[allocations >= 0]
    
    if len(allocations) == 0:
        return 0.0
    
    return np.min(allocations)


def weighted_fairness_score(allocations: np.ndarray,
                           demands: Optional[np.ndarray] = None,
                           priorities: Optional[np.ndarray] = None,
                           network_load: Optional[float] = None) -> float:
    """
    Compute weighted fairness score using multi-dimensional metrics.
    
    Args:
        allocations: Array of resource allocations
        demands: Array of user demands (optional)
        priorities: Array of user priorities (optional)
        network_load: Overall network load (optional)
        
    Returns:
        Weighted fairness score [0, 1]
    """
    allocations = np.array(allocations)
    
    # Compute current fairness metrics
    jain = jain_fairness_index(allocations)
    gini = gini_coefficient(allocations)
    
    # Convert Gini to fairness (1 - Gini)
    fairness_metric = 1.0 - gini
    
    # Estimate network load if not provided
    if network_load is None:
        if demands is not None:
            demands = np.array(demands)
            total_demand = np.sum(demands)
            total_allocated = np.sum(allocations)
            network_load = min(total_allocated / total_demand, 1.0) if total_demand > 0 else 0.0
        else:
            network_load = 0.5  # Default moderate load
    
    # Weighted combination: 70% Jain, 30% fairness metric
    weighted_score = 0.7 * jain + 0.3 * fairness_metric
    
    # Adjust based on network load (higher load -> lower fairness)
    load_factor = 1.0 - 0.2 * network_load  # Reduce by up to 20% at full load
    
    return weighted_score * load_factor


class FairnessEvaluator:
    """
    Comprehensive fairness evaluation using multiple metrics.
    """
    
    def __init__(self):
        """
        Initialize fairness evaluator.
        
        Uses traditional and vector-based fairness metrics.
        """
        # Removed: FuzzyInferenceSystem dependency
        # Using traditional and vector-based fairness metrics instead
    
    def evaluate(self, allocations: np.ndarray,
                demands: Optional[np.ndarray] = None,
                priorities: Optional[np.ndarray] = None,
                network_load: Optional[float] = None) -> Dict:
        """
        Evaluate fairness using all metrics.
        
        Args:
            allocations: Array of resource allocations
            demands: Array of user demands
            priorities: Array of user priorities
            network_load: Overall network load
            
        Returns:
            Dictionary with all fairness metrics
        """
        allocations = np.array(allocations)
        
        results = {
            'jain_index': jain_fairness_index(allocations),
            'gini_coefficient': gini_coefficient(allocations),
            'max_min_fairness': max_min_fairness(allocations),
            'proportional_fairness': alpha_fairness(allocations, alpha=1.0),
            'weighted_fairness_score': weighted_fairness_score(
                allocations, demands, priorities, network_load
            )
        }
        
        # Additional statistics
        results['mean_allocation'] = np.mean(allocations)
        results['std_allocation'] = np.std(allocations)
        results['min_allocation'] = np.min(allocations)
        results['max_allocation'] = np.max(allocations)
        results['coefficient_of_variation'] = (
            results['std_allocation'] / results['mean_allocation']
            if results['mean_allocation'] > 0 else 0.0
        )
        
        return results

