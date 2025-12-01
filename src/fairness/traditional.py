"""
Traditional fairness metrics: Jain Index, Alpha-fairness, Gini Coefficient.

Standard fairness metrics used in network resource allocation.
"""

from typing import List, Dict, Optional
import numpy as np


class TraditionalFairness:
    """
    Traditional fairness metrics for resource allocation.
    
    Implements:
    - Jain Index
    - Alpha-fairness (proportional, max-min)
    - Gini Coefficient
    """
    
    @staticmethod
    def jain_index(allocations: List[float]) -> float:
        """
        Compute Jain's Fairness Index.
        
        J = (Σx_i)² / (n * Σx_i²)
        
        Range: [1/n, 1.0]
        - 1.0 = perfectly fair (all equal)
        - 1/n = perfectly unfair (one gets all)
        
        Args:
            allocations: List of allocated resources
            
        Returns:
            Jain Index (0.0 to 1.0)
        """
        if not allocations or len(allocations) == 0:
            return 0.0
        
        allocations = np.array(allocations)
        allocations = allocations[allocations >= 0]  # Filter negative
        
        if len(allocations) == 0:
            return 0.0
        
        if np.sum(allocations) == 0:
            return 0.0
        
        numerator = np.sum(allocations) ** 2
        denominator = len(allocations) * np.sum(allocations ** 2)
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    @staticmethod
    def alpha_fairness(
        allocations: List[float],
        alpha: float = 1.0
    ) -> float:
        """
        Compute Alpha-fairness utility.
        
        For α = 0: proportional fairness (log utility)
        For α = 1: max-min fairness
        For α → ∞: max-min fairness
        
        Args:
            allocations: List of allocated resources
            alpha: Fairness parameter (0.0 to ∞)
            
        Returns:
            Alpha-fairness utility
        """
        if not allocations or len(allocations) == 0:
            return 0.0
        
        allocations = np.array(allocations)
        allocations = allocations[allocations > 0]  # Only positive
        
        if len(allocations) == 0:
            return 0.0
        
        if alpha == 0:
            # Proportional fairness: sum of logs
            return np.sum(np.log(allocations))
        elif alpha == 1:
            # Max-min fairness: sum of logs (same as α=0)
            return np.sum(np.log(allocations))
        else:
            # General alpha-fairness
            return np.sum((allocations ** (1 - alpha)) / (1 - alpha))
    
    @staticmethod
    def gini_coefficient(allocations: List[float]) -> float:
        """
        Compute Gini Coefficient.
        
        G = (2 * Σ(i * x_i)) / (n * Σx_i) - (n + 1) / n
        
        Range: [0.0, 1.0]
        - 0.0 = perfectly equal
        - 1.0 = perfectly unequal
        
        Args:
            allocations: List of allocated resources
            
        Returns:
            Gini Coefficient (0.0 to 1.0)
        """
        if not allocations or len(allocations) == 0:
            return 1.0
        
        allocations = np.array(allocations)
        allocations = allocations[allocations >= 0]
        
        if len(allocations) == 0:
            return 1.0
        
        if np.sum(allocations) == 0:
            return 1.0
        
        # Sort allocations
        sorted_allocations = np.sort(allocations)
        n = len(sorted_allocations)
        
        # Compute Gini
        numerator = 2 * np.sum((np.arange(1, n + 1)) * sorted_allocations)
        denominator = n * np.sum(sorted_allocations)
        
        if denominator == 0:
            return 1.0
        
        gini = (numerator / denominator) - ((n + 1) / n)
        
        return max(0.0, min(1.0, gini))
    
    @staticmethod
    def compute_all_metrics(allocations: List[float]) -> Dict[str, float]:
        """
        Compute all traditional fairness metrics.
        
        Args:
            allocations: List of allocated resources
            
        Returns:
            Dictionary with all metrics
        """
        return {
            "jain_index": TraditionalFairness.jain_index(allocations),
            "alpha_fairness_0": TraditionalFairness.alpha_fairness(allocations, alpha=0.0),
            "alpha_fairness_1": TraditionalFairness.alpha_fairness(allocations, alpha=1.0),
            "gini_coefficient": TraditionalFairness.gini_coefficient(allocations)
        }
    
    @staticmethod
    def compute_per_operator(
        operator_allocations: Dict[str, List[float]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute fairness metrics per operator.
        
        Args:
            operator_allocations: Dictionary mapping operator_id to list of allocations
            
        Returns:
            Dictionary mapping operator_id to metrics
        """
        results = {}
        for operator_id, allocations in operator_allocations.items():
            results[operator_id] = TraditionalFairness.compute_all_metrics(allocations)
        return results

