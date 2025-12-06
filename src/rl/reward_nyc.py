"""
Refined reward function for NYC large-scale scenario.

Implements priority satisfaction to ensure critical users (Op_C) never get dropped.
"""

from typing import Dict, List, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class RewardComponents:
    """Components of the refined reward function."""
    total_throughput: float
    priority_satisfaction: float  # Ensures Op_C (Critical) never gets dropped
    fairness_penalty: float  # Penalty for unfairness
    critical_drop_penalty: float  # Heavy penalty if critical users are dropped


def compute_reward_nyc(
    users: List[Dict],
    allocations: Dict[str, Optional],
    throughputs: Dict[str, float],
    jain_index: float,
    alpha: float = 0.4,
    beta: float = 0.4,
    gamma: float = 0.2
) -> float:
    """
    Compute refined reward for NYC scenario.
    
    Reward = (alpha * Total_Throughput) + (beta * Priority_Satisfaction) - (gamma * Fairness_Penalty)
    
    Where Priority_Satisfaction ensures Op_C (Critical) never gets dropped.
    
    Args:
        users: List of user dictionaries with 'id', 'operator', 'priority'
        allocations: Dictionary of allocations (user_id -> allocation or None)
        throughputs: Dictionary of throughputs (user_id -> throughput in bps)
        jain_index: Jain fairness index
        alpha: Weight for total throughput (default: 0.4)
        beta: Weight for priority satisfaction (default: 0.4)
        gamma: Weight for fairness penalty (default: 0.2)
    
    Returns:
        Reward value (higher is better)
    """
    # 1. Total Throughput (normalized)
    total_throughput = sum(throughputs.values()) / 1e9  # Convert to Gbps
    max_possible_throughput = len(users) * 100e6 / 1e9  # Assume 100 Mbps max per user
    throughput_normalized = min(total_throughput / max_possible_throughput, 1.0)
    
    # 2. Priority Satisfaction
    # Check if critical users (Op_C, priority=1.0) are allocated
    critical_users = [u for u in users if u.get('operator') == 'Op_C' or u.get('priority', 0) >= 1.0]
    critical_allocated = sum(1 for u in critical_users if allocations.get(u['id']) is not None)
    critical_total = len(critical_users)
    
    if critical_total > 0:
        priority_satisfaction = critical_allocated / critical_total
        # Heavy penalty if any critical user is dropped
        critical_drop_penalty = 0.0 if critical_allocated == critical_total else 0.5
    else:
        priority_satisfaction = 1.0
        critical_drop_penalty = 0.0
    
    # 3. Fairness Penalty (inverse of Jain index)
    # Lower Jain index = higher penalty
    fairness_penalty = 1.0 - jain_index
    
    # 4. Combined Reward
    reward = (
        alpha * throughput_normalized +
        beta * priority_satisfaction -
        gamma * fairness_penalty -
        critical_drop_penalty  # Heavy penalty for dropping critical users
    )
    
    return reward


def compute_reward_components(
    users: List[Dict],
    allocations: Dict[str, Optional],
    throughputs: Dict[str, float],
    jain_index: float
) -> RewardComponents:
    """
    Compute reward components separately for analysis.
    
    Returns:
        RewardComponents dataclass
    """
    total_throughput = sum(throughputs.values()) / 1e9
    
    critical_users = [u for u in users if u.get('operator') == 'Op_C' or u.get('priority', 0) >= 1.0]
    critical_allocated = sum(1 for u in critical_users if allocations.get(u['id']) is not None)
    critical_total = len(critical_users)
    
    if critical_total > 0:
        priority_satisfaction = critical_allocated / critical_total
        critical_drop_penalty = 0.0 if critical_allocated == critical_total else 0.5
    else:
        priority_satisfaction = 1.0
        critical_drop_penalty = 0.0
    
    fairness_penalty = 1.0 - jain_index
    
    return RewardComponents(
        total_throughput=total_throughput,
        priority_satisfaction=priority_satisfaction,
        fairness_penalty=fairness_penalty,
        critical_drop_penalty=critical_drop_penalty
    )

