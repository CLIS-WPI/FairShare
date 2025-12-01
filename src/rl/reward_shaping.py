"""
Reward shaping for fairness-optimized RL training.

Custom reward functions that balance efficiency and fairness.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass

from src.fairness.traditional import TraditionalFairness
from src.fairness.vector_metrics import VectorFairness, MultiDimensionalMetrics


@dataclass
class RewardComponents:
    """Components of the reward function."""
    efficiency: float  # Spectral efficiency, throughput
    fairness: float  # Fairness score
    utilization: float  # Resource utilization
    qos_satisfaction: float  # QoS requirements met
    penalty: float  # Constraint violations
    
    def total(self, weights: Dict[str, float]) -> float:
        """Compute weighted total reward."""
        return (
            weights.get("efficiency", 0.3) * self.efficiency +
            weights.get("fairness", 0.3) * self.fairness +
            weights.get("utilization", 0.2) * self.utilization +
            weights.get("qos", 0.1) * self.qos_satisfaction -
            weights.get("penalty", 0.1) * self.penalty
        )


class FairnessRewardShaping:
    """
    Reward shaping for fairness-optimized RL.
    
    Provides various reward functions that balance efficiency
    and fairness in different ways.
    """
    
    def __init__(
        self,
        fairness_weight: float = 0.5,
        fairness_metric: str = "jain",
        use_vector_fairness: bool = False
    ):
        """
        Initialize reward shaper.
        
        Args:
            fairness_weight: Weight for fairness (0.0 to 1.0)
            fairness_metric: Type of fairness metric ("jain", "alpha", "gini")
            use_vector_fairness: Whether to use vector-based fairness
        """
        self.fairness_weight = fairness_weight
        self.efficiency_weight = 1.0 - fairness_weight
        self.fairness_metric = fairness_metric
        self.use_vector_fairness = use_vector_fairness
        
        if use_vector_fairness:
            self.vector_fairness = VectorFairness()
    
    def compute_reward(
        self,
        allocations: np.ndarray,
        demands: np.ndarray,
        throughputs: Optional[np.ndarray] = None,
        latencies: Optional[np.ndarray] = None,
        qos_requirements: Optional[Dict] = None
    ) -> float:
        """
        Compute reward for given allocations.
        
        Args:
            allocations: Allocated bandwidth per user/operator
            demands: Demand per user/operator
            throughputs: Optional throughput per user
            latencies: Optional latency per user
            qos_requirements: Optional QoS requirements
            
        Returns:
            Reward value
        """
        components = self._compute_components(
            allocations, demands, throughputs, latencies, qos_requirements
        )
        
        weights = {
            "efficiency": self.efficiency_weight,
            "fairness": self.fairness_weight,
            "utilization": 0.1,
            "qos": 0.1,
            "penalty": 0.1
        }
        
        return components.total(weights)
    
    def _compute_components(
        self,
        allocations: np.ndarray,
        demands: np.ndarray,
        throughputs: Optional[np.ndarray],
        latencies: Optional[np.ndarray],
        qos_requirements: Optional[Dict]
    ) -> RewardComponents:
        """Compute individual reward components."""
        # Efficiency: spectral efficiency or throughput
        if throughputs is not None:
            efficiency = np.mean(throughputs) / 100.0  # Normalize
        else:
            # Use allocation efficiency
            satisfied_demand = np.minimum(allocations, demands)
            efficiency = np.sum(satisfied_demand) / (np.sum(demands) + 1e-6)
        
        # Fairness
        if self.use_vector_fairness and throughputs is not None and latencies is not None:
            # Vector-based fairness
            metrics = [
                MultiDimensionalMetrics(
                    throughput_mbps=t,
                    latency_ms=l,
                    access_rate=1.0 if a > 0 else 0.0,
                    coverage_quality=1.0,
                    qos_satisfaction=1.0
                )
                for t, l, a in zip(throughputs, latencies, allocations)
            ]
            fairness = self.vector_fairness.compute_weighted_fairness(metrics)
        else:
            # Traditional fairness
            if self.fairness_metric == "jain":
                fairness = TraditionalFairness.jain_index(allocations.tolist())
            elif self.fairness_metric == "alpha":
                fairness = TraditionalFairness.alpha_fairness(allocations.tolist(), alpha=1.0)
                fairness = fairness / len(allocations)  # Normalize
            elif self.fairness_metric == "gini":
                fairness = 1.0 - TraditionalFairness.gini_coefficient(allocations.tolist())
            else:
                fairness = TraditionalFairness.jain_index(allocations.tolist())
        
        # Utilization
        total_available = np.sum(allocations) + np.sum(demands - allocations)
        utilization = np.sum(allocations) / (total_available + 1e-6)
        
        # QoS satisfaction
        qos_satisfaction = 1.0
        if qos_requirements:
            # Check if QoS requirements are met
            if throughputs is not None:
                min_throughput = qos_requirements.get("min_throughput_mbps", 0.0)
                satisfied = np.sum(throughputs >= min_throughput) / len(throughputs)
                qos_satisfaction = satisfied
        
        # Penalty: constraint violations
        penalty = 0.0
        # Penalize if allocations exceed demands significantly
        excess = np.maximum(0, allocations - demands * 1.1)
        penalty = np.sum(excess) / (np.sum(demands) + 1e-6)
        
        return RewardComponents(
            efficiency=float(efficiency),
            fairness=float(fairness),
            utilization=float(utilization),
            qos_satisfaction=float(qos_satisfaction),
            penalty=float(penalty)
        )
    
    def compute_fairness_constraint_reward(
        self,
        allocations: np.ndarray,
        min_fairness: float = 0.7
    ) -> float:
        """
        Compute reward with fairness constraint.
        
        Penalizes if fairness falls below threshold.
        
        Args:
            allocations: Allocations
            min_fairness: Minimum required fairness
            
        Returns:
            Reward (negative if constraint violated)
        """
        fairness = TraditionalFairness.jain_index(allocations.tolist())
        
        if fairness < min_fairness:
            # Heavy penalty for constraint violation
            return -10.0 * (min_fairness - fairness)
        else:
            # Normal reward
            return self.compute_reward(allocations, allocations)
    
    def compute_pareto_reward(
        self,
        allocations: np.ndarray,
        throughputs: np.ndarray,
        fairness_weight: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Compute reward that explores Pareto frontier.
        
        Returns both efficiency and fairness for multi-objective optimization.
        
        Args:
            allocations: Allocations
            throughputs: Throughputs
            fairness_weight: Optional fairness weight (uses instance default if None)
            
        Returns:
            (efficiency_reward, fairness_reward)
        """
        efficiency = np.mean(throughputs) / 100.0
        fairness = TraditionalFairness.jain_index(allocations.tolist())
        
        return float(efficiency), float(fairness)

