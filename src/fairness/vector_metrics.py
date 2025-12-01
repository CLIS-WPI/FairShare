"""
Vector-based fairness metrics.

Multi-dimensional fairness evaluation considering throughput, latency,
access, coverage, and QoS simultaneously.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class MultiDimensionalMetrics:
    """Multi-dimensional performance metrics for a user/operator."""
    throughput_mbps: float
    latency_ms: float
    access_rate: float  # 0.0 to 1.0 (fraction of time with service)
    coverage_quality: float  # 0.0 to 1.0 (signal quality)
    qos_satisfaction: float  # 0.0 to 1.0 (QoS requirements met)
    
    def to_vector(self) -> np.ndarray:
        """Convert to normalized vector."""
        # Normalize each dimension to [0, 1]
        # Throughput: normalize by max expected (e.g., 100 Mbps)
        norm_throughput = min(1.0, self.throughput_mbps / 100.0)
        
        # Latency: normalize by max acceptable (e.g., 100 ms)
        # Lower latency is better, so invert
        norm_latency = max(0.0, 1.0 - (self.latency_ms / 100.0))
        
        return np.array([
            norm_throughput,
            norm_latency,
            self.access_rate,
            self.coverage_quality,
            self.qos_satisfaction
        ])


class VectorFairness:
    """
    Vector-based fairness metrics.
    
    Evaluates fairness across multiple dimensions simultaneously,
    providing a more comprehensive view than single-dimensional metrics.
    """
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize vector fairness calculator.
        
        Args:
            weights: Optional weights for each dimension
                    Default: equal weights
        """
        if weights is None:
            weights = {
                "throughput": 0.2,
                "latency": 0.2,
                "access": 0.2,
                "coverage": 0.2,
                "qos": 0.2
            }
        
        self.weights = weights
        self.dimension_names = ["throughput", "latency", "access", "coverage", "qos"]
    
    def compute_fairness_vector(
        self,
        metrics_list: List[MultiDimensionalMetrics]
    ) -> np.ndarray:
        """
        Compute fairness vector across all dimensions.
        
        For each dimension, compute a fairness score (e.g., Jain Index),
        then combine into a vector.
        
        Args:
            metrics_list: List of multi-dimensional metrics
            
        Returns:
            Fairness vector [fairness_throughput, fairness_latency, ...]
        """
        if not metrics_list:
            return np.zeros(5)
        
        # Extract each dimension
        dimensions = {
            "throughput": [m.throughput_mbps for m in metrics_list],
            "latency": [m.latency_ms for m in metrics_list],
            "access": [m.access_rate for m in metrics_list],
            "coverage": [m.coverage_quality for m in metrics_list],
            "qos": [m.qos_satisfaction for m in metrics_list]
        }
        
        # Compute Jain Index for each dimension
        from src.fairness.traditional import TraditionalFairness
        
        fairness_vector = np.array([
            TraditionalFairness.jain_index(dimensions["throughput"]),
            # For latency: lower is better, so we use inverse
            TraditionalFairness.jain_index([100.0 - l for l in dimensions["latency"]]),
            TraditionalFairness.jain_index(dimensions["access"]),
            TraditionalFairness.jain_index(dimensions["coverage"]),
            TraditionalFairness.jain_index(dimensions["qos"])
        ])
        
        return fairness_vector
    
    def compute_weighted_fairness(
        self,
        metrics_list: List[MultiDimensionalMetrics]
    ) -> float:
        """
        Compute weighted scalar fairness score.
        
        Args:
            metrics_list: List of multi-dimensional metrics
            
        Returns:
            Weighted fairness score (0.0 to 1.0)
        """
        fairness_vector = self.compute_fairness_vector(metrics_list)
        
        # Weighted sum
        weights_array = np.array([
            self.weights.get("throughput", 0.2),
            self.weights.get("latency", 0.2),
            self.weights.get("access", 0.2),
            self.weights.get("coverage", 0.2),
            self.weights.get("qos", 0.2)
        ])
        
        return np.dot(fairness_vector, weights_array)
    
    def compute_distance_fairness(
        self,
        metrics_list: List[MultiDimensionalMetrics]
    ) -> float:
        """
        Compute fairness based on distance from ideal (equal) distribution.
        
        Lower distance = more fair.
        
        Args:
            metrics_list: List of multi-dimensional metrics
            
        Returns:
            Distance-based fairness score (0.0 to 1.0, higher = more fair)
        """
        if not metrics_list:
            return 0.0
        
        # Convert to vectors
        vectors = [m.to_vector() for m in metrics_list]
        vectors_array = np.array(vectors)
        
        # Ideal: all users have the same vector (mean)
        ideal_vector = np.mean(vectors_array, axis=0)
        
        # Compute distances from ideal
        distances = [
            np.linalg.norm(v - ideal_vector) for v in vectors
        ]
        
        # Fairness: inverse of average distance (normalized)
        max_distance = np.max(distances) if distances else 1.0
        if max_distance == 0:
            return 1.0
        
        avg_distance = np.mean(distances)
        fairness = 1.0 - (avg_distance / max_distance)
        
        return max(0.0, min(1.0, fairness))
    
    def compute_per_operator_fairness(
        self,
        operator_metrics: Dict[str, List[MultiDimensionalMetrics]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute fairness metrics per operator.
        
        Args:
            operator_metrics: Dictionary mapping operator_id to metrics list
            
        Returns:
            Dictionary with fairness scores per operator
        """
        results = {}
        
        for operator_id, metrics_list in operator_metrics.items():
            fairness_vector = self.compute_fairness_vector(metrics_list)
            weighted_fairness = self.compute_weighted_fairness(metrics_list)
            distance_fairness = self.compute_distance_fairness(metrics_list)
            
            results[operator_id] = {
                "fairness_vector": fairness_vector.tolist(),
                "weighted_fairness": weighted_fairness,
                "distance_fairness": distance_fairness,
                "dimension_names": self.dimension_names
            }
        
        return results

