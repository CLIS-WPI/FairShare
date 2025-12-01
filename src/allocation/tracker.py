"""
Resource tracking for multi-operator allocation.

Tracks per-user and per-operator resource usage, performance metrics,
and allocation history.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

from src.allocation.engine import AllocationResult


@dataclass
class UserMetrics:
    """Metrics for a single user."""
    user_id: str
    operator_id: str
    
    # Allocation metrics
    allocated_bandwidth_mhz: float = 0.0
    allocated_frequency_mhz: float = 0.0
    
    # Performance metrics
    throughput_mbps: float = 0.0
    latency_ms: float = 0.0
    packet_loss_rate: float = 0.0
    
    # QoS metrics
    qos_satisfaction: float = 0.0  # 0.0 to 1.0
    
    # Fairness metrics
    fairness_score: float = 0.0
    
    # History
    allocation_history: List[Tuple[datetime, float]] = field(default_factory=list)
    throughput_history: List[Tuple[datetime, float]] = field(default_factory=list)


@dataclass
class OperatorMetrics:
    """Metrics for an operator."""
    operator_id: str
    
    # Resource metrics
    total_allocated_bandwidth_mhz: float = 0.0
    total_available_bandwidth_mhz: float = 0.0
    utilization: float = 0.0
    
    # User metrics
    num_users: int = 0
    num_served_users: int = 0
    service_rate: float = 0.0  # served / total
    
    # Performance metrics
    avg_throughput_mbps: float = 0.0
    avg_latency_ms: float = 0.0
    
    # Fairness metrics
    fairness_scores: Dict[str, float] = field(default_factory=dict)
    
    # Efficiency metrics
    spectral_efficiency_mbps_per_mhz: float = 0.0


class ResourceTracker:
    """
    Tracks resource allocation and performance metrics.
    
    Maintains per-user and per-operator metrics, allocation history,
    and computes aggregate statistics.
    """
    
    def __init__(self):
        """Initialize resource tracker."""
        self.user_metrics: Dict[str, UserMetrics] = {}
        self.operator_metrics: Dict[str, OperatorMetrics] = {}
        
        # Allocation history
        self.allocation_history: List[Tuple[datetime, List[AllocationResult]]] = []
    
    def update_allocations(
        self,
        results: List[AllocationResult],
        timestamp: datetime
    ):
        """
        Update tracker with new allocation results.
        
        Args:
            results: List of allocation results
            timestamp: Current timestamp
        """
        # Store in history
        self.allocation_history.append((timestamp, results))
        
        # Update user metrics
        for result in results:
            user_key = f"{result.operator_id}_{result.user_id}"
            
            if user_key not in self.user_metrics:
                self.user_metrics[user_key] = UserMetrics(
                    user_id=result.user_id,
                    operator_id=result.operator_id
                )
            
            user_metrics = self.user_metrics[user_key]
            user_metrics.allocated_bandwidth_mhz = result.allocated_bandwidth_mhz
            user_metrics.allocated_frequency_mhz = result.allocated_frequency_mhz
            
            # Store in history
            user_metrics.allocation_history.append(
                (timestamp, result.allocated_bandwidth_mhz)
            )
        
        # Update operator metrics
        self._update_operator_metrics()
    
    def update_user_performance(
        self,
        operator_id: str,
        user_id: str,
        throughput_mbps: float,
        latency_ms: float,
        packet_loss_rate: float = 0.0,
        qos_satisfaction: float = 0.0
    ):
        """
        Update performance metrics for a user.
        
        Args:
            operator_id: Operator ID
            user_id: User ID
            throughput_mbps: Throughput in Mbps
            latency_ms: Latency in ms
            packet_loss_rate: Packet loss rate (0.0 to 1.0)
            qos_satisfaction: QoS satisfaction (0.0 to 1.0)
        """
        user_key = f"{operator_id}_{user_id}"
        
        if user_key not in self.user_metrics:
            self.user_metrics[user_key] = UserMetrics(
                user_id=user_id,
                operator_id=operator_id
            )
        
        user_metrics = self.user_metrics[user_key]
        user_metrics.throughput_mbps = throughput_mbps
        user_metrics.latency_ms = latency_ms
        user_metrics.packet_loss_rate = packet_loss_rate
        user_metrics.qos_satisfaction = qos_satisfaction
        
        # Store in history
        from datetime import datetime
        user_metrics.throughput_history.append(
            (datetime.now(), throughput_mbps)
        )
    
    def _update_operator_metrics(self):
        """Update aggregate operator metrics."""
        # Group users by operator
        operator_users: Dict[str, List[UserMetrics]] = defaultdict(list)
        for user_metrics in self.user_metrics.values():
            operator_users[user_metrics.operator_id].append(user_metrics)
        
        # Compute operator metrics
        for operator_id, users in operator_users.items():
            if operator_id not in self.operator_metrics:
                self.operator_metrics[operator_id] = OperatorMetrics(
                    operator_id=operator_id
                )
            
            op_metrics = self.operator_metrics[operator_id]
            
            # Resource metrics
            op_metrics.total_allocated_bandwidth_mhz = sum(
                u.allocated_bandwidth_mhz for u in users
            )
            
            # User metrics
            op_metrics.num_users = len(users)
            op_metrics.num_served_users = sum(
                1 for u in users if u.allocated_bandwidth_mhz > 0
            )
            op_metrics.service_rate = (
                op_metrics.num_served_users / op_metrics.num_users
                if op_metrics.num_users > 0 else 0.0
            )
            
            # Performance metrics
            served_users = [u for u in users if u.allocated_bandwidth_mhz > 0]
            if served_users:
                op_metrics.avg_throughput_mbps = np.mean(
                    [u.throughput_mbps for u in served_users]
                )
                op_metrics.avg_latency_ms = np.mean(
                    [u.latency_ms for u in served_users]
                )
            else:
                op_metrics.avg_throughput_mbps = 0.0
                op_metrics.avg_latency_ms = 0.0
            
            # Efficiency
            if op_metrics.total_allocated_bandwidth_mhz > 0:
                op_metrics.spectral_efficiency_mbps_per_mhz = (
                    op_metrics.avg_throughput_mbps / op_metrics.total_allocated_bandwidth_mhz
                )
            else:
                op_metrics.spectral_efficiency_mbps_per_mhz = 0.0
    
    def get_operator_metrics(self, operator_id: str) -> Optional[OperatorMetrics]:
        """Get metrics for an operator."""
        return self.operator_metrics.get(operator_id)
    
    def get_user_metrics(
        self,
        operator_id: str,
        user_id: str
    ) -> Optional[UserMetrics]:
        """Get metrics for a user."""
        user_key = f"{operator_id}_{user_id}"
        return self.user_metrics.get(user_key)
    
    def get_all_operator_metrics(self) -> Dict[str, OperatorMetrics]:
        """Get all operator metrics."""
        return self.operator_metrics.copy()
    
    def get_aggregate_statistics(self) -> Dict[str, float]:
        """Get aggregate statistics across all operators."""
        if not self.operator_metrics:
            return {}
        
        all_ops = list(self.operator_metrics.values())
        
        return {
            "total_operators": len(all_ops),
            "total_users": sum(op.num_users for op in all_ops),
            "total_served_users": sum(op.num_served_users for op in all_ops),
            "avg_service_rate": np.mean([op.service_rate for op in all_ops]),
            "avg_throughput_mbps": np.mean([op.avg_throughput_mbps for op in all_ops]),
            "avg_latency_ms": np.mean([op.avg_latency_ms for op in all_ops]),
            "avg_utilization": np.mean([op.utilization for op in all_ops]),
            "avg_spectral_efficiency": np.mean([
                op.spectral_efficiency_mbps_per_mhz for op in all_ops
            ])
        }
    
    def reset(self):
        """Reset all tracking data."""
        self.user_metrics.clear()
        self.operator_metrics.clear()
        self.allocation_history.clear()

