"""
Metrics logger for LEO DSS simulations.

Phase 4: Log fairness metrics, QoS, and allocation statistics to CSV.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import os
import logging

logger = logging.getLogger(__name__)

from src.fairness.metrics import (
    jain_fairness_index,
    alpha_fairness,
    gini_coefficient,
    max_min_fairness
)
from src.fairness.traditional import TraditionalFairness
from src.fairness.vector_metrics import VectorFairness


class MetricsLogger:
    """
    Logger for simulation metrics.
    
    Tracks:
    - Fairness metrics (Jain, α-fair, Weighted Fairness, Gini)
    - QoS metrics (mean rate, cell edge rate)
    - Allocation statistics
    - Operator imbalance
    """
    
    def __init__(self, scenario_name: str, policy_name: str):
        """
        Initialize metrics logger.
        
        Args:
            scenario_name: Name of scenario
            policy_name: Name of policy (static, priority, rl)
        """
        self.scenario_name = scenario_name
        self.policy_name = policy_name
        self.traditional_fairness = TraditionalFairness()
        self.vector_fairness = VectorFairness()
        
        self.metrics_history = []
        self.current_slot = 0
        self.interference_history = []  # For NYC scenario: interference logging
    
    def update(self, slot: int, users: List[Dict], qos: Dict[str, Dict],
              allocations: Dict[str, Optional[Tuple[float, float]]], context: Optional[Dict] = None,
              interference_data: Optional[Dict] = None) -> None:
        """
        Update metrics for a time slot.
        
        Args:
            slot: Time slot index
            users: List of user dictionaries
            qos: Dictionary mapping user_id to QoS metrics (throughput, latency, outage)
            allocations: Dictionary mapping user_id to allocation (freq, sinr) or None
            context: Optional context dictionary (geometry, etc.)
        """
        self.current_slot = slot
        
        # Extract allocations and QoS
        user_ids = [u['id'] for u in users]
        allocation_values = []
        throughputs = []
        latencies = []
        outages = []
        priorities = []
        operators = []
        
        for user in users:
            user_id = user['id']
            
            # Allocation
            alloc = allocations.get(user_id)
            if alloc is not None:
                # Allocation is (freq, sinr) tuple
                allocation_values.append(alloc[1] if isinstance(alloc, tuple) else 1.0)
            else:
                allocation_values.append(0.0)
            
            # QoS
            user_qos = qos.get(user_id, {})
            throughputs.append(user_qos.get('throughput', 0.0))
            latencies.append(user_qos.get('latency', 1.0))
            outages.append(user_qos.get('outage', 1.0))
            
            # User attributes
            priorities.append(user.get('priority', 0.5))
            operators.append(user.get('operator', 0))
        
        allocation_array = np.array(allocation_values)
        throughput_array = np.array(throughputs)
        
        # Compute fairness metrics
        jain = jain_fairness_index(allocation_array)
        
        # α-fairness for multiple α values
        alpha_0 = alpha_fairness(allocation_array, alpha=0.0)  # Max-sum
        alpha_1 = alpha_fairness(allocation_array, alpha=1.0)  # Proportional
        alpha_2 = alpha_fairness(allocation_array, alpha=2.0)  # More fair
        
        # Gini coefficient
        gini = gini_coefficient(allocation_array)
        
        # Max-min fairness
        max_min = max_min_fairness(allocation_array)
        
        # Weighted fairness (multi-dimensional)
        weighted_fairness = self._compute_weighted_fairness(allocation_array, throughput_array, priorities)
        
        # QoS statistics
        # FIXED: Only compute mean_rate from allocated users (allocation > 0)
        # This ensures Priority policy with 50/500 users has correct mean rate
        allocated_mask = allocation_array > 0
        if np.any(allocated_mask):
            allocated_throughputs = throughput_array[allocated_mask]
            mean_rate = np.mean(allocated_throughputs) if len(allocated_throughputs) > 0 else 0.0
            cell_edge_rate = np.percentile(allocated_throughputs, 5) if len(allocated_throughputs) > 0 else 0.0
        else:
            # No allocations - all zeros
            mean_rate = 0.0
            cell_edge_rate = 0.0
        
        # Operator imbalance (Gini coefficient across operators)
        operator_imbalance = self._compute_operator_imbalance(allocation_array, operators)
        
        # Urban/Rural Jain Index (for geographic inequality test)
        urban_jain = None
        rural_jain = None
        if users and 'is_urban' in users[0]:
            # Separate users into urban and rural
            urban_mask = np.array([u.get('is_urban', False) for u in users])
            rural_mask = ~urban_mask
            
            if np.any(urban_mask):
                urban_allocations = allocation_array[urban_mask]
                urban_jain = jain_fairness_index(urban_allocations) if len(urban_allocations) > 0 else 0.0
            
            if np.any(rural_mask):
                rural_allocations = allocation_array[rural_mask]
                rural_jain = jain_fairness_index(rural_allocations) if len(rural_allocations) > 0 else 0.0
        
        # Record metrics
        metrics = {
            'slot': slot,
            'policy': self.policy_name,
            'jain': float(jain),
            'urban_jain': float(urban_jain) if urban_jain is not None else None,
            'rural_jain': float(rural_jain) if rural_jain is not None else None,
            'alpha_0': float(alpha_0),
            'alpha_1': float(alpha_1),
            'alpha_2': float(alpha_2),
            'weighted_fairness': float(weighted_fairness),
            'gini': float(gini),
            'max_min': float(max_min),
            'mean_rate': float(mean_rate),
            'cell_edge_rate': float(cell_edge_rate),
            'operator_imbalance': float(operator_imbalance),
            'num_users': len(users),
            'num_allocated': int(np.sum(allocation_array > 0)),
            'total_allocation': float(np.sum(allocation_array))
        }
        
        self.metrics_history.append(metrics)
        
        # Store interference data if provided (for NYC heatmap)
        if interference_data is not None:
            self.interference_history.append(interference_data)
    
    def _compute_weighted_fairness(self, allocations: np.ndarray,
                                   throughputs: np.ndarray,
                                   priorities: np.ndarray) -> float:
        """
        Compute weighted fairness from allocations and throughputs.
        
        Uses multi-dimensional fairness metrics considering throughput,
        access rate, and QoS satisfaction.
        
        Args:
            allocations: Array of allocations (bandwidth in Hz)
            throughputs: Array of throughput values (in bps)
            priorities: Array of user priorities (not used directly, but for fallback)
            
        Returns:
            Weighted fairness score (0.0 to 1.0, higher is better)
        """
        if len(allocations) == 0:
            return 0.0
        
        # Create multi-dimensional metrics for allocated users
        metrics = []
        for i in range(len(allocations)):
            if allocations[i] > 0:
                # Convert throughput from bps to Mbps
                throughput_mbps = (throughputs[i] if i < len(throughputs) else 0.0) / 1e6
                
                metrics.append({
                    'throughput_mbps': throughput_mbps,
                    'latency_ms': 0.0,  # Not available in this context
                    'access_rate': 1.0,  # User has allocation
                    'coverage_quality': 1.0,  # Assume good coverage if allocated
                    'qos_satisfaction': 1.0 if allocations[i] > 0 else 0.0
                })
        
        if len(metrics) == 0:
            # No allocations - return 0 fairness
            return 0.0
        
        # Compute weighted fairness using vector fairness
        try:
            # Convert metrics to MultiDimensionalMetrics format
            from src.fairness.vector_metrics import MultiDimensionalMetrics
            
            metric_objects = [
                MultiDimensionalMetrics(
                    throughput_mbps=m['throughput_mbps'],
                    latency_ms=m['latency_ms'],
                    access_rate=m['access_rate'],
                    coverage_quality=m['coverage_quality'],
                    qos_satisfaction=m['qos_satisfaction']
                )
                for m in metrics
            ]
            
            # Compute fairness vector
            fairness_vector = self.vector_fairness.compute_fairness_vector(metric_objects)
            
            # Compute weighted fairness
            weighted = self.vector_fairness.compute_weighted_fairness(fairness_vector)
            return float(weighted)
        except Exception as e:
            # Fallback to proportional fairness (alpha=1)
            return alpha_fairness(allocations, alpha=1.0)
    
    def _compute_operator_imbalance(self, allocations: np.ndarray,
                                   operators: List[int]) -> float:
        """
        Compute operator imbalance (Gini coefficient across operators).
        
        Args:
            allocations: Array of allocations
            operators: List of operator IDs
            
        Returns:
            Operator imbalance (Gini coefficient)
        """
        if len(allocations) == 0:
            return 0.0
        
        # Group allocations by operator
        operator_allocations = {}
        for i, op in enumerate(operators):
            if op not in operator_allocations:
                operator_allocations[op] = []
            operator_allocations[op].append(allocations[i])
        
        # Compute total allocation per operator
        operator_totals = [np.sum(operator_allocations[op]) for op in operator_allocations]
        
        if len(operator_totals) == 0:
            return 0.0
        
        # Compute Gini coefficient
        return gini_coefficient(np.array(operator_totals))
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert metrics history to pandas DataFrame.
        
        Returns:
            DataFrame with all metrics
        """
        return pd.DataFrame(self.metrics_history)
    
    def to_csv(self, output_dir: str = 'results') -> str:
        """
        Save metrics to CSV file.
        
        Args:
            output_dir: Output directory
            
        Returns:
            Path to saved CSV file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        df = self.to_dataframe()
        filename = f"{self.scenario_name}_{self.policy_name}.csv"
        filepath = os.path.join(output_dir, filename)
        
        df.to_csv(filepath, index=False)
        
        # Save interference data if available (for NYC heatmap)
        if hasattr(self, 'interference_history') and len(self.interference_history) > 0:
            import json
            interference_filepath = filepath.replace('.csv', '_interference.json')
            with open(interference_filepath, 'w') as f:
                json.dump(self.interference_history, f, indent=2)
            logger.info(f"Interference data saved to: {interference_filepath}")
        
        return filepath
    
    def get_summary(self) -> Dict:
        """
        Get summary statistics.
        
        Returns:
            Dictionary with summary metrics
        """
        if len(self.metrics_history) == 0:
            return {}
        
        df = self.to_dataframe()
        
        return {
            'mean_jain': float(df['jain'].mean()),
            'mean_weighted_fairness': float(df['weighted_fairness'].mean()),
            'mean_alpha_1': float(df['alpha_1'].mean()),
            'mean_mean_rate': float(df['mean_rate'].mean()),
            'mean_cell_edge_rate': float(df['cell_edge_rate'].mean()),
            'mean_operator_imbalance': float(df['operator_imbalance'].mean()),
            'num_slots': len(self.metrics_history)
        }


if __name__ == '__main__':
    # Test logger
    logger = MetricsLogger("test_scenario", "priority")
    
    # Simulate some metrics
    users = [
        {'id': 'u1', 'priority': 0.8, 'operator': 0},
        {'id': 'u2', 'priority': 0.5, 'operator': 1},
        {'id': 'u3', 'priority': 0.3, 'operator': 0}
    ]
    
    qos = {
        'u1': {'throughput': 50e6, 'latency': 0.1, 'outage': 0.05},
        'u2': {'throughput': 30e6, 'latency': 0.2, 'outage': 0.1},
        'u3': {'throughput': 20e6, 'latency': 0.3, 'outage': 0.15}
    }
    
    allocations = {
        'u1': (11e9, 20.0),
        'u2': (11.1e9, 18.0),
        'u3': (11.2e9, 15.0)
    }
    
    logger.update(0, users, qos, allocations)
    
    print("✓ Metrics logger test")
    print(f"  Summary: {logger.get_summary()}")
    
    # Save to CSV
    filepath = logger.to_csv()
    print(f"  Saved to: {filepath}")

