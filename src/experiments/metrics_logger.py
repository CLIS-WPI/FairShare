"""
Metrics logger for LEO DSS simulations.

Phase 4: Log fairness metrics, QoS, and allocation statistics to CSV.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import os

from src.fairness.metrics import (
    jain_fairness_index,
    alpha_fairness,
    gini_coefficient,
    max_min_fairness
)
from src.fairness.fuzzy_core import FuzzyInferenceSystem


class MetricsLogger:
    """
    Logger for simulation metrics.
    
    Tracks:
    - Fairness metrics (Jain, α-fair, Fuzzy, Gini)
    - QoS metrics (mean rate, cell edge rate)
    - Allocation statistics
    - Operator imbalance
    """
    
    def __init__(self, scenario_name: str, policy_name: str, fis: Optional[FuzzyInferenceSystem] = None):
        """
        Initialize metrics logger.
        
        Args:
            scenario_name: Name of scenario
            policy_name: Name of policy (static, priority, fuzzy)
            fis: Optional FIS for fuzzy fairness computation
        """
        self.scenario_name = scenario_name
        self.policy_name = policy_name
        self.fis = fis or FuzzyInferenceSystem(use_phase3=True)
        
        self.metrics_history = []
        self.current_slot = 0
    
    def update(self, slot: int, users: List[Dict], qos: Dict[str, Dict],
              allocations: Dict[str, Optional[Tuple[float, float]]], context: Optional[Dict] = None) -> None:
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
        
        # Fuzzy fairness (network-level)
        # Use average context if available
        if context:
            # Compute network-level fuzzy fairness
            fuzzy_fairness = self._compute_network_fuzzy_fairness(users, qos, context)
        else:
            # Fallback: use allocation-based fuzzy fairness
            fuzzy_fairness = self._compute_allocation_fuzzy_fairness(allocation_array, priorities)
        
        # QoS statistics
        mean_rate = np.mean(throughput_array) if len(throughput_array) > 0 else 0.0
        cell_edge_rate = np.percentile(throughput_array, 5) if len(throughput_array) > 0 else 0.0
        
        # Operator imbalance (Gini coefficient across operators)
        operator_imbalance = self._compute_operator_imbalance(allocation_array, operators)
        
        # Record metrics
        metrics = {
            'slot': slot,
            'policy': self.policy_name,
            'jain': float(jain),
            'alpha_0': float(alpha_0),
            'alpha_1': float(alpha_1),
            'alpha_2': float(alpha_2),
            'fuzzy_fairness': float(fuzzy_fairness),
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
    
    def _compute_network_fuzzy_fairness(self, users: List[Dict], qos: Dict[str, Dict],
                                       context: Dict) -> float:
        """
        Compute network-level fuzzy fairness from user contexts.
        
        Args:
            users: List of users
            qos: QoS metrics per user
            context: Context dictionary
            
        Returns:
            Network-level fuzzy fairness score
        """
        # Collect all user fairness scores
        fairness_scores = []
        
        for user in users:
            user_id = user['id']
            user_qos = qos.get(user_id, {})
            user_context = context.get(user_id, {})
            
            # Build FIS inputs
            inputs = {
                'throughput': np.clip(user_qos.get('throughput', 0.0) / 100e6, 0, 1),  # Normalize
                'latency': np.clip(user_qos.get('latency', 1.0) / 1.0, 0, 1),  # Normalize
                'outage': np.clip(user_qos.get('outage', 0.0), 0, 1),
                'priority': np.clip(user.get('priority', 0.5), 0, 1),
                'doppler': np.clip(user_context.get('doppler', 0.5), 0, 1),
                'elevation': np.clip(user_context.get('elevation', 0.5) / 90.0, 0, 1),  # Normalize
                'beam_load': np.clip(user_context.get('beam_load', 0.5), 0, 1)
            }
            
            # Compute fairness score
            fairness = self.fis.infer(inputs)
            fairness_scores.append(fairness)
        
        # Network-level fairness: average of user fairness scores
        if len(fairness_scores) > 0:
            return float(np.mean(fairness_scores))
        return 0.0
    
    def _compute_allocation_fuzzy_fairness(self, allocations: np.ndarray,
                                          priorities: List[float]) -> float:
        """
        Compute fuzzy fairness from allocations (fallback method).
        
        Args:
            allocations: Array of allocations
            priorities: List of priorities
            
        Returns:
            Fuzzy fairness score
        """
        # Use proportional fairness as proxy
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
            'mean_fuzzy_fairness': float(df['fuzzy_fairness'].mean()),
            'mean_alpha_1': float(df['alpha_1'].mean()),
            'mean_mean_rate': float(df['mean_rate'].mean()),
            'mean_cell_edge_rate': float(df['cell_edge_rate'].mean()),
            'mean_operator_imbalance': float(df['operator_imbalance'].mean()),
            'num_slots': len(self.metrics_history)
        }


if __name__ == '__main__':
    # Test logger
    logger = MetricsLogger("test_scenario", "fuzzy")
    
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

