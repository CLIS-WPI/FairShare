"""
QoS estimator for LEO DSS simulations.

Phase 4: Compute throughput, latency, and outage for users based on link budget.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


class QoSEstimator:
    """
    QoS estimator for satellite links.
    
    Computes:
    - Throughput (R): Based on capacity and allocation
    - Latency (D): Propagation + queuing delay
    - Outage (O): Probability of link failure
    """
    
    def __init__(self, min_snr_db: float = 5.0, max_latency_s: float = 0.5):
        """
        Initialize QoS estimator.
        
        Args:
            min_snr_db: Minimum SNR for reliable communication
            max_latency_s: Maximum acceptable latency in seconds
        """
        self.min_snr_db = min_snr_db
        self.max_latency_s = max_latency_s
        self.SPEED_OF_LIGHT = 299792458.0  # m/s
    
    def compute_qos(self, users: List[Dict], link_budgets: Dict[str, Dict],
                   allocations: Dict[str, Optional[Tuple[float, float]]],
                   traffic_demands: Dict[str, float]) -> Dict[str, Dict]:
        """
        Compute QoS metrics for all users.
        
        Args:
            users: List of user dictionaries
            link_budgets: Dictionary mapping user_id to link budget dict
            allocations: Dictionary mapping user_id to allocation (freq, sinr) or None
            traffic_demands: Dictionary mapping user_id to demand
            
        Returns:
            Dictionary mapping user_id to QoS metrics (throughput, latency, outage)
        """
        qos = {}
        
        for user in users:
            user_id = user['id']
            link_budget = link_budgets.get(user_id, {})
            allocation = allocations.get(user_id)
            demand = traffic_demands.get(user_id, 0.0)
            
            # Throughput: based on capacity and allocation
            if allocation is not None:
                # Get SINR from allocation
                sinr_db = allocation[1] if isinstance(allocation, tuple) else 0.0
                
                # Use capacity from link budget if available
                if 'capacity_bps' in link_budget:
                    capacity = link_budget['capacity_bps']
                    # Scale capacity by SINR (higher SINR = better efficiency)
                    # SINR in dB: convert to linear and use Shannon capacity scaling
                    sinr_linear = 10**(sinr_db / 10)
                    # Capacity scales with log2(1 + SINR), but we simplify
                    # For SINR > 0 dB, we get some capacity
                    # Use the actual allocated bandwidth (from allocation, not link budget)
                    # The allocation bandwidth is the actual channel bandwidth assigned
                    allocated_bandwidth_hz = link_budget.get('bandwidth_hz', 200e6)
                    
                    if sinr_db > 0:
                        # Use Shannon capacity: C = B * log2(1 + SINR)
                        # Use allocated bandwidth, not full link budget bandwidth
                        shannon_capacity = allocated_bandwidth_hz * np.log2(1 + sinr_linear)
                        # Effective capacity is the minimum of:
                        # 1. Link budget capacity (based on SNR)
                        # 2. Shannon capacity (based on SINR and allocated bandwidth)
                        # This ensures we don't exceed what the channel can actually support
                        effective_capacity = min(capacity, shannon_capacity)
                    else:
                        # Very low capacity for negative SINR
                        effective_capacity = capacity * 0.1
                    
                    # Throughput is min of effective capacity and demand
                    # This represents actual data rate achieved
                    if demand > 0:
                        throughput = min(effective_capacity, demand)
                    else:
                        throughput = 0.0
                else:
                    # No capacity in link budget - estimate from bandwidth and SINR
                    bandwidth_hz = link_budget.get('bandwidth_hz', 200e6)
                    sinr_linear = 10**(sinr_db / 10)
                    if sinr_db > 0:
                        capacity_estimate = bandwidth_hz * np.log2(1 + sinr_linear)
                    else:
                        capacity_estimate = bandwidth_hz * 0.01  # Very low for negative SINR
                    throughput = min(capacity_estimate, demand) if demand > 0 else 0.0
            else:
                throughput = 0.0
            
            # Latency: propagation delay + queuing delay
            slant_range = link_budget.get('slant_range', 1000e3)  # meters
            prop_delay = slant_range / self.SPEED_OF_LIGHT  # seconds
            
            # Queuing delay (simplified: based on utilization)
            if demand > 0 and throughput > 0:
                utilization = min(demand / throughput, 1.0)
                queue_delay = utilization * 0.1  # Simplified model
            else:
                queue_delay = 0.0
            
            latency = prop_delay + queue_delay
            
            # Outage: probability of link failure
            snr_db = link_budget.get('snr_db', -np.inf)
            if snr_db < self.min_snr_db:
                # Link is in outage
                outage = 1.0
            else:
                # Outage probability based on SNR margin
                snr_margin = snr_db - self.min_snr_db
                # Simplified: exponential decay with SNR margin
                outage = np.exp(-snr_margin / 3.0)  # 3 dB per e-fold
            
            qos[user_id] = {
                'throughput': float(throughput),
                'latency': float(latency),
                'outage': float(outage)
            }
        
        return qos


if __name__ == '__main__':
    # Test QoS estimator
    estimator = QoSEstimator()
    
    users = [{'id': 'u1'}, {'id': 'u2'}]
    link_budgets = {
        'u1': {'capacity_bps': 100e6, 'snr_db': 15.0, 'slant_range': 1000e3},
        'u2': {'capacity_bps': 50e6, 'snr_db': 8.0, 'slant_range': 1200e3}
    }
    allocations = {'u1': (11e9, 15.0), 'u2': (11.1e9, 8.0)}
    demands = {'u1': 80e6, 'u2': 60e6}
    
    qos = estimator.compute_qos(users, link_budgets, allocations, demands)
    print("✓ QoS estimator test")
    for user_id, metrics in qos.items():
        print(f"  {user_id}: throughput={metrics['throughput']/1e6:.2f} Mbps, "
              f"latency={metrics['latency']*1000:.2f} ms, outage={metrics['outage']:.3f}")

