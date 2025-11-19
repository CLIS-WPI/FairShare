"""
Traffic generator for LEO DSS simulations.

Phase 4: Generate user positions and traffic arrivals.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from .scenario_loader import ScenarioConfig


def generate_user_positions(config: ScenarioConfig, seed: Optional[int] = None) -> List[Dict]:
    """
    Generate user positions on Earth surface.
    
    Args:
        config: Scenario configuration
        seed: Random seed for reproducibility
        
    Returns:
        List of user dictionaries with id, operator, priority, lat, lon
    """
    if seed is not None:
        np.random.seed(seed)
    
    users = []
    center_lat = config.geo['center_lat_deg']
    center_lon = config.geo['center_lon_deg']
    radius_km = config.geo['radius_km']
    
    # Convert radius to degrees (approximate)
    # 1 degree latitude ≈ 111 km
    radius_deg = radius_km / 111.0
    
    # Generate positions uniformly in circular area
    for i in range(config.num_users):
        # Generate random angle and radius
        angle = np.random.uniform(0, 2 * np.pi)
        # Use square root for uniform distribution in circle
        r = np.sqrt(np.random.uniform(0, 1)) * radius_deg
        
        # Convert to lat/lon
        lat = center_lat + r * np.cos(angle)
        lon = center_lon + r * np.sin(angle)
        
        # Assign operator (round-robin)
        operator = i % config.num_operators
        
        # Assign priority (random, but higher for some users)
        if np.random.random() < 0.2:  # 20% high priority
            priority = np.random.uniform(0.7, 1.0)
        elif np.random.random() < 0.5:  # 30% medium priority
            priority = np.random.uniform(0.4, 0.7)
        else:  # 50% low priority
            priority = np.random.uniform(0.1, 0.4)
        
        users.append({
            'id': f"u_{i}",
            'operator': operator,
            'priority': priority,
            'lat': float(lat),
            'lon': float(lon)
        })
    
    return users


def generate_traffic_arrivals(
    config: ScenarioConfig,
    users: List[Dict],
    seed: Optional[int] = None
) -> Dict[float, Dict[str, float]]:
    """
    Generate traffic arrivals using Poisson model.
    
    Args:
        config: Scenario configuration
        users: List of user dictionaries
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary mapping time (seconds) to user demand dictionary
    """
    if seed is not None:
        np.random.seed(seed)
    
    traffic = {}
    num_slots = config.get_num_slots()
    slot_duration = config.slot_duration_s
    
    # Get traffic model parameters
    traffic_type = config.traffic_model.get('type', 'poisson')
    lambda_per_user = config.traffic_model.get('lambda_per_user', 0.2)
    
    # Generate traffic for each slot
    for slot_idx in range(num_slots):
        t = slot_idx * slot_duration
        slot_demands = {}
        
        for user in users:
            user_id = user['id']
            
            if traffic_type == 'poisson':
                # Poisson arrivals: number of packets per slot
                num_packets = np.random.poisson(lambda_per_user)
                # Demand is normalized (0-1)
                demand = min(num_packets * 0.1, 1.0)  # Scale to [0, 1]
            elif traffic_type == 'constant':
                # Constant demand
                demand = lambda_per_user
            elif traffic_type == 'bursty':
                # Bursty traffic: occasional high demand
                if np.random.random() < 0.1:  # 10% chance of burst
                    demand = np.random.uniform(0.8, 1.0)
                else:
                    demand = np.random.uniform(0.1, 0.3)
            else:
                # Default: uniform random
                demand = np.random.uniform(0.0, lambda_per_user)
            
            slot_demands[user_id] = float(demand)
        
        traffic[t] = slot_demands
    
    return traffic


class TrafficGenerator:
    """
    Traffic generator for LEO DSS simulations.
    
    Combines user position generation and traffic arrival generation.
    """
    
    def __init__(self, config: ScenarioConfig, seed: Optional[int] = None):
        """
        Initialize traffic generator.
        
        Args:
            config: Scenario configuration
            seed: Random seed for reproducibility
        """
        self.config = config
        self.seed = seed
        self.users = None
        self.traffic = None
    
    def generate(self) -> Dict:
        """
        Generate complete traffic scenario.
        
        Returns:
            Dictionary with 'users' and 'traffic' keys
        """
        # Generate user positions
        self.users = generate_user_positions(self.config, self.seed)
        
        # Generate traffic arrivals
        self.traffic = generate_traffic_arrivals(self.config, self.users, self.seed)
        
        return {
            'users': self.users,
            'traffic': self.traffic
        }
    
    def get_user_demand(self, user_id: str, time_s: float) -> float:
        """
        Get demand for a specific user at a specific time.
        
        Args:
            user_id: User identifier
            time_s: Time in seconds
            
        Returns:
            Demand value (0-1)
        """
        if self.traffic is None:
            return 0.0
        
        # Find closest slot
        slot_duration = self.config.slot_duration_s
        slot_idx = int(time_s / slot_duration)
        slot_time = slot_idx * slot_duration
        
        if slot_time in self.traffic:
            return self.traffic[slot_time].get(user_id, 0.0)
        
        return 0.0


if __name__ == '__main__':
    # Test generation
    from .scenario_loader import load_scenario
    
    try:
        config = load_scenario("urban_congestion")
        generator = TrafficGenerator(config, seed=42)
        result = generator.generate()
        
        print(f"✓ Generated {len(result['users'])} users")
        print(f"✓ Generated traffic for {len(result['traffic'])} slots")
        print(f"  Sample user: {result['users'][0]}")
        print(f"  Sample traffic at t=0: {list(result['traffic'][0.0].items())[:3]}")
    except Exception as e:
        print(f"⚠ Error: {e}")

