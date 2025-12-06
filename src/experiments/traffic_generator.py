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
    
    # Check if Gaussian 2D distribution is requested
    user_dist = config.geo.get('user_distribution', {})
    use_gaussian = user_dist.get('type') == 'gaussian_2d'
    
    if use_gaussian:
        # Gaussian 2D distribution (for geographic inequality test)
        gauss_center_lat = user_dist.get('center_lat', center_lat)
        gauss_center_lon = user_dist.get('center_lon', center_lon)
        std_lat = user_dist.get('std_lat', 0.15)  # Standard deviation in degrees
        std_lon = user_dist.get('std_lon', 0.15)
        urban_threshold_std = user_dist.get('urban_threshold_std', 1.0)
        
        # Generate positions using Gaussian 2D
        for i in range(config.num_users):
            # Sample from 2D Gaussian
            lat_offset = np.random.normal(0, std_lat)
            lon_offset = np.random.normal(0, std_lon)
            
            lat = gauss_center_lat + lat_offset
            lon = gauss_center_lon + lon_offset
            
            # Clip to reasonable bounds (within radius)
            radius_deg = radius_km / 111.0
            distance_deg = np.sqrt((lat - center_lat)**2 + (lon - center_lon)**2)
            if distance_deg > radius_deg:
                # Resample if outside radius
                lat_offset = np.random.normal(0, std_lat)
                lon_offset = np.random.normal(0, std_lon)
                lat = gauss_center_lat + lat_offset
                lon = gauss_center_lon + lon_offset
                # Clip to radius
                if distance_deg > radius_deg:
                    scale = radius_deg / distance_deg
                    lat = center_lat + (lat - center_lat) * scale
                    lon = center_lon + (lon - center_lon) * scale
            
            # Classify as urban or rural based on distance from center
            dist_from_center_std = np.sqrt((lat_offset/std_lat)**2 + (lon_offset/std_lon)**2)
            is_urban = dist_from_center_std <= urban_threshold_std
            
            # Assign operator (for 2 operators: Starlink and OneWeb)
            if hasattr(config, 'operators') and config.operators:
                # Equal distribution between operators
                operator = 'Op_A' if i % 2 == 0 else 'Op_B'
            else:
                operator = i % config.num_operators
            
            # Assign priority (equal for both operators in this test)
            priority = 0.5
            
            user_dict = {
                'id': f"u_{i}",
                'operator': operator,
                'priority': priority,
                'lat': float(lat),
                'lon': float(lon),
                'is_urban': bool(is_urban),
                'dist_from_center_deg': float(np.sqrt((lat - gauss_center_lat)**2 + (lon - gauss_center_lon)**2))
            }
            users.append(user_dict)
    else:
        # Original uniform distribution
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
            is_urban = None  # Not applicable for uniform distribution
        
        # Assign operator based on config
        # For NYC scenario: 60% Op_A, 30% Op_B, 10% Op_C
        if hasattr(config, 'operators') and config.operators:
            # NYC scenario with specific operator distribution
            rand_op = np.random.random()
            if rand_op < 0.6:  # 60% Op_A
                operator = 'Op_A'
            elif rand_op < 0.9:  # 30% Op_B
                operator = 'Op_B'
            else:  # 10% Op_C
                operator = 'Op_C'
        else:
            # Default: round-robin
            operator = i % config.num_operators
        
        # Assign priority based on operator (for NYC scenario)
        if hasattr(config, 'operators') and config.operators and isinstance(operator, str):
            # NYC scenario: Op_C has priority=1.0 (critical), Op_A=0.5, Op_B=0.6
            if operator == 'Op_C':
                priority = 1.0  # Critical
            elif operator == 'Op_B':
                priority = 0.6  # Medium
            else:  # Op_A
                priority = 0.5  # Normal
        else:
            # Default: random priority distribution
            rand_val = np.random.random()
            if rand_val < 0.2:  # 20% high priority
                priority = np.random.uniform(0.7, 1.0)
            elif rand_val < 0.5:  # 30% medium priority (0.2 to 0.5 = 30% of total)
                priority = np.random.uniform(0.4, 0.7)
            else:  # 50% low priority (0.5 to 1.0 = 50% of total)
                priority = np.random.uniform(0.1, 0.4)
        
        user_dict = {
            'id': f"u_{i}",
            'operator': operator,
            'priority': priority,
            'lat': float(lat),
            'lon': float(lon)
        }
        
        # Add urban/rural classification for Gaussian distribution
        if use_gaussian and is_urban is not None:
            user_dict['is_urban'] = bool(is_urban)
            # Store distance from center for later analysis
            if use_gaussian:
                dist_from_center = np.sqrt((lat - gauss_center_lat)**2 + (lon - gauss_center_lon)**2)
                user_dict['dist_from_center_deg'] = float(dist_from_center)
        
        users.append(user_dict)
    
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
                # Scale lambda to get more realistic traffic
                # lambda_per_user is packets per slot, scale it up for realistic demand
                num_packets = np.random.poisson(lambda_per_user * 10)  # Scale up for more traffic
                # Demand in actual bps (not normalized)
                # Assume average packet size ~1500 bytes = 12000 bits
                packet_size_bits = 12000  # 1500 bytes
                packets_per_second = num_packets / slot_duration
                demand_bps = packets_per_second * packet_size_bits
                # Cap at reasonable fraction of bandwidth (e.g., 80% max)
                max_demand_bps = config.bandwidth_hz * 0.8
                demand = min(demand_bps, max_demand_bps)
            elif traffic_type == 'constant':
                # Constant demand in bps (scale lambda to bps)
                # lambda_per_user is treated as fraction of bandwidth
                demand = lambda_per_user * config.bandwidth_hz
            elif traffic_type == 'bursty':
                # Bursty traffic: occasional high demand
                if np.random.random() < 0.1:  # 10% chance of burst
                    demand = np.random.uniform(0.6, 0.8) * config.bandwidth_hz
                else:
                    demand = np.random.uniform(0.1, 0.3) * config.bandwidth_hz
            else:
                # Default: uniform random (as fraction of bandwidth)
                demand = np.random.uniform(0.0, lambda_per_user) * config.bandwidth_hz
            
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
            Dictionary with 'users', 'traffic', and 'priorities' keys
        """
        # Generate user positions (includes priorities)
        self.users = generate_user_positions(self.config, self.seed)
        
        # Generate traffic arrivals
        self.traffic = generate_traffic_arrivals(self.config, self.users, self.seed)
        
        # Extract priorities as separate dictionary (FIXED: was missing)
        priorities = {user['id']: user['priority'] for user in self.users}
        
        return {
            'users': self.users,
            'traffic': self.traffic,
            'priorities': priorities,  # FIXED: Added priorities to return dict
            'user_ids': [user['id'] for user in self.users]  # Also add user_ids for convenience
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

