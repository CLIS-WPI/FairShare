"""
Synthetic data generator for LEO satellite scenarios.

Creates realistic traffic patterns, user distributions, and demand models
when real data is unavailable.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timedelta
import yaml


@dataclass
class UserProfile:
    """User profile with demand characteristics."""
    user_id: str
    location: Tuple[float, float]  # (latitude, longitude)
    operator_id: str
    demand_mbps: float
    priority: float
    traffic_type: str  # "video", "web", "iot", etc.
    mobility: bool = False


@dataclass
class TrafficPattern:
    """Traffic pattern over time."""
    timestamp: datetime
    total_demand_mbps: float
    user_demands: Dict[str, float]
    active_users: int


class SyntheticDataGenerator:
    """
    Generate synthetic but realistic data for LEO scenarios.
    
    Uses population statistics, official LEO records, and FCC benchmarks
    to create realistic distributions.
    """
    
    def __init__(
        self,
        region_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        population_density_per_km2: float = 0.1
    ):
        """
        Initialize data generator.
        
        Args:
            region_bounds: Region bounds {"lat": (min, max), "lon": (min, max)}
            population_density_per_km2: User density per km²
        """
        if region_bounds is None:
            # Default: US East Coast
            region_bounds = {
                "lat": (35.0, 45.0),
                "lon": (-85.0, -70.0)
            }
        
        self.region_bounds = region_bounds
        self.population_density = population_density_per_km2
        
        # Traffic type distributions (based on real statistics)
        self.traffic_type_weights = {
            "video": 0.4,  # 40% video streaming
            "web": 0.3,   # 30% web browsing
            "iot": 0.2,   # 20% IoT devices
            "voice": 0.1  # 10% voice calls
        }
        
        # Demand distributions per traffic type (Mbps)
        self.demand_ranges = {
            "video": (2.0, 25.0),  # Video streaming
            "web": (0.5, 5.0),     # Web browsing
            "iot": (0.01, 0.1),    # IoT devices
            "voice": (0.064, 0.128)  # Voice calls
        }
    
    def generate_user_positions(
        self,
        num_users: int,
        distribution: str = "uniform"
    ) -> List[Tuple[float, float]]:
        """
        Generate user positions.
        
        Args:
            num_users: Number of users
            distribution: Distribution type ("uniform", "gaussian", "clustered")
            
        Returns:
            List of (latitude, longitude) tuples
        """
        lat_min, lat_max = self.region_bounds["lat"]
        lon_min, lon_max = self.region_bounds["lon"]
        
        if distribution == "uniform":
            lats = np.random.uniform(lat_min, lat_max, num_users)
            lons = np.random.uniform(lon_min, lon_max, num_users)
        elif distribution == "gaussian":
            # Concentrated around center
            lat_center = (lat_min + lat_max) / 2
            lon_center = (lon_min + lon_max) / 2
            lat_std = (lat_max - lat_min) / 4
            lon_std = (lon_max - lon_min) / 4
            
            lats = np.random.normal(lat_center, lat_std, num_users)
            lons = np.random.normal(lon_center, lon_std, num_users)
            lats = np.clip(lats, lat_min, lat_max)
            lons = np.clip(lons, lon_min, lon_max)
        elif distribution == "clustered":
            # Create clusters
            n_clusters = max(3, num_users // 50)
            lats = []
            lons = []
            
            for _ in range(num_users):
                cluster_lat = np.random.uniform(lat_min, lat_max)
                cluster_lon = np.random.uniform(lon_min, lon_max)
                # Add noise around cluster center
                lat = cluster_lat + np.random.normal(0, 0.5)
                lon = cluster_lon + np.random.normal(0, 0.5)
                lats.append(np.clip(lat, lat_min, lat_max))
                lons.append(np.clip(lon, lon_min, lon_max))
            
            lats = np.array(lats)
            lons = np.array(lons)
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
        
        return list(zip(lats, lons))
    
    def generate_users(
        self,
        num_users: int,
        operators: List[str],
        distribution: str = "uniform"
    ) -> List[UserProfile]:
        """
        Generate user profiles.
        
        Args:
            num_users: Number of users
            operators: List of operator IDs
            distribution: Position distribution
            
        Returns:
            List of user profiles
        """
        positions = self.generate_user_positions(num_users, distribution)
        users = []
        
        # Assign operators (proportional or random)
        operator_assignments = np.random.choice(
            operators,
            size=num_users,
            p=[1.0/len(operators)] * len(operators)  # Equal probability
        )
        
        # Assign traffic types
        traffic_types = list(self.traffic_type_weights.keys())
        traffic_probs = list(self.traffic_type_weights.values())
        traffic_assignments = np.random.choice(
            traffic_types,
            size=num_users,
            p=traffic_probs
        )
        
        for i, (pos, operator, traffic_type) in enumerate(
            zip(positions, operator_assignments, traffic_assignments)
        ):
            # Generate demand based on traffic type
            demand_min, demand_max = self.demand_ranges[traffic_type]
            demand = np.random.uniform(demand_min, demand_max)
            
            # Generate priority (higher for video/voice)
            if traffic_type in ["video", "voice"]:
                priority = np.random.uniform(0.7, 1.0)
            else:
                priority = np.random.uniform(0.3, 0.8)
            
            user = UserProfile(
                user_id=f"user_{i:04d}",
                location=pos,
                operator_id=operator,
                demand_mbps=demand,
                priority=priority,
                traffic_type=traffic_type,
                mobility=np.random.random() < 0.1  # 10% mobile
            )
            users.append(user)
        
        return users
    
    def generate_traffic_pattern(
        self,
        users: List[UserProfile],
        timestamp: datetime,
        time_variation: bool = True
    ) -> TrafficPattern:
        """
        Generate traffic pattern for a timestamp.
        
        Args:
            users: List of users
            timestamp: Current timestamp
            time_variation: Whether to apply time-based variation
            
        Returns:
            Traffic pattern
        """
        user_demands = {}
        total_demand = 0.0
        active_users = 0
        
        # Time-based variation (peak hours, etc.)
        hour = timestamp.hour
        if time_variation:
            # Peak hours: 8-10 AM, 6-9 PM
            if 8 <= hour <= 10 or 18 <= hour <= 21:
                demand_multiplier = np.random.uniform(1.2, 1.5)
            # Off-peak: 2-5 AM
            elif 2 <= hour <= 5:
                demand_multiplier = np.random.uniform(0.5, 0.7)
            else:
                demand_multiplier = np.random.uniform(0.8, 1.2)
        else:
            demand_multiplier = 1.0
        
        for user in users:
            # Base demand
            base_demand = user.demand_mbps
            
            # Apply time variation
            demand = base_demand * demand_multiplier
            
            # Random variation
            demand *= np.random.uniform(0.8, 1.2)
            
            # Some users may be inactive
            if np.random.random() < 0.1:  # 10% inactive
                demand = 0.0
            
            user_demands[user.user_id] = demand
            total_demand += demand
            
            if demand > 0:
                active_users += 1
        
        return TrafficPattern(
            timestamp=timestamp,
            total_demand_mbps=total_demand,
            user_demands=user_demands,
            active_users=active_users
        )
    
    def generate_traffic_timeline(
        self,
        users: List[UserProfile],
        start_time: datetime,
        duration_hours: float,
        interval_minutes: int = 5
    ) -> List[TrafficPattern]:
        """
        Generate traffic timeline.
        
        Args:
            users: List of users
            start_time: Start timestamp
            duration_hours: Duration in hours
            interval_minutes: Time interval in minutes
            
        Returns:
            List of traffic patterns
        """
        patterns = []
        current_time = start_time
        end_time = start_time + timedelta(hours=duration_hours)
        
        while current_time < end_time:
            pattern = self.generate_traffic_pattern(users, current_time)
            patterns.append(pattern)
            current_time += timedelta(minutes=interval_minutes)
        
        return patterns

