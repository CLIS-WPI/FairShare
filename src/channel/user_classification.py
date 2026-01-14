"""
User location classification based on population density
"""

import numpy as np
from typing import Tuple, List


class UserLocationClassifier:
    """
    Classify user locations into urban/suburban/rural
    Based on distance from urban centers
    """
    
    # NYC Metro area centers
    URBAN_CENTERS = [
        {"name": "Manhattan", "lat": 40.7128, "lon": -74.0060, "radius_km": 15},
        {"name": "Brooklyn", "lat": 40.6782, "lon": -73.9442, "radius_km": 10},
        {"name": "Newark", "lat": 40.7357, "lon": -74.1724, "radius_km": 8},
    ]
    
    LOCATION_TYPES = {
        0: "dense_urban",
        1: "urban", 
        2: "suburban",
        3: "rural"
    }
    
    @classmethod
    def classify(cls, lat: float, lon: float) -> Tuple[int, str]:
        """
        Classify a single location
        
        Returns:
            (type_index, type_name)
        """
        min_distance = float('inf')
        
        for center in cls.URBAN_CENTERS:
            dist = cls._haversine(lat, lon, center['lat'], center['lon'])
            min_distance = min(min_distance, dist)
        
        if min_distance < 5:
            return 0, "dense_urban"
        elif min_distance < 20:
            return 1, "urban"
        elif min_distance < 50:
            return 2, "suburban"
        else:
            return 3, "rural"
    
    @classmethod
    def classify_batch(cls, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
        """
        Classify batch of locations
        
        Returns:
            Array of type indices
        """
        n = len(lats)
        types = np.zeros(n, dtype=np.int32)
        
        for i in range(n):
            types[i], _ = cls.classify(lats[i], lons[i])
        
        return types
    
    @staticmethod
    def _haversine(lat1, lon1, lat2, lon2):
        """Distance in km"""
        R = 6371
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
        return R * 2 * np.arcsin(np.sqrt(a))


class PopulationBasedTrafficGenerator:
    """
    Generate user distribution based on population density
    """
    
    def __init__(self, center_lat: float = 40.7128, center_lon: float = -74.0060):
        self.center_lat = center_lat
        self.center_lon = center_lon
    
    def generate_users(self, n_users: int, 
                       urban_ratio: float = 0.6,
                       suburban_ratio: float = 0.25,
                       rural_ratio: float = 0.15) -> dict:
        """
        Generate user positions with realistic distribution
        
        Returns:
            dict with 'lats', 'lons', 'location_types'
        """
        assert abs(urban_ratio + suburban_ratio + rural_ratio - 1.0) < 0.01
        
        n_urban = int(n_users * urban_ratio)
        n_suburban = int(n_users * suburban_ratio)
        n_rural = n_users - n_urban - n_suburban
        
        users = {
            'lats': [],
            'lons': [],
            'location_types': []
        }
        
        # Urban users: tight Gaussian around center
        urban_lats = np.random.normal(self.center_lat, 0.05, n_urban)
        urban_lons = np.random.normal(self.center_lon, 0.05, n_urban)
        users['lats'].extend(urban_lats)
        users['lons'].extend(urban_lons)
        users['location_types'].extend([0] * n_urban)  # dense_urban
        
        # Suburban users: wider ring
        angles = np.random.uniform(0, 2*np.pi, n_suburban)
        radii = np.random.uniform(0.2, 0.5, n_suburban)  # degrees
        suburban_lats = self.center_lat + radii * np.sin(angles)
        suburban_lons = self.center_lon + radii * np.cos(angles)
        users['lats'].extend(suburban_lats)
        users['lons'].extend(suburban_lons)
        users['location_types'].extend([2] * n_suburban)
        
        # Rural users: outer ring
        angles = np.random.uniform(0, 2*np.pi, n_rural)
        radii = np.random.uniform(0.5, 1.5, n_rural)  # degrees
        rural_lats = self.center_lat + radii * np.sin(angles)
        rural_lons = self.center_lon + radii * np.cos(angles)
        users['lats'].extend(rural_lats)
        users['lons'].extend(rural_lons)
        users['location_types'].extend([3] * n_rural)
        
        # Convert to arrays
        users['lats'] = np.array(users['lats'])
        users['lons'] = np.array(users['lons'])
        users['location_types'] = np.array(users['location_types'])
        
        return users

