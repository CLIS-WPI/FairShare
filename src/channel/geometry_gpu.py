"""
GPU-accelerated geometric calculations for satellite-ground communication.

Uses TensorFlow for batch processing of geometry computations.
"""

import numpy as np
import tensorflow as tf
from typing import Tuple, List, Dict, Any
from datetime import datetime


class SatelliteGeometryGPU:
    """
    GPU-accelerated geometric calculations for satellite-ground communication links.
    
    Processes multiple user-satellite pairs in parallel on GPU.
    """
    
    # Constants
    EARTH_RADIUS = 6371000.0  # meters
    SPEED_OF_LIGHT = 299792458.0  # m/s
    
    def __init__(self, device: str = '/GPU:0'):
        """
        Initialize GPU geometry calculator.
        
        Args:
            device: TensorFlow device (e.g., '/GPU:0')
        """
        self.device = device
        self._build_gpu_functions()
    
    def _build_gpu_functions(self):
        """Build TensorFlow functions for GPU computation."""
        
        @tf.function(jit_compile=True)
        def compute_elevation_batch(
            user_positions_ecef: tf.Tensor,  # [num_users, 3]
            sat_positions_ecef: tf.Tensor,   # [num_sats, 3]
            user_lats: tf.Tensor,            # [num_users]
            user_lons: tf.Tensor             # [num_users]
        ) -> tf.Tensor:
            """
            Compute elevation angles for all user-satellite pairs.
            
            Args:
                user_positions_ecef: User positions in ECEF [num_users, 3]
                sat_positions_ecef: Satellite positions in ECEF [num_sats, 3]
                user_lats: User latitudes in radians [num_users]
                user_lons: User longitudes in radians [num_users]
            
            Returns:
                Elevation angles in degrees [num_users, num_sats]
            """
            # Expand dimensions for broadcasting
            # user_positions: [num_users, 1, 3]
            # sat_positions: [1, num_sats, 3]
            user_pos = tf.expand_dims(user_positions_ecef, axis=1)  # [num_users, 1, 3]
            sat_pos = tf.expand_dims(sat_positions_ecef, axis=0)   # [1, num_sats, 3]
            
            # Vector from user to satellite
            vec_user_to_sat = sat_pos - user_pos  # [num_users, num_sats, 3]
            
            # Distance (slant range)
            slant_range = tf.linalg.norm(vec_user_to_sat, axis=2)  # [num_users, num_sats]
            
            # User position magnitude (Earth radius)
            user_r = tf.linalg.norm(user_positions_ecef, axis=1)  # [num_users]
            user_r = tf.expand_dims(user_r, axis=1)  # [num_users, 1]
            
            # Compute elevation angle
            # elevation = arcsin((sat_altitude - user_altitude) / slant_range)
            # For simplicity, use: elevation = arcsin(dot_product / (|user_pos| * |vec|))
            # More accurate: use local ENU frame
            
            # Convert to local ENU (East-North-Up) for each user
            # ENU rotation matrices for each user
            sin_lat = tf.sin(user_lats)  # [num_users]
            cos_lat = tf.cos(user_lats)
            sin_lon = tf.sin(user_lons)
            cos_lon = tf.cos(user_lons)
            
            # Expand for broadcasting
            sin_lat = tf.expand_dims(sin_lat, axis=1)  # [num_users, 1]
            cos_lat = tf.expand_dims(cos_lat, axis=1)
            sin_lon = tf.expand_dims(sin_lon, axis=1)
            cos_lon = tf.expand_dims(cos_lon, axis=1)
            
            # ENU rotation matrix components
            # R_ecef_to_enu = [[-sin_lon, cos_lon, 0],
            #                   [-sin_lat*cos_lon, -sin_lat*sin_lon, cos_lat],
            #                   [cos_lat*cos_lon, cos_lat*sin_lon, sin_lat]]
            
            # Transform vec_user_to_sat to ENU for each user
            # vec_enu = R @ vec_ecef
            vec_x = vec_user_to_sat[:, :, 0]  # [num_users, num_sats]
            vec_y = vec_user_to_sat[:, :, 1]
            vec_z = vec_user_to_sat[:, :, 2]
            
            # ENU components
            east = -sin_lon * vec_x + cos_lon * vec_y  # [num_users, num_sats]
            north = -sin_lat * cos_lon * vec_x - sin_lat * sin_lon * vec_y + cos_lat * vec_z
            up = cos_lat * cos_lon * vec_x + cos_lat * sin_lon * vec_y + sin_lat * vec_z
            
            # Elevation = arcsin(up / slant_range)
            # Avoid division by zero
            slant_range_safe = tf.maximum(slant_range, 1.0)
            sin_elevation = up / slant_range_safe
            sin_elevation = tf.clip_by_value(sin_elevation, -1.0, 1.0)
            
            elevation_rad = tf.asin(sin_elevation)
            # Convert radians to degrees: deg = rad * 180 / π
            elevation_deg = elevation_rad * 180.0 / np.pi
            
            return elevation_deg
        
        self.compute_elevation_batch = compute_elevation_batch
    
    def compute_visibility_batch(
        self,
        user_positions_ecef: np.ndarray,  # [num_users, 3]
        user_lats: np.ndarray,            # [num_users] in degrees
        user_lons: np.ndarray,            # [num_users] in degrees
        sat_positions_ecef: np.ndarray,   # [num_sats, 3]
        min_elevation_deg: float = 25.0
    ) -> np.ndarray:
        """
        Compute visibility matrix for all user-satellite pairs.
        
        Args:
            user_positions_ecef: User positions in ECEF [num_users, 3]
            user_lats: User latitudes in degrees [num_users]
            user_lons: User longitudes in degrees [num_users]
            sat_positions_ecef: Satellite positions in ECEF [num_sats, 3]
            min_elevation_deg: Minimum elevation angle in degrees
        
        Returns:
            Visibility matrix [num_users, num_sats] where True = visible
        """
        with tf.device(self.device):
            # Convert to tensors
            user_pos_tf = tf.constant(user_positions_ecef, dtype=tf.float32)
            sat_pos_tf = tf.constant(sat_positions_ecef, dtype=tf.float32)
            user_lats_rad = tf.constant(np.radians(user_lats), dtype=tf.float32)
            user_lons_rad = tf.constant(np.radians(user_lons), dtype=tf.float32)
            
            # Compute elevations
            elevations = self.compute_elevation_batch(
                user_pos_tf, sat_pos_tf, user_lats_rad, user_lons_rad
            )
            
            # Check visibility (elevation >= min_elevation)
            visibility = elevations >= min_elevation_deg
            
            # Convert back to numpy
            return visibility.numpy()
    
    def compute_slant_ranges_batch(
        self,
        user_positions_ecef: np.ndarray,  # [num_users, 3]
        sat_positions_ecef: np.ndarray    # [num_sats, 3]
    ) -> np.ndarray:
        """
        Compute slant ranges for all user-satellite pairs.
        
        Args:
            user_positions_ecef: User positions in ECEF [num_users, 3]
            sat_positions_ecef: Satellite positions in ECEF [num_sats, 3]
        
        Returns:
            Slant ranges in meters [num_users, num_sats]
        """
        with tf.device(self.device):
            user_pos_tf = tf.constant(user_positions_ecef, dtype=tf.float32)
            sat_pos_tf = tf.constant(sat_positions_ecef, dtype=tf.float32)
            
            # Expand for broadcasting
            user_pos = tf.expand_dims(user_pos_tf, axis=1)  # [num_users, 1, 3]
            sat_pos = tf.expand_dims(sat_pos_tf, axis=0)   # [1, num_sats, 3]
            
            # Vector from user to satellite
            vec = sat_pos - user_pos  # [num_users, num_sats, 3]
            
            # Slant range
            slant_range = tf.linalg.norm(vec, axis=2)  # [num_users, num_sats]
            
            return slant_range.numpy()

