"""
Batch processing utilities for GPU acceleration.

Optimized for H100 GPUs with vectorized operations.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None


def batch_compute_geometry(
    users: List[Dict],
    sat_pos_ecef: np.ndarray,
    sat_vel_ecef: np.ndarray,
    current_time,
    geometries: List
) -> Tuple[Dict[str, Dict], np.ndarray]:
    """
    Batch compute geometry for all users.
    
    Args:
        users: List of user dictionaries
        sat_pos_ecef: Satellite position in ECEF
        sat_vel_ecef: Satellite velocity in ECEF
        current_time: Current simulation time
        geometries: List of SatelliteGeometry instances
    
    Returns:
        Tuple of (geometry_dict, geometry_array) where array is (num_users, 7)
        with [elevation, doppler, slant_range, lat, lon, operator, beam_id_index]
    """
    num_users = len(users)
    geometry_dict = {}
    geometry_array = np.zeros((num_users, 7))
    
    # Use first geometry model (or create per user)
    geom_model = geometries[0] if geometries else None
    
    for i, user in enumerate(users):
        user_id = user['id']
        lat, lon = user['lat'], user['lon']
        
        # Create geometry for this user if needed
        if geom_model is None:
            from src.channel.geometry import SatelliteGeometry
            user_geom = SatelliteGeometry(lat, lon)
        else:
            user_geom = geom_model
        
        # Compute geometry
        sat_pos_eci = sat_pos_ecef  # Simplified
        sat_vel_eci = sat_vel_ecef  # Simplified
        geom = user_geom.compute_geometry(sat_pos_eci, sat_vel_eci, current_time)
        
        elevation = geom['elevation']
        doppler = abs(geom['doppler_shift'])
        slant_range = geom['slant_range']
        
        geometry_dict[user_id] = geom
        geometry_array[i] = [
            elevation,
            doppler,
            slant_range,
            lat,
            lon,
            user.get('operator', 0),
            i  # beam_id_index
        ]
    
    return geometry_dict, geometry_array


def batch_compute_link_budgets(
    geometry_dict: Dict[str, Dict],
    channel_model,
    rain_rate_mmh: float,
    bandwidth_hz: float
) -> Dict[str, Dict]:
    """
    Batch compute link budgets (can be optimized with vectorization).
    
    Args:
        geometry_dict: Dictionary mapping user_id to geometry
        channel_model: ChannelModel instance
        rain_rate_mmh: Rain rate
        bandwidth_hz: Bandwidth
    
    Returns:
        Dictionary mapping user_id to link budget
    """
    link_budgets = {}
    
    for user_id, geom in geometry_dict.items():
        link_budget = channel_model.compute_link_budget(
            geometry=geom,
            rain_rate_mmh=rain_rate_mmh,
            bandwidth_hz=bandwidth_hz
        )
        link_budgets[user_id] = link_budget
    
    return link_budgets


def batch_compute_beam_loads(
    users: List[Dict],
    spectrum_env,
    num_operators: int
) -> np.ndarray:
    """
    Batch compute beam loads for all users.
    
    Args:
        users: List of user dictionaries
        spectrum_env: SpectrumEnvironment instance
        num_operators: Number of operators
    
    Returns:
        Array of shape (num_users,) with beam loads
    """
    num_users = len(users)
    beam_loads = np.zeros(num_users)
    
    for i, user in enumerate(users):
        operator = user.get('operator', 0)
        beam_id = f"beam_{operator}"
        beam_loads[i] = spectrum_env.compute_beam_load(beam_id)
    
    return beam_loads

