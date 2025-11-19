"""
Tests for geometry module.
"""

import pytest
import numpy as np
from datetime import datetime
from src.channel.geometry import SatelliteGeometry


class TestSatelliteGeometry:
    """Test cases for SatelliteGeometry class."""
    
    def test_compute_elevation(self):
        """Test compute_elevation function."""
        geom = SatelliteGeometry(lat=42.0, lon=-71.0)
        user_xyz = np.array([6371000, 0, 0])  # User on equator
        sat_xyz = np.array([7000e3, 0, 0])  # Satellite above
        
        elevation = geom.compute_elevation(user_xyz, sat_xyz)
        
        assert isinstance(elevation, float)
        assert -90 <= elevation <= 90
    
    def test_elevation_calculation(self):
        """Test elevation angle calculation."""
        geom = SatelliteGeometry(lat=42.0, lon=-71.0)
        sat_pos = np.array([7000e3, 0, 0])  # 7000 km in x direction
        dt = datetime.now()
        
        elevation = geom.elevation((42.0, -71.0), sat_pos, dt)
        
        assert isinstance(elevation, float)
        assert -90 <= elevation <= 90
    
    def test_compute_doppler(self):
        """Test compute_doppler function."""
        geom = SatelliteGeometry(lat=42.0, lon=-71.0)
        fc = 12e9  # 12 GHz
        sat_vel = np.array([0, 7500, 0])  # 7.5 km/s
        
        doppler = geom.compute_doppler(fc, sat_vel)
        
        assert isinstance(doppler, float)
        assert abs(doppler) < 1e6  # Reasonable range for LEO
    
    def test_doppler_calculation(self):
        """Test Doppler shift calculation."""
        geom = SatelliteGeometry(lat=42.0, lon=-71.0)
        sat_pos = np.array([7000e3, 0, 0])
        sat_vel = np.array([0, 7500, 0])  # 7.5 km/s in y direction
        dt = datetime.now()
        
        doppler = geom.doppler((42.0, -71.0), sat_pos, sat_vel, dt)
        
        assert isinstance(doppler, float)
    
    def test_compute_slant_range(self):
        """Test compute_slant_range function."""
        geom = SatelliteGeometry(lat=42.0, lon=-71.0)
        user_xyz = np.array([6371000, 0, 0])
        sat_xyz = np.array([7000e3, 0, 0])
        
        distance = geom.compute_slant_range(user_xyz, sat_xyz)
        
        assert isinstance(distance, float)
        assert distance > 0
        assert distance < 1e7  # Reasonable range
    
    def test_distance_calculation(self):
        """Test slant range calculation."""
        geom = SatelliteGeometry(lat=42.0, lon=-71.0)
        sat_pos = np.array([7000e3, 0, 0])
        dt = datetime.now()
        
        distance = geom.distance((42.0, -71.0), sat_pos, dt)
        
        assert isinstance(distance, float)
        assert distance > 0
    
    def test_eci_to_ecef_conversion(self):
        """Test ECI to ECEF coordinate conversion."""
        geom = SatelliteGeometry(lat=0.0, lon=0.0)
        eci_pos = np.array([7000e3, 0, 0])
        dt = datetime.now()
        
        ecef_pos = geom.eci_to_ecef(eci_pos, dt)
        
        assert ecef_pos.shape == (3,)
        assert isinstance(ecef_pos, np.ndarray)

