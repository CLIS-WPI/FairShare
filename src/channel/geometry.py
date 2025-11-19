"""
Geometric calculations for satellite-ground communication.

Computes Doppler shift, elevation angle, azimuth, and slant range
for LEO satellite-ground station links.
"""

import numpy as np
from typing import Tuple
from datetime import datetime


class SatelliteGeometry:
    """
    Geometric calculations for satellite-ground communication links.
    """
    
    # Constants
    EARTH_RADIUS = 6371000.0  # meters
    SPEED_OF_LIGHT = 299792458.0  # m/s
    
    def __init__(self, ground_station_lat: float, ground_station_lon: float, 
                 ground_station_alt: float = 0.0):
        """
        Initialize with ground station coordinates.
        
        Args:
            ground_station_lat: Latitude in degrees
            ground_station_lon: Longitude in degrees
            ground_station_alt: Altitude above sea level in meters
        """
        self.gs_lat = np.radians(ground_station_lat)
        self.gs_lon = np.radians(ground_station_lon)
        self.gs_alt = ground_station_alt
        
        # Convert to ECEF
        self.gs_ecef = self._lat_lon_alt_to_ecef(
            ground_station_lat, ground_station_lon, ground_station_alt
        )
    
    def _lat_lon_alt_to_ecef(self, lat: float, lon: float, alt: float) -> np.ndarray:
        """Convert lat/lon/alt to ECEF coordinates."""
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        
        # WGS84 ellipsoid parameters
        a = 6378137.0  # semi-major axis
        e2 = 0.00669437999014  # first eccentricity squared
        
        N = a / np.sqrt(1 - e2 * np.sin(lat_rad)**2)
        
        x = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
        y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
        z = (N * (1 - e2) + alt) * np.sin(lat_rad)
        
        return np.array([x, y, z])
    
    def eci_to_ecef(self, position_eci: np.ndarray, dt: datetime) -> np.ndarray:
        """
        Convert ECI to ECEF coordinates accounting for Earth rotation.
        
        Args:
            position_eci: Position in ECI frame (meters)
            dt: Datetime for Earth rotation calculation
            
        Returns:
            Position in ECEF frame (meters)
        """
        # Greenwich Mean Sidereal Time
        jd = self._datetime_to_julian_day(dt)
        gmst = self._julian_day_to_gmst(jd)
        
        # Rotation matrix around Z-axis
        R = np.array([
            [np.cos(gmst), np.sin(gmst), 0],
            [-np.sin(gmst), np.cos(gmst), 0],
            [0, 0, 1]
        ])
        
        return R @ position_eci
    
    def _datetime_to_julian_day(self, dt: datetime) -> float:
        """Convert datetime to Julian day."""
        a = (14 - dt.month) // 12
        y = dt.year + 4800 - a
        m = dt.month + 12 * a - 3
        
        jdn = dt.day + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045
        jd = jdn + (dt.hour - 12) / 24.0 + dt.minute / 1440.0 + dt.second / 86400.0
        
        return jd
    
    def _julian_day_to_gmst(self, jd: float) -> float:
        """Convert Julian day to Greenwich Mean Sidereal Time in radians."""
        d = jd - 2451545.0
        gmst = 18.697374558 + 24.06570982441908 * d
        gmst = (gmst % 24.0) * np.pi / 12.0
        return gmst
    
    def compute_geometry(self, satellite_position_eci: np.ndarray, 
                        satellite_velocity_eci: np.ndarray,
                        dt: datetime) -> dict:
        """
        Compute all geometric parameters for satellite-ground link.
        
        Args:
            satellite_position_eci: Satellite position in ECI (meters)
            satellite_velocity_eci: Satellite velocity in ECI (m/s)
            dt: Current datetime
            
        Returns:
            Dictionary with elevation, azimuth, slant_range, doppler_shift
        """
        # Convert satellite position to ECEF
        sat_ecef = self.eci_to_ecef(satellite_position_eci, dt)
        
        # Vector from ground station to satellite
        r_vec = sat_ecef - self.gs_ecef
        slant_range = np.linalg.norm(r_vec)
        
        # Convert to local ENU (East-North-Up) frame
        lat, lon = self.gs_lat, self.gs_lon
        
        # Rotation matrix from ECEF to ENU
        R = np.array([
            [-np.sin(lon), np.cos(lon), 0],
            [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
            [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
        ])
        
        r_enu = R @ r_vec
        
        # Elevation angle (angle above horizon)
        elevation = np.arcsin(r_enu[2] / slant_range)
        elevation_deg = np.degrees(elevation)
        
        # Azimuth angle (compass direction, 0=North, 90=East)
        azimuth = np.arctan2(r_enu[0], r_enu[1])
        azimuth_deg = np.degrees(azimuth)
        if azimuth_deg < 0:
            azimuth_deg += 360
        
        # Doppler shift calculation
        # Convert velocity to ECEF
        sat_vel_ecef = self.eci_to_ecef(satellite_velocity_eci, dt)
        # Approximate: relative velocity along line of sight
        unit_vec = r_vec / slant_range
        radial_velocity = np.dot(sat_vel_ecef, unit_vec)
        
        # Doppler shift: f_d = -f_c * v_r / c
        # Using typical LEO frequency (e.g., 12 GHz)
        freq_carrier = 12e9  # Hz (can be parameterized)
        doppler_shift = -freq_carrier * radial_velocity / self.SPEED_OF_LIGHT
        
        return {
            'elevation': elevation_deg,
            'elevation_rad': elevation,
            'azimuth': azimuth_deg,
            'azimuth_rad': azimuth,
            'slant_range': slant_range,
            'doppler_shift': doppler_shift,
            'radial_velocity': radial_velocity
        }
    
    def is_visible(self, satellite_position_eci: np.ndarray, dt: datetime, 
                   min_elevation: float = 5.0) -> bool:
        """
        Check if satellite is visible above minimum elevation.
        
        Args:
            satellite_position_eci: Satellite position in ECI
            dt: Current datetime
            min_elevation: Minimum elevation angle in degrees
            
        Returns:
            True if satellite is visible
        """
        geom = self.compute_geometry(satellite_position_eci, 
                                    np.zeros(3), dt)
        return geom['elevation'] >= min_elevation
    
    def compute_path_loss(self, slant_range: float, frequency_hz: float) -> float:
        """
        Compute free-space path loss.
        
        Args:
            slant_range: Distance in meters
            frequency_hz: Carrier frequency in Hz
            
        Returns:
            Path loss in dB
        """
        wavelength = self.SPEED_OF_LIGHT / frequency_hz
        path_loss_db = 20 * np.log10(4 * np.pi * slant_range / wavelength)
        return path_loss_db
    
    def compute_elevation(self, user_xyz: np.ndarray, sat_xyz: np.ndarray) -> float:
        """
        Compute elevation angle between user and satellite positions.
        
        Args:
            user_xyz: User position in ECEF (meters) [x, y, z]
            sat_xyz: Satellite position in ECEF (meters) [x, y, z]
            
        Returns:
            Elevation angle in degrees
        """
        # Vector from user to satellite
        r_vec = sat_xyz - user_xyz
        slant_range = np.linalg.norm(r_vec)
        
        # User position vector (from Earth center)
        user_range = np.linalg.norm(user_xyz)
        
        # Angle between user position and line-of-sight
        cos_angle = np.dot(user_xyz, r_vec) / (user_range * slant_range)
        angle = np.arccos(np.clip(cos_angle, -1, 1))
        
        # Elevation = 90 - angle from vertical
        elevation_rad = np.pi/2 - angle
        elevation_deg = np.degrees(elevation_rad)
        
        return elevation_deg
    
    def elevation(self, user_location: Tuple[float, float],
                  satellite_position_eci: np.ndarray,
                  dt: datetime) -> float:
        """
        Compute elevation angle for user-satellite link.
        
        Args:
            user_location: (lat, lon) tuple in degrees
            satellite_position_eci: Satellite position in ECI (meters)
            dt: Current datetime
            
        Returns:
            Elevation angle in degrees
        """
        # Create temporary geometry for user location
        user_geom = SatelliteGeometry(user_location[0], user_location[1])
        geom = user_geom.compute_geometry(satellite_position_eci, np.zeros(3), dt)
        return geom['elevation']
    
    def compute_doppler(self, fc: float, sat_vel: np.ndarray, user_vel: np.ndarray = None) -> float:
        """
        Compute Doppler shift from relative velocity.
        
        Args:
            fc: Carrier frequency in Hz
            sat_vel: Satellite velocity vector (m/s)
            user_vel: User velocity vector (m/s), defaults to zero
            
        Returns:
            Doppler shift in Hz
        """
        if user_vel is None:
            user_vel = np.zeros(3)
        
        # Relative velocity
        rel_vel = sat_vel - user_vel
        
        # Assume velocity is along line-of-sight (simplified)
        # In practice, need to project onto LOS vector
        radial_velocity = np.linalg.norm(rel_vel)
        
        # Doppler shift: f_d = -f_c * v_r / c
        doppler_shift = -fc * radial_velocity / self.SPEED_OF_LIGHT
        
        return doppler_shift
    
    def doppler(self, user_location: Tuple[float, float],
               satellite_position_eci: np.ndarray,
               satellite_velocity_eci: np.ndarray,
               dt: datetime,
               frequency_hz: float = 12e9) -> float:
        """
        Compute Doppler shift for user-satellite link.
        
        Args:
            user_location: (lat, lon) tuple in degrees
            satellite_position_eci: Satellite position in ECI (meters)
            satellite_velocity_eci: Satellite velocity in ECI (m/s)
            dt: Current datetime
            frequency_hz: Carrier frequency in Hz
            
        Returns:
            Doppler shift in Hz
        """
        user_geom = SatelliteGeometry(user_location[0], user_location[1])
        geom = user_geom.compute_geometry(satellite_position_eci, satellite_velocity_eci, dt)
        return geom['doppler_shift']
    
    def compute_slant_range(self, user_xyz: np.ndarray, sat_xyz: np.ndarray) -> float:
        """
        Compute slant range (distance) between user and satellite.
        
        Args:
            user_xyz: User position in ECEF (meters) [x, y, z]
            sat_xyz: Satellite position in ECEF (meters) [x, y, z]
            
        Returns:
            Slant range in meters
        """
        r_vec = sat_xyz - user_xyz
        return np.linalg.norm(r_vec)
    
    def distance(self, user_location: Tuple[float, float],
                satellite_position_eci: np.ndarray,
                dt: datetime) -> float:
        """
        Compute slant range (distance) for user-satellite link.
        
        Args:
            user_location: (lat, lon) tuple in degrees
            satellite_position_eci: Satellite position in ECI (meters)
            dt: Current datetime
            
        Returns:
            Slant range in meters
        """
        user_geom = SatelliteGeometry(user_location[0], user_location[1])
        geom = user_geom.compute_geometry(satellite_position_eci, np.zeros(3), dt)
        return geom['slant_range']

