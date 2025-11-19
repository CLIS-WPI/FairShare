"""
TLE-based orbit propagation for LEO satellites using sgp4/skyfield.

This module provides orbit propagation capabilities for LEO satellite constellations
using Two-Line Element (TLE) data. Supports both sgp4 and skyfield libraries.
"""

import numpy as np
from typing import Tuple, Optional, List
from datetime import datetime, timedelta
try:
    from sgp4.api import Satrec, jday
    from sgp4.conveniences import jday_datetime
    SGP4_AVAILABLE = True
except ImportError:
    SGP4_AVAILABLE = False
    print("Warning: sgp4 not available. Install with: pip install sgp4")

try:
    from skyfield.api import load, EarthSatellite
    SKYFIELD_AVAILABLE = True
except ImportError:
    SKYFIELD_AVAILABLE = False
    print("Warning: skyfield not available. Install with: pip install skyfield")


class OrbitPropagator:
    """
    Orbit propagator for LEO satellites using TLE data.
    
    Supports both sgp4 and skyfield backends for propagation.
    """
    
    def __init__(self, tle_file: Optional[str] = None, backend: str = 'sgp4'):
        """
        Initialize orbit propagator.
        
        Args:
            tle_file: Path to TLE file containing satellite orbital elements
            backend: 'sgp4' or 'skyfield' for propagation engine
        """
        self.backend = backend
        self.satellites = {}
        self.ts = None
        
        if backend == 'skyfield' and SKYFIELD_AVAILABLE:
            self.ts = load.timescale()
        
        if tle_file:
            self.load_tle_file(tle_file)
    
    def load_tle_file(self, tle_file: str) -> None:
        """
        Load TLE data from file (from data/tle/ directory).
        
        Args:
            tle_file: Path to TLE file containing satellite orbital elements
        """
        satellites = {}
        
        with open(tle_file, 'r') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            # Skip empty lines or lines that start with '1 ' (TLE line 1)
            line = lines[i].strip()
            if not line or line.startswith('1 '):
                i += 1
                continue
            
            # Satellite name (non-empty line that doesn't start with '1' or '2')
            if i + 2 < len(lines):
                name = line
                line1 = lines[i + 1].strip()
                line2 = lines[i + 2].strip()
                
                # Validate TLE format
                if line1.startswith('1 ') and line2.startswith('2 '):
                    if self.backend == 'sgp4' and SGP4_AVAILABLE:
                        satellite = Satrec.twoline2rv(line1, line2)
                        satellites[name] = satellite
                    elif self.backend == 'skyfield' and SKYFIELD_AVAILABLE:
                        satellite = EarthSatellite(line1, line2, name, self.ts)
                        satellites[name] = satellite
                    
                    i += 3
                else:
                    i += 1
            else:
                i += 1
        
        self.satellites.update(satellites)
        print(f"Loaded {len(satellites)} satellites from {tle_file}")
    
    def add_satellite(self, name: str, line1: str, line2: str) -> None:
        """Add a single satellite from TLE lines."""
        if self.backend == 'sgp4' and SGP4_AVAILABLE:
            satellite = Satrec.twoline2rv(line1, line2)
            self.satellites[name] = satellite
        elif self.backend == 'skyfield' and SKYFIELD_AVAILABLE:
            satellite = EarthSatellite(line1, line2, name, self.ts)
            self.satellites[name] = satellite
    
    def propagate(self, satellite_name: str, dt: datetime) -> Tuple[np.ndarray, np.ndarray]:
        """
        Propagate satellite to given datetime.
        
        Args:
            satellite_name: Name of satellite
            dt: Target datetime
            
        Returns:
            Tuple of (position_m, velocity_m_s) in ECI frame
            Position and velocity are in meters and m/s respectively
        """
        if satellite_name not in self.satellites:
            raise ValueError(f"Satellite {satellite_name} not found")
        
        sat = self.satellites[satellite_name]
        
        if self.backend == 'sgp4' and SGP4_AVAILABLE:
            jd, fr = jday_datetime(dt)
            error, r, v = sat.sgp4(jd, fr)
            
            if error != 0:
                raise RuntimeError(f"SGP4 propagation error: {error}")
            
            # Convert from km to meters
            position = np.array(r) * 1000  # km to m
            velocity = np.array(v) * 1000  # km/s to m/s
            
        elif self.backend == 'skyfield' and SKYFIELD_AVAILABLE:
            t = self.ts.from_datetime(dt)
            geocentric = sat.at(t)
            position = geocentric.position.m  # meters
            velocity = geocentric.velocity.m_per_s  # m/s
            
        else:
            raise RuntimeError(f"Backend {self.backend} not available")
        
        return position, velocity
    
    def propagate_ecef(self, satellite_name: str, dt: datetime) -> Tuple[np.ndarray, np.ndarray]:
        """
        Propagate satellite and return position in ECEF frame.
        
        Args:
            satellite_name: Name of satellite
            dt: Target datetime
            
        Returns:
            Tuple of (position_ecef_m, velocity_ecef_m_s) in ECEF frame
        """
        # Get ECI position and velocity
        position_eci, velocity_eci = self.propagate(satellite_name, dt)
        
        # Convert ECI to ECEF (simplified rotation)
        # In practice, use proper Earth rotation matrix
        from .geometry import SatelliteGeometry
        temp_geom = SatelliteGeometry(0.0, 0.0)
        position_ecef = temp_geom.eci_to_ecef(position_eci, dt)
        
        # Velocity conversion (simplified - should account for Earth rotation)
        velocity_ecef = temp_geom.eci_to_ecef(velocity_eci, dt)
        
        return position_ecef, velocity_ecef
    
    def propagate_batch(self, satellite_name: str, dts: List[datetime]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Propagate satellite for multiple timestamps.
        
        Args:
            satellite_name: Name of satellite
            dts: List of target datetimes
            
        Returns:
            Tuple of (positions, velocities) arrays
        """
        positions = []
        velocities = []
        
        for dt in dts:
            pos, vel = self.propagate(satellite_name, dt)
            positions.append(pos)
            velocities.append(vel)
        
        return np.array(positions), np.array(velocities)
    
    def get_altitude(self, satellite_name: str, dt: datetime) -> float:
        """
        Get satellite altitude above Earth's surface.
        
        Args:
            satellite_name: Name of satellite
            dt: Target datetime
            
        Returns:
            Altitude in meters
        """
        position, _ = self.propagate(satellite_name, dt)
        altitude = np.linalg.norm(position) - 6371000  # Earth radius in meters
        return altitude
    
    def get_orbital_period(self, satellite_name: str) -> float:
        """
        Estimate orbital period from TLE data.
        
        Args:
            satellite_name: Name of satellite
            
        Returns:
            Period in seconds
        """
        if satellite_name not in self.satellites:
            raise ValueError(f"Satellite {satellite_name} not found")
        
        sat = self.satellites[satellite_name]
        
        if self.backend == 'sgp4' and SGP4_AVAILABLE:
            # Mean motion in revs per day
            n = sat.no_kozai  # radians per minute
            period_minutes = 2 * np.pi / n
            return period_minutes * 60  # seconds
        else:
            # Fallback: propagate for one orbit
            dt = datetime.now()
            pos1, _ = self.propagate(satellite_name, dt)
            # Approximate period from altitude
            alt = np.linalg.norm(pos1) - 6371000
            # Simplified: T ≈ 2π√(a³/μ) where a ≈ R_earth + alt
            mu = 3.986004418e14  # m³/s²
            a = 6371000 + alt
            period = 2 * np.pi * np.sqrt(a**3 / mu)
            return period

