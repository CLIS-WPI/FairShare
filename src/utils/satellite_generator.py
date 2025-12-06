"""
Satellite Generator for Multi-Operator Scenarios

Generates realistic satellite positions for multiple operators (Starlink, OneWeb, Kuiper).
Each operator has ~20 visible satellites above 25° elevation.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import logging
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None

logger = logging.getLogger(__name__)


class MultiOperatorSatelliteGenerator:
    """
    Generate satellite positions for multiple operators.
    
    For NYC scenario:
    - Op_A (Starlink-like): 20 satellites, altitude ~550 km
    - Op_B (OneWeb-like): 20 satellites, altitude ~1200 km  
    - Op_C (Kuiper-like): 20 satellites, altitude ~630 km
    """
    
    EARTH_RADIUS_KM = 6371.0
    
    def __init__(
        self,
        operators: List[Dict],
        center_lat: float = 40.75,
        center_lon: float = -73.975,
        min_elevation_deg: float = 25.0
    ):
        """
        Initialize satellite generator.
        
        Args:
            operators: List of operator configs with altitude, inclination, etc.
            center_lat: Center latitude for visibility calculation
            center_lon: Center longitude for visibility calculation
            min_elevation_deg: Minimum elevation angle (default: 25°)
        """
        self.operators = operators
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.min_elevation_deg = min_elevation_deg
        
        # Operator configurations
        self.operator_configs = {
            'Op_A': {'altitude_km': 550.0, 'inclination_deg': 53.0, 'num_sats': 20},
            'Op_B': {'altitude_km': 1200.0, 'inclination_deg': 87.4, 'num_sats': 20},
            'Op_C': {'altitude_km': 630.0, 'inclination_deg': 51.9, 'num_sats': 20}
        }
    
    def generate_satellite_positions(
        self,
        timestamp: datetime,
        num_sats_per_operator: int = 20
    ) -> Dict[str, List[Dict]]:
        """
        Generate satellite positions for all operators.
        
        Args:
            timestamp: Current simulation time
            num_sats_per_operator: Number of satellites per operator
            
        Returns:
            Dictionary mapping operator_id -> list of satellite positions
        """
        all_satellites = {}
        
        for op_config in self.operators:
            op_id = op_config.get('name', 'Op_A')
            op_type = op_config.get('type', 'starlink_like')
            
            # Get operator-specific config
            if op_id in self.operator_configs:
                config = self.operator_configs[op_id]
            else:
                # Default config
                config = {'altitude_km': 550.0, 'inclination_deg': 53.0, 'num_sats': num_sats_per_operator}
            
            # Generate satellites for this operator
            satellites = self._generate_operator_satellites(
                op_id=op_id,
                altitude_km=config['altitude_km'],
                inclination_deg=config['inclination_deg'],
                num_sats=config['num_sats'],
                timestamp=timestamp
            )
            
            all_satellites[op_id] = satellites
        
        return all_satellites
    
    def _generate_operator_satellites(
        self,
        op_id: str,
        altitude_km: float,
        inclination_deg: float,
        num_sats: int,
        timestamp: datetime
    ) -> List[Dict]:
        """
        Generate satellite positions for a single operator.
        
        Uses simplified orbital mechanics to generate realistic positions.
        """
        satellites = []
        
        # Calculate orbital period
        semi_major_axis_km = self.EARTH_RADIUS_KM + altitude_km
        orbital_period_s = 2 * np.pi * np.sqrt(
            (semi_major_axis_km * 1000) ** 3 / (3.986004418e14)
        )
        
        # Time-based phase (satellites move over time)
        time_phase = (timestamp.timestamp() % orbital_period_s) / orbital_period_s
        
        for sat_idx in range(num_sats):
            sat_id = f"{op_id}_sat{sat_idx:02d}"
            
            # Distribute satellites in orbital plane
            # Use RAAN (Right Ascension of Ascending Node) and mean anomaly
            raan = 2 * np.pi * sat_idx / num_sats + time_phase * 2 * np.pi
            mean_anomaly = 2 * np.pi * (sat_idx % 10) / 10 + time_phase * 2 * np.pi
            
            # Convert to ECEF position (simplified)
            # In real implementation, use proper orbital mechanics
            position_ecef = self._orbital_to_ecef(
                altitude_km=altitude_km,
                inclination_deg=inclination_deg,
                raan_rad=raan,
                mean_anomaly_rad=mean_anomaly,
                timestamp=timestamp
            )
            
            # Velocity (simplified - circular orbit)
            orbital_speed_ms = np.sqrt(3.986004418e14 / (semi_major_axis_km * 1000))
            velocity_ecef = self._compute_velocity(position_ecef, orbital_speed_ms, inclination_deg)
            
            satellites.append({
                'satellite_id': sat_id,
                'operator_id': op_id,
                'position_ecef': position_ecef,
                'velocity_ecef': velocity_ecef,
                'altitude_km': altitude_km,
                'inclination_deg': inclination_deg
            })
        
        return satellites
    
    def _orbital_to_ecef(
        self,
        altitude_km: float,
        inclination_deg: float,
        raan_rad: float,
        mean_anomaly_rad: float,
        timestamp: datetime
    ) -> np.ndarray:
        """
        Convert orbital elements to ECEF position (simplified).
        
        Generates satellite positions that are visible from the center location
        (above min_elevation_deg). Uses simplified orbital mechanics.
        """
        semi_major_axis_m = (self.EARTH_RADIUS_KM + altitude_km) * 1000
        
        # User position in ECEF
        user_lat_rad = np.radians(self.center_lat)
        user_lon_rad = np.radians(self.center_lon)
        user_r = self.EARTH_RADIUS_KM * 1000  # meters
        
        user_x = user_r * np.cos(user_lat_rad) * np.cos(user_lon_rad)
        user_y = user_r * np.cos(user_lat_rad) * np.sin(user_lon_rad)
        user_z = user_r * np.sin(user_lat_rad)
        user_pos = np.array([user_x, user_y, user_z])
        
        # Generate satellite positions that are visible from user
        # Strategy: Place satellites in a cone above the user (elevation > 25°)
        # Use mean_anomaly and raan to distribute satellites around the orbit
        
        # Calculate minimum slant range for 25° elevation
        # sin(elevation) = (sat_altitude - user_altitude) / slant_range
        # For 25°: slant_range_min = altitude / sin(25°)
        min_elevation_rad = np.radians(self.min_elevation_deg)
        slant_range_min = altitude_km * 1000 / np.sin(min_elevation_rad)
        
        # Maximum slant range (satellite at horizon would be at 0°, but we use 25°)
        # For LEO at 550km: max slant range ≈ 2000km
        max_slant_range = np.sqrt((semi_major_axis_m)**2 - (user_r)**2)
        
        # Generate a position that ensures visibility
        # Use mean_anomaly to vary the position along the orbit
        # Use raan to vary the orbital plane
        
        # OPTIMIZED: Use deterministic seed based on satellite index instead of hash
        # This avoids expensive hash computation for every satellite
        # Use a simple deterministic seed: sat_idx + timestamp seconds
        seed_value = int(timestamp.timestamp()) + int(mean_anomaly_rad * 1000) + int(raan_rad * 1000)
        np.random.seed(seed_value % 2**31)  # Use 2^31 to avoid overflow
        
        # Generate azimuth and elevation angles (elevation > 25°)
        # Azimuth: 0-360 degrees
        azimuth_rad = 2 * np.pi * (hash(f"{mean_anomaly_rad}_{raan_rad}") % 360) / 360.0
        
        # Elevation: 25° to 90° (directly overhead)
        # Use a distribution that favors higher elevations (more realistic)
        elevation_rad = np.radians(np.random.uniform(self.min_elevation_deg, 90.0))
        
        # Calculate satellite position relative to user
        # Convert spherical (azimuth, elevation, range) to ECEF
        # First, calculate slant range from elevation
        # elevation = arcsin((sat_altitude - user_altitude) / slant_range)
        # slant_range = (sat_altitude - user_altitude) / sin(elevation)
        sat_altitude_m = altitude_km * 1000
        user_altitude_m = 0.0  # Sea level
        slant_range = (sat_altitude_m - user_altitude_m) / np.sin(elevation_rad)
        
        # Ensure slant range is within reasonable bounds
        slant_range = np.clip(slant_range, slant_range_min, max_slant_range)
        
        # Convert to local ENU (East-North-Up) coordinates
        # Then rotate to ECEF
        # ENU: East = x, North = y, Up = z
        east = slant_range * np.cos(elevation_rad) * np.sin(azimuth_rad)
        north = slant_range * np.cos(elevation_rad) * np.cos(azimuth_rad)
        up = slant_range * np.sin(elevation_rad)
        
        # Rotation matrix from ENU to ECEF (at user location)
        # This is a simplified rotation - proper implementation would use full transformation
        sin_lat = np.sin(user_lat_rad)
        cos_lat = np.cos(user_lat_rad)
        sin_lon = np.sin(user_lon_rad)
        cos_lon = np.cos(user_lon_rad)
        
        # ENU to ECEF rotation matrix
        # [x_ecef]   [-sin_lon    -sin_lat*cos_lon   cos_lat*cos_lon] [east]
        # [y_ecef] = [ cos_lon    -sin_lat*sin_lon   cos_lat*sin_lon] [north]
        # [z_ecef]   [    0            cos_lat           sin_lat    ] [up]
        
        x_ecef = -sin_lon * east - sin_lat * cos_lon * north + cos_lat * cos_lon * up
        y_ecef = cos_lon * east - sin_lat * sin_lon * north + cos_lat * sin_lon * up
        z_ecef = cos_lat * north + sin_lat * up
        
        # Add to user position to get absolute ECEF
        sat_pos_ecef = user_pos + np.array([x_ecef, y_ecef, z_ecef])
        
        # Normalize to correct altitude
        sat_pos_norm = np.linalg.norm(sat_pos_ecef)
        if sat_pos_norm > 0:
            sat_pos_ecef = sat_pos_ecef * (semi_major_axis_m / sat_pos_norm)
        
        return sat_pos_ecef
    
    def _compute_velocity(
        self,
        position_ecef: np.ndarray,
        speed_ms: float,
        inclination_deg: float
    ) -> np.ndarray:
        """Compute velocity vector for circular orbit."""
        # Simplified: velocity perpendicular to position vector
        r = np.linalg.norm(position_ecef)
        if r == 0:
            return np.array([0.0, 0.0, 0.0])
        
        # Direction perpendicular to radius (tangential)
        # In circular orbit, velocity is perpendicular to position
        # Use cross product to get perpendicular direction
        north_pole = np.array([0, 0, 1])
        tangent = np.cross(position_ecef, north_pole)
        tangent = tangent / (np.linalg.norm(tangent) + 1e-10)
        
        velocity = speed_ms * tangent
        return velocity
    
    def filter_visible_satellites_batch(
        self,
        satellites: Dict[str, List[Dict]],
        user_lats: np.ndarray,
        user_lons: np.ndarray,
        user_positions_ecef: np.ndarray,
        timestamp: datetime,
        use_gpu: bool = True,
        device: str = '/GPU:0'
    ) -> List[List[Dict]]:
        """
        GPU-accelerated batch filtering of visible satellites for multiple users.
        
        Args:
            satellites: Dictionary of operator_id -> satellite list
            user_lats: User latitudes in degrees [num_users]
            user_lons: User longitudes in degrees [num_users]
            user_positions_ecef: User positions in ECEF [num_users, 3]
            timestamp: Current time
            use_gpu: Whether to use GPU acceleration
            device: TensorFlow device
        
        Returns:
            List of visible satellite lists, one per user
        """
        if not use_gpu:
            # Fallback to CPU (one-by-one)
            results = []
            for i, (lat, lon) in enumerate(zip(user_lats, user_lons)):
                visible = self.filter_visible_satellites(
                    satellites, lat, lon, timestamp
                )
                results.append(visible)
            return results
        
        # GPU-accelerated batch processing
        try:
            from src.channel.geometry_gpu import SatelliteGeometryGPU
            geom_gpu = SatelliteGeometryGPU(device=device)
        except ImportError:
            logger.warning("GPU geometry not available, falling back to CPU")
            return self.filter_visible_satellites_batch(
                satellites, user_lats, user_lons, user_positions_ecef,
                timestamp, use_gpu=False, device=device
            )
        
        # Collect all satellite positions
        all_sat_positions = []
        all_sat_metadata = []  # (op_id, sat_dict)
        
        for op_id, sat_list in satellites.items():
            for sat in sat_list:
                all_sat_positions.append(sat['position_ecef'])
                all_sat_metadata.append((op_id, sat))
        
        if not all_sat_positions:
            return [[] for _ in range(len(user_lats))]
        
        sat_positions_ecef = np.array(all_sat_positions)  # [num_sats, 3]
        
        # Compute visibility matrix on GPU
        visibility_matrix = geom_gpu.compute_visibility_batch(
            user_positions_ecef=user_positions_ecef,
            user_lats=user_lats,
            user_lons=user_lons,
            sat_positions_ecef=sat_positions_ecef,
            min_elevation_deg=self.min_elevation_deg
        )  # [num_users, num_sats]
        
        # Compute elevations and slant ranges for visible pairs
        # Note: tf is imported at module level, but we need to ensure it's available here
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for GPU batch processing")
        elevations = geom_gpu.compute_elevation_batch(
            tf.constant(user_positions_ecef, dtype=tf.float32),
            tf.constant(sat_positions_ecef, dtype=tf.float32),
            tf.constant(np.radians(user_lats), dtype=tf.float32),
            tf.constant(np.radians(user_lons), dtype=tf.float32)
        ).numpy()  # [num_users, num_sats]
        
        slant_ranges = geom_gpu.compute_slant_ranges_batch(
            user_positions_ecef, sat_positions_ecef
        )  # [num_users, num_sats]
        
        # Build results: for each user, list of visible satellites with geometry
        results = []
        for user_idx in range(len(user_lats)):
            visible_sats = []
            visible_indices = np.where(visibility_matrix[user_idx])[0]
            
            for sat_idx in visible_indices:
                op_id, sat = all_sat_metadata[sat_idx]
                sat_with_geom = sat.copy()
                sat_with_geom['geometry'] = {
                    'elevation': float(elevations[user_idx, sat_idx]),
                    'slant_range': float(slant_ranges[user_idx, sat_idx]),
                    'azimuth': 0.0,  # Can be computed if needed
                    'doppler': 0.0    # Can be computed if needed
                }
                sat_with_geom['elevation'] = float(elevations[user_idx, sat_idx])
                sat_with_geom['slant_range'] = float(slant_ranges[user_idx, sat_idx])
                visible_sats.append(sat_with_geom)
            
            # Sort by elevation (highest first)
            visible_sats.sort(key=lambda x: x.get('elevation', 0.0), reverse=True)
            results.append(visible_sats)
        
        return results
    
    def filter_visible_satellites(
        self,
        satellites: Dict[str, List[Dict]],
        user_lat: float,
        user_lon: float,
        timestamp: datetime
    ) -> List[Dict]:
        """
        Filter satellites visible to a user (elevation > min_elevation).
        
        Uses ECEF coordinates directly for accurate elevation calculation.
        
        Args:
            satellites: Dictionary of operator_id -> satellite list
            user_lat: User latitude
            user_lon: User longitude
            timestamp: Current time
            
        Returns:
            List of visible satellites with elevation > min_elevation
        """
        visible_sats = []
        
        # User position in ECEF (using WGS84)
        user_lat_rad = np.radians(user_lat)
        user_lon_rad = np.radians(user_lon)
        a = 6378137.0  # WGS84 semi-major axis
        e2 = 0.00669437999014  # WGS84 first eccentricity squared
        N = a / np.sqrt(1 - e2 * np.sin(user_lat_rad)**2)
        user_pos_ecef = np.array([
            (N + 0.0) * np.cos(user_lat_rad) * np.cos(user_lon_rad),
            (N + 0.0) * np.cos(user_lat_rad) * np.sin(user_lon_rad),
            (N * (1 - e2) + 0.0) * np.sin(user_lat_rad)
        ])
        
        # ENU rotation matrix (for converting ECEF to local ENU)
        sin_lat = np.sin(user_lat_rad)
        cos_lat = np.cos(user_lat_rad)
        sin_lon = np.sin(user_lon_rad)
        cos_lon = np.cos(user_lon_rad)
        
        R_ecef_to_enu = np.array([
            [-sin_lon, cos_lon, 0],
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat]
        ])
        
        for op_id, sat_list in satellites.items():
            for sat in sat_list:
                try:
                    sat_pos_ecef = sat['position_ecef']
                    
                    # Vector from user to satellite (in ECEF)
                    r_vec = sat_pos_ecef - user_pos_ecef
                    slant_range = np.linalg.norm(r_vec)
                    
                    if slant_range < 1e-6:  # Avoid division by zero
                        continue
                    
                    # Convert to ENU (East-North-Up) frame
                    r_enu = R_ecef_to_enu @ r_vec
                    
                    # Elevation angle (angle above horizon)
                    # elevation = arcsin(up_component / slant_range)
                    elevation_rad = np.arcsin(np.clip(r_enu[2] / slant_range, -1.0, 1.0))
                    elevation_deg = np.degrees(elevation_rad)
                    
                    # Filter by elevation (must be >= 25°)
                    if elevation_deg >= self.min_elevation_deg:
                        # Azimuth angle (compass direction, 0=North, 90=East)
                        azimuth_rad = np.arctan2(r_enu[0], r_enu[1])
                        azimuth_deg = np.degrees(azimuth_rad)
                        if azimuth_deg < 0:
                            azimuth_deg += 360
                        
                        # Create geometry dict (compatible with existing code)
                        geom = {
                            'elevation': elevation_deg,
                            'elevation_rad': elevation_rad,
                            'azimuth': azimuth_deg,
                            'azimuth_rad': azimuth_rad,
                            'slant_range': slant_range,
                            'doppler_shift': 0.0,  # Simplified - would need velocity
                            'radial_velocity': 0.0
                        }
                        
                        sat_info = {
                            'satellite_id': sat['satellite_id'],
                            'operator_id': op_id,
                            'position_ecef': sat['position_ecef'],
                            'velocity_ecef': sat['velocity_ecef'],
                            'elevation': elevation_deg,
                            'azimuth': azimuth_deg,
                            'slant_range': slant_range,
                            'geometry': geom
                        }
                        visible_sats.append(sat_info)
                except Exception as e:
                    logger.debug(f"Failed to compute geometry for {sat['satellite_id']}: {e}")
                    continue
        
        # Sort by elevation (highest first)
        visible_sats.sort(key=lambda x: x['elevation'], reverse=True)
        
        return visible_sats
    
    def select_best_server(
        self,
        visible_satellites: List[Dict],
        max_candidates: int = 12
    ) -> Tuple[Dict, List[Dict]]:
        """
        Select best serving satellite and interferers.
        
        Args:
            visible_satellites: List of visible satellites (sorted by elevation)
            max_candidates: Maximum number of candidates to consider
            
        Returns:
            Tuple of (serving_satellite, interferer_list)
        """
        if not visible_satellites:
            return None, []
        
        # Select top candidates (highest elevation = best signal)
        candidates = visible_satellites[:max_candidates]
        
        # Best server = highest elevation (strongest signal)
        serving_sat = candidates[0]
        
        # Interferers = rest of candidates
        interferers = candidates[1:]
        
        return serving_sat, interferers

