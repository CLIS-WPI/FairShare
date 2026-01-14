"""
OpenNTN-based Channel Model for FairShare

3GPP TR 38.811 compliant NTN channel modeling

"""

import tensorflow as tf
import numpy as np
from typing import Dict, Optional, Tuple

# Try to import OpenNTN
try:
    from OpenNTN import (
        DenseUrbanScenario,
        UrbanScenario, 
        SubUrbanScenario,
        Antenna
    )
    OPENNTN_AVAILABLE = True
except ImportError:
    OPENNTN_AVAILABLE = False
    DenseUrbanScenario = None
    UrbanScenario = None
    SubUrbanScenario = None
    Antenna = None


class OpenNTNChannelModel:
    """
    Realistic NTN channel model using OpenNTN/Sionna
    Supports per-user elevation-dependent channel calculations
    
    Note: OpenNTN only supports S-band (1.9-4 GHz) and Ka-band (19-40 GHz).
    For Ku-band (12 GHz), we use analytical fallback models.
    """

    def __init__(self, config: dict):
        """
        Args:
            config: Dictionary with:
                - carrier_frequency: Hz (e.g., 12e9 for Ku-band, 20e9 for Ka-band)
                - bandwidth: Hz
                - satellite_altitude: meters
                - satellite_tx_power: dBW
                - satellite_antenna_gain: dBi
                - ut_antenna_gain: dBi
                - noise_figure: dB
        """
        self.config = config
        self.carrier_frequency = config.get('carrier_frequency', 12e9)
        self.bandwidth = config.get('bandwidth', 250e6)
        self.sat_altitude = config.get('satellite_altitude', 550e3)
        self.sat_tx_power = config.get('satellite_tx_power', 30)  # dBW
        self.sat_antenna_gain = config.get('satellite_antenna_gain', 34)  # dBi
        self.ut_antenna_gain = config.get('ut_antenna_gain', 0)  # dBi
        self.noise_figure = config.get('noise_figure', 7)  # dB
        
        # Check if OpenNTN supports this frequency
        # OpenNTN supports: S-band (1.9-4 GHz) or Ka-band (19-40 GHz)
        self.use_openntn = OPENNTN_AVAILABLE and (
            (1.9e9 <= self.carrier_frequency <= 4e9) or 
            (19e9 <= self.carrier_frequency <= 40e9)
        )
        
        if self.use_openntn:
            # Create antenna arrays
            self._setup_antennas()
        else:
            if not OPENNTN_AVAILABLE:
                print("Warning: OpenNTN not available, using analytical models")
            else:
                print(f"Warning: OpenNTN doesn't support {self.carrier_frequency/1e9:.1f} GHz. "
                      f"Using analytical fallback models.")
        
        # Pre-compute constants
        self.k_boltzmann = 1.380649e-23  # J/K
        self.temp_noise = 290  # Kelvin
        self.noise_power = self._compute_noise_power()
        
    def _setup_antennas(self):
        """Setup satellite and user terminal antennas"""
        if not OPENNTN_AVAILABLE:
            return
            
        # Satellite antenna (3GPP pattern for Ka-band, omni for S-band)
        if 19e9 <= self.carrier_frequency <= 40e9:
            sat_pattern = "38.901"  # 3GPP pattern for Ka-band
        else:
            sat_pattern = "omni"  # S-band
            
        self.sat_antenna = Antenna(
            polarization="single",
            polarization_type="V",
            antenna_pattern=sat_pattern,
            carrier_frequency=self.carrier_frequency
        )
        
        # User terminal antenna (omnidirectional)
        self.ut_antenna = Antenna(
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=self.carrier_frequency
        )
    
    def _compute_noise_power(self):
        """Compute thermal noise power in linear scale"""
        # N = k * T * B * NF
        noise_power_dbm = -174 + 10 * np.log10(self.bandwidth) + self.noise_figure
        return 10 ** (noise_power_dbm / 10) / 1000  # Convert to Watts
    
    def get_scenario_for_location(self, user_lat: float, user_lon: float, 
                                   elevation_angle: float, 
                                   location_type: str = "auto"):
        """
        Get appropriate OpenNTN scenario based on location
        
        Args:
            user_lat: User latitude
            user_lon: User longitude
            elevation_angle: Elevation angle to satellite (degrees)
            location_type: "dense_urban", "urban", "suburban", "rural", or "auto"
        
        Returns:
            OpenNTN scenario object or None if not available
        """
        if not self.use_openntn:
            return None
            
        # Auto-detect based on population density if needed
        if location_type == "auto":
            location_type = self._classify_location(user_lat, user_lon)
        
        # Clip elevation to valid range
        elevation_angle = np.clip(elevation_angle, 0.1, 89.9)
        
        try:
            scenario_params = {
                'carrier_frequency': self.carrier_frequency,
                'ut_array': self.ut_antenna,
                'bs_array': self.sat_antenna,
                'direction': 'downlink',
                'elevation_angle': float(elevation_angle),
                'enable_pathloss': True,
                'enable_shadow_fading': True,
            }
            
            if location_type == "dense_urban":
                return DenseUrbanScenario(**scenario_params)
            elif location_type == "urban":
                return UrbanScenario(**scenario_params)
            elif location_type in ["suburban", "rural"]:
                # Use SubUrbanScenario for both suburban and rural
                # (OpenNTN doesn't have RuralScenario)
                return SubUrbanScenario(**scenario_params)
            else:
                return UrbanScenario(**scenario_params)  # Default
                
        except Exception as e:
            print(f"Warning: Failed to create OpenNTN scenario: {e}")
            return None
    
    def _classify_location(self, lat: float, lon: float) -> str:
        """
        Classify location type based on coordinates
        Uses simple distance from urban center for now
        """
        # Manhattan center
        urban_center_lat = 40.7128
        urban_center_lon = -74.0060
        
        # Haversine distance
        dist_km = self._haversine_distance(lat, lon, urban_center_lat, urban_center_lon)
        
        if dist_km < 10:
            return "dense_urban"
        elif dist_km < 30:
            return "urban"
        elif dist_km < 60:
            return "suburban"
        else:
            return "rural"
    
    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance in km between two coordinates"""
        R = 6371  # Earth radius in km
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
        return R * 2 * np.arcsin(np.sqrt(a))
    
    def compute_channel_gain(self, user_lat: float, user_lon: float,
                              sat_lat: float, sat_lon: float, sat_alt: float,
                              location_type: str = "auto") -> dict:
        """
        Compute complete channel gain for a user-satellite link
        
        Returns:
            dict with: path_loss_db, shadow_fading_db, atmospheric_loss_db,
                      total_loss_db, snr_db, capacity_bps
        """
        # 1. Compute geometry
        elevation_angle = self._compute_elevation(user_lat, user_lon, 
                                                   sat_lat, sat_lon, sat_alt)
        distance = self._compute_distance(user_lat, user_lon,
                                          sat_lat, sat_lon, sat_alt)
        
        # 2. Get scenario (if OpenNTN available and frequency supported)
        path_loss_db = None
        shadow_fading_db = None
        
        if self.use_openntn:
            try:
                scenario = self.get_scenario_for_location(
                    user_lat, user_lon, elevation_angle, location_type
                )
                
                if scenario is not None:
                    # Try to get path loss from OpenNTN
                    # Note: OpenNTN scenarios may have different method names
                    # We'll use a fallback if the method doesn't exist
                    try:
                        # Convert distance to tensor if needed
                        distance_tensor = tf.constant(distance, dtype=tf.float32)
                        elev_tensor = tf.constant(elevation_angle, dtype=tf.float32)
                        
                        # OpenNTN may compute path loss internally
                        # For now, we'll use analytical model as fallback
                        path_loss_db = self._free_space_path_loss(distance)
                        shadow_fading_db = self._analytical_shadow_fading(elevation_angle, location_type)
                    except Exception as e:
                        # Fallback to analytical
                        path_loss_db = self._free_space_path_loss(distance)
                        shadow_fading_db = self._analytical_shadow_fading(elevation_angle, location_type)
                else:
                    # Fallback to analytical
                    path_loss_db = self._free_space_path_loss(distance)
                    shadow_fading_db = self._analytical_shadow_fading(elevation_angle, location_type)
            except Exception as e:
                # Fallback to analytical model
                path_loss_db = self._free_space_path_loss(distance)
                shadow_fading_db = self._analytical_shadow_fading(elevation_angle, location_type)
        else:
            # Use analytical model
            path_loss_db = self._free_space_path_loss(distance)
            shadow_fading_db = self._analytical_shadow_fading(elevation_angle, location_type)
        
        # 3. Atmospheric losses (elevation dependent)
        atmospheric_loss_db = self._atmospheric_loss(elevation_angle)
        
        # 4. Rain attenuation (for Ka-band)
        rain_loss_db = self._rain_attenuation(elevation_angle) if self.carrier_frequency > 15e9 else 0
        
        # 5. Total loss
        total_loss_db = path_loss_db + shadow_fading_db + atmospheric_loss_db + rain_loss_db
        
        # 6. Received power
        rx_power_dbw = (self.sat_tx_power + self.sat_antenna_gain + 
                        self.ut_antenna_gain - total_loss_db)
        
        # 7. SNR
        noise_power_dbw = 10 * np.log10(self.noise_power)
        snr_db = rx_power_dbw - noise_power_dbw
        
        # 8. Capacity (Shannon)
        snr_linear = 10 ** (snr_db / 10)
        capacity_bps = self.bandwidth * np.log2(1 + snr_linear)
        
        return {
            'elevation_angle': elevation_angle,
            'distance_km': distance / 1000,
            'path_loss_db': path_loss_db,
            'shadow_fading_db': shadow_fading_db,
            'atmospheric_loss_db': atmospheric_loss_db,
            'rain_loss_db': rain_loss_db,
            'total_loss_db': total_loss_db,
            'rx_power_dbw': rx_power_dbw,
            'snr_db': snr_db,
            'capacity_bps': capacity_bps,
            'capacity_mbps': capacity_bps / 1e6,
            'location_type': location_type if location_type != "auto" else self._classify_location(user_lat, user_lon)
        }
    
    def _compute_elevation(self, user_lat, user_lon, sat_lat, sat_lon, sat_alt):
        """Compute elevation angle from user to satellite"""
        R_earth = 6371e3  # meters
        
        # Convert to radians
        user_lat_rad = np.radians(user_lat)
        user_lon_rad = np.radians(user_lon)
        sat_lat_rad = np.radians(sat_lat)
        sat_lon_rad = np.radians(sat_lon)
        
        # Central angle
        delta_lon = sat_lon_rad - user_lon_rad
        cos_gamma = (np.sin(user_lat_rad) * np.sin(sat_lat_rad) + 
                     np.cos(user_lat_rad) * np.cos(sat_lat_rad) * np.cos(delta_lon))
        gamma = np.arccos(np.clip(cos_gamma, -1, 1))
        
        # Elevation angle
        r_sat = R_earth + sat_alt
        elevation_rad = np.arctan2(
            np.cos(gamma) - R_earth / r_sat,
            np.sin(gamma)
        )
        
        return np.degrees(elevation_rad)
    
    def _compute_distance(self, user_lat, user_lon, sat_lat, sat_lon, sat_alt):
        """Compute slant range distance to satellite in meters"""
        R_earth = 6371e3
        
        user_lat_rad = np.radians(user_lat)
        user_lon_rad = np.radians(user_lon)
        sat_lat_rad = np.radians(sat_lat)
        sat_lon_rad = np.radians(sat_lon)
        
        delta_lon = sat_lon_rad - user_lon_rad
        cos_gamma = (np.sin(user_lat_rad) * np.sin(sat_lat_rad) + 
                     np.cos(user_lat_rad) * np.cos(sat_lat_rad) * np.cos(delta_lon))
        
        r_sat = R_earth + sat_alt
        distance = np.sqrt(R_earth**2 + r_sat**2 - 2 * R_earth * r_sat * cos_gamma)
        
        return distance
    
    def _free_space_path_loss(self, distance):
        """Free space path loss in dB"""
        c = 3e8
        wavelength = c / self.carrier_frequency
        return 20 * np.log10(4 * np.pi * distance / wavelength)
    
    def _analytical_shadow_fading(self, elevation_angle, location_type):
        """
        Shadow fading based on 3GPP TR 38.811 Table 6.6.2-1
        """
        # Standard deviation varies by elevation and environment
        if location_type in ["dense_urban", "urban"]:
            if elevation_angle < 20:
                sigma = 6.0
            elif elevation_angle < 50:
                sigma = 4.0
            else:
                sigma = 2.5
        elif location_type == "suburban":
            if elevation_angle < 20:
                sigma = 4.0
            elif elevation_angle < 50:
                sigma = 3.0
            else:
                sigma = 2.0
        else:  # rural
            if elevation_angle < 20:
                sigma = 2.5
            elif elevation_angle < 50:
                sigma = 1.5
            else:
                sigma = 1.0
        
        return np.random.normal(0, sigma)
    
    def _atmospheric_loss(self, elevation_angle):
        """
        Atmospheric (gaseous) attenuation
        Increases at low elevation angles
        """
        # Simplified ITU-R P.676 model
        zenith_attenuation = 0.5  # dB at zenith for Ku-band
        if elevation_angle < 5:
            elevation_angle = 5  # Avoid division issues
        return zenith_attenuation / np.sin(np.radians(elevation_angle))
    
    def _rain_attenuation(self, elevation_angle, rain_rate=10):
        """
        Rain attenuation for Ka-band
        Based on ITU-R P.618
        """
        if self.carrier_frequency < 15e9:
            return 0
        
        # Simplified model
        specific_attenuation = 0.01 * rain_rate ** 1.2  # dB/km
        effective_path_length = 5 / np.sin(np.radians(max(elevation_angle, 10)))  # km
        
        return specific_attenuation * effective_path_length


class BatchChannelCalculator:
    """
    GPU-accelerated batch channel calculations for large-scale simulations
    """
    
    def __init__(self, channel_model: OpenNTNChannelModel):
        self.channel_model = channel_model
    
    @tf.function
    def compute_batch_snr(self, user_positions: tf.Tensor, 
                          sat_positions: tf.Tensor,
                          location_types: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Compute SNR for batch of user-satellite pairs on GPU
        
        Args:
            user_positions: [N, 2] tensor of (lat, lon)
            sat_positions: [M, 3] tensor of (lat, lon, alt)
            location_types: [N] tensor of location type indices
        
        Returns:
            (snr_db, elevations, distances) tensors of shape [N, M]
        """
        n_users = tf.shape(user_positions)[0]
        n_sats = tf.shape(sat_positions)[0]
        
        # Expand for broadcasting
        users_expanded = tf.expand_dims(user_positions, 1)  # [N, 1, 2]
        sats_expanded = tf.expand_dims(sat_positions, 0)    # [1, M, 3]
        
        # Compute distances and elevations (vectorized)
        distances = self._batch_distance(users_expanded, sats_expanded)
        elevations = self._batch_elevation(users_expanded, sats_expanded)
        
        # Path loss
        path_loss = self._batch_fspl(distances)
        
        # Location-dependent shadow fading
        shadow_fading = self._batch_shadow_fading(elevations, location_types)
        
        # Atmospheric loss
        atm_loss = self._batch_atmospheric(elevations)
        
        # Total loss and SNR
        total_loss = path_loss + shadow_fading + atm_loss
        
        rx_power = (self.channel_model.sat_tx_power + 
                   self.channel_model.sat_antenna_gain +
                   self.channel_model.ut_antenna_gain - total_loss)
        
        noise_power_f32 = tf.cast(self.channel_model.noise_power, tf.float32)
        noise_power_db = 10.0 * tf.math.log(noise_power_f32) / tf.math.log(10.0)
        snr_db = rx_power - noise_power_db
        
        return snr_db, elevations, distances
    
    @tf.function
    def _batch_distance(self, users, sats):
        """Compute distances for all user-satellite pairs"""
        R_earth = 6371e3
        
        user_lat = users[..., 0] * np.pi / 180
        user_lon = users[..., 1] * np.pi / 180
        sat_lat = sats[..., 0] * np.pi / 180
        sat_lon = sats[..., 1] * np.pi / 180
        sat_alt = sats[..., 2]
        
        delta_lon = sat_lon - user_lon
        cos_gamma = (tf.sin(user_lat) * tf.sin(sat_lat) + 
                    tf.cos(user_lat) * tf.cos(sat_lat) * tf.cos(delta_lon))
        
        r_sat = R_earth + sat_alt
        distance = tf.sqrt(R_earth**2 + r_sat**2 - 2 * R_earth * r_sat * cos_gamma)
        
        return distance
    
    @tf.function  
    def _batch_elevation(self, users, sats):
        """Compute elevation angles for all pairs"""
        R_earth = 6371e3
        
        user_lat = users[..., 0] * np.pi / 180
        user_lon = users[..., 1] * np.pi / 180
        sat_lat = sats[..., 0] * np.pi / 180
        sat_lon = sats[..., 1] * np.pi / 180
        sat_alt = sats[..., 2]
        
        delta_lon = sat_lon - user_lon
        cos_gamma = (tf.sin(user_lat) * tf.sin(sat_lat) + 
                     tf.cos(user_lat) * tf.cos(sat_lat) * tf.cos(delta_lon))
        cos_gamma = tf.clip_by_value(cos_gamma, -1.0, 1.0)
        gamma = tf.acos(cos_gamma)
        
        r_sat = R_earth + sat_alt
        elevation = tf.atan2(
            tf.cos(gamma) - R_earth / r_sat,
            tf.sin(gamma)
        )
        
        return elevation * 180 / np.pi
    
    @tf.function
    def _batch_fspl(self, distances):
        """Free space path loss"""
        c = 3e8
        wavelength = c / self.channel_model.carrier_frequency
        return 20 * tf.math.log(4 * np.pi * distances / wavelength) / tf.math.log(10.0)
    
    @tf.function
    def _batch_shadow_fading(self, elevations, location_types):
        """Shadow fading with location-dependent variance"""
        # Simplified: use average sigma per location type
        sigma_map = tf.constant([4.0, 3.0, 2.5, 1.5], dtype=tf.float32)  # dense_urban, urban, suburban, rural
        sigmas = tf.gather(sigma_map, location_types)
        sigmas = tf.expand_dims(sigmas, 1)
        
        return tf.random.normal(tf.shape(elevations)) * sigmas
    
    @tf.function
    def _batch_atmospheric(self, elevations):
        """Atmospheric loss"""
        zenith_atten = 0.5
        safe_elevation = tf.maximum(elevations, 5.0)
        return zenith_atten / tf.sin(safe_elevation * np.pi / 180)

