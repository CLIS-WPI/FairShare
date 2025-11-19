"""
ITU-R atmospheric attenuation models for satellite links.

Implements ITU-R recommendations for rain, cloud, and tropospheric
attenuation calculations.
"""

import numpy as np
from typing import Dict, Optional


class AtmosphericModel:
    """
    ITU-R atmospheric attenuation models.
    
    Implements:
    - ITU-R P.838: Rain attenuation
    - ITU-R P.840: Cloud attenuation
    - ITU-R P.676: Tropospheric gases
    """
    
    def __init__(self, frequency_hz: float = 12e9):
        """
        Initialize atmospheric model.
        
        Args:
            frequency_hz: Carrier frequency in Hz
        """
        self.frequency_hz = frequency_hz
        self.frequency_ghz = frequency_hz / 1e9
        
        # ITU-R P.838 coefficients (for Ku band, 12 GHz)
        self._init_rain_coefficients()
    
    def _init_rain_coefficients(self):
        """Initialize ITU-R P.838 rain attenuation coefficients."""
        # Coefficients for horizontal polarization at 12 GHz
        if 10 <= self.frequency_ghz <= 20:
            # Ku band
            self.k_h = 0.0188
            self.alpha_h = 1.217
            self.k_v = 0.0168
            self.alpha_v = 1.200
        elif 20 < self.frequency_ghz <= 30:
            # Ka band
            self.k_h = 0.0367
            self.alpha_h = 1.154
            self.k_v = 0.0335
            self.alpha_v = 1.128
        else:
            # Default to Ku band
            self.k_h = 0.0188
            self.alpha_h = 1.217
            self.k_v = 0.0168
            self.alpha_v = 1.200
    
    def rain_attenuation(self, rain_rate_mmh: float, elevation_deg: float,
                        polarization: str = 'horizontal',
                        latitude_deg: float = 0.0) -> float:
        """
        Compute rain attenuation using ITU-R P.838.
        
        Args:
            rain_rate_mmh: Rain rate in mm/h (exceeded 0.01% of time)
            elevation_deg: Elevation angle in degrees
            polarization: 'horizontal' or 'vertical'
            latitude_deg: Ground station latitude in degrees
            
        Returns:
            Rain attenuation in dB
        """
        if rain_rate_mmh <= 0:
            return 0.0
        
        # Select coefficients based on polarization
        if polarization == 'horizontal':
            k = self.k_h
            alpha = self.alpha_h
        else:
            k = self.k_v
            alpha = self.alpha_v
        
        # Specific attenuation (dB/km)
        gamma = k * (rain_rate_mmh ** alpha)
        
        # Effective path length
        elev_rad = np.radians(elevation_deg)
        
        # Rain height (ITU-R P.839)
        if abs(latitude_deg) <= 36:
            h_r = 5.0 - 0.075 * (abs(latitude_deg) - 23)
        else:
            h_r = 0.0
        
        # Slant path length through rain
        if elevation_deg >= 5:
            L_s = (h_r - 0) / np.sin(elev_rad)  # km
        else:
            return np.inf  # Below horizon
        
        # Horizontal reduction factor
        L_G = L_s * np.cos(elev_rad)  # km
        
        # Reduction factor (ITU-R P.838)
        if L_G <= 0:
            r = 1.0
        else:
            r = 1 / (1 + 0.78 * np.sqrt(L_G * gamma / self.frequency_ghz) - 0.38 * (1 - np.exp(-2 * L_G)))
        
        # Vertical adjustment factor
        if elevation_deg >= 5:
            v = 1 / (1 + np.sqrt(np.sin(elev_rad)) * 
                    (31 * (1 - np.exp(-elevation_deg / 31)) * np.sqrt(L_G * gamma / self.frequency_ghz) - 0.45))
        else:
            v = 1.0
        
        # Effective path length
        L_e = L_s * r * v
        
        # Total rain attenuation
        A_rain = gamma * L_e
        
        return A_rain
    
    def cloud_attenuation(self, liquid_water_density: float,
                         elevation_deg: float, temperature_k: float = 273.15) -> float:
        """
        Compute cloud attenuation using ITU-R P.840.
        
        Args:
            liquid_water_density: Cloud liquid water density in g/m³
            elevation_deg: Elevation angle in degrees
            temperature_k: Cloud temperature in Kelvin
            
        Returns:
            Cloud attenuation in dB
        """
        if liquid_water_density <= 0:
            return 0.0
        
        # Complex permittivity of water
        f_ghz = self.frequency_ghz
        eps_prime = 77.66 + 103.3 * (300 / temperature_k - 1)
        eps_double_prime = 0.067 * eps_prime
        
        # Relaxation frequency
        f_p = 20.09 - 142 * (300 / temperature_k - 1) + 294 * (300 / temperature_k - 1)**2
        
        # Complex permittivity
        eps = eps_prime - 1j * eps_double_prime * (f_ghz / f_p) / (1 + (f_ghz / f_p)**2)
        
        # Attenuation coefficient (simplified)
        K = 0.819 * f_ghz / (eps_double_prime * (1 + (f_ghz / f_p)**2))
        
        # Path length through clouds (simplified: 2 km)
        L_cloud = 2.0 / np.sin(np.radians(elevation_deg))  # km
        
        # Total cloud attenuation
        A_cloud = K * liquid_water_density * L_cloud
        
        return A_cloud
    
    def tropospheric_gases(self, elevation_deg: float,
                          pressure_hpa: float = 1013.25,
                          temperature_k: float = 288.15,
                          water_vapor_density: float = 7.5) -> float:
        """
        Compute tropospheric gas attenuation using ITU-R P.676.
        
        Args:
            elevation_deg: Elevation angle in degrees
            pressure_hpa: Atmospheric pressure in hPa
            temperature_k: Temperature in Kelvin
            water_vapor_density: Water vapor density in g/m³
            
        Returns:
            Total gas attenuation in dB
        """
        if elevation_deg < 0:
            return np.inf
        
        # Dry air attenuation (oxygen)
        gamma_o = self._oxygen_attenuation(pressure_hpa, temperature_k)
        
        # Water vapor attenuation
        gamma_w = self._water_vapor_attenuation(pressure_hpa, temperature_k, water_vapor_density)
        
        # Total specific attenuation
        gamma_total = gamma_o + gamma_w  # dB/km
        
        # Effective path length (simplified: 8 km scale height)
        h_scale = 8.0  # km
        L_path = h_scale / np.sin(np.radians(elevation_deg))  # km
        
        # Total attenuation
        A_gases = gamma_total * L_path
        
        return A_gases
    
    def _oxygen_attenuation(self, pressure_hpa: float, temperature_k: float) -> float:
        """Compute oxygen attenuation coefficient."""
        f_ghz = self.frequency_ghz
        
        # Simplified model for Ku/Ka band
        # More accurate models available in ITU-R P.676
        if 10 <= f_ghz <= 20:
            # Ku band: oxygen absorption is minimal
            gamma_o = 0.01 * (pressure_hpa / 1013.25) * (288.15 / temperature_k)**0.5
        else:
            gamma_o = 0.02 * (pressure_hpa / 1013.25) * (288.15 / temperature_k)**0.5
        
        return gamma_o
    
    def _water_vapor_attenuation(self, pressure_hpa: float, temperature_k: float,
                                 water_vapor_density: float) -> float:
        """Compute water vapor attenuation coefficient."""
        f_ghz = self.frequency_ghz
        
        # Simplified model
        # Water vapor has resonance around 22.235 GHz
        if 20 <= f_ghz <= 25:
            gamma_w = 0.05 * water_vapor_density / 7.5
        else:
            gamma_w = 0.01 * water_vapor_density / 7.5
        
        return gamma_w
    
    def total_atmospheric_loss(self, elevation_deg: float,
                              rain_rate_mmh: float = 0.0,
                              cloud_lwc: float = 0.0,
                              pressure_hpa: float = 1013.25,
                              temperature_k: float = 288.15,
                              water_vapor_density: float = 7.5) -> Dict:
        """
        Compute total atmospheric losses.
        
        Args:
            elevation_deg: Elevation angle in degrees
            rain_rate_mmh: Rain rate in mm/h
            cloud_lwc: Cloud liquid water content in g/m³
            pressure_hpa: Atmospheric pressure in hPa
            temperature_k: Temperature in Kelvin
            water_vapor_density: Water vapor density in g/m³
            
        Returns:
            Dictionary with all attenuation components
        """
        A_rain = self.rain_attenuation(rain_rate_mmh, elevation_deg)
        A_cloud = self.cloud_attenuation(cloud_lwc, elevation_deg, temperature_k)
        A_gases = self.tropospheric_gases(elevation_deg, pressure_hpa, 
                                         temperature_k, water_vapor_density)
        
        A_total = A_rain + A_cloud + A_gases
        
        return {
            'rain_attenuation_db': A_rain,
            'cloud_attenuation_db': A_cloud,
            'gas_attenuation_db': A_gases,
            'total_attenuation_db': A_total
        }

