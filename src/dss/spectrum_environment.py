"""
Spectrum occupancy map for satellite beams.

Tracks spectrum usage across different beams and frequencies,
enabling interference-aware spectrum allocation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Beam:
    """Represents a satellite beam."""
    beam_id: str
    satellite_id: str
    center_frequency_hz: float
    bandwidth_hz: float
    power_dbm: float
    location: Tuple[float, float]  # (lat, lon)
    elevation_deg: float


class SpectrumEnvironment:
    """
    Spectrum occupancy environment for dynamic spectrum sharing.
    
    Tracks spectrum usage across beams and computes interference.
    """
    
    def __init__(self, frequency_range_hz: Tuple[float, float],
                 frequency_resolution_hz: float = 1e6):
        """
        Initialize spectrum environment.
        
        Args:
            frequency_range_hz: (min_freq, max_freq) in Hz
            frequency_resolution_hz: Frequency resolution for spectrum map
        """
        # Ensure numeric types (may come from YAML as strings)
        self.freq_min = float(frequency_range_hz[0])
        self.freq_max = float(frequency_range_hz[1])
        self.freq_resolution = float(frequency_resolution_hz)
        
        # Create frequency bins
        self.freq_bins = np.arange(self.freq_min, self.freq_max, self.freq_resolution)
        self.n_bins = len(self.freq_bins)
        
        # Spectrum occupancy map: [beam_id][frequency_bin] -> power (dBm)
        self.occupancy_map = {}
        
        # Active beams
        self.beams = {}
    
    def register_beam(self, beam: Beam) -> None:
        """
        Register a beam in the environment.
        
        Args:
            beam: Beam object
        """
        self.beams[beam.beam_id] = beam
        if beam.beam_id not in self.occupancy_map:
            self.occupancy_map[beam.beam_id] = np.zeros(self.n_bins) - np.inf
    
    def update_beam_usage(self, beam_id: str, 
                         frequency_hz: float,
                         bandwidth_hz: float,
                         power_dbm: float) -> None:
        """
        Update spectrum usage for a beam.
        
        Args:
            beam_id: Beam identifier
            frequency_hz: Center frequency in Hz
            bandwidth_hz: Bandwidth in Hz
            power_dbm: Transmit power in dBm
        """
        if beam_id not in self.occupancy_map:
            raise ValueError(f"Beam {beam_id} not registered")
        
        # Find frequency bins within bandwidth
        freq_start = frequency_hz - bandwidth_hz / 2
        freq_end = frequency_hz + bandwidth_hz / 2
        
        start_bin = int((freq_start - self.freq_min) / self.freq_resolution)
        end_bin = int((freq_end - self.freq_min) / self.freq_resolution)
        
        start_bin = max(0, min(start_bin, self.n_bins - 1))
        end_bin = max(0, min(end_bin, self.n_bins - 1))
        
        # Update occupancy
        self.occupancy_map[beam_id][start_bin:end_bin + 1] = power_dbm
    
    def get_spectrum_occupancy(self, frequency_hz: float,
                              bandwidth_hz: float,
                              exclude_beam_id: Optional[str] = None) -> float:
        """
        Get total spectrum occupancy (interference) at given frequency.
        
        Args:
            frequency_hz: Center frequency in Hz
            bandwidth_hz: Bandwidth in Hz
            exclude_beam_id: Beam to exclude from interference calculation
            
        Returns:
            Total interference power in dBm
        """
        freq_start = frequency_hz - bandwidth_hz / 2
        freq_end = frequency_hz + bandwidth_hz / 2
        
        start_bin = int((freq_start - self.freq_min) / self.freq_resolution)
        end_bin = int((freq_end - self.freq_min) / self.freq_resolution)
        
        start_bin = max(0, min(start_bin, self.n_bins - 1))
        end_bin = max(0, min(end_bin, self.n_bins - 1))
        
        # Sum interference from all beams (except excluded)
        total_interference_linear = 0.0
        
        for beam_id, occupancy in self.occupancy_map.items():
            if beam_id == exclude_beam_id:
                continue
            
            # Get power in frequency range
            power_dbm = np.max(occupancy[start_bin:end_bin + 1])
            
            if power_dbm > -np.inf:
                # Convert to linear and sum
                power_linear = 10**(power_dbm / 10)  # mW
                total_interference_linear += power_linear
        
        # Convert back to dBm
        if total_interference_linear > 0:
            # Add small epsilon to avoid log10(0) warning
            total_interference_linear_safe = max(total_interference_linear, 1e-20)
            total_interference_dbm = 10 * np.log10(total_interference_linear_safe)
        else:
            total_interference_dbm = -np.inf
        
        return total_interference_dbm
    
    def find_available_spectrum(self, bandwidth_hz: float,
                               min_sinr_db: float = 0.0,  # Lowered from 10.0 to allow more allocations
                               exclude_beam_id: Optional[str] = None,
                               link_budget_snr_db: Optional[float] = None,
                               allow_spatial_reuse: bool = True) -> List[Tuple[float, float]]:
        """
        Find available spectrum bands.
        
        Args:
            bandwidth_hz: Required bandwidth in Hz
            min_sinr_db: Minimum required SINR in dB (default: 0.0, lowered to allow allocations)
            exclude_beam_id: Beam to exclude from search
            link_budget_snr_db: Optional actual SNR from link budget (if None, uses assumed value)
            
        Returns:
            List of (center_freq, sinr) tuples for available bands
        """
        available_bands = []
        
        # Use actual link budget SNR if provided, otherwise use reasonable assumption
        if link_budget_snr_db is not None:
            # Use actual SNR from link budget
            base_snr_db = link_budget_snr_db
        else:
            # Assume reasonable SNR for LEO satellite (typically 10-20 dB)
            base_snr_db = 15.0
        
        # Scan frequency range
        for center_freq in np.arange(
            self.freq_min + bandwidth_hz / 2,
            self.freq_max - bandwidth_hz / 2,
            bandwidth_hz
        ):
            # Check if band is available
            # exclude_beam_id allows spatial reuse (different beams can use same frequency)
            interference = self.get_spectrum_occupancy(
                center_freq, bandwidth_hz, exclude_beam_id if allow_spatial_reuse else None
            )
            
            # Estimate SINR using link budget SNR
            noise_dbm = -174.0 + 10 * np.log10(bandwidth_hz)  # Thermal noise
            
            if interference > -np.inf:
                interference_linear = 10**(interference / 10)
            else:
                interference_linear = 0
            
            noise_linear = 10**(noise_dbm / 10)
            # Use link budget SNR to estimate desired signal power
            desired_linear = noise_linear * 10**(base_snr_db / 10)
            
            sinr_linear = desired_linear / (interference_linear + noise_linear)
            # Add small epsilon to avoid log10(0) warning
            sinr_linear_safe = max(sinr_linear, 1e-20)
            sinr_db = 10 * np.log10(sinr_linear_safe)
            
            if sinr_db >= min_sinr_db:
                available_bands.append((center_freq, sinr_db))
        
        return available_bands
    
    def get_spectrum_map(self) -> np.ndarray:
        """
        Get complete spectrum occupancy map.
        
        Returns:
            2D array [beam_id_index, frequency_bin] of power levels
        """
        beam_ids = list(self.occupancy_map.keys())
        n_beams = len(beam_ids)
        
        spectrum_map = np.zeros((n_beams, self.n_bins)) - np.inf
        
        for i, beam_id in enumerate(beam_ids):
            spectrum_map[i, :] = self.occupancy_map[beam_id]
        
        return spectrum_map, beam_ids
    
    def clear_beam_usage(self, beam_id: str) -> None:
        """Clear spectrum usage for a beam."""
        if beam_id in self.occupancy_map:
            self.occupancy_map[beam_id] = np.zeros(self.n_bins) - np.inf
    
    def get_available_channels(self, bandwidth_hz: float,
                              min_sinr_db: float = 0.0,  # Lowered default to allow allocations
                              link_budget_snr_db: Optional[float] = None) -> List[Tuple[float, float]]:
        """
        Get available channels for allocation.
        
        Args:
            bandwidth_hz: Required bandwidth in Hz
            min_sinr_db: Minimum required SINR in dB (default: 0.0)
            link_budget_snr_db: Optional actual SNR from link budget
            
        Returns:
            List of (center_frequency_hz, sinr_db) tuples
        """
        return self.find_available_spectrum(bandwidth_hz, min_sinr_db, link_budget_snr_db=link_budget_snr_db)
    
    def check_conflict(self, frequency_hz: float, bandwidth_hz: float,
                      exclude_beam_id: Optional[str] = None) -> bool:
        """
        Check if frequency band conflicts with existing allocations.
        
        Args:
            frequency_hz: Center frequency in Hz
            bandwidth_hz: Bandwidth in Hz
            exclude_beam_id: Beam to exclude from conflict check
            
        Returns:
            True if conflict exists, False if free
        """
        interference = self.get_spectrum_occupancy(
            frequency_hz, bandwidth_hz, exclude_beam_id
        )
        # Conflict if interference above threshold
        return interference > -100.0  # dBm threshold
    
    def allocate(self, user_id: str, bandwidth_hz: float,
                beam_id: Optional[str] = None,
                preferred_frequency_hz: Optional[float] = None,
                allow_same_beam_reallocation: bool = False) -> Optional[Tuple[float, float]]:
        """
        Allocate spectrum to a user with conflict detection.
        
        Multi-operator logic:
        - Two different beams can allocate same frequency (OK - spatial reuse)
        - Same beam cannot double-allocate same frequency (NOT OK - conflict)
        
        Args:
            user_id: User identifier
            bandwidth_hz: Required bandwidth in Hz
            beam_id: Optional beam ID (if None, finds best beam)
            preferred_frequency_hz: Optional preferred frequency
            allow_same_beam_reallocation: Allow same beam to reallocate (default: False)
            
        Returns:
            (center_frequency_hz, sinr_db) if allocation successful, None otherwise
        """
        # FIXED: Find available channels EXCLUDING current beam (allows spatial reuse)
        # This allows different beams to use same frequency
        available = self.find_available_spectrum(
            bandwidth_hz,
            min_sinr_db=0.0,
            exclude_beam_id=beam_id,  # FIXED: Exclude current beam for spatial reuse
            allow_spatial_reuse=True
        )
        
        if not available:
            return None
        
        # FIXED: Filter available channels to exclude frequencies this beam already uses
        # This allows beams to have multiple allocations at different frequencies
        if beam_id is not None and beam_id in self.occupancy_map:
            # Filter out frequencies that this beam already uses
            filtered_available = []
            for freq, sinr in available:
                freq_start = freq - bandwidth_hz / 2
                freq_end = freq + bandwidth_hz / 2
                start_bin = int((freq_start - self.freq_min) / self.freq_resolution)
                end_bin = int((freq_end - self.freq_min) / self.freq_resolution)
                start_bin = max(0, min(start_bin, self.n_bins - 1))
                end_bin = max(0, min(end_bin, self.n_bins - 1))
                
                # Check if beam already occupies this frequency
                existing_occupancy = self.occupancy_map[beam_id][start_bin:end_bin + 1]
                if not np.any(existing_occupancy > -np.inf) or allow_same_beam_reallocation:
                    # This frequency is free for this beam (or reallocation allowed)
                    filtered_available.append((freq, sinr))
            
            if filtered_available:
                available = filtered_available
            else:
                # No free frequencies for this beam
                return None
        
        # Select best channel (highest SINR) from filtered available channels
        if not available:
            return None
            
        if preferred_frequency_hz is not None:
            # Try to use preferred frequency
            for freq, sinr in available:
                if abs(freq - preferred_frequency_hz) < bandwidth_hz:
                    selected = (freq, sinr)
                    break
            else:
                # Preferred not available, use best
                selected = max(available, key=lambda x: x[1])
        else:
            selected = max(available, key=lambda x: x[1])
        
        center_freq, sinr = selected
        
        # Allocate to beam (or create new beam if needed)
        if beam_id is None:
            beam_id = f"beam_{user_id}"
        
        if beam_id not in self.beams:
            # Create default beam
            beam = Beam(
                beam_id=beam_id,
                satellite_id="unknown",
                center_frequency_hz=center_freq,
                bandwidth_hz=bandwidth_hz,
                power_dbm=40.0,
                location=(0.0, 0.0),
                elevation_deg=0.0
            )
            self.register_beam(beam)
        
        # Update occupancy
        self.update_beam_usage(beam_id, center_freq, bandwidth_hz, 40.0)
        
        return selected
    
    def update_interference_map(self) -> np.ndarray:
        """
        Update and return interference map.
        
        Returns:
            1D array [frequency_bin] of total interference power (dBm)
        """
        interference_linear = np.zeros(self.n_bins)
        
        # Sum interference from all beams (in linear domain)
        for beam_id, occupancy in self.occupancy_map.items():
            # Convert to linear and sum
            power_linear = np.where(
                occupancy > -np.inf,
                10**(occupancy / 10),
                0.0
            )
            interference_linear += power_linear
        
        # Convert back to dBm (avoid log10(0) warning)
        # Add small epsilon to avoid log10(0) for all values
        interference_linear_safe = np.maximum(interference_linear, 1e-20)
        interference_map = np.where(
            interference_linear > 1e-20,
            10 * np.log10(interference_linear_safe),
            -np.inf
        )
        
        return interference_map
    
    def compute_beam_load(self, beam_id: str) -> float:
        """
        Compute beam load (utilization).
        
        Args:
            beam_id: Beam identifier
            
        Returns:
            Beam load (0.0 to 1.0)
        """
        if beam_id not in self.occupancy_map:
            return 0.0
        
        occupancy = self.occupancy_map[beam_id]
        
        # Count occupied bins
        occupied_bins = np.sum(occupancy > -np.inf)
        
        # Load = occupied / total
        load = occupied_bins / self.n_bins
        
        return float(load)
    
    def get_beam_loads(self) -> Dict[str, float]:
        """
        Get load for all beams.
        
        Returns:
            Dictionary mapping beam_id to load (0.0 to 1.0)
        """
        loads = {}
        for beam_id in self.beams.keys():
            loads[beam_id] = self.compute_beam_load(beam_id)
        return loads

