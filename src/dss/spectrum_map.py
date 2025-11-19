"""
DySPAN-critical spectrum sensing database.

Maintains a database of spectrum sensing results for dynamic spectrum access,
enabling interference-aware spectrum allocation decisions.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field


@dataclass
class SpectrumMeasurement:
    """Single spectrum measurement."""
    timestamp: datetime
    frequency_hz: float
    bandwidth_hz: float
    power_dbm: float
    location: Tuple[float, float]  # (lat, lon)
    source_id: Optional[str] = None  # ID of source (beam, user, etc.)


@dataclass
class SpectrumMapEntry:
    """Entry in spectrum map database."""
    frequency_hz: float
    bandwidth_hz: float
    measurements: List[SpectrumMeasurement] = field(default_factory=list)
    average_power_dbm: float = -np.inf
    last_updated: Optional[datetime] = None


class SpectrumMap:
    """
    DySPAN spectrum sensing database.
    
    Maintains a time-series database of spectrum measurements for
    dynamic spectrum access decisions.
    """
    
    def __init__(self, frequency_range_hz: Tuple[float, float],
                 frequency_resolution_hz: float = 1e6,
                 measurement_ttl_seconds: float = 3600.0):
        """
        Initialize spectrum map.
        
        Args:
            frequency_range_hz: (min_freq, max_freq) in Hz
            frequency_resolution_hz: Frequency resolution
            measurement_ttl_seconds: Time-to-live for measurements
        """
        self.freq_min, self.freq_max = frequency_range_hz
        self.freq_resolution = frequency_resolution_hz
        self.ttl_seconds = measurement_ttl_seconds
        
        # Database: frequency_bin -> SpectrumMapEntry
        self.database = {}
        
        # Create frequency bins
        self.freq_bins = np.arange(self.freq_min, self.freq_max, self.freq_resolution)
    
    def add_measurement(self, measurement: SpectrumMeasurement) -> None:
        """
        Add a spectrum measurement to the database.
        
        Args:
            measurement: Spectrum measurement
        """
        # Find frequency bin
        freq_bin = int((measurement.frequency_hz - self.freq_min) / self.freq_resolution)
        
        if freq_bin not in self.database:
            self.database[freq_bin] = SpectrumMapEntry(
                frequency_hz=measurement.frequency_hz,
                bandwidth_hz=measurement.bandwidth_hz
            )
        
        entry = self.database[freq_bin]
        entry.measurements.append(measurement)
        entry.last_updated = measurement.timestamp
        
        # Update average power
        self._update_average_power(entry)
    
    def _update_average_power(self, entry: SpectrumMapEntry) -> None:
        """Update average power for an entry."""
        if len(entry.measurements) == 0:
            entry.average_power_dbm = -np.inf
            return
        
        # Convert to linear, average, convert back
        powers_linear = [10**(m.power_dbm / 10) for m in entry.measurements]
        avg_power_linear = np.mean(powers_linear)
        entry.average_power_dbm = 10 * np.log10(avg_power_linear) if avg_power_linear > 0 else -np.inf
    
    def cleanup_old_measurements(self, current_time: Optional[datetime] = None) -> int:
        """
        Remove measurements older than TTL.
        
        Args:
            current_time: Current time (default: now)
            
        Returns:
            Number of measurements removed
        """
        if current_time is None:
            current_time = datetime.now()
        
        removed_count = 0
        
        for entry in self.database.values():
            cutoff_time = current_time - timedelta(seconds=self.ttl_seconds)
            
            # Filter measurements
            original_count = len(entry.measurements)
            entry.measurements = [
                m for m in entry.measurements
                if m.timestamp > cutoff_time
            ]
            
            removed_count += original_count - len(entry.measurements)
            
            # Update average if measurements removed
            if len(entry.measurements) < original_count:
                self._update_average_power(entry)
                if len(entry.measurements) == 0:
                    entry.last_updated = None
        
        return removed_count
    
    def get_spectrum_occupancy(self, frequency_hz: float,
                              bandwidth_hz: float,
                              time_window_seconds: Optional[float] = None) -> float:
        """
        Get spectrum occupancy in a frequency band.
        
        Args:
            frequency_hz: Center frequency in Hz
            bandwidth_hz: Bandwidth in Hz
            time_window_seconds: Time window for measurements (None = all)
            
        Returns:
            Average power in dBm
        """
        freq_start = frequency_hz - bandwidth_hz / 2
        freq_end = frequency_hz + bandwidth_hz / 2
        
        start_bin = int((freq_start - self.freq_min) / self.freq_resolution)
        end_bin = int((freq_end - self.freq_min) / self.freq_resolution)
        
        start_bin = max(0, min(start_bin, len(self.freq_bins) - 1))
        end_bin = max(0, min(end_bin, len(self.freq_bins) - 1))
        
        # Collect measurements in frequency range
        powers_linear = []
        
        for bin_idx in range(start_bin, end_bin + 1):
            if bin_idx in self.database:
                entry = self.database[bin_idx]
                
                # Filter by time window if specified
                if time_window_seconds is not None:
                    cutoff_time = datetime.now() - timedelta(seconds=time_window_seconds)
                    measurements = [m for m in entry.measurements 
                                  if m.timestamp > cutoff_time]
                else:
                    measurements = entry.measurements
                
                for m in measurements:
                    if m.power_dbm > -np.inf:
                        powers_linear.append(10**(m.power_dbm / 10))
        
        # Average power
        if len(powers_linear) > 0:
            avg_power_linear = np.mean(powers_linear)
            return 10 * np.log10(avg_power_linear)
        else:
            return -np.inf
    
    def find_available_channels(self, bandwidth_hz: float,
                               max_power_dbm: float = -100.0,
                               min_channel_separation_hz: float = 0.0) -> List[Tuple[float, float]]:
        """
        Find available channels based on sensing database.
        
        Args:
            bandwidth_hz: Required bandwidth in Hz
            max_power_dbm: Maximum acceptable power level
            min_channel_separation_hz: Minimum separation between channels
            
        Returns:
            List of (center_freq, power_dbm) tuples for available channels
        """
        available_channels = []
        
        # Scan frequency range
        current_freq = self.freq_min + bandwidth_hz / 2
        
        while current_freq + bandwidth_hz / 2 <= self.freq_max:
            # Check occupancy
            power = self.get_spectrum_occupancy(current_freq, bandwidth_hz)
            
            if power <= max_power_dbm:
                available_channels.append((current_freq, power))
                # Skip ahead to avoid overlapping channels
                current_freq += bandwidth_hz + min_channel_separation_hz
            else:
                # Move to next potential channel
                current_freq += self.freq_resolution
        
        return available_channels
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about the spectrum map.
        
        Returns:
            Dictionary with statistics
        """
        total_measurements = sum(len(entry.measurements) 
                                for entry in self.database.values())
        
        active_bins = len([entry for entry in self.database.values()
                          if len(entry.measurements) > 0])
        
        if total_measurements > 0:
            avg_powers = [entry.average_power_dbm 
                         for entry in self.database.values()
                         if entry.average_power_dbm > -np.inf]
            avg_power = np.mean(avg_powers) if avg_powers else -np.inf
        else:
            avg_power = -np.inf
        
        return {
            'total_measurements': total_measurements,
            'active_frequency_bins': active_bins,
            'total_frequency_bins': len(self.freq_bins),
            'average_power_dbm': avg_power,
            'frequency_range_hz': (self.freq_min, self.freq_max),
            'frequency_resolution_hz': self.freq_resolution
        }
    
    def allocate_channel(self, user_id: str, bandwidth_hz: float,
                        max_power_dbm: float = -100.0) -> Optional[Tuple[float, float]]:
        """
        Allocate a channel from available channels (SAS-like allocation).
        
        Args:
            user_id: User identifier
            bandwidth_hz: Required bandwidth in Hz
            max_power_dbm: Maximum acceptable interference power
            
        Returns:
            (center_frequency_hz, power_dbm) if allocation successful, None otherwise
        """
        available = self.find_available_channels(
            bandwidth_hz,
            max_power_dbm=max_power_dbm
        )
        
        if not available:
            return None
        
        # Select best channel (lowest interference)
        selected = min(available, key=lambda x: x[1])
        
        # Record allocation as measurement
        measurement = SpectrumMeasurement(
            timestamp=datetime.now(),
            frequency_hz=selected[0],
            bandwidth_hz=bandwidth_hz,
            power_dbm=40.0,  # Assumed transmit power
            location=(0.0, 0.0),
            source_id=user_id
        )
        self.add_measurement(measurement)
        
        return selected
    
    def query_spectrum(self, frequency_hz: float, bandwidth_hz: float,
                     time_window_seconds: Optional[float] = None) -> Dict:
        """
        Query spectrum availability (SAS-like query with enhanced information).
        
        Args:
            frequency_hz: Center frequency in Hz
            bandwidth_hz: Bandwidth in Hz
            time_window_seconds: Time window for measurements (None = all)
            
        Returns:
            Dictionary with comprehensive spectrum information:
            - available: Free/Busy status
            - occupancy_dbm: Power level
            - occupancy_percent: Utilization percentage
            - expected_interference_dbm: Expected interference
            - sinr_estimate_db: Estimated SINR
            - best_alternative_channel: Best alternative if occupied
        """
        occupancy = self.get_spectrum_occupancy(frequency_hz, bandwidth_hz, time_window_seconds)
        
        # Calculate occupancy percentage
        freq_start = frequency_hz - bandwidth_hz / 2
        freq_end = frequency_hz + bandwidth_hz / 2
        start_bin = int((freq_start - self.freq_min) / self.freq_resolution)
        end_bin = int((freq_end - self.freq_min) / self.freq_resolution)
        start_bin = max(0, min(start_bin, len(self.freq_bins) - 1))
        end_bin = max(0, min(end_bin, len(self.freq_bins) - 1))
        
        # Count occupied bins in range
        occupied_bins = 0
        total_power_linear = 0.0
        for bin_idx in range(start_bin, end_bin + 1):
            if bin_idx in self.database:
                entry = self.database[bin_idx]
                if time_window_seconds is not None:
                    cutoff_time = datetime.now() - timedelta(seconds=time_window_seconds)
                    measurements = [m for m in entry.measurements 
                                  if m.timestamp > cutoff_time]
                else:
                    measurements = entry.measurements
                
                if len(measurements) > 0:
                    occupied_bins += 1
                    if entry.average_power_dbm > -np.inf:
                        total_power_linear += 10**(entry.average_power_dbm / 10)
        
        total_bins = end_bin - start_bin + 1
        occupancy_percent = (occupied_bins / total_bins * 100) if total_bins > 0 else 0.0
        
        # Expected interference
        expected_interference_dbm = occupancy if occupancy > -np.inf else -np.inf
        
        # Estimate SINR (simplified)
        desired_power_dbm = 40.0  # Assumed transmit power
        noise_dbm = -174.0 + 10 * np.log10(bandwidth_hz)  # Thermal noise
        
        if occupancy > -np.inf:
            interference_linear = 10**(occupancy / 10)
        else:
            interference_linear = 0
        
        noise_linear = 10**(noise_dbm / 10)
        desired_linear = 10**(desired_power_dbm / 10)
        
        sinr_linear = desired_linear / (interference_linear + noise_linear) if (interference_linear + noise_linear) > 0 else np.inf
        sinr_estimate_db = 10 * np.log10(sinr_linear) if sinr_linear < np.inf else 100.0
        
        # Find best alternative channel
        best_alternative = None
        if occupancy >= -100.0:  # If occupied, find alternative
            alternatives = self.find_available_channels(
                bandwidth_hz,
                max_power_dbm=-100.0
            )
            if alternatives:
                # Select channel with lowest interference
                best_alternative = min(alternatives, key=lambda x: x[1])
        
        return {
            'frequency_hz': frequency_hz,
            'bandwidth_hz': bandwidth_hz,
            'available': occupancy < -100.0,  # Free/Busy status
            'occupancy_dbm': occupancy,
            'occupancy_percent': occupancy_percent,
            'expected_interference_dbm': expected_interference_dbm,
            'sinr_estimate_db': sinr_estimate_db,
            'best_alternative_channel': best_alternative,  # (freq, power) or None
            'timestamp': datetime.now()
        }

