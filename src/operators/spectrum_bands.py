"""
Spectrum band management for multi-operator scenarios.

Handles frequency assignment, band allocation, and interference
between operators.
"""

from typing import List, Tuple, Optional, Dict, Set
import numpy as np
from dataclasses import dataclass
from enum import Enum


class BandType(Enum):
    """Types of spectrum bands."""
    KA_BAND = "ka"  # 26.5-40 GHz
    KU_BAND = "ku"  # 12-18 GHz
    V_BAND = "v"    # 40-75 GHz
    CUSTOM = "custom"


@dataclass
class SpectrumBand:
    """Represents a frequency band."""
    start_freq_mhz: float
    end_freq_mhz: float
    band_type: BandType
    operator_id: Optional[str] = None  # None = unassigned
    
    def get_bandwidth_mhz(self) -> float:
        """Get bandwidth in MHz."""
        return self.end_freq_mhz - self.start_freq_mhz
    
    def overlaps(self, other: 'SpectrumBand') -> bool:
        """Check if this band overlaps with another."""
        return not (
            self.end_freq_mhz <= other.start_freq_mhz or
            self.start_freq_mhz >= other.end_freq_mhz
        )
    
    def __repr__(self) -> str:
        return (
            f"SpectrumBand({self.start_freq_mhz:.1f}-{self.end_freq_mhz:.1f} MHz, "
            f"type={self.band_type.value}, operator={self.operator_id})"
        )


class SpectrumBandManager:
    """
    Manages spectrum bands for multiple operators.
    
    Handles frequency assignment, conflict detection, and
    interference management.
    """
    
    def __init__(
        self,
        total_spectrum_start_mhz: float = 10000.0,  # 10 GHz
        total_spectrum_end_mhz: float = 40000.0,    # 40 GHz
    ):
        """
        Initialize spectrum band manager.
        
        Args:
            total_spectrum_start_mhz: Start of available spectrum
            total_spectrum_end_mhz: End of available spectrum
        """
        self.total_spectrum_start_mhz = total_spectrum_start_mhz
        self.total_spectrum_end_mhz = total_spectrum_end_mhz
        self.total_bandwidth_mhz = total_spectrum_end_mhz - total_spectrum_start_mhz
        
        # All bands (assigned and unassigned)
        self.bands: List[SpectrumBand] = []
        
        # Operator assignments
        self.operator_bands: Dict[str, List[SpectrumBand]] = {}
    
    def add_band(
        self,
        start_freq_mhz: float,
        end_freq_mhz: float,
        band_type: BandType = BandType.CUSTOM,
        operator_id: Optional[str] = None
    ) -> SpectrumBand:
        """
        Add a spectrum band.
        
        Args:
            start_freq_mhz: Start frequency in MHz
            end_freq_mhz: End frequency in MHz
            band_type: Type of band
            operator_id: Optional operator assignment
            
        Returns:
            Created SpectrumBand
        """
        # Validate frequency range
        if start_freq_mhz < self.total_spectrum_start_mhz:
            raise ValueError(f"Start frequency {start_freq_mhz} below minimum")
        if end_freq_mhz > self.total_spectrum_end_mhz:
            raise ValueError(f"End frequency {end_freq_mhz} above maximum")
        if start_freq_mhz >= end_freq_mhz:
            raise ValueError("Start frequency must be < end frequency")
        
        # Check for overlaps with existing bands
        new_band = SpectrumBand(start_freq_mhz, end_freq_mhz, band_type, operator_id)
        for existing_band in self.bands:
            if new_band.overlaps(existing_band):
                if existing_band.operator_id != operator_id:
                    raise ValueError(
                        f"Band overlaps with existing band: {existing_band}"
                    )
        
        # Add band
        self.bands.append(new_band)
        
        # Track operator assignment
        if operator_id:
            if operator_id not in self.operator_bands:
                self.operator_bands[operator_id] = []
            self.operator_bands[operator_id].append(new_band)
        
        return new_band
    
    def assign_band_to_operator(
        self,
        start_freq_mhz: float,
        end_freq_mhz: float,
        operator_id: str,
        band_type: BandType = BandType.CUSTOM
    ) -> SpectrumBand:
        """Assign a spectrum band to an operator."""
        return self.add_band(start_freq_mhz, end_freq_mhz, band_type, operator_id)
    
    def get_operator_bands(self, operator_id: str) -> List[SpectrumBand]:
        """Get all bands assigned to an operator."""
        return self.operator_bands.get(operator_id, [])
    
    def get_operator_total_bandwidth(self, operator_id: str) -> float:
        """Get total bandwidth assigned to an operator."""
        bands = self.get_operator_bands(operator_id)
        return sum(band.get_bandwidth_mhz() for band in bands)
    
    def get_unassigned_bands(self) -> List[SpectrumBand]:
        """Get all unassigned bands."""
        return [band for band in self.bands if band.operator_id is None]
    
    def check_interference(
        self,
        operator1_id: str,
        operator2_id: str,
        guard_band_mhz: float = 0.0
    ) -> bool:
        """
        Check if two operators' bands interfere.
        
        Args:
            operator1_id: First operator
            operator2_id: Second operator
            guard_band_mhz: Required guard band in MHz
            
        Returns:
            True if interference detected
        """
        bands1 = self.get_operator_bands(operator1_id)
        bands2 = self.get_operator_bands(operator2_id)
        
        for band1 in bands1:
            for band2 in bands2:
                # Check overlap with guard band
                if guard_band_mhz > 0:
                    band1_guarded = SpectrumBand(
                        band1.start_freq_mhz - guard_band_mhz,
                        band1.end_freq_mhz + guard_band_mhz,
                        band1.band_type
                    )
                    if band1_guarded.overlaps(band2):
                        return True
                else:
                    if band1.overlaps(band2):
                        return True
        
        return False
    
    def get_available_spectrum(self) -> float:
        """Get total available (unassigned) spectrum in MHz."""
        assigned = sum(
            self.get_operator_total_bandwidth(op_id)
            for op_id in self.operator_bands.keys()
        )
        return self.total_bandwidth_mhz - assigned
    
    def __repr__(self) -> str:
        return (
            f"SpectrumBandManager(total={self.total_bandwidth_mhz:.1f} MHz, "
            f"operators={len(self.operator_bands)}, "
            f"available={self.get_available_spectrum():.1f} MHz)"
        )

