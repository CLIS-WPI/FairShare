"""
Operator class for multi-operator LEO satellite constellations.

Each operator (e.g., Starlink, Kuiper) has its own constellation,
spectrum bands, and resource allocation preferences.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
from enum import Enum


class OperatorType(Enum):
    """Types of LEO satellite operators."""
    STARLINK = "starlink"
    KUIPER = "kuiper"
    ONEWEB = "oneweb"
    TELESAT = "telesat"
    CUSTOM = "custom"


@dataclass
class OperatorConfig:
    """Configuration for an operator."""
    name: str
    operator_type: OperatorType
    constellation_altitude_km: float  # Altitude in km
    constellation_inclination_deg: float  # Inclination in degrees
    num_satellites: int  # Number of satellites
    num_planes: int  # Number of orbital planes
    satellites_per_plane: int  # Satellites per plane
    
    # Spectrum configuration
    spectrum_bands_mhz: List[Tuple[float, float]] = field(default_factory=list)
    # Each tuple is (start_freq_mhz, end_freq_mhz)
    
    # Beam configuration
    beams_per_satellite: int = 64
    beam_coverage_radius_km: float = 500.0
    
    # Resource preferences
    priority_weight: float = 1.0  # Relative priority (1.0 = equal)
    min_bandwidth_mhz: float = 20.0  # Minimum required bandwidth
    max_bandwidth_mhz: float = 1000.0  # Maximum available bandwidth
    
    # Traffic characteristics
    user_density_per_km2: float = 0.1  # Users per km²
    avg_demand_mbps: float = 10.0  # Average demand per user in Mbps
    
    def __post_init__(self):
        """Validate configuration."""
        if self.num_satellites != self.num_planes * self.satellites_per_plane:
            raise ValueError(
                f"num_satellites ({self.num_satellites}) must equal "
                f"num_planes ({self.num_planes}) × satellites_per_plane "
                f"({self.satellites_per_plane})"
            )


class Operator:
    """
    Represents a single LEO satellite operator.
    
    Each operator manages its own constellation, spectrum bands,
    and resource allocation preferences.
    """
    
    def __init__(self, config: OperatorConfig):
        """
        Initialize an operator.
        
        Args:
            config: Operator configuration
        """
        self.config = config
        self.name = config.name
        self.operator_type = config.operator_type
        
        # Constellation will be initialized separately
        self.constellation = None
        
        # Resource tracking
        self.allocated_bandwidth_mhz: float = 0.0
        self.total_demand_mbps: float = 0.0
        self.served_users: int = 0
        
        # Performance metrics
        self.throughput_mbps: float = 0.0
        self.latency_ms: float = 0.0
        self.coverage_percentage: float = 0.0
        
        # Fairness scores (will be computed)
        self.fairness_scores: Dict[str, float] = {}
    
    def set_constellation(self, constellation):
        """Set the constellation for this operator."""
        self.constellation = constellation
    
    def get_spectrum_bands(self) -> List[Tuple[float, float]]:
        """Get spectrum bands assigned to this operator."""
        return self.config.spectrum_bands_mhz
    
    def get_total_spectrum_mhz(self) -> float:
        """Get total spectrum bandwidth in MHz."""
        total = 0.0
        for start, end in self.config.spectrum_bands_mhz:
            total += (end - start)
        return total
    
    def update_resource_usage(
        self,
        allocated_bandwidth_mhz: float,
        served_users: int,
        throughput_mbps: float,
        latency_ms: float
    ):
        """
        Update resource usage and performance metrics.
        
        Args:
            allocated_bandwidth_mhz: Allocated bandwidth in MHz
            served_users: Number of users served
            throughput_mbps: Average throughput in Mbps
            latency_ms: Average latency in ms
        """
        self.allocated_bandwidth_mhz = allocated_bandwidth_mhz
        self.served_users = served_users
        self.throughput_mbps = throughput_mbps
        self.latency_ms = latency_ms
    
    def update_fairness_scores(self, scores: Dict[str, float]):
        """Update fairness scores for this operator."""
        self.fairness_scores.update(scores)
    
    def get_utilization(self) -> float:
        """Get spectrum utilization (0.0 to 1.0)."""
        total_spectrum = self.get_total_spectrum_mhz()
        if total_spectrum == 0:
            return 0.0
        return min(1.0, self.allocated_bandwidth_mhz / total_spectrum)
    
    def get_efficiency(self) -> float:
        """Get spectral efficiency (Mbps per MHz)."""
        if self.allocated_bandwidth_mhz == 0:
            return 0.0
        return self.throughput_mbps / self.allocated_bandwidth_mhz
    
    def __repr__(self) -> str:
        return (
            f"Operator(name='{self.name}', "
            f"type={self.operator_type.value}, "
            f"satellites={self.config.num_satellites}, "
            f"spectrum={self.get_total_spectrum_mhz():.1f} MHz)"
        )

