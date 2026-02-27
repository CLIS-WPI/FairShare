"""
Constellation configurations for multi-constellation simulation.

Defines Starlink Shell 1, OneWeb Phase 1, and Kuiper Shell 1
with proper Walker Delta parameters for the workshop revision.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ConstellationConfig:
    name: str
    num_planes: int
    sats_per_plane: int
    altitude_km: float
    inclination_deg: float
    inter_plane_phasing: bool = True
    reference: str = ""

    @property
    def total_sats(self) -> int:
        return self.num_planes * self.sats_per_plane


CONSTELLATIONS: Dict[str, ConstellationConfig] = {
    "starlink_shell1": ConstellationConfig(
        name="Starlink Shell 1",
        num_planes=72,
        sats_per_plane=22,
        altitude_km=550.0,
        inclination_deg=53.0,
        reference="FCC-22-91",
    ),
    "oneweb": ConstellationConfig(
        name="OneWeb Phase 1",
        num_planes=18,
        sats_per_plane=36,
        altitude_km=1200.0,
        inclination_deg=87.9,
        reference="ITU Filing",
    ),
    "kuiper_shell1": ConstellationConfig(
        name="Kuiper Shell 1",
        num_planes=34,
        sats_per_plane=34,
        altitude_km=630.0,
        inclination_deg=51.9,
        reference="FCC-20-102",
    ),
}


@dataclass
class CoverageArea:
    name: str = "NYC Metropolitan Area"
    center_lat: float = 40.7128
    center_lon: float = -74.0060
    urban_radius_km: float = 22.0
    suburban_inner_km: float = 22.0
    suburban_outer_km: float = 55.0
    rural_inner_km: float = 55.0
    rural_outer_km: float = 165.0


COVERAGE_AREA = CoverageArea()


SIMULATION_DEFAULTS = {
    "n_snapshots": 20,
    "snapshot_interval_sec": 30,
    "n_users": 1000,
    "urban_fraction": 0.50,
    "suburban_fraction": 0.20,
    "rural_fraction": 0.30,
    "n_monte_carlo": 50,
    "carrier_freq_ghz": 20.0,
    "bandwidth_mhz": 300,
    "sat_eirp_dbw": 45,
    "user_gain_dbi": 30,
    "noise_figure_db": 2,
    "shadow_fading_urban_db": 8,
    "shadow_fading_suburban_db": 6,
    "shadow_fading_rural_db": 4,
    "min_elevation_deg": 10.0,
    "beams_per_satellite": 7,
    "beam_3db_width_deg": 1.5,
    "beam_peak_gain_dbi": 30,
    "freq_reuse_factor": 4,
    "fairshare_quotas": {"urban": 0.40, "suburban": 0.25, "rural": 0.35},
}
