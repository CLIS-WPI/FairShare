"""
Scenario configuration loader for LEO DSS simulations.

Phase 4: Complete scenario parsing from YAML files.
"""

import yaml
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ScenarioConfig:
    """
    Scenario configuration loaded from YAML.
    
    Phase 4: Complete scenario structure with all required fields.
    """
    scenario_name: str
    sim_time_s: float
    slot_duration_s: float
    num_users: int
    num_operators: int
    carrier_frequency_hz: float
    bandwidth_hz: float
    traffic_model: Dict
    geo: Dict
    
    # Optional fields
    description: str = ""
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    time_step_seconds: Optional[float] = None
    tle_file: Optional[str] = None
    ground_stations: List[Tuple[float, float]] = field(default_factory=list)
    frequency_range_hz: Tuple[float, float] = (10e9, 12e9)
    policy: str = "fuzzy"
    users: List[Dict] = field(default_factory=list)
    channel: Dict = field(default_factory=dict)
    
    def __init__(self, yaml_path: str):
        """
        Load scenario from YAML file.
        
        Args:
            yaml_path: Path to scenario YAML file
        """
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Required fields (Phase 4 format)
        self.scenario_name = data.get('scenario_name', 'unknown')
        self.sim_time_s = data.get('sim_time_s', 600.0)
        self.slot_duration_s = data.get('slot_duration_s', 0.05)
        self.num_users = data.get('num_users', 100)
        self.num_operators = data.get('num_operators', 3)
        # Ensure numeric types (YAML may load scientific notation as string)
        carrier_freq = data.get('carrier_frequency_hz', 12e9)
        self.carrier_frequency_hz = float(carrier_freq) if carrier_freq is not None else 12e9
        
        bandwidth = data.get('bandwidth_hz', 200e6)
        self.bandwidth_hz = float(bandwidth) if bandwidth is not None else 200e6
        self.traffic_model = data.get('traffic_model', {'type': 'poisson', 'lambda_per_user': 0.2})
        self.geo = data.get('geo', {
            'center_lat_deg': 42.27,
            'center_lon_deg': -71.80,
            'radius_km': 50
        })
        
        # Optional fields (legacy format support)
        self.description = data.get('description', '')
        self.start_time = data.get('start_time', None)
        self.end_time = data.get('end_time', None)
        self.time_step_seconds = data.get('time_step_seconds', self.slot_duration_s)
        self.tle_file = data.get('tle_file', None)
        self.policy = data.get('policy', 'fuzzy')
        
        # Ground stations
        gs_list = data.get('ground_stations', [])
        if gs_list:
            self.ground_stations = [tuple(gs) if isinstance(gs, list) else gs for gs in gs_list]
        else:
            # Default to geo center
            self.ground_stations = [
                (self.geo['center_lat_deg'], self.geo['center_lon_deg'])
            ]
        
        # Frequency range - ensure numeric types
        freq_range = data.get('frequency_range_hz', [10e9, 12e9])
        if isinstance(freq_range, list):
            # Convert each element to float
            self.frequency_range_hz = tuple(float(f) for f in freq_range)
        elif isinstance(freq_range, tuple):
            self.frequency_range_hz = tuple(float(f) for f in freq_range)
        else:
            # Single value or other type
            try:
                self.frequency_range_hz = (float(freq_range[0]), float(freq_range[1]))
            except (TypeError, IndexError):
                self.frequency_range_hz = (10e9, 12e9)
        
        # Users (if explicitly defined)
        self.users = data.get('users', [])
        
        # Channel conditions
        self.channel = data.get('channel', {
            'frequency_hz': self.carrier_frequency_hz,
            'rain_rate_mmh': 0.0,
            'temperature_k': 290.0
        })
    
    def to_dict(self) -> Dict:
        """
        Convert scenario config to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            'scenario_name': self.scenario_name,
            'sim_time_s': self.sim_time_s,
            'slot_duration_s': self.slot_duration_s,
            'num_users': self.num_users,
            'num_operators': self.num_operators,
            'carrier_frequency_hz': self.carrier_frequency_hz,
            'bandwidth_hz': self.bandwidth_hz,
            'traffic_model': self.traffic_model,
            'geo': self.geo,
            'description': self.description,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'time_step_seconds': self.time_step_seconds,
            'tle_file': self.tle_file,
            'ground_stations': self.ground_stations,
            'frequency_range_hz': self.frequency_range_hz,
            'policy': self.policy,
            'users': self.users,
            'channel': self.channel
        }
    
    def get_num_slots(self) -> int:
        """Get number of time slots in simulation."""
        return int(self.sim_time_s / self.slot_duration_s)
    
    def get_start_datetime(self) -> datetime:
        """Get start time as datetime object."""
        if self.start_time:
            return datetime.fromisoformat(self.start_time)
        return datetime(2024, 1, 1, 0, 0, 0)
    
    def get_end_datetime(self) -> datetime:
        """Get end time as datetime object."""
        if self.end_time:
            return datetime.fromisoformat(self.end_time)
        start = self.get_start_datetime()
        return start + timedelta(seconds=self.sim_time_s)


def load_scenario(name: str) -> ScenarioConfig:
    """
    Load scenario by name from experiments/scenarios/ directory.
    
    Args:
        name: Scenario name (without .yaml extension)
        
    Returns:
        ScenarioConfig object
        
    Example:
        config = load_scenario("urban_congestion")
    """
    # Try different possible paths
    possible_paths = [
        f"experiments/scenarios/{name}.yaml",
        f"experiments/scenarios/{name}",
        f"src/experiments/scenarios/{name}.yaml",
        f"src/experiments/scenarios/{name}",
        os.path.join(os.path.dirname(__file__), '..', '..', 'experiments', 'scenarios', f"{name}.yaml")
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return ScenarioConfig(path)
    
    raise FileNotFoundError(f"Scenario '{name}' not found. Tried: {possible_paths}")


if __name__ == '__main__':
    # Test loading
    try:
        config = load_scenario("urban_congestion")
        print(f"✓ Loaded scenario: {config.scenario_name}")
        print(f"  Users: {config.num_users}")
        print(f"  Duration: {config.sim_time_s}s")
        print(f"  Slots: {config.get_num_slots()}")
    except FileNotFoundError as e:
        print(f"⚠ {e}")

