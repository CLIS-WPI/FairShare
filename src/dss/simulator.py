"""
Core DSS simulator using physics, fairness, and spectrum components.

Main simulation loop that integrates orbit propagation, channel modeling,
fairness evaluation, and dynamic spectrum sharing.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

try:
    from ..channel import OrbitPropagator, SatelliteGeometry, ChannelModel
    from ..fairness import TraditionalFairness, VectorFairness
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    from src.channel import OrbitPropagator, SatelliteGeometry, ChannelModel
    from src.fairness import TraditionalFairness, VectorFairness

from .spectrum_environment import SpectrumEnvironment, Beam
from .spectrum_map import SpectrumMap
from .policies import StaticPolicy, PriorityPolicy


class DSSSimulator:
    """
    Dynamic Spectrum Sharing simulator for LEO satellite networks.
    
    Integrates:
    - Orbit propagation
    - Channel modeling
    - Fairness evaluation
    - Dynamic spectrum allocation
    """
    
    def __init__(self, 
                 tle_file: Optional[str] = None,
                 ground_stations: List[Tuple[float, float]] = None,
                 frequency_range_hz: Tuple[float, float] = (10e9, 12e9),
                 policy_type: str = 'priority'):
        """
        Initialize DSS simulator.
        
        Args:
            tle_file: Path to TLE file
            ground_stations: List of (lat, lon) tuples for ground stations
            frequency_range_hz: Frequency range for spectrum sharing
            policy_type: 'static', 'priority', or 'load_adaptive'
        """
        # Orbit propagation
        self.orbit_prop = OrbitPropagator(tle_file) if tle_file else None
        
        # Ground stations
        self.ground_stations = ground_stations or [(0.0, 0.0)]
        self.geometries = [
            SatelliteGeometry(lat, lon) for lat, lon in self.ground_stations
        ]
        
        # Channel model
        self.channel_model = ChannelModel()
        
        # Fairness evaluation
        self.traditional_fairness = TraditionalFairness()
        self.vector_fairness = VectorFairness()
        
        # Spectrum environment
        self.spectrum_env = SpectrumEnvironment(frequency_range_hz)
        self.spectrum_map = SpectrumMap(frequency_range_hz)
        
        # Allocation policy
        if policy_type == 'static':
            self.policy = StaticPolicy()
        elif policy_type == 'priority':
            self.policy = PriorityPolicy()
        elif policy_type == 'load_adaptive':
            # LoadAdaptivePolicy removed - use priority instead
            self.policy = PriorityPolicy()
        else:  # Default to priority
            self.policy = PriorityPolicy()
        
        # Simulation state
        self.current_time = None
        self.satellites = {}
        self.users = {}
        self.allocation_history = []
        self.fairness_history = []
    
    def add_satellite(self, satellite_name: str, line1: str, line2: str) -> None:
        """Add satellite from TLE lines."""
        if self.orbit_prop:
            self.orbit_prop.add_satellite(satellite_name, line1, line2)
        self.satellites[satellite_name] = {
            'name': satellite_name,
            'beams': []
        }
    
    def add_beam(self, satellite_name: str, beam_id: str,
                center_freq_hz: float, bandwidth_hz: float,
                power_dbm: float, location: Tuple[float, float]) -> None:
        """Add a beam to a satellite."""
        if satellite_name not in self.satellites:
            raise ValueError(f"Satellite {satellite_name} not found")
        
        beam = Beam(beam_id, satellite_name, center_freq_hz, 
                   bandwidth_hz, power_dbm, location, 0.0)
        
        self.satellites[satellite_name]['beams'].append(beam)
        self.spectrum_env.register_beam(beam)
    
    def add_user(self, user_id: str, demand: float, priority: float = 0.5,
                location: Tuple[float, float] = None) -> None:
        """Add a user to the system."""
        self.users[user_id] = {
            'id': user_id,
            'demand': demand,
            'priority': priority,
            'location': location or (0.0, 0.0),
            'allocation': 0.0,
            'satellite': None
        }
    
    def step(self, dt: datetime, demands: Optional[Dict[str, float]] = None) -> Dict:
        """
        Perform one simulation step.
        
        Args:
            dt: Current simulation time
            demands: Optional updated user demands
            
        Returns:
            Dictionary with step results
        """
        self.current_time = dt
        
        # Update user demands if provided
        if demands:
            for user_id, demand in demands.items():
                if user_id in self.users:
                    self.users[user_id]['demand'] = demand
        
        # Propagate orbits and compute geometry
        geometries = {}
        for sat_name in self.satellites.keys():
            if self.orbit_prop:
                try:
                    pos, vel = self.orbit_prop.propagate(sat_name, dt)
                    # Compute geometry for first ground station
                    geom = self.geometries[0].compute_geometry(pos, vel, dt)
                    geometries[sat_name] = geom
                except:
                    geometries[sat_name] = None
        
        # Compute channel conditions
        channel_conditions = {}
        for sat_name, geom in geometries.items():
            if geom:
                link_budget = self.channel_model.compute_link_budget(geom)
                channel_conditions[sat_name] = link_budget
        
        # Get current allocations
        user_ids = list(self.users.keys())
        current_allocations = np.array([self.users[uid]['allocation'] for uid in user_ids])
        current_demands = np.array([self.users[uid]['demand'] for uid in user_ids])
        priorities = np.array([self.users[uid]['priority'] for uid in user_ids])
        
        # Compute network load
        total_demand = np.sum(current_demands)
        total_allocated = np.sum(current_allocations)
        available_resources = 100.0  # Total available (normalized)
        network_load = min(total_demand / available_resources, 1.0) if available_resources > 0 else 0.0
        
        # Allocate resources using policy
        if self.policy:
            if isinstance(self.policy, PriorityPolicy):
                new_allocations = self.policy.allocate(
                    current_demands, available_resources, priorities
                )
            elif policy_type == 'load_adaptive':
                new_allocations = self.policy.allocate(
                    current_demands, available_resources, network_load, priorities
                )
            else:
                new_allocations = self.policy.allocate(
                    current_demands, available_resources
                )
        else:
            # Use priority policy as default
            new_allocations = PriorityPolicy().allocate(
                current_demands, available_resources, priorities
            )
        
        # Update user allocations
        for i, user_id in enumerate(user_ids):
            self.users[user_id]['allocation'] = new_allocations[i]
        
        # Evaluate fairness
        jain_index = self.traditional_fairness.jain_index(new_allocations)
        gini_coefficient = self.traditional_fairness.gini_coefficient(new_allocations)
        fairness_results = {
            'jain_index': jain_index,
            'gini_coefficient': gini_coefficient
        }
        
        # Store history
        self.allocation_history.append({
            'time': dt,
            'allocations': new_allocations.copy(),
            'demands': current_demands.copy()
        })
        self.fairness_history.append({
            'time': dt,
            **fairness_results
        })
        
        return {
            'time': dt,
            'allocations': new_allocations,
            'demands': current_demands,
            'fairness': fairness_results,
            'network_load': network_load,
            'channel_conditions': channel_conditions
        }
    
    # Removed: _fuzzy_allocation method (replaced with priority policy as default)
        
        # Normalize
        total_allocated = np.sum(allocation)
        if total_allocated > 0:
            allocation = allocation * (available_resources / total_allocated)
        
        return allocation
    
    def run(self, start_time: datetime, end_time: datetime,
           time_step_seconds: float = 60.0) -> Dict:
        """
        Run simulation for a time period.
        
        Args:
            start_time: Simulation start time
            end_time: Simulation end time
            time_step_seconds: Time step in seconds
            
        Returns:
            Dictionary with simulation results
        """
        current = start_time
        results = []
        
        while current < end_time:
            step_result = self.step(current)
            results.append(step_result)
            current += timedelta(seconds=time_step_seconds)
        
        return {
            'results': results,
            'allocation_history': self.allocation_history,
            'fairness_history': self.fairness_history
        }

