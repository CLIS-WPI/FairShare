#!/usr/bin/env python3
"""
Parameterized Simulation Runner

Allows running simulations with custom parameters for sensitivity analysis.
"""

import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import json
from typing import List, Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.operators import (
    Operator, OperatorConfig, OperatorType,
    Constellation, SpectrumBandManager, BandType
)
from src.allocation import (
    AllocationEngine, AllocationPolicy,
    AllocationRequest, ResourceTracker
)
from src.fairness.traditional import TraditionalFairness
from src.fairness.vector_metrics import VectorFairness, MultiDimensionalMetrics
from src.data import SyntheticDataGenerator, DataValidator


def run_simulation_with_params(
    num_users: int = 100,
    operators_list: List[str] = None,
    demand_distribution: str = "clustered",
    policy: AllocationPolicy = AllocationPolicy.STATIC_EQUAL,
    available_bandwidth_mhz: float = 10000.0
) -> Dict[str, Any]:
    """
    Run simulation with specified parameters.
    
    Returns:
        Dictionary with fairness metrics and configuration
    """
    if operators_list is None:
        operators_list = ["starlink", "kuiper", "oneweb"]
    
    # Generate users
    generator = SyntheticDataGenerator(
        region_bounds={"lat": (35.0, 45.0), "lon": (-85.0, -70.0)},
        population_density_per_km2=0.15
    )
    
    users = generator.generate_users(num_users, operators_list, distribution=demand_distribution)
    
    # Generate traffic
    start_time = datetime.now()
    patterns = generator.generate_traffic_timeline(users, start_time, duration_hours=1.0, interval_minutes=10)
    current_pattern = patterns[0]
    
    # Configure operators
    spectrum_manager = SpectrumBandManager(10000.0, 40000.0)
    operators = {}
    
    operator_configs = {
        "starlink": {"altitude_km": 550.0, "inclination_deg": 53.0, "num_satellites": 100,
                     "num_planes": 10, "satellites_per_plane": 10,
                     "spectrum_bands": [(10000.0, 12000.0), (15000.0, 17000.0)]},
        "kuiper": {"altitude_km": 630.0, "inclination_deg": 51.9, "num_satellites": 80,
                   "num_planes": 8, "satellites_per_plane": 10,
                   "spectrum_bands": [(20000.0, 22000.0), (25000.0, 27000.0)]},
        "oneweb": {"altitude_km": 1200.0, "inclination_deg": 87.4, "num_satellites": 50,
                   "num_planes": 5, "satellites_per_plane": 10,
                   "spectrum_bands": [(30000.0, 32000.0)]}
    }
    
    for op_id in operators_list:
        if op_id not in operator_configs:
            continue
        config = operator_configs[op_id]
        op_config = OperatorConfig(
            name=op_id.capitalize(), operator_type=OperatorType.CUSTOM,
            constellation_altitude_km=config["altitude_km"],
            constellation_inclination_deg=config["inclination_deg"],
            num_satellites=config["num_satellites"],
            num_planes=config["num_planes"],
            satellites_per_plane=config["satellites_per_plane"],
            spectrum_bands_mhz=config["spectrum_bands"]
        )
        operator = Operator(op_config)
        for start, end in config["spectrum_bands"]:
            spectrum_manager.assign_band_to_operator(start, end, op_id, BandType.KA_BAND)
        constellation = Constellation(
            operator_name=op_id, altitude_km=config["altitude_km"],
            inclination_deg=config["inclination_deg"],
            num_satellites=config["num_satellites"],
            num_planes=config["num_planes"],
            satellites_per_plane=config["satellites_per_plane"]
        )
        operator.set_constellation(constellation)
        operators[op_id] = operator
    
    # Create requests
    requests = []
    for user in users:
        demand = current_pattern.user_demands.get(user.user_id, user.demand_mbps)
        requests.append(AllocationRequest(
            operator_id=user.operator_id, user_id=user.user_id,
            demand_mbps=demand, priority=user.priority,
            min_bandwidth_mhz=demand / 10.0, max_bandwidth_mhz=demand / 5.0
        ))
    
    # Allocate
    engine = AllocationEngine(spectrum_manager, policy)
    results = engine.allocate(requests, available_bandwidth_mhz=available_bandwidth_mhz)
    
    # Track and simulate performance
    tracker = ResourceTracker()
    tracker.update_allocations(results, current_pattern.timestamp)
    
    for result in results:
        if result.success:
            throughput = result.allocated_bandwidth_mhz * 10.0
            latency = 20.0 + np.random.normal(0, 5)
            latency = max(10.0, latency)
            user_key = f"{result.operator_id}_{result.user_id}"
            if user_key in tracker.user_metrics:
                tracker.user_metrics[user_key].throughput_mbps = throughput
                tracker.user_metrics[user_key].latency_ms = latency
                tracker.user_metrics[user_key].qos_satisfaction = 0.9 if throughput >= result.allocated_bandwidth_mhz * 8.0 else 0.7
    
    # Compute fairness
    all_allocations = [r.allocated_bandwidth_mhz for r in results if r.success]
    
    metrics = {
        "config": {
            "num_users": num_users,
            "num_operators": len(operators_list),
            "demand_distribution": demand_distribution,
            "policy": policy.value,
            "available_bandwidth_mhz": available_bandwidth_mhz
        },
        "allocation": {
            "successful": sum(1 for r in results if r.success),
            "total": len(results),
            "total_allocated_mhz": sum(r.allocated_bandwidth_mhz for r in results if r.success)
        }
    }
    
    if all_allocations:
        metrics["fairness"] = {
            "jain_index": TraditionalFairness.jain_index(all_allocations),
            "gini_coefficient": TraditionalFairness.gini_coefficient(all_allocations),
            "alpha_fairness": TraditionalFairness.alpha_fairness(all_allocations, alpha=1.0)
        }
        
        # Vector-based fairness
        metrics_list = []
        for result in results:
            if result.success:
                user_key = f"{result.operator_id}_{result.user_id}"
                user_metrics = tracker.user_metrics.get(user_key)
                if user_metrics and user_metrics.throughput_mbps > 0:
                    metrics_list.append(MultiDimensionalMetrics(
                        throughput_mbps=user_metrics.throughput_mbps,
                        latency_ms=user_metrics.latency_ms,
                        access_rate=1.0, coverage_quality=0.9,
                        qos_satisfaction=user_metrics.qos_satisfaction
                    ))
        
        if metrics_list:
            vector_fairness = VectorFairness()
            metrics["fairness"]["weighted_fairness"] = vector_fairness.compute_weighted_fairness(metrics_list)
            metrics["fairness"]["distance_fairness"] = vector_fairness.compute_distance_fairness(metrics_list)
    
    return metrics


if __name__ == "__main__":
    # Example usage
    result = run_simulation_with_params(
        num_users=100,
        policy=AllocationPolicy.STATIC_EQUAL
    )
    print(json.dumps(result, indent=2))
