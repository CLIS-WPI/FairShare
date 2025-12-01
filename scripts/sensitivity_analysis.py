#!/usr/bin/env python3
"""
Sensitivity Analysis: Test robustness across parameter variations

Tests how fairness metrics respond to:
- Different user/operator counts
- Various allocation policies  
- Different demand patterns
- Satellite configuration changes
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import json
from datetime import datetime
from itertools import product

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.operators import Operator, OperatorConfig, OperatorType, SpectrumBandManager, BandType
from src.allocation import AllocationEngine, AllocationPolicy, AllocationRequest, ResourceTracker
from src.fairness.traditional import TraditionalFairness
from src.fairness.vector_metrics import VectorFairness, MultiDimensionalMetrics
from src.data import SyntheticDataGenerator


def run_single_config(num_users, num_operators, demand_pattern, policy_name):
    """Run a single simulation configuration."""
    try:
        # Generate users
        generator = SyntheticDataGenerator(
            region_bounds={"lat": (35.0, 45.0), "lon": (-85.0, -70.0)},
            population_density_per_km2=0.15
        )
        
        operators_list = [f"operator_{i}" for i in range(num_operators)]
        users = generator.generate_users(num_users, operators_list, distribution=demand_pattern)
        
        # Create spectrum manager
        spectrum_manager = SpectrumBandManager(
            total_spectrum_start_mhz=10000.0,
            total_spectrum_end_mhz=40000.0
        )
        
        # Configure operators
        operators = {}
        spectrum_per_op = 10000.0 / num_operators
        for i, op_id in enumerate(operators_list):
            start_freq = 10000.0 + i * spectrum_per_op
            end_freq = start_freq + spectrum_per_op
            
            op_config = OperatorConfig(
                name=op_id,
                operator_type=OperatorType.CUSTOM,
                constellation_altitude_km=550.0 + i * 100,
                constellation_inclination_deg=53.0,
                num_satellites=50 + i * 10,
                num_planes=5,
                satellites_per_plane=10,
                spectrum_bands_mhz=[(start_freq, end_freq)]
            )
            operators[op_id] = Operator(op_config)
            spectrum_manager.assign_band_to_operator(start_freq, end_freq, op_id, BandType.KA_BAND)
        
        # Create requests
        requests = []
        for user in users:
            requests.append(AllocationRequest(
                operator_id=user.operator_id,
                user_id=user.user_id,
                demand_mbps=user.demand_mbps,
                priority=user.priority
            ))
        
        # Run allocation
        policy_map = {
            'static_equal': AllocationPolicy.STATIC_EQUAL,
            'static_proportional': AllocationPolicy.STATIC_PROPORTIONAL,
            'priority': AllocationPolicy.PRIORITY_BASED
        }
        
        engine = AllocationEngine(spectrum_manager, policy_map.get(policy_name, AllocationPolicy.STATIC_EQUAL))
        results = engine.allocate(requests, available_bandwidth_mhz=10000.0)
        
        # Compute fairness
        allocations = [r.allocated_bandwidth_mhz for r in results if r.success]
        if not allocations:
            return None
        
        jain = TraditionalFairness.jain_index(allocations)
        gini = TraditionalFairness.gini_coefficient(allocations)
        
        # Vector fairness
        tracker = ResourceTracker()
        tracker.update_allocations(results, datetime.now())
        
        metrics_list = []
        for result in results:
            if result.success:
                user_key = f"{result.operator_id}_{result.user_id}"
                user_metrics = tracker.user_metrics.get(user_key)
                if user_metrics:
                    metrics_list.append(MultiDimensionalMetrics(
                        throughput_mbps=result.allocated_bandwidth_mhz * 10.0,
                        latency_ms=20.0,
                        access_rate=1.0,
                        coverage_quality=0.9,
                        qos_satisfaction=0.8
                    ))
        
        vector_fairness = VectorFairness()
        weighted = vector_fairness.compute_weighted_fairness(metrics_list) if metrics_list else 0.0
        
        return {
            "jain_index": jain,
            "gini_coefficient": gini,
            "weighted_fairness": weighted,
            "users_served": len(allocations),
            "total_allocated": sum(allocations)
        }
    except Exception as e:
        print(f"  Error: {e}")
        return None


def run_sensitivity_experiment():
    """Run comprehensive sensitivity analysis."""
    print("=" * 70)
    print("SENSITIVITY ANALYSIS")
    print("=" * 70)
    
    # Parameter ranges
    num_users_range = [50, 100, 200, 500]
    num_operators_range = [2, 3, 4, 5]
    demand_patterns = ['uniform', 'clustered', 'gaussian']
    policies = ['static_equal', 'static_proportional', 'priority']
    
    total_configs = len(num_users_range) * len(num_operators_range) * len(demand_patterns) * len(policies)
    print(f"Testing {total_configs} configurations...\n")
    
    results = []
    config_idx = 0
    
    for num_users, num_operators, demand_pattern, policy in product(
        num_users_range, num_operators_range, demand_patterns, policies
    ):
        config_idx += 1
        if config_idx % 10 == 0:
            print(f"Progress: {config_idx}/{total_configs} ({100*config_idx/total_configs:.1f}%)")
        
        result = run_single_config(num_users, num_operators, demand_pattern, policy)
        if result:
            result.update({
                "num_users": num_users,
                "num_operators": num_operators,
                "demand_pattern": demand_pattern,
                "policy": policy
            })
            results.append(result)
    
    # Save results
    df = pd.DataFrame(results)
    try:
        output_dir = Path("results/sensitivity_analysis")
        output_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        output_dir = Path.home() / "fuzzy_fairness_results/sensitivity_analysis"
        output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "sensitivity_results.csv", index=False)
    
    # Analysis
    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS RESULTS")
    print("=" * 70)
    
    print("\nBy Number of Users:")
    print(df.groupby("num_users")[["jain_index", "gini_coefficient", "weighted_fairness"]].mean())
    
    print("\nBy Number of Operators:")
    print(df.groupby("num_operators")[["jain_index", "gini_coefficient", "weighted_fairness"]].mean())
    
    print("\nBy Demand Pattern:")
    print(df.groupby("demand_pattern")[["jain_index", "gini_coefficient", "weighted_fairness"]].mean())
    
    print("\nBy Policy:")
    print(df.groupby("policy")[["jain_index", "gini_coefficient", "weighted_fairness"]].mean())
    
    print(f"\n✓ Results saved to: {output_dir}")
    return df


if __name__ == "__main__":
    df = run_sensitivity_experiment()
    print("\n✓ Sensitivity analysis complete!")
