#!/usr/bin/env python3
"""
Full Simulation Workflow: Data Generation → Operator Config → Allocation → Fairness Analysis

Demonstrates the complete multi-operator LEO fairness framework workflow.
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.operators import (
    Operator, OperatorConfig, OperatorType,
    Constellation, SpectrumBandManager, BandType
)
from src.allocation import (
    AllocationEngine, AllocationPolicy,
    AllocationRequest, AllocationResult, ResourceTracker
)
from src.fairness.traditional import TraditionalFairness
from src.fairness.vector_metrics import VectorFairness, MultiDimensionalMetrics
from src.data import SyntheticDataGenerator, DataValidator


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def run_full_simulation():
    """Run complete simulation workflow."""
    
    print_section("MULTI-OPERATOR LEO FAIRNESS SIMULATION")
    print("Complete workflow: Data Generation → Operators → Allocation → Fairness")
    print()
    
    # ============================================================
    # STEP 1: Data Generation
    # ============================================================
    print_section("STEP 1: Synthetic Data Generation")
    
    generator = SyntheticDataGenerator(
        region_bounds={
            "lat": (35.0, 45.0),  # US East Coast
            "lon": (-85.0, -70.0)
        },
        population_density_per_km2=0.15
    )
    
    # Generate users for multiple operators
    operators_list = ["starlink", "kuiper", "oneweb"]
    num_users = 100
    
    print(f"Generating {num_users} users for {len(operators_list)} operators...")
    users = generator.generate_users(num_users, operators_list, distribution="clustered")
    
    print(f"✓ Generated {len(users)} users")
    for op_id in operators_list:
        count = sum(1 for u in users if u.operator_id == op_id)
        print(f"  - {op_id.capitalize()}: {count}")
    
    # Generate traffic timeline
    start_time = datetime.now()
    patterns = generator.generate_traffic_timeline(
        users, start_time, duration_hours=1.0, interval_minutes=10
    )
    print(f"✓ Generated {len(patterns)} traffic patterns")
    
    # Validate data
    validator = DataValidator()
    validation_results = validator.validate_user_distribution(users)
    passed = sum(1 for r in validation_results if r.passed)
    print(f"✓ Validation: {passed}/{len(validation_results)} checks passed")
    
    # ============================================================
    # STEP 2: Operator Configuration
    # ============================================================
    print_section("STEP 2: Operator Configuration & Spectrum Assignment")
    
    # Create spectrum manager
    spectrum_manager = SpectrumBandManager(
        total_spectrum_start_mhz=10000.0,  # 10 GHz
        total_spectrum_end_mhz=40000.0     # 40 GHz
    )
    
    # Configure operators
    operators = {}
    operator_configs = {
        "starlink": {
            "altitude_km": 550.0,
            "inclination_deg": 53.0,
            "num_satellites": 100,
            "num_planes": 10,
            "satellites_per_plane": 10,
            "spectrum_bands": [(10000.0, 12000.0), (15000.0, 17000.0)]  # 4000 MHz total
        },
        "kuiper": {
            "altitude_km": 630.0,
            "inclination_deg": 51.9,
            "num_satellites": 80,
            "num_planes": 8,
            "satellites_per_plane": 10,
            "spectrum_bands": [(20000.0, 22000.0), (25000.0, 27000.0)]  # 4000 MHz total
        },
        "oneweb": {
            "altitude_km": 1200.0,
            "inclination_deg": 87.4,
            "num_satellites": 50,
            "num_planes": 5,
            "satellites_per_plane": 10,
            "spectrum_bands": [(30000.0, 32000.0)]  # 2000 MHz total
        }
    }
    
    for op_id, config in operator_configs.items():
        # Create operator config
        op_config = OperatorConfig(
            name=op_id.capitalize(),
            operator_type=OperatorType.CUSTOM,
            constellation_altitude_km=config["altitude_km"],
            constellation_inclination_deg=config["inclination_deg"],
            num_satellites=config["num_satellites"],
            num_planes=config["num_planes"],
            satellites_per_plane=config["satellites_per_plane"],
            spectrum_bands_mhz=config["spectrum_bands"]
        )
        
        # Create operator
        operator = Operator(op_config)
        
        # Assign spectrum bands
        for start, end in config["spectrum_bands"]:
            spectrum_manager.assign_band_to_operator(start, end, op_id, BandType.KA_BAND)
        
        # Create constellation
        constellation = Constellation(
            operator_name=op_id,
            altitude_km=config["altitude_km"],
            inclination_deg=config["inclination_deg"],
            num_satellites=config["num_satellites"],
            num_planes=config["num_planes"],
            satellites_per_plane=config["satellites_per_plane"]
        )
        operator.set_constellation(constellation)
        
        operators[op_id] = operator
        
        print(f"✓ {op_id.capitalize()}:")
        print(f"    Satellites: {config['num_satellites']}")
        print(f"    Altitude: {config['altitude_km']} km")
        print(f"    Spectrum: {operator.get_total_spectrum_mhz():.0f} MHz")
    
    print(f"\n✓ Total spectrum allocated: {sum(op.get_total_spectrum_mhz() for op in operators.values()):.0f} MHz")
    print(f"✓ Available spectrum: {spectrum_manager.get_available_spectrum():.0f} MHz")
    
    # ============================================================
    # STEP 3: Resource Allocation
    # ============================================================
    print_section("STEP 3: Resource Allocation")
    
    # Use first traffic pattern for allocation
    current_pattern = patterns[0]
    
    # Create allocation requests
    requests = []
    for user in users:
        demand = current_pattern.user_demands.get(user.user_id, user.demand_mbps)
        requests.append(AllocationRequest(
            operator_id=user.operator_id,
            user_id=user.user_id,
            demand_mbps=demand,
            priority=user.priority,
            min_bandwidth_mhz=demand / 10.0,  # Simplified: 10 Mbps per MHz
            max_bandwidth_mhz=demand / 5.0
        ))
    
    print(f"Created {len(requests)} allocation requests")
    print(f"Total demand: {sum(r.demand_mbps for r in requests):.2f} Mbps")
    
    # Test each policy
    policies_to_test = [
        ("Static Equal", AllocationPolicy.STATIC_EQUAL),
        ("Static Proportional", AllocationPolicy.STATIC_PROPORTIONAL),
        ("Priority Based", AllocationPolicy.PRIORITY_BASED),
    ]
    
    allocation_results = {}
    
    for policy_name, policy in policies_to_test:
        print(f"\n--- Testing {policy_name} Policy ---")
        
        engine = AllocationEngine(spectrum_manager, policy)
        results = engine.allocate(requests, available_bandwidth_mhz=10000.0)
        
        # Create tracker for this policy
        tracker = ResourceTracker()
        tracker.update_allocations(results, current_pattern.timestamp)
        
        # Simulate performance metrics
        for result in results:
            if result.success:
                # Simulate performance based on allocation
                throughput = result.allocated_bandwidth_mhz * 10.0  # Simplified: 10 Mbps per MHz
                latency = 20.0 + np.random.normal(0, 5)  # ms
                latency = max(10.0, latency)
                
                # Update user performance in tracker
                user_metrics = tracker.user_metrics.get(result.user_id)
                if user_metrics:
                    user_metrics.throughput_mbps = throughput
                    user_metrics.latency_ms = latency
                    user_metrics.packet_loss_rate = 0.01
                    user_metrics.qos_satisfaction = 0.9 if throughput >= result.allocated_bandwidth_mhz * 8.0 else 0.7
        
        # Store results
        allocation_results[policy_name] = {
            "results": results,
            "tracker": tracker
        }
        
        successful = sum(1 for r in results if r.success)
        total_allocated = sum(r.allocated_bandwidth_mhz for r in results if r.success)
        
        print(f"  Allocated: {successful}/{len(results)} users")
        print(f"  Total bandwidth: {total_allocated:.2f} MHz")
    
    # ============================================================
    # STEP 4: Fairness Analysis
    # ============================================================
    print_section("STEP 4: Fairness Analysis")
    
    fairness_results = {}
    
    for policy_name, policy_data in allocation_results.items():
        print(f"\n--- {policy_name} Policy Fairness ---")
        
        results = policy_data["results"]
        tracker = policy_data["tracker"]
        
        # Group allocations by operator
        operator_allocations = {}
        operator_users = {}
        
        for result in results:
            if result.success:
                op_id = result.operator_id
                if op_id not in operator_allocations:
                    operator_allocations[op_id] = []
                    operator_users[op_id] = []
                
                operator_allocations[op_id].append(result.allocated_bandwidth_mhz)
                operator_users[op_id].append(result.user_id)
        
        # Traditional fairness metrics
        print("  Traditional Metrics:")
        all_allocations = [r.allocated_bandwidth_mhz for r in results if r.success]
        
        if all_allocations:
            jain = TraditionalFairness.jain_index(all_allocations)
            gini = TraditionalFairness.gini_coefficient(all_allocations)
            alpha = TraditionalFairness.alpha_fairness(all_allocations, alpha=1.0)
            
            print(f"    Jain Index: {jain:.4f}")
            print(f"    Gini Coefficient: {gini:.4f}")
            print(f"    Alpha-fairness: {alpha:.2f}")
            
            fairness_results[policy_name] = {
                "jain": jain,
                "gini": gini,
                "alpha": alpha,
                "allocations": all_allocations
            }
        
        # Per-operator fairness
        print("  Per-Operator Fairness:")
        for op_id, allocations in operator_allocations.items():
            if allocations:
                op_jain = TraditionalFairness.jain_index(allocations)
                print(f"    {op_id.capitalize()}: Jain={op_jain:.4f}, Users={len(allocations)}")
        
        # Vector-based fairness
        print("  Vector-Based Fairness:")
        vector_fairness = VectorFairness()
        
        # Create multi-dimensional metrics
        metrics_list = []
        for result in results:
            if result.success:
                # Tracker uses key format: "{operator_id}_{user_id}"
                user_key = f"{result.operator_id}_{result.user_id}"
                user_metrics = tracker.user_metrics.get(user_key)
                if user_metrics and user_metrics.throughput_mbps > 0:
                    metrics_list.append(MultiDimensionalMetrics(
                        throughput_mbps=user_metrics.throughput_mbps,
                        latency_ms=user_metrics.latency_ms,
                        access_rate=1.0 if result.success else 0.0,
                        coverage_quality=0.9,  # Simulated
                        qos_satisfaction=user_metrics.qos_satisfaction
                    ))
                elif result.success:
                    # If user metrics not found but allocation succeeded, use allocation data
                    throughput = result.allocated_bandwidth_mhz * 10.0
                    metrics_list.append(MultiDimensionalMetrics(
                        throughput_mbps=throughput,
                        latency_ms=20.0,
                        access_rate=1.0,
                        coverage_quality=0.9,
                        qos_satisfaction=0.8
                    ))
        
        if metrics_list:
            try:
                fairness_vector = vector_fairness.compute_fairness_vector(metrics_list)
                weighted_fairness = vector_fairness.compute_weighted_fairness(metrics_list)
                distance_fairness = vector_fairness.compute_distance_fairness(metrics_list)
                
                print(f"    Fairness Vector: [{', '.join(f'{v:.4f}' for v in fairness_vector)}]")
                print(f"    Weighted Fairness: {weighted_fairness:.4f}")
                print(f"    Distance Fairness: {distance_fairness:.4f}")
                
                if policy_name not in fairness_results:
                    fairness_results[policy_name] = {}
                fairness_results[policy_name]["weighted_fairness"] = weighted_fairness
                fairness_results[policy_name]["distance_fairness"] = distance_fairness
            except Exception as e:
                print(f"    Error computing vector fairness: {e}")
                if policy_name not in fairness_results:
                    fairness_results[policy_name] = {}
                fairness_results[policy_name]["weighted_fairness"] = 0.0
                fairness_results[policy_name]["distance_fairness"] = 0.0
        else:
            print("    No metrics available for vector-based fairness")
    
    # ============================================================
    # STEP 5: Summary & Comparison
    # ============================================================
    print_section("STEP 5: Policy Comparison Summary")
    
    # Create comparison table
    comparison_data = []
    for policy_name, results in fairness_results.items():
        comparison_data.append({
            "Policy": policy_name,
            "Jain Index": results.get("jain", 0.0),
            "Gini Coefficient": results.get("gini", 0.0),
            "Weighted Fairness": results.get("weighted_fairness", 0.0),
            "Distance Fairness": results.get("distance_fairness", 0.0),
            "Users Served": len(results.get("allocations", []))
        })
    
    df = pd.DataFrame(comparison_data)
    print(df.to_string(index=False))
    
    # Find best policy
    if comparison_data:
        best_jain = max(comparison_data, key=lambda x: x["Jain Index"])
        print(f"\n✓ Best Jain Index: {best_jain['Policy']} ({best_jain['Jain Index']:.4f})")
        
        best_weighted = max(comparison_data, key=lambda x: x["Weighted Fairness"])
        print(f"✓ Best Weighted Fairness: {best_weighted['Policy']} ({best_weighted['Weighted Fairness']:.4f})")
    
    # ============================================================
    # STEP 6: Export Results
    # ============================================================
    print_section("STEP 6: Exporting Results")
    
    # Create results directory if it doesn't exist
    # Try current directory first, then home directory
    try:
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)
        sim_dir = output_dir / "simulation"
        sim_dir.mkdir(exist_ok=True)
    except PermissionError:
        # Fallback to home directory
        output_dir = Path.home() / "fuzzy_fairness_results"
        output_dir.mkdir(exist_ok=True)
        sim_dir = output_dir / "simulation"
        sim_dir.mkdir(exist_ok=True)
        print(f"Note: Using {sim_dir} for results (permission issue with ./results)")
    
    # Save comparison table
    output_file = sim_dir / "policy_comparison.csv"
    df.to_csv(output_file, index=False)
    print(f"✓ Saved comparison table: {output_file}")
    
    # Save detailed results
    detailed_results = {
        "timestamp": datetime.now().isoformat(),
        "num_users": num_users,
        "operators": list(operators.keys()),
        "policies_tested": list(allocation_results.keys()),
        "fairness_results": {
            policy: {
                "jain": float(results.get("jain", 0.0)),
                "gini": float(results.get("gini", 0.0)),
                "weighted": float(results.get("weighted_fairness", 0.0)),
                "distance": float(results.get("distance_fairness", 0.0))
            }
            for policy, results in fairness_results.items()
        }
    }
    
    results_file = sim_dir / "simulation_results.json"
    with open(results_file, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    print(f"✓ Saved detailed results: {results_file}")
    
    print_section("SIMULATION COMPLETE")
    print("✓ All steps completed successfully!")
    print(f"✓ Results saved to: {sim_dir}")
    print()
    print("Next steps:")
    print("  - Review results in results/simulation/")
    print("  - Run with different policies or scenarios")
    print("  - Integrate RL agents for optimization")
    
    return detailed_results


if __name__ == "__main__":
    try:
        results = run_full_simulation()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
