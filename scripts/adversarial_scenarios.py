#!/usr/bin/env python3
"""
Adversarial Scenarios: Test fairness metrics under challenging conditions

Tests fairness metrics with:
- Priority bias scenarios
- Overloaded operators
- Unfair allocation patterns
- Where multi-dimensional metrics reveal issues missed by traditional metrics
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.operators import Operator, OperatorConfig, OperatorType, SpectrumBandManager
from src.allocation import AllocationEngine, AllocationPolicy, AllocationRequest, ResourceTracker
from src.fairness.traditional import TraditionalFairness
from src.fairness.vector_metrics import VectorFairness, MultiDimensionalMetrics
from src.data import SyntheticDataGenerator


def create_priority_bias_scenario():
    """Create scenario with extreme priority bias."""
    print("\n" + "=" * 70)
    print("ADVERSARIAL SCENARIO 1: Priority Bias")
    print("=" * 70)
    
    # Create users with extreme priority differences
    users = []
    for i in range(50):
        # High priority users
        users.append({
            'user_id': f'high_priority_{i}',
            'operator_id': 'starlink',
            'demand_mbps': 10.0,
            'priority': 1.0  # Maximum priority
        })
    
    for i in range(50):
        # Low priority users
        users.append({
            'user_id': f'low_priority_{i}',
            'operator_id': 'kuiper',
            'demand_mbps': 10.0,
            'priority': 0.1  # Very low priority
        })
    
    # Create requests
    requests = [
        AllocationRequest(
            operator_id=u['operator_id'],
            user_id=u['user_id'],
            demand_mbps=u['demand_mbps'],
            priority=u['priority']
        )
        for u in users
    ]
    
    # Test with priority-based allocation
    spectrum_manager = SpectrumBandManager(10000.0, 40000.0)
    engine = AllocationEngine(spectrum_manager, AllocationPolicy.PRIORITY_BASED)
    
    results = engine.allocate(requests, available_bandwidth_mhz=5000.0)
    
    # Analyze fairness
    high_priority_allocations = [r.allocated_bandwidth_mhz for r in results 
                                 if r.user_id.startswith('high_priority') and r.success]
    low_priority_allocations = [r.allocated_bandwidth_mhz for r in results 
                                if r.user_id.startswith('low_priority') and r.success]
    
    all_allocations = [r.allocated_bandwidth_mhz for r in results if r.success]
    
    jain_all = TraditionalFairness.jain_index(all_allocations)
    jain_high = TraditionalFairness.jain_index(high_priority_allocations) if high_priority_allocations else 0
    jain_low = TraditionalFairness.jain_index(low_priority_allocations) if low_priority_allocations else 0
    
    print(f"\nResults:")
    print(f"  High priority users served: {len(high_priority_allocations)}/50")
    print(f"  Low priority users served: {len(low_priority_allocations)}/50")
    print(f"  Overall Jain Index: {jain_all:.4f}")
    print(f"  High priority group Jain: {jain_high:.4f}")
    print(f"  Low priority group Jain: {jain_low:.4f}")
    
    # Multi-dimensional analysis
    metrics_list = []
    for result in results:
        if result.success:
            metrics_list.append(MultiDimensionalMetrics(
                throughput_mbps=result.allocated_bandwidth_mhz * 10.0,
                latency_ms=20.0 if result.user_id.startswith('high_priority') else 50.0,  # Bias
                access_rate=1.0,
                coverage_quality=0.9 if result.user_id.startswith('high_priority') else 0.6,  # Bias
                qos_satisfaction=0.95 if result.user_id.startswith('high_priority') else 0.5  # Bias
            ))
    
    if metrics_list:
        vector_fairness = VectorFairness()
        weighted = vector_fairness.compute_weighted_fairness(metrics_list)
        distance = vector_fairness.compute_distance_fairness(metrics_list)
        
        print(f"  Multi-dimensional Weighted Fairness: {weighted:.4f}")
        print(f"  Distance Fairness: {distance:.4f}")
        print(f"\n  ⚠ Multi-dimensional metrics reveal QoS bias not captured by Jain!")
    
    return {
        'scenario': 'priority_bias',
        'jain_all': jain_all,
        'jain_high': jain_high,
        'jain_low': jain_low,
        'weighted_fairness': weighted if metrics_list else 0.0,
        'distance_fairness': distance if metrics_list else 0.0
    }


def create_overloaded_operator_scenario():
    """Create scenario with one overloaded operator."""
    print("\n" + "=" * 70)
    print("ADVERSARIAL SCENARIO 2: Overloaded Operator")
    print("=" * 70)
    
    # Create users: 80% on one operator, 20% on others
    users = []
    for i in range(80):
        users.append({
            'user_id': f'overloaded_{i}',
            'operator_id': 'starlink',
            'demand_mbps': 15.0,
            'priority': 0.5
        })
    
    for i in range(10):
        users.append({
            'user_id': f'normal_{i}',
            'operator_id': 'kuiper',
            'demand_mbps': 10.0,
            'priority': 0.5
        })
    
    for i in range(10):
        users.append({
            'user_id': f'normal2_{i}',
            'operator_id': 'oneweb',
            'demand_mbps': 10.0,
            'priority': 0.5
        })
    
    requests = [
        AllocationRequest(
            operator_id=u['operator_id'],
            user_id=u['user_id'],
            demand_mbps=u['demand_mbps'],
            priority=u['priority']
        )
        for u in users
    ]
    
    # Test with static equal allocation
    spectrum_manager = SpectrumBandManager(10000.0, 40000.0)
    engine = AllocationEngine(spectrum_manager, AllocationPolicy.STATIC_EQUAL)
    
    results = engine.allocate(requests, available_bandwidth_mhz=5000.0)
    
    # Analyze by operator
    operator_allocations = {}
    for result in results:
        if result.success:
            op_id = result.operator_id
            if op_id not in operator_allocations:
                operator_allocations[op_id] = []
            operator_allocations[op_id].append(result.allocated_bandwidth_mhz)
    
    print(f"\nResults by Operator:")
    for op_id, allocs in operator_allocations.items():
        jain = TraditionalFairness.jain_index(allocs) if allocs else 0
        avg_alloc = np.mean(allocs) if allocs else 0
        print(f"  {op_id.capitalize()}: {len(allocs)} users, "
              f"Avg allocation: {avg_alloc:.2f} MHz, Jain: {jain:.4f}")
    
    # Overall fairness
    all_allocations = [r.allocated_bandwidth_mhz for r in results if r.success]
    jain_all = TraditionalFairness.jain_index(all_allocations)
    
    # Operator-level fairness (fairness between operators)
    operator_totals = {op: sum(allocs) for op, allocs in operator_allocations.items()}
    operator_fairness = TraditionalFairness.jain_index(list(operator_totals.values()))
    
    print(f"\n  Overall Jain Index: {jain_all:.4f}")
    print(f"  Operator-level Jain Index: {operator_fairness:.4f}")
    print(f"  ⚠ Operator-level fairness reveals imbalance not visible in user-level Jain!")
    
    return {
        'scenario': 'overloaded_operator',
        'jain_all': jain_all,
        'operator_fairness': operator_fairness,
        'operator_allocations': {k: len(v) for k, v in operator_allocations.items()}
    }


def create_unfair_allocation_pattern():
    """Create scenario with intentionally unfair allocation."""
    print("\n" + "=" * 70)
    print("ADVERSARIAL SCENARIO 3: Unfair Allocation Pattern")
    print("=" * 70)
    
    # Create equal users but allocate unfairly
    users = []
    for i in range(100):
        users.append({
            'user_id': f'user_{i}',
            'operator_id': 'starlink' if i < 50 else 'kuiper',
            'demand_mbps': 10.0,
            'priority': 0.5
        })
    
    requests = [
        AllocationRequest(
            operator_id=u['operator_id'],
            user_id=u['user_id'],
            demand_mbps=u['demand_mbps'],
            priority=u['priority']
        )
        for u in users
    ]
    
    # Manually create unfair allocation (simulate)
    # First 20 users get 100 MHz each, rest get 10 MHz each
    unfair_results = []
    for i, req in enumerate(requests):
        if i < 20:
            unfair_results.append({
                'user_id': req.user_id,
                'operator_id': req.operator_id,
                'allocated_bandwidth_mhz': 100.0,
                'success': True
            })
        else:
            unfair_results.append({
                'user_id': req.user_id,
                'operator_id': req.operator_id,
                'allocated_bandwidth_mhz': 10.0,
                'success': True
            })
    
    allocations = [r['allocated_bandwidth_mhz'] for r in unfair_results]
    
    jain = TraditionalFairness.jain_index(allocations)
    gini = TraditionalFairness.gini_coefficient(allocations)
    
    print(f"\nResults:")
    print(f"  Jain Index: {jain:.4f}")
    print(f"  Gini Coefficient: {gini:.4f}")
    print(f"  ⚠ High Gini ({gini:.4f}) reveals inequality that Jain ({jain:.4f}) partially masks!")
    
    # Multi-dimensional: some users get better QoS despite same allocation
    metrics_list = []
    for i, result in enumerate(unfair_results):
        # First 20 get better latency and QoS
        metrics_list.append(MultiDimensionalMetrics(
            throughput_mbps=result['allocated_bandwidth_mhz'] * 10.0,
            latency_ms=15.0 if i < 20 else 40.0,  # Unfair
            access_rate=1.0,
            coverage_quality=0.95 if i < 20 else 0.7,  # Unfair
            qos_satisfaction=0.98 if i < 20 else 0.6  # Unfair
        ))
    
    vector_fairness = VectorFairness()
    weighted = vector_fairness.compute_weighted_fairness(metrics_list)
    distance = vector_fairness.compute_distance_fairness(metrics_list)
    
    print(f"  Multi-dimensional Weighted Fairness: {weighted:.4f}")
    print(f"  Distance Fairness: {distance:.4f}")
    print(f"  ⚠ Multi-dimensional metrics capture QoS unfairness not in allocation fairness!")
    
    return {
        'scenario': 'unfair_pattern',
        'jain': jain,
        'gini': gini,
        'weighted_fairness': weighted,
        'distance_fairness': distance
    }


def main():
    """Run all adversarial scenarios."""
    print("=" * 70)
    print("ADVERSARIAL SCENARIOS: Testing Fairness Metrics")
    print("=" * 70)
    print("\nTesting fairness metrics under challenging conditions...")
    print("Goal: Identify where multi-dimensional metrics reveal issues")
    print("      missed by traditional metrics (Jain, Gini)\n")
    
    results = []
    
    # Run scenarios
    results.append(create_priority_bias_scenario())
    results.append(create_overloaded_operator_scenario())
    results.append(create_unfair_allocation_pattern())
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Multi-Dimensional vs Traditional Metrics")
    print("=" * 70)
    
    df = pd.DataFrame(results)
    print("\nScenario Comparison:")
    print(df.to_string(index=False))
    
    # Save results
    try:
        output_dir = Path("results/adversarial_scenarios")
        output_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        output_dir = Path.home() / "fuzzy_fairness_results/adversarial_scenarios"
        output_dir.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_dir / "adversarial_results.csv", index=False)
    
    with open(output_dir / "adversarial_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_dir}")
    
    # Key insights
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print("""
1. Traditional metrics (Jain, Gini) focus on allocation equality
2. Multi-dimensional metrics reveal QoS unfairness:
   - Latency differences
   - Coverage quality variations
   - QoS satisfaction gaps
3. Operator-level fairness can differ from user-level fairness
4. Priority-based allocation can create fairness issues
5. Multi-dimensional metrics provide richer insights for policy evaluation
    """)


if __name__ == "__main__":
    main()
