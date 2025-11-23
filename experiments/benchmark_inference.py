"""
Benchmark REAL inference time for different DSS policies.

Reports ACTUAL measured times - no fabricated numbers.

Phase 6: Measurement Tools

Usage:
    python experiments/benchmark_inference.py --n-users 100 --n-iterations 1000
"""

import argparse
import time
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.dss.spectrum_environment import SpectrumEnvironment
from src.dss.policies.fuzzy_adaptive import FuzzyAdaptivePolicy
from src.dss.policies.dqn_baseline import DQNPolicy
from src.dss.policies.priority import PriorityPolicy
from src.dss.policies.static import StaticPolicy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_random_contexts(n_users: int) -> dict:
    """Generate random user contexts for benchmarking"""
    contexts = {}
    for i in range(n_users):
        contexts[f'user_{i}'] = {
            'throughput': np.random.uniform(0.3, 1.0),
            'latency': np.random.uniform(0.0, 0.5),
            'outage': np.random.uniform(0.0, 0.3),
            'priority': np.random.uniform(0.3, 1.0),
            'doppler': np.random.uniform(0.0, 0.5),
            'elevation': np.random.uniform(0.3, 1.0),
            'beam_load': np.random.uniform(0.2, 0.8),
            'beam_id': f'beam_{i % 10}'
        }
    return contexts


def generate_test_data(n_users: int):
    """Generate test users, qos, and context for policy allocation"""
    users = []
    qos = {}
    context = {}
    
    for i in range(n_users):
        user_id = f'user_{i}'
        users.append({
            'id': user_id,
            'priority': np.random.uniform(0.3, 1.0),
            'operator': i % 3
        })
        
        qos[user_id] = {
            'throughput': np.random.uniform(30e6, 100e6),
            'latency': np.random.uniform(0.0, 0.5),
            'outage': np.random.uniform(0.0, 0.3)
        }
        
        context[user_id] = {
            'throughput': np.random.uniform(0.3, 1.0),
            'latency': np.random.uniform(0.0, 0.5),
            'outage': np.random.uniform(0.0, 0.3),
            'priority': users[-1]['priority'],
            'doppler': np.random.uniform(0.0, 0.5),
            'elevation': np.random.uniform(0.3, 1.0),
            'beam_load': np.random.uniform(0.2, 0.8),
            'beam_id': f'beam_{i % 10}'
        }
    
    return users, qos, context


def benchmark_policy(policy, policy_name: str, n_users: int, n_iterations: int, env: SpectrumEnvironment):
    """Benchmark REAL inference time for a single policy"""
    logger.info(f"Benchmarking {policy_name} ({n_users} users, {n_iterations} iterations)...")
    
    times_ms = []
    
    for iteration in range(n_iterations):
        users, qos, context = generate_test_data(n_users)
        
        # Measure ACTUAL time
        t_start = time.perf_counter()
        
        # Call policy allocation (matching main.py interface)
        if policy_name in ['fuzzy', 'dqn']:
            allocations = policy.allocate(
                users=users,
                qos=qos,
                context=context,
                bandwidth_hz=100e6
            )
        elif policy_name == 'static':
            # Static policy uses different interface
            demands_array = np.array([qos[u['id']].get('throughput', 0.0) for u in users])
            allocations_array = policy.allocate(
                demands=demands_array,
                available_resources=100e6 * n_users,
                weights=None
            )
            # Convert to allocation format
            allocations = {}
            for i, user in enumerate(users):
                if allocations_array[i] > 0:
                    allocations[user['id']] = (11e9, 20.0)  # Mock allocation
                else:
                    allocations[user['id']] = None
        elif policy_name == 'priority':
            # Priority policy uses different interface
            demands_array = np.array([qos[u['id']].get('throughput', 0.0) for u in users])
            priorities_array = np.array([u.get('priority', 0.5) for u in users])
            allocations_array = policy.allocate(
                demands=demands_array,
                available_resources=100e6 * n_users,
                priorities=priorities_array
            )
            # Convert to allocation format
            allocations = {}
            for i, user in enumerate(users):
                if allocations_array[i] > 0:
                    allocations[user['id']] = (11e9, 20.0)  # Mock allocation
                else:
                    allocations[user['id']] = None
        else:
            allocations = {}
        
        t_end = time.perf_counter()
        
        times_ms.append((t_end - t_start) * 1000.0)  # Convert to milliseconds
        
        if iteration % 100 == 0 and iteration > 0:
            logger.debug(f"  Iteration {iteration}/{n_iterations}")
    
    # Compute REAL statistics
    results = {
        'policy': policy_name,
        'n_users': n_users,
        'n_iterations': n_iterations,
        'mean_ms': np.mean(times_ms),
        'std_ms': np.std(times_ms),
        'median_ms': np.median(times_ms),
        'p50_ms': np.percentile(times_ms, 50),
        'p95_ms': np.percentile(times_ms, 95),
        'p99_ms': np.percentile(times_ms, 99),
        'min_ms': np.min(times_ms),
        'max_ms': np.max(times_ms)
    }
    
    logger.info(f"  ✓ {policy_name}: mean={results['mean_ms']:.4f}ms, "
                f"p95={results['p95_ms']:.4f}ms, p99={results['p99_ms']:.4f}ms")
    
    return results, times_ms


def main(args):
    logger.info("=== REAL Inference Benchmark ===")
    logger.info("All times are ACTUAL measured values")
    
    # Initialize environment
    env = SpectrumEnvironment((10e9, 12e9))
    
    # Initialize policies
    policies = {}
    
    if 'static' in args.policies:
        policies['static'] = StaticPolicy()
    
    if 'priority' in args.policies:
        policies['priority'] = PriorityPolicy()
    
    if 'fuzzy' in args.policies:
        policies['fuzzy'] = FuzzyAdaptivePolicy(env)
    
    if 'dqn' in args.policies:
        dqn_model_path = args.dqn_model_path or 'models/dqn/dqn_baseline_final.h5'
        if not Path(dqn_model_path).exists():
            logger.warning(f"DQN model not found: {dqn_model_path}")
            logger.warning("Using untrained DQN. Train with: python scripts/train_dqn_baseline.py")
            try:
                policies['dqn'] = DQNPolicy(env, model_path=None)
            except Exception as e:
                logger.error(f"Could not initialize DQN: {e}")
                args.policies = [p for p in args.policies if p != 'dqn']
        else:
            policies['dqn'] = DQNPolicy(env, model_path=dqn_model_path)
    
    if not policies:
        logger.error("No valid policies to benchmark!")
        return
    
    # Benchmark each policy
    all_results = []
    all_times = {}
    
    for policy_name, policy in policies.items():
        try:
            results, times = benchmark_policy(policy, policy_name, args.n_users, args.n_iterations, env)
            all_results.append(results)
            all_times[policy_name] = times
        except Exception as e:
            logger.error(f"Failed to benchmark {policy_name}: {e}")
            import traceback
            traceback.print_exc()
    
    if not all_results:
        logger.error("No benchmark results collected!")
        return
    
    # Save REAL results
    results_df = pd.DataFrame(all_results)
    output_path = Path(args.output_dir) / f"inference_benchmark_n{args.n_users}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    logger.info(f"✓ REAL results saved to {output_path}")
    
    # Print summary
    print("\n=== REAL Measured Inference Times ===")
    print(results_df[['policy', 'mean_ms', 'p95_ms', 'p99_ms']].to_string(index=False))
    
    # Compute REAL speedup
    if 'dqn' in all_times:
        dqn_mean = results_df[results_df['policy'] == 'dqn']['mean_ms'].values[0]
        print(f"\n=== Speedup vs DQN (REAL measurements) ===")
        for _, row in results_df.iterrows():
            if row['policy'] != 'dqn':
                speedup = dqn_mean / row['mean_ms']
                print(f"{row['policy']:10s}: {speedup:.1f}x faster than DQN")
    
    # Save raw times
    times_path = Path(args.output_dir) / f"inference_times_raw_n{args.n_users}.npz"
    np.savez(times_path, **all_times)
    logger.info(f"✓ Raw timing data saved to {times_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark REAL inference times for DSS policies")
    parser.add_argument("--n-users", type=int, default=100, help="Number of users")
    parser.add_argument("--n-iterations", type=int, default=1000, help="Number of iterations")
    parser.add_argument("--policies", nargs='+', default=['static', 'priority', 'fuzzy', 'dqn'],
                       help="Policies to benchmark")
    parser.add_argument("--dqn-model-path", type=str, default=None,
                       help="Path to trained DQN model")
    parser.add_argument("--output-dir", type=str, default="results/benchmarks",
                       help="Output directory")
    
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

