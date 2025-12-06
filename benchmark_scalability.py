#!/usr/bin/env python3
"""
Scalability Benchmark: CPU vs GPU Performance Comparison

This script tests the framework's scalability by comparing CPU and GPU
performance across different user counts (10k, 20k, 50k, 100k).

Purpose: Demonstrate GPU acceleration for large-scale scenarios (Mega-Constellations).
"""

import os
import sys
import time
import argparse
import numpy as np
from typing import Dict, List, Tuple

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠ TensorFlow not available, GPU tests will be skipped")
    tf = None

from src.experiments.scenario_loader import load_scenario, ScenarioConfig
from src.channel.channel_model_gpu import ChannelModelGPU
from src.channel.channel_model import ChannelModel


def benchmark_channel_model(
    num_users: int,
    use_gpu: bool = True,
    num_slots: int = 10
) -> Dict[str, float]:
    """
    Benchmark channel model computation for given number of users.
    
    Args:
        num_users: Number of users to simulate
        use_gpu: Whether to use GPU acceleration
        num_slots: Number of time slots to simulate
        
    Returns:
        Dictionary with timing and performance metrics
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking: {num_users:,} users, {'GPU' if use_gpu else 'CPU'}")
    print(f"{'='*60}")
    
    # Load base scenario
    try:
        config = load_scenario('paper_scenario_nyc')
    except FileNotFoundError:
        # Create minimal config for benchmark
        config = ScenarioConfig('paper_scenario_nyc')
        config.num_users = num_users
        config.carrier_frequency_hz = 12e9
        config.bandwidth_hz = 400e6
    
    # Override user count
    config.num_users = num_users
    
    # Generate test geometries (simplified - just random positions)
    from src.experiments.traffic_generator import TrafficGenerator
    generator = TrafficGenerator(config)
    users = generator.generate_users()
    
    # Generate random geometries for testing
    geometries = []
    for user in users[:num_users]:
        geometries.append({
            'slant_range': np.random.uniform(500e3, 1500e3),  # 500-1500 km
            'elevation': np.random.uniform(30.0, 90.0),  # 30-90 degrees
            'azimuth': np.random.uniform(0.0, 360.0)
        })
    
    # Benchmark computation
    total_time = 0.0
    times_per_slot = []
    
    if use_gpu and TF_AVAILABLE:
        # GPU benchmark
        try:
            # Configure GPU
            gpus = tf.config.list_physical_devices('GPU')
            if not gpus:
                print("⚠ No GPUs available, falling back to CPU")
                use_gpu = False
            else:
                print(f"✓ Using GPU: {gpus[0].name}")
                gpu_channel = ChannelModelGPU(
                    frequency_hz=config.carrier_frequency_hz,
                    use_gpu=True
                )
                
                # Warm-up
                print("  Warming up GPU...")
                warmup_geom = [geometries[0]]
                _ = gpu_channel.compute_link_budgets_batch(
                    warmup_geom,
                    rain_rate_mmh=2.0,
                    bandwidth_hz=config.bandwidth_hz
                )
                
                # Actual benchmark
                print(f"  Running {num_slots} slots...")
                for slot in range(num_slots):
                    slot_start = time.time()
                    
                    # Process in batches if needed
                    batch_size = 10000  # Process 10k users per batch
                    for i in range(0, len(geometries), batch_size):
                        batch_geoms = geometries[i:i+batch_size]
                        _ = gpu_channel.compute_link_budgets_batch(
                            batch_geoms,
                            rain_rate_mmh=2.0,
                            bandwidth_hz=config.bandwidth_hz
                        )
                    
                    slot_time = time.time() - slot_start
                    times_per_slot.append(slot_time)
                    total_time += slot_time
                    
                    if (slot + 1) % 5 == 0:
                        print(f"    Slot {slot+1}/{num_slots}: {slot_time*1000:.2f}ms")
                
        except Exception as e:
            print(f"⚠ GPU benchmark failed: {e}")
            print("  Falling back to CPU...")
            use_gpu = False
    
    if not use_gpu:
        # CPU benchmark
        print("  Using CPU...")
        cpu_channel = ChannelModel(
            frequency_hz=config.carrier_frequency_hz
        )
        
        # Warm-up
        print("  Warming up CPU...")
        warmup_geom = geometries[0]
        _ = cpu_channel.compute_link_budget(
            slant_range=warmup_geom['slant_range'],
            elevation=warmup_geom['elevation'],
            rain_rate_mmh=2.0,
            bandwidth_hz=config.bandwidth_hz
        )
        
        # Actual benchmark
        print(f"  Running {num_slots} slots...")
        for slot in range(num_slots):
            slot_start = time.time()
            
            for geom in geometries:
                _ = cpu_channel.compute_link_budget(
                    slant_range=geom['slant_range'],
                    elevation=geom['elevation'],
                    rain_rate_mmh=2.0,
                    bandwidth_hz=config.bandwidth_hz
                )
            
            slot_time = time.time() - slot_start
            times_per_slot.append(slot_time)
            total_time += slot_time
            
            if (slot + 1) % 5 == 0:
                print(f"    Slot {slot+1}/{num_slots}: {slot_time*1000:.2f}ms")
    
    # Calculate statistics
    avg_time = total_time / num_slots
    min_time = min(times_per_slot)
    max_time = max(times_per_slot)
    std_time = np.std(times_per_slot)
    
    throughput = num_users / avg_time  # users per second
    
    results = {
        'num_users': num_users,
        'device': 'GPU' if use_gpu else 'CPU',
        'total_time': total_time,
        'avg_time_per_slot': avg_time,
        'min_time': min_time,
        'max_time': max_time,
        'std_time': std_time,
        'throughput': throughput,
        'num_slots': num_slots
    }
    
    print(f"\n  Results:")
    print(f"    Total time: {total_time:.3f}s")
    print(f"    Avg per slot: {avg_time*1000:.2f}ms")
    print(f"    Throughput: {throughput:.0f} users/s")
    print(f"    Min/Max: {min_time*1000:.2f}ms / {max_time*1000:.2f}ms")
    
    return results


def run_scalability_benchmark(
    user_counts: List[int] = [10000, 20000, 50000, 100000],
    num_slots: int = 10,
    gpu_id: int = 0
) -> Tuple[List[Dict], List[Dict]]:
    """
    Run scalability benchmark comparing CPU and GPU performance.
    
    Args:
        user_counts: List of user counts to test
        num_slots: Number of slots per test
        gpu_id: GPU ID to use (or 'cpu' for CPU-only)
        
    Returns:
        Tuple of (cpu_results, gpu_results)
    """
    print("="*60)
    print("SCALABILITY BENCHMARK: CPU vs GPU")
    print("="*60)
    print(f"Testing user counts: {user_counts}")
    print(f"Slots per test: {num_slots}")
    print()
    
    # Configure GPU visibility
    if gpu_id != 'cpu' and TF_AVAILABLE:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        print(f"✓ Configured CUDA_VISIBLE_DEVICES={gpu_id}")
    
    cpu_results = []
    gpu_results = []
    
    for num_users in user_counts:
        # CPU benchmark
        print(f"\n{'#'*60}")
        print(f"CPU Benchmark: {num_users:,} users")
        print(f"{'#'*60}")
        cpu_result = benchmark_channel_model(
            num_users=num_users,
            use_gpu=False,
            num_slots=num_slots
        )
        cpu_results.append(cpu_result)
        
        # GPU benchmark (if available)
        if TF_AVAILABLE and gpu_id != 'cpu':
            print(f"\n{'#'*60}")
            print(f"GPU Benchmark: {num_users:,} users")
            print(f"{'#'*60}")
            gpu_result = benchmark_channel_model(
                num_users=num_users,
                use_gpu=True,
                num_slots=num_slots
            )
            gpu_results.append(gpu_result)
        else:
            print("\n⚠ Skipping GPU benchmark (TensorFlow not available or CPU-only mode)")
    
    return cpu_results, gpu_results


def plot_results(cpu_results: List[Dict], gpu_results: List[Dict], output_file: str = 'scalability_benchmark.png'):
    """Plot scalability benchmark results."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠ Matplotlib not available, skipping plot generation")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Extract data
    cpu_users = [r['num_users'] for r in cpu_results]
    cpu_times = [r['avg_time_per_slot'] * 1000 for r in cpu_results]  # Convert to ms
    
    if gpu_results:
        gpu_users = [r['num_users'] for r in gpu_results]
        gpu_times = [r['avg_time_per_slot'] * 1000 for r in gpu_results]
    
    # Plot 1: Time per slot
    ax1.plot(cpu_users, cpu_times, 'o-', label='CPU', linewidth=2, markersize=8)
    if gpu_results:
        ax1.plot(gpu_users, gpu_times, 's-', label='GPU', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Users', fontsize=12)
    ax1.set_ylabel('Time per Slot (ms)', fontsize=12)
    ax1.set_title('Computation Time vs User Count', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    # Plot 2: Speedup
    if gpu_results:
        speedups = [cpu_times[i] / gpu_times[i] for i in range(len(cpu_times))]
        ax2.plot(cpu_users, speedups, 'o-', label='GPU Speedup', linewidth=2, markersize=8, color='green')
        ax2.axhline(y=1.0, color='r', linestyle='--', label='Break-even', alpha=0.7)
        ax2.set_xlabel('Number of Users', fontsize=12)
        ax2.set_ylabel('Speedup (CPU time / GPU time)', fontsize=12)
        ax2.set_title('GPU Speedup vs User Count', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {output_file}")


def save_results(cpu_results: List[Dict], gpu_results: List[Dict], output_file: str = 'scalability_benchmark.csv'):
    """Save benchmark results to CSV."""
    import csv
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'num_users', 'device', 'avg_time_per_slot_ms', 'min_time_ms', 
            'max_time_ms', 'std_time_ms', 'throughput_users_per_sec', 'num_slots'
        ])
        writer.writeheader()
        
        for result in cpu_results:
            writer.writerow({
                'num_users': result['num_users'],
                'device': 'CPU',
                'avg_time_per_slot_ms': result['avg_time_per_slot'] * 1000,
                'min_time_ms': result['min_time'] * 1000,
                'max_time_ms': result['max_time'] * 1000,
                'std_time_ms': result['std_time'] * 1000,
                'throughput_users_per_sec': result['throughput'],
                'num_slots': result['num_slots']
            })
        
        for result in gpu_results:
            writer.writerow({
                'num_users': result['num_users'],
                'device': 'GPU',
                'avg_time_per_slot_ms': result['avg_time_per_slot'] * 1000,
                'min_time_ms': result['min_time'] * 1000,
                'max_time_ms': result['max_time'] * 1000,
                'std_time_ms': result['std_time'] * 1000,
                'throughput_users_per_sec': result['throughput'],
                'num_slots': result['num_slots']
            })
    
    print(f"✓ Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Scalability Benchmark: CPU vs GPU')
    parser.add_argument('--user-counts', type=int, nargs='+', 
                       default=[10000, 20000, 50000, 100000],
                       help='User counts to test (default: 10k, 20k, 50k, 100k)')
    parser.add_argument('--slots', type=int, default=10,
                       help='Number of slots per test (default: 10)')
    parser.add_argument('--gpu-id', type=str, default='0',
                       help='GPU ID to use, or "cpu" for CPU-only (default: 0)')
    parser.add_argument('--output-dir', type=str, default='results/benchmark',
                       help='Output directory for results (default: results/benchmark)')
    
    args = parser.parse_args()
    
    # Convert gpu_id
    if args.gpu_id.lower() == 'cpu':
        gpu_id = 'cpu'
    else:
        try:
            gpu_id = int(args.gpu_id)
        except ValueError:
            print(f"⚠ Invalid GPU ID: {args.gpu_id}, using CPU")
            gpu_id = 'cpu'
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run benchmark
    cpu_results, gpu_results = run_scalability_benchmark(
        user_counts=args.user_counts,
        num_slots=args.slots,
        gpu_id=gpu_id
    )
    
    # Save results
    csv_file = os.path.join(args.output_dir, 'scalability_benchmark.csv')
    save_results(cpu_results, gpu_results, csv_file)
    
    # Plot results
    plot_file = os.path.join(args.output_dir, 'scalability_benchmark.png')
    plot_results(cpu_results, gpu_results, plot_file)
    
    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    print(f"\nCPU Results:")
    for r in cpu_results:
        print(f"  {r['num_users']:,} users: {r['avg_time_per_slot']*1000:.2f}ms/slot")
    
    if gpu_results:
        print(f"\nGPU Results:")
        for r in gpu_results:
            print(f"  {r['num_users']:,} users: {r['avg_time_per_slot']*1000:.2f}ms/slot")
        
        print(f"\nSpeedup:")
        for i, cpu_r in enumerate(cpu_results):
            if i < len(gpu_results):
                gpu_r = gpu_results[i]
                speedup = cpu_r['avg_time_per_slot'] / gpu_r['avg_time_per_slot']
                print(f"  {cpu_r['num_users']:,} users: {speedup:.2f}x")
    
    print(f"\n✓ Results saved to: {args.output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()

