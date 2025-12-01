"""
Main simulation script for FairShare: Multi-Operator Dynamic Spectrum Sharing in LEO Satellite.

Runs full physics-based simulation with orbit propagation, channel modeling,
fairness evaluation, and dynamic spectrum sharing.

GPU Optimization:
- Multi-GPU support with MirroredStrategy
- XLA JIT compilation
- GPU placement optimization
"""

import argparse
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import yaml
import os
import sys

# Add src to path for module imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# GPU Optimization Layer
# Note: TensorFlow import is delayed until after GPU configuration in main()
TF_AVAILABLE = False
tf = None
strategy = None

from src.dss import DSSSimulator
from src.dss.spectrum_environment import SpectrumEnvironment
from src.dss.spectrum_map import SpectrumMap
from src.dss.policies.static import StaticPolicy
from src.dss.policies.priority import PriorityPolicy
# Removed: FuzzyAdaptivePolicy (replaced with RL-based policies)
from src.channel import OrbitPropagator, SatelliteGeometry, ChannelModel
from src.fairness import TraditionalFairness, VectorFairness
from src.experiments import (
    ScenarioConfig, load_scenario,
    TrafficGenerator,
    MetricsLogger
)
from src.experiments.qos_estimator import QoSEstimator
from src.visualization import (
    plot_fairness_radar,
    plot_fairness_over_time,
    plot_spectrum_heatmap
)


def run_simulation(
    scenario_name: str,
    policy_name: str = 'priority',
    output_dir: str = 'results',
    use_gpu: bool = True,
    gpu_id: Optional[int] = None,
    duration_s: Optional[float] = None,
    dqn_model_path: Optional[str] = None
) -> None:
    """
    Run complete slot-based simulation from scenario.
    
    Phase 4: Complete end-to-end simulation with slot-based loop.
    
    Args:
        scenario_name: Scenario name (e.g., 'urban_congestion')
        policy_name: Policy name ('static', 'priority', 'rl')
        output_dir: Output directory for results
        use_gpu: Enable GPU acceleration
        gpu_id: Specific GPU ID to use (None = use all, 'cpu' = CPU only)
        duration_s: Override simulation duration in seconds
    """
    # GPU configuration and TensorFlow setup
    global strategy
    if TF_AVAILABLE and use_gpu and gpu_id != 'cpu':
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            # GPU visibility was already set via CUDA_VISIBLE_DEVICES in main()
            print(f"✓ Using {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")
            
            # Enable memory growth to avoid allocating all memory
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    print(f"Warning: Could not set memory growth for {gpu.name}: {e}")
            
            # Enable XLA JIT compilation for performance
            tf.config.optimizer.set_jit(True)
            print("✓ XLA JIT compilation enabled")
            
            # Create MirroredStrategy for multi-GPU training
            visible_gpus = tf.config.get_visible_devices('GPU')
            if len(visible_gpus) > 1:
                strategy = tf.distribute.MirroredStrategy()
                print(f"✓ MirroredStrategy initialized for {len(visible_gpus)} GPUs")
                print(f"  Number of replicas: {strategy.num_replicas_in_sync}")
            else:
                strategy = None
                print("✓ Single GPU mode")
        else:
            print("⚠ No GPUs found, using CPU")
            strategy = None
            tf.config.optimizer.set_jit(False)
    else:
        print("✓ Using CPU")
        strategy = None
        if TF_AVAILABLE:
            tf.config.optimizer.set_jit(False)
    
    # Load scenario
    try:
        config = load_scenario(scenario_name)
    except FileNotFoundError:
        # Try as file path
        config = ScenarioConfig(scenario_name)
    
    # Override duration if specified
    if duration_s is not None:
        config.sim_time_s = duration_s
    
    print(f"✓ Loaded scenario: {config.scenario_name}")
    print(f"  Users: {config.num_users}, Operators: {config.num_operators}")
    print(f"  Duration: {config.sim_time_s}s, Slots: {config.get_num_slots()}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize FIS (try GPU-accelerated version first)
    # Initialize fairness metrics
    traditional_fairness = TraditionalFairness()
    vector_fairness = VectorFairness()
    print("✓ Fairness metrics initialized")
    
    # Initialize components
    print("Initializing simulation components...")
    
    # Orbit propagator
    orbit_prop = None
    if config.tle_file and os.path.exists(config.tle_file):
        orbit_prop = OrbitPropagator(config.tle_file)
        print(f"✓ Loaded TLE file: {config.tle_file}")
    else:
        print("⚠ No TLE file, using default satellite position")
    
    # Geometry models (one per ground station)
    geometries = []
    for lat, lon in config.ground_stations:
        geom = SatelliteGeometry(lat, lon)
        geometries.append(geom)
    print(f"✓ Initialized {len(geometries)} geometry model(s)")
    
    # Channel model
    channel_model = ChannelModel(
        frequency_hz=config.carrier_frequency_hz,
        scenario='urban'
    )
    print("✓ Channel model initialized")
    
    # Spectrum environment and map
    spectrum_env = SpectrumEnvironment(config.frequency_range_hz)
    spectrum_map = SpectrumMap(config.frequency_range_hz)
    print("✓ Spectrum environment initialized")
    
    # Policy (use GPU-accelerated version if available)
    if policy_name == "static":
        policy = StaticPolicy()
    elif policy_name == "priority":
        policy = PriorityPolicy()
    elif policy_name == "rl" or policy_name == "dqn":
        # DQN baseline policy
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for DQN policy. Install with: pip install tensorflow")
        from src.dss.policies.dqn_baseline import DQNPolicy
        model_path = dqn_model_path or 'models/dqn/dqn_baseline_final.h5'
        if not os.path.exists(model_path):
            print(f"⚠ Warning: DQN model not found at {model_path}")
            print("  Using untrained DQN policy. Train with: python scripts/train_dqn_baseline.py")
            model_path = None
        policy = DQNPolicy(spectrum_env, model_path=model_path)
        print(f"✓ DQN policy initialized (model: {model_path if model_path else 'untrained'})")
    else:
        raise ValueError(f"Unknown policy: {policy_name}")
    print(f"✓ Policy initialized: {policy_name}")
    
    # Traffic generator
    traffic_gen = TrafficGenerator(config, seed=42)
    traffic_data = traffic_gen.generate()
    users = traffic_data['users']
    traffic = traffic_data['traffic']
    # FIXED: Store priorities for use in priority policy
    priorities_dict = traffic_data.get('priorities', {})
    print(f"✓ Generated {len(users)} users and {len(traffic)} traffic slots")
    if priorities_dict:
        print(f"✓ Priorities available for {len(priorities_dict)} users")
    
    # QoS estimator
    qos_estimator = QoSEstimator()
    
    # Metrics logger
    metrics_logger = MetricsLogger(config.scenario_name, policy_name, fis)
    
    # Simulation loop (slot-based)
    print(f"\n{'='*60}")
    print(f"Starting simulation: {config.sim_time_s}s, {config.get_num_slots()} slots")
    print(f"{'='*60}\n")
    
    start_time = config.get_start_datetime()
    num_slots = config.get_num_slots()
    slot_duration = config.slot_duration_s
    
    # Default satellite (if no TLE)
    default_sat_name = "STARLINK-1000"
    if orbit_prop and len(orbit_prop.satellites) > 0:
        sat_names = list(orbit_prop.satellites.keys())
        sat_name = sat_names[0]
    else:
        sat_name = default_sat_name
    
    # Main simulation loop
    for slot_idx in range(num_slots):
        current_time = start_time + timedelta(seconds=slot_idx * slot_duration)
        t = slot_idx * slot_duration
        
        if slot_idx % max(1, num_slots // 10) == 0:
            print(f"Slot {slot_idx}/{num_slots} (t={t:.2f}s)")
        
        # 1. Propagate satellites
        if orbit_prop and sat_name in orbit_prop.satellites:
            try:
                sat_pos_ecef, sat_vel_ecef = orbit_prop.propagate_ecef(sat_name, current_time)
            except:
                # Fallback: default position
                sat_pos_ecef = np.array([7000e3, 0, 0])
                sat_vel_ecef = np.array([0, 7500, 0])
        else:
            # Default position (simplified)
            sat_pos_ecef = np.array([7000e3, 0, 0])
            sat_vel_ecef = np.array([0, 7500, 0])
        
        # 2. Compute geometry and context per user (batch processing)
        user_context = {}
        link_budgets = {}
        geometries_list = []
        
        # FIXED: Add temporal variation to channel conditions (for dynamic Jain index)
        # Add random fading that changes each slot
        slot_fading_db = np.random.normal(0, 4)  # 4 dB std deviation per slot
        
        # First pass: compute all geometries
        for user in users:
            user_id = user['id']
            lat, lon = user['lat'], user['lon']
            
            # Create geometry for this user
            user_geom = SatelliteGeometry(lat, lon)
            
            # Convert ECEF to ECI for geometry computation (simplified)
            sat_pos_eci = sat_pos_ecef  # Simplified
            sat_vel_eci = sat_vel_ecef  # Simplified
            
            # Compute geometry
            geom = user_geom.compute_geometry(sat_pos_eci, sat_vel_eci, current_time)
            geometries_list.append(geom)
        
        # Batch compute link budgets on GPU if available
        if TF_AVAILABLE and use_gpu and gpu_id != 'cpu' and len(users) > 10:
            try:
                from src.channel.channel_model_gpu import ChannelModelGPU
                gpu_channel = ChannelModelGPU(
                    frequency_hz=config.carrier_frequency_hz,
                    use_gpu=True
                )
                link_budgets_list = gpu_channel.compute_link_budgets_batch(
                    geometries_list,
                    rain_rate_mmh=config.channel.get('rain_rate_mmh', 0.0),
                    bandwidth_hz=config.bandwidth_hz
                )
                # Convert to dictionary
                for i, user in enumerate(users):
                    link_budgets[user['id']] = link_budgets_list[i]
            except (ImportError, Exception):
                # Fallback to CPU
                for i, user in enumerate(users):
                    link_budgets[user['id']] = channel_model.compute_link_budget(
                        geometry=geometries_list[i],
                        rain_rate_mmh=config.channel.get('rain_rate_mmh', 0.0),
                        bandwidth_hz=config.bandwidth_hz
                    )
        else:
            # CPU fallback
            for i, user in enumerate(users):
                link_budgets[user['id']] = channel_model.compute_link_budget(
                    geometry=geometries_list[i],
                    rain_rate_mmh=config.channel.get('rain_rate_mmh', 0.0),
                    bandwidth_hz=config.bandwidth_hz
                )
        
        # Build context dictionaries
        for i, user in enumerate(users):
            user_id = user['id']
            geom = geometries_list[i]
            link_budget = link_budgets[user_id]
            
            elevation = geom['elevation']
            doppler = abs(geom['doppler_shift'])
            slant_range = geom['slant_range']
            
            # Get beam load
            operator = user.get('operator', 0)
            beam_id = f"beam_{operator}"
            beam_load = spectrum_env.compute_beam_load(beam_id)
            
            # Store context
            user_context[user_id] = {
                'elevation': elevation,
                'doppler': doppler,
                'beam_load': beam_load,
                'beam_id': beam_id,
                'slant_range': slant_range,
                'link_budget': link_budget
            }
        
        # 3. Get traffic demands for this slot
        slot_demands = traffic.get(t, {})
        
        # 4. Compute QoS (initial, before allocation)
        # Use previous allocations for initial QoS estimate
        initial_allocations = {uid: None for uid in [u['id'] for u in users]}
        qos = qos_estimator.compute_qos(users, link_budgets, initial_allocations, slot_demands)
        
        # FIXED: Add temporal variation to QoS (for dynamic Jain index)
        # Add random variation to throughput, latency, outage per slot
        for user_id in qos:
            # Add temporal variation (changes each slot)
            throughput_var = np.random.normal(0, 0.05)  # 5% std deviation
            latency_var = np.random.normal(0, 0.03)    # 3% std deviation
            outage_var = np.random.normal(0, 0.02)     # 2% std deviation
            
            # Apply variation (clip to valid ranges)
            qos[user_id]['throughput'] = np.clip(
                qos[user_id].get('throughput', 0.5) + throughput_var,
                0.0, 1.0
            )
            qos[user_id]['latency'] = np.clip(
                qos[user_id].get('latency', 0.5) + latency_var,
                0.0, 1.0
            )
            qos[user_id]['outage'] = np.clip(
                qos[user_id].get('outage', 0.5) + outage_var,
                0.0, 1.0
            )
        
        # 5. Call DSS policy
        # FIXED: Use per-user bandwidth (20 MHz) instead of total spectrum bandwidth
        per_user_bandwidth_hz = 20e6  # 20 MHz per user (reasonable for LEO)
        
        if policy_name == "rl" or policy_name == "dqn":
            # RL/DQN policy allocation
            allocations = policy.allocate(
                users=users,
                qos=qos,
                context=user_context,
                bandwidth_hz=per_user_bandwidth_hz  # FIXED: Use per-user bandwidth
            )
        elif policy_name == "static":
            # Static policy: equal allocation
            demands_array = np.array([slot_demands.get(u['id'], 0.0) for u in users])
            # FIXED: Use same resource constraint as Priority for fair comparison
            total_spectrum_bw = config.frequency_range_hz[1] - config.frequency_range_hz[0]
            available_resources_static = min(
                total_spectrum_bw * 0.8,  # Same constraint as Priority (80%)
                per_user_bandwidth_hz * len(users)
            )
            allocations_array = policy.allocate(
                demands=demands_array,
                available_resources=available_resources_static,
                weights=None
            )
            # Convert to allocation format
            allocations = {}
            for i, user in enumerate(users):
                if allocations_array[i] > 0:
                    # Allocate a channel
                    channels = spectrum_env.get_available_channels(per_user_bandwidth_hz)  # FIXED: Use per-user bandwidth
                    if channels:
                        freq, sinr = channels[0]
                        allocations[user['id']] = (freq, sinr)
                    else:
                        allocations[user['id']] = None
                else:
                    allocations[user['id']] = None
        elif policy_name == "priority":
            # Priority policy
            # FIXED: Get priorities from priorities_dict (set at function start)
            priorities_array = np.array([priorities_dict.get(u['id'], u.get('priority', 0.5)) for u in users])
            
            demands_array = np.array([slot_demands.get(u['id'], 0.0) for u in users])
            # FIXED: Constrain available resources to create competition
            # Use total spectrum bandwidth instead of per-user * num_users
            # This forces Priority Policy to make choices
            total_spectrum_bw = config.frequency_range_hz[1] - config.frequency_range_hz[0]  # Total spectrum
            available_resources = min(
                total_spectrum_bw * 0.8,  # Use 80% of total spectrum (allows more users while maintaining competition)
                per_user_bandwidth_hz * len(users)  # But don't exceed per-user * num_users
            )
            allocations_array = policy.allocate(
                demands=demands_array,
                available_resources=available_resources,
                priorities=priorities_array
            )
            # FIXED: Convert to allocation format, respecting priority order
            # Sort users by priority (higher priority first) for spectrum allocation
            user_priority_pairs = [(users[i], allocations_array[i], priorities_array[i]) 
                                   for i in range(len(users))]
            # Sort by priority (descending) - high priority users get spectrum first
            user_priority_pairs.sort(key=lambda x: x[2], reverse=True)
            
            allocations = {}
            # Allocate spectrum in priority order
            # High-priority users get spectrum first, may exhaust available channels
            for user, alloc_amount, priority in user_priority_pairs:
                if alloc_amount > 0:
                    # Allocate spectrum to this user (in priority order)
                    beam_id = f"beam_{user.get('operator', 0)}"
                    allocation = spectrum_env.allocate(
                        user_id=user['id'],
                        bandwidth_hz=per_user_bandwidth_hz,
                        beam_id=beam_id
                    )
                    allocations[user['id']] = allocation
                else:
                    allocations[user['id']] = None
        
        # 6. Update spectrum environment
        spectrum_env.update_interference_map()
        
        # 7. Recompute QoS with actual allocations
        qos = qos_estimator.compute_qos(users, link_budgets, allocations, slot_demands)
        
        # 8. Update metrics logger
        metrics_logger.update(slot_idx, users, qos, allocations, user_context)
        
        # 9. Clear allocations for next slot (per-slot allocation model)
        # This allows fresh allocations in each slot
        # Note: In a real system, allocations might persist, but for this simulation
        # we use per-slot allocation to test the policy each slot
        for beam_id in list(spectrum_env.beams.keys()):
            spectrum_env.clear_beam_usage(beam_id)
    
    print(f"\n{'='*60}")
    print("Simulation complete!")
    print(f"{'='*60}\n")
    
    # Save metrics to CSV
    csv_path = metrics_logger.to_csv(output_dir)
    print(f"✓ Metrics saved to: {csv_path}")
    
    # Print summary
    summary = metrics_logger.get_summary()
    print("\nSummary Statistics:")
    print(f"  Mean Jain Index: {summary.get('mean_jain', 0):.3f}")
    print(f"  Mean Weighted Fairness: {summary.get('mean_weighted_fairness', 0):.3f}")
    print(f"  Mean α-fairness (α=1): {summary.get('mean_alpha_1', 0):.3f}")
    print(f"  Mean Rate: {summary.get('mean_mean_rate', 0)/1e6:.2f} Mbps")
    print(f"  Cell Edge Rate: {summary.get('mean_cell_edge_rate', 0)/1e6:.2f} Mbps")
    print(f"  Operator Imbalance: {summary.get('mean_operator_imbalance', 0):.3f}")
    
    print(f"\n✓ Results saved to {output_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='FairShare: Deep Fairness Benchmarking for Multi-Operator Dynamic Spectrum Sharing in LEO Satellite'
    )
    parser.add_argument(
        '--scenario',
        type=str,
        default='urban_congestion',
        help='Scenario name (e.g., urban_congestion, rural_coverage) or path to YAML file'
    )
    parser.add_argument(
        '--policy',
        type=str,
        default='priority',
        choices=['static', 'priority', 'rl', 'dqn'],
        help='Allocation policy'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--gpu-id',
        type=str,
        default=None,
        help='GPU ID (0, 1, ...) or "cpu" for CPU-only'
    )
    parser.add_argument(
        '--duration-s',
        type=float,
        default=None,
        help='Override simulation duration in seconds'
    )
    parser.add_argument(
        '--dqn-model-path',
        type=str,
        default=None,
        help='Path to trained DQN model (for --policy dqn). Default: models/dqn/dqn_baseline_final.h5'
    )
    
    args = parser.parse_args()
    
    # Parse GPU ID and configure GPU BEFORE TensorFlow initializes
    gpu_id = None
    use_gpu = True
    if args.gpu_id == 'cpu':
        use_gpu = False
        gpu_id = None
    elif args.gpu_id is not None:
        try:
            gpu_id = int(args.gpu_id)
            use_gpu = True
        except ValueError:
            print(f"⚠ Invalid GPU ID: {args.gpu_id}, using CPU")
            use_gpu = False
            gpu_id = None
    
    # Configure GPU visibility BEFORE importing TensorFlow
    # For H100 x 2, use both GPUs (don't restrict to single GPU)
    if use_gpu and gpu_id != 'cpu':
        if gpu_id is not None and isinstance(gpu_id, int):
            # Single GPU mode
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            print(f"✓ Configured CUDA_VISIBLE_DEVICES={gpu_id} (single GPU)")
        else:
            # Multi-GPU mode (use all available GPUs)
            # Don't set CUDA_VISIBLE_DEVICES to allow access to all GPUs
            print("✓ Multi-GPU mode: Using all available GPUs")
    
    # Now import TensorFlow (after GPU configuration)
    global TF_AVAILABLE, tf, strategy
    try:
        import tensorflow as tf
        TF_AVAILABLE = True
    except ImportError:
        print("⚠ TensorFlow not available, GPU optimization disabled")
        TF_AVAILABLE = False
        tf = None
    
    run_simulation(
        scenario_name=args.scenario,
        policy_name=args.policy,
        output_dir=args.output,
        use_gpu=use_gpu,
        gpu_id=gpu_id,
        duration_s=args.duration_s,
        dqn_model_path=args.dqn_model_path
    )


if __name__ == '__main__':
    main()

