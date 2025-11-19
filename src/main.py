"""
Main simulation script for fuzzy-fairness DSS in LEO satellite networks.

Runs full physics-based simulation with orbit propagation, channel modeling,
fuzzy fairness evaluation, and dynamic spectrum sharing.

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
try:
    import tensorflow as tf
    
    # Configure GPU visibility (use all GPUs by default)
    # For specific GPU: tf.config.set_visible_devices(['GPU:0'], 'GPU')
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Found {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")
        
        # Enable memory growth to avoid allocating all memory
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"Error setting memory growth for {gpu.name}: {e}")
        
        # Enable XLA JIT compilation for performance
        tf.config.optimizer.set_jit(True)
        print("✓ XLA JIT compilation enabled")
        
        # Create MirroredStrategy for multi-GPU training
        if len(gpus) > 1:
            strategy = tf.distribute.MirroredStrategy()
            print(f"✓ MirroredStrategy initialized for {len(gpus)} GPUs")
            print(f"  Number of replicas: {strategy.num_replicas_in_sync}")
        else:
            strategy = None
            print("✓ Single GPU mode")
    else:
        print("⚠ No GPUs found, using CPU")
        strategy = None
        tf.config.optimizer.set_jit(False)
    
    TF_AVAILABLE = True
except ImportError:
    print("⚠ TensorFlow not available, GPU optimization disabled")
    TF_AVAILABLE = False
    strategy = None

from src.dss import DSSSimulator
from src.dss.spectrum_environment import SpectrumEnvironment
from src.dss.spectrum_map import SpectrumMap
from src.dss.policies.static import StaticPolicy
from src.dss.policies.priority import PriorityPolicy
from src.dss.policies.fuzzy_adaptive import FuzzyAdaptivePolicy
from src.channel import OrbitPropagator, SatelliteGeometry, ChannelModel
from src.fairness import FairnessEvaluator, FuzzyInferenceSystem
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
    policy_name: str = 'fuzzy',
    output_dir: str = 'results',
    use_gpu: bool = True,
    gpu_id: Optional[int] = None,
    duration_s: Optional[float] = None
) -> None:
    """
    Run complete slot-based simulation from scenario.
    
    Phase 4: Complete end-to-end simulation with slot-based loop.
    
    Args:
        scenario_name: Scenario name (e.g., 'urban_congestion')
        policy_name: Policy name ('static', 'priority', 'fuzzy')
        output_dir: Output directory for results
        use_gpu: Enable GPU acceleration
        gpu_id: Specific GPU ID to use (None = use all, 'cpu' = CPU only)
        duration_s: Override simulation duration in seconds
    """
    # GPU configuration
    if TF_AVAILABLE and use_gpu and gpu_id != 'cpu':
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            if gpu_id is not None and isinstance(gpu_id, int):
                # Use specific GPU
                tf.config.set_visible_devices([gpus[gpu_id]], 'GPU')
                print(f"✓ Using GPU {gpu_id}: {gpus[gpu_id].name}")
            else:
                # Use all GPUs
                print(f"✓ Using all {len(gpus)} GPU(s)")
        else:
            print("⚠ No GPUs found, using CPU")
    else:
        print("✓ Using CPU")
    
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
    
    # Initialize FIS
    if TF_AVAILABLE and strategy is not None:
        with strategy.scope():
            fis = FuzzyInferenceSystem(use_phase3=True)
            print("✓ Fairness model initialized with MirroredStrategy")
    else:
        fis = FuzzyInferenceSystem(use_phase3=True)
        print("✓ Fairness model initialized (CPU)")
    
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
    
    # Policy
    if policy_name == "static":
        policy = StaticPolicy()
    elif policy_name == "priority":
        policy = PriorityPolicy()
    elif policy_name == "fuzzy":
        policy = FuzzyAdaptivePolicy(spectrum_env, spectrum_map, fis)
    else:
        raise ValueError(f"Unknown policy: {policy_name}")
    print(f"✓ Policy initialized: {policy_name}")
    
    # Traffic generator
    traffic_gen = TrafficGenerator(config, seed=42)
    traffic_data = traffic_gen.generate()
    users = traffic_data['users']
    traffic = traffic_data['traffic']
    print(f"✓ Generated {len(users)} users and {len(traffic)} traffic slots")
    
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
        
        # 2. Compute geometry and context per user
        user_context = {}
        link_budgets = {}
        
        for user in users:
            user_id = user['id']
            lat, lon = user['lat'], user['lon']
            
            # Create geometry for this user
            user_geom = SatelliteGeometry(lat, lon)
            
            # Convert ECEF to ECI for geometry computation (simplified)
            # In practice, need proper conversion
            sat_pos_eci = sat_pos_ecef  # Simplified
            sat_vel_eci = sat_vel_ecef  # Simplified
            
            # Compute geometry
            geom = user_geom.compute_geometry(sat_pos_eci, sat_vel_eci, current_time)
            
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
                'slant_range': slant_range
            }
            
            # Compute link budget
            link_budget = channel_model.compute_link_budget(
                geometry=geom,
                rain_rate_mmh=config.channel.get('rain_rate_mmh', 0.0),
                bandwidth_hz=config.bandwidth_hz
            )
            link_budgets[user_id] = link_budget
        
        # 3. Get traffic demands for this slot
        slot_demands = traffic.get(t, {})
        
        # 4. Compute QoS (initial, before allocation)
        # Use previous allocations for initial QoS estimate
        initial_allocations = {uid: None for uid in [u['id'] for u in users]}
        qos = qos_estimator.compute_qos(users, link_budgets, initial_allocations, slot_demands)
        
        # 5. Call DSS policy
        if policy_name == "fuzzy":
            allocations = policy.allocate(
                users=users,
                qos=qos,
                context=user_context,
                bandwidth_hz=config.bandwidth_hz,
                alpha=0.7
            )
        elif policy_name == "static":
            # Static policy: equal allocation
            demands_array = np.array([slot_demands.get(u['id'], 0.0) for u in users])
            allocations_array = policy.allocate(
                demands=demands_array,
                available_resources=config.bandwidth_hz * len(users),
                weights=None
            )
            # Convert to allocation format
            allocations = {}
            for i, user in enumerate(users):
                if allocations_array[i] > 0:
                    # Allocate a channel
                    channels = spectrum_env.get_available_channels(config.bandwidth_hz)
                    if channels:
                        freq, sinr = channels[0]
                        allocations[user['id']] = (freq, sinr)
                    else:
                        allocations[user['id']] = None
                else:
                    allocations[user['id']] = None
        elif policy_name == "priority":
            # Priority policy
            demands_array = np.array([slot_demands.get(u['id'], 0.0) for u in users])
            priorities_array = np.array([u.get('priority', 0.5) for u in users])
            allocations_array = policy.allocate(
                demands=demands_array,
                available_resources=config.bandwidth_hz * len(users),
                priorities=priorities_array
            )
            # Convert to allocation format
            allocations = {}
            for i, user in enumerate(users):
                if allocations_array[i] > 0:
                    channels = spectrum_env.get_available_channels(config.bandwidth_hz)
                    if channels:
                        freq, sinr = channels[0]
                        allocations[user['id']] = (freq, sinr)
                    else:
                        allocations[user['id']] = None
                else:
                    allocations[user['id']] = None
        
        # 6. Update spectrum environment
        spectrum_env.update_interference_map()
        
        # 7. Recompute QoS with actual allocations
        qos = qos_estimator.compute_qos(users, link_budgets, allocations, slot_demands)
        
        # 8. Update metrics logger
        metrics_logger.update(slot_idx, users, qos, allocations, user_context)
    
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
    print(f"  Mean Fuzzy Fairness: {summary.get('mean_fuzzy_fairness', 0):.3f}")
    print(f"  Mean α-fairness (α=1): {summary.get('mean_alpha_1', 0):.3f}")
    print(f"  Mean Rate: {summary.get('mean_mean_rate', 0)/1e6:.2f} Mbps")
    print(f"  Cell Edge Rate: {summary.get('mean_cell_edge_rate', 0)/1e6:.2f} Mbps")
    print(f"  Operator Imbalance: {summary.get('mean_operator_imbalance', 0):.3f}")
    
    print(f"\n✓ Results saved to {output_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Fuzzy-Fairness DSS Simulator for LEO Satellites (Phase 4)'
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
        default='fuzzy',
        choices=['static', 'priority', 'fuzzy'],
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
    
    args = parser.parse_args()
    
    # Parse GPU ID
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
    
    run_simulation(
        scenario_name=args.scenario,
        policy_name=args.policy,
        output_dir=args.output,
        use_gpu=use_gpu,
        gpu_id=gpu_id,
        duration_s=args.duration_s
    )


if __name__ == '__main__':
    main()

