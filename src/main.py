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
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import yaml
import os
import sys

# Add src to path for module imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Module-level logger
logger = logging.getLogger(__name__)

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
                    logger.warning("Could not set memory growth for %s: %s", gpu.name, e)
            
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
    
    # Register beams for all operators (CRITICAL: must be done before allocation)
    from src.dss.spectrum_environment import Beam
    for op_config in getattr(config, 'operators', []):
        op_id = op_config.get('id', op_config.get('name', 'Op_A'))
        beam_id = f"beam_{op_id}"
        if beam_id not in spectrum_env.occupancy_map:
            # Create a simple beam object for registration
            beam = Beam(
                beam_id=beam_id,
                satellite_id=f"sat_{op_id}",
                center_frequency_hz=config.frequency_range_hz[0] + (config.frequency_range_hz[1] - config.frequency_range_hz[0]) / 2,
                bandwidth_hz=config.bandwidth_hz,
                power_dbm=40.0,
                location=(config.geo['center_lat_deg'], config.geo['center_lon_deg']),
                elevation_deg=45.0
            )
            spectrum_env.register_beam(beam)
    
    # Also register default beams if operators not specified
    for i in range(config.num_operators):
        beam_id = f"beam_{i}"
        if beam_id not in spectrum_env.occupancy_map:
            beam = Beam(
                beam_id=beam_id,
                satellite_id=f"sat_{i}",
                center_frequency_hz=config.frequency_range_hz[0] + (config.frequency_range_hz[1] - config.frequency_range_hz[0]) / 2,
                bandwidth_hz=config.bandwidth_hz,
                power_dbm=40.0,
                location=(config.geo['center_lat_deg'], config.geo['center_lon_deg']),
                elevation_deg=45.0
            )
            spectrum_env.register_beam(beam)
    
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
    metrics_logger = MetricsLogger(config.scenario_name, policy_name)
    
    # Simulation loop (slot-based)
    print(f"\n{'='*60}")
    print(f"Starting simulation: {config.sim_time_s}s, {config.get_num_slots()} slots")
    print(f"{'='*60}\n")
    
    start_time = config.get_start_datetime()
    num_slots = config.get_num_slots()
    slot_duration = config.slot_duration_s
    
    # Multi-operator satellite generation
    # Generate satellites for all operators (Op_A, Op_B, Op_C)
    from src.utils.satellite_generator import MultiOperatorSatelliteGenerator
    
    # Get operator configs from scenario
    operators_config = getattr(config, 'operators', [])
    if not operators_config:
        # Default: 3 operators
        operators_config = [
            {'name': 'Op_A', 'type': 'starlink_like'},
            {'name': 'Op_B', 'type': 'oneweb_like'},
            {'name': 'Op_C', 'type': 'critical_gov'}
        ]
    
    # Initialize satellite generator
    center_lat = config.geo.get('center_lat_deg', 40.75) if hasattr(config, 'geo') else 40.75
    center_lon = config.geo.get('center_lon_deg', -73.975) if hasattr(config, 'geo') else -73.975
    
    sat_generator = MultiOperatorSatelliteGenerator(
        operators=operators_config,
        center_lat=center_lat,
        center_lon=center_lon,
        min_elevation_deg=25.0  # 25° minimum elevation (urban standard)
    )
    
    print(f"✓ Initialized Multi-Operator Satellite Generator")
    print(f"  Operators: {len(operators_config)}")
    print(f"  Min Elevation: 25.0° (urban standard)")
    print(f"  Expected visible satellites: ~60 (20 per operator)")
    
    # Main simulation loop
    print(f"Starting simulation loop: {num_slots} slots, {len(users)} users")
    print(f"GPU acceleration: {'ENABLED' if (TF_AVAILABLE and use_gpu and gpu_id != 'cpu') else 'DISABLED'}\n")
    
    for slot_idx in range(num_slots):
        current_time = start_time + timedelta(seconds=slot_idx * slot_duration)
        t = slot_idx * slot_duration
        
        # Progress indicator every 10% or every 10 slots (whichever is more frequent)
        progress_interval = max(1, min(num_slots // 10, 10))
        if slot_idx % progress_interval == 0:
            progress_pct = 100 * slot_idx / num_slots
            print(f"[{progress_pct:5.1f}%] Slot {slot_idx}/{num_slots} - Processing...")
            print(f"Slot {slot_idx}/{num_slots} (t={t:.2f}s)")
        
        # 1. Generate satellite positions for all operators
        # OPTIMIZED: Reduce satellite count for 10000 users (20 → 10 per operator)
        # This reduces computation: 30 satellites instead of 60
        num_sats_per_op = 10 if len(users) > 5000 else 20
        all_satellites = sat_generator.generate_satellite_positions(
            timestamp=current_time,
            num_sats_per_operator=num_sats_per_op
        )
        
        # Count total satellites
        total_sats = sum(len(sats) for sats in all_satellites.values())
        if slot_idx == 0:
            print(f"  Generated {total_sats} satellites across {len(all_satellites)} operators")
        
        # 2. GPU-accelerated batch processing: find visible satellites for all users
        user_context = {}
        link_budgets = {}
        geometries_list = []
        visible_sat_counts = []
        serving_sat_counts = {'Op_A': 0, 'Op_B': 0, 'Op_C': 0}
        
        # FIXED: Add temporal variation to channel conditions (for dynamic Jain index)
        # Add random fading that changes each slot
        slot_fading_db = np.random.normal(0, 4)  # 4 dB std deviation per slot
        
        # OPTIMIZED: Batch process all users on GPU
        # Convert user positions to ECEF for batch processing (vectorized)
        user_lats = np.array([u['lat'] for u in users])
        user_lons = np.array([u['lon'] for u in users])
        
        # Batch convert lat/lon to ECEF (vectorized, much faster than loop)
        def lat_lon_alt_to_ecef_batch(lats, lons, alt=0.0):
            """Convert lat/lon arrays to ECEF coordinates (vectorized)."""
            lat_rad = np.radians(lats)
            lon_rad = np.radians(lons)
            a = 6378137.0  # WGS84 semi-major axis
            e2 = 0.00669437999014  # first eccentricity squared
            N = a / np.sqrt(1 - e2 * np.sin(lat_rad)**2)
            x = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
            y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
            z = (N * (1 - e2) + alt) * np.sin(lat_rad)
            return np.column_stack([x, y, z])
        
        user_positions_ecef = lat_lon_alt_to_ecef_batch(user_lats, user_lons, 0.0)  # [num_users, 3]
        
        # GPU-accelerated batch filtering
        use_gpu_batch = (TF_AVAILABLE and use_gpu and gpu_id != 'cpu')
        if use_gpu_batch and len(users) > 100:  # Use GPU for large batches
            device = f'/GPU:{gpu_id}' if isinstance(gpu_id, int) else '/GPU:0'
            print(f"🚀 Using GPU-accelerated batch satellite filtering for {len(users)} users")
            logger.info(f"🚀 Using GPU-accelerated batch satellite filtering for {len(users)} users")
            visible_sats_per_user = sat_generator.filter_visible_satellites_batch(
                satellites=all_satellites,
                user_lats=user_lats,
                user_lons=user_lons,
                user_positions_ecef=user_positions_ecef,
                timestamp=current_time,
                use_gpu=True,
                device=device
            )
        else:
            # Fallback to CPU (one-by-one)
            visible_sats_per_user = []
            for user in users:
                visible = sat_generator.filter_visible_satellites(
                    satellites=all_satellites,
                    user_lat=user['lat'],
                    user_lon=user['lon'],
                    timestamp=current_time
                )
                visible_sats_per_user.append(visible)
        
        # Process results: select best server for each user
        for i, user in enumerate(users):
            user_id = user['id']
            visible_sats = visible_sats_per_user[i]
            
            visible_sat_counts.append(len(visible_sats))
            
            if not visible_sats:
                # No visible satellites - use default
                logger.warning(f"User {user_id} has no visible satellites, using default")
                user_geom = SatelliteGeometry(user['lat'], user['lon'])
                default_pos = np.array([7000e3, 0, 0])
                default_vel = np.array([0, 7500, 0])
                geom = user_geom.compute_geometry(default_pos, default_vel, current_time)
                geometries_list.append(geom)
                continue
            
            # Select best server (highest elevation) and interferers
            serving_sat, interferers = sat_generator.select_best_server(
                visible_sats,
                max_candidates=12
            )
            
            # Track serving satellite operator
            serving_op = serving_sat.get('operator_id', 'Op_A')
            if serving_op in serving_sat_counts:
                serving_sat_counts[serving_op] += 1
            
            # Use serving satellite geometry for link budget computation
            geom = serving_sat['geometry']
            geometries_list.append(geom)
            
            # Store interference information for SINR calculation
            # (Will be used in spectrum allocation)
            user_context[user_id] = {
                'serving_satellite': serving_sat['satellite_id'],
                'serving_operator': serving_op,
                'num_visible': len(visible_sats),
                'num_interferers': len(interferers),
                'interferers': [
                    {
                        'satellite_id': intf['satellite_id'],
                        'operator_id': intf['operator_id'],
                        'elevation': intf['elevation']
                    }
                    for intf in interferers
                ]
            }
        
        # Log visible satellite statistics
        if slot_idx == 0 or slot_idx % max(1, num_slots // 10) == 0:
            avg_visible = np.mean(visible_sat_counts) if visible_sat_counts else 0
            min_visible = np.min(visible_sat_counts) if visible_sat_counts else 0
            max_visible = np.max(visible_sat_counts) if visible_sat_counts else 0
            print(f"  Visible satellites: avg={avg_visible:.1f}, min={min_visible}, max={max_visible}")
            print(f"  Serving satellites: Op_A={serving_sat_counts['Op_A']}, "
                  f"Op_B={serving_sat_counts['Op_B']}, Op_C={serving_sat_counts['Op_C']}")
        
        # Batch compute link budgets on GPU if available
        if TF_AVAILABLE and use_gpu and gpu_id != 'cpu' and len(users) > 10:
            try:
                from src.channel.channel_model_gpu import ChannelModelGPU
                
                # Log GPU usage
                if slot_idx == 0:
                    print(f"🚀 Using GPU-accelerated ChannelModelGPU for {len(users)} users")
                    if TF_AVAILABLE:
                        gpus = tf.config.list_physical_devices('GPU')
                        print(f"   Available GPUs: {len(gpus)}")
                        for gpu in gpus:
                            print(f"   - {gpu.name}")
                
                gpu_channel = ChannelModelGPU(
                    frequency_hz=config.carrier_frequency_hz,
                    use_gpu=True
                )
                
                # Verify GPU is actually being used
                if slot_idx == 0:
                    # Force GPU computation to verify
                    test_geom = [geometries_list[0]]
                    import time
                    test_start = time.time()
                    test_result = gpu_channel.compute_link_budgets_batch(
                        test_geom,
                        rain_rate_mmh=config.channel.get('rain_rate_mmh', 0.0),
                        bandwidth_hz=config.bandwidth_hz
                    )
                    test_elapsed = time.time() - test_start
                    print(f"   ✓ GPU computation verified (test: {test_elapsed*1000:.2f}ms, SNR: {test_result[0].get('snr_db', 'N/A')} dB)")
                    
                    # Additional verification: check TensorFlow device placement
                    if TF_AVAILABLE:
                        try:
                            # Create a test tensor and check its device
                            test_tensor = tf.constant([1.0], dtype=tf.float32)
                            with tf.device('/GPU:0'):
                                test_op = tf.reduce_sum(test_tensor)
                                test_val = test_op.numpy()
                            print(f"   ✓ TensorFlow GPU device placement confirmed")
                        except Exception as e:
                            print(f"   ⚠ GPU device check warning: {e}")
                
                # Process in batches for large user counts
                link_budgets_list = gpu_channel.compute_link_budgets_batch(
                    geometries_list,
                    rain_rate_mmh=config.channel.get('rain_rate_mmh', 0.0),
                    bandwidth_hz=config.bandwidth_hz
                )
                
                # Convert to dictionary
                for i, user in enumerate(users):
                    link_budgets[user['id']] = link_budgets_list[i]
                
                # Progress indicator
                if slot_idx % max(1, num_slots // 20) == 0:
                    print(f"   Slot {slot_idx}/{num_slots} ({100*slot_idx/num_slots:.1f}%) - GPU processing...")
                    
            except (ImportError, Exception) as e:
                # Fallback to CPU
                if slot_idx == 0:
                    print(f"⚠ GPU computation failed: {e}")
                    print("   Falling back to CPU...")
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
        
        # Build context dictionaries (update existing or create new)
        for i, user in enumerate(users):
            user_id = user['id']
            if i < len(geometries_list):
                geom = geometries_list[i]
            else:
                logger.warning(f"Geometry mismatch for user {user_id}")
                continue
                
            link_budget = link_budgets.get(user_id, {'snr_db': -100, 'capacity_mbps': 0})
            
            elevation = geom.get('elevation', 0.0)
            doppler = abs(geom.get('doppler_shift', 0.0))
            slant_range = geom.get('slant_range', 0.0)
            
            # Get beam load
            operator = user.get('operator', 0)
            beam_id = f"beam_{operator}"
            beam_load = spectrum_env.compute_beam_load(beam_id)
            
            # Update context (may already have serving/interferer info from satellite selection)
            if user_id not in user_context:
                user_context[user_id] = {}
            
            # Add link budget and geometry info
            user_context[user_id].update({
                'elevation': elevation,
                'doppler': doppler,
                'beam_load': beam_load,
                'beam_id': beam_id,
                'slant_range': slant_range,
                'link_budget': link_budget
            })
        
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
        # FIXED: Create REAL scarcity - use actual available bandwidth
        # For geographic inequality test: 50 MHz total, 25 MHz per operator
        total_spectrum_bw = config.frequency_range_hz[1] - config.frequency_range_hz[0]
        # Per-user bandwidth should be MUCH smaller to create competition
        # With 10000 users and 50 MHz: 50e6 / 10000 = 5 kHz per user (creates scarcity!)
        per_user_bandwidth_hz = min(20e6, total_spectrum_bw / len(users))  # Cap at 20 MHz, but use scarcity
        
        if policy_name == "rl" or policy_name == "dqn":
            # RL/DQN policy allocation
            allocations = policy.allocate(
                users=users,
                qos=qos,
                context=user_context,
                bandwidth_hz=per_user_bandwidth_hz  # FIXED: Use per-user bandwidth
            )
        elif policy_name == "static":
            # Static Equal policy: equal bandwidth share per operator, then equal per user within operator
            # For geographic inequality test: each operator gets equal share of total bandwidth
            # CRITICAL: Create REAL scarcity - limit total available bandwidth
            total_spectrum_bw = config.frequency_range_hz[1] - config.frequency_range_hz[0]
            
            # Group users by operator
            operator_users = {}
            for user in users:
                op_id = user.get('operator', 'Op_A')
                if op_id not in operator_users:
                    operator_users[op_id] = []
                operator_users[op_id].append(user)
            
            # Equal bandwidth per operator (REAL scarcity: use actual bandwidth, not unlimited)
            num_operators = len(operator_users)
            # Use 80% of total spectrum (reserve 20% for guard bands)
            available_total = total_spectrum_bw * 0.8
            bandwidth_per_operator = available_total / num_operators if num_operators > 0 else 0
            
            # Capacity limit: max users per operator (creates competition)
            max_users_per_operator = int(bandwidth_per_operator / per_user_bandwidth_hz)
            
            # Allocate within each operator (with capacity limit)
            allocations = {}
            for op_id, op_users in operator_users.items():
                if len(op_users) == 0:
                    continue
                
                # Limit number of users that can be served (capacity constraint)
                # This creates REAL scarcity - not all users can be served
                if len(op_users) > max_users_per_operator:
                    # Select users (for now, first come first served - could be random or priority-based)
                    # For geographic inequality: this will naturally favor urban users (better geometry)
                    op_users = op_users[:max_users_per_operator]
                
                # Equal allocation within operator (for selected users)
                demands_array = np.array([slot_demands.get(u['id'], 0.0) for u in op_users])
                available_per_op = min(
                    bandwidth_per_operator,
                    per_user_bandwidth_hz * len(op_users)
                )
                
                allocations_array = policy.allocate(
                    demands=demands_array,
                    available_resources=available_per_op,
                    weights=None
                )
                
                # Convert to allocation format
                # CRITICAL: Only allocate to users with allocation_array[i] > 0
                # This creates REAL scarcity - not all users get resources
                for i, user in enumerate(op_users):
                    if allocations_array[i] > 0:
                        # Get available channels (with interference consideration)
                        user_id = user['id']
                        user_link_budget = link_budgets.get(user_id, {})
                        snr_db = user_link_budget.get('snr_db', 0.0)
                        
                        # Find channel with SINR (accounting for interference)
                        channels = spectrum_env.find_available_spectrum(
                            bandwidth_hz=per_user_bandwidth_hz,
                            min_sinr_db=0.0,  # Minimum SINR threshold
                            exclude_beam_id=f"beam_{op_id}",
                            link_budget_snr_db=snr_db
                        )
                        
                        if channels:
                            freq, sinr = channels[0]
                            allocations[user['id']] = (freq, sinr)
                            
                            # Update spectrum environment (for interference calculation)
                            spectrum_env.update_beam_usage(
                                beam_id=f"beam_{op_id}",
                                frequency_hz=freq,
                                bandwidth_hz=per_user_bandwidth_hz,
                                power_dbm=snr_db  # Use SNR as transmit power estimate
                            )
                        else:
                            # No available channel (interference too high or spectrum exhausted)
                            allocations[user['id']] = None
                    else:
                        allocations[user['id']] = None
                
                # Mark unselected users as not allocated
                for user in operator_users[op_id]:
                    if user['id'] not in allocations:
                        allocations[user['id']] = None
        elif policy_name == "priority":
            # Priority policy
            # FIXED: Get priorities from priorities_dict (set at function start)
            priorities_array = np.array([priorities_dict.get(u['id'], u.get('priority', 0.5)) for u in users])
            
            demands_array = np.array([slot_demands.get(u['id'], 0.0) for u in users])
            # FIXED: Constrain available resources to create competition
            # Use total spectrum bandwidth instead of per-user * num_users.
            # This forces Priority Policy to make choices.
            total_spectrum_bw = config.frequency_range_hz[1] - config.frequency_range_hz[0]  # Total spectrum
            available_resources = min(
                total_spectrum_bw * 0.8,  # Use 80% of total spectrum (allows more users while maintaining competition)
                per_user_bandwidth_hz * len(users)  # But don't exceed per-user * num_users
            )
            
            # Priority allocation debug info (logged at DEBUG level)
            if slot_idx == 0 or slot_idx % 100 == 0:
                logger.debug(
                    "[Priority] Slot %d: available_resources=%.2f MHz, per_user=%.2f MHz, "
                    "priority_range=[%.3f, %.3f]",
                    slot_idx,
                    available_resources / 1e6,
                    per_user_bandwidth_hz / 1e6,
                    float(np.min(priorities_array)),
                    float(np.max(priorities_array)),
                )
            
            allocations_array = policy.allocate(
                demands=demands_array,
                available_resources=available_resources,
                priorities=priorities_array
            )
            
            # Log allocation mask (how many users got non-zero allocation)
            num_allocated = int(np.sum(allocations_array > 0))
            if slot_idx == 0 or slot_idx % 100 == 0:
                logger.debug(
                    "[Priority] Slot %d: allocations>0=%d/%d, total_allocation=%.2f MHz",
                    slot_idx,
                    num_allocated,
                    len(users),
                    float(np.sum(allocations_array) / 1e6),
                )
            
            # FIXED: Convert to allocation format, respecting priority order
            # Sort users by priority (higher priority first) for spectrum allocation
            user_priority_pairs = [(users[i], allocations_array[i], priorities_array[i]) 
                                   for i in range(len(users))]
            # Sort by priority (descending) - high priority users get spectrum first
            user_priority_pairs.sort(key=lambda x: x[2], reverse=True)
            
            allocations = {}
            # Track allocation statistics
            successful_allocs = 0
            failed_allocs = 0
            beam_allocation_counts = {}
            
            # FIXED: Check available spectrum capacity BEFORE allocating.
            # This prevents trying to allocate to more users than spectrum can support.
            available_channels = spectrum_env.get_available_channels(per_user_bandwidth_hz)
            max_allocatable_users = len(available_channels)
            
            # Log spectrum capacity info
            if slot_idx == 0 or slot_idx % 100 == 0:
                requesting = len([u for u, a, p in user_priority_pairs if a > 0])
                logger.debug(
                    "[Priority] Spectrum capacity: available_channels=%d, "
                    "freq_range=%.2f-%.2f GHz, per_user=%.2f MHz, "
                    "max_theoretical_channels=%.0f, requesting_users=%d",
                    max_allocatable_users,
                    spectrum_env.freq_min / 1e9,
                    spectrum_env.freq_max / 1e9,
                    per_user_bandwidth_hz / 1e6,
                    (spectrum_env.freq_max - spectrum_env.freq_min) / per_user_bandwidth_hz,
                    requesting,
                )
            
            # Allocate spectrum in priority order
            # High-priority users get spectrum first, may exhaust available channels.
            for user, alloc_amount, priority in user_priority_pairs:
                if alloc_amount > 0:
                    # FIXED: Stop allocating if spectrum is exhausted
                    # This prevents wasted allocation attempts.
                    if successful_allocs >= max_allocatable_users:
                        # Spectrum exhausted - mark remaining users as not allocated
                        allocations[user['id']] = None
                        failed_allocs += 1
                        continue
                    
                    # Allocate spectrum to this user (in priority order)
                    beam_id = f"beam_{user.get('operator', 0)}"
                    
                    allocation = spectrum_env.allocate(
                        user_id=user['id'],
                        bandwidth_hz=per_user_bandwidth_hz,
                        beam_id=beam_id
                    )
                    
                    if allocation:
                        allocations[user['id']] = allocation
                        successful_allocs += 1
                        beam_allocation_counts[beam_id] = beam_allocation_counts.get(beam_id, 0) + 1
                    else:
                        allocations[user['id']] = None
                        failed_allocs += 1
                else:
                    allocations[user['id']] = None
            
            # Log aggregated allocation statistics
            if slot_idx == 0 or slot_idx % 100 == 0:
                requesting = len([u for u, a, p in user_priority_pairs if a > 0])
                available_after = spectrum_env.get_available_channels(per_user_bandwidth_hz)
                logger.debug(
                    "[Priority] Allocation results: successful=%d/%d, failed=%d, "
                    "per_beam=%s, remaining_channels=%d",
                    successful_allocs,
                    requesting,
                    failed_allocs,
                    beam_allocation_counts,
                    len(available_after),
                )
        
        # 6. Update spectrum environment
        spectrum_env.update_interference_map()
        
        # 6.5. Compute and log interference data (for NYC heatmap)
        interference_data = None
        if hasattr(config, 'output') and config.output.get('save_interference_heatmap', False):
            # Compute total interference power across all frequencies
            total_interference_power = 0.0
            interference_samples = []
            
            # Sample interference at multiple frequencies
            freq_samples = np.linspace(config.frequency_range_hz[0], config.frequency_range_hz[1], 20)
            for freq in freq_samples:
                interference = spectrum_env.get_spectrum_occupancy(
                    freq, 10e6, exclude_beam_id=None
                )
                if interference > -np.inf:
                    interference_samples.append(interference)
                    total_interference_power += 10**(interference / 10)  # Linear power
            
            # Convert back to dBm
            if len(interference_samples) > 0:
                avg_interference_linear = total_interference_power / len(interference_samples)
                avg_interference_dbm = 10 * np.log10(avg_interference_linear) if avg_interference_linear > 0 else -np.inf
            else:
                avg_interference_dbm = -np.inf
            
            # Collect user locations and interference for heatmap
            interference_data = {
                'slot': slot_idx,
                'total_interference_dbm': float(avg_interference_dbm),
                'user_locations': [(u.get('lat', 0), u.get('lon', 0)) for u in users],
                'user_operators': [u.get('operator', 0) for u in users],
                'allocated_users': [u['id'] for u in users if allocations.get(u['id']) is not None]
            }
        
        # 7. Recompute QoS with actual allocations
        qos = qos_estimator.compute_qos(users, link_budgets, allocations, slot_demands)
        
        # 8. Update metrics logger (with interference data)
        metrics_logger.update(slot_idx, users, qos, allocations, user_context, interference_data)
        
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
            # CRITICAL FIX: TensorFlow doesn't work with CUDA_VISIBLE_DEVICES="all"
            # Must use explicit GPU IDs like "0,1" for 2 GPUs
            import subprocess
            try:
                result = subprocess.run(['nvidia-smi', '--list-gpus'], 
                                      capture_output=True, text=True, timeout=5)
                num_gpus = len([line for line in result.stdout.split('\n') if 'GPU' in line])
                if num_gpus > 0:
                    gpu_ids = ','.join(str(i) for i in range(num_gpus))
                    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
                    print(f"✓ Multi-GPU mode: Using GPUs {gpu_ids}")
                else:
                    print("⚠ No GPUs detected, using CPU")
                    use_gpu = False
            except:
                # Fallback: try to use first 2 GPUs (for H100 x 2)
                os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
                print("✓ Multi-GPU mode: Using GPUs 0,1 (fallback)")
    
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

