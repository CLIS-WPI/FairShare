"""
Train DQN baseline for spectrum allocation.

Phase 6: ML baseline training script.

Usage:
    python scripts/train_dqn_baseline.py --scenario urban_congestion_phase4 --episodes 10000
"""

import argparse
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
import json
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.dss.policies.dqn_baseline import DQNPolicy
from src.dss.spectrum_environment import SpectrumEnvironment
from src.experiments import load_scenario, TrafficGenerator
from src.channel import OrbitPropagator, SatelliteGeometry, ChannelModel
from src.experiments.qos_estimator import QoSEstimator
from src.fairness.metrics import jain_fairness_index

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compute_reward(user_contexts: dict, allocations: dict, jain_index: float) -> float:
    """
    Compute reward for DQN training.
    
    Reward = weighted sum of:
    - Jain fairness index
    - Number of successful allocations
    - Average throughput
    
    Args:
        user_contexts: Dictionary of user contexts
        allocations: Dictionary of allocations (user_id -> allocation or None)
        jain_index: Jain fairness index
    
    Returns:
        Reward value
    """
    n_allocated = len([a for a in allocations.values() if a is not None])
    n_total = len(user_contexts)
    allocation_ratio = n_allocated / max(n_total, 1)
    
    # Compute average throughput for allocated users
    avg_throughput = 0.0
    if allocations:
        throughputs = []
        for uid, alloc in allocations.items():
            if alloc is not None and uid in user_contexts:
                # Get normalized throughput from context
                ctx = user_contexts[uid]
                throughput_norm = ctx.get('throughput', 0.0)
                throughputs.append(throughput_norm)
        
        if throughputs:
            avg_throughput = np.mean(throughputs)
    
    # Combined reward
    reward = 0.5 * jain_index + 0.3 * allocation_ratio + 0.2 * avg_throughput
    
    return reward


def train_dqn(args):
    """Main DQN training loop"""
    logger.info("=== DQN Training ===")
    
    # Load scenario
    try:
        config = load_scenario(args.scenario)
    except FileNotFoundError:
        # Try as file path
        from src.experiments.scenario_loader import ScenarioConfig
        config = ScenarioConfig(args.scenario)
    
    logger.info(f"Loaded scenario: {config.scenario_name}")
    
    # Get frequency range from config
    freq_range = config.frequency_range_hz
    if isinstance(freq_range, tuple) and len(freq_range) == 2:
        freq_min, freq_max = float(freq_range[0]), float(freq_range[1])
    else:
        freq_min, freq_max = 10e9, 12e9  # Default
    
    # Initialize environment
    env = SpectrumEnvironment(
        frequency_range_hz=(freq_min, freq_max),
        frequency_resolution_hz=1e6
    )
    
    # Initialize DQN policy
    policy = DQNPolicy(
        spectrum_env=env,
        state_dim=7,
        action_dim=args.action_dim,
        learning_rate=args.lr,
        epsilon=args.epsilon,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size
    )
    
    logger.info(f"DQN Policy initialized: action_dim={args.action_dim}")
    
    # Initialize components
    orbit_prop = None
    if config.tle_file:
        try:
            orbit_prop = OrbitPropagator(config.tle_file)
        except Exception as e:
            logger.warning(f"Could not load TLE file: {e}")
    
    channel_model = ChannelModel(frequency_hz=config.carrier_frequency_hz)
    qos_estimator = QoSEstimator()
    
    # Training loop
    episode_rewards = []
    episode_losses = []
    
    for episode in range(args.episodes):
        # Generate traffic for this episode
        traffic_gen = TrafficGenerator(config, seed=args.seed + episode)
        traffic_data = traffic_gen.generate()
        users = traffic_data['users']
        traffic = traffic_data['traffic']
        
        # Reset environment
        env = SpectrumEnvironment(
            frequency_range_hz=(freq_min, freq_max),
            frequency_resolution_hz=1e6
        )
        policy.spectrum_env = env  # Update policy's environment reference
        
        episode_reward = 0.0
        episode_loss = 0.0
        n_steps = 0
        
        # Get time slots
        timestamps = sorted(traffic.keys())[:args.max_steps_per_episode]
        
        # Simulate time slots for this episode
        for slot_idx, t in enumerate(timestamps):
            # Build user contexts (simplified for training speed)
            user_contexts = {}
            qos = {}
            
            for user in users:
                user_id = user['id']
                
                # Mock context (in full version, compute from channel model)
                # For training, use random but realistic values
                context = {
                    'throughput': np.random.uniform(0.3, 1.0),
                    'latency': np.random.uniform(0.0, 0.5),
                    'outage': np.random.uniform(0.0, 0.3),
                    'priority': user.get('priority', 0.5),
                    'doppler': np.random.uniform(0.0, 0.5),
                    'elevation': np.random.uniform(0.3, 1.0),
                    'beam_load': np.random.uniform(0.2, 0.8),
                    'beam_id': f'beam_{user.get("operator", 0)}'
                }
                user_contexts[user_id] = context
                
                # Mock QoS
                qos[user_id] = {
                    'throughput': context['throughput'] * 100e6,  # Convert to bps
                    'latency': context['latency'],
                    'outage': context['outage']
                }
            
            # Convert contexts to states
            user_ids = list(user_contexts.keys())
            states = np.array([policy._context_to_state(user_contexts[uid]) for uid in user_ids])
            
            # Select actions and allocate
            allocations = policy.allocate(
                users=users,
                qos=qos,
                context=user_contexts,
                bandwidth_hz=config.bandwidth_hz
            )
            
            # Compute Jain index for reward
            if allocations:
                throughputs = []
                for uid in allocations.keys():
                    if allocations[uid] is not None and uid in user_contexts:
                        throughputs.append(user_contexts[uid]['throughput'])
                
                if throughputs:
                    jain_index = jain_fairness_index(np.array(throughputs))
                else:
                    jain_index = 0.0
            else:
                jain_index = 0.0
            
            # Compute reward
            reward = compute_reward(user_contexts, allocations, jain_index)
            episode_reward += reward
            
            # Store experiences and train (simplified - sample subset for speed)
            sample_size = min(10, len(user_ids))
            sampled_indices = np.random.choice(len(user_ids), sample_size, replace=False)
            
            for idx in sampled_indices:
                user_id = user_ids[idx]
                state = states[idx]
                # Get action that was selected (simplified - use random for now)
                action = np.random.randint(0, policy.action_dim)
                next_state = state  # Simplified (same state)
                done = (slot_idx == len(timestamps) - 1)
                
                policy.store_experience(state, action, reward, next_state, done)
            
            # Train
            if len(policy.replay_buffer) >= policy.batch_size:
                loss = policy.train_step()
                episode_loss += loss
                n_steps += 1
        
        # Update target network every N episodes
        if episode % args.target_update_freq == 0:
            policy.update_target_network()
        
        # Logging
        avg_loss = episode_loss / max(n_steps, 1)
        episode_rewards.append(episode_reward)
        episode_losses.append(avg_loss)
        
        if episode % args.log_freq == 0:
            avg_reward_100 = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            logger.info(f"Episode {episode}/{args.episodes} | Reward: {episode_reward:.3f} | Avg(100): {avg_reward_100:.3f} | Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if episode % args.save_freq == 0 and episode > 0:
            checkpoint_path = Path(args.output_dir) / f"dqn_checkpoint_ep{episode}.h5"
            policy.save(str(checkpoint_path))
            logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    # Save final model
    final_path = Path(args.output_dir) / "dqn_baseline_final.h5"
    policy.save(str(final_path))
    
    # Save training history
    history = {
        'episode_rewards': [float(r) for r in episode_rewards],
        'episode_losses': [float(l) for l in episode_losses],
        'config': vars(args),
        'final_episode': args.episodes
    }
    history_path = Path(args.output_dir) / "dqn_training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info(f"✓ Training complete. Model saved to {final_path}")
    logger.info(f"✓ Training history saved to {history_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN baseline for spectrum allocation")
    parser.add_argument("--scenario", type=str, default="urban_congestion_phase4", 
                       help="Scenario name or path to YAML file")
    parser.add_argument("--episodes", type=int, default=10000, 
                       help="Number of training episodes")
    parser.add_argument("--max-steps-per-episode", type=int, default=50, 
                       help="Max steps per episode")
    parser.add_argument("--action-dim", type=int, default=20, 
                       help="Number of discrete actions")
    parser.add_argument("--lr", type=float, default=0.001, 
                       help="Learning rate")
    parser.add_argument("--epsilon", type=float, default=0.1, 
                       help="Exploration rate")
    parser.add_argument("--buffer-size", type=int, default=10000, 
                       help="Replay buffer size")
    parser.add_argument("--batch-size", type=int, default=64, 
                       help="Batch size")
    parser.add_argument("--target-update-freq", type=int, default=100, 
                       help="Target network update frequency")
    parser.add_argument("--log-freq", type=int, default=100, 
                       help="Logging frequency")
    parser.add_argument("--save-freq", type=int, default=1000, 
                       help="Model save frequency")
    parser.add_argument("--output-dir", type=str, default="models/dqn", 
                       help="Output directory")
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Train
    train_dqn(args)

