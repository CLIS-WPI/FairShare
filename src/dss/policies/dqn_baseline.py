"""
DQN-based spectrum allocation baseline for comparison with fuzzy-adaptive DSS.

Phase 6: ML baseline for comparison studies.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None

from ..spectrum_environment import SpectrumEnvironment

logger = logging.getLogger(__name__)


class DQNPolicy:
    """
    Deep Q-Network baseline for dynamic spectrum sharing.
    
    State: 7-dimensional vector per user [throughput, latency, outage, priority, 
                                          doppler, elevation, beam_load]
    Action: Discrete spectrum allocation (channel index)
    """
    
    def __init__(
        self,
        spectrum_env: SpectrumEnvironment,
        state_dim: int = 7,
        action_dim: int = 20,  # Number of discrete channels
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        buffer_size: int = 10000,
        batch_size: int = 64,
        model_path: Optional[str] = None
    ):
        """
        Initialize DQN policy.
        
        Args:
            spectrum_env: Spectrum environment instance
            state_dim: State dimension (default: 7 for Phase 3 inputs)
            action_dim: Number of discrete actions (channels)
            learning_rate: Learning rate for Q-network
            gamma: Discount factor
            epsilon: Exploration rate (for training)
            buffer_size: Experience replay buffer size
            batch_size: Training batch size
            model_path: Path to pre-trained model (optional)
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for DQN policy. Install with: pip install tensorflow")
        
        self.spectrum_env = spectrum_env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        
        # Q-network
        self.q_network = self._build_network()
        self.target_network = self._build_network()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        
        # Experience replay buffer
        self.replay_buffer = []
        self.buffer_size = buffer_size
        
        # Load pre-trained model if available
        if model_path:
            self.load(model_path)
        else:
            # Initialize target network with same weights
            self.update_target_network()
        
        logger.info(f"DQN Policy initialized: state_dim={state_dim}, action_dim={action_dim}")
    
    def _build_network(self) -> tf.keras.Model:
        """Build Q-network: state -> Q(s,a) for all actions"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(self.state_dim,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_dim, activation='linear')
        ])
        return model
    
    def _context_to_state(self, context: Dict) -> np.ndarray:
        """
        Convert user context to state vector (normalize to [0,1]).
        
        Args:
            context: User context dictionary with keys:
                - throughput, latency, outage, priority, doppler, elevation, beam_load
        
        Returns:
            Normalized state vector of shape (state_dim,)
        """
        # Extract values with defaults
        state = np.array([
            context.get('throughput', 0.0),
            context.get('latency', 0.0),
            context.get('outage', 0.1),
            context.get('priority', 0.5),
            context.get('doppler', 0.0),
            context.get('elevation', 0.5),
            context.get('beam_load', 0.5)
        ], dtype=np.float32)
        
        # Normalize to [0, 1] (assuming inputs are already normalized)
        return np.clip(state, 0.0, 1.0)
    
    def select_action(self, state: np.ndarray, training: bool = False) -> int:
        """
        Epsilon-greedy action selection.
        
        Args:
            state: State vector
            training: If True, use epsilon-greedy; if False, use greedy
        
        Returns:
            Selected action (channel index)
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_dim)
        
        # Greedy action
        state_tensor = tf.constant(state[np.newaxis, :], dtype=tf.float32)
        q_values = self.q_network(state_tensor, training=False)
        return int(np.argmax(q_values[0].numpy()))
    
    def allocate(
        self,
        users: List[Dict],
        qos: Dict[str, Dict],
        context: Dict[str, Dict],
        bandwidth_hz: float = 100e6,
        **kwargs
    ) -> Dict[str, Optional[Tuple[float, float]]]:
        """
        Allocate spectrum to users using trained DQN policy.
        
        This method matches the interface of FuzzyAdaptivePolicy.allocate()
        
        Args:
            users: List of user dictionaries
            qos: QoS estimates per user (dict: user_id -> qos_dict)
            context: User context per user (dict: user_id -> context_dict)
            bandwidth_hz: Bandwidth to allocate per user
            **kwargs: Additional arguments (ignored)
        
        Returns:
            Dict[user_id -> (frequency_hz, bandwidth_hz) or None]
        """
        allocations = {}
        
        # Convert contexts to state vectors
        user_ids = [user['id'] for user in users]
        states = []
        for user_id in user_ids:
            user_ctx = context.get(user_id, {})
            # Enhance context with QoS if available
            if user_id in qos:
                user_ctx = {**user_ctx, **qos[user_id]}
            state = self._context_to_state(user_ctx)
            states.append(state)
        
        states = np.array(states, dtype=np.float32)
        
        # Get Q-values for all users (batch prediction)
        states_tensor = tf.constant(states, dtype=tf.float32)
        q_values_batch = self.q_network(states_tensor, training=False).numpy()
        
        # Select actions (channels) for each user
        actions = np.argmax(q_values_batch, axis=1)
        
        # Convert actions to frequency allocations
        freq_range = (self.spectrum_env.freq_min, self.spectrum_env.freq_max)
        channel_bw = bandwidth_hz
        n_channels = max(1, int((freq_range[1] - freq_range[0]) / channel_bw))
        n_channels = min(n_channels, self.action_dim)  # Cap at action_dim
        
        # Sort users by Q-value (highest first) for priority allocation
        user_q_values = np.max(q_values_batch, axis=1)
        sorted_indices = np.argsort(-user_q_values)  # Descending order
        
        for idx in sorted_indices:
            user_id = user_ids[idx]
            action = actions[idx]
            
            # Map action to frequency
            channel_idx = action % n_channels
            frequency_hz = freq_range[0] + channel_idx * channel_bw
            
            # Get beam_id from context
            user_ctx = context.get(user_id, {})
            beam_id = user_ctx.get('beam_id', 'beam_0')
            
            # Try to allocate
            try:
                allocation = self.spectrum_env.allocate(
                    user_id=user_id,
                    bandwidth_hz=bandwidth_hz,
                    beam_id=beam_id,
                    preferred_frequency_hz=frequency_hz
                )
                
                if allocation:
                    allocations[user_id] = allocation
            except Exception as e:
                logger.debug(f"Allocation failed for {user_id}: {e}")
                allocations[user_id] = None
        
        logger.debug(f"DQN allocated spectrum to {len([a for a in allocations.values() if a])}/{len(user_ids)} users")
        return allocations
    
    def store_experience(self, state: np.ndarray, action: int, reward: float, 
                        next_state: np.ndarray, done: bool):
        """
        Store experience in replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        if len(self.replay_buffer) >= self.buffer_size:
            self.replay_buffer.pop(0)
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    def train_step(self) -> float:
        """
        Single training step using experience replay.
        
        Returns:
            Training loss
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        # Sample batch
        batch_indices = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in batch_indices]
        
        states = np.array([s for s, _, _, _, _ in batch], dtype=np.float32)
        actions = np.array([a for _, a, _, _, _ in batch], dtype=np.int32)
        rewards = np.array([r for _, _, r, _, _ in batch], dtype=np.float32)
        next_states = np.array([ns for _, _, _, ns, _ in batch], dtype=np.float32)
        dones = np.array([d for _, _, _, _, d in batch], dtype=np.float32)
        
        # Compute target Q-values
        next_q_values = self.target_network(next_states, training=False)
        # Convert to numpy if tensor
        if hasattr(next_q_values, 'numpy'):
            next_q_max = np.max(next_q_values.numpy(), axis=1)
        else:
            next_q_max = np.max(next_q_values, axis=1)
        targets = rewards + self.gamma * next_q_max * (1 - dones)
        
        # Update Q-network
        with tf.GradientTape() as tape:
            q_values = self.q_network(states, training=True)
            action_masks = tf.one_hot(actions, self.action_dim)
            q_action = tf.reduce_sum(q_values * action_masks, axis=1)
            loss = tf.reduce_mean(tf.square(targets - q_action))
        
        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))
        
        return float(loss.numpy())
    
    def update_target_network(self):
        """Copy weights from Q-network to target network"""
        self.target_network.set_weights(self.q_network.get_weights())
    
    def save(self, path: str):
        """Save trained model"""
        self.q_network.save(path)
        logger.info(f"DQN model saved to {path}")
    
    def load(self, path: str):
        """Load trained model"""
        self.q_network = tf.keras.models.load_model(path)
        self.update_target_network()
        logger.info(f"DQN model loaded from {path}")

