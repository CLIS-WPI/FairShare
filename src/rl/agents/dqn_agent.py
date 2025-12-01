"""
DQN (Deep Q-Network) agent for fairness-optimized allocation.
"""

from typing import Dict, Any, Optional
import numpy as np

try:
    from stable_baselines3 import DQN
    from stable_baselines3.common.callbacks import BaseCallback
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

from src.rl.agents.base_agent import BaseRLAgent
from src.rl.environment import FairnessRLEnvironment, ActionType


class DQNAgent(BaseRLAgent):
    """
    DQN agent for fairness-optimized spectrum allocation.
    
    Uses stable-baselines3 DQN implementation.
    Requires discrete action space.
    """
    
    def __init__(
        self,
        env: FairnessRLEnvironment,
        policy: str = "MlpPolicy",
        learning_rate: float = 1e-4,
        buffer_size: int = 100000,
        learning_starts: int = 1000,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: int = 4,
        gradient_steps: int = 1,
        target_update_interval: int = 1000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        **kwargs
    ):
        """
        Initialize DQN agent.
        
        Args:
            env: RL environment (must have discrete action space)
            policy: Policy network type
            learning_rate: Learning rate
            buffer_size: Replay buffer size
            learning_starts: Steps before learning starts
            batch_size: Batch size
            tau: Hard update coefficient (1.0 = hard update)
            gamma: Discount factor
            train_freq: Training frequency
            gradient_steps: Gradient steps per update
            target_update_interval: Target network update interval
            exploration_fraction: Fraction of training for exploration
            exploration_initial_eps: Initial epsilon
            exploration_final_eps: Final epsilon
            **kwargs: Additional DQN parameters
        """
        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3 required for DQN agent")
        
        if env.action_type != ActionType.DISCRETE:
            raise ValueError("DQN requires discrete action space")
        
        self.env = env
        self.agent = DQN(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            target_update_interval=target_update_interval,
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps,
            verbose=1,
            **kwargs
        )
    
    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> np.ndarray:
        """Predict action."""
        return self.agent.predict(observation, deterministic=deterministic)[0]
    
    def train(
        self,
        total_timesteps: int,
        callback: Optional[BaseCallback] = None,
        **kwargs
    ):
        """Train the agent."""
        self.agent.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            **kwargs
        )
    
    def save(self, path: str):
        """Save agent."""
        self.agent.save(path)
    
    def load(self, path: str):
        """Load agent."""
        self.agent = DQN.load(path, env=self.env)

