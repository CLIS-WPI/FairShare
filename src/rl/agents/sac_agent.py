"""
SAC (Soft Actor-Critic) agent for fairness-optimized allocation.
"""

from typing import Dict, Any, Optional
import numpy as np

try:
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

from src.rl.agents.base_agent import BaseRLAgent
from src.rl.environment import FairnessRLEnvironment


class SACAgent(BaseRLAgent):
    """
    SAC agent for fairness-optimized spectrum allocation.
    
    Uses stable-baselines3 SAC implementation.
    Good for continuous action spaces.
    """
    
    def __init__(
        self,
        env: FairnessRLEnvironment,
        policy: str = "MlpPolicy",
        learning_rate: float = 3e-4,
        buffer_size: int = 100000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: int = 1,
        gradient_steps: int = 1,
        **kwargs
    ):
        """
        Initialize SAC agent.
        
        Args:
            env: RL environment
            policy: Policy network type
            learning_rate: Learning rate
            buffer_size: Replay buffer size
            learning_starts: Steps before learning starts
            batch_size: Batch size
            tau: Soft update coefficient
            gamma: Discount factor
            train_freq: Training frequency
            gradient_steps: Gradient steps per update
            **kwargs: Additional SAC parameters
        """
        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3 required for SAC agent")
        
        self.env = env
        self.agent = SAC(
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
        self.agent = SAC.load(path, env=self.env)

