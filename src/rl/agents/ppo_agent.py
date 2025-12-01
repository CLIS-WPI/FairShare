"""
PPO (Proximal Policy Optimization) agent for fairness-optimized allocation.
"""

from typing import Dict, Any, Optional
import numpy as np

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

from src.rl.agents.base_agent import BaseRLAgent
from src.rl.environment import FairnessRLEnvironment


class FairnessCallback(BaseCallback):
    """Callback to track fairness during training."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.fairness_history = []
        self.efficiency_history = []
    
    def _on_step(self) -> bool:
        """Called at each step."""
        # Extract fairness from info if available
        if 'fairness_scores' in self.locals.get('infos', [{}])[0]:
            fairness = np.mean(self.locals['infos'][0]['fairness_scores'])
            self.fairness_history.append(fairness)
        return True


class PPOAgent(BaseRLAgent):
    """
    PPO agent for fairness-optimized spectrum allocation.
    
    Uses stable-baselines3 PPO implementation.
    """
    
    def __init__(
        self,
        env: FairnessRLEnvironment,
        policy: str = "MlpPolicy",
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        **kwargs
    ):
        """
        Initialize PPO agent.
        
        Args:
            env: RL environment
            policy: Policy network type
            learning_rate: Learning rate
            n_steps: Steps per update
            batch_size: Batch size
            n_epochs: Number of epochs per update
            gamma: Discount factor
            gae_lambda: GAE lambda
            clip_range: PPO clip range
            ent_coef: Entropy coefficient
            vf_coef: Value function coefficient
            **kwargs: Additional PPO parameters
        """
        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3 required for PPO agent")
        
        self.env = env
        self.agent = PPO(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            verbose=1,
            **kwargs
        )
        
        self.callback = FairnessCallback()
    
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
        """
        Train the agent.
        
        Args:
            total_timesteps: Total training timesteps
            callback: Optional callback
            **kwargs: Additional training parameters
        """
        if callback is None:
            callback = self.callback
        
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
        self.agent = PPO.load(path, env=self.env)
    
    def get_fairness_history(self) -> list:
        """Get fairness history from training."""
        return self.callback.fairness_history

