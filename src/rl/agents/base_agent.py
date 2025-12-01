"""
Base RL agent interface for fairness-optimized allocation.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np


class BaseRLAgent(ABC):
    """Base class for RL agents."""
    
    @abstractmethod
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """
        Predict action from observation.
        
        Args:
            observation: Current state observation
            deterministic: Whether to use deterministic policy
            
        Returns:
            Action array
        """
        pass
    
    @abstractmethod
    def train(self, total_timesteps: int, **kwargs):
        """
        Train the agent.
        
        Args:
            total_timesteps: Total training timesteps
            **kwargs: Additional training parameters
        """
        pass
    
    @abstractmethod
    def save(self, path: str):
        """Save agent to file."""
        pass
    
    @abstractmethod
    def load(self, path: str):
        """Load agent from file."""
        pass

