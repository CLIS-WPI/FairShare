"""
RL Environment for fairness-optimized spectrum allocation.

Gymnasium-compatible environment for training RL agents to optimize
spectrum sharing under fairness constraints.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
from enum import Enum

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    try:
        import gym
        from gym import spaces
        GYM_AVAILABLE = True
    except ImportError:
        GYM_AVAILABLE = False


class ActionType(Enum):
    """Types of actions."""
    CONTINUOUS = "continuous"  # Continuous bandwidth allocation
    DISCRETE = "discrete"  # Discrete allocation levels


@dataclass
class RLState:
    """State representation for RL agent."""
    # Operator states
    operator_demands: np.ndarray  # Demand per operator
    operator_allocations: np.ndarray  # Current allocation per operator
    operator_utilizations: np.ndarray  # Utilization per operator
    
    # User states
    user_demands: np.ndarray  # Demand per user
    user_priorities: np.ndarray  # Priority per user
    
    # System state
    available_bandwidth: float
    total_users: int
    num_operators: int
    
    # Fairness state
    current_fairness_scores: np.ndarray  # Fairness per operator
    
    def to_vector(self) -> np.ndarray:
        """Convert to feature vector."""
        return np.concatenate([
            self.operator_demands,
            self.operator_allocations,
            self.operator_utilizations,
            self.user_demands,
            self.user_priorities,
            np.array([self.available_bandwidth]),
            np.array([self.total_users]),
            np.array([self.num_operators]),
            self.current_fairness_scores
        ])


class FairnessRLEnvironment:
    """
    RL Environment for fairness-optimized spectrum allocation.
    
    Compatible with Gymnasium/Gym interface for training RL agents.
    """
    
    def __init__(
        self,
        num_operators: int = 2,
        num_users: int = 100,
        max_bandwidth_mhz: float = 1000.0,
        action_type: ActionType = ActionType.CONTINUOUS,
        fairness_weight: float = 0.5
    ):
        """
        Initialize RL environment.
        
        Args:
            num_operators: Number of operators
            num_users: Number of users
            max_bandwidth_mhz: Maximum available bandwidth
            action_type: Type of action space
            fairness_weight: Weight for fairness in reward (0.0 to 1.0)
        """
        if not GYM_AVAILABLE:
            raise ImportError("Gymnasium/Gym required for RL environment")
        
        self.num_operators = num_operators
        self.num_users = num_users
        self.max_bandwidth_mhz = max_bandwidth_mhz
        self.action_type = action_type
        self.fairness_weight = fairness_weight
        
        # State space dimension
        # [operator_demands, allocations, utilizations, user_demands, priorities,
        #  available_bw, total_users, num_ops, fairness_scores]
        state_dim = (
            num_operators * 3 +  # operator states
            num_users * 2 +  # user states
            3 +  # system states
            num_operators  # fairness scores
        )
        
        # Action space
        if action_type == ActionType.CONTINUOUS:
            # Continuous: allocation per operator [0, 1] (fraction of max)
            self.action_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(num_operators,),
                dtype=np.float32
            )
        else:
            # Discrete: allocation levels per operator
            self.action_space = spaces.MultiDiscrete(
                [10] * num_operators  # 10 levels per operator
            )
        
        # Observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32
        )
        
        # Current state
        self.state: Optional[RLState] = None
        self.current_step = 0
        self.max_steps = 1000
        
        # Allocation engine (will be set externally)
        self.allocation_engine = None
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment.
        
        Returns:
            Initial observation and info dict
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize random state
        self._initialize_state()
        self.current_step = 0
        
        observation = self.state.to_vector()
        info = {
            "step": self.current_step,
            "fairness_scores": self.state.current_fairness_scores.copy()
        }
        
        return observation, info
    
    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step.
        
        Args:
            action: Action from agent
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Convert action to allocations
        allocations = self._action_to_allocations(action)
        
        # Apply allocations (simplified - in real implementation,
        # this would use the allocation engine)
        self._apply_allocations(allocations)
        
        # Compute reward
        reward = self._compute_reward(allocations)
        
        # Update state
        self._update_state()
        
        # Check termination
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        # Observation
        observation = self.state.to_vector()
        
        # Info
        info = {
            "step": self.current_step,
            "allocations": allocations,
            "fairness_scores": self.state.current_fairness_scores.copy(),
            "utilizations": self.state.operator_utilizations.copy()
        }
        
        return observation, reward, terminated, truncated, info
    
    def _initialize_state(self):
        """Initialize random state."""
        # Random demands and priorities
        operator_demands = np.random.uniform(100, 500, self.num_operators)
        operator_allocations = np.random.uniform(0, 200, self.num_operators)
        operator_utilizations = operator_allocations / (self.max_bandwidth_mhz / self.num_operators)
        
        user_demands = np.random.uniform(1, 20, self.num_users)
        user_priorities = np.random.uniform(0.5, 1.0, self.num_users)
        
        current_fairness = np.random.uniform(0.5, 1.0, self.num_operators)
        
        self.state = RLState(
            operator_demands=operator_demands,
            operator_allocations=operator_allocations,
            operator_utilizations=operator_utilizations,
            user_demands=user_demands,
            user_priorities=user_priorities,
            available_bandwidth=self.max_bandwidth_mhz,
            total_users=self.num_users,
            num_operators=self.num_operators,
            current_fairness_scores=current_fairness
        )
    
    def _action_to_allocations(self, action: np.ndarray) -> np.ndarray:
        """Convert action to bandwidth allocations."""
        if self.action_type == ActionType.CONTINUOUS:
            # Action is fraction [0, 1] of max bandwidth per operator
            max_per_operator = self.max_bandwidth_mhz / self.num_operators
            allocations = action * max_per_operator
        else:
            # Discrete: convert to continuous
            max_per_operator = self.max_bandwidth_mhz / self.num_operators
            allocations = (action / 9.0) * max_per_operator
        
        return allocations
    
    def _apply_allocations(self, allocations: np.ndarray):
        """Apply allocations to state."""
        # Update operator allocations
        self.state.operator_allocations = allocations.copy()
        
        # Update utilizations
        max_per_operator = self.max_bandwidth_mhz / self.num_operators
        self.state.operator_utilizations = allocations / max_per_operator
    
    def _compute_reward(
        self,
        allocations: np.ndarray
    ) -> float:
        """
        Compute reward: efficiency + fairness.
        
        r = α × efficiency + β × fairness
        """
        from src.fairness.traditional import TraditionalFairness
        
        # Efficiency: total throughput (simplified)
        efficiency = np.sum(allocations) / self.max_bandwidth_mhz
        
        # Fairness: Jain Index of allocations
        fairness = TraditionalFairness.jain_index(allocations.tolist())
        
        # Combined reward
        efficiency_weight = 1.0 - self.fairness_weight
        reward = (
            efficiency_weight * efficiency +
            self.fairness_weight * fairness
        )
        
        return float(reward)
    
    def _update_state(self):
        """Update state after step."""
        # In real implementation, this would update based on
        # actual allocation results and performance metrics
        # For now, simplified random updates
        self.state.operator_demands += np.random.normal(0, 10, self.num_operators)
        self.state.operator_demands = np.maximum(0, self.state.operator_demands)
        
        # Update fairness scores (simplified)
        from src.fairness.traditional import TraditionalFairness
        self.state.current_fairness_scores = np.array([
            TraditionalFairness.jain_index(self.state.operator_allocations.tolist())
        ] * self.num_operators)

