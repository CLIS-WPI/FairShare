"""
Unit tests for RL module.
"""

import pytest
import numpy as np

from src.rl.environment import (
    FairnessRLEnvironment, RLState, ActionType
)
from src.rl.reward_shaping import (
    FairnessRewardShaping, RewardComponents
)


class TestRLEnvironment:
    """Tests for RL environment."""
    
    def test_environment_creation(self):
        """Test environment creation."""
        env = FairnessRLEnvironment(
            num_operators=2,
            num_users=10,
            max_bandwidth_mhz=1000.0,
            action_type=ActionType.CONTINUOUS
        )
        
        assert env.num_operators == 2
        assert env.num_users == 10
        assert env.max_bandwidth_mhz == 1000.0
    
    def test_environment_reset(self):
        """Test environment reset."""
        env = FairnessRLEnvironment(num_operators=2, num_users=10)
        
        obs, info = env.reset()
        
        assert obs is not None
        assert isinstance(obs, np.ndarray)
        assert "step" in info
        assert env.current_step == 0
    
    def test_environment_step(self):
        """Test environment step."""
        env = FairnessRLEnvironment(
            num_operators=2,
            num_users=10,
            action_type=ActionType.CONTINUOUS
        )
        
        obs, _ = env.reset()
        
        # Random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert obs is not None
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "allocations" in info
    
    def test_state_to_vector(self):
        """Test state to vector conversion."""
        state = RLState(
            operator_demands=np.array([100.0, 150.0]),
            operator_allocations=np.array([50.0, 75.0]),
            operator_utilizations=np.array([0.5, 0.75]),
            user_demands=np.array([10.0] * 5),
            user_priorities=np.array([0.8] * 5),
            available_bandwidth=1000.0,
            total_users=5,
            num_operators=2,
            current_fairness_scores=np.array([0.8, 0.9])
        )
        
        vector = state.to_vector()
        assert isinstance(vector, np.ndarray)
        assert len(vector) > 0


class TestRewardShaping:
    """Tests for reward shaping."""
    
    def test_reward_shaping_creation(self):
        """Test reward shaper creation."""
        shaper = FairnessRewardShaping(
            fairness_weight=0.5,
            fairness_metric="jain"
        )
        
        assert shaper.fairness_weight == 0.5
        assert shaper.efficiency_weight == 0.5
    
    def test_reward_components(self):
        """Test reward components."""
        components = RewardComponents(
            efficiency=0.8,
            fairness=0.7,
            utilization=0.9,
            qos_satisfaction=0.85,
            penalty=0.1
        )
        
        weights = {
            "efficiency": 0.3,
            "fairness": 0.3,
            "utilization": 0.2,
            "qos": 0.1,
            "penalty": 0.1
        }
        
        total = components.total(weights)
        assert isinstance(total, float)
    
    def test_compute_reward(self):
        """Test reward computation."""
        shaper = FairnessRewardShaping(fairness_weight=0.5)
        
        allocations = np.array([100.0, 150.0, 80.0])
        demands = np.array([120.0, 140.0, 90.0])
        
        reward = shaper.compute_reward(allocations, demands)
        assert isinstance(reward, float)
    
    def test_fairness_constraint_reward(self):
        """Test fairness constraint reward."""
        shaper = FairnessRewardShaping()
        
        # Fair allocation
        fair_allocations = np.array([100.0, 100.0, 100.0])
        reward_fair = shaper.compute_fairness_constraint_reward(
            fair_allocations, min_fairness=0.7
        )
        
        # Unfair allocation
        unfair_allocations = np.array([200.0, 10.0, 5.0])
        reward_unfair = shaper.compute_fairness_constraint_reward(
            unfair_allocations, min_fairness=0.7
        )
        
        # Fair should have higher reward
        assert reward_fair > reward_unfair

