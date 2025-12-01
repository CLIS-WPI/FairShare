"""
Reinforcement Learning for Fairness-Optimized Spectrum Allocation

RL agents that optimize spectrum sharing under fairness constraints.
"""

from .environment import (
    FairnessRLEnvironment, RLState, ActionType
)
from .reward_shaping import (
    FairnessRewardShaping, RewardComponents
)

__all__ = [
    'FairnessRLEnvironment',
    'RLState',
    'ActionType',
    'FairnessRewardShaping',
    'RewardComponents',
]

