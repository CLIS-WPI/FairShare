"""
Synthetic Data Generation and Validation

Generate realistic traffic patterns and user distributions,
and validate against public datasets and benchmarks.
"""

from .generator import SyntheticDataGenerator, UserProfile, TrafficPattern
from .validator import DataValidator, ValidationResult

__all__ = [
    'SyntheticDataGenerator',
    'UserProfile',
    'TrafficPattern',
    'DataValidator',
    'ValidationResult',
]

