"""
Advanced Fairness Metrics

Multi-dimensional, vector-based, and learned fairness evaluation.
"""

from .traditional import TraditionalFairness
from .vector_metrics import VectorFairness, MultiDimensionalMetrics
from .learned_metrics import (
    LearnedFairness, AllocationProfile, LearnedFairnessFallback
)

__all__ = [
    'TraditionalFairness',
    'VectorFairness',
    'MultiDimensionalMetrics',
    'LearnedFairness',
    'AllocationProfile',
    'LearnedFairnessFallback',
]
