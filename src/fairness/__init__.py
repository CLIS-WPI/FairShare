"""Fuzzy fairness evaluation for dynamic spectrum sharing."""

from .membership import TriangularMF, TrapezoidalMF, GaussianMF, MembershipFunctionSet
from .rule_base import FuzzyRule, FuzzyRuleBase
from .fuzzy_core import FuzzyInferenceSystem, AdaptiveFIS
from .metrics import (
    jain_fairness_index, alpha_fairness, gini_coefficient,
    max_min_fairness, fuzzy_fairness_score, FairnessEvaluator
)

__all__ = [
    'TriangularMF', 'TrapezoidalMF', 'GaussianMF', 'MembershipFunctionSet',
    'FuzzyRule', 'FuzzyRuleBase',
    'FuzzyInferenceSystem', 'AdaptiveFIS',
    'jain_fairness_index', 'alpha_fairness', 'gini_coefficient',
    'max_min_fairness', 'fuzzy_fairness_score', 'FairnessEvaluator'
]

