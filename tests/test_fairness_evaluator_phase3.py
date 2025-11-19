"""
Phase 3: Tests for FairnessEvaluator with Phase 3 FIS.
"""

import pytest
import numpy as np
from src.fairness.metrics import FairnessEvaluator
from src.fairness.fuzzy_core import FuzzyInferenceSystem


class TestFairnessEvaluatorPhase3:
    """Test FairnessEvaluator with Phase 3 FIS."""
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization with Phase 3 FIS."""
        fis = FuzzyInferenceSystem(use_phase3=True)
        evaluator = FairnessEvaluator(fis)
        
        assert evaluator.fis is not None
        assert evaluator.fis.use_phase3 == True
    
    def test_evaluate_high_fairness_case(self):
        """Test evaluation for high fairness case."""
        fis = FuzzyInferenceSystem(use_phase3=True)
        evaluator = FairnessEvaluator(fis)
        
        # High fairness scenario: equal allocations
        allocations = np.array([1.0, 1.0, 1.0, 1.0])
        demands = np.array([1.0, 1.0, 1.0, 1.0])
        priorities = np.array([0.5, 0.5, 0.5, 0.5])
        
        results = evaluator.evaluate(
            allocations=allocations,
            demands=demands,
            priorities=priorities,
            network_load=0.5
        )
        
        assert results['jain_index'] > 0.9
        assert results['fuzzy_fairness_score'] > 0.5
    
    def test_evaluate_low_fairness_case(self):
        """Test evaluation for low fairness case."""
        fis = FuzzyInferenceSystem(use_phase3=True)
        evaluator = FairnessEvaluator(fis)
        
        # Low fairness scenario: very unequal allocations
        allocations = np.array([0.1, 0.1, 0.1, 1.0])
        demands = np.array([1.0, 1.0, 1.0, 1.0])
        priorities = np.array([0.5, 0.5, 0.5, 0.5])
        
        results = evaluator.evaluate(
            allocations=allocations,
            demands=demands,
            priorities=priorities,
            network_load=0.8
        )
        
        assert results['jain_index'] < 0.5
        assert results['fuzzy_fairness_score'] < 0.5
    
    def test_consistency(self):
        """Test consistency: same inputs should give same results."""
        fis = FuzzyInferenceSystem(use_phase3=True)
        evaluator = FairnessEvaluator(fis)
        
        allocations = np.array([0.5, 0.6, 0.7, 0.8])
        demands = np.array([0.5, 0.6, 0.7, 0.8])
        priorities = np.array([0.5, 0.6, 0.7, 0.8])
        
        results1 = evaluator.evaluate(allocations, demands, priorities, 0.6)
        results2 = evaluator.evaluate(allocations, demands, priorities, 0.6)
        
        assert abs(results1['fuzzy_fairness_score'] - results2['fuzzy_fairness_score']) < 1e-6


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

