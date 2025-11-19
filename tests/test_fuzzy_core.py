"""
Tests for fuzzy inference system core.
"""

import unittest
import numpy as np
from src.fairness import FuzzyInferenceSystem, FuzzyRuleBase


class TestFuzzyCore(unittest.TestCase):
    """Test fuzzy inference system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.fis = FuzzyInferenceSystem()
    
    def test_inference_light_load(self):
        """Test inference for light load scenario."""
        inputs = {'load': 0.2, 'fairness': 0.5, 'priority': 0.5}
        output = self.fis.infer(inputs)
        
        # Light load should result in high fairness score
        self.assertGreater(output, 0.5)
        self.assertLessEqual(output, 1.0)
    
    def test_inference_heavy_load(self):
        """Test inference for heavy load scenario."""
        inputs = {'load': 0.9, 'fairness': 0.3, 'priority': 0.5}
        output = self.fis.infer(inputs)
        
        # Heavy load with low fairness should result in lower score
        self.assertGreaterEqual(output, 0.0)
        self.assertLessEqual(output, 1.0)
    
    def test_explain_inference(self):
        """Test inference explanation."""
        inputs = {'load': 0.5, 'fairness': 0.5, 'priority': 0.5}
        explanation = self.fis.explain_inference(inputs)
        
        self.assertIn('inputs', explanation)
        self.assertIn('fuzzified_inputs', explanation)
        self.assertIn('active_rules', explanation)
        self.assertIn('output', explanation)
    
    def test_batch_infer(self):
        """Test batch inference."""
        input_list = [
            {'load': 0.2, 'fairness': 0.5, 'priority': 0.5},
            {'load': 0.5, 'fairness': 0.5, 'priority': 0.5},
            {'load': 0.9, 'fairness': 0.3, 'priority': 0.5}
        ]
        
        outputs = self.fis.batch_infer(input_list)
        
        self.assertEqual(len(outputs), 3)
        self.assertTrue(np.all((outputs >= 0) & (outputs <= 1)))


if __name__ == '__main__':
    unittest.main()

