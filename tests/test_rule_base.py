"""
Tests for fuzzy rule base.
"""

import unittest
from src.fairness import FuzzyRuleBase, FuzzyRule


class TestRuleBase(unittest.TestCase):
    """Test fuzzy rule base."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.rule_base = FuzzyRuleBase()
    
    def test_rule_count(self):
        """Test that rule base has 15 rules."""
        rules = self.rule_base.get_rules()
        self.assertEqual(len(rules), 15)
    
    def test_evaluate_rules(self):
        """Test rule evaluation."""
        inputs = {'load': 0.5, 'fairness': 0.5, 'priority': 0.5}
        conclusion_strengths = self.rule_base.evaluate_rules(inputs)
        
        self.assertIn('low', conclusion_strengths)
        self.assertIn('medium', conclusion_strengths)
        self.assertIn('high', conclusion_strengths)
        
        # All strengths should be in [0, 1]
        for strength in conclusion_strengths.values():
            self.assertGreaterEqual(strength, 0.0)
            self.assertLessEqual(strength, 1.0)
    
    def test_get_active_rules(self):
        """Test getting active rules."""
        inputs = {'load': 0.3, 'fairness': 0.6, 'priority': 0.7}
        active_rules = self.rule_base.get_active_rules(inputs, threshold=0.1)
        
        self.assertGreater(len(active_rules), 0)
        
        for rule, strength in active_rules:
            self.assertIsInstance(rule, FuzzyRule)
            self.assertGreaterEqual(strength, 0.1)


if __name__ == '__main__':
    unittest.main()

