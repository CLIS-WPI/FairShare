"""
Fuzzy rule base for fairness evaluation.

Implements a 15-rule fuzzy inference system for dynamic spectrum sharing
fairness assessment.
"""

from typing import List, Dict, Tuple
from .membership import MembershipFunctionSet, create_fairness_membership_functions, \
                       create_load_membership_functions, create_priority_membership_functions


class FuzzyRule:
    """
    Single fuzzy rule in the form: IF conditions THEN conclusion.
    """
    
    def __init__(self, conditions: Dict[str, str], conclusion: str, weight: float = 1.0):
        """
        Initialize fuzzy rule.
        
        Args:
            conditions: Dictionary mapping variable names to linguistic labels
            conclusion: Output linguistic label
            weight: Rule weight (0-1)
        """
        self.conditions = conditions
        self.conclusion = conclusion
        self.weight = weight
    
    def evaluate_antecedent(self, inputs: Dict[str, float], 
                           membership_sets: Dict[str, MembershipFunctionSet]) -> float:
        """
        Evaluate rule antecedent (IF part) using minimum (AND) operator.
        
        Args:
            inputs: Dictionary of crisp input values
            membership_sets: Dictionary of membership function sets
            
        Returns:
            Firing strength of the rule
        """
        firing_strengths = []
        
        for var_name, label in self.conditions.items():
            if var_name not in inputs:
                return 0.0
            
            if var_name not in membership_sets:
                return 0.0
            
            membership = membership_sets[var_name].get_membership(label, inputs[var_name])
            firing_strengths.append(membership)
        
        # Minimum (AND) operator
        firing_strength = min(firing_strengths) if firing_strengths else 0.0
        
        return firing_strength * self.weight
    
    def __repr__(self) -> str:
        """String representation of rule."""
        conditions_str = " AND ".join([f"{var}={label}" 
                                      for var, label in self.conditions.items()])
        return f"IF {conditions_str} THEN fairness={self.conclusion} (weight={self.weight})"


class FuzzyRuleBase:
    """
    15-rule fuzzy system for fairness evaluation.
    
    Rules consider:
    - Network load (light, moderate, heavy)
    - Current fairness (low, medium, high)
    - User priority (low, normal, high)
    """
    
    def __init__(self):
        """Initialize rule base with 15 rules."""
        self.rules = []
        self._create_rules()
        
        # Initialize membership function sets
        self.membership_sets = {
            'load': create_load_membership_functions(),
            'fairness': create_fairness_membership_functions(),
            'priority': create_priority_membership_functions()
        }
    
    def _create_rules(self):
        """Create the 15-rule fuzzy system."""
        # Rule 1-3: Light load scenarios
        self.rules.append(FuzzyRule(
            {'load': 'light', 'fairness': 'low'},
            'high', weight=1.0
        ))
        self.rules.append(FuzzyRule(
            {'load': 'light', 'fairness': 'medium'},
            'high', weight=1.0
        ))
        self.rules.append(FuzzyRule(
            {'load': 'light', 'fairness': 'high'},
            'high', weight=0.9
        ))
        
        # Rule 4-6: Moderate load scenarios
        self.rules.append(FuzzyRule(
            {'load': 'moderate', 'fairness': 'low'},
            'high', weight=1.0
        ))
        self.rules.append(FuzzyRule(
            {'load': 'moderate', 'fairness': 'medium'},
            'medium', weight=1.0
        ))
        self.rules.append(FuzzyRule(
            {'load': 'moderate', 'fairness': 'high'},
            'medium', weight=0.8
        ))
        
        # Rule 7-9: Heavy load scenarios
        self.rules.append(FuzzyRule(
            {'load': 'heavy', 'fairness': 'low'},
            'medium', weight=1.0
        ))
        self.rules.append(FuzzyRule(
            {'load': 'heavy', 'fairness': 'medium'},
            'low', weight=1.0
        ))
        self.rules.append(FuzzyRule(
            {'load': 'heavy', 'fairness': 'high'},
            'low', weight=0.9
        ))
        
        # Rule 10-12: Priority-aware rules (high priority users)
        self.rules.append(FuzzyRule(
            {'load': 'light', 'priority': 'high'},
            'high', weight=1.2
        ))
        self.rules.append(FuzzyRule(
            {'load': 'moderate', 'priority': 'high'},
            'high', weight=1.1
        ))
        self.rules.append(FuzzyRule(
            {'load': 'heavy', 'priority': 'high'},
            'medium', weight=1.0
        ))
        
        # Rule 13-15: Priority-aware rules (low priority users)
        self.rules.append(FuzzyRule(
            {'load': 'light', 'priority': 'low'},
            'high', weight=0.9
        ))
        self.rules.append(FuzzyRule(
            {'load': 'moderate', 'priority': 'low'},
            'medium', weight=0.9
        ))
        self.rules.append(FuzzyRule(
            {'load': 'heavy', 'priority': 'low'},
            'low', weight=0.8
        ))
    
    def get_rules(self) -> List[FuzzyRule]:
        """Get all rules in the rule base."""
        return self.rules
    
    def evaluate_rules(self, inputs: Dict[str, float]) -> Dict[str, float]:
        """
        Evaluate all rules and return firing strengths for each conclusion.
        
        Args:
            inputs: Dictionary of crisp input values
            
        Returns:
            Dictionary mapping conclusion labels to aggregated firing strengths
        """
        conclusion_strengths = {'low': 0.0, 'medium': 0.0, 'high': 0.0}
        
        for rule in self.rules:
            firing_strength = rule.evaluate_antecedent(inputs, self.membership_sets)
            
            # Maximum (OR) operator for same conclusion
            conclusion = rule.conclusion
            if conclusion in conclusion_strengths:
                conclusion_strengths[conclusion] = max(
                    conclusion_strengths[conclusion], 
                    firing_strength
                )
        
        return conclusion_strengths
    
    def get_active_rules(self, inputs: Dict[str, float], 
                        threshold: float = 0.1) -> List[Tuple[FuzzyRule, float]]:
        """
        Get rules that fire above threshold.
        
        Args:
            inputs: Dictionary of crisp input values
            threshold: Minimum firing strength
            
        Returns:
            List of (rule, firing_strength) tuples
        """
        active = []
        
        for rule in self.rules:
            firing_strength = rule.evaluate_antecedent(inputs, self.membership_sets)
            if firing_strength >= threshold:
                active.append((rule, firing_strength))
        
        return active

