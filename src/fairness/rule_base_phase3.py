"""
Phase 3: Comprehensive Rule Base for Mamdani FIS

At least 12 strong rules for fairness evaluation in LEO satellite DSS.
Rules consider all 7 input variables.
"""

from typing import List, Dict
from .rule_base import FuzzyRule
from .membership_phase3 import build_all_membership_functions


class Phase3RuleBase:
    """
    Comprehensive rule base with at least 12 rules for fairness evaluation.
    
    Rules consider:
    - Throughput (Low, Medium, High)
    - Latency (Good, Acceptable, Poor)
    - Outage (Rare, Occasional, Frequent)
    - Priority (Low, Normal, High)
    - Doppler (Low, Medium, High)
    - Elevation (Low, Medium, High)
    - Beam Load (Light, Moderate, Heavy)
    
    Output: Fairness (Very-Low, Low, Medium, High, Very-High)
    """
    
    def __init__(self):
        """Initialize rule base with comprehensive rules."""
        self.membership_sets = build_all_membership_functions()
        self.rules = []
        self._build_rules()
    
    def _build_rules(self):
        """
        Build comprehensive rule base with at least 12 strong rules.
        
        Rule format: (conditions_dict, conclusion_label, weight)
        """
        # Rule 1: Poor latency + Frequent outage → Very-Low fairness
        self.rules.append(FuzzyRule(
            {"latency": "Poor", "outage": "Frequent"},
            "Very-Low",
            weight=1.0
        ))
        
        # Rule 2: High priority + Rare outage → High fairness
        self.rules.append(FuzzyRule(
            {"priority": "High", "outage": "Rare"},
            "High",
            weight=1.2
        ))
        
        # Rule 3: Low elevation + High doppler → Low fairness
        self.rules.append(FuzzyRule(
            {"elevation": "Low", "doppler": "High"},
            "Low",
            weight=1.0
        ))
        
        # Rule 4: Heavy beam load + Low throughput → Low fairness
        self.rules.append(FuzzyRule(
            {"beam_load": "Heavy", "throughput": "Low"},
            "Low",
            weight=1.1
        ))
        
        # Rule 5: High elevation + High throughput → Very-High fairness
        self.rules.append(FuzzyRule(
            {"elevation": "High", "throughput": "High"},
            "Very-High",
            weight=1.2
        ))
        
        # Rule 6: Low priority + Heavy beam load → Very-Low fairness
        self.rules.append(FuzzyRule(
            {"priority": "Low", "beam_load": "Heavy"},
            "Very-Low",
            weight=1.0
        ))
        
        # Rule 7: Good latency + High throughput + High elevation → Very-High fairness
        self.rules.append(FuzzyRule(
            {"latency": "Good", "throughput": "High", "elevation": "High"},
            "Very-High",
            weight=1.3
        ))
        
        # Rule 8: Poor latency + Low throughput → Very-Low fairness
        self.rules.append(FuzzyRule(
            {"latency": "Poor", "throughput": "Low"},
            "Very-Low",
            weight=1.1
        ))
        
        # Rule 9: High priority + Light beam load → High fairness
        self.rules.append(FuzzyRule(
            {"priority": "High", "beam_load": "Light"},
            "High",
            weight=1.1
        ))
        
        # Rule 10: Medium elevation + Medium throughput + Normal priority → Medium fairness
        self.rules.append(FuzzyRule(
            {"elevation": "Medium", "throughput": "Medium", "priority": "Normal"},
            "Medium",
            weight=1.0
        ))
        
        # Rule 11: Low doppler + High elevation + Rare outage → High fairness
        self.rules.append(FuzzyRule(
            {"doppler": "Low", "elevation": "High", "outage": "Rare"},
            "High",
            weight=1.2
        ))
        
        # Rule 12: Frequent outage + Heavy beam load → Very-Low fairness
        self.rules.append(FuzzyRule(
            {"outage": "Frequent", "beam_load": "Heavy"},
            "Very-Low",
            weight=1.1
        ))
        
        # Rule 13: High priority + Good latency + High throughput → Very-High fairness
        self.rules.append(FuzzyRule(
            {"priority": "High", "latency": "Good", "throughput": "High"},
            "Very-High",
            weight=1.3
        ))
        
        # Rule 14: Low elevation + High doppler + Frequent outage → Very-Low fairness
        self.rules.append(FuzzyRule(
            {"elevation": "Low", "doppler": "High", "outage": "Frequent"},
            "Very-Low",
            weight=1.2
        ))
        
        # Rule 15: Moderate beam load + Acceptable latency + Medium throughput → Medium fairness
        self.rules.append(FuzzyRule(
            {"beam_load": "Moderate", "latency": "Acceptable", "throughput": "Medium"},
            "Medium",
            weight=1.0
        ))
        
        # Rule 16: High priority + Low doppler + High elevation → Very-High fairness
        self.rules.append(FuzzyRule(
            {"priority": "High", "doppler": "Low", "elevation": "High"},
            "Very-High",
            weight=1.2
        ))
    
    def get_rules(self) -> List[FuzzyRule]:
        """Get all rules in the rule base."""
        return self.rules
    
    def evaluate_rules(self, inputs: Dict[str, float]) -> Dict[str, float]:
        """
        Evaluate all rules and return firing strengths for each conclusion.
        
        Uses min operator for AND (antecedent) and max operator for OR (conclusion).
        
        Args:
            inputs: Dictionary of crisp input values (normalized 0-1)
            
        Returns:
            Dictionary mapping conclusion labels to aggregated firing strengths
        """
        conclusion_strengths = {
            'Very-Low': 0.0,
            'Low': 0.0,
            'Medium': 0.0,
            'High': 0.0,
            'Very-High': 0.0
        }
        
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
                        threshold: float = 0.01) -> List[tuple]:
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

