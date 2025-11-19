"""
Unified Fuzzy Inference System (FIS) engine.

Implements Mamdani-type fuzzy inference with defuzzification
for fairness evaluation in dynamic spectrum sharing.

Phase 3: Complete implementation with 7 inputs + 1 output.
"""

import numpy as np
from typing import Dict, List, Optional
from .rule_base import FuzzyRuleBase
from .rule_base_phase3 import Phase3RuleBase
from .membership import MembershipFunctionSet, create_fairness_membership_functions
from .membership_phase3 import build_all_membership_functions, create_fairness_output_membership_functions


class FuzzyInferenceSystem:
    """
    Unified FIS engine for fuzzy logic inference.
    
    Phase 3: Complete Mamdani FIS with 7 inputs + 1 output.
    
    Supports:
    - Mamdani inference (min-max + centroid defuzzification)
    - Multiple defuzzification methods (centroid, bisector, MOM, LOM)
    - Rule evaluation and aggregation
    - 7 input variables: throughput, latency, outage, priority, doppler, elevation, beam_load
    - 1 output variable: fairness (Very-Low, Low, Medium, High, Very-High)
    """
    
    def __init__(self, rule_base: Optional[FuzzyRuleBase] = None,
                 defuzzification_method: str = 'centroid',
                 use_phase3: bool = True):
        """
        Initialize FIS engine.
        
        Args:
            rule_base: Fuzzy rule base (default: creates Phase3RuleBase)
            defuzzification_method: 'centroid', 'bisector', 'mom', 'lom'
            use_phase3: Use Phase 3 rule base with 7 inputs (default: True)
        """
        if use_phase3 and rule_base is None:
            # Use Phase 3 rule base
            self.rule_base = Phase3RuleBase()
            self.output_mfs = create_fairness_output_membership_functions()
            self.membership_sets = build_all_membership_functions()
        else:
            # Use legacy rule base
            self.rule_base = rule_base or FuzzyRuleBase()
            self.output_mfs = create_fairness_membership_functions()
            self.membership_sets = None
        
        self.defuzz_method = defuzzification_method
        self.use_phase3 = use_phase3
    
    def infer(self, inputs: Dict[str, float]) -> float:
        """
        Perform complete fuzzy inference: fuzzification -> rule evaluation -> 
        aggregation -> defuzzification.
        
        Args:
            inputs: Dictionary of crisp input values
            
        Returns:
            Defuzzified output (crisp value)
        """
        # Step 1: Evaluate rules
        conclusion_strengths = self.rule_base.evaluate_rules(inputs)
        
        # Step 2: Aggregate output membership functions
        aggregated_mf = self._aggregate_outputs(conclusion_strengths)
        
        # Step 3: Defuzzify
        output = self._defuzzify(aggregated_mf)
        
        return output
    
    def _aggregate_outputs(self, conclusion_strengths: Dict[str, float]) -> np.ndarray:
        """
        Aggregate output membership functions using maximum operator.
        
        Args:
            conclusion_strengths: Dictionary mapping labels to firing strengths
            
        Returns:
            Aggregated membership function as array
        """
        # Create output domain
        x = np.linspace(0, 1, 1000)
        aggregated = np.zeros_like(x)
        
        # For each conclusion, clip membership function and take maximum
        for label, strength in conclusion_strengths.items():
            if strength > 0:
                mf = self.output_mfs.functions[label]
                membership = np.array([mf(xi) for xi in x])
                # Clip by firing strength (min operator)
                clipped = np.minimum(membership, strength)
                # Aggregate using maximum (OR operator)
                aggregated = np.maximum(aggregated, clipped)
        
        return aggregated
    
    def _defuzzify(self, aggregated_mf: np.ndarray) -> float:
        """
        Defuzzify aggregated membership function.
        
        Phase 3: Centroid defuzzification (Center of Gravity).
        
        Args:
            aggregated_mf: Aggregated membership function array
            
        Returns:
            Defuzzified crisp value [0, 1]
        """
        # Use higher resolution for better accuracy
        if len(aggregated_mf) < 200:
            x = np.linspace(0, 1, 200)
            # Interpolate if needed
            from scipy.interpolate import interp1d
            if len(aggregated_mf) > 1:
                x_old = np.linspace(0, 1, len(aggregated_mf))
                interp_func = interp1d(x_old, aggregated_mf, kind='linear', 
                                      fill_value='extrapolate')
                aggregated_mf = interp_func(x)
            else:
                aggregated_mf = np.zeros(200)
        else:
            x = np.linspace(0, 1, len(aggregated_mf))
        
        if self.defuzz_method == 'centroid':
            # Center of gravity (COG)
            total_area = np.sum(aggregated_mf)
            if total_area == 0:
                return 0.5  # Default if no firing
            return np.sum(x * aggregated_mf) / total_area
        
        elif self.defuzz_method == 'bisector':
            # Bisector of area
            total_area = np.sum(aggregated_mf)
            if total_area == 0:
                return 0.5
            
            cumulative = np.cumsum(aggregated_mf)
            half_area = total_area / 2.0
            
            idx = np.searchsorted(cumulative, half_area)
            if idx >= len(x):
                idx = len(x) - 1
            return x[idx]
        
        elif self.defuzz_method == 'mom':
            # Mean of Maximum
            max_val = np.max(aggregated_mf)
            if max_val == 0:
                return 0.5
            
            max_indices = np.where(aggregated_mf == max_val)[0]
            return np.mean(x[max_indices])
        
        elif self.defuzz_method == 'lom':
            # Largest of Maximum
            max_val = np.max(aggregated_mf)
            if max_val == 0:
                return 0.5
            
            max_indices = np.where(aggregated_mf == max_val)[0]
            return x[max_indices[-1]]
        
        else:
            raise ValueError(f"Unknown defuzzification method: {self.defuzz_method}")
    
    def batch_infer(self, input_list: List[Dict[str, float]]) -> np.ndarray:
        """
        Perform inference for multiple input sets.
        
        Args:
            input_list: List of input dictionaries
            
        Returns:
            Array of defuzzified outputs
        """
        return np.array([self.infer(inputs) for inputs in input_list])
    
    def get_firing_strengths(self, inputs: Dict[str, float]) -> Dict[str, float]:
        """
        Get firing strengths for all rules.
        
        Args:
            inputs: Dictionary of crisp input values
            
        Returns:
            Dictionary mapping conclusion labels to firing strengths
        """
        return self.rule_base.evaluate_rules(inputs)
    
    def explain_inference(self, inputs: Dict[str, float]) -> Dict:
        """
        Provide detailed explanation of inference process.
        
        Args:
            inputs: Dictionary of crisp input values
            
        Returns:
            Dictionary with inference details
        """
        # Fuzzify inputs
        fuzzified = {}
        for var_name, value in inputs.items():
            if var_name in self.rule_base.membership_sets:
                fuzzified[var_name] = self.rule_base.membership_sets[var_name].fuzzify(value)
        
        # Get active rules
        active_rules = self.rule_base.get_active_rules(inputs, threshold=0.01)
        
        # Get firing strengths
        firing_strengths = self.get_firing_strengths(inputs)
        
        # Perform inference
        output = self.infer(inputs)
        
        return {
            'inputs': inputs,
            'fuzzified_inputs': fuzzified,
            'active_rules': [(str(rule), strength) for rule, strength in active_rules],
            'firing_strengths': firing_strengths,
            'output': output,
            'defuzzification_method': self.defuzz_method
        }


class AdaptiveFIS:
    """
    Adaptive FIS that can adjust rule weights based on performance.
    """
    
    def __init__(self, base_fis: FuzzyInferenceSystem):
        """
        Initialize adaptive FIS.
        
        Args:
            base_fis: Base FIS to adapt
        """
        self.fis = base_fis
        self.rule_weights_history = []
    
    def update_rule_weights(self, performance_feedback: Dict[str, float]):
        """
        Update rule weights based on performance feedback.
        
        Args:
            performance_feedback: Dictionary mapping rule indices to performance scores
        """
        # Simple adaptation: adjust weights based on feedback
        for idx, (rule, _) in enumerate(self.fis.rule_base.get_active_rules({})):
            if idx in performance_feedback:
                # Update weight (clamped to [0.1, 2.0])
                new_weight = rule.weight * (1 + 0.1 * performance_feedback[idx])
                rule.weight = np.clip(new_weight, 0.1, 2.0)
        
        self.rule_weights_history.append([r.weight for r in self.fis.rule_base.rules])

