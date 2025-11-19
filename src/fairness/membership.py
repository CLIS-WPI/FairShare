"""
Membership functions for fuzzy logic system.

Implements Triangular Membership Functions (TMF) and other fuzzy sets
for fairness evaluation in dynamic spectrum sharing.
"""

import numpy as np
from typing import Callable, Tuple


class TriangularMF:
    """
    Triangular Membership Function.
    
    Defined by three points: left, center (peak), right.
    """
    
    def __init__(self, left: float, center: float, right: float):
        """
        Initialize triangular membership function.
        
        Args:
            left: Left boundary (membership = 0)
            center: Peak point (membership = 1)
            right: Right boundary (membership = 0)
        """
        if not (left <= center <= right):
            raise ValueError("Must have left <= center <= right")
        
        self.left = left
        self.center = center
        self.right = right
    
    def __call__(self, x: float) -> float:
        """
        Evaluate membership function at x.
        
        Args:
            x: Input value
            
        Returns:
            Membership degree [0, 1]
        """
        if x < self.left or x > self.right:
            return 0.0
        elif x == self.center:
            return 1.0
        elif x < self.center:
            return (x - self.left) / (self.center - self.left)
        else:
            return (self.right - x) / (self.right - self.center)
    
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate membership function for array of inputs.
        
        Args:
            x: Array of input values
            
        Returns:
            Array of membership degrees
        """
        return np.array([self.__call__(xi) for xi in x])


class TrapezoidalMF:
    """
    Trapezoidal Membership Function.
    
    Defined by four points: left, left_peak, right_peak, right.
    """
    
    def __init__(self, left: float, left_peak: float, 
                 right_peak: float, right: float):
        """
        Initialize trapezoidal membership function.
        
        Args:
            left: Left boundary (membership = 0)
            left_peak: Start of plateau (membership = 1)
            right_peak: End of plateau (membership = 1)
            right: Right boundary (membership = 0)
        """
        if not (left <= left_peak <= right_peak <= right):
            raise ValueError("Must have left <= left_peak <= right_peak <= right")
        
        self.left = left
        self.left_peak = left_peak
        self.right_peak = right_peak
        self.right = right
    
    def __call__(self, x: float) -> float:
        """
        Evaluate membership function at x.
        
        Args:
            x: Input value
            
        Returns:
            Membership degree [0, 1]
        """
        if x < self.left or x > self.right:
            return 0.0
        elif self.left_peak <= x <= self.right_peak:
            return 1.0
        elif x < self.left_peak:
            return (x - self.left) / (self.left_peak - self.left)
        else:
            return (self.right - x) / (self.right - self.right_peak)
    
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate membership function for array of inputs."""
        return np.array([self.__call__(xi) for xi in x])


class GaussianMF:
    """
    Gaussian Membership Function.
    """
    
    def __init__(self, center: float, sigma: float):
        """
        Initialize Gaussian membership function.
        
        Args:
            center: Mean (peak) of Gaussian
            sigma: Standard deviation
        """
        self.center = center
        self.sigma = sigma
    
    def __call__(self, x: float) -> float:
        """Evaluate membership function at x."""
        return np.exp(-0.5 * ((x - self.center) / self.sigma) ** 2)
    
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate membership function for array of inputs."""
        return np.exp(-0.5 * ((x - self.center) / self.sigma) ** 2)


class MembershipFunctionSet:
    """
    Collection of membership functions for fuzzy variables.
    
    Used to define linguistic variables like "low", "medium", "high".
    """
    
    def __init__(self, name: str):
        """
        Initialize membership function set.
        
        Args:
            name: Name of the fuzzy variable
        """
        self.name = name
        self.functions = {}
    
    def add_function(self, label: str, mf: Callable) -> None:
        """
        Add a membership function with a linguistic label.
        
        Args:
            label: Linguistic label (e.g., "low", "medium", "high")
            mf: Membership function object
        """
        self.functions[label] = mf
    
    def fuzzify(self, x: float) -> dict:
        """
        Fuzzify a crisp input value.
        
        Args:
            x: Crisp input value
            
        Returns:
            Dictionary mapping labels to membership degrees
        """
        return {label: mf(x) for label, mf in self.functions.items()}
    
    def get_membership(self, label: str, x: float) -> float:
        """
        Get membership degree for a specific label.
        
        Args:
            label: Linguistic label
            x: Input value
            
        Returns:
            Membership degree
        """
        if label not in self.functions:
            raise ValueError(f"Label '{label}' not found")
        return self.functions[label](x)


def create_fairness_membership_functions() -> MembershipFunctionSet:
    """
    Create standard membership functions for fairness metrics.
    
    Returns:
        MembershipFunctionSet for fairness (0-1 scale)
    """
    mfs = MembershipFunctionSet("fairness")
    
    # Low fairness: 0.0 to 0.4
    mfs.add_function("low", TriangularMF(0.0, 0.0, 0.4))
    
    # Medium fairness: 0.2 to 0.8
    mfs.add_function("medium", TriangularMF(0.2, 0.5, 0.8))
    
    # High fairness: 0.6 to 1.0
    mfs.add_function("high", TriangularMF(0.6, 1.0, 1.0))
    
    return mfs


def create_load_membership_functions() -> MembershipFunctionSet:
    """
    Create membership functions for network load.
    
    Returns:
        MembershipFunctionSet for load (0-1 scale)
    """
    mfs = MembershipFunctionSet("load")
    
    mfs.add_function("light", TriangularMF(0.0, 0.0, 0.4))
    mfs.add_function("moderate", TriangularMF(0.2, 0.5, 0.8))
    mfs.add_function("heavy", TriangularMF(0.6, 1.0, 1.0))
    
    return mfs


def create_priority_membership_functions() -> MembershipFunctionSet:
    """
    Create membership functions for user priority.
    
    Returns:
        MembershipFunctionSet for priority (0-1 scale)
    """
    mfs = MembershipFunctionSet("priority")
    
    mfs.add_function("low", TriangularMF(0.0, 0.0, 0.5))
    mfs.add_function("normal", TriangularMF(0.2, 0.5, 0.8))
    mfs.add_function("high", TriangularMF(0.5, 1.0, 1.0))
    
    return mfs

