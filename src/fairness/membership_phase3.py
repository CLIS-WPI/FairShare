"""
Phase 3: Complete Membership Functions for Mamdani FIS

7 Input Variables:
1. Throughput → Low, Medium, High
2. Latency → Good, Acceptable, Poor
3. Outage → Rare, Occasional, Frequent
4. Priority → Low, Normal, High
5. Doppler → Low, Medium, High
6. Elevation → Low, Medium, High
7. Beam Load → Light, Moderate, Heavy

1 Output Variable:
- Fairness → Very-Low, Low, Medium, High, Very-High
"""

import numpy as np
from typing import Callable
from .membership import TriangularMF, MembershipFunctionSet


def _mf_triangular(x: float, a: float, b: float, c: float) -> float:
    """
    Triangular membership function.
    
    Args:
        x: Input value
        a: Left boundary (membership = 0)
        b: Center/peak (membership = 1)
        c: Right boundary (membership = 0)
        
    Returns:
        Membership degree [0, 1]
    """
    if x < a or x > c:
        return 0.0
    elif x == b:
        return 1.0
    elif x < b:
        return (x - a) / (b - a) if (b - a) > 0 else 0.0
    else:
        return (c - x) / (c - b) if (c - b) > 0 else 0.0


def create_throughput_membership_functions() -> MembershipFunctionSet:
    """
    Create membership functions for Throughput (normalized 0-1).
    
    Returns:
        MembershipFunctionSet for throughput
    """
    mfs = MembershipFunctionSet("throughput")
    
    # Low: 0.0 to 0.4
    mfs.add_function("Low", TriangularMF(0.0, 0.0, 0.4))
    
    # Medium: 0.2 to 0.8
    mfs.add_function("Medium", TriangularMF(0.2, 0.5, 0.8))
    
    # High: 0.6 to 1.0
    mfs.add_function("High", TriangularMF(0.6, 1.0, 1.0))
    
    return mfs


def create_latency_membership_functions() -> MembershipFunctionSet:
    """
    Create membership functions for Latency (normalized 0-1, lower is better).
    
    Returns:
        MembershipFunctionSet for latency
    """
    mfs = MembershipFunctionSet("latency")
    
    # Good: 0.0 to 0.3 (low latency)
    mfs.add_function("Good", TriangularMF(0.0, 0.0, 0.3))
    
    # Acceptable: 0.2 to 0.7
    mfs.add_function("Acceptable", TriangularMF(0.2, 0.45, 0.7))
    
    # Poor: 0.6 to 1.0 (high latency)
    mfs.add_function("Poor", TriangularMF(0.6, 1.0, 1.0))
    
    return mfs


def create_outage_membership_functions() -> MembershipFunctionSet:
    """
    Create membership functions for Outage probability (normalized 0-1).
    
    Returns:
        MembershipFunctionSet for outage
    """
    mfs = MembershipFunctionSet("outage")
    
    # Rare: 0.0 to 0.3 (low outage)
    mfs.add_function("Rare", TriangularMF(0.0, 0.0, 0.3))
    
    # Occasional: 0.2 to 0.7
    mfs.add_function("Occasional", TriangularMF(0.2, 0.45, 0.7))
    
    # Frequent: 0.6 to 1.0 (high outage)
    mfs.add_function("Frequent", TriangularMF(0.6, 1.0, 1.0))
    
    return mfs


def create_priority_membership_functions() -> MembershipFunctionSet:
    """
    Create membership functions for Priority (normalized 0-1).
    
    Returns:
        MembershipFunctionSet for priority
    """
    mfs = MembershipFunctionSet("priority")
    
    # Low: 0.0 to 0.4
    mfs.add_function("Low", TriangularMF(0.0, 0.0, 0.4))
    
    # Normal: 0.3 to 0.7
    mfs.add_function("Normal", TriangularMF(0.3, 0.5, 0.7))
    
    # High: 0.6 to 1.0
    mfs.add_function("High", TriangularMF(0.6, 1.0, 1.0))
    
    return mfs


def create_doppler_membership_functions() -> MembershipFunctionSet:
    """
    Create membership functions for Doppler shift (normalized 0-1).
    
    Returns:
        MembershipFunctionSet for doppler
    """
    mfs = MembershipFunctionSet("doppler")
    
    # Low: 0.0 to 0.4 (low Doppler)
    mfs.add_function("Low", TriangularMF(0.0, 0.0, 0.4))
    
    # Medium: 0.3 to 0.7
    mfs.add_function("Medium", TriangularMF(0.3, 0.5, 0.7))
    
    # High: 0.6 to 1.0 (high Doppler)
    mfs.add_function("High", TriangularMF(0.6, 1.0, 1.0))
    
    return mfs


def create_elevation_membership_functions() -> MembershipFunctionSet:
    """
    Create membership functions for Elevation angle (normalized 0-1, higher is better).
    
    Returns:
        MembershipFunctionSet for elevation
    """
    mfs = MembershipFunctionSet("elevation")
    
    # Low: 0.0 to 0.4 (low elevation)
    mfs.add_function("Low", TriangularMF(0.0, 0.0, 0.4))
    
    # Medium: 0.3 to 0.7
    mfs.add_function("Medium", TriangularMF(0.3, 0.5, 0.7))
    
    # High: 0.6 to 1.0 (high elevation)
    mfs.add_function("High", TriangularMF(0.6, 1.0, 1.0))
    
    return mfs


def create_beam_load_membership_functions() -> MembershipFunctionSet:
    """
    Create membership functions for Beam Load (normalized 0-1).
    
    Returns:
        MembershipFunctionSet for beam_load
    """
    mfs = MembershipFunctionSet("beam_load")
    
    # Light: 0.0 to 0.4
    mfs.add_function("Light", TriangularMF(0.0, 0.0, 0.4))
    
    # Moderate: 0.3 to 0.7
    mfs.add_function("Moderate", TriangularMF(0.3, 0.5, 0.7))
    
    # Heavy: 0.6 to 1.0
    mfs.add_function("Heavy", TriangularMF(0.6, 1.0, 1.0))
    
    return mfs


def create_fairness_output_membership_functions() -> MembershipFunctionSet:
    """
    Create membership functions for Fairness output (0-1 scale).
    
    Returns:
        MembershipFunctionSet for fairness output with 5 levels
    """
    mfs = MembershipFunctionSet("fairness")
    
    # Very-Low: 0.0 to 0.3
    mfs.add_function("Very-Low", TriangularMF(0.0, 0.0, 0.3))
    
    # Low: 0.2 to 0.5
    mfs.add_function("Low", TriangularMF(0.2, 0.35, 0.5))
    
    # Medium: 0.4 to 0.7
    mfs.add_function("Medium", TriangularMF(0.4, 0.55, 0.7))
    
    # High: 0.6 to 0.9
    mfs.add_function("High", TriangularMF(0.6, 0.75, 0.9))
    
    # Very-High: 0.8 to 1.0
    mfs.add_function("Very-High", TriangularMF(0.8, 1.0, 1.0))
    
    return mfs


def build_all_membership_functions() -> dict:
    """
    Build all membership function sets for Phase 3 FIS.
    
    Returns:
        Dictionary mapping variable names to MembershipFunctionSet
    """
    return {
        'throughput': create_throughput_membership_functions(),
        'latency': create_latency_membership_functions(),
        'outage': create_outage_membership_functions(),
        'priority': create_priority_membership_functions(),
        'doppler': create_doppler_membership_functions(),
        'elevation': create_elevation_membership_functions(),
        'beam_load': create_beam_load_membership_functions(),
        'fairness': create_fairness_output_membership_functions()
    }

