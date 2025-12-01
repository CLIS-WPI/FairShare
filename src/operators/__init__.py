"""
Multi-Operator LEO Constellation Management

This module handles multiple LEO satellite operators (e.g., Starlink, Kuiper)
competing for spectrum resources.
"""

from .operator import Operator, OperatorConfig, OperatorType
from .constellation import Constellation, SatelliteState
from .spectrum_bands import SpectrumBand, SpectrumBandManager, BandType

__all__ = [
    'Operator',
    'OperatorConfig',
    'OperatorType',
    'Constellation',
    'SatelliteState',
    'SpectrumBand',
    'SpectrumBandManager',
    'BandType',
]

