"""
Tests for spectrum conflict detection (multi-operator logic).
"""

import pytest
import numpy as np
from src.dss.spectrum_environment import SpectrumEnvironment, Beam


class TestSpectrumConflict:
    """Test conflict detection in SpectrumEnvironment."""
    
    def test_different_beams_same_frequency_ok(self):
        """Test: Two different beams can allocate same frequency (OK)."""
        env = SpectrumEnvironment((10e9, 12e9))
        
        # Register two different beams
        beam1 = Beam("beam1", "sat1", 11e9, 100e6, 40.0, (0.0, 0.0), 30.0)
        beam2 = Beam("beam2", "sat2", 11e9, 100e6, 40.0, (1.0, 1.0), 30.0)
        
        env.register_beam(beam1)
        env.register_beam(beam2)
        
        # Allocate same frequency to different beams (should work)
        env.update_beam_usage("beam1", 11e9, 100e6, 40.0)
        env.update_beam_usage("beam2", 11e9, 100e6, 40.0)
        
        # Both should be allocated
        load1 = env.compute_beam_load("beam1")
        load2 = env.compute_beam_load("beam2")
        
        assert load1 > 0
        assert load2 > 0
        print("✓ Different beams can use same frequency (spatial reuse)")
    
    def test_same_beam_double_allocation_not_ok(self):
        """Test: Same beam cannot double-allocate same frequency (NOT OK)."""
        env = SpectrumEnvironment((10e9, 12e9))
        
        beam1 = Beam("beam1", "sat1", 11e9, 100e6, 40.0, (0.0, 0.0), 30.0)
        env.register_beam(beam1)
        
        # First allocation
        allocation1 = env.allocate("user1", 100e6, beam_id="beam1", preferred_frequency_hz=11e9)
        assert allocation1 is not None
        
        # Second allocation to same beam, same frequency (should fail)
        allocation2 = env.allocate("user2", 100e6, beam_id="beam1", preferred_frequency_hz=11e9)
        
        # Should either return None or different frequency
        if allocation2 is not None:
            # If not None, must be different frequency
            assert abs(allocation2[0] - 11e9) > 50e6, "Same beam allocated same frequency twice!"
        
        print("✓ Same beam cannot double-allocate same frequency")
    
    def test_conflict_detection(self):
        """Test conflict detection method."""
        env = SpectrumEnvironment((10e9, 12e9))
        
        beam1 = Beam("beam1", "sat1", 11e9, 100e6, 40.0, (0.0, 0.0), 30.0)
        env.register_beam(beam1)
        env.update_beam_usage("beam1", 11e9, 100e6, 40.0)
        
        # Check conflict
        has_conflict = env.check_conflict(11e9, 100e6, exclude_beam_id="beam1")
        assert not has_conflict, "Should not conflict with itself"
        
        has_conflict_other = env.check_conflict(11e9, 100e6, exclude_beam_id=None)
        assert has_conflict_other, "Should conflict with beam1"
        
        print("✓ Conflict detection works correctly")

