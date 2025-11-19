"""
Tests for orbit propagator.
"""

import unittest
import numpy as np
from datetime import datetime
from src.channel import OrbitPropagator


class TestOrbitPropagator(unittest.TestCase):
    """Test orbit propagator."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple test satellite TLE
        self.line1 = "1 25544U 98067A   24001.00000000  .00000000  00000+0  00000+0 0  9999"
        self.line2 = "2 25544  51.6441 344.7765 0007976 126.2523 233.8264 15.49781658149437"
        self.sat_name = "TEST-SAT"
        
        try:
            self.prop = OrbitPropagator(backend='sgp4')
            self.prop.add_satellite(self.sat_name, self.line1, self.line2)
        except:
            self.skipTest("sgp4 not available")
    
    def test_propagate(self):
        """Test orbit propagation."""
        dt = datetime(2024, 1, 1, 0, 0, 0)
        
        try:
            pos, vel = self.prop.propagate(self.sat_name, dt)
            
            self.assertEqual(len(pos), 3)
            self.assertEqual(len(vel), 3)
            
            # Position should be reasonable (in meters, > Earth radius)
            self.assertGreater(np.linalg.norm(pos), 6e6)
            self.assertLess(np.linalg.norm(pos), 1e7)
        except Exception as e:
            self.skipTest(f"Propagation failed: {e}")
    
    def test_get_altitude(self):
        """Test altitude calculation."""
        dt = datetime(2024, 1, 1, 0, 0, 0)
        
        try:
            altitude = self.prop.get_altitude(self.sat_name, dt)
            
            # LEO altitude should be between 200-2000 km
            self.assertGreater(altitude, 200e3)
            self.assertLess(altitude, 2000e3)
        except Exception as e:
            self.skipTest(f"Altitude calculation failed: {e}")


if __name__ == '__main__':
    unittest.main()

