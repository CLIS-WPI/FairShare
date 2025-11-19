"""
Integration tests for channel model with OpenNTN and Sionna.
"""

import unittest
import numpy as np
from datetime import datetime
from src.channel import OrbitPropagator, SatelliteGeometry, ChannelModel


class TestChannelModel(unittest.TestCase):
    """Test channel model with OpenNTN/Sionna integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test satellite TLE
        self.line1 = "1 25544U 98067A   24001.00000000  .00000000  00000+0  00000+0 0  9999"
        self.line2 = "2 25544  51.6441 344.7765 0007976 126.2523 233.8264 15.49781658149437"
        
        # Initialize components
        self.geometry = SatelliteGeometry(40.0, -100.0)
        self.channel_model = ChannelModel(
            frequency_hz=12e9,
            use_sionna=False  # Disable for basic tests
        )
    
    def test_antenna_gain_tx(self):
        """Test transmit antenna gain."""
        gain_boresight = self.channel_model.antenna_gain_tx(0.0)
        gain_off_axis = self.channel_model.antenna_gain_tx(10.0)
        
        self.assertGreater(gain_boresight, gain_off_axis)
        self.assertGreater(gain_boresight, 0)
    
    def test_antenna_gain_rx(self):
        """Test receive antenna gain."""
        gain_boresight = self.channel_model.antenna_gain_rx(0.0)
        gain_off_axis = self.channel_model.antenna_gain_rx(10.0)
        
        self.assertGreater(gain_boresight, gain_off_axis)
        self.assertGreater(gain_boresight, 0)
    
    def test_free_space_path_loss(self):
        """Test free-space path loss calculation."""
        distance = 1000e3  # 1000 km
        path_loss = self.channel_model.free_space_path_loss(distance)
        
        # Path loss should be positive and reasonable for 12 GHz
        self.assertGreater(path_loss, 150)  # At least 150 dB
        self.assertLess(path_loss, 200)  # Less than 200 dB
    
    def test_ntn_path_loss(self):
        """Test NTN path loss calculation."""
        distance = 1000e3  # 1000 km
        elevation = 45.0  # 45 degrees
        
        path_loss = self.channel_model.ntn_path_loss(distance, elevation)
        
        self.assertGreater(path_loss, 0)
        self.assertIsInstance(path_loss, (float, np.floating))
    
    def test_rain_attenuation(self):
        """Test rain attenuation calculation."""
        # No rain
        no_rain = self.channel_model.rain_attenuation(0.0, 45.0)
        self.assertEqual(no_rain, 0.0)
        
        # Light rain
        light_rain = self.channel_model.rain_attenuation(5.0, 45.0)
        self.assertGreater(light_rain, 0)
        
        # Heavy rain
        heavy_rain = self.channel_model.rain_attenuation(25.0, 45.0)
        self.assertGreater(heavy_rain, light_rain)
    
    def test_link_budget(self):
        """Test complete link budget calculation."""
        # Create geometry
        sat_position = np.array([7000e3, 0, 0])  # 7000 km altitude
        sat_velocity = np.array([0, 7500, 0])  # ~7.5 km/s
        dt = datetime(2024, 1, 1, 0, 0, 0)
        
        geometry = self.geometry.compute_geometry(sat_position, sat_velocity, dt)
        
        # Compute link budget
        link_budget = self.channel_model.compute_link_budget(
            geometry,
            rain_rate_mmh=0.0,
            tx_power_dbm=40.0,
            bandwidth_hz=100e6
        )
        
        # Check required keys
        required_keys = [
            'tx_power_dbm', 'rx_power_dbm', 'path_loss_db',
            'snr_db', 'capacity_bps', 'elevation_deg'
        ]
        for key in required_keys:
            self.assertIn(key, link_budget)
        
        # Check reasonable values
        self.assertGreater(link_budget['snr_db'], -20)  # SNR should be reasonable
        self.assertGreater(link_budget['capacity_bps'], 0)
    
    def test_sinr_calculation(self):
        """Test SINR calculation."""
        desired_link = {
            'rx_power_dbm': -100.0
        }
        interference_links = [
            {'rx_power_dbm': -110.0},
            {'rx_power_dbm': -115.0}
        ]
        noise_power_dbm = -120.0
        
        sinr = self.channel_model.compute_sinr(
            desired_link, interference_links, noise_power_dbm
        )
        
        self.assertGreater(sinr, 0)
        self.assertIsInstance(sinr, (float, np.floating))


class TestChannelModelSionna(unittest.TestCase):
    """Test Sionna integration (requires Sionna to be installed)."""
    
    def setUp(self):
        """Set up test fixtures."""
        try:
            import tensorflow as tf
            import sionna
            self.sionna_available = True
            self.channel_model = ChannelModel(use_sionna=True)
        except ImportError:
            self.sionna_available = False
            self.skipTest("Sionna not available")
    
    def test_sionna_channel_application(self):
        """Test applying Sionna channel."""
        if not self.sionna_available:
            self.skipTest("Sionna not available")
        
        import tensorflow as tf
        
        # Create test signal
        signal = tf.random.normal([100, 64], dtype=tf.complex64)
        snr_db = 10.0
        
        # Apply channel
        received = self.channel_model.apply_channel_sionna(signal, snr_db)
        
        # Check output shape
        self.assertEqual(received.shape, signal.shape)
        self.assertEqual(received.dtype, signal.dtype)


if __name__ == '__main__':
    unittest.main()

