"""
Tests for channel model (TR38.811 NTN with OpenNTN + Sionna).
"""

import pytest
import numpy as np
from src.channel.channel_model import ChannelModel


class TestChannelModel:
    """Test cases for ChannelModel class."""
    
    def test_channel_model_initialization(self):
        """Test channel model initialization."""
        model = ChannelModel()
        
        assert model is not None
        assert hasattr(model, 'free_space_path_loss')
        assert hasattr(model, 'ntn_path_loss')
        assert hasattr(model, 'antenna_gain_tx')
        assert hasattr(model, 'antenna_gain_rx')
    
    def test_free_space_path_loss(self):
        """Test free space path loss calculation."""
        model = ChannelModel()
        distance = 1000e3  # 1000 km
        
        path_loss = model.free_space_path_loss(distance)
        
        assert isinstance(path_loss, float)
        assert path_loss > 0
        assert path_loss < 200  # Reasonable range for 1000 km at 12 GHz
    
    def test_ntn_path_loss(self):
        """Test TR38.811 NTN path loss calculation."""
        model = ChannelModel()
        distance = 1000e3  # 1000 km
        elevation = 30.0  # 30 degrees
        
        path_loss = model.ntn_path_loss(distance, elevation)
        
        assert isinstance(path_loss, float)
        assert path_loss > 0
    
    def test_atmospheric_loss(self):
        """Test atmospheric loss calculation."""
        model = ChannelModel()
        elevation = 30.0  # 30 degrees
        
        loss = model.atmospheric_loss(elevation)
        
        assert isinstance(loss, float)
        assert loss >= 0
    
    def test_atmospheric_loss_low_elevation(self):
        """Test atmospheric loss for very low elevation angles."""
        model = ChannelModel()
        elevation = 2.0  # Very low elevation
        
        loss = model.atmospheric_loss(elevation)
        
        # Should return inf for very low elevations (< 5 deg)
        assert loss == np.inf or loss > 100
    
    def test_rain_attenuation(self):
        """Test rain attenuation calculation (ITU-R P.838)."""
        model = ChannelModel()
        rain_rate = 10.0  # 10 mm/h
        elevation = 30.0
        
        rain_loss = model.rain_attenuation(rain_rate, elevation)
        
        assert isinstance(rain_loss, float)
        assert rain_loss >= 0
    
    def test_antenna_gain_tx(self):
        """Test satellite transmit antenna gain G_tx(theta)."""
        model = ChannelModel()
        
        # Boresight
        gain_0 = model.antenna_gain_tx(0.0)
        assert gain_0 == model.sat_antenna_gain
        
        # Off-boresight
        gain_off = model.antenna_gain_tx(10.0)
        assert gain_off < gain_0
    
    def test_antenna_gain_rx(self):
        """Test ground station receive antenna gain G_rx(theta)."""
        model = ChannelModel()
        
        # Boresight
        gain_0 = model.antenna_gain_rx(0.0)
        assert gain_0 == model.gs_antenna_gain
        
        # Off-boresight
        gain_off = model.antenna_gain_rx(10.0)
        assert gain_off < gain_0
    
    def test_shadowing_loss(self):
        """Test shadowing loss calculation."""
        model = ChannelModel()
        elevation = 30.0
        
        # LOS shadowing
        shadowing_los = model.shadowing_loss(elevation, 'los')
        assert isinstance(shadowing_los, float)
        
        # NLOS shadowing (should be higher)
        shadowing_nlos = model.shadowing_loss(elevation, 'nlos')
        assert shadowing_nlos > shadowing_los

