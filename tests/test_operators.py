"""
Unit tests for operators module.
"""

import pytest
import numpy as np
from datetime import datetime

from src.operators import (
    Operator, OperatorConfig, OperatorType,
    Constellation, SatelliteState,
    SpectrumBand, SpectrumBandManager, BandType
)


class TestOperator:
    """Tests for Operator class."""
    
    def test_operator_creation(self):
        """Test operator creation."""
        config = OperatorConfig(
            name="Starlink",
            operator_type=OperatorType.STARLINK,
            constellation_altitude_km=550.0,
            constellation_inclination_deg=53.0,
            num_satellites=100,
            num_planes=10,
            satellites_per_plane=10,
            spectrum_bands_mhz=[(10000.0, 12000.0), (15000.0, 17000.0)]
        )
        
        operator = Operator(config)
        assert operator.name == "Starlink"
        assert operator.operator_type == OperatorType.STARLINK
        assert operator.get_total_spectrum_mhz() == 4000.0  # 2000 + 2000
    
    def test_operator_resource_tracking(self):
        """Test resource tracking."""
        config = OperatorConfig(
            name="Test",
            operator_type=OperatorType.CUSTOM,
            constellation_altitude_km=500.0,
            constellation_inclination_deg=45.0,
            num_satellites=50,
            num_planes=5,
            satellites_per_plane=10
        )
        
        operator = Operator(config)
        operator.update_resource_usage(
            allocated_bandwidth_mhz=100.0,
            served_users=10,
            throughput_mbps=50.0,
            latency_ms=20.0
        )
        
        assert operator.allocated_bandwidth_mhz == 100.0
        assert operator.served_users == 10
        assert operator.throughput_mbps == 50.0
    
    def test_operator_utilization(self):
        """Test utilization calculation."""
        config = OperatorConfig(
            name="Test",
            operator_type=OperatorType.CUSTOM,
            constellation_altitude_km=500.0,
            constellation_inclination_deg=45.0,
            num_satellites=50,
            num_planes=5,
            satellites_per_plane=10,
            spectrum_bands_mhz=[(10000.0, 11000.0)]  # 1000 MHz
        )
        
        operator = Operator(config)
        operator.update_resource_usage(50.0, 5, 25.0, 15.0)
        
        utilization = operator.get_utilization()
        assert 0.0 <= utilization <= 1.0
        assert abs(utilization - 0.05) < 0.01  # 50/1000


class TestConstellation:
    """Tests for Constellation class."""
    
    def test_constellation_creation(self):
        """Test constellation creation."""
        constellation = Constellation(
            operator_name="Test",
            altitude_km=550.0,
            inclination_deg=53.0,
            num_satellites=100,
            num_planes=10,
            satellites_per_plane=10
        )
        
        assert constellation.num_satellites == 100
        assert len(constellation.satellites) == 100
        assert len(constellation.satellite_ids) == 100
    
    def test_constellation_propagation(self):
        """Test constellation propagation."""
        constellation = Constellation(
            operator_name="Test",
            altitude_km=550.0,
            inclination_deg=53.0,
            num_satellites=10,
            num_planes=2,
            satellites_per_plane=5
        )
        
        timestamp = datetime.now()
        constellation.propagate(timestamp)
        
        for sat in constellation.satellites:
            assert sat.timestamp == timestamp
    
    def test_constellation_coverage(self):
        """Test coverage calculation."""
        constellation = Constellation(
            operator_name="Test",
            altitude_km=550.0,
            inclination_deg=53.0,
            num_satellites=10,
            num_planes=2,
            satellites_per_plane=5
        )
        
        # Mock user positions (ECEF coordinates)
        user_positions = np.array([
            [6371000.0, 0.0, 0.0],  # User 1
            [0.0, 6371000.0, 0.0],  # User 2
        ])
        
        timestamp = datetime.now()
        coverage = constellation.get_coverage(user_positions, timestamp)
        
        assert isinstance(coverage, dict)
        assert len(coverage) == len(constellation.satellites)


class TestSpectrumBands:
    """Tests for SpectrumBand and SpectrumBandManager."""
    
    def test_spectrum_band_creation(self):
        """Test spectrum band creation."""
        band = SpectrumBand(
            start_freq_mhz=10000.0,
            end_freq_mhz=12000.0,
            band_type=BandType.KA_BAND,
            operator_id="starlink"
        )
        
        assert band.get_bandwidth_mhz() == 2000.0
        assert band.operator_id == "starlink"
    
    def test_spectrum_band_overlap(self):
        """Test band overlap detection."""
        band1 = SpectrumBand(10000.0, 12000.0, BandType.KA_BAND)
        band2 = SpectrumBand(11000.0, 13000.0, BandType.KA_BAND)
        band3 = SpectrumBand(13000.0, 15000.0, BandType.KA_BAND)
        
        assert band1.overlaps(band2) is True
        assert band1.overlaps(band3) is False
    
    def test_spectrum_manager_creation(self):
        """Test spectrum manager creation."""
        manager = SpectrumBandManager(
            total_spectrum_start_mhz=10000.0,
            total_spectrum_end_mhz=40000.0
        )
        
        assert manager.total_bandwidth_mhz == 30000.0
    
    def test_spectrum_manager_add_band(self):
        """Test adding bands to manager."""
        manager = SpectrumBandManager()
        
        band = manager.add_band(
            10000.0, 12000.0,
            BandType.KA_BAND,
            operator_id="starlink"
        )
        
        assert band.operator_id == "starlink"
        assert len(manager.bands) == 1
        assert "starlink" in manager.operator_bands
    
    def test_spectrum_manager_operator_bands(self):
        """Test getting operator bands."""
        manager = SpectrumBandManager()
        
        manager.assign_band_to_operator(10000.0, 12000.0, "starlink")
        manager.assign_band_to_operator(15000.0, 17000.0, "kuiper")
        
        starlink_bands = manager.get_operator_bands("starlink")
        assert len(starlink_bands) == 1
        assert starlink_bands[0].get_bandwidth_mhz() == 2000.0
        
        total_bw = manager.get_operator_total_bandwidth("starlink")
        assert total_bw == 2000.0
    
    def test_spectrum_manager_interference(self):
        """Test interference detection."""
        manager = SpectrumBandManager()
        
        manager.assign_band_to_operator(10000.0, 12000.0, "starlink")
        manager.assign_band_to_operator(13000.0, 15000.0, "kuiper")
        
        # No interference
        assert manager.check_interference("starlink", "kuiper") is False
        
        # Create new manager for overlapping test (to avoid ValueError)
        manager2 = SpectrumBandManager()
        manager2.assign_band_to_operator(10000.0, 12000.0, "starlink")
        # Try to add overlapping band - should raise error or be handled
        try:
            manager2.assign_band_to_operator(11500.0, 13500.0, "kuiper")
            # If no error, check interference
            assert manager2.check_interference("starlink", "kuiper") is True
        except ValueError:
            # Expected: overlapping bands not allowed for different operators
            pass

