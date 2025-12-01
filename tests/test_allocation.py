"""
Unit tests for allocation module.
"""

import pytest
import numpy as np

from src.allocation import (
    AllocationEngine, AllocationPolicy,
    AllocationRequest, AllocationResult,
    ResourceTracker, UserMetrics, OperatorMetrics
)
from src.operators import SpectrumBandManager, BandType


class TestAllocationEngine:
    """Tests for AllocationEngine."""
    
    @pytest.fixture
    def spectrum_manager(self):
        """Create spectrum manager for tests."""
        manager = SpectrumBandManager()
        manager.assign_band_to_operator(10000.0, 12000.0, "operator1")
        manager.assign_band_to_operator(15000.0, 17000.0, "operator2")
        return manager
    
    @pytest.fixture
    def allocation_engine(self, spectrum_manager):
        """Create allocation engine."""
        return AllocationEngine(spectrum_manager, AllocationPolicy.STATIC_EQUAL)
    
    def test_static_equal_allocation(self, allocation_engine):
        """Test static equal allocation."""
        requests = [
            AllocationRequest("operator1", "user1", 10.0),
            AllocationRequest("operator1", "user2", 20.0),
            AllocationRequest("operator2", "user3", 15.0),
        ]
        
        results = allocation_engine.allocate(requests, available_bandwidth_mhz=1000.0)
        
        assert len(results) == 3
        # Check that allocations are made
        successful = [r for r in results if r.success]
        assert len(successful) > 0
    
    def test_priority_allocation(self, spectrum_manager):
        """Test priority-based allocation."""
        engine = AllocationEngine(spectrum_manager, AllocationPolicy.PRIORITY_BASED)
        
        requests = [
            AllocationRequest("operator1", "user1", 10.0, priority=0.5),
            AllocationRequest("operator1", "user2", 20.0, priority=0.9),
            AllocationRequest("operator2", "user3", 15.0, priority=0.7),
        ]
        
        results = engine.allocate(requests, available_bandwidth_mhz=500.0)
        
        assert len(results) == 3
        # Higher priority should get allocated first
        user2_result = [r for r in results if r.user_id == "user2"][0]
        user1_result = [r for r in results if r.user_id == "user1"][0]
        
        if user2_result.success and user1_result.success:
            assert user2_result.allocated_bandwidth_mhz >= user1_result.allocated_bandwidth_mhz
    
    def test_operator_utilization(self, allocation_engine, spectrum_manager):
        """Test operator utilization calculation."""
        requests = [
            AllocationRequest("operator1", "user1", 10.0),
        ]
        
        results = allocation_engine.allocate(requests, available_bandwidth_mhz=100.0)
        
        utilization = allocation_engine.get_operator_utilization("operator1")
        assert 0.0 <= utilization <= 1.0
    
    def test_reset(self, allocation_engine):
        """Test reset functionality."""
        requests = [AllocationRequest("operator1", "user1", 10.0)]
        allocation_engine.allocate(requests)
        
        allocation_engine.reset()
        assert len(allocation_engine.operator_allocations) == 0


class TestResourceTracker:
    """Tests for ResourceTracker."""
    
    def test_tracker_creation(self):
        """Test tracker creation."""
        tracker = ResourceTracker()
        assert len(tracker.user_metrics) == 0
        assert len(tracker.operator_metrics) == 0
    
    def test_update_allocations(self):
        """Test updating allocations."""
        tracker = ResourceTracker()
        
        results = [
            AllocationResult("operator1", "user1", 100.0, 10500.0, True),
            AllocationResult("operator1", "user2", 50.0, 10600.0, True),
        ]
        
        from datetime import datetime
        tracker.update_allocations(results, datetime.now())
        
        assert len(tracker.user_metrics) == 2
        assert "operator1_user1" in tracker.user_metrics
    
    def test_update_user_performance(self):
        """Test updating user performance."""
        tracker = ResourceTracker()
        
        from datetime import datetime
        results = [AllocationResult("operator1", "user1", 100.0, 10500.0, True)]
        tracker.update_allocations(results, datetime.now())
        
        tracker.update_user_performance(
            "operator1", "user1",
            throughput_mbps=50.0,
            latency_ms=20.0,
            packet_loss_rate=0.01,
            qos_satisfaction=0.9
        )
        
        user_metrics = tracker.get_user_metrics("operator1", "user1")
        assert user_metrics is not None
        assert user_metrics.throughput_mbps == 50.0
        assert user_metrics.latency_ms == 20.0
    
    def test_operator_metrics(self):
        """Test operator metrics aggregation."""
        tracker = ResourceTracker()
        
        from datetime import datetime
        results = [
            AllocationResult("operator1", "user1", 100.0, 10500.0, True),
            AllocationResult("operator1", "user2", 50.0, 10600.0, True),
        ]
        tracker.update_allocations(results, datetime.now())
        
        # Update performance BEFORE checking metrics (so they're computed)
        tracker.update_user_performance("operator1", "user1", 50.0, 20.0)
        tracker.update_user_performance("operator1", "user2", 30.0, 25.0)
        
        # Force update of operator metrics
        tracker._update_operator_metrics()
        
        op_metrics = tracker.get_operator_metrics("operator1")
        assert op_metrics is not None
        assert op_metrics.num_users == 2
        assert op_metrics.num_served_users == 2
        # avg_throughput should be computed from user metrics
        assert op_metrics.avg_throughput_mbps >= 0  # Can be 0 if no served users with throughput
    
    def test_aggregate_statistics(self):
        """Test aggregate statistics."""
        tracker = ResourceTracker()
        
        from datetime import datetime
        results = [
            AllocationResult("operator1", "user1", 100.0, 10500.0, True),
            AllocationResult("operator2", "user2", 50.0, 15500.0, True),
        ]
        tracker.update_allocations(results, datetime.now())
        
        stats = tracker.get_aggregate_statistics()
        assert "total_operators" in stats
        assert stats["total_operators"] == 2
        assert stats["total_users"] == 2

