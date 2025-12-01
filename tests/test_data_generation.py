"""
Unit tests for data generation module.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta

from src.data.generator import (
    SyntheticDataGenerator, UserProfile, TrafficPattern
)
from src.data.validator import DataValidator, ValidationResult


class TestSyntheticDataGenerator:
    """Tests for synthetic data generator."""
    
    def test_generator_creation(self):
        """Test generator creation."""
        generator = SyntheticDataGenerator()
        assert generator.region_bounds is not None
        assert generator.population_density > 0
    
    def test_generate_user_positions(self):
        """Test user position generation."""
        generator = SyntheticDataGenerator()
        
        positions = generator.generate_user_positions(10, distribution="uniform")
        
        assert len(positions) == 10
        for lat, lon in positions:
            assert isinstance(lat, float)
            assert isinstance(lon, float)
    
    def test_generate_users(self):
        """Test user generation."""
        generator = SyntheticDataGenerator()
        
        operators = ["starlink", "kuiper"]
        users = generator.generate_users(20, operators)
        
        assert len(users) == 20
        for user in users:
            assert isinstance(user, UserProfile)
            assert user.operator_id in operators
            assert user.demand_mbps > 0
            assert 0.0 <= user.priority <= 1.0
    
    def test_generate_traffic_pattern(self):
        """Test traffic pattern generation."""
        generator = SyntheticDataGenerator()
        
        users = generator.generate_users(10, ["starlink"])
        timestamp = datetime.now()
        
        pattern = generator.generate_traffic_pattern(users, timestamp)
        
        assert isinstance(pattern, TrafficPattern)
        assert pattern.timestamp == timestamp
        assert len(pattern.user_demands) == len(users)
        assert pattern.total_demand_mbps >= 0
    
    def test_generate_traffic_timeline(self):
        """Test traffic timeline generation."""
        generator = SyntheticDataGenerator()
        
        users = generator.generate_users(10, ["starlink"])
        start_time = datetime.now()
        
        patterns = generator.generate_traffic_timeline(
            users, start_time, duration_hours=1.0, interval_minutes=10
        )
        
        assert len(patterns) == 6  # 1 hour / 10 minutes = 6 intervals
        assert all(isinstance(p, TrafficPattern) for p in patterns)


class TestDataValidator:
    """Tests for data validator."""
    
    def test_validator_creation(self):
        """Test validator creation."""
        validator = DataValidator()
        assert validator.fcc_benchmarks is not None
        assert validator.leo_benchmarks is not None
    
    def test_validate_user_distribution(self):
        """Test user distribution validation."""
        validator = DataValidator()
        
        generator = SyntheticDataGenerator()
        users = generator.generate_users(100, ["starlink"])
        
        results = validator.validate_user_distribution(users)
        
        assert len(results) > 0
        assert all(isinstance(r, ValidationResult) for r in results)
    
    def test_validate_traffic_patterns(self):
        """Test traffic pattern validation."""
        validator = DataValidator()
        
        generator = SyntheticDataGenerator()
        users = generator.generate_users(50, ["starlink"])
        patterns = generator.generate_traffic_timeline(
            users, datetime.now(), duration_hours=2.0
        )
        
        results = validator.validate_traffic_patterns(patterns)
        
        assert len(results) > 0
        assert all(isinstance(r, ValidationResult) for r in results)
    
    def test_validate_all(self):
        """Test complete validation."""
        validator = DataValidator()
        
        generator = SyntheticDataGenerator()
        users = generator.generate_users(100, ["starlink", "kuiper"])
        patterns = generator.generate_traffic_timeline(
            users, datetime.now(), duration_hours=1.0
        )
        
        operator_performance = {
            "starlink": {"avg_throughput_mbps": 95.0},
            "kuiper": {"avg_throughput_mbps": 75.0}
        }
        
        results = validator.validate_all(users, patterns, operator_performance)
        
        assert "user_distribution" in results
        assert "traffic_patterns" in results
        assert "operator_statistics" in results

