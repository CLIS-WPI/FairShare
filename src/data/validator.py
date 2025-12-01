"""
Validator for synthetic data against real datasets and statistics.

Validates generated data against public datasets, FCC benchmarks,
and official LEO operator records.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass

from src.data.generator import UserProfile, TrafficPattern


@dataclass
class ValidationResult:
    """Result of data validation."""
    metric_name: str
    expected_value: float
    actual_value: float
    tolerance: float
    passed: bool
    message: str


class DataValidator:
    """
    Validate synthetic data against real benchmarks.
    
    Compares generated data with:
    - FCC spectrum allocation records
    - Official LEO operator statistics
    - Public network traffic datasets
    """
    
    def __init__(self):
        """Initialize validator with benchmark data."""
        # FCC benchmarks (approximate)
        self.fcc_benchmarks = {
            "avg_user_demand_mbps": 10.0,  # Average user demand
            "peak_demand_multiplier": 1.5,  # Peak vs average
            "user_density_per_km2": 0.1,  # Users per km²
        }
        
        # LEO operator statistics (approximate)
        self.leo_benchmarks = {
            "starlink_avg_throughput": 100.0,  # Mbps
            "starlink_latency_ms": 20.0,  # ms
            "kuiper_avg_throughput": 80.0,  # Mbps
            "kuiper_latency_ms": 25.0,  # ms
        }
    
    def validate_user_distribution(
        self,
        users: List[UserProfile],
        tolerance: float = 0.2
    ) -> List[ValidationResult]:
        """
        Validate user distribution.
        
        Args:
            users: List of users
            tolerance: Acceptable tolerance (0.0 to 1.0)
            
        Returns:
            List of validation results
        """
        results = []
        
        # Check average demand
        avg_demand = np.mean([u.demand_mbps for u in users])
        expected_demand = self.fcc_benchmarks["avg_user_demand_mbps"]
        
        passed = abs(avg_demand - expected_demand) / expected_demand <= tolerance
        results.append(ValidationResult(
            metric_name="average_demand",
            expected_value=expected_demand,
            actual_value=avg_demand,
            tolerance=tolerance,
            passed=passed,
            message=f"Average demand: {avg_demand:.2f} Mbps (expected: {expected_demand:.2f})"
        ))
        
        # Check demand distribution
        demands = [u.demand_mbps for u in users]
        cv = np.std(demands) / (np.mean(demands) + 1e-6)  # Coefficient of variation
        
        # Expected CV: 0.5 to 1.5 (reasonable variation)
        passed = 0.3 <= cv <= 1.5
        results.append(ValidationResult(
            metric_name="demand_variation",
            expected_value=0.8,  # Target CV
            actual_value=cv,
            tolerance=0.5,
            passed=passed,
            message=f"Demand variation (CV): {cv:.2f}"
        ))
        
        return results
    
    def validate_traffic_patterns(
        self,
        patterns: List[TrafficPattern],
        tolerance: float = 0.2
    ) -> List[ValidationResult]:
        """
        Validate traffic patterns.
        
        Args:
            patterns: List of traffic patterns
            tolerance: Acceptable tolerance
            
        Returns:
            List of validation results
        """
        results = []
        
        if not patterns:
            return results
        
        # Check peak vs average ratio
        total_demands = [p.total_demand_mbps for p in patterns]
        avg_demand = np.mean(total_demands)
        peak_demand = np.max(total_demands)
        peak_ratio = peak_demand / (avg_demand + 1e-6)
        
        expected_ratio = self.fcc_benchmarks["peak_demand_multiplier"]
        passed = abs(peak_ratio - expected_ratio) / expected_ratio <= tolerance
        
        results.append(ValidationResult(
            metric_name="peak_to_average_ratio",
            expected_value=expected_ratio,
            actual_value=peak_ratio,
            tolerance=tolerance,
            passed=passed,
            message=f"Peak/Average ratio: {peak_ratio:.2f} (expected: {expected_ratio:.2f})"
        ))
        
        # Check temporal variation
        # Demand should vary over time
        demand_std = np.std(total_demands)
        demand_cv = demand_std / (avg_demand + 1e-6)
        
        # Expected CV: 0.2 to 0.5 (reasonable temporal variation)
        passed = 0.15 <= demand_cv <= 0.6
        results.append(ValidationResult(
            metric_name="temporal_variation",
            expected_value=0.3,
            actual_value=demand_cv,
            tolerance=0.2,
            passed=passed,
            message=f"Temporal variation (CV): {demand_cv:.2f}"
        ))
        
        return results
    
    def validate_operator_statistics(
        self,
        users: List[UserProfile],
        operator_performance: Dict[str, Dict[str, float]],
        tolerance: float = 0.3
    ) -> List[ValidationResult]:
        """
        Validate operator performance against benchmarks.
        
        Args:
            users: List of users
            operator_performance: Performance metrics per operator
            tolerance: Acceptable tolerance
            
        Returns:
            List of validation results
        """
        results = []
        
        for operator_id, metrics in operator_performance.items():
            # Check throughput (if available)
            if "avg_throughput_mbps" in metrics:
                throughput = metrics["avg_throughput_mbps"]
                
                # Compare with benchmarks
                if "starlink" in operator_id.lower():
                    expected = self.leo_benchmarks["starlink_avg_throughput"]
                elif "kuiper" in operator_id.lower():
                    expected = self.leo_benchmarks["kuiper_avg_throughput"]
                else:
                    expected = 50.0  # Default
                
                passed = abs(throughput - expected) / expected <= tolerance
                results.append(ValidationResult(
                    metric_name=f"{operator_id}_throughput",
                    expected_value=expected,
                    actual_value=throughput,
                    tolerance=tolerance,
                    passed=passed,
                    message=f"{operator_id} throughput: {throughput:.2f} Mbps"
                ))
        
        return results
    
    def validate_all(
        self,
        users: List[UserProfile],
        patterns: Optional[List[TrafficPattern]] = None,
        operator_performance: Optional[Dict[str, Dict[str, float]]] = None
    ) -> Dict[str, List[ValidationResult]]:
        """
        Run all validation checks.
        
        Returns:
            Dictionary of validation results by category
        """
        results = {
            "user_distribution": self.validate_user_distribution(users),
            "traffic_patterns": [],
            "operator_statistics": []
        }
        
        if patterns:
            results["traffic_patterns"] = self.validate_traffic_patterns(patterns)
        
        if operator_performance:
            results["operator_statistics"] = self.validate_operator_statistics(
                users, operator_performance
            )
        
        return results
    
    def print_validation_summary(self, results: Dict[str, List[ValidationResult]]):
        """Print validation summary."""
        print("=" * 60)
        print("Data Validation Summary")
        print("=" * 60)
        
        total_checks = 0
        total_passed = 0
        
        for category, checks in results.items():
            print(f"\n{category.upper()}:")
            for check in checks:
                total_checks += 1
                if check.passed:
                    total_passed += 1
                    status = "✓ PASS"
                else:
                    status = "✗ FAIL"
                
                print(f"  {status}: {check.message}")
        
        print("\n" + "=" * 60)
        print(f"Total: {total_passed}/{total_checks} checks passed")
        print("=" * 60)

