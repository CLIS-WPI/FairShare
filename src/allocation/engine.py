"""
Resource Allocation Engine for Multi-Operator LEO Spectrum Sharing.

Handles dynamic spectrum allocation across multiple operators,
supporting various allocation policies (static, priority, RL-based).
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass
from enum import Enum

from src.operators import Operator, SpectrumBandManager


class AllocationPolicy(Enum):
    """Types of allocation policies."""
    STATIC_EQUAL = "static_equal"
    STATIC_PROPORTIONAL = "static_proportional"
    PRIORITY_BASED = "priority_based"
    RL_AGENT = "rl_agent"
    HYBRID = "hybrid"


@dataclass
class AllocationRequest:
    """Request for spectrum allocation."""
    operator_id: str
    user_id: str
    demand_mbps: float
    priority: float = 1.0
    min_bandwidth_mhz: float = 0.0
    max_bandwidth_mhz: float = 1000.0
    qos_requirements: Optional[Dict[str, float]] = None


@dataclass
class AllocationResult:
    """Result of spectrum allocation."""
    operator_id: str
    user_id: str
    allocated_bandwidth_mhz: float
    allocated_frequency_mhz: float  # Center frequency
    success: bool
    reason: str = ""


class AllocationEngine:
    """
    Core resource allocation engine for multi-operator scenarios.
    
    Supports multiple allocation policies and tracks resource usage
    per operator and per user.
    """
    
    def __init__(
        self,
        spectrum_manager: SpectrumBandManager,
        policy: AllocationPolicy = AllocationPolicy.STATIC_EQUAL
    ):
        """
        Initialize allocation engine.
        
        Args:
            spectrum_manager: Spectrum band manager
            policy: Allocation policy to use
        """
        self.spectrum_manager = spectrum_manager
        self.policy = policy
        
        # Resource tracking
        self.operator_allocations: Dict[str, float] = {}  # MHz per operator
        self.user_allocations: Dict[str, AllocationResult] = {}
        self.operator_demands: Dict[str, float] = {}  # Total demand per operator
        
        # RL agent (if using RL policy)
        self.rl_agent = None
    
    def set_rl_agent(self, agent):
        """Set RL agent for RL-based allocation."""
        if self.policy == AllocationPolicy.RL_AGENT:
            self.rl_agent = agent
        else:
            raise ValueError("RL agent can only be set for RL_AGENT policy")
    
    def allocate(
        self,
        requests: List[AllocationRequest],
        available_bandwidth_mhz: Optional[float] = None
    ) -> List[AllocationResult]:
        """
        Allocate spectrum resources to requests.
        
        Args:
            requests: List of allocation requests
            available_bandwidth_mhz: Available bandwidth (None = use all)
            
        Returns:
            List of allocation results
        """
        if available_bandwidth_mhz is None:
            available_bandwidth_mhz = self.spectrum_manager.get_available_spectrum()
        
        # Group requests by operator
        operator_requests: Dict[str, List[AllocationRequest]] = {}
        for req in requests:
            if req.operator_id not in operator_requests:
                operator_requests[req.operator_id] = []
            operator_requests[req.operator_id].append(req)
        
        # Allocate based on policy
        if self.policy == AllocationPolicy.STATIC_EQUAL:
            return self._allocate_static_equal(operator_requests, available_bandwidth_mhz)
        elif self.policy == AllocationPolicy.STATIC_PROPORTIONAL:
            return self._allocate_static_proportional(operator_requests, available_bandwidth_mhz)
        elif self.policy == AllocationPolicy.PRIORITY_BASED:
            return self._allocate_priority_based(requests, available_bandwidth_mhz)
        elif self.policy == AllocationPolicy.RL_AGENT:
            return self._allocate_rl_agent(requests, available_bandwidth_mhz)
        else:
            raise ValueError(f"Unknown policy: {self.policy}")
    
    def _allocate_static_equal(
        self,
        operator_requests: Dict[str, List[AllocationRequest]],
        available_bandwidth_mhz: float
    ) -> List[AllocationResult]:
        """Allocate equal bandwidth to each operator."""
        num_operators = len(operator_requests)
        if num_operators == 0:
            return []
        
        bandwidth_per_operator = available_bandwidth_mhz / num_operators
        results = []
        
        for operator_id, requests in operator_requests.items():
            # Get operator's spectrum bands
            operator_bands = self.spectrum_manager.get_operator_bands(operator_id)
            if not operator_bands:
                # No spectrum assigned to this operator
                for req in requests:
                    results.append(AllocationResult(
                        operator_id=operator_id,
                        user_id=req.user_id,
                        allocated_bandwidth_mhz=0.0,
                        allocated_frequency_mhz=0.0,
                        success=False,
                        reason="No spectrum assigned to operator"
                    ))
                continue
            
            # Distribute bandwidth among users
            num_users = len(requests)
            if num_users == 0:
                continue
            
            bandwidth_per_user = bandwidth_per_operator / num_users
            
            for req in requests:
                # Find available frequency in operator's bands
                allocated_freq = self._find_available_frequency(
                    operator_bands, bandwidth_per_user
                )
                
                if allocated_freq > 0:
                    results.append(AllocationResult(
                        operator_id=operator_id,
                        user_id=req.user_id,
                        allocated_bandwidth_mhz=bandwidth_per_user,
                        allocated_frequency_mhz=allocated_freq,
                        success=True,
                        reason="Static equal allocation"
                    ))
                    self.operator_allocations[operator_id] = \
                        self.operator_allocations.get(operator_id, 0.0) + bandwidth_per_user
                else:
                    results.append(AllocationResult(
                        operator_id=operator_id,
                        user_id=req.user_id,
                        allocated_bandwidth_mhz=0.0,
                        allocated_frequency_mhz=0.0,
                        success=False,
                        reason="No available frequency"
                    ))
        
        return results
    
    def _allocate_static_proportional(
        self,
        operator_requests: Dict[str, List[AllocationRequest]],
        available_bandwidth_mhz: float
    ) -> List[AllocationResult]:
        """Allocate bandwidth proportional to operator's total demand."""
        # Calculate total demand per operator
        operator_demands = {}
        total_demand = 0.0
        
        for operator_id, requests in operator_requests.items():
            demand = sum(req.demand_mbps for req in requests)
            operator_demands[operator_id] = demand
            total_demand += demand
        
        if total_demand == 0:
            return []
        
        results = []
        
        for operator_id, requests in operator_requests.items():
            # Proportional allocation
            operator_share = operator_demands[operator_id] / total_demand
            operator_bandwidth = available_bandwidth_mhz * operator_share
            
            # Get operator's spectrum bands
            operator_bands = self.spectrum_manager.get_operator_bands(operator_id)
            if not operator_bands:
                for req in requests:
                    results.append(AllocationResult(
                        operator_id=operator_id,
                        user_id=req.user_id,
                        allocated_bandwidth_mhz=0.0,
                        allocated_frequency_mhz=0.0,
                        success=False,
                        reason="No spectrum assigned to operator"
                    ))
                continue
            
            # Distribute among users (proportional to demand)
            user_demands = {req.user_id: req.demand_mbps for req in requests}
            total_user_demand = sum(user_demands.values())
            
            for req in requests:
                if total_user_demand > 0:
                    user_share = user_demands[req.user_id] / total_user_demand
                    user_bandwidth = operator_bandwidth * user_share
                else:
                    user_bandwidth = operator_bandwidth / len(requests)
                
                allocated_freq = self._find_available_frequency(
                    operator_bands, user_bandwidth
                )
                
                if allocated_freq > 0:
                    results.append(AllocationResult(
                        operator_id=operator_id,
                        user_id=req.user_id,
                        allocated_bandwidth_mhz=user_bandwidth,
                        allocated_frequency_mhz=allocated_freq,
                        success=True,
                        reason="Static proportional allocation"
                    ))
                    self.operator_allocations[operator_id] = \
                        self.operator_allocations.get(operator_id, 0.0) + user_bandwidth
                else:
                    results.append(AllocationResult(
                        operator_id=operator_id,
                        user_id=req.user_id,
                        allocated_bandwidth_mhz=0.0,
                        allocated_frequency_mhz=0.0,
                        success=False,
                        reason="No available frequency"
                    ))
        
        return results
    
    def _allocate_priority_based(
        self,
        requests: List[AllocationRequest],
        available_bandwidth_mhz: float
    ) -> List[AllocationResult]:
        """Allocate based on priority (higher priority first)."""
        # Sort by priority (descending)
        sorted_requests = sorted(
            requests,
            key=lambda r: (r.priority, r.demand_mbps),
            reverse=True
        )
        
        results = []
        remaining_bandwidth = available_bandwidth_mhz
        
        for req in sorted_requests:
            operator_bands = self.spectrum_manager.get_operator_bands(req.operator_id)
            if not operator_bands:
                results.append(AllocationResult(
                    operator_id=req.operator_id,
                    user_id=req.user_id,
                    allocated_bandwidth_mhz=0.0,
                    allocated_frequency_mhz=0.0,
                    success=False,
                    reason="No spectrum assigned to operator"
                ))
                continue
            
            # Allocate based on demand, but respect available bandwidth
            allocated_bandwidth = min(
                req.demand_mbps / 10.0,  # Convert Mbps to MHz (simplified)
                remaining_bandwidth,
                req.max_bandwidth_mhz
            )
            
            if allocated_bandwidth >= req.min_bandwidth_mhz:
                allocated_freq = self._find_available_frequency(
                    operator_bands, allocated_bandwidth
                )
                
                if allocated_freq > 0:
                    results.append(AllocationResult(
                        operator_id=req.operator_id,
                        user_id=req.user_id,
                        allocated_bandwidth_mhz=allocated_bandwidth,
                        allocated_frequency_mhz=allocated_freq,
                        success=True,
                        reason="Priority-based allocation"
                    ))
                    remaining_bandwidth -= allocated_bandwidth
                    self.operator_allocations[req.operator_id] = \
                        self.operator_allocations.get(req.operator_id, 0.0) + allocated_bandwidth
                else:
                    results.append(AllocationResult(
                        operator_id=req.operator_id,
                        user_id=req.user_id,
                        allocated_bandwidth_mhz=0.0,
                        allocated_frequency_mhz=0.0,
                        success=False,
                        reason="No available frequency"
                    ))
            else:
                results.append(AllocationResult(
                    operator_id=req.operator_id,
                    user_id=req.user_id,
                    allocated_bandwidth_mhz=0.0,
                    allocated_frequency_mhz=0.0,
                    success=False,
                    reason="Insufficient bandwidth"
                ))
        
        return results
    
    def _allocate_rl_agent(
        self,
        requests: List[AllocationRequest],
        available_bandwidth_mhz: float
    ) -> List[AllocationResult]:
        """Allocate using RL agent."""
        if self.rl_agent is None:
            raise ValueError("RL agent not set")
        
        # Convert requests to state representation
        state = self._requests_to_state(requests, available_bandwidth_mhz)
        
        # Get action from RL agent
        action = self.rl_agent.predict(state)
        
        # Convert action to allocation results
        return self._action_to_allocations(action, requests)
    
    def _find_available_frequency(
        self,
        bands: List,
        required_bandwidth_mhz: float
    ) -> float:
        """
        Find available frequency in operator's bands.
        
        Args:
            bands: List of SpectrumBand objects
            required_bandwidth_mhz: Required bandwidth
            
        Returns:
            Center frequency in MHz, or 0.0 if not available
        """
        for band in bands:
            bandwidth = band.get_bandwidth_mhz()
            if bandwidth >= required_bandwidth_mhz:
                # Return center frequency
                return (band.start_freq_mhz + band.end_freq_mhz) / 2.0
        
        return 0.0
    
    def _requests_to_state(
        self,
        requests: List[AllocationRequest],
        available_bandwidth_mhz: float
    ) -> np.ndarray:
        """Convert requests to state representation for RL."""
        # This is a placeholder - will be implemented with RL module
        # State: [num_requests, total_demand, available_bandwidth, ...]
        num_requests = len(requests)
        total_demand = sum(req.demand_mbps for req in requests)
        
        return np.array([
            num_requests,
            total_demand,
            available_bandwidth_mhz
        ])
    
    def _action_to_allocations(
        self,
        action: np.ndarray,
        requests: List[AllocationRequest]
    ) -> List[AllocationResult]:
        """Convert RL action to allocation results."""
        # Placeholder - will be implemented with RL module
        results = []
        for i, req in enumerate(requests):
            if i < len(action):
                allocated_bandwidth = action[i] * req.max_bandwidth_mhz
                results.append(AllocationResult(
                    operator_id=req.operator_id,
                    user_id=req.user_id,
                    allocated_bandwidth_mhz=allocated_bandwidth,
                    allocated_frequency_mhz=0.0,  # Will be determined by RL
                    success=allocated_bandwidth > 0,
                    reason="RL agent allocation"
                ))
        return results
    
    def get_operator_utilization(self, operator_id: str) -> float:
        """Get utilization for an operator (0.0 to 1.0)."""
        total_bandwidth = self.spectrum_manager.get_operator_total_bandwidth(operator_id)
        if total_bandwidth == 0:
            return 0.0
        allocated = self.operator_allocations.get(operator_id, 0.0)
        return min(1.0, allocated / total_bandwidth)
    
    def reset(self):
        """Reset allocation state."""
        self.operator_allocations.clear()
        self.user_allocations.clear()
        self.operator_demands.clear()

