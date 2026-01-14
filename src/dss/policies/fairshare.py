"""
FairShare: Geographic-Aware Spectrum Allocation Policy

Ensures minimum allocation quota for each geographic region
while maintaining reasonable spectral efficiency.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


class FairSharePolicy:
    """
    Geographic-aware allocation that guarantees minimum access
    for rural/suburban users while considering channel quality.
    
    Key idea: 
    1. Reserve quota for each geographic region
    2. Within each region, allocate by channel quality
    3. Redistribute unused quota to other regions
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        Args:
            config: Dictionary with:
                - urban_quota: minimum fraction for urban (default: 0.40)
                - suburban_quota: minimum fraction for suburban (default: 0.25)
                - rural_quota: minimum fraction for rural (default: 0.35)
                - efficiency_weight: balance fairness vs efficiency (0-1, default: 0.3)
        """
        if config is None:
            config = {}
        self.urban_quota = config.get('urban_quota', 0.40)
        self.suburban_quota = config.get('suburban_quota', 0.25)
        self.rural_quota = config.get('rural_quota', 0.35)
        self.efficiency_weight = config.get('efficiency_weight', 0.3)
        
        # Validate quotas sum to 1
        total = self.urban_quota + self.suburban_quota + self.rural_quota
        if abs(total - 1.0) > 0.01:
            # Normalize if close
            scale = 1.0 / total
            self.urban_quota *= scale
            self.suburban_quota *= scale
            self.rural_quota *= scale
    
    def allocate(self, 
                 demands: np.ndarray,
                 available_resources: float,
                 priorities: Optional[np.ndarray] = None,
                 users: Optional[List[Dict]] = None,
                 link_budgets: Optional[Dict[str, Dict]] = None,
                 location_types: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Allocate spectrum with geographic fairness guarantees.
        
        Args:
            demands: [N] array of user demands
            available_resources: total bandwidth in Hz
            priorities: [N] array of priorities (optional, not used in FairShare)
            users: list of user dicts (optional, for location_type extraction)
            link_budgets: dict mapping user_id to link budget (optional, for SNR)
            location_types: [N] array of location types (0=urban, 1=suburban, 2=rural, 3=rural)
        
        Returns:
            allocations: [N] array of allocated bandwidth per user (Hz)
        """
        n_users = len(demands)
        allocations = np.zeros(n_users)
        
        # Extract location types if not provided
        if location_types is None and users is not None:
            # Try to get from users
            location_types = np.array([
                self._get_location_type(user) for user in users
            ])
        elif location_types is None:
            # Default: assume all urban (fallback)
            location_types = np.zeros(n_users, dtype=int)
        
        # Extract SNR values from link_budgets if available
        snr_values = np.zeros(n_users)
        if link_budgets is not None and users is not None:
            for i, user in enumerate(users):
                user_id = user.get('id', i)
                link_budget = link_budgets.get(user_id, {})
                snr_values[i] = link_budget.get('snr_db', 0.0)
        elif priorities is not None:
            # Use priorities as proxy for SNR if link_budgets not available
            snr_values = priorities * 10.0  # Scale to dB range
        
        # Step 1: Separate users by geographic type
        # Map: 0=dense_urban, 1=urban -> urban (0)
        #      2=suburban -> suburban (1)
        #      3=rural -> rural (2)
        urban_mask = (location_types == 0) | (location_types == 1)
        suburban_mask = location_types == 2
        rural_mask = location_types == 3
        
        urban_indices = np.where(urban_mask)[0]
        suburban_indices = np.where(suburban_mask)[0]
        rural_indices = np.where(rural_mask)[0]
        
        # Step 2: Calculate bandwidth quota for each region
        urban_bw = available_resources * self.urban_quota
        suburban_bw = available_resources * self.suburban_quota
        rural_bw = available_resources * self.rural_quota
        
        # Step 3: Calculate how many users to allocate per region based on quota
        # Total users to allocate (30% of all users for scarcity scenario)
        total_users_to_allocate = max(1, int(n_users * 0.30))
        
        # Calculate number of users per region based on quota
        urban_users_to_alloc = max(1, int(total_users_to_allocate * self.urban_quota))
        suburban_users_to_alloc = max(1, int(total_users_to_allocate * self.suburban_quota))
        rural_users_to_alloc = max(1, int(total_users_to_allocate * self.rural_quota))
        
        # Ensure we don't exceed available users in each region
        urban_users_to_alloc = min(urban_users_to_alloc, len(urban_indices))
        suburban_users_to_alloc = min(suburban_users_to_alloc, len(suburban_indices))
        rural_users_to_alloc = min(rural_users_to_alloc, len(rural_indices))
        
        # Step 4: Within each region, allocate by SNR (best users first)
        def allocate_region(indices, region_bw, n_users_to_alloc):
            """Allocate within a geographic region"""
            if len(indices) == 0 or n_users_to_alloc == 0:
                return np.array([], dtype=int), np.array([])
            
            # Sort by SNR (descending)
            region_snr = snr_values[indices]
            sorted_order = np.argsort(region_snr)[::-1]
            sorted_indices = indices[sorted_order]
            
            # Allocate to top N users by SNR
            allocated_indices = sorted_indices[:n_users_to_alloc]
            
            # Per-user bandwidth (equal within region)
            per_user_bw = region_bw / n_users_to_alloc
            allocated_bw = np.full(len(allocated_indices), per_user_bw)
            
            return allocated_indices, allocated_bw
        
        # Allocate each region
        urban_alloc_idx, urban_alloc_bw = allocate_region(
            urban_indices, urban_bw, urban_users_to_alloc
        )
        suburban_alloc_idx, suburban_alloc_bw = allocate_region(
            suburban_indices, suburban_bw, suburban_users_to_alloc
        )
        rural_alloc_idx, rural_alloc_bw = allocate_region(
            rural_indices, rural_bw, rural_users_to_alloc
        )
        
        # Step 4: Fill in allocations array
        for idx, bw in zip(urban_alloc_idx, urban_alloc_bw):
            allocations[idx] = bw
        for idx, bw in zip(suburban_alloc_idx, suburban_alloc_bw):
            allocations[idx] = bw
        for idx, bw in zip(rural_alloc_idx, rural_alloc_bw):
            allocations[idx] = bw
        
        # Ensure allocations don't exceed demands
        allocations = np.minimum(allocations, demands)
        
        return allocations
    
    def _get_location_type(self, user: Dict) -> int:
        """Extract location type from user dict"""
        # Try different possible keys
        if 'location_type' in user:
            loc_type = user['location_type']
            if isinstance(loc_type, str):
                # Map string to int
                mapping = {'dense_urban': 0, 'urban': 1, 'suburban': 2, 'rural': 3}
                return mapping.get(loc_type, 0)
            return int(loc_type)
        elif 'is_urban' in user:
            return 0 if user['is_urban'] else 3
        else:
            return 0  # Default to urban


class WeightedFairSharePolicy:
    """
    Alternative: Weight-based allocation that balances fairness and efficiency.
    
    Score = (1 - α) × FairnessScore + α × EfficiencyScore
    
    where:
    - FairnessScore = 1.0 for rural, 0.5 for suburban, 0.0 for urban
    - EfficiencyScore = normalized SNR
    - α = efficiency_weight (0 = pure fairness, 1 = pure efficiency)
    """
    
    def __init__(self, config: Optional[dict] = None):
        if config is None:
            config = {}
        self.efficiency_weight = config.get('efficiency_weight', 0.3)
        self.allocation_fraction = config.get('allocation_fraction', 0.30)
    
    def allocate(self,
                 demands: np.ndarray,
                 available_resources: float,
                 priorities: Optional[np.ndarray] = None,
                 users: Optional[List[Dict]] = None,
                 link_budgets: Optional[Dict[str, Dict]] = None,
                 location_types: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Allocate using weighted scoring.
        """
        n_users = len(demands)
        allocations = np.zeros(n_users)
        
        # Extract location types
        if location_types is None and users is not None:
            location_types = np.array([
                self._get_location_type(user) for user in users
            ])
        elif location_types is None:
            location_types = np.zeros(n_users, dtype=int)
        
        # Extract SNR values
        snr_values = np.zeros(n_users)
        if link_budgets is not None and users is not None:
            for i, user in enumerate(users):
                user_id = user.get('id', i)
                link_budget = link_budgets.get(user_id, {})
                snr_values[i] = link_budget.get('snr_db', 0.0)
        elif priorities is not None:
            snr_values = priorities * 10.0
        
        # Fairness score: rural gets boost
        fairness_score = np.zeros(n_users)
        urban_mask = (location_types == 0) | (location_types == 1)
        suburban_mask = location_types == 2
        rural_mask = location_types == 3
        
        fairness_score[urban_mask] = 0.0  # urban
        fairness_score[suburban_mask] = 0.5  # suburban
        fairness_score[rural_mask] = 1.0  # rural (highest priority)
        
        # Efficiency score: normalized SNR
        snr_min, snr_max = snr_values.min(), snr_values.max()
        if snr_max > snr_min:
            efficiency_score = (snr_values - snr_min) / (snr_max - snr_min)
        else:
            efficiency_score = np.ones(n_users) * 0.5
        
        # Combined score
        α = self.efficiency_weight
        combined_score = (1 - α) * fairness_score + α * efficiency_score
        
        # Select top users by combined score
        n_allocate = max(1, int(n_users * self.allocation_fraction))
        top_indices = np.argsort(combined_score)[::-1][:n_allocate]
        
        # Equal bandwidth to selected users
        per_user_bw = available_resources / n_allocate
        allocations[top_indices] = per_user_bw
        
        # Ensure allocations don't exceed demands
        allocations = np.minimum(allocations, demands)
        
        return allocations
    
    def _get_location_type(self, user: Dict) -> int:
        """Extract location type from user dict"""
        if 'location_type' in user:
            loc_type = user['location_type']
            if isinstance(loc_type, str):
                mapping = {'dense_urban': 0, 'urban': 1, 'suburban': 2, 'rural': 3}
                return mapping.get(loc_type, 0)
            return int(loc_type)
        elif 'is_urban' in user:
            return 0 if user['is_urban'] else 3
        else:
            return 0

