"""
Fuzzy Adaptive Policy for Dynamic Spectrum Sharing

Phase 3: Integration with Mamdani FIS for fairness-based allocation.

This policy uses fuzzy logic to adaptively allocate spectrum resources
based on multiple criteria:
- Throughput
- Latency
- Outage
- Priority
- Doppler
- Elevation
- Beam Load
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from ..spectrum_map import SpectrumMap
from ..spectrum_environment import SpectrumEnvironment
from ...fairness.fuzzy_core import FuzzyInferenceSystem


class FuzzyAdaptivePolicy:
    """
    Fuzzy adaptive policy that uses Mamdani FIS for fairness-based allocation.
    
    Phase 3: Complete integration with 7-input FIS.
    
    This policy:
    1. Collects context per user (throughput, latency, outage, etc.)
    2. Applies FIS to compute fairness score
    3. Allocates spectrum based on fairness score
    """
    
    def __init__(
        self,
        spectrum_env: SpectrumEnvironment,
        spectrum_map: Optional[SpectrumMap] = None,
        fuzzy_system: Optional[FuzzyInferenceSystem] = None,
        alpha: float = 0.7
    ):
        """
        Initialize fuzzy adaptive policy.
        
        Args:
            spectrum_env: Spectrum environment instance
            spectrum_map: Optional spectrum map instance
            fuzzy_system: Optional fuzzy inference system (creates Phase 3 FIS if None)
            alpha: Weight for fairness vs priority (0-1, higher = more fairness)
        """
        self.spectrum_env = spectrum_env
        self.spectrum_map = spectrum_map
        self.fuzzy_system = fuzzy_system or self._create_default_fuzzy_system()
        self.alpha = alpha
        
    def _create_default_fuzzy_system(self) -> FuzzyInferenceSystem:
        """Create Phase 3 fuzzy inference system for adaptive allocation."""
        from ...fairness.fuzzy_core import FuzzyInferenceSystem
        return FuzzyInferenceSystem(use_phase3=True)
    
    def allocate(
        self,
        users: List[Dict],
        qos: Dict[str, Dict],
        context: Dict[str, Dict],
        bandwidth_hz: float = 100e6,
        alpha: Optional[float] = None
    ) -> Dict[str, Optional[Tuple[float, float]]]:
        """
        Allocate spectrum using fuzzy adaptive policy.
        
        Phase 4: Complete allocation algorithm with ranking and conflict detection.
        
        Args:
            users: List of user dictionaries with id, priority, operator
            qos: Dictionary mapping user_id to QoS metrics (throughput, latency, outage)
            context: Dictionary mapping user_id to context (elevation, doppler, beam_load)
            bandwidth_hz: Required bandwidth in Hz
            alpha: Weight for fairness vs priority (0-1, higher = more fairness). 
                   If None, uses instance default.
            
        Returns:
            Dictionary mapping user_id to (center_freq, sinr) or None
        """
        if alpha is None:
            alpha = self.alpha
        
        # Step 1: Build context vector for each user and compute fairness scores
        user_scores = []
        
        for user in users:
            user_id = user['id']
            user_qos = qos.get(user_id, {})
            user_context = context.get(user_id, {})
            
            # Normalize inputs to [0, 1]
            # Throughput: normalize by max capacity (e.g., 100 Mbps)
            throughput_norm = np.clip(user_qos.get('throughput', 0.0) / 100e6, 0, 1)
            
            # Latency: normalize by max latency (e.g., 1.0 s), lower is better
            latency_norm = np.clip(1.0 - (user_qos.get('latency', 1.0) / 1.0), 0, 1)
            
            # Outage: already normalized [0, 1]
            outage_norm = np.clip(user_qos.get('outage', 0.5), 0, 1)
            
            # Priority: already normalized [0, 1]
            priority_norm = np.clip(user.get('priority', 0.5), 0, 1)
            
            # Doppler: normalize by max Doppler (e.g., 100 kHz)
            doppler_norm = np.clip(user_context.get('doppler', 0.0) / 100e3, 0, 1)
            
            # Elevation: normalize by 90 degrees
            elevation_norm = np.clip(user_context.get('elevation', 0.0) / 90.0, 0, 1)
            
            # Beam load: already normalized [0, 1]
            beam_load_norm = np.clip(user_context.get('beam_load', 0.5), 0, 1)
            
            # Build FIS inputs
            fis_inputs = {
                'throughput': throughput_norm,
                'latency': latency_norm,
                'outage': outage_norm,
                'priority': priority_norm,
                'doppler': doppler_norm,
                'elevation': elevation_norm,
                'beam_load': beam_load_norm
            }
            
            # Apply FIS to get fairness score
            fairness_score = self.fuzzy_system.infer(fis_inputs)
            
            # Combine fairness and priority: score_u = alpha * fairness + (1 - alpha) * priority
            score_u = alpha * fairness_score + (1 - alpha) * priority_norm
            
            user_scores.append({
                'user_id': user_id,
                'user': user,
                'score': score_u,
                'fairness_score': fairness_score,
                'priority': priority_norm,
                'context': user_context
            })
        
        # Step 2: Sort users by score (descending)
        user_scores.sort(key=lambda x: x['score'], reverse=True)
        
        # Step 3: Allocate spectrum to users in order
        allocations = {}
        
        for user_data in user_scores:
            user_id = user_data['user_id']
            user = user_data['user']
            user_context = user_data['context']
            
            # Get available channels (use lower min_sinr to allow more allocations)
            # Try with actual link budget SNR if available
            user_link_budget = user_context.get('link_budget', {})
            link_snr_db = user_link_budget.get('snr_db') if isinstance(user_link_budget, dict) else None
            
            # Get beam_id for spatial reuse (exclude this beam from interference check)
            beam_id = user_context.get('beam_id', f"beam_{user.get('operator', 0)}_{user_id}")
            
            # Use find_available_spectrum directly to pass exclude_beam_id for spatial reuse
            available_channels = self.spectrum_env.find_available_spectrum(
                bandwidth_hz,
                min_sinr_db=0.0,  # Lower threshold to allow allocations
                exclude_beam_id=beam_id,  # Allow spatial reuse (different beams can share frequency)
                link_budget_snr_db=link_snr_db,
                allow_spatial_reuse=True
            )
            
            if not available_channels:
                # No channels available - try with even lower threshold
                available_channels = self.spectrum_env.find_available_spectrum(
                    bandwidth_hz,
                    min_sinr_db=-10.0,  # Very low threshold as fallback
                    exclude_beam_id=beam_id,
                    link_budget_snr_db=link_snr_db,
                    allow_spatial_reuse=True
                )
            
            if not available_channels:
                # Still no channels available
                allocations[user_id] = None
                continue
            
            # Select best channel (highest SINR)
            best_channel = max(available_channels, key=lambda x: x[1])
            center_freq, sinr = best_channel
            
            # Try to allocate (beam_id already set above)
            allocation = self.spectrum_env.allocate(
                user_id=user_id,
                bandwidth_hz=bandwidth_hz,
                beam_id=beam_id,
                preferred_frequency_hz=center_freq
            )
            
            if allocation is None:
                # Conflict detected, try next best channel
                for freq, sinr_candidate in available_channels:
                    if freq != center_freq:
                        allocation = self.spectrum_env.allocate(
                            user_id=user_id,
                            bandwidth_hz=bandwidth_hz,
                            beam_id=beam_id,
                            preferred_frequency_hz=freq
                        )
                        if allocation is not None:
                            break
            
            allocations[user_id] = allocation
        
        return allocations
    
    def evaluate_fairness(self, user_context: Dict[str, Dict]) -> Dict[str, float]:
        """
        Evaluate fairness score for multiple users.
        
        Args:
            user_context: Dictionary mapping user_id to context dict
            
        Returns:
            Dictionary mapping user_id to fairness score
        """
        fairness_scores = {}
        
        for user_id, context in user_context.items():
            inputs = {
                'throughput': np.clip(context.get('throughput', 0.5), 0, 1),
                'latency': np.clip(context.get('latency', 0.5), 0, 1),
                'outage': np.clip(context.get('outage', 0.5), 0, 1),
                'priority': np.clip(context.get('priority', 0.5), 0, 1),
                'doppler': np.clip(context.get('doppler', 0.5), 0, 1),
                'elevation': np.clip(context.get('elevation', 0.5), 0, 1),
                'beam_load': np.clip(context.get('beam_load', 0.5), 0, 1)
            }
            
            fairness_score = self.fuzzy_system.infer(inputs)
            fairness_scores[user_id] = fairness_score
        
        return fairness_scores
    
    def update_fuzzy_rules(self, rules: List[Dict]):
        """Update fuzzy inference rules dynamically."""
        if self.fuzzy_system:
            self.fuzzy_system.update_rules(rules)
    
    def get_policy_stats(self) -> Dict:
        """Get statistics about policy performance."""
        return {
            'policy_type': 'fuzzy_adaptive',
            'fuzzy_system_active': self.fuzzy_system is not None,
            'total_allocations': len(self.spectrum_map.allocations)
        }

