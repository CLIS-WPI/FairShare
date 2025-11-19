"""
GPU-accelerated Fuzzy Adaptive Policy for Dynamic Spectrum Sharing.

Optimized for H100 GPUs with batch processing.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from ..spectrum_map import SpectrumMap
from ..spectrum_environment import SpectrumEnvironment

# Try to import GPU-accelerated FIS
try:
    from ...fairness.fuzzy_core_gpu import FuzzyInferenceSystemGPU
    GPU_FIS_AVAILABLE = True
except ImportError:
    GPU_FIS_AVAILABLE = False
    from ...fairness.fuzzy_core import FuzzyInferenceSystem

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None


class FuzzyAdaptivePolicyGPU:
    """
    GPU-accelerated fuzzy adaptive policy with batch processing.
    
    Optimized for H100 GPUs:
    - Batch inference for all users simultaneously
    - GPU-accelerated fairness score computation
    - Vectorized operations
    """
    
    def __init__(
        self,
        spectrum_env: SpectrumEnvironment,
        spectrum_map: Optional[SpectrumMap] = None,
        fuzzy_system: Optional[FuzzyInferenceSystemGPU] = None,
        alpha: float = 0.7,
        use_gpu: bool = True,
        batch_size: int = 4096  # Increased for better GPU utilization (H100)
    ):
        """
        Initialize GPU-accelerated fuzzy adaptive policy.
        
        Args:
            spectrum_env: Spectrum environment instance
            spectrum_map: Optional spectrum map instance
            fuzzy_system: Optional GPU-accelerated fuzzy inference system
            alpha: Weight for fairness vs priority (0-1)
            use_gpu: Enable GPU acceleration
            batch_size: Batch size for processing
        """
        self.spectrum_env = spectrum_env
        self.spectrum_map = spectrum_map
        self.alpha = alpha
        self.use_gpu = use_gpu and TF_AVAILABLE and GPU_FIS_AVAILABLE
        self.batch_size = batch_size
        
        if self.use_gpu:
            self.fuzzy_system = fuzzy_system or FuzzyInferenceSystemGPU(use_gpu=True, batch_size=batch_size)
        else:
            # Fallback to CPU
            from ...fairness.fuzzy_core import FuzzyInferenceSystem
            self.fuzzy_system = FuzzyInferenceSystem(use_phase3=True)
    
    def allocate(
        self,
        users: List[Dict],
        qos: Dict[str, Dict],
        context: Dict[str, Dict],
        bandwidth_hz: float = 100e6,
        alpha: Optional[float] = None
    ) -> Dict[str, Optional[Tuple[float, float]]]:
        """
        Allocate spectrum using GPU-accelerated fuzzy adaptive policy.
        
        Optimized for batch processing on H100 GPUs.
        
        Args:
            users: List of user dictionaries
            qos: Dictionary mapping user_id to QoS metrics
            context: Dictionary mapping user_id to context
            bandwidth_hz: Required bandwidth in Hz
            alpha: Weight for fairness vs priority
        
        Returns:
            Dictionary mapping user_id to (center_freq, sinr) or None
        """
        if alpha is None:
            alpha = self.alpha
        
        num_users = len(users)
        
        if self.use_gpu and num_users > 1:
            # GPU-accelerated batch processing
            return self._allocate_batch_gpu(users, qos, context, bandwidth_hz, alpha)
        else:
            # Fallback to CPU (original implementation)
            return self._allocate_cpu(users, qos, context, bandwidth_hz, alpha)
    
    def _allocate_batch_gpu(
        self,
        users: List[Dict],
        qos: Dict[str, Dict],
        context: Dict[str, Dict],
        bandwidth_hz: float,
        alpha: float
    ) -> Dict[str, Optional[Tuple[float, float]]]:
        """GPU-accelerated batch allocation."""
        num_users = len(users)
        
        # Prepare batch inputs for GPU
        batch_inputs = np.zeros((num_users, 7), dtype=np.float32)
        user_ids = []
        priorities = []
        beam_ids = []
        
        for i, user in enumerate(users):
            user_id = user['id']
            user_ids.append(user_id)
            
            user_qos = qos.get(user_id, {})
            user_context = context.get(user_id, {})
            
            # Normalize inputs
            throughput_norm = np.clip(user_qos.get('throughput', 0.0) / 100e6, 0, 1)
            latency_norm = np.clip(1.0 - (user_qos.get('latency', 1.0) / 1.0), 0, 1)
            outage_norm = np.clip(user_qos.get('outage', 0.5), 0, 1)
            priority_norm = np.clip(user.get('priority', 0.5), 0, 1)
            doppler_norm = np.clip(user_context.get('doppler', 0.0) / 100e3, 0, 1)
            elevation_norm = np.clip(user_context.get('elevation', 0.0) / 90.0, 0, 1)
            beam_load_norm = np.clip(user_context.get('beam_load', 0.5), 0, 1)
            
            batch_inputs[i] = [
                throughput_norm,
                latency_norm,
                outage_norm,
                priority_norm,
                doppler_norm,
                elevation_norm,
                beam_load_norm
            ]
            
            priorities.append(priority_norm)
            
            # Get beam_id
            operator = user.get('operator', 0)
            beam_id = f"beam_{operator}"
            beam_ids.append(beam_id)
        
        # Batch inference on GPU
        fairness_scores = self.fuzzy_system.infer_batch(batch_inputs)
        
        # Combine fairness and priority (vectorized) - move to GPU if possible
        if TF_AVAILABLE and self.use_gpu:
            with tf.device('/GPU:0'):
                priorities_tf = tf.constant(priorities, dtype=tf.float32)
                fairness_tf = tf.constant(fairness_scores, dtype=tf.float32)
                scores_tf = alpha * fairness_tf + (1 - alpha) * priorities_tf
                scores = scores_tf.numpy()
        else:
            priorities_array = np.array(priorities)
            scores = alpha * fairness_scores + (1 - alpha) * priorities_array
        
        # Sort by score (descending) - can be done on GPU but argsort is faster on CPU
        sorted_indices = np.argsort(scores)[::-1]
        
        # Allocate spectrum
        allocations = {}
        for idx in sorted_indices:
            user_id = user_ids[idx]
            user = users[idx]
            beam_id = beam_ids[idx]
            
            # Get link budget for SINR calculation
            link_budget = context.get(user_id, {}).get('link_budget', {})
            link_snr_db = link_budget.get('snr_db', 15.0)
            
            # Find available spectrum with spatial reuse
            available_channels = self.spectrum_env.find_available_spectrum(
                bandwidth_hz,
                min_sinr_db=0.0,
                exclude_beam_id=beam_id,
                link_budget_snr_db=link_snr_db,
                allow_spatial_reuse=True
            )
            
            if not available_channels:
                # Try with lower threshold
                available_channels = self.spectrum_env.find_available_spectrum(
                    bandwidth_hz,
                    min_sinr_db=-10.0,
                    exclude_beam_id=beam_id,
                    link_budget_snr_db=link_snr_db,
                    allow_spatial_reuse=True
                )
            
            if not available_channels:
                allocations[user_id] = None
                continue
            
            # Select best channel
            best_channel = max(available_channels, key=lambda x: x[1])
            center_freq, sinr = best_channel
            
            # Allocate
            allocation = self.spectrum_env.allocate(
                user_id=user_id,
                bandwidth_hz=bandwidth_hz,
                beam_id=beam_id,
                preferred_frequency_hz=center_freq
            )
            
            allocations[user_id] = allocation
        
        return allocations
    
    def _allocate_cpu(
        self,
        users: List[Dict],
        qos: Dict[str, Dict],
        context: Dict[str, Dict],
        bandwidth_hz: float,
        alpha: float
    ) -> Dict[str, Optional[Tuple[float, float]]]:
        """CPU fallback allocation (original implementation)."""
        from .fuzzy_adaptive import FuzzyAdaptivePolicy
        cpu_policy = FuzzyAdaptivePolicy(
            self.spectrum_env,
            self.spectrum_map,
            self.fuzzy_system if not isinstance(self.fuzzy_system, FuzzyInferenceSystemGPU) else None,
            alpha
        )
        return cpu_policy.allocate(users, qos, context, bandwidth_hz, alpha)

