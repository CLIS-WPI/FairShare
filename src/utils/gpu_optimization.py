"""
GPU optimization utilities for maximizing H100 utilization.

Includes batch processing, vectorization, and GPU memory management.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None


def batch_normalize_on_gpu(
    values: np.ndarray,
    min_val: float,
    max_val: float,
    inverse: bool = False
) -> np.ndarray:
    """
    Normalize values on GPU for better performance.
    
    Args:
        values: Input values
        min_val: Minimum value for normalization
        max_val: Maximum value for normalization
        inverse: If True, higher values map to lower normalized values
    
    Returns:
        Normalized values
    """
    if not TF_AVAILABLE:
        # CPU fallback
        normalized = np.clip((values - min_val) / (max_val - min_val), 0, 1)
        if inverse:
            normalized = 1.0 - normalized
        return normalized
    
    # GPU-accelerated normalization
    with tf.device('/GPU:0'):
        values_tf = tf.constant(values, dtype=tf.float32)
        normalized_tf = tf.clip_by_value(
            (values_tf - min_val) / (max_val - min_val),
            0.0, 1.0
        )
        if inverse:
            normalized_tf = 1.0 - normalized_tf
        return normalized_tf.numpy()


def batch_combine_scores_gpu(
    fairness_scores: np.ndarray,
    priorities: np.ndarray,
    alpha: float
) -> np.ndarray:
    """
    Combine fairness and priority scores on GPU.
    
    Args:
        fairness_scores: Array of fairness scores
        priorities: Array of priority values
        alpha: Weight for fairness (0-1)
    
    Returns:
        Combined scores
    """
    if not TF_AVAILABLE:
        # CPU fallback
        return alpha * fairness_scores + (1 - alpha) * priorities
    
    # GPU-accelerated combination
    with tf.device('/GPU:0'):
        fairness_tf = tf.constant(fairness_scores, dtype=tf.float32)
        priorities_tf = tf.constant(priorities, dtype=tf.float32)
        combined_tf = alpha * fairness_tf + (1 - alpha) * priorities_tf
        return combined_tf.numpy()


def prepare_batch_inputs_gpu(
    users: List[Dict],
    qos: Dict[str, Dict],
    context: Dict[str, Dict],
    batch_size: int = 2048
) -> Tuple[np.ndarray, List, List, List]:
    """
    Prepare batch inputs for GPU processing.
    
    Args:
        users: List of user dictionaries
        qos: Dictionary mapping user_id to QoS metrics
        context: Dictionary mapping user_id to context
        batch_size: Batch size for processing
    
    Returns:
        Tuple of (batch_inputs, user_ids, priorities, beam_ids)
    """
    num_users = len(users)
    
    # Pre-allocate arrays
    batch_inputs = np.zeros((num_users, 7), dtype=np.float32)
    user_ids = []
    priorities = []
    beam_ids = []
    
    # Collect all data first (CPU)
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
    
    return batch_inputs, user_ids, priorities, beam_ids

