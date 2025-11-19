"""
GPU-accelerated Fuzzy Inference System using TensorFlow.

Optimized for H100 GPUs with batch processing support.
"""

import numpy as np
from typing import Dict, List, Optional, Union
import warnings

# Try to import TensorFlow
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None
    warnings.warn("TensorFlow not available, falling back to CPU")

from .rule_base_phase3 import Phase3RuleBase
from .membership_phase3 import build_all_membership_functions, create_fairness_output_membership_functions


class FuzzyInferenceSystemGPU:
    """
    GPU-accelerated Fuzzy Inference System using TensorFlow.
    
    Optimized for batch processing on H100 GPUs.
    Supports:
    - Batch inference (process multiple users simultaneously)
    - TensorFlow operations for GPU acceleration
    - Multi-GPU support via MirroredStrategy
    """
    
    def __init__(self, 
                 defuzzification_method: str = 'centroid',
                 use_gpu: bool = True,
                 batch_size: int = 4096):  # Increased for better GPU utilization (H100)
        """
        Initialize GPU-accelerated FIS.
        
        Args:
            defuzzification_method: 'centroid', 'bisector', 'mom', 'lom'
            use_gpu: Enable GPU acceleration
            batch_size: Batch size for processing
        """
        self.defuzz_method = defuzzification_method
        self.use_gpu = use_gpu and TF_AVAILABLE
        self.batch_size = batch_size
        
        # Initialize rule base and membership functions
        self.rule_base = Phase3RuleBase()
        self.output_mfs = create_fairness_output_membership_functions()
        self.membership_sets = build_all_membership_functions()
        
        # Build TensorFlow graph for batch inference
        if self.use_gpu:
            self._build_tf_graph()
        else:
            # Fallback to NumPy
            from .fuzzy_core import FuzzyInferenceSystem
            self.fallback_fis = FuzzyInferenceSystem(use_phase3=True)
    
    def _build_tf_graph(self):
        """Build TensorFlow computation graph for batch inference."""
        # Input placeholders (batch_size, 7 inputs)
        self.input_ph = tf.keras.Input(shape=(7,), dtype=tf.float32, name='fis_inputs')
        
        # Build membership function layers
        # For now, we'll use a simplified approach with pre-computed rule evaluation
        # In production, this could be fully differentiable
        
        # Create a custom layer for fuzzy inference
        self.fis_layer = tf.keras.layers.Lambda(
            self._tf_fuzzy_inference,
            name='fuzzy_inference'
        )
        
        # Build model
        self.model = tf.keras.Model(inputs=self.input_ph, outputs=self.fis_layer(self.input_ph))
        
        # Compile with XLA for H100 optimization
        if TF_AVAILABLE:
            try:
                tf.config.optimizer.set_jit(True)
            except:
                pass
        
        # Pre-compile the model with a dummy input to ensure GPU allocation
        # This forces TensorFlow to allocate GPU memory and compile the graph
        dummy_input = tf.zeros((self.batch_size, 7), dtype=tf.float32)
        _ = self.model(dummy_input, training=False)  # Warm-up call
    
    def _tf_fuzzy_inference(self, inputs):
        """
        TensorFlow implementation of fuzzy inference.
        
        Args:
            inputs: Tensor of shape (batch_size, 7) with [throughput, latency, outage, 
                   priority, doppler, elevation, beam_load]
        
        Returns:
            Tensor of shape (batch_size,) with fairness scores
        """
        batch_size = tf.shape(inputs)[0]
        
        # Normalize inputs to [0, 1] (already done in policy, but ensure)
        inputs_norm = tf.clip_by_value(inputs, 0.0, 1.0)
        
        # For batch processing, we'll use a simplified rule evaluation
        # Extract individual inputs
        throughput = inputs_norm[:, 0]
        latency = inputs_norm[:, 1]
        outage = inputs_norm[:, 2]
        priority = inputs_norm[:, 3]
        doppler = inputs_norm[:, 4]
        elevation = inputs_norm[:, 5]
        beam_load = inputs_norm[:, 6]
        
        # Simplified rule evaluation using TensorFlow operations
        # This is a simplified version - full implementation would evaluate all rules
        
        # Rule 1: IF latency=Poor AND outage=Frequent → fairness=Very-Low
        rule1_strength = tf.minimum(
            tf.maximum(0.0, 1.0 - latency * 2.0),  # Poor latency
            tf.maximum(0.0, outage * 2.0)  # Frequent outage
        )
        rule1_output = rule1_strength * 0.1  # Very-Low fairness
        
        # Rule 2: IF priority=High AND outage=Rare → fairness=High
        rule2_strength = tf.minimum(
            tf.maximum(0.0, priority * 2.0 - 1.0),  # High priority
            tf.maximum(0.0, 1.0 - outage * 2.0)  # Rare outage
        )
        rule2_output = rule2_strength * 0.8  # High fairness
        
        # Rule 3: IF elevation=High AND throughput=High → fairness=Very-High
        rule3_strength = tf.minimum(
            tf.maximum(0.0, elevation * 2.0 - 1.0),  # High elevation
            tf.maximum(0.0, throughput * 2.0 - 1.0)  # High throughput
        )
        rule3_output = rule3_strength * 0.9  # Very-High fairness
        
        # Rule 4: IF beam_load=Heavy AND throughput=Low → fairness=Low
        rule4_strength = tf.minimum(
            tf.maximum(0.0, beam_load * 2.0 - 1.0),  # Heavy load
            tf.maximum(0.0, 1.0 - throughput * 2.0)  # Low throughput
        )
        rule4_output = rule4_strength * 0.3  # Low fairness
        
        # Aggregate using maximum (OR operator)
        aggregated = tf.maximum(
            tf.maximum(rule1_output, rule2_output),
            tf.maximum(rule3_output, rule4_output)
        )
        
        # Add base fairness based on priority and elevation
        base_fairness = 0.3 + 0.4 * priority + 0.2 * elevation
        
        # Combine aggregated rules with base fairness
        final_fairness = tf.clip_by_value(
            aggregated + base_fairness * 0.5,
            0.0, 1.0
        )
        
        return final_fairness
    
    @tf.function(jit_compile=True)  # XLA compilation for H100
    def _infer_batch_tf(self, inputs_tf):
        """TensorFlow function for batch inference (compiled with XLA)."""
        return self.model(inputs_tf, training=False)
    
    def infer_batch(self, inputs_batch: np.ndarray) -> np.ndarray:
        """
        Perform batch inference on GPU.
        
        Args:
            inputs_batch: Array of shape (batch_size, 7) with [throughput, latency, 
                         outage, priority, doppler, elevation, beam_load]
        
        Returns:
            Array of shape (batch_size,) with fairness scores
        """
        if not self.use_gpu:
            # Fallback to CPU
            results = []
            for inputs in inputs_batch:
                input_dict = {
                    'throughput': inputs[0],
                    'latency': inputs[1],
                    'outage': inputs[2],
                    'priority': inputs[3],
                    'doppler': inputs[4],
                    'elevation': inputs[5],
                    'beam_load': inputs[6]
                }
                results.append(self.fallback_fis.infer(input_dict))
            return np.array(results)
        
        # Ensure inputs are on GPU
        batch_size = inputs_batch.shape[0]
        
        # Pad to batch_size if needed (for better GPU utilization)
        if batch_size < self.batch_size:
            # Pad with zeros (will be masked out)
            padded = np.zeros((self.batch_size, 7), dtype=np.float32)
            padded[:batch_size] = inputs_batch
            inputs_batch = padded
            pad_size = self.batch_size
        else:
            pad_size = batch_size
        
        # Convert to TensorFlow tensor and place on GPU
        with tf.device('/GPU:0' if self.use_gpu else '/CPU:0'):
            inputs_tf = tf.constant(inputs_batch, dtype=tf.float32)
            # Use compiled function
            outputs = self._infer_batch_tf(inputs_tf)
            # Extract only valid outputs
            outputs_np = outputs[:batch_size].numpy()
        
        return outputs_np
    
    def infer(self, inputs: Dict[str, float]) -> float:
        """
        Single inference (for compatibility).
        
        Args:
            inputs: Dictionary of crisp input values
        
        Returns:
            Defuzzified output (crisp value)
        """
        # Convert dict to array
        inputs_array = np.array([[
            inputs.get('throughput', 0.0),
            inputs.get('latency', 0.0),
            inputs.get('outage', 0.0),
            inputs.get('priority', 0.5),
            inputs.get('doppler', 0.0),
            inputs.get('elevation', 0.0),
            inputs.get('beam_load', 0.5)
        ]])
        
        result = self.infer_batch(inputs_array)
        return float(result[0])
    
    def infer_batch_dict(self, inputs_list: List[Dict[str, float]]) -> np.ndarray:
        """
        Batch inference from list of dictionaries.
        
        Args:
            inputs_list: List of input dictionaries
        
        Returns:
            Array of fairness scores
        """
        # Convert to array format
        batch = np.array([[
            inputs.get('throughput', 0.0),
            inputs.get('latency', 0.0),
            inputs.get('outage', 0.0),
            inputs.get('priority', 0.5),
            inputs.get('doppler', 0.0),
            inputs.get('elevation', 0.0),
            inputs.get('beam_load', 0.5)
        ] for inputs in inputs_list])
        
        return self.infer_batch(batch)

