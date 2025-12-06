"""
GPU-accelerated channel model for batch processing.

Optimized for H100 GPUs with vectorized operations.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None


class ChannelModelGPU:
    """
    GPU-accelerated channel model for batch link budget computation.
    
    Processes multiple users simultaneously on GPU.
    """
    
    # Constants
    SPEED_OF_LIGHT = 299792458.0  # m/s
    BOLTZMANN = 1.380649e-23  # J/K
    
    def __init__(self, frequency_hz: float = 12e9, 
                 satellite_antenna_gain_dbi: float = 30.0,
                 ground_antenna_gain_dbi: float = 40.0,
                 temperature_k: float = 290.0,
                 use_gpu: bool = True):
        """
        Initialize GPU-accelerated channel model.
        
        Args:
            frequency_hz: Carrier frequency in Hz
            satellite_antenna_gain_dbi: Satellite antenna gain in dBi
            ground_antenna_gain_dbi: Ground station antenna gain in dBi
            temperature_k: System noise temperature in Kelvin
            use_gpu: Enable GPU acceleration
        """
        self.frequency_hz = float(frequency_hz)
        self.frequency_ghz = self.frequency_hz / 1e9
        self.sat_antenna_gain = satellite_antenna_gain_dbi
        self.gs_antenna_gain = ground_antenna_gain_dbi
        self.temperature_k = temperature_k
        self.use_gpu = use_gpu and TF_AVAILABLE
        
        if self.use_gpu:
            self._build_gpu_functions()
    
    def _build_gpu_functions(self):
        """Build GPU-accelerated computation functions."""
        # Use JIT compilation for GPU performance
        @tf.function(jit_compile=True, reduce_retracing=True)
        def compute_path_loss_batch(slant_ranges, elevations):
            """Batch path loss computation on GPU."""
            # Free space path loss: FSPL = 20*log10(4*pi*d*f/c)
            # Simplified NTN path loss model
            d_km = slant_ranges / 1000.0  # Convert to km
            f_ghz = tf.constant(self.frequency_ghz, dtype=tf.float32)
            c_km_per_s = self.SPEED_OF_LIGHT / 1000.0
            
            # Clip elevation to avoid NaN
            elev_clipped = tf.clip_by_value(elevations, 0.1, 89.9)
            elev_rad = tf.cast(elev_clipped * np.pi / 180.0, tf.float32)
            
            # FSPL
            fspl_db = 20.0 * tf.math.log(
                4.0 * np.pi * d_km * f_ghz * 1e9 / self.SPEED_OF_LIGHT
            ) / tf.math.log(10.0)
            
            # Elevation correction
            correction = tf.where(
                elev_clipped < 10.0,
                20.0 * tf.math.log(tf.sin(elev_rad)) / tf.math.log(10.0),
                tf.zeros_like(elev_clipped)
            )
            
            return fspl_db + correction
        
        @tf.function(jit_compile=True, reduce_retracing=True)
        def compute_snr_batch(rx_powers, bandwidths):
            """Batch SNR computation on GPU."""
            # Noise power
            noise_power_db = 10.0 * tf.math.log(
                self.BOLTZMANN * self.temperature_k * bandwidths
            ) / tf.math.log(10.0)
            noise_power_dbm = noise_power_db + 30.0
            
            # SNR
            snr_db = rx_powers - noise_power_dbm
            return snr_db
        
        self._compute_path_loss_batch = compute_path_loss_batch
        self._compute_snr_batch = compute_snr_batch
    
    def compute_link_budgets_batch(
        self,
        geometries: List[Dict],
        rain_rate_mmh: float = 0.0,
        tx_power_dbm: float = 40.0,
        bandwidth_hz: float = 100e6
    ) -> List[Dict]:
        """
        Compute link budgets for multiple users in batch.
        
        Args:
            geometries: List of geometry dictionaries
            rain_rate_mmh: Rain rate in mm/h
            tx_power_dbm: Transmit power in dBm
            bandwidth_hz: Signal bandwidth in Hz
        
        Returns:
            List of link budget dictionaries
        """
        num_users = len(geometries)
        
        # Batch size for H100: process in chunks of 1000 to avoid OOM
        # H100 has 80GB memory, can handle ~1000 users per batch
        batch_size = 1000
        
        if not self.use_gpu or num_users < 10:
            # Fallback to CPU for small batches
            from .channel_model import ChannelModel
            cpu_model = ChannelModel(self.frequency_hz)
            return [
                cpu_model.compute_link_budget(geom, rain_rate_mmh, tx_power_dbm, bandwidth_hz)
                for geom in geometries
            ]
        
        # Process in batches for large user counts (NYC scenario: 5000 users)
        if num_users > batch_size:
            results = []
            for i in range(0, num_users, batch_size):
                batch_geometries = geometries[i:i+batch_size]
                batch_results = self._compute_batch_single(batch_geometries, rain_rate_mmh, tx_power_dbm, bandwidth_hz)
                results.extend(batch_results)
            return results
        else:
            return self._compute_batch_single(geometries, rain_rate_mmh, tx_power_dbm, bandwidth_hz)
    
    def _compute_batch_single(
        self,
        geometries: List[Dict],
        rain_rate_mmh: float,
        tx_power_dbm: float,
        bandwidth_hz: float
    ) -> List[Dict]:
        """Compute link budgets for a single batch (internal method)."""
        
        # Extract arrays for batch processing
        slant_ranges = np.array([g['slant_range'] for g in geometries], dtype=np.float32)
        elevations = np.array([g['elevation'] for g in geometries], dtype=np.float32)
        elevations = np.clip(elevations, 0.1, 90.0)
        
        # Compute on GPU - FORCE execution with synchronous operations
        # Use explicit device placement and ensure all ops run on GPU
        # CRITICAL: Use tf.config.run_functions_eagerly(False) for graph mode
        # and force synchronous execution to ensure GPU is actually used
        
        # Ensure we're in graph mode for performance
        tf.config.run_functions_eagerly(False)
        
        with tf.device('/GPU:0'):
            # Create tensors directly on GPU
            slant_ranges_tf = tf.constant(slant_ranges, dtype=tf.float32)
            elevations_tf = tf.constant(elevations, dtype=tf.float32)
            
            # Path loss - this MUST execute on GPU
            path_loss_tf = self._compute_path_loss_batch(slant_ranges_tf, elevations_tf)
            
            # Rain attenuation
            rain_loss_tf = tf.constant(rain_rate_mmh * 0.1, dtype=tf.float32)
            
            # Antenna gains
            sat_gain = tf.constant(self.sat_antenna_gain, dtype=tf.float32)
            gs_gain = tf.constant(self.gs_antenna_gain, dtype=tf.float32)
            
            # Shadowing
            shadowing_tf = tf.where(
                elevations_tf > 30.0,
                tf.constant(2.0, dtype=tf.float32),  # LOS
                tf.constant(8.0, dtype=tf.float32)   # NLOS
            )
            
            # Received power
            rx_power_tf = (tx_power_dbm + sat_gain + gs_gain - 
                          path_loss_tf - rain_loss_tf - shadowing_tf)
            
            # SNR
            bandwidths_tf = tf.constant(bandwidth_hz, dtype=tf.float32)
            snr_tf = self._compute_snr_batch(rx_power_tf, bandwidths_tf)
            
            # Capacity (Shannon)
            snr_linear_tf = tf.pow(10.0, snr_tf / 10.0)
            capacity_tf = bandwidths_tf * tf.math.log(1.0 + snr_linear_tf) / tf.math.log(2.0)
            
            # FORCE GPU execution by computing a reduction that requires all values
            # This ensures the entire computation graph executes on GPU
            # Use a more complex reduction to force GPU computation
            combined_result = (tf.reduce_sum(path_loss_tf) + 
                             tf.reduce_sum(rx_power_tf) + 
                             tf.reduce_sum(snr_tf) + 
                             tf.reduce_sum(capacity_tf))
            
            # Force synchronous execution - this blocks until GPU computation completes
            # Use tf.identity to ensure the computation is actually executed
            with tf.device('/GPU:0'):
                sync_result = tf.identity(combined_result)
                # Force evaluation by converting to numpy - this waits for GPU
                _ = sync_result.numpy()
            
            # Now convert to NumPy (all computation already done on GPU)
            # These conversions will be fast since computation is already done
            path_loss = path_loss_tf.numpy()
            rx_power = rx_power_tf.numpy()
            snr = snr_tf.numpy()
            capacity = capacity_tf.numpy()
        
        # Build result dictionaries
        results = []
        for i in range(num_users):
            results.append({
                'tx_power_dbm': tx_power_dbm,
                'rx_power_dbm': float(rx_power[i]),
                'path_loss_db': float(path_loss[i]),
                'rain_attenuation_db': float(rain_loss_tf.numpy()),
                'shadowing_loss_db': float(shadowing_tf.numpy()[i]),
                'sat_antenna_gain_db': self.sat_antenna_gain,
                'gs_antenna_gain_db': self.gs_antenna_gain,
                'snr_db': float(snr[i]),
                'capacity_bps': float(capacity[i]),
                'bandwidth_hz': bandwidth_hz,
                'link_state': 'los' if elevations[i] > 30 else 'nlos',
                'elevation_deg': float(elevations[i]),
                'slant_range': float(slant_ranges[i])
            })
        
        return results

