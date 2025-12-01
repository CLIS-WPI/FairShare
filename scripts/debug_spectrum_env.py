#!/usr/bin/env python3
"""
Debug spectrum environment to understand why allocations fail.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.dss.spectrum_environment import SpectrumEnvironment
from src.experiments import load_scenario

# Load scenario
config = load_scenario("urban_congestion_phase4")

# Initialize environment
# Get frequency resolution from config or use default
freq_res = getattr(config, 'frequency_resolution_hz', 1e6)  # Default 1 MHz
env = SpectrumEnvironment(
    frequency_range_hz=(config.frequency_range_hz[0], config.frequency_range_hz[1]),
    frequency_resolution_hz=freq_res
)

print("=" * 60)
print("SPECTRUM ENVIRONMENT DEBUG")
print("=" * 60)
# Get frequency range from config
freq_range = config.frequency_range_hz
print(f"Frequency range: {freq_range[0]/1e9:.2f} - {freq_range[1]/1e9:.2f} GHz")
print(f"Frequency resolution: {freq_res/1e6:.1f} MHz")
print(f"Total bandwidth: {(freq_range[1] - freq_range[0])/1e9:.2f} GHz")

# Try to allocate 500 users
print("\n" + "=" * 60)
print("TESTING ALLOCATIONS")
print("=" * 60)

bandwidth_hz = 100e6  # 100 MHz per user
successful = 0
failed = 0
failure_reasons = {}

for i in range(500):
    allocation = env.allocate(
        user_id=f"user_{i}",
        bandwidth_hz=bandwidth_hz,
        beam_id=f"beam_{i % 10}",
        preferred_frequency_hz=None
    )
    
    if allocation:
        successful += 1
        if successful <= 5:
            print(f"✓ User {i}: {allocation[0]/1e9:.2f} GHz")
    else:
        failed += 1
        if failed <= 5:
            print(f"✗ User {i}: FAILED")

print("\n" + "=" * 60)
print(f"Results: {successful} successful, {failed} failed")
print(f"Success rate: {successful/500*100:.1f}%")
print("=" * 60)

# Check spectrum occupancy
print("\nSpectrum occupancy:")
try:
    mid_freq = (freq_range[0] + freq_range[1])/2
    occupancy = env.get_spectrum_occupancy(
        frequency_hz=mid_freq,
        bandwidth_hz=100e6
    )
    print(f"  Mid-band occupancy: {occupancy:.2f} dB")
except Exception as e:
    print(f"  Could not compute occupancy: {e}")

# Check available channels
try:
    available = env.get_available_channels(bandwidth_hz=100e6, min_sinr_db=0.0)
    print(f"  Available channels: {len(available)}")
except Exception as e:
    print(f"  Could not get available channels: {e}")

