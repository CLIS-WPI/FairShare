#!/usr/bin/env python3
"""
Debug SpectrumEnvironment allocation logic to understand why only ~10 users are allocated.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.dss.spectrum_environment import SpectrumEnvironment

print("=" * 60)
print("SPECTRUM ENVIRONMENT ALLOCATION DEBUG")
print("=" * 60)

# Initialize environment
env = SpectrumEnvironment((10e9, 12e9), frequency_resolution_hz=1e6)
per_user_bw = 20e6  # 20 MHz per user

print(f"\nEnvironment:")
print(f"  Frequency range: {env.freq_min/1e9:.2f} - {env.freq_max/1e9:.2f} GHz")
print(f"  Total bandwidth: {(env.freq_max - env.freq_min)/1e9:.2f} GHz")
print(f"  Per-user bandwidth: {per_user_bw/1e6:.1f} MHz")
print(f"  Theoretical max users: {int((env.freq_max - env.freq_min) / per_user_bw)}")

# Test sequential allocations
print("\n" + "=" * 60)
print("TESTING SEQUENTIAL ALLOCATIONS")
print("=" * 60)

successful = 0
failed = 0
failure_reasons = {}

for i in range(100):
    beam_id = f"beam_{i % 3}"  # 3 beams for spatial reuse
    
    # Try to allocate
    allocation = env.allocate(
        user_id=f"user_{i}",
        bandwidth_hz=per_user_bw,
        beam_id=beam_id,
        preferred_frequency_hz=None
    )
    
    if allocation:
        successful += 1
        if successful <= 10:
            freq, sinr = allocation
            print(f"✓ User {i:3d} (beam {beam_id:8s}): {freq/1e9:.3f} GHz, SINR: {sinr:.1f} dB")
    else:
        failed += 1
        # Check why it failed
        available = env.get_available_channels(per_user_bw, min_sinr_db=0.0)
        if len(available) == 0:
            reason = "No available channels"
        else:
            reason = f"{len(available)} channels available but allocation failed"
        
        if reason not in failure_reasons:
            failure_reasons[reason] = 0
        failure_reasons[reason] += 1
        
        if failed <= 10:
            print(f"✗ User {i:3d} (beam {beam_id:8s}): FAILED - {reason}")

print("\n" + "=" * 60)
print(f"Results: {successful} successful, {failed} failed")
print(f"Success rate: {successful/100*100:.1f}%")
print("\nFailure reasons:")
for reason, count in failure_reasons.items():
    print(f"  {reason}: {count}")

# Check spectrum occupancy after allocations
print("\n" + "=" * 60)
print("SPECTRUM OCCUPANCY ANALYSIS")
print("=" * 60)

# Get occupancy map
spectrum_map, beam_ids = env.get_spectrum_map()
print(f"  Beams: {len(beam_ids)}")
print(f"  Frequency bins: {env.n_bins}")

# Count occupied bins per beam
for beam_id in beam_ids:
    occupied = np.sum(spectrum_map[beam_ids.index(beam_id), :] > -np.inf)
    print(f"  {beam_id}: {occupied} bins occupied ({occupied/env.n_bins*100:.1f}%)")

# Check available channels
available = env.get_available_channels(per_user_bw, min_sinr_db=0.0)
print(f"\n  Available channels (20 MHz, SINR >= 0 dB): {len(available)}")

if len(available) > 0:
    print(f"  First 5 channels:")
    for i, (freq, sinr) in enumerate(available[:5]):
        print(f"    {i+1}. {freq/1e9:.3f} GHz, SINR: {sinr:.1f} dB")

# Test conflict detection
print("\n" + "=" * 60)
print("CONFLICT DETECTION TEST")
print("=" * 60)

# Try to allocate same frequency to different beams (should work - spatial reuse)
test_freq = 11e9
beam1 = "beam_0"
beam2 = "beam_1"

allocation1 = env.allocate("test_user_1", per_user_bw, beam_id=beam1, preferred_frequency_hz=test_freq)
allocation2 = env.allocate("test_user_2", per_user_bw, beam_id=beam2, preferred_frequency_hz=test_freq)

print(f"  Allocate {test_freq/1e9:.3f} GHz to {beam1}: {'✓' if allocation1 else '✗'}")
print(f"  Allocate {test_freq/1e9:.3f} GHz to {beam2}: {'✓' if allocation2 else '✗'}")

if allocation1 and allocation2:
    print("  ✓ Spatial reuse works (different beams can share frequency)")
else:
    print("  ✗ Spatial reuse not working")

# Try to allocate same frequency to same beam (should fail - conflict)
allocation3 = env.allocate("test_user_3", per_user_bw, beam_id=beam1, preferred_frequency_hz=test_freq)
print(f"  Allocate {test_freq/1e9:.3f} GHz to {beam1} again: {'✗ (conflict - expected)' if not allocation3 else '✓ (unexpected - should conflict!)'}")

print("\n" + "=" * 60)

