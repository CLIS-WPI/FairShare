#!/usr/bin/env python3
"""Check if Jain index varies over time"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd

for policy in ['static_equal', 'static_proportional', 'priority', 'rl']:
    csv_file = f'results/urban_congestion_{policy}.csv'
    
    try:
        df = pd.read_csv(csv_file)
        
        print(f"\n{policy.upper()} POLICY:")
        print(f"  Jain: min={df['jain'].min():.4f}, max={df['jain'].max():.4f}, std={df['jain'].std():.6f}")
        print(f"  Rate: min={df['mean_rate'].min()/1e6:.2f} Mbps, max={df['mean_rate'].max()/1e6:.2f} Mbps")
        print(f"  Allocated: min={df['num_allocated'].min()}, max={df['num_allocated'].max()}")
        
        # Check if ANY variation exists
        if df['jain'].std() < 0.001:
            print(f"  ⚠️ WARNING: Jain is CONSTANT (no variation)")
        
        # Check if num_allocated varies
        if df['num_allocated'].std() < 0.1:
            print(f"  ⚠️ WARNING: num_allocated is CONSTANT (no variation)")
    
    except FileNotFoundError:
        print(f"\n{policy.upper()}: File not found")
    except Exception as e:
        print(f"\n{policy.upper()}: Error - {e}")

