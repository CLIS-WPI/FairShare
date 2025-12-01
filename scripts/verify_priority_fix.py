#!/usr/bin/env python3
"""Verify Priority policy now differs from Static"""

import pandas as pd
import sys
import os

def verify_priority_fix():
    print("=" * 60)
    print("VERIFYING PRIORITY POLICY FIX")
    print("=" * 60)
    
    # Load results
    try:
        df_static = pd.read_csv('results/urban_congestion_static.csv')
        df_priority = pd.read_csv('results/urban_congestion_priority.csv')
        
        # Compare key metrics
        static_jain = df_static['jain'].mean()
        priority_jain = df_priority['jain'].mean()
        
        static_rate = df_static['mean_rate'].mean() / 1e6  # Convert to Mbps
        priority_rate = df_priority['mean_rate'].mean() / 1e6
        
        static_alloc = df_static['num_allocated'].mean()
        priority_alloc = df_priority['num_allocated'].mean()
        
        print(f"\nStatic Policy:")
        print(f"  Jain Index:     {static_jain:.4f} ± {df_static['jain'].std():.4f}")
        print(f"  Mean Rate:      {static_rate:.2f} Mbps")
        print(f"  Allocated Users: {static_alloc:.0f}")
        
        print(f"\nPriority Policy:")
        print(f"  Jain Index:     {priority_jain:.4f} ± {df_priority['jain'].std():.4f}")
        print(f"  Mean Rate:      {priority_rate:.2f} Mbps")
        print(f"  Allocated Users: {priority_alloc:.0f}")
        
        print("\n" + "=" * 60)
        
        # Check if different
        jain_diff = abs(static_jain - priority_jain)
        rate_diff = abs(static_rate - priority_rate)
        alloc_diff = abs(static_alloc - priority_alloc)
        
        if jain_diff > 0.01 or rate_diff > 0.1 or alloc_diff > 10:
            print("✓ SUCCESS: Priority differs from Static")
            print(f"  Jain difference:    {jain_diff:.4f}")
            print(f"  Rate difference:    {rate_diff:.2f} Mbps")
            print(f"  Allocation diff:    {alloc_diff:.0f} users")
            
            # Priority should have LOWER Jain (less fair, favors high priority)
            if priority_jain < static_jain:
                print("\n✓ Expected: Priority has lower fairness (favors high-priority users)")
                print("  This is CORRECT behavior - priority policy should be less fair")
                print("  because it gives more resources to high-priority users")
            else:
                print("\n⚠️  Unexpected: Priority has higher fairness than Static")
                print("  This might indicate priorities are not being used effectively")
            
            return True
        else:
            print("✗ FAILED: Priority still identical to Static")
            print(f"  Jain diff: {jain_diff:.4f} (need > 0.01)")
            print(f"  Rate diff: {rate_diff:.2f} Mbps (need > 0.1)")
            print(f"  Allocation diff: {alloc_diff:.0f} (need > 10)")
            print("\n  Check if fix was applied correctly")
            return False
            
    except FileNotFoundError as e:
        print(f"✗ ERROR: Missing file - {e}")
        print("  Run Priority simulation first:")
        print("    python -m src.main --scenario urban_congestion_phase4 --policy priority --duration-s 60")
        return False
    except KeyError as e:
        print(f"✗ ERROR: Missing column in CSV - {e}")
        print("  Check CSV file format")
        return False
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = verify_priority_fix()
    sys.exit(0 if success else 1)

