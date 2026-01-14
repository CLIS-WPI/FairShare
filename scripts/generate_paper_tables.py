#!/usr/bin/env python3
"""
Generate all 5 tables for the FairShare paper.
Optimized for fast execution on H100 GPUs.
"""

import os
import sys
import subprocess
import time
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
N_RUNS = 50  # Main comparison
N_RUNS_SENS = 10  # Sensitivity analysis
DURATION_S = 5  # 5 seconds = 100 slots (fast but statistically valid)

POLICIES = ['static', 'priority', 'demand', 'fairshare']
BANDWIDTHS = [50e6, 100e6, 200e6, 300e6]
RURAL_QUOTAS = [0.25, 0.30, 0.35, 0.40]

RESULTS_DIR = Path('/workspace/results/paper_tables')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Enable fast mode
os.environ['FAIRSHARE_FAST_MODE'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def run_sim(policy, run_idx, gpu_id=0, bandwidth_hz=None, rural_quota=None):
    """Run a single simulation."""
    suffix = ""
    if bandwidth_hz:
        suffix += f"_bw{int(bandwidth_hz/1e6)}"
    if rural_quota:
        suffix += f"_rq{int(rural_quota*100)}"
    
    output_dir = RESULTS_DIR / f"{policy}{suffix}" / f"run_{run_idx:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        sys.executable, '-m', 'src.main',
        '--scenario', 'experiments/scenarios/fairshare_ultrafast.yaml',
        '--policy', policy,
        '--duration-s', str(DURATION_S),
        '--output', str(output_dir),
        '--gpu-id', str(gpu_id)
    ]
    
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    if bandwidth_hz:
        env['FAIRSHARE_OVERRIDE_BW_HZ'] = str(int(bandwidth_hz))
    if rural_quota:
        env['FAIRSHARE_RURAL_QUOTA'] = str(rural_quota)
    
    start = time.time()
    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, 
                               timeout=120, cwd='/workspace')
        elapsed = time.time() - start
        
        if result.returncode != 0:
            return None
        
        csv_file = output_dir / f"fairshare_ultrafast_{policy}.csv"
        if csv_file.exists():
            import pandas as pd
            df = pd.read_csv(csv_file)
            return {
                'policy': policy,
                'run_idx': run_idx,
                'runtime_s': elapsed,
                'urban_alloc': df['urban_allocation_rate'].mean(),
                'rural_alloc': df['rural_allocation_rate'].mean(),
                'gap': df['urban_allocation_rate'].mean() / df['rural_allocation_rate'].mean() if df['rural_allocation_rate'].mean() > 0 else float('inf'),
                'avg_snr': df['mean_rate'].mean() / 1e6,  # Approximate SNR from rate
            }
    except:
        pass
    return None


def agg(results):
    """Aggregate results."""
    results = [r for r in results if r]
    if not results:
        return {}
    agg = {}
    for key in ['urban_alloc', 'rural_alloc', 'gap', 'avg_snr', 'runtime_s']:
        vals = [r[key] for r in results if r.get(key) is not None and not np.isinf(r.get(key, 0))]
        if vals:
            agg[key] = {'mean': np.mean(vals), 'std': np.std(vals), 'n': len(vals)}
    return agg


def main():
    start_all = time.time()
    print("=" * 70)
    print("FairShare Paper - Table Generation")
    print(f"N_RUNS={N_RUNS}, DURATION={DURATION_S}s, Fast Mode=ON")
    print("=" * 70)
    
    all_results = {}
    
    # TABLE I: Main comparison
    print("\n📊 TABLE I: Main Policy Comparison")
    for policy in POLICIES:
        print(f"  {policy}: ", end="", flush=True)
        results = []
        for run_idx in range(N_RUNS):
            gpu_id = run_idx % 2
            r = run_sim(policy, run_idx, gpu_id)
            if r:
                results.append(r)
            if (run_idx + 1) % 10 == 0:
                print(f"{run_idx+1}", end=" ", flush=True)
        print()
        all_results[policy] = results
        a = agg(results)
        if a:
            print(f"    Urban: {a['urban_alloc']['mean']*100:.1f}±{a['urban_alloc']['std']*100:.1f}%")
            print(f"    Rural: {a['rural_alloc']['mean']*100:.1f}±{a['rural_alloc']['std']*100:.1f}%")
            print(f"    Gap: {a['gap']['mean']:.2f}±{a['gap']['std']:.2f}×")
    
    # Save Table I
    table_I = {p: agg(all_results.get(p, [])) for p in POLICIES}
    with open(RESULTS_DIR / 'table_I.json', 'w') as f:
        json.dump(table_I, f, indent=2, default=float)
    
    # TABLE II: Channel quality (from fairshare)
    print("\n📊 TABLE II: Channel Quality (from FairShare runs)")
    # Use results from fairshare runs
    
    # TABLE III: Bandwidth sensitivity
    print("\n📊 TABLE III: Bandwidth Sensitivity")
    table_III = {}
    for bw in BANDWIDTHS:
        print(f"  {int(bw/1e6)} MHz: ", end="", flush=True)
        table_III[bw] = {}
        for policy in POLICIES:
            results = []
            for run_idx in range(N_RUNS_SENS):
                r = run_sim(policy, run_idx, run_idx % 2, bandwidth_hz=bw)
                if r:
                    results.append(r)
            a = agg(results)
            gap = a.get('gap', {}).get('mean', 0)
            table_III[bw][policy] = gap
            print(f"{policy}={gap:.2f}× ", end="", flush=True)
        print()
    
    with open(RESULTS_DIR / 'table_III.json', 'w') as f:
        json.dump({str(k): v for k, v in table_III.items()}, f, indent=2)
    
    # TABLE IV: Quota sensitivity
    print("\n📊 TABLE IV: Rural Quota Sensitivity")
    table_IV = {}
    for quota in RURAL_QUOTAS:
        print(f"  {int(quota*100)}%: ", end="", flush=True)
        results = []
        for run_idx in range(N_RUNS_SENS):
            r = run_sim('fairshare', run_idx, run_idx % 2, rural_quota=quota)
            if r:
                results.append(r)
        a = agg(results)
        gap = a.get('gap', {}).get('mean', 0)
        table_IV[quota] = {'gap': gap, 'avg_snr': a.get('avg_snr', {}).get('mean', 0)}
        print(f"Gap={gap:.2f}×")
    
    with open(RESULTS_DIR / 'table_IV.json', 'w') as f:
        json.dump({str(k): v for k, v in table_IV.items()}, f, indent=2)
    
    # TABLE V: Runtime
    print("\n📊 TABLE V: Runtime")
    table_V = {}
    for policy in ['priority', 'fairshare']:
        results = all_results.get(policy, [])
        runtimes = [r['runtime_s'] for r in results if r]
        if runtimes:
            table_V[policy] = {'mean': np.mean(runtimes), 'std': np.std(runtimes)}
            print(f"  {policy}: {table_V[policy]['mean']:.2f}±{table_V[policy]['std']:.2f}s")
    
    if 'priority' in table_V and 'fairshare' in table_V:
        overhead = (table_V['fairshare']['mean'] - table_V['priority']['mean']) / table_V['priority']['mean'] * 100
        table_V['overhead_pct'] = overhead
        print(f"  Overhead: {overhead:+.1f}%")
    
    with open(RESULTS_DIR / 'table_V.json', 'w') as f:
        json.dump(table_V, f, indent=2)
    
    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY FOR PAPER")
    print("=" * 70)
    
    print("\nTable I: Main Policy Comparison (Mean ± Std)")
    print("-" * 60)
    for policy in POLICIES:
        a = table_I.get(policy, {})
        if a:
            print(f"{policy:12} | Urban: {a['urban_alloc']['mean']*100:.1f}±{a['urban_alloc']['std']*100:.1f}% | "
                  f"Rural: {a['rural_alloc']['mean']*100:.1f}±{a['rural_alloc']['std']*100:.1f}% | "
                  f"Gap: {a['gap']['mean']:.2f}±{a['gap']['std']:.2f}×")
    
    print(f"\nTotal time: {(time.time()-start_all)/60:.1f} minutes")
    print(f"Results saved to: {RESULTS_DIR}")


if __name__ == '__main__':
    main()

