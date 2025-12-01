"""
Extract REAL measured results from experiments for DySPAN paper.

This script only reads actual CSV files - NO fabricated numbers.

Phase 6: Result Extraction

Usage:
    python scripts/extract_paper_results.py --scenario urban_congestion_phase4
"""

import argparse
import pandas as pd
from pathlib import Path
import json
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def check_file_exists(filepath: Path, description: str):
    """Check if required result file exists"""
    if not filepath.exists():
        print(f"❌ ERROR: {description} not found at {filepath}")
        print(f"   Run experiments first!")
        return False
    return True


def load_simulation_results(scenario: str):
    """Load REAL simulation results from CSV files"""
    print("\n=== Loading Simulation Results ===")
    
    results_dir = Path("results")
    data = []
    
    for policy in ['static', 'priority', 'fuzzy', 'dqn']:
        csv_file = results_dir / f"{scenario}_{policy}.csv"
        
        if not check_file_exists(csv_file, f"{policy} simulation results"):
            continue
        
        df = pd.read_csv(csv_file)
        
        # Extract REAL statistics
        result = {
            'Policy': policy.capitalize(),
            'Jain_mean': float(df['jain'].mean()),
            'Jain_std': float(df['jain'].std()),
            'Fuzzy_mean': float(df['fuzzy_fairness'].mean()),
            'Fuzzy_std': float(df['fuzzy_fairness'].std()),
            'Alpha_mean': float(df['alpha_1'].mean()),
            'Alpha_std': float(df['alpha_1'].std()),
            'Gini_mean': float(df['gini'].mean()) if 'gini' in df.columns else 0.0,
            'Gini_std': float(df['gini'].std()) if 'gini' in df.columns else 0.0,
            'Rate_mean': float(df['mean_rate'].mean()),
            'Rate_std': float(df['mean_rate'].std()),
        }
        
        data.append(result)
        print(f"✓ Loaded {policy}: Jain={result['Jain_mean']:.4f}±{result['Jain_std']:.4f}")
    
    if not data:
        print("❌ No simulation results found!")
        return None
    
    return pd.DataFrame(data)


def load_inference_results():
    """Load REAL inference benchmark results"""
    print("\n=== Loading Inference Benchmark ===")
    
    benchmark_file = Path("results/benchmarks/inference_benchmark_n100.csv")
    
    if not check_file_exists(benchmark_file, "Inference benchmark"):
        return None
    
    df = pd.read_csv(benchmark_file)
    print(f"✓ Loaded inference times for {len(df)} policies")
    
    for _, row in df.iterrows():
        print(f"  {row['policy']:10s}: {row['mean_ms']:.4f} ms (mean)")
    
    return df


def load_ablation_results(scenario: str):
    """Load REAL ablation study results"""
    print("\n=== Loading Ablation Study ===")
    
    ablation_file = Path(f"results/ablation/ablation_study_{scenario}.csv")
    
    if not check_file_exists(ablation_file, "Ablation study"):
        return None
    
    df = pd.read_csv(ablation_file)
    print(f"✓ Loaded ablation results for {len(df)} configurations")
    
    for _, row in df.iterrows():
        print(f"  {row['configuration']:20s}: Jain={row['jain_mean']:.4f}")
    
    return df


def compute_comparisons(sim_df, inf_df):
    """Compute REAL comparison metrics"""
    print("\n=== Computing Comparisons ===")
    
    if sim_df is None or len(sim_df) < 2:
        print("⚠️  Not enough data for comparisons")
        return
    
    # Fairness comparison
    fuzzy_row = sim_df[sim_df['Policy'] == 'Fuzzy']
    dqn_row = sim_df[sim_df['Policy'] == 'Dqn']
    
    if not fuzzy_row.empty and not dqn_row.empty:
        fuzzy_jain = fuzzy_row['Jain_mean'].values[0]
        dqn_jain = dqn_row['Jain_mean'].values[0]
        
        if fuzzy_jain > dqn_jain:
            improvement = ((fuzzy_jain - dqn_jain) / dqn_jain) * 100
            print(f"✓ Fuzzy is {improvement:.2f}% BETTER than DQN in fairness")
        else:
            degradation = ((dqn_jain - fuzzy_jain) / dqn_jain) * 100
            performance_ratio = (fuzzy_jain / dqn_jain) * 100
            print(f"⚠️  Fuzzy is {degradation:.2f}% worse than DQN")
            print(f"    (Achieves {performance_ratio:.1f}% of DQN performance)")
    
    # Speed comparison
    if inf_df is not None:
        fuzzy_time = inf_df[inf_df['policy'] == 'fuzzy']
        dqn_time = inf_df[inf_df['policy'] == 'dqn']
        
        if not fuzzy_time.empty and not dqn_time.empty:
            fuzzy_ms = fuzzy_time['mean_ms'].values[0]
            dqn_ms = dqn_time['mean_ms'].values[0]
            if fuzzy_ms > 0:
                speedup = dqn_ms / fuzzy_ms
                print(f"✓ Fuzzy is {speedup:.1f}x FASTER than DQN")


def generate_latex_tables(sim_df, inf_df, ablation_df, output_dir: Path):
    """Generate LaTeX tables from REAL data"""
    print("\n=== Generating LaTeX Tables ===")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Table 1: Fairness comparison
    if sim_df is not None and len(sim_df) > 0:
        table1 = sim_df.copy()
        # Format for paper
        table1['Jain Index'] = table1.apply(
            lambda row: f"${row['Jain_mean']:.3f} \\pm {row['Jain_std']:.3f}$", axis=1)
        table1['α-Fairness'] = table1.apply(
            lambda row: f"${row['Alpha_mean']:.3f} \\pm {row['Alpha_std']:.3f}$", axis=1)
        
        latex_file = output_dir / "table1_fairness.tex"
        try:
            table1[['Policy', 'Jain Index', 'α-Fairness']].to_latex(
                latex_file, index=False, escape=False)
            print(f"✓ Table 1 saved: {latex_file}")
        except Exception as e:
            print(f"⚠️  Could not generate LaTeX table: {e}")
            # Save as CSV instead
            table1[['Policy', 'Jain Index', 'α-Fairness']].to_csv(
                output_dir / "table1_fairness.csv", index=False)
            print(f"✓ Table 1 saved as CSV: {output_dir / 'table1_fairness.csv'}")
    
    # Table 2: Computational overhead
    if inf_df is not None and len(inf_df) > 0:
        table2 = inf_df.copy()
        latex_file = output_dir / "table2_inference.tex"
        try:
            table2[['policy', 'mean_ms', 'p95_ms', 'p99_ms']].to_latex(
                latex_file, index=False, float_format="%.4f")
            print(f"✓ Table 2 saved: {latex_file}")
        except Exception as e:
            print(f"⚠️  Could not generate LaTeX table: {e}")
            table2[['policy', 'mean_ms', 'p95_ms', 'p99_ms']].to_csv(
                output_dir / "table2_inference.csv", index=False)
            print(f"✓ Table 2 saved as CSV: {output_dir / 'table2_inference.csv'}")
    
    # Table 3: Ablation study
    if ablation_df is not None and len(ablation_df) > 0:
        table3 = ablation_df.copy()
        latex_file = output_dir / "table3_ablation.tex"
        try:
            table3[['configuration', 'n_inputs', 'jain_mean']].to_latex(
                latex_file, index=False, float_format="%.4f")
            print(f"✓ Table 3 saved: {latex_file}")
        except Exception as e:
            print(f"⚠️  Could not generate LaTeX table: {e}")
            table3[['configuration', 'n_inputs', 'jain_mean']].to_csv(
                output_dir / "table3_ablation.csv", index=False)
            print(f"✓ Table 3 saved as CSV: {output_dir / 'table3_ablation.csv'}")


def save_summary_json(sim_df, inf_df, ablation_df, output_file: Path):
    """Save all REAL results to JSON"""
    summary = {
        'simulation_results': sim_df.to_dict('records') if sim_df is not None else None,
        'inference_results': inf_df.to_dict('records') if inf_df is not None else None,
        'ablation_results': ablation_df.to_dict('records') if ablation_df is not None else None,
    }
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Complete results saved to: {output_file}")


def main(args):
    print("=" * 60)
    print("EXTRACTING REAL EXPERIMENTAL RESULTS FOR PAPER")
    print("=" * 60)
    
    # Load REAL results
    sim_df = load_simulation_results(args.scenario)
    inf_df = load_inference_results()
    ablation_df = load_ablation_results(args.scenario)
    
    # Compute comparisons
    compute_comparisons(sim_df, inf_df)
    
    # Generate outputs
    output_dir = Path("results/paper_tables")
    generate_latex_tables(sim_df, inf_df, ablation_df, output_dir)
    
    # Save complete summary
    summary_file = Path("results/REAL_RESULTS_SUMMARY.json")
    save_summary_json(sim_df, inf_df, ablation_df, summary_file)
    
    print("\n" + "=" * 60)
    print("✓ RESULT EXTRACTION COMPLETE")
    print("=" * 60)
    print("\n⚠️  IMPORTANT:")
    print("   Use ONLY these numbers in your paper")
    print("   All values are from actual measurements")
    print("   No fabricated or hypothetical data")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract REAL experimental results for paper")
    parser.add_argument("--scenario", type=str, default="urban_congestion_phase4",
                       help="Scenario name")
    args = parser.parse_args()
    main(args)

