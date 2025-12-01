#!/usr/bin/env python3
"""
Visualization and Reporting: Generate comparative plots for simulation results

Creates:
- Time evolution plots
- Operator/user fairness comparisons
- Efficiency vs fairness trade-offs
- Multi-dimensional fairness insights
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fairness.traditional import TraditionalFairness
from src.fairness.vector_metrics import VectorFairness


def load_simulation_results(results_dir: str = "~/fuzzy_fairness_results/simulation"):
    """Load simulation results from JSON and CSV files."""
    results_dir = Path(results_dir).expanduser()
    
    # Load JSON
    json_file = results_dir / "simulation_results.json"
    if json_file.exists():
        with open(json_file, 'r') as f:
            json_data = json.load(f)
    else:
        json_data = {}
    
    # Load CSV
    csv_file = results_dir / "policy_comparison.csv"
    if csv_file.exists():
        df = pd.read_csv(csv_file)
    else:
        df = pd.DataFrame()
    
    return json_data, df


def plot_policy_comparison(df: pd.DataFrame, output_dir: Path):
    """Create bar plot comparing policies across metrics."""
    if df.empty:
        print("No data for policy comparison")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Policy Comparison: Fairness Metrics', fontsize=16, fontweight='bold')
    
    policies = df['Policy'].values
    x = np.arange(len(policies))
    width = 0.35
    
    # Jain Index
    axes[0, 0].bar(x, df['Jain Index'], width, alpha=0.8, color='steelblue')
    axes[0, 0].set_ylabel('Jain Index', fontsize=12)
    axes[0, 0].set_title('Jain Index (Higher is Better)', fontsize=12)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(policies, rotation=15, ha='right')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].set_ylim([0, 1.1])
    
    # Gini Coefficient
    axes[0, 1].bar(x, df['Gini Coefficient'], width, alpha=0.8, color='coral')
    axes[0, 1].set_ylabel('Gini Coefficient', fontsize=12)
    axes[0, 1].set_title('Gini Coefficient (Lower is Better)', fontsize=12)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(policies, rotation=15, ha='right')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].set_ylim([0, 1.1])
    
    # Weighted Fairness
    axes[1, 0].bar(x, df['Weighted Fairness'], width, alpha=0.8, color='mediumseagreen')
    axes[1, 0].set_ylabel('Weighted Fairness', fontsize=12)
    axes[1, 0].set_title('Multi-Dimensional Fairness', fontsize=12)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(policies, rotation=15, ha='right')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    axes[1, 0].set_ylim([0, 1.1])
    
    # Users Served
    axes[1, 1].bar(x, df['Users Served'], width, alpha=0.8, color='gold')
    axes[1, 1].set_ylabel('Users Served', fontsize=12)
    axes[1, 1].set_title('Service Coverage', fontsize=12)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(policies, rotation=15, ha='right')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = output_dir / "policy_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_fairness_radar(df: pd.DataFrame, output_dir: Path):
    """Create radar chart comparing all fairness dimensions."""
    if df.empty:
        return
    
    # Prepare data for radar chart
    # Normalize Gini (invert since lower is better)
    df_normalized = df.copy()
    if 'Gini Coefficient' in df_normalized.columns:
        df_normalized['Gini Inverted'] = 1.0 - df_normalized['Gini Coefficient']
        metrics = ['Jain Index', 'Weighted Fairness', 'Distance Fairness', 'Gini Inverted']
    else:
        metrics = ['Jain Index', 'Weighted Fairness', 'Distance Fairness']
    
    # Number of variables
    N = len(metrics)
    
    # Compute angles for each metric
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Plot each policy
    colors = plt.cm.tab10(np.linspace(0, 1, len(df_normalized)))
    for idx, row in df_normalized.iterrows():
        values = [row[m] for m in metrics]
        values += values[:1]  # Complete the circle
        
        policy_name = row.get('Policy', f'Policy {idx}')
        ax.plot(angles, values, 'o-', linewidth=2, label=policy_name, color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])
    
    # Add labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
    ax.grid(True)
    
    plt.title('Multi-Dimensional Fairness Comparison', size=16, fontweight='bold', pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    output_path = output_dir / "fairness_radar.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_efficiency_fairness_tradeoff(df: pd.DataFrame, output_dir: Path):
    """Plot efficiency vs fairness trade-off."""
    if df.empty or 'Weighted Fairness' not in df.columns:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Use Users Served as proxy for efficiency
    if 'Users Served' in df.columns:
        efficiency = df['Users Served']
        fairness = df['Weighted Fairness']
        
        scatter = ax.scatter(efficiency, fairness, s=200, alpha=0.6, c=range(len(df)), cmap='viridis')
        
        # Annotate points
        for idx, row in df.iterrows():
            ax.annotate(row['Policy'], 
                       (efficiency.iloc[idx], fairness.iloc[idx]),
                       xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        ax.set_xlabel('Efficiency (Users Served)', fontsize=12)
        ax.set_ylabel('Fairness (Weighted)', fontsize=12)
        ax.set_title('Efficiency vs Fairness Trade-off', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = output_dir / "efficiency_fairness_tradeoff.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {output_path}")


def generate_summary_report(json_data: dict, df: pd.DataFrame, output_dir: Path):
    """Generate a text summary report."""
    report_path = output_dir / "summary_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("SIMULATION RESULTS SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        
        if json_data:
            f.write(f"Timestamp: {json_data.get('timestamp', 'N/A')}\n")
            f.write(f"Number of Users: {json_data.get('num_users', 'N/A')}\n")
            f.write(f"Operators: {', '.join(json_data.get('operators', []))}\n")
            f.write(f"Policies Tested: {', '.join(json_data.get('policies_tested', []))}\n\n")
        
        if not df.empty:
            f.write("Policy Comparison:\n")
            f.write("-" * 70 + "\n")
            f.write(df.to_string(index=False))
            f.write("\n\n")
            
            # Best policy analysis
            best_jain = df.loc[df['Jain Index'].idxmax()]
            best_weighted = df.loc[df['Weighted Fairness'].idxmax()]
            
            f.write("Best Policies:\n")
            f.write("-" * 70 + "\n")
            f.write(f"Best Jain Index: {best_jain['Policy']} ({best_jain['Jain Index']:.4f})\n")
            f.write(f"Best Weighted Fairness: {best_weighted['Policy']} ({best_weighted['Weighted Fairness']:.4f})\n\n")
            
            # Insights
            f.write("Key Insights:\n")
            f.write("-" * 70 + "\n")
            if best_jain['Jain Index'] > 0.9:
                f.write("• Excellent fairness achieved (Jain > 0.9)\n")
            if best_weighted['Weighted Fairness'] > 0.8:
                f.write("• Strong multi-dimensional fairness (Weighted > 0.8)\n")
            
            # Compare traditional vs multi-dimensional
            if 'Weighted Fairness' in df.columns and 'Jain Index' in df.columns:
                correlation = df['Jain Index'].corr(df['Weighted Fairness'])
                f.write(f"• Correlation (Jain vs Weighted): {correlation:.3f}\n")
                if correlation < 0.7:
                    f.write("  → Multi-dimensional metrics reveal different insights!\n")
    
    print(f"✓ Saved: {report_path}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize simulation results')
    parser.add_argument('--results-dir', type=str, 
                       default='~/fuzzy_fairness_results/simulation',
                       help='Results directory')
    parser.add_argument('--output-dir', type=str, default='plots',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Load results
    print("Loading simulation results...")
    json_data, df = load_simulation_results(args.results_dir)
    
    if df.empty:
        print("⚠ No results found. Please run simulation first.")
        return
    
    # Create output directory
    try:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        output_dir = Path.home() / "fuzzy_fairness_results/visualizations"
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Note: Using {output_dir} for results (permission issue with {args.output_dir})")
    
    print(f"\nGenerating visualizations...")
    print(f"Output directory: {output_dir}\n")
    
    # Generate plots
    plot_policy_comparison(df, output_dir)
    plot_fairness_radar(df, output_dir)
    plot_efficiency_fairness_tradeoff(df, output_dir)
    generate_summary_report(json_data, df, output_dir)
    
    print(f"\n✓ All visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
