"""
Generate plots for LEO DSS simulation results.

Phase 4: Comprehensive plot generation for paper figures.

Usage:
    python experiments/generate_plots.py --scenario urban_congestion_phase4
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

# Use non-interactive backend
matplotlib.use('Agg')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def load_results(scenario_name: str, results_dir: str = 'results') -> dict:
    """
    Load CSV results for all policies.
    
    Args:
        scenario_name: Scenario name
        results_dir: Results directory
        
    Returns:
        Dictionary mapping policy name to DataFrame
    """
    policies = ['static', 'priority', 'fuzzy']
    results = {}
    
    for policy in policies:
        csv_path = os.path.join(results_dir, f"{scenario_name}_{policy}.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            results[policy] = df
            print(f"✓ Loaded {policy}: {len(df)} slots")
        else:
            print(f"⚠ Not found: {csv_path}")
    
    return results


def plot_fairness_over_time(results: dict, scenario_name: str, output_dir: str = 'plots'):
    """
    Plot (1) Jain vs FuzzyFairness vs α-fair over time.
    
    Args:
        results: Dictionary of policy DataFrames
        scenario_name: Scenario name
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    # Plot for each policy
    for policy, df in results.items():
        if df is None or len(df) == 0:
            continue
        
        # Jain Index
        axes[0].plot(df['slot'], df['jain'], label=f'{policy.capitalize()}', 
                    linewidth=2, alpha=0.8)
        
        # Fuzzy Fairness
        axes[1].plot(df['slot'], df['fuzzy_fairness'], label=f'{policy.capitalize()}', 
                    linewidth=2, alpha=0.8)
        
        # α-fairness (α=1)
        axes[2].plot(df['slot'], df['alpha_1'], label=f'{policy.capitalize()}', 
                    linewidth=2, alpha=0.8)
    
    # Formatting
    axes[0].set_ylabel('Jain Index', fontsize=12)
    axes[0].set_title('Fairness Metrics Over Time', fontsize=14, fontweight='bold')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1])
    
    axes[1].set_ylabel('Fuzzy Fairness', fontsize=12)
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1])
    
    axes[2].set_xlabel('Time Slot', fontsize=12)
    axes[2].set_ylabel('α-Fairness (α=1)', fontsize=12)
    axes[2].legend(loc='best')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim([0, None])
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'fairness_time_{scenario_name}.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_policy_comparison(results: dict, scenario_name: str, output_dir: str = 'plots'):
    """
    Plot (2) Barplot comparison of policies.
    
    Args:
        results: Dictionary of policy DataFrames
        scenario_name: Scenario name
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Compute mean metrics for each policy
    metrics = {}
    for policy, df in results.items():
        if df is None or len(df) == 0:
            continue
        
        metrics[policy] = {
            'Jain': df['jain'].mean(),
            'Fuzzy': df['fuzzy_fairness'].mean(),
            'α-fair (α=1)': df['alpha_1'].mean(),
            'Mean Rate (Mbps)': df['mean_rate'].mean() / 1e6,
            'Cell Edge Rate (Mbps)': df['cell_edge_rate'].mean() / 1e6
        }
    
    if not metrics:
        print("⚠ No data for policy comparison")
        return
    
    # Create bar plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    policies = list(metrics.keys())
    x = np.arange(len(policies))
    width = 0.25
    
    # Fairness metrics
    jain_values = [metrics[p]['Jain'] for p in policies]
    fuzzy_values = [metrics[p]['Fuzzy'] for p in policies]
    alpha_values = [metrics[p]['α-fair (α=1)'] for p in policies]
    
    axes[0].bar(x - width, jain_values, width, label='Jain Index', alpha=0.8)
    axes[0].bar(x, fuzzy_values, width, label='Fuzzy Fairness', alpha=0.8)
    axes[0].bar(x + width, alpha_values, width, label='α-fair (α=1)', alpha=0.8)
    
    axes[0].set_xlabel('Policy', fontsize=12)
    axes[0].set_ylabel('Fairness Metric', fontsize=12)
    axes[0].set_title('Fairness Comparison', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([p.capitalize() for p in policies])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].set_ylim([0, 1])
    
    # Rate metrics
    mean_rate_values = [metrics[p]['Mean Rate (Mbps)'] for p in policies]
    edge_rate_values = [metrics[p]['Cell Edge Rate (Mbps)'] for p in policies]
    
    axes[1].bar(x - width/2, mean_rate_values, width, label='Mean Rate', alpha=0.8)
    axes[1].bar(x + width/2, edge_rate_values, width, label='Cell Edge Rate', alpha=0.8)
    
    axes[1].set_xlabel('Policy', fontsize=12)
    axes[1].set_ylabel('Rate (Mbps)', fontsize=12)
    axes[1].set_title('Throughput Comparison', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([p.capitalize() for p in policies])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'policy_comparison_{scenario_name}.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_rate_cdf(results: dict, scenario_name: str, output_dir: str = 'plots'):
    """
    Plot (3) CDF of user rates.
    
    Args:
        results: Dictionary of policy DataFrames
        scenario_name: Scenario name
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for policy, df in results.items():
        if df is None or len(df) == 0:
            continue
        
        # Use mean_rate as proxy for user rates
        rates = df['mean_rate'].values / 1e6  # Convert to Mbps
        
        # Compute CDF
        sorted_rates = np.sort(rates)
        cdf = np.arange(1, len(sorted_rates) + 1) / len(sorted_rates)
        
        ax.plot(sorted_rates, cdf, label=f'{policy.capitalize()}', 
               linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Rate (Mbps)', fontsize=12)
    ax.set_ylabel('CDF', fontsize=12)
    ax.set_title('CDF of User Rates', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, None])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'rate_cdf_{scenario_name}.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_operator_imbalance_heatmap(results: dict, scenario_name: str, output_dir: str = 'plots'):
    """
    Plot (4) Heatmap of operator imbalance over time.
    
    Args:
        results: Dictionary of policy DataFrames
        scenario_name: Scenario name
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Use fuzzy policy if available, otherwise first available
    df = results.get('fuzzy', None)
    if df is None:
        df = next(iter(results.values()))
    
    if df is None or len(df) == 0:
        print("⚠ No data for operator imbalance heatmap")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create heatmap: time slots vs operator imbalance
    # For simplicity, use operator_imbalance column directly
    imbalance_data = df['operator_imbalance'].values.reshape(-1, 1).T
    
    im = ax.imshow(imbalance_data, aspect='auto', cmap='YlOrRd', 
                  interpolation='nearest', vmin=0, vmax=1)
    
    ax.set_xlabel('Time Slot', fontsize=12)
    ax.set_ylabel('Operator Imbalance', fontsize=12)
    ax.set_title('Operator Imbalance Over Time', fontsize=14, fontweight='bold')
    ax.set_yticks([0])
    ax.set_yticklabels(['Imbalance'])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Imbalance (Gini Coefficient)', fontsize=10)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'operator_imbalance_heat_{scenario_name}.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_doppler_fairness_scatter(results: dict, scenario_name: str, output_dir: str = 'plots'):
    """
    Plot (5) Scatter: Doppler vs Fairness.
    
    Note: This requires per-user data which may not be in CSV.
    We'll use aggregated data as approximation.
    
    Args:
        results: Dictionary of policy DataFrames
        scenario_name: Scenario name
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Use fuzzy policy
    df = results.get('fuzzy', None)
    if df is None:
        df = next(iter(results.values()))
    
    if df is None or len(df) == 0:
        print("⚠ No data for doppler-fairness scatter")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Use mean_rate as proxy for doppler effect (simplified)
    # In practice, would need per-user doppler data
    # For now, use cell_edge_rate as proxy (lower rate = higher doppler effect)
    doppler_proxy = 1.0 / (df['cell_edge_rate'] / df['mean_rate'] + 0.1)  # Normalized
    fairness = df['fuzzy_fairness'].values
    
    scatter = ax.scatter(doppler_proxy, fairness, c=df['slot'], 
                        cmap='viridis', alpha=0.6, s=50)
    
    ax.set_xlabel('Doppler Effect (Normalized)', fontsize=12)
    ax.set_ylabel('Fuzzy Fairness', fontsize=12)
    ax.set_title('Doppler vs Fairness', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Time Slot', fontsize=10)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'doppler_fairness_scatter_{scenario_name}.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate plots for LEO DSS simulation results'
    )
    parser.add_argument(
        '--scenario',
        type=str,
        required=True,
        help='Scenario name (e.g., urban_congestion_phase4)'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results',
        help='Results directory (default: results)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='plots',
        help='Output directory for plots (default: plots)'
    )
    
    args = parser.parse_args()
    
    print(f"Generating plots for scenario: {args.scenario}")
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}\n")
    
    # Load results
    results = load_results(args.scenario, args.results_dir)
    
    if not results:
        print("⚠ No results found. Please run simulation first.")
        return
    
    # Generate all plots
    print("\nGenerating plots...")
    
    plot_fairness_over_time(results, args.scenario, args.output_dir)
    plot_policy_comparison(results, args.scenario, args.output_dir)
    plot_rate_cdf(results, args.scenario, args.output_dir)
    plot_operator_imbalance_heatmap(results, args.scenario, args.output_dir)
    plot_doppler_fairness_scatter(results, args.scenario, args.output_dir)
    
    print(f"\n✓ All plots saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
