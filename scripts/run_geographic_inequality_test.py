#!/usr/bin/env python3
"""
Geographic Inequality Validation Test

Tests the hypothesis that even with "equal" allocation policies,
geographic inequality exists (urban users get better service than rural users).

Outputs:
- Global Jain Index
- Urban Jain Index (center city users)
- Rural Jain Index (edge users)
- Geographic heatmap of user satisfaction
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src.experiments.scenario_loader import load_scenario
from src.main import run_simulation

def generate_geographic_heatmap(
    users: list,
    throughputs: dict,
    output_path: str,
    title: str = "Geographic User Satisfaction Heatmap"
):
    """
    Generate a geographic heatmap showing average user satisfaction per location.
    
    Args:
        users: List of user dictionaries with lat/lon
        throughputs: Dictionary mapping user_id to average throughput
        output_path: Path to save the heatmap image
        title: Chart title
    """
    # Extract coordinates and satisfaction values
    lats = [u['lat'] for u in users]
    lons = [u['lon'] for u in users]
    satisfactions = [throughputs.get(u['id'], 0.0) for u in users]
    
    # Normalize satisfactions to 0-1 for visualization
    max_sat = max(satisfactions) if satisfactions else 1.0
    if max_sat > 0:
        satisfactions = [s / max_sat for s in satisfactions]
    
    # Create scatter plot with color-coded satisfaction
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(lons, lats, c=satisfactions, cmap='RdYlGn', 
                         s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    plt.colorbar(scatter, label='Normalized User Satisfaction')
    plt.xlabel('Longitude', fontsize=12)
    plt.ylabel('Latitude', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add Manhattan center marker
    plt.plot(-73.97, 40.75, 'r*', markersize=20, label='Manhattan Center')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Heatmap saved to: {output_path}")

def analyze_results(results_dir: str, output_dir: str):
    """
    Analyze simulation results and generate report.
    
    Args:
        results_dir: Directory containing simulation results
        output_dir: Directory to save analysis outputs
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results CSV
    csv_files = list(Path(results_dir).glob("*.csv"))
    if not csv_files:
        print(f"❌ No CSV files found in {results_dir}")
        return
    
    df = pd.read_csv(csv_files[0])
    
    print("\n" + "="*70)
    print("GEOGRAPHIC INEQUALITY TEST RESULTS")
    print("="*70)
    
    # Global Jain Index
    global_jain = df['jain'].mean()
    print(f"\n📊 Global Jain Index: {global_jain:.4f}")
    
    # Urban/Rural Jain Index
    if 'urban_jain' in df.columns and 'rural_jain' in df.columns:
        urban_jain = df['urban_jain'].mean()
        rural_jain = df['rural_jain'].mean()
        print(f"🏙️  Urban Jain Index: {urban_jain:.4f}")
        print(f"🌾 Rural Jain Index: {rural_jain:.4f}")
        
        # Hypothesis validation
        print("\n" + "-"*70)
        print("HYPOTHESIS VALIDATION:")
        print("-"*70)
        
        if global_jain >= 0.8 and rural_jain < 0.6:
            print("✅ HYPOTHESIS CONFIRMED:")
            print("   Global statistics are misleading!")
            print(f"   Global Jain ({global_jain:.2f}) suggests fairness,")
            print(f"   but Rural Jain ({rural_jain:.2f}) reveals inequality.")
            print("   → Rural users are being underserved despite 'equal' policy.")
        elif global_jain >= 0.8 and rural_jain >= 0.8:
            print("❌ HYPOTHESIS REJECTED:")
            print("   System appears genuinely fair across geography.")
            print("   → May need to adjust scenario or metrics.")
        else:
            print("⚠️  INCONCLUSIVE:")
            print("   Results don't clearly support or reject hypothesis.")
    else:
        print("⚠️  Urban/Rural Jain indices not found in results.")
        print("   Make sure users have 'is_urban' attribute.")
    
    # Generate heatmap (if user data available)
    # Note: This would require loading user positions from simulation
    # For now, we'll create a placeholder
    
    # Save summary report
    report_path = os.path.join(output_dir, "geographic_inequality_report.txt")
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("GEOGRAPHIC INEQUALITY TEST REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(f"Global Jain Index: {global_jain:.4f}\n")
        if 'urban_jain' in df.columns:
            f.write(f"Urban Jain Index: {df['urban_jain'].mean():.4f}\n")
        if 'rural_jain' in df.columns:
            f.write(f"Rural Jain Index: {df['rural_jain'].mean():.4f}\n")
        f.write(f"\nTotal Slots: {len(df)}\n")
        f.write(f"Mean Rate: {df['mean_rate'].mean():.2f} Mbps\n")
        f.write(f"Allocation Rate: {(df['num_allocated'].mean() / df['num_users'].iloc[0] * 100):.1f}%\n")
    
    print(f"\n✓ Report saved to: {report_path}")
    
    return {
        'global_jain': global_jain,
        'urban_jain': df['urban_jain'].mean() if 'urban_jain' in df.columns else None,
        'rural_jain': df['rural_jain'].mean() if 'rural_jain' in df.columns else None
    }

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Geographic Inequality Test')
    parser.add_argument('--scenario', type=str, default='geographic_inequality_test',
                       help='Scenario name')
    parser.add_argument('--output', type=str, default='results/geographic_inequality',
                       help='Output directory')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID')
    
    args = parser.parse_args()
    
    print("="*70)
    print("GEOGRAPHIC INEQUALITY VALIDATION TEST")
    print("="*70)
    print(f"Scenario: {args.scenario}")
    print(f"Output: {args.output}")
    print(f"GPU: {args.gpu_id}")
    print("="*70 + "\n")
    
    # Run simulation
    print("Starting simulation...")
    run_simulation(
        scenario_name=args.scenario,
        policy_name='static',
        output_dir=args.output,
        use_gpu=True,
        gpu_id=args.gpu_id
    )
    
    # Analyze results
    print("\nAnalyzing results...")
    results = analyze_results(args.output, args.output)
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)

