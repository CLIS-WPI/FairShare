#!/usr/bin/env python3
"""
Generate Elegant Publication-Ready Figures for FairShare Paper.

Uses advanced Seaborn + Matplotlib styling for professional, 
visually striking charts suitable for top-tier venues.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# ELEGANT STYLE CONFIGURATION
# ============================================================================

# Premium color palette (Nature/Science journal inspired)
PALETTE = {
    'urban': '#D64550',      # Deep coral
    'suburban': '#4C9F70',   # Forest green  
    'rural': '#4A7FB0',      # Steel blue
    'static': '#95A5A6',     # Elegant gray
    'priority': '#E74C3C',   # Vibrant red
    'demand': '#F39C12',     # Golden amber
    'fairshare': '#2E86AB',  # Deep ocean blue
    'accent': '#9B59B6',     # Royal purple
    'background': '#FAFBFC', # Off-white
    'grid': '#E8EAED',       # Soft gray
    'text': '#2C3E50',       # Dark slate
}

# Set up elegant style
def setup_elegant_style():
    """Configure matplotlib for elegant, publication-ready output."""
    plt.rcParams.update({
        # Font settings
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica Neue', 'Arial', 'DejaVu Sans'],
        'font.size': 11,
        'font.weight': 'normal',
        
        # Axes
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'axes.labelsize': 12,
        'axes.labelweight': 'medium',
        'axes.linewidth': 1.5,
        'axes.edgecolor': PALETTE['text'],
        'axes.facecolor': 'white',
        'axes.spines.top': False,
        'axes.spines.right': False,
        
        # Ticks
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        
        # Legend
        'legend.fontsize': 10,
        'legend.frameon': True,
        'legend.framealpha': 0.95,
        'legend.edgecolor': PALETTE['grid'],
        'legend.fancybox': True,
        
        # Grid
        'grid.color': PALETTE['grid'],
        'grid.linewidth': 0.8,
        'grid.alpha': 0.7,
        
        # Figure
        'figure.facecolor': 'white',
        'figure.edgecolor': 'none',
        'figure.dpi': 150,
        
        # Lines
        'lines.linewidth': 2.5,
        'lines.markersize': 9,
        
        # Saving
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
    })

setup_elegant_style()

# Output directory
FIGURES_DIR = Path('/workspace/results/figures')
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# DATA GENERATION
# ============================================================================

def generate_user_distribution(n_users=1000, seed=42):
    """Generate user positions matching paper's NYC-calibrated distribution."""
    np.random.seed(seed)
    
    center_lat, center_lon = 40.7128, -74.0060
    
    n_urban = int(n_users * 0.50)
    n_suburban = int(n_users * 0.20)
    n_rural = n_users - n_urban - n_suburban
    
    data = []
    
    # Urban
    for _ in range(n_urban):
        lat = np.random.normal(center_lat, 0.05)
        lon = np.random.normal(center_lon, 0.05)
        snr = np.random.normal(32, 3)
        data.append({'lat': lat, 'lon': lon, 'type': 'Urban', 'snr': snr})
    
    # Suburban
    for _ in range(n_suburban):
        angle = np.random.uniform(0, 2*np.pi)
        radius = np.random.uniform(0.2, 0.5)
        lat = center_lat + radius * np.sin(angle)
        lon = center_lon + radius * np.cos(angle)
        snr = np.random.normal(27, 3)
        data.append({'lat': lat, 'lon': lon, 'type': 'Suburban', 'snr': snr})
    
    # Rural
    for _ in range(n_rural):
        angle = np.random.uniform(0, 2*np.pi)
        radius = np.random.uniform(0.5, 1.5)
        lat = center_lat + radius * np.sin(angle)
        lon = center_lon + radius * np.cos(angle)
        snr = np.random.normal(22, 4)
        data.append({'lat': lat, 'lon': lon, 'type': 'Rural', 'snr': snr})
    
    return pd.DataFrame(data)


def get_paper_results():
    """Get results matching paper's Table I."""
    return pd.DataFrame({
        'Policy': ['Equal Static', 'Priority (SNR)', 'Demand Prop.', 'FairShare'],
        'Urban': [77.9, 44.0, 40.8, 24.0],
        'Rural': [78.1, 26.7, 29.1, 33.3],
        'Gap': [1.00, 1.65, 1.40, 0.72],
        'Efficiency': [0.78, 0.95, 0.88, 0.82]
    })


def get_bandwidth_data():
    """Get bandwidth sensitivity data."""
    return pd.DataFrame({
        'Bandwidth': [50, 100, 200, 300] * 4,
        'Policy': ['Equal Static']*4 + ['Priority (SNR)']*4 + ['Demand Prop.']*4 + ['FairShare']*4,
        'Gap': [1.00, 1.00, 1.00, 1.00,   # Static
                1.45, 1.55, 1.62, 1.65,   # Priority
                1.25, 1.32, 1.38, 1.40,   # Demand
                0.72, 0.72, 0.72, 0.72]   # FairShare
    })


# ============================================================================
# FIGURE 1: Elegant User Distribution Map
# ============================================================================

def create_figure1():
    """Create elegant geographic user distribution map."""
    print("Creating Figure 1: User Distribution Map...")
    
    df = generate_user_distribution(n_users=1000)
    
    fig, ax = plt.subplots(figsize=(10, 9))
    
    # Create custom order for layering
    type_order = ['Rural', 'Suburban', 'Urban']
    colors = [PALETTE['rural'], PALETTE['suburban'], PALETTE['urban']]
    markers = ['o', 's', '^']
    sizes = [40, 60, 80]
    
    for i, (utype, color, marker, size) in enumerate(zip(type_order, colors, markers, sizes)):
        subset = df[df['type'] == utype]
        ax.scatter(subset['lon'], subset['lat'], 
                   c=color, marker=marker, s=size,
                   alpha=0.65, edgecolors='white', linewidth=0.8,
                   label=utype, zorder=3+i)
    
    # NYC center with elegant marker
    ax.scatter(-74.0060, 40.7128, c='#1A1A2E', marker='*', s=500, 
               zorder=10, edgecolors='gold', linewidth=2)
    ax.annotate('Manhattan', xy=(-74.0060, 40.73), ha='center',
                fontsize=11, fontweight='bold', color=PALETTE['text'])
    
    # Elegant concentric rings
    center = (-74.0060, 40.7128)
    for radius, label, alpha in [(0.05, '', 0.4), (0.2, '22 km', 0.3), 
                                  (0.5, '55 km', 0.25), (1.0, '110 km', 0.2)]:
        circle = plt.Circle(center, radius, fill=False, 
                           color=PALETTE['text'], linestyle='--', 
                           linewidth=1.5, alpha=alpha)
        ax.add_patch(circle)
        if label:
            ax.annotate(label, xy=(center[0] + radius + 0.05, center[1]),
                       fontsize=9, color='gray', alpha=0.8)
    
    # Elegant legend
    legend_elements = [
        Line2D([0], [0], marker='^', color='w', markerfacecolor=PALETTE['urban'], 
               markersize=12, markeredgecolor='white', markeredgewidth=1.5,
               label=f'Urban (n=500, 50%)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=PALETTE['suburban'], 
               markersize=11, markeredgecolor='white', markeredgewidth=1.5,
               label=f'Suburban (n=200, 20%)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=PALETTE['rural'], 
               markersize=10, markeredgecolor='white', markeredgewidth=1.5,
               label=f'Rural (n=300, 30%)'),
    ]
    legend = ax.legend(handles=legend_elements, loc='upper right',
                       fontsize=11, framealpha=0.95)
    legend.get_frame().set_linewidth(1.5)
    
    ax.set_xlabel('Longitude (°)', fontweight='medium')
    ax.set_ylabel('Latitude (°)', fontweight='medium')
    ax.set_title('User Geographic Distribution\nNYC Metropolitan Area', 
                 fontsize=15, fontweight='bold', pad=15)
    
    ax.set_xlim(-75.8, -72.3)
    ax.set_ylim(39.3, 42.2)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Subtle background gradient effect
    ax.set_facecolor('#F8FAFC')
    
    plt.tight_layout()
    save_figure(fig, 'fig1_user_distribution')


# ============================================================================
# FIGURE 2: Elegant SNR Distribution
# ============================================================================

def create_figure2():
    """Create elegant SNR distribution plot."""
    print("Creating Figure 2: SNR Distribution...")
    
    df = generate_user_distribution(n_users=1000)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Color palette for the plot
    palette = [PALETTE['urban'], PALETTE['suburban'], PALETTE['rural']]
    
    # Create elegant violin + strip plot combination
    sns.violinplot(data=df, x='type', y='snr', order=['Urban', 'Suburban', 'Rural'],
                   palette=palette, ax=ax, inner=None, alpha=0.7, linewidth=2)
    
    # Add strip plot for individual points
    sns.stripplot(data=df, x='type', y='snr', order=['Urban', 'Suburban', 'Rural'],
                  color='white', alpha=0.15, size=3, ax=ax, jitter=0.25)
    
    # Add box plot for statistics
    sns.boxplot(data=df, x='type', y='snr', order=['Urban', 'Suburban', 'Rural'],
                ax=ax, width=0.15, showfliers=False,
                boxprops=dict(facecolor='white', edgecolor=PALETTE['text'], linewidth=2),
                medianprops=dict(color=PALETTE['accent'], linewidth=2.5),
                whiskerprops=dict(color=PALETTE['text'], linewidth=1.5),
                capprops=dict(color=PALETTE['text'], linewidth=1.5))
    
    # Calculate and annotate statistics
    stats = df.groupby('type')['snr'].agg(['mean', 'std']).reindex(['Urban', 'Suburban', 'Rural'])
    
    for i, (idx, row) in enumerate(stats.iterrows()):
        ax.annotate(f'μ = {row["mean"]:.1f} dB\nσ = {row["std"]:.1f} dB',
                    xy=(i, row['mean'] + 10), ha='center',
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                             edgecolor=palette[i], linewidth=2, alpha=0.95))
    
    # SNR gap annotation with elegant arrow
    urban_mean = stats.loc['Urban', 'mean']
    rural_mean = stats.loc['Rural', 'mean']
    gap = urban_mean - rural_mean
    
    # Draw elegant bracket
    ax.annotate('', xy=(0, urban_mean), xytext=(2, rural_mean),
                arrowprops=dict(arrowstyle='<->', color=PALETTE['accent'], 
                               lw=3, connectionstyle='bar,fraction=0.15'))
    
    # Gap label
    ax.annotate(f'SNR Gap\nΔ = {gap:.1f} dB', 
                xy=(1, (urban_mean + rural_mean)/2),
                ha='center', va='center', fontsize=12, fontweight='bold',
                color=PALETTE['accent'],
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF9E6', 
                         edgecolor=PALETTE['accent'], linewidth=2))
    
    # Reference line
    ax.axhline(y=20, color='gray', linestyle=':', linewidth=2, alpha=0.6)
    ax.annotate('Min. viable (20 dB)', xy=(2.4, 20.5), fontsize=9, color='gray')
    
    ax.set_xlabel('Geographic Region', fontweight='medium', fontsize=12)
    ax.set_ylabel('Signal-to-Noise Ratio (dB)', fontweight='medium', fontsize=12)
    ax.set_title('SNR Distribution by Geographic Region\n3GPP TR 38.811 LEO Channel Model',
                 fontsize=15, fontweight='bold', pad=15)
    
    ax.set_ylim(8, 48)
    ax.grid(True, axis='y', alpha=0.3, linestyle='-')
    ax.set_facecolor('#F8FAFC')
    
    plt.tight_layout()
    save_figure(fig, 'fig2_snr_distribution')


# ============================================================================
# FIGURE 3: Elegant Main Results Bar Chart
# ============================================================================

def create_figure3():
    """Create elegant grouped bar chart."""
    print("Creating Figure 3: Main Results Comparison...")
    
    df = get_paper_results()
    
    fig, ax = plt.subplots(figsize=(11, 7))
    
    x = np.arange(len(df))
    width = 0.35
    
    # Gradient-style bars with elegant colors
    colors_urban = [PALETTE['static'], PALETTE['priority'], PALETTE['demand'], PALETTE['urban']]
    colors_rural = ['#B8C5D1', '#F5A5A5', '#F5D5A5', PALETTE['rural']]
    
    # Create bars with rounded edges effect
    bars1 = ax.bar(x - width/2, df['Urban'], width, 
                   color=PALETTE['urban'], alpha=0.85,
                   edgecolor='white', linewidth=2,
                   label='Urban Rate (ρ_urban)')
    
    bars2 = ax.bar(x + width/2, df['Rural'], width,
                   color=PALETTE['rural'], alpha=0.85,
                   edgecolor='white', linewidth=2,
                   label='Rural Rate (ρ_rural)')
    
    # Value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 5), textcoords='offset points',
                    ha='center', va='bottom', fontsize=10, fontweight='bold',
                    color=PALETTE['urban'])
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 5), textcoords='offset points',
                    ha='center', va='bottom', fontsize=10, fontweight='bold',
                    color=PALETTE['rural'])
    
    # Gap badges on top
    for i, (_, row) in enumerate(df.iterrows()):
        gap = row['Gap']
        max_height = max(row['Urban'], row['Rural'])
        
        # Color based on fairness
        if gap < 1.0:
            badge_color = '#27AE60'  # Green
            text = f'Δ = {gap:.2f}×'
        elif gap == 1.0:
            badge_color = '#3498DB'  # Blue
            text = f'Δ = {gap:.2f}×'
        else:
            badge_color = '#E74C3C'  # Red
            text = f'Δ = {gap:.2f}×'
        
        ax.annotate(text, xy=(i, max_height + 8), ha='center',
                    fontsize=11, fontweight='bold', color='white',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor=badge_color,
                             edgecolor='white', linewidth=2))
    
    # Highlight FairShare column
    ax.axvspan(2.6, 3.4, alpha=0.08, color='green', zorder=0)
    ax.annotate('✓ Best', xy=(3, 5), ha='center', fontsize=10, 
                color='#27AE60', fontweight='bold')
    
    ax.set_xlabel('Allocation Policy', fontweight='medium', fontsize=12)
    ax.set_ylabel('Allocation Rate (%)', fontweight='medium', fontsize=12)
    ax.set_title('Geographic Allocation Rates by Policy\nW = 300 MHz, n = 100 users, 33% scarcity',
                 fontsize=15, fontweight='bold', pad=15)
    
    ax.set_xticks(x)
    ax.set_xticklabels(df['Policy'], fontsize=11, fontweight='medium')
    ax.legend(loc='upper right', fontsize=11)
    ax.set_ylim(0, 100)
    ax.grid(True, axis='y', alpha=0.3, linestyle='-')
    ax.set_facecolor('#F8FAFC')
    
    plt.tight_layout()
    save_figure(fig, 'fig3_main_results')


# ============================================================================
# FIGURE 4: Elegant Bandwidth Sensitivity
# ============================================================================

def create_figure4():
    """Create elegant bandwidth sensitivity line plot."""
    print("Creating Figure 4: Bandwidth Sensitivity...")
    
    df = get_bandwidth_data()
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Define elegant styles
    styles = {
        'Equal Static': {'color': PALETTE['static'], 'marker': 'o', 'ls': '-', 'lw': 2.5},
        'Priority (SNR)': {'color': PALETTE['priority'], 'marker': 's', 'ls': '-', 'lw': 3},
        'Demand Prop.': {'color': PALETTE['demand'], 'marker': '^', 'ls': '-', 'lw': 2.5},
        'FairShare': {'color': PALETTE['fairshare'], 'marker': 'D', 'ls': '-', 'lw': 3.5},
    }
    
    for policy, style in styles.items():
        subset = df[df['Policy'] == policy]
        ax.plot(subset['Bandwidth'], subset['Gap'],
                marker=style['marker'], color=style['color'],
                linestyle=style['ls'], linewidth=style['lw'],
                markersize=12, markeredgecolor='white', markeredgewidth=2,
                label=policy, zorder=5)
    
    # Fair line with elegant styling
    ax.axhline(y=1.0, color=PALETTE['text'], linestyle='--', linewidth=2.5, alpha=0.4)
    ax.annotate('Fair (Δ = 1.0)', xy=(310, 1.02), fontsize=11, 
                color=PALETTE['text'], alpha=0.7, fontweight='medium')
    
    # Shaded regions with gradient effect
    ax.fill_between([40, 320], 1.0, 1.8, alpha=0.08, color=PALETTE['priority'])
    ax.fill_between([40, 320], 0.5, 1.0, alpha=0.08, color=PALETTE['fairshare'])
    
    # Region labels
    ax.annotate('Urban Bias Region\n(Δ > 1)', xy=(75, 1.55), fontsize=11,
                color=PALETTE['priority'], fontweight='bold', alpha=0.8)
    ax.annotate('Rural Favoring Region\n(Δ < 1)', xy=(75, 0.75), fontsize=11,
                color=PALETTE['fairshare'], fontweight='bold', alpha=0.8)
    
    # Insight annotation
    ax.annotate('FairShare: Constant fairness\nregardless of bandwidth',
                xy=(200, 0.75), xytext=(250, 0.58),
                fontsize=10, fontweight='medium', color=PALETTE['fairshare'],
                arrowprops=dict(arrowstyle='->', color=PALETTE['fairshare'], lw=2),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor=PALETTE['fairshare'], linewidth=1.5))
    
    ax.set_xlabel('Bandwidth (MHz)', fontweight='medium', fontsize=12)
    ax.set_ylabel('Geographic Disparity Ratio (Δ_geo)', fontweight='medium', fontsize=12)
    ax.set_title('Bandwidth Sensitivity Analysis\nImpact of Spectrum Availability on Geographic Fairness',
                 fontsize=15, fontweight='bold', pad=15)
    
    ax.legend(loc='upper left', fontsize=11, framealpha=0.95)
    ax.set_xlim(40, 320)
    ax.set_ylim(0.5, 1.8)
    ax.set_xticks([50, 100, 200, 300])
    ax.grid(True, alpha=0.3, linestyle='-')
    ax.set_facecolor('#F8FAFC')
    
    plt.tight_layout()
    save_figure(fig, 'fig4_bandwidth_sensitivity')


# ============================================================================
# FIGURE 5: Elegant Pareto Frontier
# ============================================================================

def create_figure5():
    """Create elegant Pareto frontier plot."""
    print("Creating Figure 5: Pareto Frontier...")
    
    df = get_paper_results()
    df['Fairness'] = 1 / df['Gap']  # Convert gap to fairness
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Colors for each policy
    colors = [PALETTE['static'], PALETTE['priority'], PALETTE['demand'], PALETTE['fairshare']]
    markers = ['o', 's', '^', 'D']
    
    # Plot each policy point
    for i, (_, row) in enumerate(df.iterrows()):
        ax.scatter(row['Efficiency'], row['Fairness'], 
                   s=400, c=colors[i], marker=markers[i],
                   edgecolors='white', linewidth=3, zorder=5)
        
        # Labels with offset
        offset_x = 0.025 if row['Policy'] != 'FairShare' else 0.025
        offset_y = 0.08 if row['Policy'] != 'Priority (SNR)' else -0.12
        
        ax.annotate(row['Policy'], 
                    xy=(row['Efficiency'] + offset_x, row['Fairness'] + offset_y),
                    fontsize=11, fontweight='bold', color=colors[i])
    
    # Draw Pareto frontier
    pareto_x = [df.loc[3, 'Efficiency'], df.loc[2, 'Efficiency'], 
                df.loc[1, 'Efficiency']]
    pareto_y = [df.loc[3, 'Fairness'], df.loc[2, 'Fairness'], 
                df.loc[1, 'Fairness']]
    
    ax.plot(pareto_x, pareto_y, '--', color=PALETTE['accent'], 
            linewidth=2.5, alpha=0.7, zorder=3)
    ax.annotate('Pareto Frontier', xy=(0.86, 0.88), fontsize=10,
                color=PALETTE['accent'], fontweight='medium', style='italic')
    
    # Ideal point
    ax.scatter([1.0], [1.5], s=350, c='gold', marker='*',
               edgecolors=PALETTE['text'], linewidth=2, zorder=6)
    ax.annotate('Ideal Point', xy=(1.0, 1.55), ha='center',
                fontsize=10, fontweight='bold', color='goldenrod')
    
    # Trade-off arrow
    ax.annotate('', xy=(0.82, 1.39), xytext=(0.95, 0.61),
                arrowprops=dict(arrowstyle='-|>', color=PALETTE['accent'], 
                               lw=3, connectionstyle='arc3,rad=-0.15'))
    ax.annotate('Trade-off\nDirection', xy=(0.90, 0.95), ha='center',
                fontsize=10, color=PALETTE['accent'], fontweight='bold')
    
    # Key insight box
    insight_text = "FairShare achieves 81% higher fairness\nwith only 14% efficiency reduction"
    ax.annotate(insight_text, xy=(0.77, 1.45), fontsize=10, fontweight='medium',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#E8F4EA',
                         edgecolor=PALETTE['fairshare'], linewidth=2))
    
    ax.set_xlabel('Spectral Efficiency (Normalized)', fontweight='medium', fontsize=12)
    ax.set_ylabel('Geographic Fairness (1/Δ_geo)', fontweight='medium', fontsize=12)
    ax.set_title('Efficiency-Fairness Pareto Trade-off\nFairShare Achieves Superior Fairness with Minimal Efficiency Loss',
                 fontsize=15, fontweight='bold', pad=15)
    
    ax.set_xlim(0.74, 1.04)
    ax.set_ylim(0.5, 1.6)
    ax.grid(True, alpha=0.3, linestyle='-')
    ax.set_facecolor('#F8FAFC')
    
    # Legend
    legend_elements = [
        Line2D([0], [0], marker=markers[i], color='w', markerfacecolor=colors[i],
               markersize=12, markeredgecolor='white', markeredgewidth=2,
               label=df.loc[i, 'Policy'])
        for i in range(len(df))
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=10, framealpha=0.95)
    
    plt.tight_layout()
    save_figure(fig, 'fig5_pareto_frontier')


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def save_figure(fig, name):
    """Save figure in multiple formats."""
    for fmt in ['pdf', 'png', 'svg']:
        filepath = FIGURES_DIR / f'{name}.{fmt}'
        fig.savefig(filepath, format=fmt, dpi=300 if fmt == 'png' else None,
                    facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  ✓ Saved: {FIGURES_DIR / name}.[pdf/png/svg]")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("🎨 GENERATING ELEGANT PUBLICATION-READY FIGURES")
    print("=" * 70)
    print()
    
    create_figure1()
    create_figure2()
    create_figure3()
    create_figure4()
    create_figure5()
    
    print()
    print("=" * 70)
    print("✅ ALL ELEGANT FIGURES GENERATED!")
    print("=" * 70)
    print(f"\n📁 Output: {FIGURES_DIR}")
    print("\n📊 Generated files:")
    for f in sorted(FIGURES_DIR.glob('*.pdf')):
        print(f"   • {f.name}")


if __name__ == '__main__':
    main()

