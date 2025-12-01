"""
Radar charts for fairness metrics visualization.

Creates radar/spider charts showing multiple fairness dimensions,
useful for DySPAN paper visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import matplotlib.patches as mpatches


def plot_fairness_radar(metrics: Dict[str, float],
                       title: str = "Fairness Metrics Radar Chart",
                       save_path: Optional[str] = None) -> plt.Figure:
    """
    Create radar chart of fairness metrics.
    
    Args:
        metrics: Dictionary of metric names to values (0-1 scale)
        title: Chart title
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    # Metric labels
    labels = list(metrics.keys())
    values = [metrics[label] for label in labels]
    
    # Number of variables
    N = len(labels)
    
    # Compute angle for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    # Add values
    values += values[:1]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Plot
    ax.plot(angles, values, 'o-', linewidth=2, label='Fairness Metrics')
    ax.fill(angles, values, alpha=0.25)
    
    # Add labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
    ax.grid(True)
    
    plt.title(title, size=16, fontweight='bold', pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_comparison_radar(metrics_list: List[Dict[str, float]],
                         labels: List[str],
                         title: str = "Fairness Comparison",
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Create comparison radar chart for multiple scenarios.
    
    Args:
        metrics_list: List of metric dictionaries
        labels: Labels for each scenario
        title: Chart title
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    # Get all metric names (assume all have same keys)
    if len(metrics_list) == 0:
        raise ValueError("Need at least one metrics dictionary")
    
    metric_names = list(metrics_list[0].keys())
    N = len(metric_names)
    
    # Compute angles
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
    
    # Colors
    colors = plt.cm.tab10(np.linspace(0, 1, len(metrics_list)))
    
    # Plot each scenario
    for i, (metrics, label) in enumerate(zip(metrics_list, labels)):
        values = [metrics.get(name, 0) for name in metric_names]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=label, color=colors[i])
        ax.fill(angles, values, alpha=0.15, color=colors[i])
    
    # Add labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
    ax.grid(True)
    
    plt.title(title, size=16, fontweight='bold', pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_fairness_over_time(fairness_history: List[Dict],
                           metric_name: str = 'weighted_fairness',
                           title: str = "Fairness Over Time",
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot fairness metric over time.
    
    Args:
        fairness_history: List of fairness dictionaries from simulation
        metric_name: Name of metric to plot
        title: Chart title
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    times = [entry['time'] for entry in fairness_history]
    values = [entry.get(metric_name, 0) for entry in fairness_history]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(times, values, linewidth=2, marker='o', markersize=4)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

