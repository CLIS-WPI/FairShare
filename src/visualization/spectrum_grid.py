"""
Heatmap visualization of spectrum usage and interference.

Creates heatmaps showing beam usage, interference patterns, and
spectrum occupancy for DySPAN analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from datetime import datetime


def plot_spectrum_heatmap(spectrum_map: np.ndarray,
                          beam_ids: List[str],
                          frequency_bins: np.ndarray,
                          title: str = "Spectrum Occupancy Heatmap",
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot heatmap of spectrum occupancy across beams.
    
    Args:
        spectrum_map: 2D array [beam, frequency_bin] of power levels (dBm)
        beam_ids: List of beam identifiers
        frequency_bins: Frequency bin centers in Hz
        title: Chart title
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Convert to GHz for readability
    freq_ghz = frequency_bins / 1e9
    
    # Create heatmap
    im = ax.imshow(spectrum_map, aspect='auto', cmap='hot',
                   interpolation='nearest', origin='lower')
    
    # Set labels
    ax.set_xlabel('Frequency (GHz)', fontsize=12)
    ax.set_ylabel('Beam ID', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Set frequency ticks
    n_ticks = 10
    tick_indices = np.linspace(0, len(freq_ghz) - 1, n_ticks).astype(int)
    ax.set_xticks(tick_indices)
    ax.set_xticklabels([f'{freq_ghz[i]:.2f}' for i in tick_indices])
    
    # Set beam labels
    if len(beam_ids) <= 20:
        ax.set_yticks(range(len(beam_ids)))
        ax.set_yticklabels(beam_ids)
    else:
        # Show subset of labels
        tick_indices = np.linspace(0, len(beam_ids) - 1, 10).astype(int)
        ax.set_yticks(tick_indices)
        ax.set_yticklabels([beam_ids[i] for i in tick_indices])
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Power (dBm)', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_interference_map(interference_map: np.ndarray,
                         frequency_bins: np.ndarray,
                         title: str = "Interference Map",
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot interference map across frequency.
    
    Args:
        interference_map: 1D array of interference power (dBm) per frequency bin
        frequency_bins: Frequency bin centers in Hz
        title: Chart title
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    freq_ghz = frequency_bins / 1e9
    
    ax.plot(freq_ghz, interference_map, linewidth=2, color='red')
    ax.fill_between(freq_ghz, interference_map, alpha=0.3, color='red')
    
    ax.set_xlabel('Frequency (GHz)', fontsize=12)
    ax.set_ylabel('Interference Power (dBm)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_beam_usage_over_time(usage_history: List[Dict],
                             beam_ids: List[str],
                             title: str = "Beam Usage Over Time",
                             save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot beam usage over time.
    
    Args:
        usage_history: List of usage dictionaries with 'time' and beam usage
        beam_ids: List of beam identifiers
        title: Chart title
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    times = [entry['time'] for entry in usage_history]
    
    # Plot each beam
    colors = plt.cm.tab20(np.linspace(0, 1, len(beam_ids)))
    
    for i, beam_id in enumerate(beam_ids):
        usage = [entry.get(beam_id, 0) for entry in usage_history]
        ax.plot(times, usage, label=beam_id, linewidth=2, color=colors[i])
    
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Usage (normalized)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_spectrum_allocation_comparison(allocation_maps: List[np.ndarray],
                                       labels: List[str],
                                       frequency_bins: np.ndarray,
                                       title: str = "Spectrum Allocation Comparison",
                                       save_path: Optional[str] = None) -> plt.Figure:
    """
    Compare spectrum allocations across different policies/scenarios.
    
    Args:
        allocation_maps: List of 2D allocation arrays
        labels: Labels for each allocation
        frequency_bins: Frequency bin centers
        title: Chart title
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    n_scenarios = len(allocation_maps)
    
    fig, axes = plt.subplots(n_scenarios, 1, figsize=(14, 4 * n_scenarios))
    
    if n_scenarios == 1:
        axes = [axes]
    
    freq_ghz = frequency_bins / 1e9
    
    for i, (allocation_map, label) in enumerate(zip(allocation_maps, labels)):
        ax = axes[i]
        
        im = ax.imshow(allocation_map, aspect='auto', cmap='viridis',
                      interpolation='nearest', origin='lower')
        
        ax.set_ylabel('Beam ID', fontsize=10)
        ax.set_title(f'{label}', fontsize=12, fontweight='bold')
        
        # Frequency ticks
        n_ticks = 10
        tick_indices = np.linspace(0, len(freq_ghz) - 1, n_ticks).astype(int)
        ax.set_xticks(tick_indices)
        ax.set_xticklabels([f'{freq_ghz[j]:.2f}' for j in tick_indices])
        
        plt.colorbar(im, ax=ax, label='Allocation')
    
    axes[-1].set_xlabel('Frequency (GHz)', fontsize=12)
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

