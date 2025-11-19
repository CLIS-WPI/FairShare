"""
3D surface plots for elevation and Doppler visualization.

Creates 3D surface plots showing elevation angles and Doppler shifts
over time and location, useful for understanding satellite dynamics.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Dict, Optional, Tuple
from datetime import datetime


def plot_elevation_surface(elevation_data: np.ndarray,
                          times: List[datetime],
                          locations: List[Tuple[float, float]],
                          title: str = "Elevation Angle Surface",
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot 3D surface of elevation angles over time and location.
    
    Args:
        elevation_data: 2D array [time, location] of elevation angles (degrees)
        times: List of time points
        locations: List of (lat, lon) tuples
        title: Chart title
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create meshgrid
    time_indices = np.arange(len(times))
    location_indices = np.arange(len(locations))
    T, L = np.meshgrid(time_indices, location_indices)
    
    # Plot surface
    surf = ax.plot_surface(T, L, elevation_data.T, cmap='coolwarm',
                          linewidth=0, antialiased=True, alpha=0.8)
    
    # Labels
    ax.set_xlabel('Time Index', fontsize=10)
    ax.set_ylabel('Location Index', fontsize=10)
    ax.set_zlabel('Elevation Angle (degrees)', fontsize=10)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Elevation (deg)')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_doppler_surface(doppler_data: np.ndarray,
                        times: List[datetime],
                        locations: List[Tuple[float, float]],
                        title: str = "Doppler Shift Surface",
                        save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot 3D surface of Doppler shifts over time and location.
    
    Args:
        doppler_data: 2D array [time, location] of Doppler shifts (Hz)
        times: List of time points
        locations: List of (lat, lon) tuples
        title: Chart title
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create meshgrid
    time_indices = np.arange(len(times))
    location_indices = np.arange(len(locations))
    T, L = np.meshgrid(time_indices, location_indices)
    
    # Plot surface
    surf = ax.plot_surface(T, L, doppler_data.T / 1e3, cmap='seismic',
                          linewidth=0, antialiased=True, alpha=0.8)
    
    # Labels
    ax.set_xlabel('Time Index', fontsize=10)
    ax.set_ylabel('Location Index', fontsize=10)
    ax.set_zlabel('Doppler Shift (kHz)', fontsize=10)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Doppler (kHz)')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_elevation_doppler_contour(elevation_data: np.ndarray,
                                  doppler_data: np.ndarray,
                                  times: List[datetime],
                                  locations: List[Tuple[float, float]],
                                  title: str = "Elevation-Doppler Contour",
                                  save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot contour of elevation vs Doppler.
    
    Args:
        elevation_data: 2D array [time, location] of elevation angles
        doppler_data: 2D array [time, location] of Doppler shifts
        times: List of time points
        locations: List of (lat, lon) tuples
        title: Chart title
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Flatten data
    elev_flat = elevation_data.flatten()
    dopp_flat = doppler_data.flatten() / 1e3  # Convert to kHz
    
    # Create 2D histogram
    H, xedges, yedges = np.histogram2d(elev_flat, dopp_flat, bins=50)
    
    # Plot contour
    X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
    contour = ax.contourf(X, Y, H.T, levels=20, cmap='viridis')
    
    ax.set_xlabel('Elevation Angle (degrees)', fontsize=12)
    ax.set_ylabel('Doppler Shift (kHz)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.colorbar(contour, ax=ax, label='Count')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_satellite_trajectory_3d(positions: np.ndarray,
                                 ground_station: Tuple[float, float, float],
                                 title: str = "Satellite Trajectory",
                                 save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot 3D trajectory of satellite relative to ground station.
    
    Args:
        positions: Array of satellite positions [time, xyz] in ECEF (meters)
        ground_station: Ground station position (x, y, z) in ECEF
        title: Chart title
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectory
    ax.plot(positions[:, 0] / 1e6, positions[:, 1] / 1e6, positions[:, 2] / 1e6,
           linewidth=2, label='Satellite Trajectory')
    
    # Plot ground station
    ax.scatter([ground_station[0] / 1e6], [ground_station[1] / 1e6],
              [ground_station[2] / 1e6], color='red', s=100,
              label='Ground Station', marker='^')
    
    # Labels
    ax.set_xlabel('X (1000 km)', fontsize=10)
    ax.set_ylabel('Y (1000 km)', fontsize=10)
    ax.set_zlabel('Z (1000 km)', fontsize=10)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    
    # Equal aspect ratio
    max_range = np.array([
        positions[:, 0].max() - positions[:, 0].min(),
        positions[:, 1].max() - positions[:, 1].min(),
        positions[:, 2].max() - positions[:, 2].min()
    ]).max() / 1e6
    mid_x = (positions[:, 0].max() + positions[:, 0].min()) / 2 / 1e6
    mid_y = (positions[:, 1].max() + positions[:, 1].min()) / 2 / 1e6
    mid_z = (positions[:, 2].max() + positions[:, 2].min()) / 2 / 1e6
    
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

