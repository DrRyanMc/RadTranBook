#!/usr/bin/env python3
"""
Simple 3D Visualization Using Matplotlib for Crooked Pipe IMC

This version uses matplotlib which is more reliable across platforms,
though not as high quality as PyVista.

Usage:
    python visualize_crooked_pipe_simple.py <solution_file.npz>
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import sys


def load_solution(filename):
    """Load IMC solution from npz file"""
    data = np.load(filename, allow_pickle=True)
    
    solution = {
        'r_centers': data['r_centers'],
        'z_centers': data['z_centers'],
        'T_final': data['T_final'],
        'Tr_final': data['Tr_final'],
    }
    
    return solution


def create_artistic_colormap():
    """Create artistic colormap: deep blue -> purple -> orange -> yellow"""
    from matplotlib.colors import LinearSegmentedColormap
    
    colors = [
        '#0a0e27',  # Deep blue-black
        '#1a237e',  # Deep blue
        '#673ab7',  # Purple
        '#ff6f00',  # Orange
        '#ffd54f',  # Gold
        '#ffeb3b',  # Bright yellow
    ]
    cmap = LinearSegmentedColormap.from_list('artistic', colors, N=256)
    return cmap


def visualize_3d_cutaway(solution_file, output_file='crooked_pipe_3d.png', 
                        wedge_angle=75, n_theta=40,downsample=1):
    """
    Create 3D cutaway visualization using matplotlib
    """
    print(f"Creating 3D cutaway from {solution_file}...")
    
    # Load data
    sol = load_solution(solution_file)
    r_centers = sol['r_centers'][::downsample]
    z_centers = sol['z_centers'][::downsample]
    T_rz = sol['T_final'][::downsample, ::downsample]
    
    print(f"  Data shape: {len(r_centers)} × {len(z_centers)}")
    print(f"  Temperature range: {T_rz.min():.4f} to {T_rz.max():.4f} keV")
    
    # Create revolution coordinates (partial for wedge)
    theta_start = 0
    theta_end = 2*np.pi - np.radians(wedge_angle)
    theta = np.linspace(theta_start, theta_end, n_theta)
    
    # Create 3D meshgrid
    R, THETA, Z = np.meshgrid(r_centers, theta, z_centers, indexing='ij')
    X = R * np.cos(THETA)
    Y = R * np.sin(THETA)
    
    # Expand temperature
    T_3d = np.repeat(T_rz[:, np.newaxis, :], n_theta, axis=1)
    
    # Create figure
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get colormap
    cmap = create_artistic_colormap()
    norm = plt.Normalize(vmin=T_rz.min(), vmax=T_rz.max())
    
    # Plot outer surface (r=max)
    print("  Plotting outer surface...")
    surf_outer = ax.plot_surface(
        X[-1, :, :], Y[-1, :, :], Z[-1, :, :],
        facecolors=cmap(norm(T_3d[-1, :, :])),
        shade=True, alpha=0.9, antialiased=True
    )
    
    # Plot inner radial cuts (the wedge faces)
    print("  Plotting cut faces...")
    # First face (theta = 0)
    surf_cut1 = ax.plot_surface(
        X[:, 0, :], Y[:, 0, :], Z[:, 0, :],
        facecolors=cmap(norm(T_rz[:, :])),
        shade=True, alpha=0.8, antialiased=True
    )
    
    # Second face (theta = max)
    surf_cut2 = ax.plot_surface(
        X[:, -1, :], Y[:, -1, :], Z[:, -1, :],
        facecolors=cmap(norm(T_rz[:, :])),
        shade=True, alpha=0.8, antialiased=True
    )
    
    # Plot bottom surface (z=0)
    print("  Plotting bottom...")
    surf_bottom = ax.plot_surface(
        X[:, :, 0], Y[:, :, 0], Z[:, :, 0],
        facecolors=cmap(norm(T_3d[:, :, 0])),
        shade=True, alpha=0.7, antialiased=True
    )
    
    # Plot top surface (z=max)
    print("  Plotting top...")
    surf_top = ax.plot_surface(
        X[:, :, -1], Y[:, :, -1], Z[:, :, -1],
        facecolors=cmap(norm(T_3d[:, :, -1])),
        shade=True, alpha=0.7, antialiased=True
    )
    
    # Add interface lines (ghosted)
    for z_int in [2.5, 3.0, 4.0, 4.5]:
        theta_line = np.linspace(theta_start, theta_end, 50)
        r_line = np.linspace(0, r_centers[-1], 30)
        R_int, THETA_int = np.meshgrid(r_line, theta_line)
        X_int = R_int * np.cos(THETA_int)
        Y_int = R_int * np.sin(THETA_int)
        Z_int = np.full_like(X_int, z_int)
        ax.plot_surface(X_int, Y_int, Z_int, color='gray', alpha=0.05)
    
    # Set view angle
    ax.view_init(elev=20, azim=45)
    
    # Labels
    ax.set_xlabel('x (cm)', fontsize=12, labelpad=10)
    ax.set_ylabel('y (cm)', fontsize=12, labelpad=10)
    ax.set_zlabel('z (cm)', fontsize=12, labelpad=10)
    ax.set_title(f'Crooked Pipe 3D - Wedge {wedge_angle}°', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Set equal aspect ratio
    max_range = max(z_centers[-1], 2*r_centers[-1])
    ax.set_xlim([-max_range/2, max_range/2])
    ax.set_ylim([-max_range/2, max_range/2])
    ax.set_zlim([0, z_centers[-1]])
    
    # Add colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, aspect=15, pad=0.1)
    cbar.set_label('Temperature (keV)', fontsize=12)
    
    plt.tight_layout()
    print(f"  Saving to {output_file}...")
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_file}")
    plt.close()


def visualize_isosurfaces(solution_file, output_file='crooked_pipe_iso.png',
                         n_levels=4, wedge_angle=75, downsample=5):
    """
    Create visualization with temperature isosurfaces
    """
    print(f"Creating isosurface visualization from {solution_file}...")
    
    # Load data
    sol = load_solution(solution_file)
    r_centers = sol['r_centers'][::downsample]
    z_centers = sol['z_centers'][::downsample]
    T_rz = sol['T_final'][::downsample, ::downsample]
    
    print(f"  Data shape: {len(r_centers)} × {len(z_centers)}")
    
    # Create revolution coordinates
    theta = np.linspace(0, 2*np.pi - np.radians(wedge_angle), 30)
    R, THETA, Z = np.meshgrid(r_centers, theta, z_centers, indexing='ij')
    X = R * np.cos(THETA)
    Y = R * np.sin(THETA)
    T_3d = np.repeat(T_rz[:, np.newaxis, :], len(theta), axis=1)
    
    # Create figure with multiple views
    fig = plt.figure(figsize=(18, 6))
    
    cmap = create_artistic_colormap()
    T_levels = np.linspace(T_rz.min() + 0.1*(T_rz.max()-T_rz.min()), 
                          T_rz.max()*0.8, n_levels)
    
    # View 1: 3/4 view
    ax1 = fig.add_subplot(131, projection='3d')
    for T_level in T_levels:
        mask = np.abs(T_3d - T_level) < 0.01
        if mask.any():
            ax1.scatter(X[mask], Y[mask], Z[mask], 
                       c=[T_level]*mask.sum(), cmap=cmap,
                       vmin=T_rz.min(), vmax=T_rz.max(),
                       alpha=0.3, s=1)
    ax1.view_init(elev=20, azim=45)
    ax1.set_title('3/4 View', fontsize=12, fontweight='bold')
    ax1.set_xlabel('x (cm)')
    ax1.set_ylabel('y (cm)')
    ax1.set_zlabel('z (cm)')
    
    # View 2: Side view
    ax2 = fig.add_subplot(132, projection='3d')
    for T_level in T_levels:
        mask = np.abs(T_3d - T_level) < 0.01
        if mask.any():
            ax2.scatter(X[mask], Y[mask], Z[mask], 
                       c=[T_level]*mask.sum(), cmap=cmap,
                       vmin=T_rz.min(), vmax=T_rz.max(),
                       alpha=0.3, s=1)
    ax2.view_init(elev=0, azim=90)
    ax2.set_title('Side View', fontsize=12, fontweight='bold')
    ax2.set_xlabel('x (cm)')
    ax2.set_ylabel('y (cm)')
    ax2.set_zlabel('z (cm)')
    
    # View 3: Top view
    ax3 = fig.add_subplot(133, projection='3d')
    for T_level in T_levels:
        mask = np.abs(T_3d - T_level) < 0.01
        if mask.any():
            ax3.scatter(X[mask], Y[mask], Z[mask], 
                       c=[T_level]*mask.sum(), cmap=cmap,
                       vmin=T_rz.min(), vmax=T_rz.max(),
                       alpha=0.3, s=1)
    ax3.view_init(elev=90, azim=0)
    ax3.set_title('Top View', fontsize=12, fontweight='bold')
    ax3.set_xlabel('x (cm)')
    ax3.set_ylabel('y (cm)')
    ax3.set_zlabel('z (cm)')
    
    plt.suptitle('Temperature Isosurfaces - Multiple Views', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    print(f"  Saving to {output_file}...")
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_file}")
    plt.close()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python visualize_crooked_pipe_simple.py <solution_file.npz>")
        sys.exit(1)
    
    solution_file = sys.argv[1]
    
    print("="*60)
    print("Crooked Pipe 3D Visualization (Matplotlib)")
    print("="*60)
    
    # Generate all three options
    print("\nOption 1: 75° wedge cutaway")
    visualize_3d_cutaway(solution_file, 
                        output_file='option1_matplotlib_75deg.png',
                        wedge_angle=75, downsample=1)
    
    print("\nOption 2: 60° wedge cutaway")
    visualize_3d_cutaway(solution_file,
                        output_file='option2_matplotlib_60deg.png',
                        wedge_angle=60, downsample=1)
    
    print("\nOption 3: 90° wedge cutaway")
    visualize_3d_cutaway(solution_file,
                        output_file='option3_matplotlib_90deg.png',
                        wedge_angle=90, downsample=1)
    
    print("\n" + "="*60)
    print("All visualizations complete!")
    print("="*60)
