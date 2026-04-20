#!/usr/bin/env python3
"""
Dramatic 3D Visualization for Crooked Pipe - Floating colorful surfaces

Creates vibrant, eye-catching visualizations with no grids, focusing on
temperature variations with enhanced color contrast.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import PowerNorm, LogNorm, Normalize
from mpl_toolkits.mplot3d import Axes3D
import sys


def load_solution(filename):
    """Load IMC solution from npz file"""
    data = np.load(filename, allow_pickle=True)
    return {
        'r_centers': data['r_centers'],
        'z_centers': data['z_centers'],
        'T_final': data['T_final'],
        'Tr_final': data['Tr_final'],
    }


def create_hot_colormap():
    """Create vibrant colormap emphasizing hot regions"""
    from matplotlib.colors import LinearSegmentedColormap
    
    # Dramatic fire colors
    colors = [
        '#000033',  # Deep blue (cold)
        '#0066ff',  # Bright blue
        '#00ffff',  # Cyan
        '#00ff00',  # Green
        '#ffff00',  # Yellow
        '#ff6600',  # Orange
        '#ff0000',  # Red
        '#ff00ff',  # Magenta (hottest)
    ]
    return LinearSegmentedColormap.from_list('hot', colors, N=256)


def create_artistic_colormap():
    """Vibrant artistic palette"""
    from matplotlib.colors import LinearSegmentedColormap
    
    colors = [
        '#1a0033',  # Deep purple-black
        '#4a148c',  # Deep purple
        '#7b1fa2',  # Purple
        '#d500f9',  # Bright magenta
        '#ff6d00',  # Orange
        '#ffd600',  # Gold
        '#ffea00',  # Bright yellow
        '#ffffff',  # White hot
    ]
    return LinearSegmentedColormap.from_list('artistic', colors, N=256)


def visualize_dramatic_cutaway(solution_file, output_file='dramatic_cutaway.png',
                               wedge_angle=75, downsample=4, use_log=False):
    """
    Create dramatic cutaway with enhanced colors and no grids
    """
    print(f"Creating dramatic visualization from {solution_file}...")
    
    # Load data
    sol = load_solution(solution_file)
    r_centers = sol['r_centers'][::downsample]
    z_centers = sol['z_centers'][::downsample]
    T_rz = sol['T_final'][::downsample, ::downsample]
    
    print(f"  Temperature range: {T_rz.min():.4f} to {T_rz.max():.4f} keV")
    
    # Enhance contrast by adjusting the range - focus on variations above minimum
    T_min_display = T_rz.min() + 0.02  # Start colormap slightly above minimum
    T_max_display = T_rz.max()
    
    # Clip extremely cold regions to show more variation
    T_display = np.clip(T_rz, T_min_display, T_max_display)
    
    print(f"  Display range: {T_min_display:.4f} to {T_max_display:.4f} keV")
    
    # Create revolution coordinates
    theta = np.linspace(0, 2*np.pi - np.radians(wedge_angle), 40)
    R, THETA, Z = np.meshgrid(r_centers, theta, z_centers, indexing='ij')
    X = R * np.cos(THETA)
    Y = R * np.sin(THETA)
    T_3d = np.repeat(T_display[:, np.newaxis, :], len(theta), axis=1)
    
    # Create figure with black background for dramatic effect
    fig = plt.figure(figsize=(16, 14), facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='black')
    
    # Get colormap
    cmap = create_artistic_colormap()
    
    # Use power norm for better contrast (gamma < 1 brightens, > 1 darkens)
    norm = PowerNorm(gamma=0.6, vmin=T_min_display, vmax=T_max_display)
    
    # Plot surfaces with NO grid lines
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')
    
    # Plot outer surface (r=max) - brightest
    print("  Rendering outer surface...")
    surf_outer = ax.plot_surface(
        X[-1, :, :], Y[-1, :, :], Z[-1, :, :],
        facecolors=cmap(norm(T_3d[-1, :, :])),
        shade=True, alpha=1.0, antialiased=True,
        linewidth=0, edgecolor='none'
    )
    
    # Plot cut faces showing interior
    print("  Rendering cut faces...")
    surf_cut1 = ax.plot_surface(
        X[:, 0, :], Y[:, 0, :], Z[:, 0, :],
        facecolors=cmap(norm(T_display[:, :])),
        shade=True, alpha=0.95, antialiased=True,
        linewidth=0, edgecolor='none'
    )
    
    surf_cut2 = ax.plot_surface(
        X[:, -1, :], Y[:, -1, :], Z[:, -1, :],
        facecolors=cmap(norm(T_display[:, :])),
        shade=True, alpha=0.95, antialiased=True,
        linewidth=0, edgecolor='none'
    )
    
    # Plot bottom (source region) - should be hottest
    print("  Rendering bottom surface...")
    surf_bottom = ax.plot_surface(
        X[:, :, 0], Y[:, :, 0], Z[:, :, 0],
        facecolors=cmap(norm(T_3d[:, :, 0])),
        shade=True, alpha=0.9, antialiased=True,
        linewidth=0, edgecolor='none'
    )
    
    # Plot top surface
    print("  Rendering top surface...")
    surf_top = ax.plot_surface(
        X[:, :, -1], Y[:, :, -1], Z[:, :, -1],
        facecolors=cmap(norm(T_3d[:, :, -1])),
        shade=True, alpha=0.85, antialiased=True,
        linewidth=0, edgecolor='none'
    )
    
    # Dramatic viewing angle
    ax.view_init(elev=25, azim=45)
    
    # Remove axes completely
    ax.set_axis_off()
    
    # Add colorbar with black background
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.7, aspect=20, pad=0.02)
    cbar.set_label('Temperature (keV)', fontsize=14, color='white', weight='bold')
    cbar.ax.tick_params(colors='white', labelsize=12)
    cbar.outline.set_edgecolor('white')
    
    # Title with white text
    plt.title('Crooked Pipe - Radiation Transport', 
             fontsize=18, color='white', weight='bold', pad=20)
    
    plt.tight_layout()
    print(f"  Saving to {output_file}...")
    plt.savefig(output_file, dpi=300, bbox_inches='tight', 
               facecolor='black', edgecolor='none')
    print(f"  Saved: {output_file}")
    plt.close()


def visualize_floating_isosurfaces(solution_file, output_file='floating_iso.png',
                                   wedge_angle=75, n_levels=6, downsample=3):
    """
    Show floating temperature isosurfaces with vibrant colors
    """
    print(f"Creating floating isosurfaces from {solution_file}...")
    
    # Load data
    sol = load_solution(solution_file)
    r_centers = sol['r_centers'][::downsample]
    z_centers = sol['z_centers'][::downsample]
    T_rz = sol['T_final'][::downsample, ::downsample]
    
    print(f"  Temperature range: {T_rz.min():.4f} to {T_rz.max():.4f} keV")
    
    # Focus on temperature variations
    T_threshold = T_rz.min() + 0.015  # Only show regions above cold baseline
    T_levels = np.linspace(T_threshold, T_rz.max()*0.95, n_levels)
    
    # Revolution
    theta = np.linspace(0, 2*np.pi - np.radians(wedge_angle), 50)
    
    # Create figure with black background
    fig = plt.figure(figsize=(16, 14), facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='black')
    
    # Remove grid and axes
    ax.grid(False)
    ax.set_axis_off()
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    cmap = create_hot_colormap()
    
    print(f"  Creating {n_levels} isosurfaces...")
    
    # Create isosurfaces as scatter plots
    for idx, T_level in enumerate(T_levels):
        print(f"    Level {idx+1}: T = {T_level:.4f} keV")
        
        # Find points near this temperature
        mask = np.abs(T_rz - T_level) < 0.005
        
        if not mask.any():
            continue
        
        # Get r, z coordinates of these points
        r_iso, z_iso = [], []
        for i in range(len(r_centers)):
            for j in range(len(z_centers)):
                if mask[i, j]:
                    r_iso.append(r_centers[i])
                    z_iso.append(z_centers[j])
        
        if not r_iso:
            continue
        
        r_iso = np.array(r_iso)
        z_iso = np.array(z_iso)
        
        # Revolve around axis
        X_iso, Y_iso, Z_iso = [], [], []
        for r_val, z_val in zip(r_iso, z_iso):
            for th in theta:
                X_iso.append(r_val * np.cos(th))
                Y_iso.append(r_val * np.sin(th))
                Z_iso.append(z_val)
        
        X_iso = np.array(X_iso)
        Y_iso = np.array(Y_iso)
        Z_iso = np.array(Z_iso)
        
        # Color by temperature
        color = cmap((T_level - T_rz.min()) / (T_rz.max() - T_rz.min()))
        
        # Opacity increases with temperature
        alpha = 0.3 + 0.6 * (T_level - T_threshold) / (T_rz.max() - T_threshold)
        
        ax.scatter(X_iso, Y_iso, Z_iso, 
                  c=[color], s=2, alpha=alpha, edgecolors='none')
    
    # Viewing angle
    ax.view_init(elev=20, azim=50)
    
    # Add colorbar
    norm = Normalize(vmin=T_rz.min(), vmax=T_rz.max())
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.7, aspect=20, pad=0.02)
    cbar.set_label('Temperature (keV)', fontsize=14, color='white', weight='bold')
    cbar.ax.tick_params(colors='white', labelsize=12)
    cbar.outline.set_edgecolor('white')
    
    plt.title('Temperature Isosurfaces', 
             fontsize=18, color='white', weight='bold', pad=20)
    
    plt.tight_layout()
    print(f"  Saving to {output_file}...")
    plt.savefig(output_file, dpi=300, bbox_inches='tight',
               facecolor='black', edgecolor='none')
    print(f"  Saved: {output_file}")
    plt.close()


def visualize_sliced_volume(solution_file, output_file='sliced_volume.png',
                           wedge_angle=90, downsample=3):
    """
    Show multiple slices through the volume with vibrant colors
    """
    print(f"Creating sliced volume from {solution_file}...")
    
    # Load data
    sol = load_solution(solution_file)
    r_centers = sol['r_centers'][::downsample]
    z_centers = sol['z_centers'][::downsample]
    T_rz = sol['T_final'][::downsample, ::downsample]
    
    # Enhance display range
    T_min_display = T_rz.min() + 0.015
    T_max_display = T_rz.max()
    T_display = np.clip(T_rz, T_min_display, T_max_display)
    
    # Create partial revolution
    theta = np.linspace(0, 2*np.pi - np.radians(wedge_angle), 35)
    R, THETA, Z = np.meshgrid(r_centers, theta, z_centers, indexing='ij')
    X = R * np.cos(THETA)
    Y = R * np.sin(THETA)
    T_3d = np.repeat(T_display[:, np.newaxis, :], len(theta), axis=1)
    
    # Figure with black background
    fig = plt.figure(figsize=(16, 14), facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='black')
    
    # Clean appearance
    ax.grid(False)
    ax.set_axis_off()
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    cmap = create_artistic_colormap()
    norm = PowerNorm(gamma=0.5, vmin=T_min_display, vmax=T_max_display)
    
    # Plot multiple z-slices at different heights
    z_slice_indices = [0, len(z_centers)//4, len(z_centers)//2, 
                      3*len(z_centers)//4, -1]
    
    print("  Rendering slices...")
    for z_idx in z_slice_indices:
        ax.plot_surface(
            X[:, :, z_idx], Y[:, :, z_idx], Z[:, :, z_idx],
            facecolors=cmap(norm(T_3d[:, :, z_idx])),
            shade=True, alpha=0.7, antialiased=True,
            linewidth=0, edgecolor='none'
        )
    
    # Plot outer surface
    ax.plot_surface(
        X[-1, :, :], Y[-1, :, :], Z[-1, :, :],
        facecolors=cmap(norm(T_3d[-1, :, :])),
        shade=True, alpha=0.5, antialiased=True,
        linewidth=0, edgecolor='none'
    )
    
    # View angle
    ax.view_init(elev=30, azim=40)
    
    # Colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.7, aspect=20, pad=0.02)
    cbar.set_label('Temperature (keV)', fontsize=14, color='white', weight='bold')
    cbar.ax.tick_params(colors='white', labelsize=12)
    cbar.outline.set_edgecolor('white')
    
    plt.title('Multi-Slice Volume View', 
             fontsize=18, color='white', weight='bold', pad=20)
    
    plt.tight_layout()
    print(f"  Saving to {output_file}...")
    plt.savefig(output_file, dpi=300, bbox_inches='tight',
               facecolor='black', edgecolor='none')
    print(f"  Saved: {output_file}")
    plt.close()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python visualize_crooked_pipe_dramatic.py <solution_file.npz>")
        sys.exit(1)
    
    solution_file = sys.argv[1]
    
    print("="*70)
    print("DRAMATIC 3D VISUALIZATIONS - Crooked Pipe")
    print("="*70)
    
    # Option 1: Dramatic cutaway with enhanced colors
    print("\n[1/3] Creating dramatic cutaway...")
    visualize_dramatic_cutaway(solution_file, 
                              output_file='dramatic_cutaway_75deg.png',
                              wedge_angle=75, downsample=3)
    
    # Option 2: Floating isosurfaces
    print("\n[2/3] Creating floating isosurfaces...")
    visualize_floating_isosurfaces(solution_file,
                                   output_file='dramatic_isosurfaces.png',
                                   wedge_angle=75, n_levels=6, downsample=3)
    
    # Option 3: Sliced volume
    print("\n[3/3] Creating sliced volume view...")
    visualize_sliced_volume(solution_file,
                           output_file='dramatic_sliced.png',
                           wedge_angle=90, downsample=3)
    
    print("\n" + "="*70)
    print("All dramatic visualizations complete!")
    print("="*70)
