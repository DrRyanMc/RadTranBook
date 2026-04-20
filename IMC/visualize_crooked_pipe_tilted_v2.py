#!/usr/bin/env python3
"""
Tilted wedge + floating isosurfaces/streamlines visualization

Creates asymmetric view with thin colored wedge and transparent effects.
- Transparent background
- Angle adjusted so radiation flows "into the page"
- Gradient-based streamlines
- Uses refined mesh for detail
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import PowerNorm
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


def create_vibrant_colormap():
    """Vibrant artistic colormap"""
    from matplotlib.colors import LinearSegmentedColormap
    
    colors = [
        '#0d1b2a',  # Deep blue-black
        '#1b263b',  # Dark blue
        '#415a77',  # Steel blue
        '#778da9',  # Light blue
        '#e0e1dd',  # Cool white
        '#ffd93d',  # Gold
        '#ff6b35',  # Orange-red
        '#ff006e',  # Hot pink
    ]
    return LinearSegmentedColormap.from_list('vibrant', colors, N=256)


def visualize_wedge_with_isosurfaces(solution_file, output_file='wedge_iso_tilted.png',
                                     wedge_angle=15, n_isosurfaces=5, downsample=2):
    """
    Create thin wedge slice with floating transparent isosurfaces
    """
    print(f"Creating wedge + isosurfaces from {solution_file}...")
    print(f"  Using downsample factor: {downsample}")
    
    # Load data
    sol = load_solution(solution_file)
    r_centers = sol['r_centers'][::downsample]
    z_centers = sol['z_centers'][::downsample]
    T_rz = sol['T_final'][::downsample, ::downsample]
    
    print(f"  Mesh size: {len(r_centers)} × {len(z_centers)}")
    print(f"  Temperature range: {T_rz.min():.4f} to {T_rz.max():.4f} keV")
    
    # Enhanced display range
    T_min_display = T_rz.min() + 0.02
    T_max_display = T_rz.max()
    T_display = np.clip(T_rz, T_min_display, T_max_display)
    
    # Figure with transparent background
    fig = plt.figure(figsize=(18, 14), facecolor='none')
    ax = fig.add_subplot(111, projection='3d', facecolor='none')
    
    # Clean appearance - no grids
    ax.grid(False)
    ax.set_axis_off()
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    cmap = create_vibrant_colormap()
    norm = PowerNorm(gamma=0.5, vmin=T_min_display, vmax=T_max_display)
    
    # Draw the thin colored wedge slice
    print("  Rendering wedge slice...")
    wedge_theta = np.linspace(0, np.radians(wedge_angle), 25)
    
    R_wedge, THETA_wedge, Z_wedge = np.meshgrid(r_centers, wedge_theta, z_centers, indexing='ij')
    X_wedge = R_wedge * np.cos(THETA_wedge)
    Y_wedge = R_wedge * np.sin(THETA_wedge)
    T_wedge = np.repeat(T_display[:, np.newaxis, :], len(wedge_theta), axis=1)
    
    # Plot wedge surfaces
    ax.plot_surface(X_wedge[-1, :, :], Y_wedge[-1, :, :], Z_wedge[-1, :, :],
                   facecolors=cmap(norm(T_wedge[-1, :, :])),
                   shade=True, alpha=1.0, antialiased=True,
                   linewidth=0, edgecolor='none')
    
    ax.plot_surface(X_wedge[:, 0, :], Y_wedge[:, 0, :], Z_wedge[:, 0, :],
                   facecolors=cmap(norm(T_display[:, :])),
                   shade=True, alpha=1.0, antialiased=True,
                   linewidth=0, edgecolor='none')
    
    ax.plot_surface(X_wedge[:, -1, :], Y_wedge[:, -1, :], Z_wedge[:, -1, :],
                   facecolors=cmap(norm(T_display[:, :])),
                   shade=True, alpha=1.0, antialiased=True,
                   linewidth=0, edgecolor='none')
    
    ax.plot_surface(X_wedge[:, :, 0], Y_wedge[:, :, 0], Z_wedge[:, :, 0],
                   facecolors=cmap(norm(T_wedge[:, :, 0])),
                   shade=True, alpha=0.95, antialiased=True,
                   linewidth=0, edgecolor='none')
    
    ax.plot_surface(X_wedge[:, :, -1], Y_wedge[:, :, -1], Z_wedge[:, :, -1],
                   facecolors=cmap(norm(T_wedge[:, :, -1])),
                   shade=True, alpha=0.9, antialiased=True,
                   linewidth=0, edgecolor='none')
    
    # Add floating transparent isosurfaces around the wedge
    print(f"  Creating {n_isosurfaces} floating isosurfaces...")
    
    T_threshold = T_rz.min() + 0.02
    T_levels = np.linspace(T_threshold, T_rz.max()*0.9, n_isosurfaces)
    
    iso_theta = np.linspace(0, 2*np.pi, 60)
    
    for idx, T_level in enumerate(T_levels):
        print(f"    Isosurface {idx+1}: T = {T_level:.4f} keV")
        
        tolerance = 0.008
        mask = np.abs(T_rz - T_level) < tolerance
        
        if not mask.any():
            continue
        
        points_r, points_z = [], []
        for i in range(len(r_centers)):
            for j in range(len(z_centers)):
                if mask[i, j]:
                    points_r.append(r_centers[i])
                    points_z.append(z_centers[j])
        
        if len(points_r) < 10:
            continue
        
        points_r = np.array(points_r)
        points_z = np.array(points_z)
        
        X_iso, Y_iso, Z_iso = [], [], []
        for r_val, z_val in zip(points_r, points_z):
            for theta in iso_theta:
                X_iso.append(r_val * np.cos(theta))
                Y_iso.append(r_val * np.sin(theta))
                Z_iso.append(z_val)
        
        X_iso = np.array(X_iso)
        Y_iso = np.array(Y_iso)
        Z_iso = np.array(Z_iso)
        
        color_val = (T_level - T_min_display) / (T_max_display - T_min_display)
        color = cmap(norm(T_level))
        alpha = 0.15 + 0.3 * color_val
        
        ax.scatter(X_iso, Y_iso, Z_iso,
                  c=[color], s=3, alpha=alpha, edgecolors='none')
    
    # Viewing angle - radiation flows into page (low elevation, azimuth 125°)
    ax.view_init(elev=15, azim=125)
    
    max_range = max(z_centers[-1], 2*r_centers[-1])
    ax.set_xlim([-max_range*0.6, max_range*0.6])
    ax.set_ylim([-max_range*0.6, max_range*0.6])
    ax.set_zlim([0, z_centers[-1]])
    
    # No labels, no colorbar - clean visualization
    plt.tight_layout(pad=0)
    print(f"  Saving to {output_file}...")
    plt.savefig(output_file, dpi=300, bbox_inches='tight',
               facecolor='none', edgecolor='none', transparent=True)
    print(f"  Saved: {output_file}")
    plt.close()


def visualize_wedge_with_streamlines(solution_file, output_file='wedge_stream_tilted.png',
                                     wedge_angle=20, downsample=3):
    """
    Thin wedge with gradient-based streamlines showing heat flow direction
    """
    print(f"Creating wedge + gradient streamlines from {solution_file}...")
    print(f"  Using downsample factor: {downsample}")
    
    # Load data
    sol = load_solution(solution_file)
    r_centers = sol['r_centers'][::downsample]
    z_centers = sol['z_centers'][::downsample]
    T_rz = sol['T_final'][::downsample, ::downsample]
    
    print(f"  Mesh size: {len(r_centers)} × {len(z_centers)}")
    
    # Enhanced range
    T_min_display = T_rz.min() + 0.02
    T_max_display = T_rz.max()
    T_display = np.clip(T_rz, T_min_display, T_max_display)
    
    # Calculate temperature gradient
    print("  Computing temperature gradient...")
    dr = np.diff(r_centers).mean()
    dz = np.diff(z_centers).mean()
    grad_T_r = np.gradient(T_rz, dr, axis=0)
    grad_T_z = np.gradient(T_rz, dz, axis=1)
    
    # Figure with transparent background
    fig = plt.figure(figsize=(18, 14), facecolor='none')
    ax = fig.add_subplot(111, projection='3d', facecolor='none')
    ax.grid(False)
    ax.set_axis_off()
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    cmap = create_vibrant_colormap()
    norm = PowerNorm(gamma=0.5, vmin=T_min_display, vmax=T_max_display)
    
    # Thin wedge
    print("  Rendering wedge slice...")
    wedge_theta = np.linspace(0, np.radians(wedge_angle), 20)
    R_wedge, THETA_wedge, Z_wedge = np.meshgrid(r_centers, wedge_theta, z_centers, indexing='ij')
    X_wedge = R_wedge * np.cos(THETA_wedge)
    Y_wedge = R_wedge * np.sin(THETA_wedge)
    T_wedge = np.repeat(T_display[:, np.newaxis, :], len(wedge_theta), axis=1)
    
    ax.plot_surface(X_wedge[-1, :, :], Y_wedge[-1, :, :], Z_wedge[-1, :, :],
                   facecolors=cmap(norm(T_wedge[-1, :, :])),
                   shade=True, alpha=1.0, antialiased=True,
                   linewidth=0, edgecolor='none')
    
    ax.plot_surface(X_wedge[:, 0, :], Y_wedge[:, 0, :], Z_wedge[:, 0, :],
                   facecolors=cmap(norm(T_display[:, :])),
                   shade=True, alpha=1.0, antialiased=True,
                   linewidth=0, edgecolor='none')
    
    # Curved gradient-based streamlines (more organic paths)
    print("  Adding smooth gradient-based flow streamlines...")
    n_streamlines = 20
    stream_r = np.linspace(0.2, r_centers[-1]*0.8, n_streamlines)
    
    for r_stream in stream_r:
        i_r = np.argmin(np.abs(r_centers - r_stream))
        
        # Only use 4 angles for cleaner look
        for angle_offset in [45, 135, 225, 315]:
            theta_stream = np.radians(angle_offset)
            
            # Start from fewer positions for less clutter
            for z_start_idx in range(10, len(z_centers)-10, 12):
                streamline_x, streamline_y, streamline_z = [], [], []
                
                # Current position
                r_curr = r_stream
                z_curr = z_centers[z_start_idx]
                theta_curr = theta_stream
                
                # Integrate streamline with smooth curves
                for step in range(15):
                    i_r_curr = np.clip(np.argmin(np.abs(r_centers - r_curr)), 0, len(r_centers)-1)
                    z_idx_curr = np.clip(np.argmin(np.abs(z_centers - z_curr)), 0, len(z_centers)-1)
                    
                    if z_idx_curr <= 0 or z_idx_curr >= len(z_centers)-1:
                        break
                    if r_curr < 0.1 or r_curr > r_centers[-1]:
                        break
                    
                    # Record position
                    x_pt = r_curr * np.cos(theta_curr)
                    y_pt = r_curr * np.sin(theta_curr)
                    streamline_x.append(x_pt)
                    streamline_y.append(y_pt)
                    streamline_z.append(z_curr)
                    
                    # Get gradient (heat flows from hot to cold)
                    gr = -grad_T_r[i_r_curr, z_idx_curr]
                    gz = -grad_T_z[i_r_curr, z_idx_curr]
                    
                    # Add some swirl for aesthetic curves
                    tangential_component = 0.3 * np.sin(step * 0.4)
                    
                    # Normalize and step
                    mag = np.sqrt(gr**2 + gz**2) + 1e-10
                    step_size = 0.08  # Smaller steps for smoother curves
                    
                    r_curr += (gr / mag) * step_size
                    z_curr += (gz / mag) * step_size
                    theta_curr += tangential_component * 0.2  # Gentle spiral
      No labels, no colorbar - clean visualization
    plt.tight_layout(pad=0
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.65, aspect=18, pad=0.02)
    cbar.set_label('Temperature (keV)', fontsize=15, color='black', weight='bold')
    cbar.ax.tick_params(colors='black', labelsize=13)
    cbar.outline.set_edgecolor('black')
    
    plt.title('Heat Flow - Gradient-Based Streamlines', 
             fontsize=20, color='black', weight='bold', pad=25)
    
    plt.tight_layout()
    print(f"  Saving to {output_file}...")
    plt.savefig(output_file, dpi=300, bbox_inches='tight',
               facecolor='none', edgecolor='none', transparent=True)
    print(f"  Saved: {output_file}")
    plt.close()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python visualize_crooked_pipe_tilted.py <solution_file.npz>")
        sys.exit(1)
    
    solution_file = sys.argv[1]
    
    print("="*75)
    print("TILTED WEDGE + ISOSURFACES/STREAMLINES")
    print("Using refined mesh solution (168×354) for high detail")
    print("="*75)
    
    # Option 1: Thin wedge + floating isosurfaces
    print("\n[1/2] Creating wedge with floating isosurfaces (RECOMMENDED)...")
    visualize_wedge_with_isosurfaces(solution_file,
                                     output_file='tilted_wedge_isosurfaces.png',
                                     wedge_angle=15,
                                     n_isosurfaces=5,
                                     downsample=2)
    
    # Option 2: Thin wedge + gradient streamlines
    print("\n[2/2] Creating wedge with gradient-based streamlines...")
    visualize_wedge_with_streamlines(solution_file,
                                    output_file='tilted_wedge_streamlines.png',
                                    wedge_angle=20,
                                    downsample=3)
    
    print("\n" + "="*75)
    print("Tilted visualizations complete!")
    print("="*75)
