#!/usr/bin/env python3
"""
Tilted wedge + floating isosurfaces visualization

Creates a thin colored wedge slice with transparent isosurfaces floating around it
to show temperature structure without full axisymmetric rendering.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import PowerNorm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
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
    Create a thin wedge slice plus floating transparent isosurfaces
    Using refined mesh for more detail
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
    
    # ========================================
    # PART 1: Draw the thin colored wedge slice
    # ========================================
    print("  Rendering wedge slice...")
    wedge_theta = np.linspace(0, np.radians(wedge_angle), 25)
    
    R_wedge, THETA_wedge, Z_wedge = np.meshgrid(r_centers, wedge_theta, z_centers, indexing='ij')
    X_wedge = R_wedge * np.cos(THETA_wedge)
    Y_wedge = R_wedge * np.sin(THETA_wedge)
    T_wedge = np.repeat(T_display[:, np.newaxis, :], len(wedge_theta), axis=1)
    
    # Plot all surfaces of the thin wedge
    # Outer surface (r=max)
    ax.plot_surface(
        X_wedge[-1, :, :], Y_wedge[-1, :, :], Z_wedge[-1, :, :],
        facecolors=cmap(norm(T_wedge[-1, :, :])),
        shade=True, alpha=1.0, antialiased=True,
        linewidth=0, edgecolor='none'
    )
    
    # Front face (theta=0)
    ax.plot_surface(
        X_wedge[:, 0, :], Y_wedge[:, 0, :], Z_wedge[:, 0, :],
        facecolors=cmap(norm(T_display[:, :])),
        shade=True, alpha=1.0, antialiased=True,
        linewidth=0, edgecolor='none'
    )
    
    # Back face (theta=max)
    ax.plot_surface(
        X_wedge[:, -1, :], Y_wedge[:, -1, :], Z_wedge[:, -1, :],
        facecolors=cmap(norm(T_display[:, :])),
        shade=True, alpha=1.0, antialiased=True,
        linewidth=0, edgecolor='none'
    )
    
    # Bottom surface
    ax.plot_surface(
        X_wedge[:, :, 0], Y_wedge[:, :, 0], Z_wedge[:, :, 0],
        facecolors=cmap(norm(T_wedge[:, :, 0])),
        shade=True, alpha=0.95, antialiased=True,
        linewidth=0, edgecolor='none'
    )
    
    # Top surface
    ax.plot_surface(
        X_wedge[:, :, -1], Y_wedge[:, :, -1], Z_wedge[:, :, -1],
        facecolors=cmap(norm(T_wedge[:, :, -1])),
        shade=True, alpha=0.9, antialiased=True,
        linewidth=0, edgecolor='none'
    )
    
    # ========================================
    # PART 2: Add floating transparent isosurfaces around the wedge
    # ========================================
    print(f"  Creating {n_isosurfaces} floating isosurfaces...")
    
    # Temperature levels for isosurfaces
    T_threshold = T_rz.min() + 0.02
    T_levels = np.linspace(T_threshold, T_rz.max()*0.9, n_isosurfaces)
    
    # Full revolution for isosurfaces (360°)
    iso_theta = np.linspace(0, 2*np.pi, 60)
    
    for idx, T_level in enumerate(T_levels):
        print(f"    Isosurface {idx+1}: T = {T_level:.4f} keV")
        
        # Find points near this temperature
        tolerance = 0.008
        mask = np.abs(T_rz - T_level) < tolerance
        
        if not mask.any():
            continue
        
        # Extract contour points
        points_r, points_z = [], []
        for i in range(len(r_centers)):
            for j in range(len(z_centers)):
                if mask[i, j]:
                    points_r.append(r_centers[i])
                    points_z.append(z_centers[j])
        
        if len(points_r) < 10:  # Skip if too few points
            continue
        
        points_r = np.array(points_r)
        points_z = np.array(points_z)
        
        # Revolve these points around the axis
        X_iso, Y_iso, Z_iso = [], [], []
        for r_val, z_val in zip(points_r, points_z):
            for theta in iso_theta:
                X_iso.append(r_val * np.cos(theta))
                Y_iso.append(r_val * np.sin(theta))
                Z_iso.append(z_val)
        
        X_iso = np.array(X_iso)
        Y_iso = np.array(Y_iso)
        Z_iso = np.array(Z_iso)
        
        # Color varies with temperature
        color_val = (T_level - T_min_display) / (T_max_display - T_min_display)
        color = cmap(norm(T_level))
        
        # Opacity increases with temperature
        alpha = 0.15 + 0.3 * color_val
        
        # Plot as semi-transparent scatter
        ax.scatter(X_iso, Y_iso, Z_iso,
                  c=[color], s=3, alpha=alpha, edgecolors='none')
    
    # ========================================
    # PART 3: Viewing angle - radiation flows "into the page"
    # ========================================
    # Low elevation to look along z-axis (radiation flows from bottom upward)
    # Azimuth oriented so flow goes into page
    ax.view_init(elev=15, azim=125)
    
    # Zoom in a bit
    max_range = max(z_centers[-1], 2*r_centers[-1])
    ax.set_xlim([-max_range*0.6, max_range*0.6])
    ax.set_ylim([-max_range*0.6, max_range*0.6])
    ax.set_zlim([0, z_centers[-1]])
    
    # Colorbar with black text for transparent background
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.65, aspect=18, pad=0.02)
    cbar.set_label('Temperature (keV)', fontsize=15, color='black', weight='bold')
    cbar.ax.tick_params(colors='black', labelsize=13)
    cbar.outline.set_edgecolor('black')
    cbar.outline.set_linewidth(1.5)
    
    # Title with black text
    plt.title('Radiation Transport - Temperature Structure', 
             fontsize=20, color='black', weight='bold', pad=25)
    
    plt.tight_layout()
    print(f"  Saving to {output_file}...")
    plt.savefig(output_file, dpi=300, bbox_inches='tight',
               facecolor='none', edgecolor='none', transparent=True)
    print(f"  Saved: {output_file}")
    plt.close()


def visualize_wedge_with_streamlines(solution_file, output_file='wedge_stream_tilted.png',
                                     wedge_angle=20, downsample=3):
    """
    Thin wedge with gradient-based streamlines showing heat flow direction
    Uses refined mesh for better detail
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
    
    # Calculate temperature gradient (heat flows from hot to cold, opposite to gradient)
    print("  Computing temperature gradient...")
    dr = np.diff(r_centers).mean()
    dz = np.diff(z_centers).mean()
    grad_T_r = np.gradient(T_rz, dr, axis=0)  # ∂T/∂r
    grad_T_z = np.gradient(T_rz, dz, axis=1)  # ∂T/∂z
    
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
    
    # Gradient-based streamlines showing heat flow direction
    print("  Adding gradient-based flow streamlines...")
    n_streamlines = 25
    stream_r = np.linspace(0.15, r_centers[-1]*0.85, n_streamlines)
    
    for r_stream in stream_r:
        # Find closest r index
        i_r = np.argmin(np.abs(r_centers - r_stream))
        
        # Create streamlines at several angles around the cylinder
        for angle_offset in [30, 90, 150, 210, 270, 330]:
            theta_stream = np.radians(angle_offset)
            x_base = r_stream * np.cos(theta_stream)
            y_base = r_stream * np.sin(theta_stream)
            
            # Follow gradient direction (heat flows opposite to ∇T)
            # Start from several z positions and trace the flow
            for z_start_idx in range(5, len(z_centers)-5, 8):
                # Integrate streamline using gradient
                streamline_x, streamline_y, streamline_z = [x_base], [y_base], [z_centers[z_start_idx]]
                x_curr, y_curr, z_idx = x_base, y_base, z_start_idx
                
                # Follow flow for a few steps
                for step in range(8):
                    if z_idx <= 0 or z_idx >= len(z_centers)-1:
                        break
                    
                    # Get gradient at current position (negative for heat flow direction)
                    gr = -grad_T_r[i_r, z_idx]
                    gz = -grad_T_z[i_r, z_idx]
               with black text for transparent background
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
               facecolor='none', edgecolor='none', transparent=True
                    x_curr = r_new * np.cos(theta_stream)
                    y_curr = r_new * np.sin(theta_stream)
                    z_idx = z_new_idx
                    
                    streamline_x.append(x_curr)
                    streamline_y.append(y_curr)
                    streamline_z.append(z_centers[z_idx])
                
                # Plot streamline colored by temperature at starting point
                if len(streamline_x) > 2:
                    T_stream = T_display[i_r, z_start_idx]
                    color = cmap(norm(T_stream))
                    ax.plot(streamline_x, streamline_y, streamline_z,
                           color=color, alpha=0.6, linewidth=2.0)
    
    # View angle - radiation flows into page
    ax.view_init(elev=15, azim=125r, :]
            
            # Plot as colored line
            for j in range(len(z_centers)-1):
                color = cmap(norm(T_line[j]))
                ax.plot([x_stream, x_stream], 
                       [y_stream, y_stream],
                       [z_line[j], z_line[j+1]],
                       color=color, alpha=0.4, linewidth=1.5)
    
    # Tilted view
    ax.view_init(elev=30, azim=60)
    
    # Colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.65, aspect=18, pad=0.02)
    cbar.set_label('Temperature (keV)', fontsize=15, color='white', weight='bold')
    cbar.ax.tick_params(colors='white', labelsize=13)
    cbar.outline.set_edgecolor('white')
    
    plt.title('Radiation Flow - Streamlines', 
             fontsize=20, color='white', weight='bold', pad=25)
    
    plt.tight_layout()
    print(f"  Saving to {output_file}...")
    plt.savefig(output_file, dpi=300, bbox_inches='tight',
               facecolor='black', edgecolor='none')
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
    
    # Option 1: Thin wedge + floating isosurfaces (MAIN OPTION)
    print("\n[1/2] Creating wedge with floating isosurfaces (RECOMMENDED)...")
    visualize_wedge_with_isosurfaces(solution_file,
                                     output_file='tilted_wedge_isosurfaces.png',
                                     wedge_angle=15,
                                     n_isosurfaces=5,
                                     downsample=2)  # Less downsampling for refined mesh
    
    # Option 2: Thin wedge + gradient streamlines
    print("\n[2/2] Creating wedge with gradient-based streamlines...")
    visualize_wedge_with_streamlines(solution_file,
                                    output_file='tilted_wedge_streamlines.png',
                                    wedge_angle=20,
                                    downsample=3)
    
    print("\n" + "="*75)
    print("Tilted visualizations complete!")
    print("="*75)
