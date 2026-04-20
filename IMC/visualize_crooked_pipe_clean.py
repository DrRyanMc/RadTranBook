#!/usr/bin/env python3
"""
Clean tilted wedge visualization - No labels, smooth curves

Creates publication-ready visualization with:
- Transparent background
- No labels or colorbar
- Smooth curved streamlines
- Tilted angle (radiation flows into page)
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
    """Vibrant artistic colormap with smooth alpha ramp at low end"""
    from matplotlib.colors import LinearSegmentedColormap

    # Define color stops and alpha ramp for smooth gradient with transparent low end
    cdict = {
        'red': [
            (0.0,  0.051, 0.051),
            (0.13, 0.106, 0.106),
            (0.29, 0.255, 0.255),
            (0.43, 0.467, 0.467),
            (0.57, 0.878, 0.878),
            (0.71, 1.0,   1.0),
            (0.86, 1.0,   1.0),
            (1.0,  1.0,   1.0)
        ],
        'green': [
            (0.0,  0.106, 0.106),
            (0.13, 0.149, 0.149),
            (0.29, 0.353, 0.353),
            (0.43, 0.553, 0.553),
            (0.57, 0.882, 0.882),
            (0.71, 0.851, 0.851),
            (0.86, 0.419, 0.419),
            (1.0,  0.027, 0.027)
        ],
        'blue': [
            (0.0,  0.165, 0.165),
            (0.13, 0.231, 0.231),
            (0.29, 0.467, 0.467),
            (0.43, 0.663, 0.663),
            (0.57, 0.866, 0.866),
            (0.71, 0.239, 0.239),
            (0.86, 0.208, 0.208),
            (1.0,  0.431, 0.431)
        ],
        'alpha': [
            (0.0,  0.0,   0.0),
            (0.05, 0.0,   0.0),
            (0.08, 0.7,   0.7),  # Max alpha reduced to 0.7 for better streamline visibility
            (1.0,  0.7,   0.7)
        ]
    }
    return LinearSegmentedColormap('vibrant_transparent', cdict, N=256)
        #               (0.43, 0.553, 0.553),
        #               (0.57, 0.882, 0.882),
        #               (0.71, 0.851, 0.851),
        #               (0.86, 0.419, 0.419),
        #               (1.0,  0.027, 0.027)],
        #     'blue':  [(0.0,  0.165, 0.165),
        #               (0.13, 0.231, 0.231),
        #               (0.29, 0.467, 0.467),
        #               (0.43, 0.663, 0.663),
        #               (0.57, 0.866, 0.866),
        #               (0.71, 0.239, 0.239),
        #               (0.86, 0.208, 0.208),
        #               (1.0,  0.431, 0.431)],
        #     'alpha': [(0.0,  0.0,   0.0),     # Fully transparent at low end
        #               (0.05, 0.0,   0.0),     # Still transparent
        #               (0.08, 1.0,   1.0),     # Fade in quickly
        #               (1.0,  1.0,   1.0)]     # Opaque for rest
        # }
        # return LinearSegmentedColormap('vibrant_transparent', cdict, N=256)


def visualize_wedge_with_isosurfaces(solution_file, output_file='wedge_iso_clean.png',
                                     wedge_angle=15, n_isosurfaces=5, downsample=2):
    """
    Create thin wedge slice with floating transparent isosurfaces
    Clean version - no labels
    """
    print(f"Creating clean wedge + isosurfaces from {solution_file}...")
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
    fig = plt.figure(figsize=(16, 12), facecolor='none')
    ax = fig.add_subplot(111, projection='3d', facecolor='none')
    
    # Clean appearance - no grids, no axes
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

    # Set axis limits for consistent aspect ratio
    max_range = max(z_centers[-1], 2*r_centers[-1])
    ax.set_xlim([-max_range*0.6, max_range*0.6])
    ax.set_ylim([-max_range*0.6, max_range*0.6])
    ax.set_zlim([0, z_centers[-1]])

    # Set axis limits for consistent aspect ratio
    max_range = max(z_centers[-1], 2*r_centers[-1])
    ax.set_xlim([-max_range*0.6, max_range*0.6])
    ax.set_ylim([-max_range*0.6, max_range*0.6])
    ax.set_zlim([0, z_centers[-1]])

    # Set axis limits to match isosurfaces for consistent aspect ratio
    max_range = max(z_centers[-1], 2*r_centers[-1])
    ax.set_xlim([-max_range*0.6, max_range*0.6])
    ax.set_ylim([-max_range*0.6, max_range*0.6])
    ax.set_zlim([0, z_centers[-1]])
    
    # Add floating transparent isosurfaces
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
    
    # Viewing angle - above and to the left, showing cut and streamlines arcing over top
    ax.view_init(elev=35, azim=100)
    
    max_range = max(z_centers[-1], 2*r_centers[-1])
    ax.set_xlim([-max_range*0.6, max_range*0.6])
    ax.set_ylim([-max_range*0.6, max_range*0.6])
    ax.set_zlim([0, z_centers[-1]])
    
    # No labels, no colorbar
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    print(f"  Saving to {output_file}...")
    plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0,
               facecolor='none', edgecolor='none', transparent=True)
    print(f"  Saved: {output_file}")
    plt.close()


def visualize_wedge_with_streamlines(solution_file, output_file='wedge_stream_clean.png',
                                     wedge_angle=20, downsample=3):
    """
    Thin wedge with smooth curved streamlines
    Clean version - no labels
    """


    
    print(f"Creating clean wedge + streamlines from {solution_file}...")
    print(f"  Using downsample factor: {downsample}")
    

    # Load data
    sol = load_solution(solution_file)
    r_centers = sol['r_centers'][::downsample]
    z_centers = sol['z_centers'][::downsample]
    T_rz = sol['T_final'][::downsample, ::downsample]

    print(f"  Mesh size: {len(r_centers)} × {len(z_centers)}")

    # Interpolate to higher resolution for wedge surface
    from scipy.interpolate import RegularGridInterpolator
    upscale = 3  # 3x finer grid
    r_hr = np.linspace(r_centers[0], r_centers[-1], len(r_centers)*upscale)
    z_hr = np.linspace(z_centers[0], z_centers[-1], len(z_centers)*upscale)
    interp_func = RegularGridInterpolator((r_centers, z_centers), T_rz)
    R_hr, Z_hr = np.meshgrid(r_hr, z_hr, indexing='ij')
    pts_hr = np.stack([R_hr.ravel(), Z_hr.ravel()], axis=-1)
    T_hr = interp_func(pts_hr).reshape(R_hr.shape)

    # Enhanced range
    T_min_display = T_hr.min()  # Use true minimum for transparency
    T_max_display = T_hr.max()
    T_display_hr = np.clip(T_hr, T_min_display, T_max_display)
    
    # Calculate temperature gradient
    print("  Computing temperature gradient...")
    dr = np.diff(r_centers).mean()
    dz = np.diff(z_centers).mean()
    grad_T_r = np.gradient(T_rz, dr, axis=0)
    grad_T_z = np.gradient(T_rz, dz, axis=1)
    
    # Figure with transparent background
    fig = plt.figure(figsize=(16, 12), facecolor='none')
    ax = fig.add_subplot(111, projection='3d', facecolor='none')
    ax.grid(False)
    ax.set_axis_off()
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    cmap = create_vibrant_colormap()
    norm = PowerNorm(gamma=0.5, vmin=T_min_display, vmax=T_max_display)
    
    # Thin wedge (high-res for surface)
    print("  Rendering wedge slice...")
    wedge_theta = np.linspace(0, np.radians(wedge_angle), 20)
    R_wedge_hr, THETA_wedge_hr, Z_wedge_hr = np.meshgrid(r_hr, wedge_theta, z_hr, indexing='ij')
    X_wedge_hr = R_wedge_hr * np.cos(THETA_wedge_hr)
    Y_wedge_hr = R_wedge_hr * np.sin(THETA_wedge_hr)
    T_wedge_hr = np.repeat(T_display_hr[:, np.newaxis, :], len(wedge_theta), axis=1)

    ax.plot_surface(X_wedge_hr[-1, :, :], Y_wedge_hr[-1, :, :], Z_wedge_hr[-1, :, :],
                   facecolors=cmap(norm(T_wedge_hr[-1, :, :])),
                   shade=True, alpha=1.0, antialiased=True,
                   linewidth=0, edgecolor='none')

    ax.plot_surface(X_wedge_hr[:, 0, :], Y_wedge_hr[:, 0, :], Z_wedge_hr[:, 0, :],
                   facecolors=cmap(norm(T_display_hr[:, :])),
                   shade=True, antialiased=True,
                   linewidth=0, edgecolor='none')

    # Set axis limits for consistent aspect ratio
    max_range = max(z_centers[-1], 2*r_centers[-1])
    ax.set_xlim([-max_range*0.6, max_range*0.6])
    ax.set_ylim([-max_range*0.6, max_range*0.6])
    ax.set_zlim([0, z_centers[-1]])
    
    # Smooth curved streamlines
    print("  Adding smooth curved streamlines...")
    # Restore previous working streamline logic
    print("  Adding gradient-based flow streamlines...")
    n_streamlines = 20
    stream_r = np.linspace(0.15, r_centers[-1]*0.85, n_streamlines)
    # Use more angles, including above/front (0°, 60°, 120°, 180°, 240°, 300°)
    angle_offsets = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360]
    for r_stream in stream_r:
        i_r = np.argmin(np.abs(r_centers - r_stream))
        for angle_offset in angle_offsets:
            theta_stream = np.radians(angle_offset)
            x_base = r_stream * np.cos(theta_stream)
            y_base = r_stream * np.sin(theta_stream)
            # Start from several z positions and trace the flow
            for z_start_idx in range(5, len(z_centers)-5, max(1, len(z_centers)//10)):
                streamline_x, streamline_y, streamline_z = [x_base], [y_base], [z_centers[z_start_idx]]
                r_curr = r_stream
                z_curr = z_centers[z_start_idx]
                theta_curr = theta_stream
                # Integrate streamline with smooth curves
                for step in range(12):
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
                    # Get gradient (heat flows opposite to gradient)
                    gr = -grad_T_r[i_r_curr, z_idx_curr]
                    gz = -grad_T_z[i_r_curr, z_idx_curr]
                    # Add gentle swirl for aesthetic curves
                    tangential_component = 0.3 * np.sin(step * 0.4)
                    # Normalize and step
                    mag = np.sqrt(gr**2 + gz**2) + 1e-10
                    step_size = 0.08
                    r_curr += (gr / mag) * step_size
                    z_curr += (gz / mag) * step_size
                    theta_curr += tangential_component * 0.2
                # Plot smooth streamline
                if len(streamline_x) > 2:
                    # Use nearest indices in high-res grid for color
                    i_r_hr = np.clip(np.argmin(np.abs(r_hr - r_curr)), 0, len(r_hr)-1)
                    z_idx_hr = np.clip(np.argmin(np.abs(z_hr - z_curr)), 0, len(z_hr)-1)
                    T_stream = T_display_hr[i_r_hr, z_idx_hr]
                    color = cmap(norm(T_stream))
                    ax.plot(streamline_x, streamline_y, streamline_z,
                            color=color, alpha=0.5, linewidth=1.8)
    
    # View angle - above and to the left, showing cut and streamlines arcing over top
    ax.view_init(elev=-155, azim=275,roll=105)
    
    # No labels, no colorbar
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    print(f"  Saving to {output_file}...")
    plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0,
               facecolor='none', edgecolor='none', transparent=True)
    print(f"  Saved: {output_file}")
    plt.close()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python visualize_crooked_pipe_tilted_v2.py <solution_file.npz>")
        sys.exit(1)
    
    solution_file = sys.argv[1]
    
    print("="*75)
    print("CLEAN TILTED VISUALIZATIONS - No labels, smooth curves")
    print("Using refined mesh solution (168×354) for high detail")
    print("="*75)
    
    # Option 1: Thin wedge + floating isosurfaces
    print("\n[1/2] Creating clean wedge with isosurfaces...")
    # visualize_wedge_with_isosurfaces(solution_file,
    #                                  output_file='tilted_wedge_isosurfaces.png',
    #                                  wedge_angle=15,
    #                                  n_isosurfaces=5,
    #                                  downsample=1)
    
    # Option 2: Thin wedge + smooth streamlines
    print("\n[2/2] Creating clean wedge with smooth streamlines...")
    visualize_wedge_with_streamlines(solution_file,
                                    output_file='tilted_wedge_streamlines.png',
                                    wedge_angle=20,
                                    downsample=1)
    
    print("\n" + "="*75)
    print("Clean visualizations complete!")
    print("="*75)
