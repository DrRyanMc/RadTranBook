#!/usr/bin/env python3
"""
3D Visualization Post-Processing for Crooked Pipe IMC Problem

This script creates publication-quality 3D visualizations of the axisymmetric
Crooked Pipe solution by revolving the (r,z) data around the z-axis.

Usage:
    python visualize_crooked_pipe_3d.py <solution_file.npz> --viz-type [cutaway|dual|timeseries]
    
Examples:
    python visualize_crooked_pipe_3d.py crooked_pipe_imc_solution_refined_Nb100000_114x582.npz --viz-type cutaway
    python visualize_crooked_pipe_3d.py crooked_pipe_imc_solution_refined_Nb100000_114x582.npz --viz-type dual --time 5.0
"""

import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib import cm
import sys

try:
    import pyvista as pv
    HAVE_PYVISTA = True
except ImportError:
    HAVE_PYVISTA = False
    print("Warning: PyVista not installed. Install with: pip install pyvista")
    print("Falling back to matplotlib 3D (limited quality)")


def load_solution(filename):
    """Load IMC solution from npz file"""
    data = np.load(filename, allow_pickle=True)
    
    solution = {
        'r_centers': data['r_centers'],
        'z_centers': data['z_centers'],
        'T_final': data['T_final'],
        'Tr_final': data['Tr_final'],
        'times': data['times'],
        'fiducial_data': data['fiducial_data'].item() if 'fiducial_data' in data else None,
        'fiducial_data_rad': data['fiducial_data_rad'].item() if 'fiducial_data_rad' in data else None,
    }
    
    return solution


def revolve_to_3d(r_centers, z_centers, T_rz, n_theta=60, downsample_factor=2):
    """
    Revolve 2D axisymmetric (r,z) data to 3D (x,y,z) cylindrical coordinates
    
    Parameters:
    -----------
    r_centers : array (nr,)
        Radial coordinates (cell centers)
    z_centers : array (nz,)
        Axial coordinates (cell centers)
    T_rz : array (nr, nz)
        Temperature field in (r,z)
    n_theta : int
        Number of angular divisions
    downsample_factor : int
        Factor by which to downsample r and z (for speed)
        
    Returns:
    --------
    grid : pyvista.StructuredGrid
        3D unstructured grid with temperature data
    """
    # Downsample for speed
    r_sample = r_centers[::downsample_factor]
    z_sample = z_centers[::downsample_factor]
    T_sample = T_rz[::downsample_factor, ::downsample_factor]
    
    print(f"  Downsampled: {len(r_centers)}×{len(z_centers)} -> {len(r_sample)}×{len(z_sample)}")
    
    # Create angular coordinate
    theta = np.linspace(0, 2*np.pi, n_theta)
    
    # Create meshgrid
    R, THETA, Z = np.meshgrid(r_sample, theta, z_sample, indexing='ij')
    
    # Convert to Cartesian
    X = R * np.cos(THETA)
    Y = R * np.sin(THETA)
    
    # Expand temperature to 3D (same value at all theta for axisymmetry)
    T_3d = np.repeat(T_sample[:, np.newaxis, :], n_theta, axis=1)
    
    # Create structured grid
    grid = pv.StructuredGrid(X, Y, Z)
    grid['Temperature'] = T_3d.flatten(order='F')
    
    return grid


def create_colormap_artistic():
    """Create an artistic colormap: deep blue -> purple -> orange -> yellow"""
    from matplotlib.colors import LinearSegmentedColormap
    
    colors = [
        (0.0, '#0a0e27'),  # Deep blue-black
        (0.2, '#1a237e'),  # Deep blue
        (0.4, '#673ab7'),  # Purple
        (0.6, '#ff6f00'),  # Orange
        (0.8, '#ffd54f'),  # Gold
        (1.0, '#ffeb3b'),  # Bright yellow
    ]
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('artistic', 
                                             [c[1] for c in colors],
                                             N=n_bins)
    return cmap


def visualize_cutaway(solution_file, output_file='crooked_pipe_cutaway.png', 
                     wedge_angle=75, n_isosurfaces=4, resolution=1920,
                     time_index=-1):
    """
    Create cut-away view with nested isosurfaces
    
    Parameters:
    -----------
    solution_file : str
        Path to .npz solution file
    output_file : str
        Output image filename
    wedge_angle : float
        Angle of wedge to remove (degrees)
    n_isosurfaces : int
        Number of temperature isosurfaces to show
    resolution : int
        Image width in pixels
    time_index : int
        Which time snapshot to use (-1 for final)
    """
    if not HAVE_PYVISTA:
        print("PyVista required for cutaway visualization")
        return
    
    print(f"Creating cutaway visualization from {solution_file}...")
    
    # Load data
    sol = load_solution(solution_file)
    r_centers = sol['r_centers']
    z_centers = sol['z_centers']
    
    # Use final temperature or specific time
    T_rz = sol['T_final']
    
    print(f"  Temperature range: {T_rz.min():.4f} to {T_rz.max():.4f} keV")
    
    # Create 3D grid (downsample for speed)
    print("  Revolving to 3D...")
    grid = revolve_to_3d(r_centers, z_centers, T_rz, n_theta=60, downsample_factor=3)
    
    # Set up plotter with white background for publication
    # Try interactive mode for macOS compatibility
    try:
        plotter = pv.Plotter(off_screen=True, window_size=[resolution, resolution])
    except:
        print("  Note: Off-screen rendering not available, using interactive mode")
        print("  Please screenshot the window manually")
        plotter = pv.Plotter(window_size=[resolution, resolution])
    plotter.set_background('white')
    
    # Create artistic colormap
    cmap = create_colormap_artistic()
    
    # Create isosurfaces
    T_min, T_max = T_rz.min(), T_rz.max()
    T_levels = np.linspace(T_min + 0.1*(T_max-T_min), T_max*0.9, n_isosurfaces)
    
    print(f"  Creating {n_isosurfaces} isosurfaces...")
    for idx, T_level in enumerate(T_levels):
        opacity = 0.3 + 0.5 * (idx / (n_isosurfaces-1))  # More opaque for hotter
        
        contour = grid.contour([T_level], scalars='Temperature')
        
        # Clip to remove wedge
        normal = [np.sin(np.radians(wedge_angle)), -np.cos(np.radians(wedge_angle)), 0]
        clipped = contour.clip(normal=normal, origin=[0, 0, 0])
        
        plotter.add_mesh(clipped, 
                        scalars='Temperature',
                        cmap=cmap,
                        opacity=opacity,
                        smooth_shading=True,
                        show_scalar_bar=False)
    
    # Add material interface boundaries (ghosted)
    # Interface locations at z = 2.5, 3.0, 4.0, 4.5
    for z_int in [2.5, 3.0, 4.0, 4.5]:
        # Create thin disk at interface
        disk = pv.Disc(center=[0, 0, z_int], inner=0, outer=2.0, r_res=2, c_res=30)
        clipped_disk = disk.clip(normal=normal, origin=[0, 0, 0])
        plotter.add_mesh(clipped_disk, color='gray', opacity=0.1, line_width=1)
    
    # Camera setup - 3/4 view from above
    plotter.camera_position = [(8, -6, 8), (0, 0, 3.5), (0, 0, 1)]
    plotter.camera.zoom(1.2)
    
    # Add lighting for photorealism
    light1 = pv.Light(position=(10, -10, 10), light_type='scene light')
    light2 = pv.Light(position=(-5, 5, 8), light_type='scene light', intensity=0.3)
    plotter.add_light(light1)
    plotter.add_light(light2)
    
    # Add scalar bar
    plotter.add_scalar_bar(title='Temperature (keV)',
                          n_labels=5,
                          position_x=0.05,
                          position_y=0.05,
                          width=0.6,
                          height=0.08,
                          title_font_size=16,
                          label_font_size=14,
                          color='black')
    
    print(f"  Rendering to {output_file}...")
    plotter.show(screenshot=output_file)
    print(f"  Saved: {output_file}")


def visualize_dual_volumes(solution_file, output_file='crooked_pipe_dual.png',
                          resolution=1920, time_index=-1):
    """
    Create dual semi-transparent volumes showing material and radiation temperatures
    
    Parameters:
    -----------
    solution_file : str
        Path to .npz solution file
    output_file : str
        Output image filename
    resolution : int
        Image width in pixels
    time_index : int
        Which time snapshot to use (-1 for final)
    """
    if not HAVE_PYVISTA:
        print("PyVista required for dual volume visualization")
        return
    
    print(f"Creating dual-volume visualization from {solution_file}...")
    
    # Load data
    sol = load_solution(solution_file)
    r_centers = sol['r_centers']
    z_centers = sol['z_centers']
    T_mat = sol['T_final']
    T_rad = sol['Tr_final']
    
    print(f"  Material T range: {T_mat.min():.4f} to {T_mat.max():.4f} keV")
    print(f"  Radiation T range: {T_rad.min():.4f} to {T_rad.max():.4f} keV")
    
    # Create 3D grids (downsample for speed)
    print("  Revolving to 3D...")
    grid_mat = revolve_to_3d(r_centers, z_centers, T_mat, n_theta=50, downsample_factor=3)
    grid_rad = revolve_to_3d(r_centers, z_centers, T_rad, n_theta=50, downsample_factor=3)
    
    # Set up plotter
    plotter = pv.Plotter(off_screen=True, window_size=[resolution, resolution])
    plotter.set_background('white')
    
    # Warm colormap for material (reds/oranges)
    cmap_mat = plt.cm.get_cmap('YlOrRd')
    # Cool colormap for radiation (blues/cyans)
    cmap_rad = plt.cm.get_cmap('cool')
    
    # Volume render material temperature
    print("  Adding material temperature volume...")
    plotter.add_volume(grid_mat,
                      scalars='Temperature',
                      cmap=cmap_mat,
                      opacity='sigmoid_5',  # Softer opacity function
                      shade=True)
    
    # Add radiation temperature as isosurface with offset
    print("  Adding radiation temperature isosurface...")
    T_rad_level = T_rad.max() * 0.5
    contour_rad = grid_rad.contour([T_rad_level], scalars='Temperature')
    
    plotter.add_mesh(contour_rad,
                    scalars='Temperature',
                    cmap=cmap_rad,
                    opacity=0.4,
                    smooth_shading=True,
                    show_scalar_bar=False)
    
    # Camera setup
    plotter.camera_position = [(7, -7, 9), (0, 0, 3.5), (0, 0, 1)]
    plotter.camera.zoom(1.3)
    
    # Lighting
    light = pv.Light(position=(10, -10, 12), light_type='scene light')
    plotter.add_light(light)
    
    print(f"  Rendering to {output_file}...")
    plotter.show(screenshot=output_file)
    print(f"  Saved: {output_file}")


def visualize_time_series(solution_file, output_file='crooked_pipe_timeseries.png',
                         times=[1.0, 10.0, 100.0, 500.0], resolution=3840):
    """
    Create time series showing evolution (needs checkpoint files at multiple times)
    
    NOTE: This requires running the simulation with --output-times matching
    the requested times, so checkpoint files exist.
    
    For now, this creates a single time view as demonstration.
    """
    print("Time series visualization requires multiple solution files.")
    print("For now, creating single-time view. Future: load multiple checkpoints.")
    
    # For demonstration, just create a nice single view
    visualize_cutaway(solution_file, output_file=output_file, 
                     wedge_angle=60, n_isosurfaces=5, resolution=resolution)


def quick_preview_matplotlib(solution_file, output_file='crooked_pipe_preview.png'):
    """
    Quick 3D preview using matplotlib (lower quality but no dependencies)
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    print(f"Creating matplotlib 3D preview from {solution_file}...")
    
    # Load data
    sol = load_solution(solution_file)
    r_centers = sol['r_centers']
    z_centers = sol['z_centers']
    T_rz = sol['T_final']
    
    # Create coarser 3D grid for matplotlib
    n_theta = 30
    theta = np.linspace(0, 2*np.pi, n_theta)
    
    # Sample data (too dense crashes matplotlib)
    r_sample = r_centers[::3]
    z_sample = z_centers[::3]
    T_sample = T_rz[::3, ::3]
    
    R, THETA, Z = np.meshgrid(r_sample, theta, z_sample, indexing='ij')
    X = R * np.cos(THETA)
    Y = R * np.sin(THETA)
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot outer surface
    cmap = create_colormap_artistic()
    T_outer = T_sample[:, :, -1]
    ax.plot_surface(X[:, :, -1], Y[:, :, -1], Z[:, :, -1],
                   facecolors=cmap(T_outer / T_rz.max()),
                   shade=True, alpha=0.8)
    
    # Plot a cut plane
    theta_cut = n_theta // 4
    T_cut = np.repeat(T_sample[:, theta_cut:theta_cut+1, :], 10, axis=1)
    theta_plane = np.linspace(0, np.pi/2, 10)
    R_plane, THETA_plane, Z_plane = np.meshgrid(r_sample, theta_plane, z_sample, indexing='ij')
    X_plane = R_plane * np.cos(THETA_plane)
    Y_plane = R_plane * np.sin(THETA_plane)
    
    ax.plot_surface(X_plane, Y_plane, Z_plane,
                   facecolors=cmap(T_cut / T_rz.max()),
                   shade=True, alpha=0.6)
    
    ax.set_xlabel('x (cm)')
    ax.set_ylabel('y (cm)')
    ax.set_zlabel('z (cm)')
    ax.set_title('Crooked Pipe 3D (Preview)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"Saved: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Create 3D visualizations of Crooked Pipe IMC solution',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('solution_file', 
                       help='Path to solution .npz file')
    parser.add_argument('--viz-type', 
                       choices=['cutaway', 'dual', 'timeseries', 'preview'],
                       default='cutaway',
                       help='Type of visualization to create')
    parser.add_argument('--output', '-o',
                       help='Output filename (auto-generated if not specified)')
    parser.add_argument('--resolution', type=int, default=1920,
                       help='Image resolution (width in pixels)')
    parser.add_argument('--wedge-angle', type=float, default=75,
                       help='Wedge angle for cutaway view (degrees)')
    parser.add_argument('--n-isosurfaces', type=int, default=4,
                       help='Number of isosurfaces for cutaway view')
    
    args = parser.parse_args()
    
    # Auto-generate output filename if not provided
    if args.output is None:
        base = args.solution_file.replace('.npz', '')
        args.output = f'{base}_3d_{args.viz_type}.png'
    
    # Check if solution file exists
    import os
    if not os.path.exists(args.solution_file):
        print(f"Error: Solution file not found: {args.solution_file}")
        sys.exit(1)
    
    # Create visualization
    if args.viz_type == 'cutaway':
        visualize_cutaway(args.solution_file, 
                         output_file=args.output,
                         wedge_angle=args.wedge_angle,
                         n_isosurfaces=args.n_isosurfaces,
                         resolution=args.resolution)
    elif args.viz_type == 'dual':
        visualize_dual_volumes(args.solution_file,
                              output_file=args.output,
                              resolution=args.resolution)
    elif args.viz_type == 'timeseries':
        visualize_time_series(args.solution_file,
                             output_file=args.output,
                             resolution=args.resolution)
    elif args.viz_type == 'preview':
        quick_preview_matplotlib(args.solution_file,
                                output_file=args.output)
    
    print("\nDone! Review the image and adjust parameters as needed.")
    print("\nTips for publication quality:")
    print("  - Use --resolution 3840 for very high-res (4K)")
    print("  - Adjust --wedge-angle (try 60, 75, 90)")
    print("  - Try --n-isosurfaces 5 or 6 for more detail")
    print("  - Edit lighting/camera in code for perfect angle")


if __name__ == '__main__':
    main()
