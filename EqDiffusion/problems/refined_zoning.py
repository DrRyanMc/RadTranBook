#!/usr/bin/env python3
"""
Mesh Refinement Test Problem - 2D Cylindrical (r-z) Geometry

This problem tests radiation diffusion through a complex geometry with
spatially-varying material properties:
- Optically thick regions: σ_R = 200 cm⁻¹, c_v = 0.5 GJ/(cm³·keV)
- Optically thin regions: σ_R = 0.2 cm⁻¹, c_v = 0.0005 GJ/(cm³·keV)

Source: T = 300 eV (0.3 keV) at z=0.0 and r∈(0.0, 0.5)
"""

import numpy as np
import matplotlib.pyplot as plt
from twoDFV import RadiationDiffusionSolver2D, temperature_from_Er, A_RAD
from numba import njit

T_init = 0.01  # keV

from plotfuncs import *
# =============================================================================
# MATERIAL PROPERTY FUNCTIONS
# =============================================================================

@njit
def is_optically_thick(r, z):
    """
    Determine if location (r, z) is in optically thick region
    
    There are two optically thin channels
    """
    #lower thin region
    if (r >= 1.0 and r<=2.0) and (z<2.0):
        return False  # optically thin
    
    # if (r >= 3.0 and r<=4.0) and (z<2.0):
    #     return False  # optically thin
    if (r >=3.0 and r<=4.0) and (z>3.0):
        return False  # optically thin
    return True



@njit
def rosseland_opacity(Er, r, z):
    """
    Spatially-varying Rosseland opacity
    
    Parameters:
    -----------
    Er : float
        Radiation energy density (unused for this problem)
    r : float
        Radial coordinate (cm)
    z : float
        Axial coordinate (cm)
    
    Returns:
    --------
    sigma_R : float
        Rosseland opacity (cm⁻¹)
    """
    if is_optically_thick(r, z):
        return 200.0  # cm⁻¹
    else:
        return 0.2    # cm⁻¹


@njit
def specific_heat(T, r, z):
    """
    Spatially-varying specific heat 
    
    Parameters:
    -----------
    T : float
        Temperature (keV) (unused for this problem)
    r : float
        Radial coordinate (cm)
    z : float
        Axial coordinate (cm)
    
    Returns:
    --------
    c_v : float
        Specific heat (GJ/(cm³·keV)) [Note: this is ρ*c_v combined]
    """
    if is_optically_thick(r, z):
        return 0.5      # GJ/(cm³·keV)
    else:
        return 0.0005   # GJ/(cm³·keV)


@njit
def material_energy(T, r, z):
    """
    Material energy density  
    
    e(T, r, z) = ρ*c_v(r, z) * T
    Since c_v already includes ρ, we just multiply by T
    """
    cv = specific_heat(T, r, z)
    return cv * T


# =============================================================================
# MESH GENERATION FOR INTERFACE REFINEMENT
# =============================================================================

def generate_refined_z_faces(z_min, z_max, interface_locations, n_refine, n_coarse, refine_width=0.05):
    """
    Generate z-direction face positions with refinement at material interfaces
    
    Parameters:
    -----------
    z_min, z_max : float
        Domain boundaries in z
    interface_locations : list of float
        Z-coordinates where material interfaces occur (e.g., [2.5, 3.0, 4.0, 4.5])
    n_refine : int
        Number of refined cells to subdivide each coarse cell in refinement zones
    n_coarse : int
        Number of coarse cells across entire domain
    refine_width : float
        Width (in cm) around each interface to apply refinement (e.g., 0.05 cm)
    
    Returns:
    --------
    z_faces : ndarray
        Face positions in z with logarithmic refinement at interfaces.
        Cells are smallest at interfaces and grow logarithmically away.
        No cell is wider than (z_max - z_min) / n_coarse.
    """
    # Define refinement zone width around each interface
      # cm on each side of interface
    
    # First, create a uniform coarse grid
    dz_coarse = (z_max - z_min) / n_coarse
    z_coarse_faces = np.linspace(z_min, z_max, n_coarse + 1)
    
    # Sort interfaces
    interfaces = sorted(interface_locations)
    
    # Mark which coarse cells should be refined and store nearest interface
    refine_info = {}  # i -> nearest interface z
    
    for z_int in interfaces:
        for i in range(n_coarse):
            cell_left = z_coarse_faces[i]
            cell_right = z_coarse_faces[i + 1]
            
            # Refine if cell overlaps with refinement zone [z_int - refine_width, z_int + refine_width]
            if (cell_left <= z_int + refine_width) and (cell_right >= z_int - refine_width):
                # Store the interface location for this cell
                if i not in refine_info:
                    refine_info[i] = z_int
                else:
                    # If multiple interfaces affect this cell, use the nearest one
                    if abs(z_int - (cell_left + cell_right)/2) < abs(refine_info[i] - (cell_left + cell_right)/2):
                        refine_info[i] = z_int
    
    # Build the final face list by subdividing marked cells
    z_faces_list = [z_min]
    
    for i in range(n_coarse):
        cell_left = z_coarse_faces[i]
        cell_right = z_coarse_faces[i + 1]
        
        if i in refine_info:
            z_int = refine_info[i]
            # Subdivide this cell with logarithmic spacing around the interface
            z_refined = create_log_spacing_around_interface(
                cell_left, cell_right, z_int, n_refine
            )
            z_faces_list.extend(z_refined)
        else:
            # Keep as single coarse cell
            z_faces_list.append(cell_right)
    
    return np.array(z_faces_list)


def create_log_spacing_around_interface(z_left, z_right, z_int, n_cells):
    """
    Create logarithmically-spaced cell faces within [z_left, z_right],
    with smallest cells near z_int.
    
    Parameters:
    -----------
    z_left, z_right : float
        Bounds of the region to subdivide
    z_int : float
        Interface location where cells should be finest
    n_cells : int
        Number of cells to create
    
    Returns:
    --------
    faces : list
        Face positions (excluding z_left, which is already in the list)
    """
    # Determine if interface is within the cell or outside
    if z_int <= z_left:
        # Interface is to the left, grow cells from left to right
        return create_log_spacing_one_sided(z_left, z_right, n_cells, from_left=True)
    elif z_int >= z_right:
        # Interface is to the right, grow cells from right to left
        return create_log_spacing_one_sided(z_left, z_right, n_cells, from_left=False)
    else:
        # Interface is inside the cell, split into two regions
        n_left = max(1, int(n_cells * (z_int - z_left) / (z_right - z_left)))
        n_right = n_cells - n_left
        
        faces_left = create_log_spacing_one_sided(z_left, z_int, n_left, from_left=False)
        faces_right = create_log_spacing_one_sided(z_int, z_right, n_right, from_left=True)
        
        return faces_left + faces_right


def create_log_spacing_one_sided(z_start, z_end, n_cells, from_left=True):
    """
    Create logarithmically-spaced cells in one direction.
    
    Parameters:
    -----------
    z_start, z_end : float
        Start and end of region
    n_cells : int
        Number of cells
    from_left : bool
        If True, smallest cells at z_start, growing toward z_end
        If False, smallest cells at z_end, growing from z_start
    
    Returns:
    --------
    faces : list
        Face positions (excluding z_start)
    """
    if n_cells <= 1:
        return [z_end]
    
    width = z_end - z_start
    
    # Growth factor for logarithmic spacing
    # We want: w + w*r + w*r^2 + ... + w*r^(n-1) = width
    # Choose r such that the ratio of largest to smallest cell is reasonable
    # A typical choice is r = 1.1 to 1.3
    # Better: solve for r given that we want smooth growth
    # Use r such that cells grow by a factor across the region
    
    # For n cells with growth factor r:
    # width = w * (r^n - 1) / (r - 1)
    # We need to choose r. A reasonable default is r = 1.2
    # Or we can solve for r to give a specific ratio of largest to smallest
    
    # Let's use a ratio of 10:1 for max to min cell size (adjustable)
    max_ratio = 5.0
    r = max_ratio ** (1.0 / (n_cells - 1)) if n_cells > 1 else 1.0
    
    # Calculate first cell width
    if abs(r - 1.0) < 1e-10:
        w0 = width / n_cells
        cell_widths = [w0] * n_cells
    else:
        w0 = width * (r - 1.0) / (r**n_cells - 1.0)
        cell_widths = [w0 * r**i for i in range(n_cells)]
    
    # If from_left=False, reverse the widths so small cells are at the end
    if not from_left:
        cell_widths = cell_widths[::-1]
    
    # Build face positions
    faces = []
    current_pos = z_start
    for w in cell_widths:
        current_pos += w
        faces.append(current_pos)
    
    # Ensure last face is exactly at z_end (fix floating point errors)
    faces[-1] = z_end
    
    return faces


# =============================================================================
# BOUNDARY CONDITION FUNCTIONS
# =============================================================================

def bc_left_axis(Er_boundary, r_val, z_val, geometry, time=0.0):
    """Left boundary (r=0, axis): Reflecting symmetry"""
    return .5, 1.0/(3.0*rosseland_opacity(T_init, r_val, z_val)), 0.0 


def bc_right_open(Er_boundary, r_val, z_val, geometry, time=0.0):
    """Right boundary (r_max): Open/free boundary"""
    # Open boundary: Er = 0 (or small value)
    return .5, 1.0/(3.0*rosseland_opacity(T_init, r_val, z_val)), 0.0  # Neumann: dEr/dz = 0


def bc_bottom_source(Er_boundary, r_val, z_val, geometry, time=0.0):
    """
    Bottom boundary (z=0): Reflecting with source region
    
    Source: T = 300 eV = 0.3 keV at r ∈ (0.0, 0.5)
    """

    T_init = 0.01  # keV
    if (time<500.): #(r_val >= 1.0) and (r_val < 2.0):
        # Source region: T = 0.3 keV => Er = a*T^4
        T_source = 0.3  # keV
        Er_source = A_RAD * T_source**4
        return .5, 1.0/(3.0*rosseland_opacity(T_init, r_val, z_val)), Er_source/2  #1.0, 0.0, Er_source # Dirichlet: Er = Er_source
    else:
        # vacuum elsewhere
        return .5, 1.0/(3.0*rosseland_opacity(T_init, r_val, z_val)), 0.0  # Neumann: dEr/dz = 0


def bc_top_open(Er_boundary, r_val, z_val, geometry, time=0.0):
    """Top boundary (z_max): Open/free boundary"""
    T_init = 0.01  # keV)
    if (time<500.): #(r_val >= 3.0) and (r_val < 4.0):
        # Source region: T = 0.3 keV => Er = a*T^4
        T_source = 0.3  # keV
        Er_source = A_RAD * T_source**4
        return .5, 1.0/(3.0*rosseland_opacity(T_init, r_val, z_val)), Er_source/2  #1.0, 0.0, Er_source #.5, 1.0/(3.0*rosseland_opacity(T_init, r_val, z_val)), Er_source/2  # Dirichlet: Er = Er_source
    else:
        return .5, 1.0/(3.0*rosseland_opacity(T_init, r_val, z_val)), 0.0  # Neumann: dEr/dz = 0


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_material_properties(solver):
    """Plot the material property distribution"""
    r_centers = solver.coord1_centers
    z_centers = solver.coord2_centers
    
    # Create material property arrays
    opacity_field = np.zeros((solver.n1_cells, solver.n2_cells))
    cv_field = np.zeros((solver.n1_cells, solver.n2_cells))
    
    for i in range(solver.n1_cells):
        for j in range(solver.n2_cells):
            r = r_centers[i]
            z = z_centers[j]
            opacity_field[i, j] = rosseland_opacity(0.0, r, z)
            cv_field[i, j] = specific_heat(0.0, r, z)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    R, Z = np.meshgrid(r_centers, z_centers, indexing='ij')
    
    # Plot opacity
    im1 = ax1.pcolormesh(Z, R, opacity_field, shading='auto', cmap='RdYlBu_r')
    ax1.set_xlabel('z (cm)', fontsize=12)
    ax1.set_ylabel('x (cm)', fontsize=12)
    ax1.set_title('Rosseland Opacity σ_R (cm⁻¹)', fontsize=13, fontweight='bold')
    cbar1 = plt.colorbar(im1, ax=ax1, label='σ_R (cm⁻¹)')
    ax1.set_aspect('equal')
    
    # Plot specific heat
    im2 = ax2.pcolormesh(Z, R, cv_field, shading='auto', cmap='RdYlBu_r')
    ax2.set_xlabel('z (cm)', fontsize=12)
    ax2.set_ylabel('x (cm)', fontsize=12)
    ax2.set_title('Specific Heat ρc_v (GJ/(cm³·keV))', fontsize=13, fontweight='bold')
    cbar2 = plt.colorbar(im2, ax=ax2, label='ρc_v (GJ/(cm³·keV))')
    ax2.set_aspect('equal')
    
    plt.suptitle('Material Properties', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('refinement_test_materials.png', dpi=150, bbox_inches='tight')
    print("Saved: refinement_test_materials.png")
    plt.close()


def plot_solution(solver, title="Refinement Test Solution"):
    """Plot temperature and radiation energy density"""
    r_centers, z_centers, Er_2d = solver.get_solution()
    T_2d = temperature_from_Er(Er_2d)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    R, Z = np.meshgrid(r_centers, z_centers, indexing='ij')
    
    # Plot temperature
    im1 = ax1.pcolormesh(Z, R, T_2d, shading='auto', cmap='RdYlBu_r', edgecolors='black', linewidth=0.5)
    ax1.set_xlabel('z (cm)', fontsize=12)
    ax1.set_ylabel('x (cm)', fontsize=12)
    ax1.set_title('Temperature (keV)', fontsize=13, fontweight='bold')
    cbar1 = plt.colorbar(im1, ax=ax1, label='T (keV)')
    ax1.set_aspect('equal')
    
    # Add source region indicator
    ax1.axhline(y=0.5, color='cyan', linestyle='--', linewidth=2, alpha=0.7, label='r=0.5 (source boundary)')
    ax1.axvline(x=0.0, color='cyan', linestyle='--', linewidth=2, alpha=0.7)
    ax1.legend(loc='upper right', fontsize=9)
    
    # Plot radiation energy density
    im2 = ax2.pcolormesh(Z, R, Er_2d, shading='auto', cmap='RdYlBu_r')
    ax2.set_xlabel('z (cm)', fontsize=12)
    ax2.set_ylabel('x (cm)', fontsize=12)
    ax2.set_title('Radiation Energy Density (GJ/cm³)', fontsize=13, fontweight='bold')
    cbar2 = plt.colorbar(im2, ax=ax2, label='E_r (GJ/cm³)')
    ax2.set_aspect('equal')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('refined_mesh_solution.png', dpi=150, bbox_inches='tight')
    print("Saved: refined_mesh_solution.png")
    plt.show()


def plot_cross_sections(solver):
    """Plot cross sections at different locations"""
    r_centers, z_centers, Er_2d = solver.get_solution()
    T_2d = temperature_from_Er(Er_2d)
    
    z_values = [1, 1.95, 2.05, 2.95, 3.05, 4.0]
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']
    line_types = ['-', '--', '-.', '-', '--', '-.']
    r_values = [1.5, 1.95, 3.05, 3.5]
    
    # Figure 1: Radial profiles at different z (linear scale)
    fig1, ax1 = plt.subplots(1, 1, figsize=(8, 6))
    for z_val, color in zip(z_values, colors):
        j = np.argmin(np.abs(z_centers - z_val))
        ax1.plot(r_centers, T_2d[:, j], color=color, linewidth=2, 
                 label=f'z = {z_centers[j]:.2f} cm', linestyle=line_types[z_values.index(z_val)])
    ax1.set_xlabel('x (cm)', fontsize=11)
    ax1.set_ylabel('Temperature (keV)', fontsize=11)
    #ax1.set_title('Radial Temperature Profiles', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    show('refined_mesh_radial_linear.pdf')
    print("Saved: refined_mesh_radial_linear.pdf")
    plt.close()
    
    # Figure 2: Axial profiles at different r (linear scale)
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
    for r_val, color in zip(r_values, colors):
        i = np.argmin(np.abs(r_centers - r_val))
        ax2.plot(z_centers, T_2d[i, :], color=color, linewidth=2, 
                 label=f'r = {r_centers[i]:.2f} cm', 
                 linestyle=line_types[r_values.index(r_val)])
    ax2.set_xlabel('z (cm)', fontsize=11)
    ax2.set_ylabel('Temperature (keV)', fontsize=11)
    #ax2.set_title('Axial Temperature Profiles', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    show('refined_mesh_axial_linear.pdf')
    print("Saved: refined_mesh_axial_linear.pdf")
    plt.close()
    
    # Figure 3: Radial profiles at different z (log scale)
    fig3, ax3 = plt.subplots(1, 1, figsize=(8, 6))
    for z_val, color in zip(z_values, colors):
        j = np.argmin(np.abs(z_centers - z_val))
        ax3.semilogy(r_centers, T_2d[:, j], color=color, linewidth=2, 
                     label=f'z = {z_centers[j]:.2f} cm', 
                     linestyle=line_types[z_values.index(z_val)])
    ax3.set_xlabel('x (cm)', fontsize=11)
    ax3.set_ylabel('Temperature (keV)', fontsize=11)
    #ax3.set_title('Radial Temperature Profiles (log scale)', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    show('refined_mesh_radial_log.pdf')
    print("Saved: refined_mesh_radial_log.pdf")
    plt.close()
    
    # Figure 4: Axial profiles at different r (log scale)
    fig4, ax4 = plt.subplots(1, 1, figsize=(8, 6))
    for r_val, color in zip(r_values, colors):
        i = np.argmin(np.abs(r_centers - r_val))
        ax4.semilogy(z_centers, T_2d[i, :], color=color, linewidth=2, 
                     label=f'r = {r_centers[i]:.2f} cm', 
                     linestyle=line_types[r_values.index(r_val)])
    ax4.set_xlabel('z (cm)', fontsize=11)
    ax4.set_ylabel('Temperature (keV)', fontsize=11)
    #ax4.set_title('Axial Temperature Profiles (log scale)', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    plt.tight_layout()
    show('refined_mesh_axial_log.pdf')
    print("Saved: refined_mesh_axial_log.pdf")
    plt.close()


def plot_fiducial_points_history(times, fiducial_temps):
    """
    Plot temperature history at fiducial points on log-log scale
    
    Parameters:
    -----------
    times : ndarray
        Array of times (ns)
    fiducial_temps : dict
        Dictionary with keys as point labels and values as temperature arrays
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    
    # Different markers for each point
    markers = ['o', 's', '^', 'd', 'v']
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    
    # Plot each fiducial point
    for idx, (label, temps) in enumerate(fiducial_temps.items()):
        ax.loglog(times, temps, 
                 marker=markers[idx % len(markers)],
                 color=colors[idx % len(colors)],
                 linewidth=2,
                 markersize=6,
                 markevery=max(1, len(times)//20),  # Show markers periodically
                 label=label,
                 alpha=0.8)
    
    ax.set_xlabel('Time (ns)', fontsize=15, fontweight='bold')
    ax.set_ylabel('Temperature (keV)', fontsize=15, fontweight='bold')
    #ax.set_title('Temperature History at Fiducial Points', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12, loc='best', framealpha=0.9)
    ax.grid(True, which='both', alpha=0.3, linestyle='--')
    ax.grid(True, which='minor', alpha=0.15, linestyle=':')
    
    plt.tight_layout()
    show('refined_mesh_fiducial_history.pdf', close_after=True)
    plt.show()


def plot_colormap_at_time(solver, time_value, save_prefix='refined_mesh', first_one=False):
    """
    Plot temperature colormap at a specific time
    
    Parameters:
    -----------
    solver : RadiationDiffusionSolver2D
        Solver object with current solution
    time_value : float
        Current time value (ns)
    save_prefix : str
        Prefix for saved filename
    first_one : bool
        If True, add special title or annotations for the first plot
    """
    r_centers, z_centers, Er_2d = solver.get_solution()
    T_2d = temperature_from_Er(Er_2d)
    
    #make the figure equal aspect ratio
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    if first_one:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6*1.275))

    R, Z = np.meshgrid(r_centers, z_centers, indexing='ij')
    
    # Plot temperature
    im = ax.pcolormesh(Z, R, T_2d, shading='auto', cmap='plasma', vmin=0.0, vmax=0.3)
    ax.set_xlabel('z (cm)', fontsize=12)
    ax.set_ylabel('x (cm)', fontsize=12)
    #ax.set_title(f'Temperature at t = {time_value:.3f} ns', fontsize=14, fontweight='bold')
    
    #only put colorbar on first one and make it horizontal across the top
    if first_one:
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', location='top', pad=0.15)
        cbar.set_label('Temperature (keV)', fontsize=11, fontweight='bold')
    
    ax.set_aspect('equal')
    
    # Add source region indicator
    #ax.axhline(y=0.5, color='cyan', linestyle='--', linewidth=1.5, alpha=0.5)
    #ax.axvline(x=0.0, color='cyan', linestyle='--', linewidth=1.5, alpha=0.5)
    
    plt.tight_layout()
    filename = f'{save_prefix}_t_{time_value:.3f}ns.png'
    if first_one:
        show(filename, cbar_ax=cbar.ax, close_after=True)
    else:
        show(filename, close_after=True)
    print(f"Saved: {filename}")
    #plot_solution(solver, title=f'Temperature at t = {time_value:.3f} ns')


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main(output_times=None, use_refined_mesh=False, dt_initial=1e-3, dt_max=1.0, dt_increase_factor=1.1):
    """
    Run the Crooked Pipe test problem
    
    Parameters:
    -----------
    output_times : list or None
        List of times (ns) at which to save colormap plots.
        If None, defaults to [1.0, 5.0, 10.0, 100.0]
    use_refined_mesh : bool
        If True, use mesh with refinement at material interfaces
    dt_initial : float
        Initial time step (ns)
    dt_max : float
        Maximum time step (ns)
    dt_increase_factor : float
        Factor by which to increase dt each step (e.g., 1.1 for 10% increase)
    """
    if output_times is None:
        output_times = [1.0, 5.0, 10.0,100.0]
    
    print("="*70)
    print("CROOKED PIPE TEST PROBLEM")
    print("="*70)
    print("2D Cylindrical (r-z) Geometry")
    print("Domain: r ∈ [0.0, 2.0] cm, z ∈ [0.0, 7.0] cm")
    print()
    print("Material Properties:")
    print("  Optically thick: σ_R = 200 cm⁻¹, ρc_v = 0.5 GJ/(cm³·keV)")
    print("  Optically thin:  σ_R = 0.2 cm⁻¹,  ρc_v = 0.0005 GJ/(cm³·keV)")
    print()
    print("Boundary Conditions:")
    print("  Left (r=0): Reflecting (symmetry axis)")
    print("  Right (r_max): Open")
    print("  Bottom (z=0): Source at r<0.5 (T=0.3 keV), Reflecting elsewhere")
    print("  Top (z_max): Open")
    print()
    print(f"Output times for colormaps: {output_times} ns")
    print(f"Using refined mesh at interfaces: {use_refined_mesh}")
    print(f"Time stepping: dt_initial = {dt_initial} ns, dt_max = {dt_max} ns, increase factor = {dt_increase_factor}")
    print()
    
    # Create solver
    print("Setting up solver...")
    dt = dt_initial
    rmin = 0.
    if use_refined_mesh:
        # Generate custom mesh with refinement at material interfaces
        # Interface locations: z = 2.5, 3.0, 4.0, 4.5 cm
        r_faces = generate_refined_z_faces(
            z_min=rmin+0.0, 
            z_max=rmin+5.0,
            interface_locations=[rmin+3.0,rmin+4.0],
            n_refine=10,   # 10 cells in each refinement zone
            n_coarse=100   # Approximate target for coarse regions
        )
        z_faces = generate_refined_z_faces(
            z_min=0.0, 
            z_max=5.0,
            interface_locations=[3.0],
            n_refine=10,   # 10 cells in each refinement zone
            n_coarse=100   # Approximate target for coarse regions
        )
        print("z_faces:", z_faces)
        print("="*40)
        print("r_faces:", r_faces)
        print("="*40)
        #make a plot of the mesh to verify
        plt.figure(figsize=(6,6))
        for r in r_faces:
            plt.plot([0,5],[r, r], color='black', linewidth=0.25, alpha=0.5)
        for z in z_faces:
            plt.plot([z, z], [0,5], color='black', linewidth=0.25, alpha=0.5)
        plt.xlim(0, 5)
        plt.ylim(0, 5)
        plt.xlabel('z (cm)', fontsize=12)
        plt.ylabel('x (cm)', fontsize=12)
        #plt.title(f'Refined Mesh for Crooked Pipe Nr = {len(r_faces)-1}, Nz = {len(z_faces)-1}', fontsize=14, fontweight='bold')
        plt.gca().set_aspect('equal')
        plt.tight_layout()
        #print mesh info, including the minimum dr and dz
        dr_min = np.min(np.diff(r_faces))
        dz_min = np.min(np.diff(z_faces))
        print(f"  Custom mesh generated:")
        print(f"  Minimum dr = {dr_min:.5f} cm, Minimum dz = {dz_min:.5f} cm")
        
        print(f"  Custom mesh: {len(r_faces)-1} × {len(z_faces)-1} cells")
        print(f"  Refinement at z = 2.5, 3.0, 4.0, 4.5 cm")
        
        show("refinement_test_refined_mesh.pdf", close_after=False)
        solver = RadiationDiffusionSolver2D(
            coord1_faces=r_faces,
            coord2_faces=z_faces,
            geometry='cartesian',
            dt=dt,
            max_newton_iter=30,
            newton_tol=1e-6,
            rosseland_opacity_func=rosseland_opacity,
            specific_heat_func=specific_heat,
            material_energy_func=material_energy,
            left_bc_func=bc_left_axis,
            right_bc_func=bc_right_open,
            bottom_bc_func=bc_bottom_source,
            top_bc_func=bc_top_open,
            theta=1.0,
            use_jfnk=False
        )
    else:
        # Use uniform mesh (original setup)
        solver = RadiationDiffusionSolver2D(
            coord1_min=0.0,
            coord1_max=5.0,
            n1_cells=55,      # radial cells
            coord2_min=0.0,
            coord2_max=5.0,
            n2_cells=55,     # axial cells
            geometry='cartesian',
            dt=dt,
            max_newton_iter=30,
            newton_tol=1e-6,
            rosseland_opacity_func=rosseland_opacity,
            specific_heat_func=specific_heat,
            material_energy_func=material_energy,
            left_bc_func=bc_left_axis,
            right_bc_func=bc_right_open,
            bottom_bc_func=bc_bottom_source,
            top_bc_func=bc_top_open,
            theta=1.0,
            use_jfnk=False
        )
    
    # Initial condition: uniform low background
    print("Setting initial condition...")
    Er_background = A_RAD*T_init**4  # Low background radiation energy
    solver.set_initial_condition(Er_background)
    
    # Plot material properties
    print("\nPlotting material properties...")
    plot_material_properties(solver)
    
    # Define fiducial points for tracking
    fiducial_points = {
        'Point 1: r=1.5, z=1.95': (1.5, 1.95),
        'Point 2: r=1.5, z=2.05': (1.5, 2.05),
        'Point 3: r=3.5, z=3.05': (3.5, 3.05),
        'Point 4: r=3.5, z=2.95': (3.5, 2.95),
    }
    
    # Initialize storage for fiducial point temperatures
    r_centers = solver.coord1_centers
    z_centers = solver.coord2_centers
    
    # Find indices for each fiducial point
    fiducial_indices = {}
    for label, (r_val, z_val) in fiducial_points.items():
        i = np.argmin(np.abs(r_centers - r_val))
        j = np.argmin(np.abs(z_centers - z_val))
        fiducial_indices[label] = (i, j)
        print(f"{label}: grid point (r={r_centers[i]:.3f}, z={z_centers[j]:.3f})")
    
    # Storage for time history
    times = [0.0]  # Include initial time
    fiducial_temps = {label: [T_init] for label in fiducial_points.keys()}
    
    # Track which output times have been saved
    output_times_saved = set()
    
    # Time evolution
    print("\nTime stepping...")
    t_final = max(output_times) if output_times else 10.0  # ns
    save_interval = 1  # Save every step
    t = 0.0
    step = 0
    first_one = True
    
    while t < t_final:
        step += 1
        
        # Store the current dt before any modifications
        dt_current = solver.dt
        
        # Check if we need to decrease dt to hit an output time exactly
        hit_output_time = False
        for output_t in sorted(output_times):
            if output_t > t and t + solver.dt > output_t:
                # We would overshoot this output time, so adjust dt to hit it exactly
                solver.dt = output_t - t
                hit_output_time = True
                break
        
        # Check if we need to decrease dt because t_final-t < dt
        if t + solver.dt > t_final:
            solver.dt = t_final - t
        
        t += solver.dt
        solver.current_time = t  # Update solver's time tracker
        
        if step % 10 == 0 or step == 1:
            print(f"Time step {step}, t = {t:.4e} ns, dt = {solver.dt:.4e} ns")
        
        Er_prev = solver.Er.copy()
        
        # Advance one time step
        if solver.use_jfnk:
            solver.Er = solver.newton_step_jfnk(Er_prev, verbose=False)
        else:
            solver.Er = solver.newton_step_direct(Er_prev, verbose=False)
        
        solver.Er_old = Er_prev.copy()
        
        # Check if we should save a colormap at this time
        for output_t in output_times:
            if output_t not in output_times_saved and abs(t - output_t) < solver.eps:
                print(f"  Saving colormap at t = {t:.3f} ns (requested: {output_t} ns)")
                plot_colormap_at_time(solver, t, first_one=first_one)
                output_times_saved.add(output_t)
                first_one = False
        
        # Save fiducial point data
        if step % save_interval == 0:
            times.append(t)
            
            _, _, Er_2d = solver.get_solution()
            T_2d = temperature_from_Er(Er_2d)
            
            for label, (i, j) in fiducial_indices.items():
                fiducial_temps[label].append(T_2d[i, j])
        
        # Restore and increase time step for next iteration (but don't exceed dt_max)
        # If we hit an output time, restore the dt we would have used, then increase it
        if hit_output_time:
            solver.dt = dt_current
        solver.dt = min(solver.dt * dt_increase_factor, dt_max)
    
    # Output final time if not already done
    if t_final not in output_times_saved and abs(t - t_final) < solver.eps:
        print(f"  Saving colormap at t = {t_final:.3f} ns (final time)")
        plot_colormap_at_time(solver, t_final, first_one=first_one)
    # Convert to numpy arrays
    times = np.array(times)
    for label in fiducial_temps.keys():
        fiducial_temps[label] = np.array(fiducial_temps[label])
        #print(times.shape, fiducial_temps[label].shape)
    # Plot solution
    print("\nPlotting solution...")
    plot_solution(solver, f"Mesh refinement (t = {t:.3f} ns)")
    plot_cross_sections(solver)
    
    # Plot fiducial point history
    print("\nPlotting fiducial point temperature history...")
    plot_fiducial_points_history(times, fiducial_temps)
    
    # Print some diagnostics
    r_centers, z_centers, Er_2d = solver.get_solution()
    T_2d = temperature_from_Er(Er_2d)
    
    #save final T_2d to a npz file with a meaningful name
    #that includes whether refined mesh was used and the number of cells
    mesh_type = "refined" if use_refined_mesh else "uniform"
    n_r_cells = solver.n1_cells
    n_z_cells = solver.n2_cells
    np.savez_compressed(f"refined_mesh_solution_{mesh_type}_{n_r_cells}x{n_z_cells}.npz",
                        r_centers=r_centers,
                        z_centers=z_centers,
                        T_2d=T_2d)
    print(f"Saved final solution to refined_mesh_solution_{mesh_type}_{n_r_cells}x{n_z_cells}.npz")

    print("\n" + "="*70)
    print("SOLUTION SUMMARY")
    print("="*70)
    print(f"Time: {t:.3f} ns")
    print(f"Total steps: {step}")
    print(f"Temperature range: {T_2d.min():.4f} to {T_2d.max():.4f} keV")
    print(f"Max temperature location: r={r_centers[np.unravel_index(T_2d.argmax(), T_2d.shape)[0]]:.3f} cm, "
          f"z={z_centers[np.unravel_index(T_2d.argmax(), T_2d.shape)[1]]:.3f} cm")
    
    # Print fiducial point final temperatures
    print("\nFiducial point final temperatures:")
    for label, (i, j) in fiducial_indices.items():
        print(f"  {label}: T = {T_2d[i, j]:.4f} keV")
    
    print("\n" + "="*70)
    print("Refinement test completed successfully!")
    print("="*70)
    
    return solver


if __name__ == "__main__":
    solver = main(output_times=[1.0, 5.0, 10.0,20.0,100.0,300.0,500.0,501.0,505.0,700.0,1000.0]
                  , use_refined_mesh=True, dt_max=10)#, 2.0, 5.0, 10.0])
