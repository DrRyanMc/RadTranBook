#!/usr/bin/env python3
"""
Crooked Pipe Test Problem - 2D Cylindrical (r-z) Geometry
NON-EQUILIBRIUM VERSION

This problem tests non-equilibrium radiation diffusion through a complex geometry 
with spatially-varying material properties:
- Optically thick regions: σ_R = 200 cm⁻¹, c_v = 0.5 GJ/(cm³·keV)
- Optically thin regions: σ_R = 0.2 cm⁻¹, c_v = 0.0005 GJ/(cm³·keV)

Initial Condition: Cold material everywhere (T = 0.01 keV)
Source: T = 300 eV (0.3 keV) at z=0.0 and r∈(0.0, 0.5)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from functools import partial
from twoDFV import NonEquilibriumRadiationDiffusionSolver2D, C_LIGHT, A_RAD, flux_limiter_larsen

# Add utils to path for plotting
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'utils'))
from plotfuncs import show

T_init = 0.01  # keV

# =============================================================================
# MATERIAL PROPERTY FUNCTIONS
# =============================================================================

def is_optically_thick(r, z):
    """
    Determine if location (r, z) is in optically thick region
    
    Based on Crooked Pipe geometry:
    - For r ∈ [0.0, 0.5]:
        - z ∈ [0.0, 2.5]: optically thin
        - z ∈ [2.5, 7.0]: OPTICALLY THICK
    - For r ∈ [0.5, 2.0]:
        - z ∈ [0.0, 3.0]: OPTICALLY THICK
        - z ∈ [3.0, 4.5]: optically thin
        - z ∈ [4.5, 7.0]: OPTICALLY THICK
        
    Vectorized to handle scalar or array inputs
    """
    # Check if inputs are scalar
    scalar_input = np.isscalar(r) and np.isscalar(z)
    
    # Convert to arrays if needed
    r = np.atleast_1d(r)
    z = np.atleast_1d(z)
    
    # Start with all thick (default)
    result = np.ones_like(r, dtype=bool)
    
    # Lower thin region: r < 0.5 and (z < 3.0 or z > 4.0)
    lower_thin = (r < 0.5) & ((z < 3.0) | (z > 4.0))
    result[lower_thin] = False
    
    # Ascending thin region (left): r < 1.5 and (z > 2.5 and z < 3.0)
    ascending_left_thin = (r < 1.5) & (z > 2.5) & (z < 3.0)
    result[ascending_left_thin] = False
    
    # Ascending thin region (right): r < 1.5 and (z > 4.0 and z < 4.5)
    ascending_right_thin = (r < 1.5) & (z > 4.0) & (z < 4.5)
    result[ascending_right_thin] = False
    
    # Top thin region: (r >= 1.0 and r < 1.5) and (z > 2.5 and z < 4.5)
    top_thin = (r >= 1.0) & (r < 1.5) & (z > 2.5) & (z < 4.5)
    result[top_thin] = False
    
    # Return scalar if input was scalar
    if scalar_input:
        return bool(result[0])
    return result


def rosseland_opacity(T, r, z):
    """
    Spatially-varying Rosseland opacity for crooked pipe
    Temperature-independent for this problem
    """
    # Check if inputs are scalar
    scalar_input = np.isscalar(T) and np.isscalar(r) and np.isscalar(z)
    
    thick = is_optically_thick(r, z)
    result = np.where(thick, 200.0, 0.2)  # 200.0 cm⁻¹ if thick, 0.2 cm⁻¹ if thin
    
    # Return scalar if all inputs were scalar
    if scalar_input:
        return float(result)
    return result


def planck_opacity(T, r, z):
    """Planck opacity (same as Rosseland for this problem)"""
    return rosseland_opacity(T, r, z)


def specific_heat(T, r, z):
    """
    Spatially-varying specific heat for crooked pipe
    Note: This is ρc_v combined
    """
    # Check if inputs are scalar
    scalar_input = np.isscalar(T) and np.isscalar(r) and np.isscalar(z)
    
    thick = is_optically_thick(r, z)
    result = np.where(thick, 0.5, 0.0005)  # 0.5 GJ/(cm³·keV) if thick, 0.0005 if thin
    
    # Return scalar if all inputs were scalar
    if scalar_input:
        return float(result)
    return result


def material_energy(T, r, z):
    """
    Material energy density for crooked pipe
    e(T, r, z) = ρc_v(r, z) * T
    """
    cv = specific_heat(T, r, z)
    return cv * T


def inverse_material_energy(e, r, z):
    """
    Inverse: T from e
    T = e / (ρc_v)
    """
    cv = specific_heat(0.0, r, z)  # T-independent
    return e / cv


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
        Width (in cm) around each interface to apply refinement
    
    Returns:
    --------
    z_faces : ndarray
        Face positions in z with logarithmic refinement at interfaces.
        Cells are smallest at interfaces and grow logarithmically away.
    """
    # Create a uniform coarse grid
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
            print(f"  Refining cell [{cell_left:.3f}, {cell_right:.3f}] around interface at z={z_int:.3f}")
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
    """
    if n_cells <= 1:
        return [z_end]
    
    width = z_end - z_start
    
    # Growth factor for logarithmic spacing
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

def bc_left_axis(phi, pos, t, boundary='left', geometry='cylindrical'):
    """Left boundary (r=0, axis): Reflecting symmetry"""
    return 0.0, 1.0, 0.0


def bc_right_open(phi, pos, t, boundary='right', geometry='cylindrical'):
    """
    Right boundary (r_max): Vacuum/open boundary
    A*phi - B*dphi/dn = C
    """
    r, z = pos
    T_avg = 0.1  # Estimate for diffusion coefficient
    sigma_R = rosseland_opacity(T_avg, r, z)
    D = 1.0 / (3.0 * sigma_R)
    return 0.5, D, 0.0


def bc_bottom_source(phi, pos, t, boundary='bottom', geometry='cylindrical'):
    """
    Bottom boundary (z=0): Radiation source at r < 0.5
    Source: Blackbody at T = 0.3 keV
    Vacuum/reflecting elsewhere
    """
    r, z = pos
    
    if r < 0.5:
        # Source region: blackbody radiation at T_source
        T_source = 0.3  # keV
        phi_source = C_LIGHT * A_RAD * T_source**4
        # Mixed BC: combination of Dirichlet and Neumann
        # For incoming radiation, use A=0.5, B=D/(3*sigma), C=phi_source/2
        r_loc = r
        z_loc = 0.0
        T_avg = 0.1  # Estimate for diffusion coefficient
        sigma_R = rosseland_opacity(T_avg, r_loc, z_loc)
        D = 1.0 / (3.0 * sigma_R)
        return 0.5, D, phi_source / 2
    else:
        # Vacuum elsewhere
        r_loc = r
        z_loc = 0.0
        T_avg = 0.1  # Estimate for diffusion coefficient
        sigma_R = rosseland_opacity(T_avg, r_loc, z_loc)
        D = 1.0 / (3.0 * sigma_R)
        return 0.5, D, 0.0


def bc_top_open(phi, pos, t, boundary='top', geometry='cylindrical'):
    """
    Top boundary (z_max): Vacuum/open boundary
    """
    r, z = pos
    T_avg = 0.1  # Estimate for diffusion coefficient
    sigma_R = rosseland_opacity(T_avg, r, z)
    D = 1.0 / (3.0 * sigma_R)
    return 0.5, D, 0.0


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_material_properties(r_centers, z_centers, r_faces=None, z_faces=None):
    """Plot the material property distribution with optional mesh overlay"""
    
    # Create material property arrays
    nr = len(r_centers)
    nz = len(z_centers)
    opacity_field = np.zeros((nr, nz))
    cv_field = np.zeros((nr, nz))
    
    for i in range(nr):
        for j in range(nz):
            r = r_centers[i]
            z = z_centers[j]
            opacity_field[i, j] = rosseland_opacity(0.0, r, z)
            cv_field[i, j] = specific_heat(0.0, r, z)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    R, Z = np.meshgrid(r_centers, z_centers, indexing='ij')
    
    # Plot opacity
    im1 = ax1.pcolormesh(Z, R, opacity_field, shading='auto', cmap='RdYlBu_r')
    ax1.set_xlabel('z (cm)', fontsize=12)
    ax1.set_ylabel('r (cm)', fontsize=12)
    ax1.set_title('Rosseland Opacity σ_R (cm⁻¹)', fontsize=13, fontweight='bold')
    cbar1 = plt.colorbar(im1, ax=ax1, label='σ_R (cm⁻¹)')
    ax1.set_aspect('equal')
    
    # Overlay mesh lines if provided
    if r_faces is not None and z_faces is not None:
        # Sample mesh lines (not all of them to avoid clutter)
        n_lines = 10
        r_step = max(1, len(r_faces) // n_lines)
        z_step = max(1, len(z_faces) // n_lines)
        for r in r_faces[::r_step]:
            ax1.axhline(r, color='black', alpha=0.2, linewidth=0.5)
        for z in z_faces[::z_step]:
            ax1.axvline(z, color='black', alpha=0.2, linewidth=0.5)
    
    # Plot specific heat
    im2 = ax2.pcolormesh(Z, R, cv_field, shading='auto', cmap='RdYlBu_r')
    ax2.set_xlabel('z (cm)', fontsize=12)
    ax2.set_ylabel('r (cm)', fontsize=12)
    ax2.set_title('Specific Heat ρc_v (GJ/(cm³·keV))', fontsize=13, fontweight='bold')
    cbar2 = plt.colorbar(im2, ax=ax2, label='ρc_v (GJ/(cm³·keV))')
    ax2.set_aspect('equal')
    
    # Overlay mesh lines
    if r_faces is not None and z_faces is not None:
        for r in r_faces[::r_step]:
            ax2.axhline(r, color='black', alpha=0.2, linewidth=0.5)
        for z in z_faces[::z_step]:
            ax2.axvline(z, color='black', alpha=0.2, linewidth=0.5)
    
    plt.suptitle('Crooked Pipe Material Properties (Non-Equilibrium)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('crooked_pipe_noneq_materials.png', dpi=150, bbox_inches='tight')
    print("Saved: crooked_pipe_noneq_materials.png")
    plt.close()


def plot_mesh(solver):
    """Plot the computational mesh"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot all mesh lines
    for r in solver.x_faces:
        ax.plot([solver.y_faces[0], solver.y_faces[-1]], [r, r], 
                color='black', linewidth=0.3, alpha=0.5)
    for z in solver.y_faces:
        ax.plot([z, z], [solver.x_faces[0], solver.x_faces[-1]], 
                color='black', linewidth=0.3, alpha=0.5)
    
    ax.set_xlabel('z (cm)', fontsize=12)
    ax.set_ylabel('r (cm)', fontsize=12)
    ax.set_title(f'Computational Mesh ({solver.nx_cells} × {solver.ny_cells} cells)', 
                 fontsize=13, fontweight='bold')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('crooked_pipe_noneq_mesh.png', dpi=150, bbox_inches='tight')
    print("Saved: crooked_pipe_noneq_mesh.png")
    plt.close()


def plot_solution(solver, time_value, save_prefix='crooked_pipe_noneq', show_mesh=False, first_one=False):
    """Plot material temperature and radiation temperature as separate figures"""
    T_2d = solver.get_T_2d()
    phi_2d = solver.get_phi_2d()
    Er_2d = phi_2d / C_LIGHT
    T_rad_2d = (Er_2d / A_RAD)**0.25
    
    r_centers = solver.x_centers
    z_centers = solver.y_centers
    
    R, Z = np.meshgrid(r_centers, z_centers, indexing='ij')
    
    # PLOT 1: Material temperature
    fig1, ax1 = plt.subplots(1, 1, figsize=(8, 3))
    if first_one:
        fig1, ax1 = plt.subplots(1, 1, figsize=(8, 3*1.275))
    
    im1 = ax1.pcolormesh(Z, R, T_2d, shading='auto', cmap='plasma', vmin=0.0, vmax=0.3)
    ax1.set_xlabel('z (cm)', fontsize=12)
    ax1.set_ylabel('r (cm)', fontsize=12)
    #ax1.set_title(f'Material Temperature T at t = {time_value:.3f} ns', fontsize=13, fontweight='bold')
    
    if first_one:
        cbar1 = plt.colorbar(im1, ax=ax1, orientation='horizontal', location='top', pad=0.15, label='T (keV)')
    
    ax1.set_aspect('equal')
    
    # Overlay sample mesh lines if requested
    if show_mesh:
        r_faces = solver.x_faces
        z_faces = solver.y_faces
        # Sample every Nth line to avoid clutter
        n_lines = 15
        r_step = max(1, len(r_faces) // n_lines)
        z_step = max(1, len(z_faces) // n_lines)
        for r in r_faces[::r_step]:
            ax1.axhline(r, color='white', alpha=0.2, linewidth=0.3)
        for z in z_faces[::z_step]:
            ax1.axvline(z, color='white', alpha=0.2, linewidth=0.3)
    
    plt.tight_layout()
    filename1 = f'{save_prefix}_material_t_{time_value:.3f}ns.png'
    if first_one:
        show(filename1, close_after=True, cbar_ax=cbar1.ax)
    else:
        show(filename1, close_after=True)
    print(f"Saved: {filename1}")
    
    # PLOT 2: Radiation temperature
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 3))
    if first_one:
        fig2, ax2 = plt.subplots(1, 1, figsize=(8, 3*1.275))
    
    im2 = ax2.pcolormesh(Z, R, T_rad_2d, shading='auto', cmap='plasma', vmin=0.0, vmax=0.3)
    ax2.set_xlabel('z (cm)', fontsize=12)
    ax2.set_ylabel('r (cm)', fontsize=12)
    #ax2.set_title(f'Radiation Temperature T_rad at t = {time_value:.3f} ns', fontsize=13, fontweight='bold')
    
    if first_one:
        cbar2 = plt.colorbar(im2, ax=ax2, orientation='horizontal', location='top', pad=0.15, label=r'$T_\mathrm{r}$ (keV)')
    
    ax2.set_aspect('equal')
    
    # Overlay sample mesh lines if requested
    if show_mesh:
        for r in r_faces[::r_step]:
            ax2.axhline(r, color='white', alpha=0.2, linewidth=0.3)
        for z in z_faces[::z_step]:
            ax2.axvline(z, color='white', alpha=0.2, linewidth=0.3)
    
    plt.tight_layout()
    filename2 = f'{save_prefix}_radiation_t_{time_value:.3f}ns.png'
    if first_one:
        show(filename2, close_after=True, cbar_ax=cbar2.ax)
    else:
        show(filename2, close_after=True)
    print(f"Saved: {filename2}")


def plot_fiducial_history(times, fiducial_data):
    """
    Plot temperature history at fiducial points
    Shows both material temperature T and radiation temperature T_rad
    """
    # Different markers and colors for each point
    markers = ['o', 's', '^', 'd', 'v']
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    
    # Plot 1: Material temperature
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 7))
    
    for idx, (label, data) in enumerate(fiducial_data.items()):
        ax1.loglog(times, data['T_mat'],
                  marker=markers[idx % len(markers)],
                  color=colors[idx % len(colors)],
                  linewidth=2,
                  markersize=6,
                  markevery=max(1, len(times)//20),
                  label=label,
                  alpha=0.8)
    
    ax1.set_xlabel('Time (ns)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Material Temperature T (keV)', fontsize=13, fontweight='bold')
    #ax1.set_title('Material Temperature History', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='best', framealpha=0.9)
    ax1.grid(True, which='both', alpha=0.3, linestyle='--')
    ax1.grid(True, which='minor', alpha=0.15, linestyle=':')
    
    plt.tight_layout()
    filename1 = 'crooked_pipe_noneq_fiducial_history_material.pdf'
    show(filename1, close_after=True)
    print(f"Saved: {filename1}")
    
    # Plot 2: Radiation temperature
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 7))
    
    for idx, (label, data) in enumerate(fiducial_data.items()):
        ax2.loglog(times, data['T_rad'],
                  marker=markers[idx % len(markers)],
                  color=colors[idx % len(colors)],
                  linewidth=2,
                  markersize=6,
                  markevery=max(1, len(times)//20),
                  label=label,
                  alpha=0.8)
    
    ax2.set_xlabel('Time (ns)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Radiation Temperature T_rad (keV)', fontsize=13, fontweight='bold')
    #ax2.set_title('Radiation Temperature History', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10, loc='best', framealpha=0.9)
    ax2.grid(True, which='both', alpha=0.3, linestyle='--')
    ax2.grid(True, which='minor', alpha=0.15, linestyle=':')
    
    plt.tight_layout()
    filename2 = 'crooked_pipe_noneq_fiducial_history_radiation.pdf'
    show(filename2, close_after=True)
    print(f"Saved: {filename2}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main(output_times=None, use_refined_mesh=False, dt_initial=1e-3, dt_max=10.0, dt_increase_factor=1.1):
    """
    Run the non-equilibrium Crooked Pipe test problem
    
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
        output_times = [1.0, 5.0, 10.0, 100.0, 200.0, 500.0, 1000.0]
    
    print("="*80)
    print("CROOKED PIPE TEST PROBLEM - NON-EQUILIBRIUM")
    print("="*80)
    print("2D Cylindrical (r-z) Geometry")
    print("Domain: r ∈ [0.0, 2.0] cm, z ∈ [0.0, 7.0] cm")
    print()
    print("Material Properties:")
    print("  Optically thick: σ_R = 200 cm⁻¹, ρc_v = 0.5 GJ/(cm³·keV)")
    print("  Optically thin:  σ_R = 0.2 cm⁻¹,  ρc_v = 0.0005 GJ/(cm³·keV)")
    print()
    print("Initial Condition: Cold material T = 0.01 keV everywhere")
    print()
    print("Boundary Conditions:")
    print("  Left (r=0): Reflecting (symmetry axis)")
    print("  Right (r_max): Vacuum/open")
    print("  Bottom (z=0): Source at r<0.5 (T=0.3 keV), Vacuum elsewhere")
    print("  Top (z_max): Vacuum/open")
    print()
    print(f"Output times for colormaps: {output_times} ns")
    print(f"Using refined mesh at interfaces: {use_refined_mesh}")
    print(f"Time stepping: dt_initial = {dt_initial} ns, dt_max = {dt_max} ns, increase factor = {dt_increase_factor}")
    print("="*80)
    
    # Create solver
    print("\nSetting up solver...")
    
    # Setup boundary functions
    boundary_funcs = {
        'left': bc_left_axis,
        'right': bc_right_open,
        'bottom': bc_bottom_source,
        'top': bc_top_open
    }
    
    if use_refined_mesh:
        # Generate custom mesh with refinement at material interfaces
        # Interface locations: z = 2.5, 3.0, 4.0, 4.5 cm
        r_faces = generate_refined_z_faces(
            z_min=0.0,
            z_max=2.0,
            interface_locations=[0.5, 1.0, 1.5],
            n_refine=10,   # 10 cells in each refinement zone
            n_coarse=60,   # Approximate target for coarse regions
            refine_width=0.01
        )
        z_faces = generate_refined_z_faces(
            z_min=0.0,
            z_max=7.0,
            interface_locations=[2.5, 3.0, 4.0, 4.5],
            n_refine=10,   # 10 cells in each refinement zone
            n_coarse=210,  # Approximate target for coarse regions
            refine_width=0.01
        )
        
        # Print mesh info
        dr_min = np.min(np.diff(r_faces))
        dz_min = np.min(np.diff(z_faces))
        print(f"  Custom mesh generated:")
        print(f"  Minimum dr = {dr_min:.5f} cm, Minimum dz = {dz_min:.5f} cm")
        print(f"  Custom mesh: {len(r_faces)-1} × {len(z_faces)-1} cells")
        print(f"  Refinement at z = 2.5, 3.0, 4.0, 4.5 cm and r = 0.5, 1.0, 1.5 cm")
        
        solver = NonEquilibriumRadiationDiffusionSolver2D(
            x_faces=r_faces,
            y_faces=z_faces,
            geometry='cylindrical',
            dt=dt_initial,
            max_newton_iter=30,
            newton_tol=1e-6,
            rosseland_opacity_func=rosseland_opacity,
            planck_opacity_func=planck_opacity,
            specific_heat_func=specific_heat,
            material_energy_func=material_energy,
            inverse_material_energy_func=inverse_material_energy,
            boundary_funcs=boundary_funcs,
            theta=1.0,
            flux_limiter_func=partial(flux_limiter_larsen, n=2)
        )
    else:
        # Use uniform mesh (original setup)
        solver = NonEquilibriumRadiationDiffusionSolver2D(
            x_min=0.0, x_max=2.0, nx_cells=60,
            y_min=0.0, y_max=7.0, ny_cells=210,
            geometry='cylindrical',
            dt=dt_initial,
            max_newton_iter=30,
            newton_tol=1e-6,
            rosseland_opacity_func=rosseland_opacity,
            planck_opacity_func=planck_opacity,
            specific_heat_func=specific_heat,
            material_energy_func=material_energy,
            inverse_material_energy_func=inverse_material_energy,
            boundary_funcs=boundary_funcs,
            theta=1.0,
            flux_limiter_func=partial(flux_limiter_larsen, n=2)
        )
    
    # Initial condition: cold material
    print("\nSetting initial condition...")
    T_init = 0.01  # keV
    phi_init = C_LIGHT * A_RAD * T_init**4
    solver.set_initial_condition(phi_init=phi_init, T_init=T_init)
    
    # Determine mesh type for file naming
    mesh_type = "refined" if use_refined_mesh else "uniform"
    
    # Plot material properties
    print("\nPlotting material properties...")
    plot_material_properties(solver.x_centers, solver.y_centers,
                           solver.x_faces, solver.y_faces)
    
    print("\nPlotting computational mesh...")
    plot_mesh(solver)
    
    # Define fiducial points for tracking
    fiducial_points = {
        'Point 1: r=0.0, z=0.25': (0.0, 0.25),
        'Point 2: r=0.0, z=2.75': (0.0, 2.75),
        'Point 3: r=1.25, z=3.5': (1.25, 3.5),
        'Point 4: r=0.0, z=4.25': (0.0, 4.25),
        'Point 5: r=0.0, z=6.75': (0.0, 6.75)
    }
    
    # Initialize storage for fiducial point temperatures
    r_centers = solver.x_centers
    z_centers = solver.y_centers
    
    # Find indices for each fiducial point
    fiducial_indices = {}
    for label, (r_val, z_val) in fiducial_points.items():
        i = np.argmin(np.abs(r_centers - r_val))
        j = np.argmin(np.abs(z_centers - z_val))
        fiducial_indices[label] = (i, j)
        print(f"{label}: grid point (r={r_centers[i]:.3f}, z={z_centers[j]:.3f})")
    
    # Storage for time history
    times = [0.0]  # Include initial time
    fiducial_data = {label: {'T_mat': [T_init], 'T_rad': [T_init]} 
                     for label in fiducial_points.keys()}
    
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
        
        if step % 10 == 0 or step == 1:
            print(f"  Step {step}, t = {t:.4e} ns, dt = {solver.dt:.4e} ns")
        
        # Advance one time step
        solver.time_step(n_steps=1, verbose=False)
        t += solver.dt
        
        # Check if we should save a colormap at this time
        for output_t in output_times:
            if output_t not in output_times_saved and abs(t - output_t) < 1e-6:
                print(f"  Saving colormap at t = {t:.3f} ns (requested: {output_t} ns)")
                plot_solution(solver, t, save_prefix=f'crooked_pipe_noneq_{mesh_type}', 
                            show_mesh=False, first_one=first_one)
                output_times_saved.add(output_t)
                first_one = False
        
        # Save fiducial point data
        if step % save_interval == 0:
            times.append(t)
            
            T_2d = solver.get_T_2d()
            phi_2d = solver.get_phi_2d()
            Er_2d = phi_2d / C_LIGHT
            T_rad_2d = (Er_2d / A_RAD)**0.25
            
            for label, (i, j) in fiducial_indices.items():
                fiducial_data[label]['T_mat'].append(T_2d[i, j])
                fiducial_data[label]['T_rad'].append(T_rad_2d[i, j])
        
        # Restore and increase time step for next iteration (but don't exceed dt_max)
        # If we hit an output time, restore the dt we would have used, then increase it
        if hit_output_time:
            solver.dt = dt_current
        solver.dt = min(solver.dt * dt_increase_factor, dt_max)
    
    # Plot final time if not already done
    if t_final not in output_times_saved:
        print(f"  Saving colormap at final time t = {t_final:.3f} ns")
        plot_solution(solver, t_final, save_prefix=f'crooked_pipe_noneq_{mesh_type}',
                    show_mesh=False, first_one=first_one)
    
    # Convert to numpy arrays
    times = np.array(times)
    for label in fiducial_data.keys():
        fiducial_data[label]['T_mat'] = np.array(fiducial_data[label]['T_mat'])
        fiducial_data[label]['T_rad'] = np.array(fiducial_data[label]['T_rad'])
    
    # Store solution and mesh info in npz file
    T_2d = solver.get_T_2d()
    phi_2d = solver.get_phi_2d()
    Er_2d = phi_2d / C_LIGHT
    T_rad_2d = (Er_2d / A_RAD)**0.25
    
    npz_filename = f'crooked_pipe_noneq_solution_{mesh_type}_{solver.nx_cells}x{solver.ny_cells}.npz'
    np.savez(npz_filename,
             r_centers=r_centers,
             z_centers=z_centers,
             T_2d=T_2d,
             T_rad_2d=T_rad_2d,
             phi_2d=phi_2d,
             Er_2d=Er_2d,
             times=times,
             fiducial_data=fiducial_data)
    print(f"Saved: {npz_filename}")
    
    # Plot fiducial point history
    print("\nPlotting fiducial point temperature history...")
    plot_fiducial_history(times, fiducial_data)
    
    # Print some diagnostics
    print("\n" + "="*80)
    print("SOLUTION SUMMARY")
    print("="*80)
    print(f"Time: {t:.3f} ns")
    print(f"Total steps: {step}")
    print(f"Material temperature range: {T_2d.min():.4f} to {T_2d.max():.4f} keV")
    print(f"Radiation temperature range: {T_rad_2d.min():.4f} to {T_rad_2d.max():.4f} keV")
    print(f"Max material temperature location: r={r_centers[np.unravel_index(T_2d.argmax(), T_2d.shape)[0]]:.3f} cm, "
          f"z={z_centers[np.unravel_index(T_2d.argmax(), T_2d.shape)[1]]:.3f} cm")
    
    # Print fiducial point final temperatures
    print("\nFiducial point final temperatures:")
    for label, (i, j) in fiducial_indices.items():
        T_mat = T_2d[i, j]
        T_rad = T_rad_2d[i, j]
        print(f"  {label}:")
        print(f"    T_mat = {T_mat:.4f} keV, T_rad = {T_rad:.4f} keV")
    
    print("\n" + "="*80)
    print("Crooked Pipe non-equilibrium test completed successfully!")
    print("="*80)
    
    return solver

#0.001,0.01,0.1,1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0
if __name__ == "__main__":
    solver = main(output_times=[1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 
                                200.0, 500.0, 1000.0], 
                  use_refined_mesh=False,
                  dt_initial=1e-3,
                  dt_max=10.0,
                  dt_increase_factor=1.1)
