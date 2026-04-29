#!/usr/bin/env python3
"""
Crooked Pipe Test Problem - 2D Cylindrical (r-z) Geometry
MULTIGROUP NON-EQUILIBRIUM VERSION

This problem tests multigroup non-equilibrium radiation diffusion through a
complex geometry with spatially-varying material properties.

Multigroup setup: each energy group has a different opacity from a power-law
spectrum model, while material density follows the crooked-pipe thick/thin map.

Material regions:
- Optically thick regions: rho = 10 g/cm^3
- Optically thin regions:  rho = 0.01 g/cm^3

Material model (matched to 1-D Marshak power-law):
- σ(T,E) = 10.0 * rho * T^{-1/2} * E^{-3} cm^-1
- c_v = 0.05 GJ/(g*keV), so rho*c_v varies with region density

Initial condition: cold material everywhere (T = 0.01 keV)
Source: T = 300 eV (0.3 keV) at z=0 and r in (0, 0.5)
"""

import sys
import os
import time as _time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
from numba import njit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from multigroup_diffusion_solver_2d_parallel import (
    MultigroupDiffusionSolver2DParallel as MultigroupDiffusionSolver2D,
)
from multigroup_diffusion_solver_2d import (
    C_LIGHT,
    A_RAD,
    Bg_multigroup,
    flux_limiter_larsen,
)

# Add utils to path for plotting
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'utils'))
from plotfuncs import show

T_init = 0.01  # keV
RHO_THICK = 2.0  # g/cm^3
RHO_THIN = 0.01   # g/cm^3
CV_MASS = 0.05    # GJ/(g*keV)

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


def material_density(r, z):
    """Region-dependent density for thick/thin material map."""
    scalar_input = np.isscalar(r) and np.isscalar(z)
    thick = is_optically_thick(r, z)
    result = np.where(thick, RHO_THICK, RHO_THIN)
    if scalar_input:
        return float(result)
    return result


def powerlaw_opacity_at_energy(T, E, rho=1.0):
    """
    Power-law opacity at specific energy:
    sigma(T,E) = 10.0 * rho * T^{-1/2} * E^{-3}
    """
    T_safe = np.maximum(T, 1e-2)
    return np.minimum(10.0 * rho * (T_safe ** -0.5) * (E ** -3.0), 1e14)


def make_powerlaw_opacity_func(E_low, E_high):
    """Create group opacity using geometric mean over group boundaries."""
    def opacity_func(T, r, z):
        rho_local = material_density(r, z)
        sigma_low = powerlaw_opacity_at_energy(T, E_low, rho_local)
        sigma_high = powerlaw_opacity_at_energy(T, E_high, rho_local)
        return np.sqrt(sigma_low * sigma_high)
    return opacity_func


def make_powerlaw_diffusion_func(E_low, E_high):
    """Create group diffusion function D_g = 1/(3*sigma_g)."""
    sigma_func = make_powerlaw_opacity_func(E_low, E_high)

    def diffusion_func(T, r, z):
        sigma = sigma_func(T, r, z)
        return 1.0 / (3.0 * sigma)

    return diffusion_func


def rosseland_opacity(T, r, z):
    """Fallback gray opacity used only in generic helpers/plots if needed."""
    rho_local = material_density(r, z)
    return powerlaw_opacity_at_energy(T, 1.0, rho_local)


def planck_opacity(T, r, z):
    """Fallback gray Planck opacity used only in generic helpers/plots if needed."""
    return rosseland_opacity(T, r, z)


def specific_heat(T, r, z):
    """
    Region-dependent volumetric heat capacity rho*c_v.

    Uses c_v = 0.05 GJ/(g*keV) from the 1-D Marshak power-law setup.
    """
    # Check if inputs are scalar
    scalar_input = np.isscalar(T) and np.isscalar(r) and np.isscalar(z)

    rho_local = np.atleast_1d(material_density(r, z))
    result = rho_local * CV_MASS

    # Return scalar if all inputs were scalar
    if scalar_input:
        return float(result.flat[0])
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
    return 0.5, 2.0, 0.0


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
        return 0.5, 2.0, phi_source / 2
    else:
        # Vacuum elsewhere
        r_loc = r
        z_loc = 0.0
        T_avg = 0.1  # Estimate for diffusion coefficient
        sigma_R = rosseland_opacity(T_avg, r_loc, z_loc)
        D = 1.0 / (3.0 * sigma_R)
        return 0.5, 2.0, 0.0


def bc_top_open(phi, pos, t, boundary='top', geometry='cylindrical'):
    """
    Top boundary (z_max): Vacuum/open boundary
    """
    r, z = pos
    T_avg = 0.1  # Estimate for diffusion coefficient
    sigma_R = rosseland_opacity(T_avg, r, z)
    D = 1.0 / (3.0 * sigma_R)
    return 0.5, 2.0, 0.0


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_material_properties(r_centers, z_centers, r_faces=None, z_faces=None, n_groups=10):
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
    plt.savefig(f'crooked_pipe_{n_groups}g_noneq_materials.png', dpi=150, bbox_inches='tight')
    print(f"Saved: crooked_pipe_{n_groups}g_noneq_materials.png")
    plt.close()


def plot_mesh(solver, n_groups=10):
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
    plt.savefig(f'crooked_pipe_{n_groups}g_noneq_mesh.png', dpi=150, bbox_inches='tight')
    print(f"Saved: crooked_pipe_{n_groups}g_noneq_mesh.png")
    plt.close()


def plot_solution(solver, time_value, save_prefix='crooked_pipe_noneq', show_mesh=False, first_one=False):
    """Plot material temperature and radiation temperature as separate figures"""
    T_2d = solver.T.reshape(solver.nx_cells, solver.ny_cells)
    phi_2d = np.sum(solver.phi_g_stored, axis=0).reshape(solver.nx_cells, solver.ny_cells)
    Er_2d = phi_2d / C_LIGHT
    T_rad_2d = (Er_2d / A_RAD)**0.25
    
    r_centers = solver.x_centers
    z_centers = solver.y_centers
    
    R, Z = np.meshgrid(r_centers, z_centers, indexing='ij')
    first_one = True  # need colorbar every time because scale changes across time steps, so can't reuse previous one
    # PLOT 1: Material temperature
    fig1, ax1 = plt.subplots(1, 1, figsize=(8, 3))
    if first_one:
        fig1, ax1 = plt.subplots(1, 1, figsize=(8, 3*1.275))
    
    #make max color equal to current max rounded to nearest 0.01 keV for better comparison across time steps
    max_T = np.max(T_2d)
    vmax_T = np.ceil(max_T * 100) / 100.0
    im1 = ax1.pcolormesh(Z, R, T_2d, shading='auto', cmap='plasma', vmin=T_init, vmax=vmax_T)
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
    filename1 = f'{save_prefix}_material_t_{time_value:.5f}ns.png'
    if first_one:
        show(filename1, close_after=True, cbar_ax=cbar1.ax)
    else:
        show(filename1, close_after=True)
    print(f"Saved: {filename1}")
    
    # PLOT 2: Radiation temperature
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 3))
    if first_one:
        fig2, ax2 = plt.subplots(1, 1, figsize=(8, 3*1.275))
    
    #make max color equal to current max rounded to nearest 0.01 keV for better comparison across time steps
    max_T_rad = np.max(T_rad_2d)
    vmax_T_rad = np.ceil(max_T_rad * 100) / 100.0
    im2 = ax2.pcolormesh(Z, R, T_rad_2d, shading='auto', cmap='plasma', vmin=T_init, vmax=vmax_T_rad)
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
    filename2 = f'{save_prefix}_radiation_t_{time_value:.5f}ns.png'
    if first_one:
        show(filename2, close_after=True, cbar_ax=cbar2.ax)
    else:
        show(filename2, close_after=True)
    print(f"Saved: {filename2}")


def plot_fiducial_history(times, fiducial_data, n_groups=10):
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
    filename1 = f'crooked_pipe_{n_groups}g_noneq_fiducial_history_material.pdf'
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
    filename2 = f'crooked_pipe_{n_groups}g_noneq_fiducial_history_radiation.pdf'
    show(filename2, close_after=True)
    print(f"Saved: {filename2}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def _default_n_threads(n_groups):
    """Half of available CPUs, capped at n_groups."""
    cpu_count = os.cpu_count() or 1
    return min(n_groups, max(1, cpu_count // 2))


# =============================================================================
# CHECKPOINT HELPERS
# =============================================================================

def save_checkpoint(solver, step, t, times, fiducial_labels, fiducial_data,
                    output_times_saved, filename):
    """
    Save full solver state + run history to an NPZ checkpoint file.

    All data is stored as plain NumPy arrays (no pickle) so the file can be
    reloaded independently of the class definitions.
    """
    fiducial_T_mat = np.array([fiducial_data[lbl]['T_mat'] for lbl in fiducial_labels])
    fiducial_T_rad = np.array([fiducial_data[lbl]['T_rad'] for lbl in fiducial_labels])

    np.savez(
        filename,
        # Solver fields
        chkpt_T               = solver.T,
        chkpt_E_r             = solver.E_r,
        chkpt_T_old           = solver.T_old,
        chkpt_E_r_old         = solver.E_r_old,
        chkpt_phi_g_stored    = solver.phi_g_stored,
        chkpt_phi_g_fraction  = solver.phi_g_fraction,
        # Scalar run state (stored as 0-d arrays)
        chkpt_t               = np.float64(t),
        chkpt_dt              = np.float64(solver.dt),
        chkpt_step            = np.int64(step),
        # Time history & output bookkeeping
        chkpt_times           = np.array(times, dtype=np.float64),
        chkpt_output_saved    = np.array(sorted(output_times_saved), dtype=np.float64),
        # Fiducial history (shape: n_points × n_times_so_far)
        chkpt_fiduc_T_mat     = fiducial_T_mat,
        chkpt_fiduc_T_rad     = fiducial_T_rad,
    )


def load_checkpoint(filename, solver, fiducial_labels, fiducial_data):
    """
    Restore solver state and run history from an NPZ checkpoint.

    Modifies *solver* in-place and returns (step, t, times, fiducial_data,
    output_times_saved) ready to be dropped back into the time loop.
    """
    ckpt = np.load(filename)

    # Restore solver fields
    solver.T[:]                = ckpt['chkpt_T']
    solver.E_r[:]              = ckpt['chkpt_E_r']
    solver.T_old[:]            = ckpt['chkpt_T_old']
    solver.E_r_old[:]          = ckpt['chkpt_E_r_old']
    solver.phi_g_stored[:]     = ckpt['chkpt_phi_g_stored']
    solver.phi_g_fraction[:]   = ckpt['chkpt_phi_g_fraction']
    solver.t                   = float(ckpt['chkpt_t'])
    solver.dt                  = float(ckpt['chkpt_dt'])

    step              = int(ckpt['chkpt_step'])
    t                 = float(ckpt['chkpt_t'])
    times             = list(ckpt['chkpt_times'])
    output_times_saved = set(float(v) for v in ckpt['chkpt_output_saved'])

    # Restore fiducial histories
    fiduc_T_mat = ckpt['chkpt_fiduc_T_mat']   # (n_points, n_times)
    fiduc_T_rad = ckpt['chkpt_fiduc_T_rad']
    for idx, lbl in enumerate(fiducial_labels):
        fiducial_data[lbl]['T_mat'] = list(fiduc_T_mat[idx])
        fiducial_data[lbl]['T_rad'] = list(fiduc_T_rad[idx])

    return step, t, times, fiducial_data, output_times_saved


def main(
    output_times=None,
    n_groups=10,
    t_final=1000.0,
    dt_initial=1e-3,
    dt_max=10.0,
    dt_increase_factor=1.1,
    bc_t_start=0.05,
    bc_t_end=0.5,
    bc_ramp_time=20.0,
    use_refined_mesh=False,
    n_refine=10,
    n_coarse_r=60,
    n_coarse_z=210,
    refine_width=0.01,
    n_threads=None,
    restart_file=None,
    checkpoint_file=None,
    checkpoint_every=0,
):
    """Run the multigroup non-equilibrium Crooked Pipe test problem.

    Multigroup setup: each group uses a distinct power-law opacity over its
    own energy interval.
    
    Parameters:
    -----------
    use_refined_mesh : bool
        If True, use mesh with refinement at material interfaces
    n_refine : int
        Number of refined cells per coarse cell in refinement zones
    n_coarse_r : int
        Number of coarse cells in r-direction
    n_coarse_z : int
        Number of coarse cells in z-direction
    refine_width : float
        Width (in cm) around each interface to apply refinement
    """
    if output_times is None:
        output_times = [1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]

    output_times = sorted(set(float(t) for t in output_times))
    output_times_in_window = [t for t in output_times if 0.0 < t <= t_final]

    print("=" * 80)
    print("CROOKED PIPE TEST PROBLEM - MULTIGROUP NON-EQUILIBRIUM")
    print("=" * 80)
    print("2D Cylindrical (r-z) Geometry")
    print("Domain: r in [0.0, 2.0] cm, z in [0.0, 7.0] cm")
    print(f"Groups: {n_groups} (group-dependent power-law opacities)")
    print("Material properties:")
    print("  Opacity law: sigma(T,E) = 10.0 * rho * T^{-1/2} * E^{-3} cm^-1")
    print(f"  Thick region: rho = {RHO_THICK} g/cm^3, rho*c_v = {RHO_THICK * CV_MASS:.4f} GJ/(cm^3*keV)")
    print(f"  Thin region:  rho = {RHO_THIN} g/cm^3, rho*c_v = {RHO_THIN * CV_MASS:.4f} GJ/(cm^3*keV)")
    print(f"  c_v (mass-specific) = {CV_MASS} GJ/(g*keV)")
    print("Boundary conditions:")
    print("  Left (r=0): Reflecting (symmetry axis)")
    print("  Right (r_max): Vacuum/open")
    print("  Bottom (z=0): Source at r<0.5 with ramped T_bc(t), Vacuum elsewhere")
    print("  Top (z_max): Vacuum/open")
    print(f"Boundary temperature ramp: T_bc={bc_t_start} -> {bc_t_end} keV over {bc_ramp_time} ns")
    print(f"Final simulation time: {t_final} ns")
    print(f"Output times for colormaps: {output_times_in_window} ns")
    print(f"Time stepping: dt_initial={dt_initial} ns, dt_max={dt_max} ns, growth={dt_increase_factor}")
    if use_refined_mesh:
        print(f"Mesh refinement: ENABLED at interfaces (n_refine={n_refine}, width={refine_width} cm)")
    else:
        print("Mesh refinement: DISABLED (uniform mesh)")
    print("=" * 80)

    x_min, x_max = 0.0, 2.0
    y_min, y_max = 0.0, 7.0
    nx_cells, ny_cells = n_coarse_r, n_coarse_z
    energy_edges = np.logspace(1e-4, np.log10(10.0), n_groups + 1)

    def boundary_temperature(time_ns):
        """Linear ramp in boundary temperature from bc_t_start to bc_t_end."""
        if bc_ramp_time <= 0.0:
            return bc_t_end
        f = np.clip(time_ns / bc_ramp_time, 0.0, 1.0)
        return bc_t_start + (bc_t_end - bc_t_start) * f

    group_absorption_funcs = [
        make_powerlaw_opacity_func(energy_edges[g], energy_edges[g + 1])
        for g in range(n_groups)
    ]
    group_diffusion_funcs = [
        make_powerlaw_diffusion_func(energy_edges[g], energy_edges[g + 1])
        for g in range(n_groups)
    ]

    def make_bottom_source_bc(group_idx):
        def bottom_bc(phi, pos, t, boundary='bottom', geometry='cylindrical'):
            r, z = pos
            t_eval = max(float(t), 0.0)
            T_bc = boundary_temperature(t_eval)
            D = group_diffusion_funcs[group_idx](T_bc, r, 0.0)
            if r < 0.5:
                B_groups = Bg_multigroup(energy_edges, T_bc)
                chi = B_groups / np.sum(B_groups)
                F_total = (A_RAD * C_LIGHT * T_bc**4) / 2.0
                C_group = chi[group_idx] * F_total
                return 0.5, 2.0, C_group
            return 0.5, 2.0, 0.0
        return bottom_bc

    def make_open_bc(group_idx):
        def open_bc(phi, pos, t, boundary='right', geometry='cylindrical'):
            r, z = pos
            D = group_diffusion_funcs[group_idx](0.1, r, z)
            return 0.5, 2.0, 0.0
        return open_bc

    left_bcs = [bc_left_axis] * n_groups
    right_bcs = [make_open_bc(g) for g in range(n_groups)]
    bottom_bcs = [make_bottom_source_bc(g) for g in range(n_groups)]
    top_bcs = [make_open_bc(g) for g in range(n_groups)]
    boundary_funcs = {
        'left': left_bcs,
        'right': right_bcs,
        'bottom': bottom_bcs,
        'top': top_bcs,
    }

    if n_threads is None:
        n_threads = _default_n_threads(n_groups)
    print(f"  Parallel solver: {n_threads} threads ({os.cpu_count() or 1} CPUs available)")

    print("\nSetting up multigroup solver...")

    max_newton_iter = 5
    newton_tol = 1e-6
    if use_refined_mesh:
        # Generate custom mesh with refinement at material interfaces
        # Interface locations in r: 0.5, 1.0, 1.5 cm
        # Interface locations in z: 2.5, 3.0, 4.0, 4.5 cm
        print("  Generating refined mesh at material interfaces...")
        r_faces = generate_refined_z_faces(
            z_min=x_min,
            z_max=x_max,
            interface_locations=[0.5, 1.0, 1.5],
            n_refine=n_refine,
            n_coarse=n_coarse_r,
            refine_width=refine_width
        )
        z_faces = generate_refined_z_faces(
            z_min=y_min,
            z_max=y_max,
            interface_locations=[2.5, 3.0, 4.0, 4.5],
            n_refine=n_refine,
            n_coarse=n_coarse_z,
            refine_width=refine_width
        )
        
        # Print mesh info
        dr_min = np.min(np.diff(r_faces))
        dz_min = np.min(np.diff(z_faces))
        print(f"  Refined mesh: {len(r_faces)-1} × {len(z_faces)-1} cells")
        print(f"  Minimum dr = {dr_min:.5f} cm, Minimum dz = {dz_min:.5f} cm")
        
        solver = MultigroupDiffusionSolver2D(
            n_groups=n_groups,
            x_faces=r_faces,
            y_faces=z_faces,
            energy_edges=energy_edges,
            geometry='cylindrical',
            dt=dt_initial,
            diffusion_coeff_funcs=group_diffusion_funcs,
            absorption_coeff_funcs=group_absorption_funcs,
            boundary_funcs=boundary_funcs,
            flux_limiter_funcs=flux_limiter_larsen,
            specific_heat_func=specific_heat,
            material_energy_func=material_energy,
            inverse_material_energy_func=inverse_material_energy,
            max_newton_iter=max_newton_iter,
            newton_tol=newton_tol,
            theta=1.0,
            n_threads=n_threads,
        )
    else:
        # Use uniform mesh
        print(f"  Using uniform mesh: {nx_cells} × {ny_cells} cells")
        solver = MultigroupDiffusionSolver2D(
            n_groups=n_groups,
            x_min=x_min, x_max=x_max, nx_cells=nx_cells,
            y_min=y_min, y_max=y_max, ny_cells=ny_cells,
            energy_edges=energy_edges,
            geometry='cylindrical',
            dt=dt_initial,
            diffusion_coeff_funcs=group_diffusion_funcs,
            absorption_coeff_funcs=group_absorption_funcs,
            boundary_funcs=boundary_funcs,
            flux_limiter_funcs=flux_limiter_larsen,
            specific_heat_func=specific_heat,
            material_energy_func=material_energy,
            inverse_material_energy_func=inverse_material_energy,
            max_newton_iter=max_newton_iter,
            newton_tol=newton_tol,
            theta=1.0,
            n_threads=n_threads,
        )

    # ------------------------------------------------------------------ #
    # Fiducial point index map  (built before restart so it's always set)
    # ------------------------------------------------------------------ #
    fiducial_points = {
        'Point 1: r=0.0, z=0.25': (0.0, 0.25),
        'Point 2: r=0.0, z=2.75': (0.0, 2.75),
        'Point 3: r=1.25, z=3.5': (1.25, 3.5),
        'Point 4: r=0.0, z=4.25': (0.0, 4.25),
        'Point 5: r=0.0, z=6.75': (0.0, 6.75),
    }
    fiducial_labels = list(fiducial_points.keys())
    r_centers = solver.x_centers
    z_centers = solver.y_centers
    fiducial_indices = {}
    for label, (r_val, z_val) in fiducial_points.items():
        i = np.argmin(np.abs(r_centers - r_val))
        j = np.argmin(np.abs(z_centers - z_val))
        fiducial_indices[label] = (i, j)

    # Default checkpoint filename
    if checkpoint_file is None:
        mesh_tag = "refined" if use_refined_mesh else "uniform"
        checkpoint_file = (
            f'crooked_pipe_checkpoint_{n_groups}g_{mesh_tag}'
            f'_{solver.nx_cells}x{solver.ny_cells}.npz'
        )

    # ------------------------------------------------------------------ #
    # Initial condition  OR  restart from checkpoint                      #
    # ------------------------------------------------------------------ #
    if restart_file is not None:
        print(f"\nLoading restart from: {restart_file}")
        fiducial_data = {lbl: {'T_mat': [], 'T_rad': []} for lbl in fiducial_labels}
        times = []
        step, t, times, fiducial_data, output_times_saved = load_checkpoint(
            restart_file, solver, fiducial_labels, fiducial_data
        )
        print(f"  Resumed at t = {t:.6e} ns, step {step}")
        for label, (ri, zi) in fiducial_indices.items():
            print(f"  {label}: grid (r={r_centers[ri]:.3f}, z={z_centers[zi]:.3f})")
    else:
        print("\nSetting initial condition...")
        T_cold = 0.01
        E_r_cold = A_RAD * T_cold**4
        B_groups_cold = Bg_multigroup(energy_edges, T_cold)
        chi_cold = B_groups_cold / np.sum(B_groups_cold)
        solver.T[:] = T_cold
        solver.T_old[:] = T_cold
        solver.E_r[:] = E_r_cold
        solver.E_r_old[:] = E_r_cold
        solver.phi_g_fraction[:, :] = chi_cold[:, np.newaxis]
        for g in range(n_groups):
            solver.phi_g_stored[g, :] = chi_cold[g] * E_r_cold * C_LIGHT
        solver.t = 0.0

        print("\nPlotting material properties...")
        plot_material_properties(solver.x_centers, solver.y_centers,
                                 solver.x_faces, solver.y_faces, n_groups=n_groups)
        print("\nPlotting computational mesh...")
        plot_mesh(solver, n_groups=n_groups)

        for label, (ri, zi) in fiducial_indices.items():
            print(f"{label}: grid point (r={r_centers[ri]:.3f}, z={z_centers[zi]:.3f})")

        times = [0.0]
        fiducial_data = {
            lbl: {'T_mat': [T_cold], 'T_rad': [T_cold]}
            for lbl in fiducial_labels
        }
        output_times_saved = set()
        step = 0
        t = 0.0

    first_one = (len(output_times_saved) == 0)
    last_save_t = max(output_times_saved, default=0.0)

    # ------------------------------------------------------------------ #
    # Time loop                                                           #
    # ------------------------------------------------------------------ #
    n_threads_display = getattr(solver, 'n_threads', 1)
    print(f"\nTime stepping  [{n_groups} groups, {n_threads_display} thread(s)]")
    print(f"  Checkpoint file: {checkpoint_file}")
    if checkpoint_every > 0:
        print(f"  Periodic checkpoints every {checkpoint_every} steps")
    print()

    _t_run_start = _time.perf_counter()

    while t < t_final:
        step += 1
        dt_current = solver.dt

        # Snap dt to next requested output time or t_final
        hit_output_time = False
        for output_t in output_times_in_window:
            if output_t > t and t + solver.dt > output_t:
                solver.dt = output_t - t
                hit_output_time = True
                break
        if t + solver.dt > t_final:
            solver.dt = t_final - t

        t_new = t + solver.dt

        # Progress fraction towards the next save point
        remaining_saves = [
            ot for ot in output_times_in_window
            if ot not in output_times_saved and ot > t
        ]
        next_save_t = remaining_saves[0] if remaining_saves else t_final
        span = next_save_t - last_save_t
        pct = 100.0 * (t - last_save_t) / span if span > 1e-30 else 100.0

        print(
            f"Step {step:5d} | t = {t:.4e} → {t_new:.4e} ns"
            f" | {pct:5.1f}% to save @ t = {next_save_t:.4e} ns",
            flush=True,
        )
        thread_str = (
            f"{n_threads_display} threads" if n_threads_display > 1 else "serial"
        )
        print(
            f"  Solving {n_groups} groups ({thread_str}) ...",
            flush=True,
        )

        _t_step_start = _time.perf_counter()
        info = solver.step(verbose=False, gmres_tol=1e-8, gmres_maxiter=300,
                           use_preconditioner=False)
        _dt_wall = _time.perf_counter() - _t_step_start
        _elapsed  = _time.perf_counter() - _t_run_start
        t = solver.t

        n_newt  = info['newton_iterations']
        n_gmres = info.get('total_gmres_iterations', 0)
        r_E     = info['final_residuals']['r_E']
        r_T     = info['final_residuals']['r_T']
        warn    = "  *** MAX NEWTON ***" if n_newt >= solver.max_newton_iter else ""
        print(
            f"  Done  | Newton: {n_newt} | GMRES: {n_gmres}"
            f" | r_E = {r_E:.2e}, r_T = {r_T:.2e}"
            f" | step {_dt_wall:.1f}s  elapsed {_elapsed:.0f}s{warn}",
            flush=True,
        )

        # Update fiducial history first so checkpoint includes this step
        times.append(t)
        T_2d   = solver.T.reshape(solver.nx_cells, solver.ny_cells)
        Er_2d  = solver.E_r.reshape(solver.nx_cells, solver.ny_cells)
        T_rad_2d = np.maximum(Er_2d / A_RAD, 0.0)**0.25
        for label, (ri, zi) in fiducial_indices.items():
            fiducial_data[label]['T_mat'].append(T_2d[ri, zi])
            fiducial_data[label]['T_rad'].append(T_rad_2d[ri, zi])

        # Save colormap + checkpoint at output times
        for output_t in output_times_in_window:
            if output_t not in output_times_saved and abs(t - output_t) < 1e-6:
                print(
                    f"  >>> Saving colormap at t = {t:.6e} ns"
                    f" (target: {output_t} ns)",
                    flush=True,
                )
                plot_solution(
                    solver, t,
                    save_prefix=f'crooked_pipe_{n_groups}g_noneq_{mesh_tag}',
                    show_mesh=False,
                    first_one=first_one,
                )
                output_times_saved.add(output_t)
                last_save_t = t
                first_one = False
                print(
                    f"  >>> Checkpoint → {checkpoint_file}",
                    flush=True,
                )
                save_checkpoint(
                    solver, step, t, times,
                    fiducial_labels, fiducial_data,
                    output_times_saved, checkpoint_file,
                )

        # Periodic checkpoint (if requested and not already saved above)
        if checkpoint_every > 0 and step % checkpoint_every == 0:
            print(
                f"  >>> Periodic checkpoint (step {step}) → {checkpoint_file}",
                flush=True,
            )
            save_checkpoint(
                solver, step, t, times,
                fiducial_labels, fiducial_data,
                output_times_saved, checkpoint_file,
            )

        if hit_output_time:
            solver.dt = dt_current
        solver.dt = min(solver.dt * dt_increase_factor, dt_max)

    if t_final not in output_times_saved:
        print(f"  Saving colormap at final time t = {t_final:.3f} ns")
        plot_solution(solver, t_final, save_prefix=f'crooked_pipe_{n_groups}g_noneq_{mesh_tag}', show_mesh=False, first_one=first_one)

    times = np.array(times)
    for label in fiducial_data:
        fiducial_data[label]['T_mat'] = np.array(fiducial_data[label]['T_mat'])
        fiducial_data[label]['T_rad'] = np.array(fiducial_data[label]['T_rad'])

    T_2d = solver.T.reshape(solver.nx_cells, solver.ny_cells)
    phi_2d = np.sum(solver.phi_g_stored, axis=0).reshape(solver.nx_cells, solver.ny_cells)
    Er_2d = solver.E_r.reshape(solver.nx_cells, solver.ny_cells)
    T_rad_2d = np.maximum(Er_2d / A_RAD, 0.0)**0.25
    E_r_groups_3d = (solver.phi_g_stored / C_LIGHT).reshape(n_groups, solver.nx_cells, solver.ny_cells)

    npz_filename = f'crooked_pipe_{n_groups}g_noneq_solution_{mesh_tag}_{solver.nx_cells}x{solver.ny_cells}.npz'
    np.savez(
        npz_filename,
        r_centers=r_centers,
        z_centers=z_centers,
        energy_edges=energy_edges,
        T_2d=T_2d,
        T_rad_2d=T_rad_2d,
        phi_2d=phi_2d,
        Er_2d=Er_2d,
        E_r_groups_3d=E_r_groups_3d,
        times=times,
        fiducial_data=fiducial_data,
    )
    print(f"Saved: {npz_filename}")

    print("\nPlotting fiducial point temperature history...")
    plot_fiducial_history(times, fiducial_data, n_groups=n_groups)

    print("\n" + "=" * 80)
    print("SOLUTION SUMMARY")
    print("=" * 80)
    print(f"Time: {t:.3f} ns")
    print(f"Total steps: {step}")
    print(f"Material temperature range: {T_2d.min():.4f} to {T_2d.max():.4f} keV")
    print(f"Radiation temperature range: {T_rad_2d.min():.4f} to {T_rad_2d.max():.4f} keV")
    print("\nFiducial point final temperatures:")
    for label, (i, j) in fiducial_indices.items():
        print(f"  {label}:")
        print(f"    T_mat = {T_2d[i, j]:.4f} keV, T_rad = {T_rad_2d[i, j]:.4f} keV")

    print("\n" + "=" * 80)
    print("Crooked Pipe multigroup non-equilibrium test completed successfully!")
    print("=" * 80)
    return solver

#0.001,0.01,0.1,1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0
def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run multigroup non-equilibrium Crooked Pipe test problem.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n-groups",
        type=int,
        default=10,
        help="Number of energy groups",
    )
    parser.add_argument(
        "--t-final",
        type=float,
        default=50.0,
        help="Final simulation time (ns)",
    )
    parser.add_argument(
        "--dt-initial",
        type=float,
        default=1e-4,
        help="Initial time step (ns)",
    )
    parser.add_argument(
        "--dt-max",
        type=float,
        default=0.01,
        help="Maximum time step (ns)",
    )
    parser.add_argument(
        "--dt-growth",
        type=float,
        default=1.1,
        help="Time step growth factor",
    )
    parser.add_argument(
        "--output-times",
        type=str,
        default=None,
        help="Comma-separated output times (ns), e.g. '1.0,10.0,100.0,1000.0'. If not provided, uses default set.",
    )
    parser.add_argument(
        "--bc-t-start",
        type=float,
        default=0.05,
        help="Initial boundary source temperature (keV)",
    )
    parser.add_argument(
        "--bc-t-end",
        type=float,
        default=0.5,
        help="Final boundary source temperature (keV)",
    )
    parser.add_argument(
        "--bc-ramp-time",
        type=float,
        default=20.0,
        help="Time to ramp boundary temperature from start to end (ns)",
    )
    parser.add_argument(
        "--use-refined-mesh",
        action="store_true",
        help="Enable mesh refinement at material interfaces",
    )
    parser.add_argument(
        "--n-refine",
        type=int,
        default=10,
        help="Number of refined cells per coarse cell in refinement zones",
    )
    parser.add_argument(
        "--n-coarse-r",
        type=int,
        default=60,
        help="Number of coarse cells in r-direction",
    )
    parser.add_argument(
        "--n-coarse-z",
        type=int,
        default=210,
        help="Number of coarse cells in z-direction",
    )
    parser.add_argument(
        "--refine-width",
        type=float,
        default=0.05,
        help="Width (in cm) around each interface to apply refinement",
    )
    parser.add_argument(
        "--n-threads",
        type=int,
        default=None,
        help="Number of threads for parallel group solves. "
             "Default: min(n_groups, cpu_count // 2).",
    )
    parser.add_argument(
        "--restart-file",
        type=str,
        default=None,
        help="Path to a checkpoint NPZ file to restart from.",
    )
    parser.add_argument(
        "--checkpoint-file",
        type=str,
        default=None,
        help="Path for the checkpoint NPZ file written during the run. "
             "Default: auto-generated from mesh type and size.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Save a checkpoint every N steps (0 = only at output times).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # Parse output times
    if args.output_times:
        output_times = [float(t.strip()) for t in args.output_times.split(",") if t.strip()]
    else:
        output_times = None

    solver = main(
        output_times=output_times,
        n_groups=args.n_groups,
        t_final=args.t_final,
        dt_initial=args.dt_initial,
        dt_max=args.dt_max,
        dt_increase_factor=args.dt_growth,
        bc_t_start=args.bc_t_start,
        bc_t_end=args.bc_t_end,
        bc_ramp_time=args.bc_ramp_time,
        use_refined_mesh=args.use_refined_mesh,
        n_refine=args.n_refine,
        n_coarse_r=args.n_coarse_r,
        n_coarse_z=args.n_coarse_z,
        refine_width=args.refine_width,
        n_threads=args.n_threads,
        restart_file=args.restart_file,
        checkpoint_file=args.checkpoint_file,
        checkpoint_every=args.checkpoint_every,
    )
