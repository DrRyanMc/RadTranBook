#!/usr/bin/env python3
"""
Mesh Refinement Test Problem - 2D Cartesian (x-z) Geometry
NON-EQUILIBRIUM VERSION

This problem tests non-equilibrium radiation diffusion through a complex geometry 
with spatially-varying material properties:
- Optically thick regions: σ_R = 200 cm⁻¹, c_v = 0.5 GJ/(cm³·keV)
- Optically thin regions: σ_R = 0.2 cm⁻¹, c_v = 0.0005 GJ/(cm³·keV)

Initial condition: Cold material everywhere (T = 0.01 keV)
Radiation sources: Boundaries at z=0 and z=z_max with T_source = 0.3 keV
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from functools import partial
from twoDFV import NonEquilibriumRadiationDiffusionSolver2D, C_LIGHT, A_RAD, RHO, flux_limiter_larsen

# Add utils to path for plotting
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'utils'))
from plotfuncs import show

# Add utils to path for plotting
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'utils'))
from plotfuncs import show
# =============================================================================
# MATERIAL PROPERTY FUNCTIONS
# =============================================================================

def is_optically_thick(x, z):
    """
    Determine if location (x, z) is in optically thick region
    
    There are two optically thin channels
    Vectorized to handle scalar or array inputs
    """
    # Check if inputs are scalar
    scalar_input = np.isscalar(x) and np.isscalar(z)
    
    # Convert to arrays if needed
    x = np.atleast_1d(x)
    z = np.atleast_1d(z)
    
    # Start with all thick
    result = np.ones_like(x, dtype=bool)
    
    # Lower thin region: (x in [1.0, 2.0]) and (z < 2.0)
    lower_thin = (x >= 1.0) & (x <= 2.0) & (z < 2.0)
    result[lower_thin] = False
    
    # Upper thin region: (x in [3.0, 4.0]) and (z > 3.0)
    upper_thin = (x >= 3.0) & (x <= 4.0) & (z > 3.0)
    result[upper_thin] = False
    
    # Return scalar if input was scalar
    if scalar_input:
        return bool(result[0])
    return result


def rosseland_opacity(T, x, z):
    """Spatially-varying Rosseland opacity (temperature-independent for this problem)"""
    # Check if inputs are scalar
    scalar_input = np.isscalar(T) and np.isscalar(x) and np.isscalar(z)
    
    thick = is_optically_thick(x, z)
    result = np.where(thick, 200.0, 0.2)  # 200.0 cm⁻¹ if thick, 0.2 cm⁻¹ if thin
    
    # Return scalar if all inputs were scalar
    if scalar_input:
        return float(result)
    return result


def planck_opacity(T, x, z):
    """Planck opacity (same as Rosseland for this problem)"""
    return rosseland_opacity(T, x, z)


def specific_heat(T, x, z):
    """Spatially-varying specific heat (ρc_v combined)"""
    # Check if inputs are scalar
    scalar_input = np.isscalar(T) and np.isscalar(x) and np.isscalar(z)
    
    thick = is_optically_thick(x, z)
    result = np.where(thick, 0.5, 0.05)  # 0.5 GJ/(cm³·keV) if thick, 0.05 if thin
    
    # Return scalar if all inputs were scalar
    if scalar_input:
        return float(result)
    return result


def material_energy(T, x, z):
    """Material energy density: e = ρc_v * T"""
    cv = specific_heat(T, x, z)
    return cv * T


def inverse_material_energy(e, x, z):
    """Inverse: T from e"""
    cv = specific_heat(0.0, x, z)  # T-independent
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
        Z-coordinates where material interfaces occur
    n_refine : int
        Number of refined cells to subdivide each coarse cell in refinement zones
    n_coarse : int
        Number of coarse cells across entire domain
    refine_width : float
        Width (in cm) around each interface to apply refinement
    
    Returns:
    --------
    z_faces : ndarray
        Face positions in z with logarithmic refinement at interfaces
    """
    # Create a uniform coarse grid
    dz_coarse = (z_max - z_min) / n_coarse
    z_coarse_faces = np.linspace(z_min, z_max, n_coarse + 1)
    
    # Sort interfaces
    interfaces = sorted(interface_locations)
    
    # Mark which coarse cells should be refined
    refine_info = {}  # i -> nearest interface z
    
    for z_int in interfaces:
        for i in range(n_coarse):
            cell_left = z_coarse_faces[i]
            cell_right = z_coarse_faces[i + 1]
            
            # Refine if cell overlaps with refinement zone
            if (cell_left <= z_int + refine_width) and (cell_right >= z_int - refine_width):
                if i not in refine_info:
                    refine_info[i] = z_int
                else:
                    # Use nearest interface
                    if abs(z_int - (cell_left + cell_right)/2) < abs(refine_info[i] - (cell_left + cell_right)/2):
                        refine_info[i] = z_int
    
    # Build final face list
    z_faces_list = [z_min]
    
    for i in range(n_coarse):
        cell_left = z_coarse_faces[i]
        cell_right = z_coarse_faces[i + 1]
        
        if i in refine_info:
            z_int = refine_info[i]
            z_refined = create_log_spacing_around_interface(
                cell_left, cell_right, z_int, n_refine
            )
            z_faces_list.extend(z_refined)
        else:
            z_faces_list.append(cell_right)
    
    return np.array(z_faces_list)


def create_log_spacing_around_interface(z_left, z_right, z_int, n_cells):
    """Create logarithmically-spaced cells around interface"""
    if z_int <= z_left:
        return create_log_spacing_one_sided(z_left, z_right, n_cells, from_left=True)
    elif z_int >= z_right:
        return create_log_spacing_one_sided(z_left, z_right, n_cells, from_left=False)
    else:
        # Interface inside cell, split into two regions
        n_left = max(1, int(n_cells * (z_int - z_left) / (z_right - z_left)))
        n_right = n_cells - n_left
        
        faces_left = create_log_spacing_one_sided(z_left, z_int, n_left, from_left=False)
        faces_right = create_log_spacing_one_sided(z_int, z_right, n_right, from_left=True)
        
        return faces_left + faces_right


def create_log_spacing_one_sided(z_start, z_end, n_cells, from_left=True):
    """Create logarithmically-spaced cells in one direction"""
    if n_cells <= 1:
        return [z_end]
    
    width = z_end - z_start
    
    # Growth factor for logarithmic spacing
    max_ratio = 5.0
    r = max_ratio ** (1.0 / (n_cells - 1)) if n_cells > 1 else 1.0
    
    # Calculate cell widths
    if abs(r - 1.0) < 1e-10:
        w0 = width / n_cells
        cell_widths = [w0] * n_cells
    else:
        w0 = width * (r - 1.0) / (r**n_cells - 1.0)
        cell_widths = [w0 * r**i for i in range(n_cells)]
    
    if not from_left:
        cell_widths = cell_widths[::-1]
    
    # Build face positions
    faces = []
    current_pos = z_start
    for w in cell_widths:
        current_pos += w
        faces.append(current_pos)
    
    faces[-1] = z_end  # Fix floating point errors
    
    return faces


# =============================================================================
# BOUNDARY CONDITION FUNCTIONS
# =============================================================================

def bc_left_reflecting(phi, pos, t, boundary='left', geometry='cartesian'):
    """Left boundary (x=x_min): Reflecting"""
    return 0.0, 1.0, 0.0


def bc_right_reflecting(phi, pos, t, boundary='right', geometry='cartesian'):
    """Right boundary (x=x_max): Reflecting"""
    return 0.0, 1.0, 0.0


def bc_bottom_source(phi, pos, t, boundary='bottom', geometry='cartesian'):
    """
    Bottom boundary (z=0): Radiation source at x ∈ [1.0, 2.0]
    Source: Blackbody at T = 0.3 keV
    Aligned with lower thin region
    """
    x, z = pos
    
    if t < 500.0: # and (x >= 1.0 and x <= 2.0):
        # Source region: blackbody radiation at T_source
        T_source = 0.3  # keV
        phi_source = C_LIGHT * A_RAD * T_source**4
        # Mixed BC: combination of Dirichlet and Neumann
        # For incoming radiation, use A=0.5, B=D/(3*sigma), C=phi_source/2
        x_loc = x
        z_loc = 0.0
        T_avg = 0.1  # Estimate for diffusion coefficient
        sigma_R = rosseland_opacity(T_avg, x_loc, z_loc)
        D = 1.0 / (3.0 * sigma_R)
        return 0.5, D, phi_source / 2
    else:
        # Vacuum elsewhere
        x_loc = x
        z_loc = 0.0
        T_avg = 0.1  # Estimate for diffusion coefficient
        sigma_R = rosseland_opacity(T_avg, x_loc, z_loc)
        D = 1.0 / (3.0 * sigma_R)
        return 0.5, D, 0.0


def bc_top_source(phi, pos, t, boundary='top', geometry='cartesian'):
    """
    Top boundary (z=z_max): Radiation source at x ∈ [3.0, 4.0]
    Source: Blackbody at T = 0.3 keV
    Aligned with upper thin region
    """
    x, z = pos
    
    if t < 500.0: # and (x >= 3.0 and x <= 4.0):
        # Source region
        T_source = 0.3  # keV
        phi_source = C_LIGHT * A_RAD * T_source**4
        x_loc = x
        z_loc = 5.0  # Approximate z_max
        T_avg = 0.1
        sigma_R = rosseland_opacity(T_avg, x_loc, z_loc)
        D = 1.0 / (3.0 * sigma_R)
        return 0.5, D, phi_source / 2
    else:
        # Vacuum elsewhere
        x_loc = x
        z_loc = 5.0  # Approximate z_max
        T_avg = 0.1
        sigma_R = rosseland_opacity(T_avg, x_loc, z_loc)
        D = 1.0 / (3.0 * sigma_R)
        return 0.5, D, 0.0


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_material_properties(x_centers, z_centers, x_faces=None, z_faces=None):
    """Plot the material property distribution with optional mesh overlay"""
    
    # Create material property arrays
    nx = len(x_centers)
    nz = len(z_centers)
    opacity_field = np.zeros((nx, nz))
    cv_field = np.zeros((nx, nz))
    
    for i in range(nx):
        for j in range(nz):
            x = x_centers[i]
            z = z_centers[j]
            opacity_field[i, j] = rosseland_opacity(0.0, x, z)
            cv_field[i, j] = specific_heat(0.0, x, z)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    X, Z = np.meshgrid(x_centers, z_centers, indexing='ij')
    
    # Plot opacity
    im1 = ax1.pcolormesh(Z, X, opacity_field, shading='auto', cmap='RdYlBu_r')
    ax1.set_xlabel('z (cm)', fontsize=12)
    ax1.set_ylabel('x (cm)', fontsize=12)
    #ax1.set_title('Rosseland Opacity σ_R (cm⁻¹)', fontsize=13, fontweight='bold')
    cbar1 = plt.colorbar(im1, ax=ax1, label='σ_R (cm⁻¹)')
    ax1.set_aspect('equal')
    
    # Overlay mesh lines if provided
    if x_faces is not None and z_faces is not None:
        # Sample mesh lines (not all of them to avoid clutter)
        n_lines = 10
        x_step = max(1, len(x_faces) // n_lines)
        z_step = max(1, len(z_faces) // n_lines)
        for x in x_faces[::x_step]:
            ax1.axhline(x, color='black', alpha=0.2, linewidth=0.5)
        for z in z_faces[::z_step]:
            ax1.axvline(z, color='black', alpha=0.2, linewidth=0.5)
    
    # Plot specific heat
    im2 = ax2.pcolormesh(Z, X, cv_field, shading='auto', cmap='RdYlBu_r')
    ax2.set_xlabel('z (cm)', fontsize=12)
    ax2.set_ylabel('x (cm)', fontsize=12)
    #ax2.set_title('Specific Heat ρc_v (GJ/(cm³·keV))', fontsize=13, fontweight='bold')
    cbar2 = plt.colorbar(im2, ax=ax2, label='ρc_v (GJ/(cm³·keV))')
    ax2.set_aspect('equal')
    
    # Overlay mesh lines
    if x_faces is not None and z_faces is not None:
        for x in x_faces[::x_step]:
            ax2.axhline(x, color='black', alpha=0.2, linewidth=0.5)
        for z in z_faces[::z_step]:
            ax2.axvline(z, color='black', alpha=0.2, linewidth=0.5)
    
    #plt.suptitle('Material Properties', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('refined_zoning_noneq_materials.png', dpi=150, bbox_inches='tight')
    print("Saved: refined_zoning_noneq_materials.png")
    plt.close()


def plot_solution(solver, time_value, save_prefix='refined_zoning_noneq', show_mesh=False, first_one=False):
    """Plot temperature and radiation temperature with optional mesh overlay"""
    T_2d = solver.get_T_2d()
    phi_2d = solver.get_phi_2d()
    Er_2d = phi_2d / C_LIGHT
    T_rad_2d = (Er_2d / A_RAD)**0.25
    
    x_centers = solver.x_centers
    z_centers = solver.y_centers
    
    X, Z = np.meshgrid(x_centers, z_centers, indexing='ij')
    
    # PLOT 1: Material temperature
    fig1, ax1 = plt.subplots(1, 1, figsize=(6, 6))
    if first_one:
        fig1, ax1 = plt.subplots(1, 1, figsize=(6, 6*1.275))
    
    im1 = ax1.pcolormesh(Z, X, T_2d, shading='auto', cmap='plasma', vmin=0.0, vmax=0.3)
    ax1.set_xlabel('z (cm)', fontsize=15)
    ax1.set_ylabel('x (cm)', fontsize=15)
    #ax1.set_title(f'Material Temperature at t = {time_value:.2f} ns', fontsize=13, fontweight='bold')
    
    if first_one:
        cbar1 = plt.colorbar(im1, ax=ax1, orientation='horizontal', location='top', pad=0.15, label='T (keV)')
    
    ax1.set_aspect('equal')
    
    # Overlay sample mesh lines if requested
    if show_mesh:
        x_faces = solver.x_faces
        z_faces = solver.y_faces
        # Sample every Nth line to avoid clutter
        n_lines = 15
        x_step = max(1, len(x_faces) // n_lines)
        z_step = max(1, len(z_faces) // n_lines)
        for x in x_faces[::x_step]:
            ax1.axhline(x, color='white', alpha=0.2, linewidth=0.3)
        for z in z_faces[::z_step]:
            ax1.axvline(z, color='white', alpha=0.2, linewidth=0.3)
    
    plt.tight_layout()
    filename1 = f'{save_prefix}_material_t_{time_value:.2f}ns.png'
    if first_one:
        show(filename1, close_after=True, cbar_ax=cbar1.ax)
    else:
        show(filename1, close_after=True)
    print(f"Saved: {filename1}")
    
    # PLOT 2: Radiation temperature
    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 6))
    if first_one:
        fig2, ax2 = plt.subplots(1, 1, figsize=(6, 6*1.275))
    
    im2 = ax2.pcolormesh(Z, X, T_rad_2d, shading='auto', cmap='plasma', vmin=0.0, vmax=0.3)
    ax2.set_xlabel('z (cm)', fontsize=15)
    ax2.set_ylabel('x (cm)')
    #ax2.set_title(f'Radiation Temperature at t = {time_value:.2f} ns', fontsize=13, fontweight='bold')
    
    if first_one:
        cbar2 = plt.colorbar(im2, ax=ax2, orientation='horizontal', location='top', pad=0.15, label=r'$T_\mathrm{r}$ (keV)')
    
    ax2.set_aspect('equal')
    
    # Mark thin region boundaries
    # ax2.axhline(1.0, xmin=0.0, xmax=0.4, color='cyan', linestyle='--', linewidth=1.5, alpha=0.5, label='Thin region boundaries')
    # ax2.axhline(2.0, xmin=0.0, xmax=0.4, color='cyan', linestyle='--', linewidth=1.5, alpha=0.5)
    # ax2.axhline(3.0, xmin=0.6, xmax=1.0, color='cyan', linestyle='--', linewidth=1.5, alpha=0.5)
    # ax2.axhline(4.0, xmin=0.6, xmax=1.0, color='cyan', linestyle='--', linewidth=1.5, alpha=0.5)
    
    # Mark source regions at boundaries
    # ax2.plot([0, 0], [1.0, 2.0], 'r-', linewidth=4, alpha=0.8, label='Radiation sources')
    # ax2.plot([5.0, 5.0], [3.0, 4.0], 'r-', linewidth=4, alpha=0.8)
    
    # Overlay sample mesh lines
    if show_mesh:
        for x in x_faces[::x_step]:
            ax2.axhline(x, color='white', alpha=0.2, linewidth=0.3)
        for z in z_faces[::z_step]:
            ax2.axvline(z, color='white', alpha=0.2, linewidth=0.3)
    
    #ax2.legend(fontsize=9, loc='lower right')
    
    plt.tight_layout()
    filename2 = f'{save_prefix}_radiation_t_{time_value:.2f}ns.png'
    if first_one:
        show(filename2, close_after=True, cbar_ax=cbar2.ax)
    else:
        show(filename2, close_after=True)
    print(f"Saved: {filename2}")


def plot_mesh(solver):
    """Plot the computational mesh"""
    x_faces = solver.x_faces
    z_faces = solver.y_faces
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Full mesh view
    for x in x_faces:
        ax1.axhline(x, color='black', alpha=0.3, linewidth=0.5)
    for z in z_faces:
        ax1.axvline(z, color='black', alpha=0.3, linewidth=0.5)
    
    ax1.set_xlabel('z (cm)', fontsize=12)
    ax1.set_ylabel('x (cm)', fontsize=12)
    #ax1.set_title(f'Computational Mesh ({len(x_faces)-1} × {len(z_faces)-1} cells)', 
    #              fontsize=13, fontweight='bold')
    ax1.set_xlim(z_faces[0], z_faces[-1])
    ax1.set_ylim(x_faces[0], x_faces[-1])
    ax1.set_aspect('equal')
    ax1.grid(False)
    
    # Zoomed view of interface regions
    for x in x_faces:
        ax2.axhline(x, color='black', alpha=0.5, linewidth=0.5)
    for z in z_faces:
        ax2.axvline(z, color='black', alpha=0.5, linewidth=0.5)
    
    # Highlight material interface regions
    ax2.axhline(1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Material interfaces')
    ax2.axhline(2.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.axhline(3.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.axhline(4.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.axvline(2.0, color='blue', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.axvline(3.0, color='blue', linestyle='--', linewidth=1.5, alpha=0.7)
    
    ax2.set_xlabel('z (cm)', fontsize=15)
    ax2.set_ylabel('x (cm)', fontsize=15)
    #ax2.set_title('Mesh at Material Interfaces (zoom)', fontsize=13, fontweight='bold')
    ax2.set_xlim(1.5, 3.5)
    ax2.set_ylim(0.5, 4.5)
    ax2.set_aspect('equal')
    ax2.legend(fontsize=9)
    ax2.grid(False)
    
    # Add text annotations
    dx_min = np.min(np.diff(x_faces))
    dx_max = np.max(np.diff(x_faces))
    dz_min = np.min(np.diff(z_faces))
    dz_max = np.max(np.diff(z_faces))
    
    info_text = f'Δx: [{dx_min:.5f}, {dx_max:.5f}] cm\nΔz: [{dz_min:.5f}, {dz_max:.5f}] cm'
    ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    #plt.suptitle('Computational Mesh Visualization', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('refined_zoning_noneq_mesh.png', dpi=150, bbox_inches='tight')
    print("Saved: refined_zoning_noneq_mesh.png")
    plt.close()


def plot_fiducial_history(times, fiducial_data):
    """Plot temperature history at fiducial points"""
    markers = ['o', 's', '^', 'd']
    colors = ['blue', 'red', 'green', 'purple']
    
    # Plot 1: Material temperature
    fig1, ax1 = plt.subplots(1, 1, figsize=(8, 5))
    
    for idx, (label, data) in enumerate(fiducial_data.items()):
        T_mat = data['T_mat']
        
        ax1.loglog(times, T_mat,
                  marker=markers[idx % len(markers)],
                  color=colors[idx % len(colors)],
                  linewidth=2, markersize=6,
                  markevery=max(1, len(times)//20),
                  label=label, alpha=0.8)
    
    ax1.set_xlabel('Time (ns)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Material Temperature (keV)', fontsize=13, fontweight='bold')
    #ax1.set_title('Material Temperature History', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='best', framealpha=0.9)
    ax1.grid(True, which='both', alpha=0.3, linestyle='--')
    ax1.grid(True, which='minor', alpha=0.15, linestyle=':')
    
    plt.tight_layout()
    show('refined_zoning_noneq_history_material.pdf', close_after=True)
    print("Saved: refined_zoning_noneq_history_material.pdf")
    
    # Plot 2: Radiation temperature
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 5))
    
    for idx, (label, data) in enumerate(fiducial_data.items()):
        T_rad = data['T_rad']
        
        ax2.loglog(times, T_rad,
                  marker=markers[idx % len(markers)],
                  color=colors[idx % len(colors)],
                  linewidth=2, markersize=6,
                  markevery=max(1, len(times)//20),
                  label=label, alpha=0.8)
    
    ax2.set_xlabel('Time (ns)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Radiation Temperature (keV)', fontsize=13, fontweight='bold')
    #ax2.set_title('Radiation Temperature History', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10, loc='best', framealpha=0.9)
    ax2.grid(True, which='both', alpha=0.3, linestyle='--')
    ax2.grid(True, which='minor', alpha=0.15, linestyle=':')
    
    plt.tight_layout()
    show('refined_zoning_noneq_history_radiation.pdf', close_after=True)
    print("Saved: refined_zoning_noneq_history_radiation.pdf")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main(output_times=None, use_refined_mesh=True, dt_initial=1e-3, dt_max=1.0):
    """
    Run the non-equilibrium refined zoning test problem
    
    Parameters:
    -----------
    output_times : list or None
        List of times (ns) at which to save solution plots
    use_refined_mesh : bool
        If True, use mesh with refinement at material interfaces
    dt_initial : float
        Initial time step (ns)
    dt_max : float
        Maximum time step (ns)
    """
    if output_times is None:
        output_times = [1.] #, 10.0, 100.0, 500.0]
    
    print("="*80)
    print("NON-EQUILIBRIUM REFINED ZONING TEST")
    print("="*80)
    print("2D Cartesian (x-z) Geometry")
    print("Domain: x ∈ [0.0, 5.0] cm, z ∈ [0.0, 5.0] cm")
    print()
    print("Material Properties:")
    print("  Optically thick: σ_R = 200 cm⁻¹, ρc_v = 0.5 GJ/(cm³·keV)")
    print("  Optically thin:  σ_R = 0.2 cm⁻¹,  ρc_v = 0.0005 GJ/(cm³·keV)")
    print()
    print("Initial Condition: Cold material T = 0.01 keV everywhere")
    print()
    print("Boundary Conditions:")
    print("  Left/Right (x): Reflecting")
    print("  Bottom (z=0): Radiation source at x ∈ (1.0, 2.0), T_source = 0.3 keV")
    print("  Top (z_max): Radiation source at x ∈ (3.0, 4.0), T_source = 0.3 keV")
    print()
    print(f"Output times: {output_times} ns")
    print(f"Using refined mesh: {use_refined_mesh}")
    print("="*80)
    
    # Create solver with custom mesh
    print("\nSetting up solver...")
    
    # Setup boundary functions
    boundary_funcs = {
        'left': bc_left_reflecting,
        'right': bc_right_reflecting,
        'bottom': bc_bottom_source,
        'top': bc_top_source
    }
    
    if use_refined_mesh:
        # Generate refined mesh at material interfaces
        x_faces = generate_refined_z_faces(
            z_min=0.0,
            z_max=5.0,
            interface_locations=[3.0, 4.0],
            n_refine=10,
            n_coarse=100
        )
        z_faces = generate_refined_z_faces(
            z_min=0.0,
            z_max=5.0,
            interface_locations=[3.0],
            n_refine=10,
            n_coarse=100
        )
        
        print(f"  Refined mesh: {len(x_faces)-1} × {len(z_faces)-1} cells")
        dx_min = np.min(np.diff(x_faces))
        dz_min = np.min(np.diff(z_faces))
        print(f"  Minimum dx = {dx_min:.5f} cm, Minimum dz = {dz_min:.5f} cm")
        
        # Create solver with custom faces
        solver = NonEquilibriumRadiationDiffusionSolver2D(
            x_faces=x_faces,
            y_faces=z_faces,
            geometry='cartesian',
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
        # Use uniform mesh
        solver = NonEquilibriumRadiationDiffusionSolver2D(
            x_min=0.0, x_max=5.0, nx_cells=100,
            y_min=0.0, y_max=5.0, ny_cells=100,
            geometry='cartesian',
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
    
    # Plot material properties and mesh
    print("\nPlotting material properties...")
    plot_material_properties(solver.x_centers, solver.y_centers, 
                           solver.x_faces, solver.y_faces)
    
    print("\nPlotting computational mesh...")
    plot_mesh(solver)
    
    # Define fiducial points
    fiducial_points = {
        'Point1: x=1.5, z=1.95': (1.5, 1.95),
        'Point2: x=1.5, z=2.05': (1.5, 2.05),
        'Point3: x=3.5, z=3.05': (3.5, 3.05),
        'Point4: x=3.5, z=2.95': (3.5, 2.95),
    }
    
    # Find indices
    fiducial_indices = {}
    for label, (x_val, z_val) in fiducial_points.items():
        i = np.argmin(np.abs(solver.x_centers - x_val))
        j = np.argmin(np.abs(solver.y_centers - z_val))
        fiducial_indices[label] = (i, j)
        print(f"{label}: grid point (x={solver.x_centers[i]:.3f}, z={solver.y_centers[j]:.3f})")
    
    # Storage for history
    times = [0.0]
    fiducial_data = {label: {'T_mat': [T_init], 'T_rad': [T_init]} 
                     for label in fiducial_points.keys()}
    
    output_times_saved = set()
    
    # Time evolution
    print("\nTime stepping...")
    t_final = max(output_times)
    t = 0.0
    step = 0
    first_one = True
    
    while t < t_final:
        step += 1
        
        # Adjust dt to hit output times
        for output_t in sorted(output_times):
            if output_t > t and t + solver.dt > output_t:
                solver.dt = output_t - t
                break
        
        if t + solver.dt > t_final:
            solver.dt = t_final - t
        
        # Advance
        solver.time_step(n_steps=1, verbose=False)
        t += solver.dt
        
        if step % 10 == 0 or step == 1:
            print(f"  Step {step}, t = {t:.4e} ns, dt = {solver.dt:.4e} ns")
        
        # Save solution plots at output times
        for output_t in output_times:
            if output_t not in output_times_saved and abs(t - output_t) < 1e-6:
                print(f"  Saving solution at t = {t:.2f} ns")
                plot_solution(solver, t, first_one=first_one)
                output_times_saved.add(output_t)
                first_one = False
        
        # Save fiducial data
        if step % 1 == 0:
            times.append(t)
            T_2d = solver.get_T_2d()
            phi_2d = solver.get_phi_2d()
            Er_2d = phi_2d / C_LIGHT
            T_rad_2d = (Er_2d / A_RAD)**0.25
            
            for label, (i, j) in fiducial_indices.items():
                fiducial_data[label]['T_mat'].append(T_2d[i, j])
                fiducial_data[label]['T_rad'].append(T_rad_2d[i, j])
        
        # Increase dt (adaptive)
        solver.dt = min(solver.dt * 1.1, dt_max)
    
    # Convert to arrays
    times = np.array(times)
    for label in fiducial_data.keys():
        fiducial_data[label]['T_mat'] = np.array(fiducial_data[label]['T_mat'])
        fiducial_data[label]['T_rad'] = np.array(fiducial_data[label]['T_rad'])
    
    # Plot fiducial history
    print("\nPlotting fiducial point history...")
    plot_fiducial_history(times, fiducial_data)
    
    # Final summary
    T_2d = solver.get_T_2d()
    phi_2d = solver.get_phi_2d()
    
    print("\n" + "="*80)
    print("SOLUTION SUMMARY")
    print("="*80)
    print(f"Final time: {t:.2f} ns")
    print(f"Total steps: {step}")
    print(f"Material temperature range: {T_2d.min():.4f} to {T_2d.max():.4f} keV")
    print(f"Radiation temperature range: {(phi_2d.min()/C_LIGHT/A_RAD)**0.25:.4f} to {(phi_2d.max()/C_LIGHT/A_RAD)**0.25:.4f} keV")
    
    print("\nFiducial point final temperatures:")
    for label, (i, j) in fiducial_indices.items():
        T_mat = T_2d[i, j]
        T_rad = (phi_2d[i, j] / C_LIGHT / A_RAD)**0.25
        print(f"  {label}:")
        print(f"    T_mat = {T_mat:.4f} keV, T_rad = {T_rad:.4f} keV")
    
    print("\n" + "="*80)
    print("Non-equilibrium refined zoning test completed!")
    print("="*80)
    
    return solver


if __name__ == "__main__":
    #solver = main(output_times=[.001,.01,0.1,1.0, 10.0, 100.0, 500.0], dt_max=10.0)
    solver = main(output_times=[1.0,10.,100.0,501.0,700.,1000.0], dt_max=10.0)
    #solver = main(output_times=[0.001,.01,0.1,1.0,10.], dt_max=10.0)
