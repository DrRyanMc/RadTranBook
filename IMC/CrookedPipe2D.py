#!/usr/bin/env python3
"""
Crooked Pipe Test Problem - 2D Cylindrical (r-z) Geometry
IMC VERSION

This problem tests non-equilibrium radiation transport through a complex geometry 
with spatially-varying material properties:
- Optically thick regions: σ_a = 200 cm⁻¹, c_v = 0.5 GJ/(cm³·keV)
- Optically thin regions: σ_a = 0.2 cm⁻¹, c_v = 0.0005 GJ/(cm³·keV)

Initial Condition: Cold material everywhere (T = 0.01 keV)
Source: Position-dependent boundary source at z=0 for r<0.5, T = 300 eV (0.3 keV)
"""
import argparse
import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import IMC2D as imc2d
import IMC2D_CarterForest as imc2d_cf

# Add utils to path for plotting
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'utils'))
from plotfuncs import show

# Physical constants from IMC2D
C_LIGHT = imc2d.__c  # 29.98 cm/ns
A_RAD = imc2d.__a    # 0.01372 GJ/(cm³ keV⁴)

T_INIT = 0.01  # keV

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


def opacity_func(T, r, z):
    """
    Spatially-varying absorption opacity for crooked pipe
    Temperature-independent for this problem
    
    Used by IMC for absorption cross-section.
    """
    # Check if inputs are scalar
    scalar_input = np.isscalar(T) and np.isscalar(r) and np.isscalar(z)
    
    thick = is_optically_thick(r, z)
    result = np.where(thick, 200.0, 0.2)  # 200.0 cm⁻¹ if thick, 0.2 cm⁻¹ if thin
    
    # Return scalar if all inputs were scalar
    if scalar_input:
        return float(result)
    return result


def specific_heat_func(T, r, z):
    """
    Spatially-varying specific heat for crooked pipe
    Note: This is ρc_v combined [GJ/(cm³·keV)]
    """
    # Check if inputs are scalar
    scalar_input = np.isscalar(T) and np.isscalar(r) and np.isscalar(z)
    
    thick = is_optically_thick(r, z)
    result = np.where(thick, 0.5, 0.0005)  # 0.5 GJ/(cm³·keV) if thick, 0.0005 if thin
    
    # Return scalar if all inputs were scalar
    if scalar_input:
        return float(result)
    return result


def eos(T, r, z):
    """
    Material energy density for crooked pipe
    e(T, r, z) = ρc_v(r, z) * T
    """
    cv = specific_heat_func(T, r, z)
    return cv * T


def inv_eos(e, r, z):
    """
    Inverse: T from e
    T = e / (ρc_v)
    """
    cv = specific_heat_func(0.0, r, z)  # T-independent
    return e / cv


def cv(T, r, z):
    """Wrapper for specific_heat_func to match IMC interface"""
    return specific_heat_func(T, r, z)


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
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_material_properties(r_centers, z_centers, mesh_tag='uniform'):
    """Plot the material property distribution"""
    
    # Create material property arrays
    nr = len(r_centers)
    nz = len(z_centers)
    opacity_field = np.zeros((nr, nz))
    cv_field = np.zeros((nr, nz))
    
    for i in range(nr):
        for j in range(nz):
            opacity_field[i, j] = opacity_func(0.0, r_centers[i], z_centers[j])
            cv_field[i, j] = specific_heat_func(0.0, r_centers[i], z_centers[j])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    R, Z = np.meshgrid(r_centers, z_centers, indexing='ij')
    
    # Plot opacity
    im1 = ax1.pcolormesh(Z, R, opacity_field, shading='auto', cmap='RdYlBu_r')
    ax1.set_xlabel('z (cm)', fontsize=12)
    ax1.set_ylabel('r (cm)', fontsize=12)
    ax1.set_title('Absorption Opacity σ_a (cm⁻¹)', fontsize=13, fontweight='bold')
    cbar1 = plt.colorbar(im1, ax=ax1, label='σ_a (cm⁻¹)')
    ax1.set_aspect('equal')
    
    # Plot specific heat
    im2 = ax2.pcolormesh(Z, R, cv_field, shading='auto', cmap='RdYlBu_r')
    ax2.set_xlabel('z (cm)', fontsize=12)
    ax2.set_ylabel('r (cm)', fontsize=12)
    ax2.set_title('Specific Heat ρc_v (GJ/(cm³·keV))', fontsize=13, fontweight='bold')
    cbar2 = plt.colorbar(im2, ax=ax2, label='ρc_v (GJ/(cm³·keV))')
    ax2.set_aspect('equal')
    
    plt.suptitle('Crooked Pipe Material Properties (IMC)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fname = f'crooked_pipe_imc_materials_{mesh_tag}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"Saved: {fname}")
    plt.close()


def plot_mesh(r_edges, z_edges, mesh_tag='uniform'):
    """Plot the computational mesh"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot all mesh lines
    for r in r_edges:
        ax.plot([z_edges[0], z_edges[-1]], [r, r], 
                color='black', linewidth=0.3, alpha=0.5)
    for z in z_edges:
        ax.plot([z, z], [r_edges[0], r_edges[-1]], 
                color='black', linewidth=0.3, alpha=0.5)
    
    ax.set_xlabel('z (cm)', fontsize=12)
    ax.set_ylabel('r (cm)', fontsize=12)
    ax.set_title(f'Computational Mesh ({len(r_edges)-1} × {len(z_edges)-1} cells)', 
                 fontsize=13, fontweight='bold')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    filename = f'crooked_pipe_imc_mesh_{mesh_tag}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


def plot_solution(T_mat, T_rad, r_centers, z_centers, time_value, save_prefix='crooked_pipe_imc', first_one=False):
    """Plot material temperature and radiation temperature"""
    
    R, Z = np.meshgrid(r_centers, z_centers, indexing='ij')
    
    # PLOT 1: Material temperature
    fig1, ax1 = plt.subplots(1, 1, figsize=(8, 3))
    if first_one:
        fig1, ax1 = plt.subplots(1, 1, figsize=(8, 3*1.275))
    
    im1 = ax1.pcolormesh(Z, R, T_mat, shading='auto', cmap='plasma', vmin=0.0, vmax=0.3)
    ax1.set_xlabel('z (cm)', fontsize=12)
    ax1.set_ylabel('r (cm)', fontsize=12)
    
    if first_one:
        cbar1 = plt.colorbar(im1, ax=ax1, orientation='horizontal', location='top', pad=0.15, label='T (keV)')
    
    ax1.set_aspect('equal')
    
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
    
    im2 = ax2.pcolormesh(Z, R, T_rad, shading='auto', cmap='plasma', vmin=0.0, vmax=0.3)
    ax2.set_xlabel('z (cm)', fontsize=12)
    ax2.set_ylabel('r (cm)', fontsize=12)
    
    if first_one:
        cbar2 = plt.colorbar(im2, ax=ax2, orientation='horizontal', location='top', pad=0.15, label=r'$T_\mathrm{r}$ (keV)')
    
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    filename2 = f'{save_prefix}_radiation_t_{time_value:.3f}ns.png'
    if first_one:
        show(filename2, close_after=True, cbar_ax=cbar2.ax)
    else:
        show(filename2, close_after=True)
    print(f"Saved: {filename2}")


def plot_fiducial_history(times, fiducial_data, fiducial_data_rad, mesh_tag='uniform'):
    """
    Plot temperature history at fiducial points (both material and radiation)
    """
    # Different markers and colors for each point
    markers = ['o', 's', '^', 'd', 'v']
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot material temperature
    for idx, (label, temps) in enumerate(fiducial_data.items()):
        ax1.loglog(times, temps,
                  marker=markers[idx % len(markers)],
                  color=colors[idx % len(colors)],
                  linewidth=2,
                  markersize=6,
                  markevery=max(1, len(times)//20),
                  label=label,
                  alpha=0.8)
    
    ax1.set_xlabel('Time (ns)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Material Temperature T (keV)', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10, loc='best', framealpha=0.9)
    ax1.grid(True, which='both', alpha=0.3, linestyle='--')
    ax1.grid(True, which='minor', alpha=0.15, linestyle=':')
    
    # Plot radiation temperature
    for idx, (label, temps_rad) in enumerate(fiducial_data_rad.items()):
        ax2.loglog(times, temps_rad,
                  marker=markers[idx % len(markers)],
                  color=colors[idx % len(colors)],
                  linewidth=2,
                  markersize=6,
                  markevery=max(1, len(times)//20),
                  label=label,
                  alpha=0.8)
    
    ax2.set_xlabel('Time (ns)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Radiation Temperature Tr (keV)', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10, loc='best', framealpha=0.9)
    ax2.grid(True, which='both', alpha=0.3, linestyle='--')
    ax2.grid(True, which='minor', alpha=0.15, linestyle=':')
    
    plt.tight_layout()
    filename = f'crooked_pipe_imc_fiducial_history_{mesh_tag}.pdf'
    show(filename, close_after=True)
    print(f"Saved: {filename}")


# =============================================================================
# CHECKPOINT HELPERS
# =============================================================================

def save_checkpoint(state, step_count, times, fiducial_data, fiducial_data_rad, output_times_saved,
                    current_dt, filename, method='fc', fastpath_threshold=0.0,
                    no_progress_rel_tol=1e-12, no_progress_max_streak=8,
                    profile_phase_totals_s=None, profile_event_totals=None,
                    profile_transported_particles_total=0, profile_step_count=0):
    """
    Save full IMC state + run history to an NPZ checkpoint file.
    
    All data is stored as plain NumPy arrays so the file can be reloaded 
    independently of the class definitions.
    """
    # Convert fiducial_data dict to arrays
    fiducial_labels = list(fiducial_data.keys())
    fiducial_temps = np.array([fiducial_data[lbl] for lbl in fiducial_labels])
    fiducial_temps_rad = np.array([fiducial_data_rad[lbl] for lbl in fiducial_labels])
    if profile_phase_totals_s is None:
        profile_phase_totals_s = np.zeros(4, dtype=np.float64)
    if profile_event_totals is None:
        profile_event_totals = np.zeros(8, dtype=np.int64)
    
    # Save checkpoint
    np.savez(
        filename,
        # Material state
        chkpt_temperature=state.temperature,
        chkpt_radiation_temperature=state.radiation_temperature,
        chkpt_internal_energy=state.internal_energy,
        # Census particles (if any)
        chkpt_weights=state.weights if hasattr(state, 'weights') else np.array([]),
        chkpt_mu=state.mu if hasattr(state, 'mu') else np.array([]),
        chkpt_eta=state.eta if hasattr(state, 'eta') else np.array([]),
        chkpt_particle_times=state.times if hasattr(state, 'times') else np.array([]),
        chkpt_r=state.r if hasattr(state, 'r') else np.array([]),
        chkpt_z=state.z if hasattr(state, 'z') else np.array([]),
        # Scalar run state
        chkpt_time=np.float64(state.time),
        chkpt_current_dt=np.float64(current_dt),
        chkpt_step_count=np.int64(step_count),
        # Time history & output bookkeeping
        chkpt_times=np.array(times, dtype=np.float64),
        chkpt_output_saved=np.array(sorted(output_times_saved), dtype=np.float64),
        # Fiducial history
        chkpt_fiducial_labels=np.array(fiducial_labels, dtype=object),
        chkpt_fiducial_temps=fiducial_temps,
        chkpt_fiducial_temps_rad=fiducial_temps_rad,
        # Run metadata + profiling summary accumulators
        chkpt_method=np.array(method, dtype=object),
        chkpt_fastpath_threshold=np.float64(fastpath_threshold),
        chkpt_no_progress_rel_tol=np.float64(no_progress_rel_tol),
        chkpt_no_progress_max_streak=np.int64(no_progress_max_streak),
        chkpt_profile_phase_totals_s=np.array(profile_phase_totals_s, dtype=np.float64),
        chkpt_profile_event_totals=np.array(profile_event_totals, dtype=np.int64),
        chkpt_profile_transported_particles_total=np.int64(profile_transported_particles_total),
        chkpt_profile_step_count=np.int64(profile_step_count),
    )


def load_checkpoint(filename, state, fiducial_points):
    """
    Restore IMC state and run history from an NPZ checkpoint.
    
    Modifies *state* in-place and returns (step_count, times, fiducial_data,
    fiducial_data_rad, output_times_saved, current_dt, metadata) ready to continue the time loop.
    """
    ckpt = np.load(filename, allow_pickle=True)
    
    # Restore material state
    state.temperature[:] = ckpt['chkpt_temperature']
    state.radiation_temperature[:] = ckpt['chkpt_radiation_temperature']
    state.internal_energy[:] = ckpt['chkpt_internal_energy']
    state.time = float(ckpt['chkpt_time'])
    
    # Restore census particles
    state.weights = ckpt['chkpt_weights']
    state.mu = ckpt['chkpt_mu']
    state.eta = ckpt['chkpt_eta']
    state.times = ckpt['chkpt_particle_times']
    state.r = ckpt['chkpt_r']
    state.z = ckpt['chkpt_z']
    
    # Restore run state
    step_count = int(ckpt['chkpt_step_count'])
    current_dt = float(ckpt['chkpt_current_dt'])
    times = list(ckpt['chkpt_times'])
    output_times_saved = set(float(v) for v in ckpt['chkpt_output_saved'])
    
    # Restore fiducial histories
    fiducial_labels = ckpt['chkpt_fiducial_labels']
    fiducial_temps = ckpt['chkpt_fiducial_temps']
    fiducial_temps_rad = ckpt['chkpt_fiducial_temps_rad']
    fiducial_data = {}
    fiducial_data_rad = {}
    for idx, lbl in enumerate(fiducial_labels):
        fiducial_data[str(lbl)] = list(fiducial_temps[idx])
        fiducial_data_rad[str(lbl)] = list(fiducial_temps_rad[idx])

    metadata = {
        'method': str(ckpt['chkpt_method']) if 'chkpt_method' in ckpt else 'fc',
        'fastpath_threshold': float(ckpt['chkpt_fastpath_threshold']) if 'chkpt_fastpath_threshold' in ckpt else 0.0,
        'no_progress_rel_tol': float(ckpt['chkpt_no_progress_rel_tol']) if 'chkpt_no_progress_rel_tol' in ckpt else 1e-12,
        'no_progress_max_streak': int(ckpt['chkpt_no_progress_max_streak']) if 'chkpt_no_progress_max_streak' in ckpt else 8,
        'profile_phase_totals_s': ckpt['chkpt_profile_phase_totals_s'].astype(np.float64) if 'chkpt_profile_phase_totals_s' in ckpt else np.zeros(4, dtype=np.float64),
        'profile_event_totals': ckpt['chkpt_profile_event_totals'].astype(np.int64) if 'chkpt_profile_event_totals' in ckpt else np.zeros(8, dtype=np.int64),
        'profile_transported_particles_total': int(ckpt['chkpt_profile_transported_particles_total']) if 'chkpt_profile_transported_particles_total' in ckpt else 0,
        'profile_step_count': int(ckpt['chkpt_profile_step_count']) if 'chkpt_profile_step_count' in ckpt else 0,
    }
    
    return step_count, times, fiducial_data, fiducial_data_rad, output_times_saved, current_dt, metadata


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main(output_times=None, nr=60, nz=210, dt_initial=0.001, dt_max=10.0,
         dt_increase_factor=1.1, Ntarget=20000, Nboundary=10000, Nmax=100000,
         max_events_per_particle=10**6, restart_file=None, checkpoint_every=0,
         use_refined_mesh=False, n_refine=10, refine_width=0.05,
         method='fc', fastpath_threshold=0.0,
         no_progress_rel_tol=1e-12, no_progress_max_streak=8):
    """
    Run the IMC Crooked Pipe test problem
    
    Parameters:
    -----------
    output_times : list or None
        List of times (ns) at which to save colormap plots.
        If None, defaults to [1.0, 5.0, 10.0, 100.0]
    nr : int
        Number of cells in r direction (uniform mesh) or coarse cells (refined mesh)
    nz : int
        Number of cells in z direction (uniform mesh) or coarse cells (refined mesh)
    dt_initial : float
        Initial time step (ns)
    dt_max : float
        Maximum time step (ns)
    dt_increase_factor : float
        Factor by which to increase dt each step (e.g., 1.1 for 10% increase)
    Ntarget : int
        Target number of source particles per step
    Nboundary : int
        Number of boundary source particles per step (emitted from z=0, r<0.5)
    Nmax : int
        Maximum census particles after combing
    max_events_per_particle : int
        Maximum scattering events per particle
    restart_file : str or None
        Path to checkpoint file to restart from
    checkpoint_every : int
        Save checkpoint every N steps (0 to disable periodic checkpoints)
    use_refined_mesh : bool
        If True, use mesh with refinement at material interfaces
    n_refine : int
        Number of refined cells to subdivide each coarse cell in refinement zones
    refine_width : float
        Width (in cm) around each interface to apply refinement
    method : str
        Transport backend: 'fc' (Fleck-Cummings) or 'cf' (Carter-Forest)
    fastpath_threshold : float
        CF-only stiff-regime fastpath threshold (disabled when 0.0)
    no_progress_rel_tol : float
        CF-only relative tolerance for no-progress distance-to-census detection
    no_progress_max_streak : int
        CF-only number of consecutive no-progress events before forcing census
    """
    method = method.lower()
    if method not in ('fc', 'cf'):
        raise ValueError("method must be 'fc' or 'cf'")
    backend = imc2d if method == 'fc' else imc2d_cf

    if output_times is None:
        output_times = [1.0, 5.0, 10.0, 100.0, 200.0, 500.0, 1000.0]
    
    print("="*80)
    print("CROOKED PIPE TEST PROBLEM - IMC 2D")
    print("="*80)
    print("2D Cylindrical (r-z) Geometry")
    print("Domain: r ∈ [0.0, 2.0] cm, z ∈ [0.0, 7.0] cm")
    print()
    print("Material Properties:")
    print("  Optically thick: σ_a = 200 cm⁻¹, ρc_v = 0.5 GJ/(cm³·keV)")
    print("  Optically thin:  σ_a = 0.2 cm⁻¹,  ρc_v = 0.0005 GJ/(cm³·keV)")
    print()
    print("Initial Condition: Cold material T = 0.01 keV everywhere")
    print()
    print("Boundary Conditions:")
    print("  Left (r=0): Reflecting (symmetry axis)")
    print("  Right (r_max): Vacuum/open")
    print("  Bottom (z=0): Blackbody source at T=0.3 keV for r<0.5 cm")
    print("  Top (z_max): Vacuum/open")
    print()
    print(f"Mesh type: {'refined at interfaces' if use_refined_mesh else 'uniform'}")
    if use_refined_mesh:
        print(f"  Coarse cells: {nr} × {nz}")
        print(f"  Refinement: {n_refine} cells per coarse cell, width={refine_width} cm")
    else:
        print(f"  Cells: {nr} × {nz}")
    print(f"Output times: {output_times} ns")
    print(f"Ntarget={Ntarget}, Nboundary={Nboundary}, Nmax={Nmax}")
    print(f"Time stepping: dt_initial={dt_initial} ns, dt_max={dt_max} ns, increase_factor={dt_increase_factor}")
    print(f"Method: {method.upper()}")
    print(f"max_events={max_events_per_particle}")
    if method == 'cf':
        print(f"CF fastpath threshold={fastpath_threshold}")
        print(f"CF no-progress rel tol={no_progress_rel_tol}")
        print(f"CF no-progress max streak={no_progress_max_streak}")
    print("="*80)
    
    # Determine mesh tag for output filenames (include particle info and dt_max)
    mesh_tag = "refined" if use_refined_mesh else "uniform"
    particle_tag = f"Nb{Nboundary}" if Nboundary > 0 else f"Nt{Ntarget}"
    dtmax_tag = f"dtmax{dt_max}"
    run_tag = f"{method}_{mesh_tag}_{particle_tag}_{dtmax_tag}"
    
    # Create mesh
    if use_refined_mesh:
        print("\nGenerating refined mesh...")
        # Material interfaces:
        # r = 0.5, 1.0, 1.5 cm (transitions between thin/thick regions)
        # z = 2.5, 3.0, 4.0, 4.5 cm (transitions between thin/thick regions)
        interface_locations_r = [0.5, 1.0, 1.5]
        interface_locations_z = [2.5, 3.0, 4.0, 4.5]
        
        # Refined r-direction faces
        r_edges = generate_refined_z_faces(
            0.0, 2.0, interface_locations_r, n_refine, nr, refine_width
        )
        
        # Refined z-direction faces
        z_edges = generate_refined_z_faces(
            0.0, 7.0, interface_locations_z, n_refine, nz, refine_width
        )
        
        # Print mesh info
        dr_min = np.min(np.diff(r_edges))
        dz_min = np.min(np.diff(z_edges))
        print(f"  Minimum dr = {dr_min:.5f} cm, Minimum dz = {dz_min:.5f} cm")
        print(f"  Final mesh: {len(r_edges)-1} × {len(z_edges)-1} cells")
        print(f"  Refinement at r = 0.5, 1.0, 1.5 cm and z = 2.5, 3.0, 4.0, 4.5 cm")
    else:
        r_edges = np.linspace(0.0, 2.0, nr + 1)
        z_edges = np.linspace(0.0, 7.0, nz + 1)
    
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    
    # Create mesh grids for position-dependent material properties
    R_grid, Z_grid = np.meshgrid(r_centers, z_centers, indexing='ij')
    
    # Create wrapper functions that use the mesh grid for spatially-varying properties
    def eos_wrapper(T):
        """EOS wrapper that applies spatially-varying c_v"""
        cv_grid = specific_heat_func(T, R_grid, Z_grid)
        return cv_grid * T
    
    def inv_eos_wrapper(e):
        """Inverse EOS wrapper that applies spatially-varying c_v"""
        cv_grid = specific_heat_func(0.0, R_grid, Z_grid)  # T-independent
        return e / cv_grid
    
    def cv_wrapper(T):
        """Specific heat wrapper that applies spatially-varying c_v"""
        return specific_heat_func(T, R_grid, Z_grid)
    
    def opacity_wrapper(T):
        """Opacity wrapper that applies spatially-varying opacity"""
        return opacity_func(T, R_grid, Z_grid)
    
    # Plot material properties
    print("\nPlotting material properties...")
    plot_material_properties(r_centers, z_centers, run_tag)
    
    # Plot mesh
    print("Plotting mesh...")
    plot_mesh(r_edges, z_edges, run_tag)
    
    # Initial condition: cold material
    print("\nSetting initial condition...")
    nr_actual = len(r_edges) - 1
    nz_actual = len(z_edges) - 1
    T_init_2d = np.full((nr_actual, nz_actual), T_INIT)
    Tr_init_2d = np.full((nr_actual, nz_actual), T_INIT)
    
    # Boundary source configuration
    # Source at bottom boundary (z=0) where r < 0.5, with T = 0.3 keV
    T_source = 0.3  # keV
    source_2d = np.zeros((nr_actual, nz_actual))  # No volumetric source
    
    # Define position-dependent boundary source function
    # Returns temperature (keV) for emission, 0.0 for no emission
    def boundary_source_func(r, z, side):
        """Return boundary temperature at position (r, z) on specified side"""
        if side == 'zmin' and r < 0.5:
            return T_source  # Emit at 0.3 keV where r < 0.5
        return 0.0  # No emission elsewhere
    
    print(f"  Boundary source: T = {T_source} keV at z=0 for r < 0.5 cm")
    print(f"  Source region area: {np.pi * 0.5**2:.3f} cm²")
    
    # Initialize IMC state
    print("\nInitializing IMC simulation...")
    state = backend.init_simulation(
        Ntarget,
        T_init_2d,
        Tr_init_2d,
        r_edges,
        z_edges,
        eos_wrapper,
        inv_eos_wrapper,
        geometry='rz',
    )
    
    # Boundary conditions:
    # (left, right, bottom, top)
    # left (r=0): reflecting axis
    # right (r=2.0): vacuum
    # bottom (z=0): position-dependent source via boundary_source_func (T ignored)
    # top (z=7.0): vacuum
    
    T_boundary = (0.0, 0.0, 0.0, 0.0)  # All vacuum (bottom uses boundary_source_func instead)
    reflect = (True, False, False, False)  # Only left boundary (axis) is reflecting
    
    # Define fiducial points for tracking
    fiducial_points = {
        'Point 1: r=0.0, z=0.25': (0.0, 0.25),
        'Point 2: r=0.0, z=2.75': (0.0, 2.75),
        'Point 3: r=1.25, z=3.5': (1.25, 3.5),
        'Point 4: r=0.0, z=4.25': (0.0, 4.25),
        'Point 5: r=0.0, z=6.75': (0.0, 6.75)
    }
    
    # Find indices for each fiducial point
    fiducial_indices = {}
    for label, (r_val, z_val) in fiducial_points.items():
        i = np.argmin(np.abs(r_centers - r_val))
        j = np.argmin(np.abs(z_centers - z_val))
        fiducial_indices[label] = (i, j)
        print(f"  {label} -> cell ({i}, {j}) at (r={r_centers[i]:.3f}, z={z_centers[j]:.3f})")
    
    # Default checkpoint filename
    checkpoint_file = f'crooked_pipe_imc_checkpoint_{run_tag}_{len(r_edges)-1}x{len(z_edges)-1}.npz'

    # Profiling accumulators saved to checkpoints/solution files
    profile_phase_totals_s = np.zeros(4, dtype=np.float64)
    profile_event_totals = np.zeros(8, dtype=np.int64)
    profile_transported_particles_total = 0
    profile_step_count = 0
    wall_start = time.perf_counter()
    
    # =============================================================================
    # Initial condition OR restart from checkpoint
    # =============================================================================
    if restart_file is not None:
        print(f"\nLoading restart from: {restart_file}")
        step_count, times, fiducial_data, fiducial_data_rad, output_times_saved, current_dt, metadata = load_checkpoint(
            restart_file, state, fiducial_points
        )
        if metadata['method'] != method:
            raise ValueError(f"Restart checkpoint method={metadata['method']} does not match requested method={method}")
        if method == 'cf':
            no_progress_rel_tol = metadata.get('no_progress_rel_tol', no_progress_rel_tol)
            no_progress_max_streak = metadata.get('no_progress_max_streak', no_progress_max_streak)
        profile_phase_totals_s = metadata['profile_phase_totals_s']
        profile_event_totals = metadata['profile_event_totals']
        profile_transported_particles_total = metadata['profile_transported_particles_total']
        profile_step_count = metadata['profile_step_count']
        print(f"  Resumed at t = {state.time:.6e} ns, step {step_count}")
        print(f"  Census particles: {len(state.weights)}")
        print(f"  Current dt: {current_dt:.6e} ns")
    else:
        # Fresh start
        print("\nStarting from initial conditions...")
        step_count = 0
        current_dt = dt_initial
        times = [0.0]
        fiducial_data = {label: [T_INIT] for label in fiducial_points.keys()}
        fiducial_data_rad = {label: [T_INIT] for label in fiducial_points.keys()}
        output_times_saved = set()
    
    # Calculate initial energy for status output
    # For cylindrical geometry, volumes are 2π * r * Δr * Δz
    # But since we're in (r,z) coordinates, we integrate with factor of 2πr
    dr = r_edges[1:] - r_edges[:-1]
    dz = z_edges[1:] - z_edges[:-1]
    # Volume factors for each cell: 2π * r_center * Δr * Δz
    r_volumes = 2.0 * np.pi * r_centers[:, None] * dr[:, None] * dz[None, :]
    
    # Material energy (using internal_energy from state)
    initial_E_mat = np.sum(state.internal_energy * r_volumes)
    # Radiation energy (from radiation_temperature)
    E_rad_density = A_RAD * state.radiation_temperature**4
    initial_E_rad = np.sum(E_rad_density * r_volumes)
    initial_E_tot = initial_E_mat + initial_E_rad
    initial_N = len(state.weights) if hasattr(state, 'weights') else 0
    
    print(f"\nCheckpoint file: {checkpoint_file}")
    if checkpoint_every > 0:
        print(f"Periodic checkpoints every {checkpoint_every} steps")
    
    # Track which output times have been saved
    # (already initialized from checkpoint if restarting)
    
    # Time evolution
    print("\nTime stepping...")
    print("Time", "N", "Total Energy", "Total Internal Energy", "Total Radiation Energy",
          "Boundary Emission", "Lost Energy", sep='\t')
    print("===============================================================================================================")
    
    # Print initial/restart state
    if restart_file is None:
        print("{:.6f}".format(0.0), initial_N,
              "{:.6f}".format(initial_E_tot),
              "{:.6f}".format(initial_E_mat),
              "{:.6f}".format(initial_E_rad),
              "{:.6f}".format(0.0), "{:.6f}".format(0.0), sep='\t')
    
    t_final = max(output_times) if output_times else 10.0  # ns
    first_one = (len(output_times_saved) == 0)  # First plot if no outputs saved yet
    
    while state.time < t_final - 1e-12:
        # Determine time step for this iteration
        step_dt = min(current_dt, t_final - state.time)
        
        # Check if we need to reduce dt to hit an output time exactly
        hit_output_time = False
        for tout in output_times:
            if tout not in output_times_saved and state.time < tout < state.time + step_dt:
                step_dt = tout - state.time
                hit_output_time = True
                break
        
        # Advance one time step
        step_kwargs = dict(
            state=state,
            Ntarget=Ntarget,
            Nboundary=Nboundary,
            Nsource=0,
            Nmax=Nmax,
            T_boundary=T_boundary,
            dt=step_dt,
            edges1=r_edges,
            edges2=z_edges,
            sigma_a_func=opacity_wrapper,
            inv_eos=inv_eos_wrapper,
            cv=cv_wrapper,
            source=source_2d,
            reflect=reflect,
            geometry='rz',
            rz_linear_source=True,
            max_events_per_particle=max_events_per_particle,
            boundary_source_func=boundary_source_func,
        )
        if method == 'cf':
            step_kwargs['fastpath_threshold'] = fastpath_threshold
            step_kwargs['no_progress_rel_tol'] = no_progress_rel_tol
            step_kwargs['no_progress_max_streak'] = no_progress_max_streak

        t_backend_start = time.perf_counter()
        state, info = backend.step(
            **step_kwargs,
        )
        t_backend_elapsed = time.perf_counter() - t_backend_start
        
        step_count += 1
        
        # Print progress
        if step_count % 10 == 0:
            total_energy = info['total_internal_energy'] + info['total_radiation_energy']
            print("{:.6f}".format(state.time), info['N_particles'],
                  "{:.6f}".format(total_energy),
                  "{:.6f}".format(info['total_internal_energy']),
                  "{:.6f}".format(info['total_radiation_energy']),
                  "{:.6f}".format(info['boundary_emission']),
                  "{:.6e}".format(info['energy_loss']), sep='\t')
        
        # Store time history
        times.append(state.time)
        for label, (i, j) in fiducial_indices.items():
            fiducial_data[label].append(state.temperature[i, j])
            fiducial_data_rad[label].append(state.radiation_temperature[i, j])

        # Aggregate per-step profiling data from backend
        if 'profiling' in info:
            phase = info['profiling'].get('phase_times_s', {})
            profile_phase_totals_s[0] += float(phase.get('sampling', 0.0))
            profile_phase_totals_s[1] += float(phase.get('transport', 0.0))
            profile_phase_totals_s[2] += float(phase.get('postprocess', 0.0))
            profile_phase_totals_s[3] += float(phase.get('total', 0.0))
            events = info['profiling'].get('transport_events', {})
            profile_event_totals[0] += int(events.get('total', 0))
            profile_event_totals[1] += int(events.get('boundary_crossings', 0))
            profile_event_totals[2] += int(events.get('absorption_continue_events', 0))
            profile_event_totals[3] += int(events.get('census_events', 0))
            profile_event_totals[4] += int(events.get('absorption_capture_events', 0))
            profile_event_totals[5] += int(events.get('weight_floor_kills', 0))
            profile_event_totals[6] += int(events.get('reflections', 0))
            profile_event_totals[7] += int(events.get('event_cap_hits', 0))
            profile_transported_particles_total += int(events.get('n_particles_transported', 0))
            profile_step_count += 1
        else:
            # Fallback for backends without detailed profiling payload.
            profile_phase_totals_s[1] += t_backend_elapsed
            profile_phase_totals_s[3] += t_backend_elapsed
            profile_step_count += 1
        
        # Save snapshots at output times
        for tout in output_times:
            if tout not in output_times_saved and abs(state.time - tout) < 1e-9:
                print(f"  >> Saving snapshot at t={state.time:.3f} ns")
                plot_solution(state.temperature.copy(), state.radiation_temperature.copy(),
                            r_centers, z_centers, state.time, save_prefix=f'crooked_pipe_imc_{run_tag}', first_one=first_one)
                output_times_saved.add(tout)
                first_one = False
                # Save checkpoint at output time
                print(f"  >> Checkpoint → {checkpoint_file}")
                save_checkpoint(state, step_count, times, fiducial_data, fiducial_data_rad,
                              output_times_saved, current_dt, checkpoint_file,
                              method=method,
                              fastpath_threshold=fastpath_threshold,
                              no_progress_rel_tol=no_progress_rel_tol,
                              no_progress_max_streak=no_progress_max_streak,
                              profile_phase_totals_s=profile_phase_totals_s,
                              profile_event_totals=profile_event_totals,
                              profile_transported_particles_total=profile_transported_particles_total,
                              profile_step_count=profile_step_count)
                break
        
        # Periodic checkpoint (if requested and not already saved above)
        if checkpoint_every > 0 and step_count % checkpoint_every == 0:
            if not any(abs(state.time - tout) < 1e-9 for tout in output_times if tout not in output_times_saved):
                print(f"  >> Periodic checkpoint (step {step_count}) → {checkpoint_file}")
                save_checkpoint(state, step_count, times, fiducial_data, fiducial_data_rad,
                              output_times_saved, current_dt, checkpoint_file,
                              method=method,
                              fastpath_threshold=fastpath_threshold,
                              no_progress_rel_tol=no_progress_rel_tol,
                              no_progress_max_streak=no_progress_max_streak,
                              profile_phase_totals_s=profile_phase_totals_s,
                              profile_event_totals=profile_event_totals,
                              profile_transported_particles_total=profile_transported_particles_total,
                              profile_step_count=profile_step_count)
        
        # Increase time step for next iteration (capped at dt_max)
        if not hit_output_time:
            current_dt = min(current_dt * dt_increase_factor, dt_max)
    
    # Convert to numpy arrays
    times = np.array(times)
    for label in fiducial_data.keys():
        fiducial_data[label] = np.array(fiducial_data[label])
        fiducial_data_rad[label] = np.array(fiducial_data_rad[label])
    
    wall_elapsed_s = time.perf_counter() - wall_start
    avg_events_per_particle = (
        float(profile_event_totals[0]) / float(profile_transported_particles_total)
        if profile_transported_particles_total > 0 else 0.0
    )

    # Store solution in npz file
    npz_filename = f'crooked_pipe_imc_solution_{run_tag}_{len(r_edges)-1}x{len(z_edges)-1}.npz'
    np.savez(npz_filename,
             r_centers=r_centers,
             z_centers=z_centers,
             T_final=state.temperature,
             Tr_final=state.radiation_temperature,
             times=times,
             fiducial_data=fiducial_data,
             fiducial_data_rad=fiducial_data_rad,
             method=np.array(method, dtype=object),
             fastpath_threshold=np.float64(fastpath_threshold),
             no_progress_rel_tol=np.float64(no_progress_rel_tol),
             no_progress_max_streak=np.int64(no_progress_max_streak),
             profile_phase_keys=np.array(['sampling', 'transport', 'postprocess', 'total'], dtype=object),
             profile_phase_totals_s=profile_phase_totals_s,
             profile_event_keys=np.array([
                 'total',
                 'boundary_crossings',
                 'absorption_continue_events',
                 'census_events',
                 'absorption_capture_events',
                 'weight_floor_kills',
                 'reflections',
                 'event_cap_hits',
             ], dtype=object),
             profile_event_totals=profile_event_totals,
             profile_transported_particles_total=np.int64(profile_transported_particles_total),
             profile_step_count=np.int64(profile_step_count),
             profile_avg_events_per_particle=np.float64(avg_events_per_particle),
             profile_wall_elapsed_s=np.float64(wall_elapsed_s))
    print(f"\nSaved: {npz_filename}")
    
    # Plot fiducial point history
    print("\nPlotting fiducial point temperature history...")
    plot_fiducial_history(times, fiducial_data, fiducial_data_rad, run_tag)
    
    # Print some diagnostics
    print("\n" + "="*80)
    print("SOLUTION SUMMARY")
    print("="*80)
    print(f"Final time: {state.time:.3f} ns")
    print(f"Total steps: {step_count}")
    print(f"Temperature range: {state.temperature.min():.4f} to {state.temperature.max():.4f} keV")
    
    i_max, j_max = np.unravel_index(state.temperature.argmax(), state.temperature.shape)
    print(f"Max temperature location: r={r_centers[i_max]:.3f} cm, z={z_centers[j_max]:.3f} cm")
    
    # Print fiducial point final temperatures
    print("\nFiducial point final temperatures:")
    for label, (i, j) in fiducial_indices.items():
        T_final = state.temperature[i, j]
        print(f"  {label}: T = {T_final:.4f} keV")
    
    print("\n" + "="*80)
    print("Crooked Pipe IMC test completed successfully!")
    print("="*80)
    
    return state


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run IMC Crooked Pipe test problem.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["fc", "cf"],
        default="fc",
        help="Transport backend: fc (Fleck-Cummings) or cf (Carter-Forest)",
    )
    parser.add_argument(
        "--nr",
        type=int,
        default=60,
        help="Number of cells in r direction",
    )
    parser.add_argument(
        "--nz",
        type=int,
        default=210,
        help="Number of cells in z direction",
    )
    parser.add_argument(
        "--dt-initial",
        type=float,
        default=0.001,
        help="Initial time step (ns)",
    )
    parser.add_argument(
        "--dt-max",
        type=float,
        default=10.0,
        help="Maximum time step (ns)",
    )
    parser.add_argument(
        "--dt-increase-factor",
        type=float,
        default=1.1,
        help="Factor by which to increase dt each step (e.g., 1.1 for 10%% increase)",
    )
    parser.add_argument(
        "--Ntarget",
        type=int,
        default=20000,
        help="Target particles per step",
    )
    parser.add_argument(
        "--Nboundary",
        type=int,
        default=10000,
        help="Boundary source particles per step (emitted from z=0, r<0.5)",
    )
    parser.add_argument(
        "--Nmax",
        type=int,
        default=100000,
        help="Maximum census particles",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=10**6,
        help="Maximum scattering events per particle",
    )
    parser.add_argument(
        "--output-times",
        type=str,
        default=None,
        help="Comma-separated output times (ns), e.g. '1.0,10.0,100.0'",
    )
    parser.add_argument(
        "--restart-file",
        type=str,
        default=None,
        help="Path to checkpoint file to restart from",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Save checkpoint every N steps (0 to disable periodic checkpoints)",
    )
    parser.add_argument(
        "--use-refined-mesh",
        action="store_true",
        help="Use mesh with refinement at material interfaces",
    )
    parser.add_argument(
        "--n-refine",
        type=int,
        default=10,
        help="Number of refined cells to subdivide each coarse cell in refinement zones",
    )
    parser.add_argument(
        "--refine-width",
        type=float,
        default=0.05,
        help="Width (in cm) around each interface to apply refinement",
    )
    parser.add_argument(
        "--fastpath-threshold",
        type=float,
        default=0.0,
        help="CF-only fastpath threshold (lambda cutoff; 0 disables)",
    )
    parser.add_argument(
        "--no-progress-rel-tol",
        type=float,
        default=1e-12,
        help="CF-only relative tolerance for no-progress loop detection",
    )
    parser.add_argument(
        "--no-progress-max-streak",
        type=int,
        default=8,
        help="CF-only consecutive no-progress events before forcing census",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    if args.output_times:
        output_times = [float(t) for t in args.output_times.split(',')]
    else:
        output_times = [1.0, 5.0, 10.0, 100.0, 200.0, 500.0, 1000.0]

    state = main(
        output_times=output_times,
        nr=args.nr,
        nz=args.nz,
        dt_initial=args.dt_initial,
        dt_max=args.dt_max,
        dt_increase_factor=args.dt_increase_factor,
        Ntarget=args.Ntarget,
        Nboundary=args.Nboundary,
        Nmax=args.Nmax,
        max_events_per_particle=args.max_events,
        restart_file=args.restart_file,
        checkpoint_every=args.checkpoint_every,
        use_refined_mesh=args.use_refined_mesh,
        n_refine=args.n_refine,
        refine_width=args.refine_width,
        method=args.method,
        fastpath_threshold=args.fastpath_threshold,
        no_progress_rel_tol=args.no_progress_rel_tol,
        no_progress_max_streak=args.no_progress_max_streak,
    )
