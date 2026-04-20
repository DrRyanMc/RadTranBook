#!/usr/bin/env python3
"""2D IMC simulation of the refined_zoning_noneq diffusion problem.

Geometry: 2D Cartesian, xy-mode in IMC2D (x = x-direction, y = z-direction).
Domain:   x ∈ [0, 5] cm,  z ∈ [0, 5] cm

Materials (temperature-independent, position-dependent):
  Optically thick (default):    σ_a = 200 cm⁻¹,  ρc_v = 0.5  GJ/(cm³·keV)
  Lower thin channel x∈[1,2], z<2:  σ_a = 0.2  cm⁻¹,  ρc_v = 0.05 GJ/(cm³·keV)
  Upper thin channel x∈[3,4], z>3:  σ_a = 0.2  cm⁻¹,  ρc_v = 0.05 GJ/(cm³·keV)

Initial condition: T = 0.01 keV everywhere (cold).
Boundary conditions:
  Left (x=0) / Right (x=5): reflecting (vacuum outside treated as perfect mirror)
  Bottom (z=0): Lambertian source at T_bc = 0.3 keV for x ∈ [1.0, 2.0], turns off at t>=500 ns
  Top (z=5): Lambertian source at T_bc = 0.3 keV for x ∈ [3.0, 4.0], turns off at t>=500 ns
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import IMC2D as imc2d

# Add utils to path for plotting
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'utils'))
from plotfuncs import show

__a = imc2d.__a
__c = imc2d.__c

# ── material constants ────────────────────────────────────────────────────────
SIG_THICK = 200.0   # cm⁻¹
SIG_THIN  =   0.2   # cm⁻¹
CV_THICK  =   0.5   # GJ/(cm³·keV)
CV_THIN   =   0.05  # GJ/(cm³·keV)

T_INIT = 0.01   # keV  (uniform cold start)
T_BC   = 0.3    # keV  (boundary temperature, bottom and top)
T_CUTOFF = 500.0  # ns  (time when boundary source turns off)


# ── geometry ──────────────────────────────────────────────────────────────────
def is_optically_thin(x_arr, z_arr):
    """Return bool array: True where cells are in a thin channel."""
    lower = (x_arr >= 1.0) & (x_arr <= 2.0) & (z_arr <  2.0)
    upper = (x_arr >= 3.0) & (x_arr <= 4.0) & (z_arr >  3.0)
    return lower | upper


def build_material_arrays(x_centers, z_centers):
    """Pre-build 2-D σ_a and ρc_v arrays (shape nx × nz)."""
    X, Z = np.meshgrid(x_centers, z_centers, indexing="ij")
    thin = is_optically_thin(X, Z)
    sigma = np.where(thin, SIG_THIN,  SIG_THICK)
    cv    = np.where(thin, CV_THIN,   CV_THICK)
    return sigma, cv


def make_problem_funcs(sigma_arr, cv_arr):
    """Return (sigma_a_func, eos, inv_eos, cv_func) closures for IMC2D.

    All functions accept a 2-D temperature or energy array and return a 2-D
    array of the same shape.  Because the material properties are assumed
    temperature-independent here, the position map is pre-built once.
    """
    def sigma_a_func(T):
        return sigma_arr               # shape (nx, nz), ignores T

    def eos(T):
        return cv_arr * T              # internal energy density e = ρc_v T

    def inv_eos(u):
        return u / cv_arr              # T = e / ρc_v

    def cv_func(T):
        return cv_arr                  # ignores T

    return sigma_a_func, eos, inv_eos, cv_func


# ── mesh generation with interface refinement ─────────────────────────────────
def create_log_spacing_one_sided(z_start, z_end, n_cells, from_left=True):
    """Create logarithmically-spaced cells in one direction."""
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


def create_log_spacing_around_interface(z_left, z_right, z_int, n_cells):
    """Create logarithmically-spaced cells around interface."""
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


def generate_refined_faces(coord_min, coord_max, interface_locations, n_refine, n_coarse, refine_width=0.05):
    """
    Generate face positions with logarithmic refinement at material interfaces.
    
    Parameters:
    -----------
    coord_min, coord_max : float
        Domain boundaries
    interface_locations : list of float
        Coordinates where material interfaces occur
    n_refine : int
        Number of refined cells to subdivide each coarse cell in refinement zones
    n_coarse : int
        Number of coarse cells across entire domain
    refine_width : float
        Width (in cm) around each interface to apply refinement
    
    Returns:
    --------
    faces : ndarray
        Face positions with logarithmic refinement at interfaces
    """
    # Create a uniform coarse grid
    dcoord_coarse = (coord_max - coord_min) / n_coarse
    coarse_faces = np.linspace(coord_min, coord_max, n_coarse + 1)
    
    # Sort interfaces
    interfaces = sorted(interface_locations)
    
    # Mark which coarse cells should be refined
    refine_info = {}  # i -> nearest interface coord
    
    for coord_int in interfaces:
        for i in range(n_coarse):
            cell_left = coarse_faces[i]
            cell_right = coarse_faces[i + 1]
            
            # Refine if cell overlaps with refinement zone
            if (cell_left <= coord_int + refine_width) and (cell_right >= coord_int - refine_width):
                if i not in refine_info:
                    refine_info[i] = coord_int
                else:
                    # Use nearest interface
                    if abs(coord_int - (cell_left + cell_right)/2) < abs(refine_info[i] - (cell_left + cell_right)/2):
                        refine_info[i] = coord_int
    
    # Build final face list
    faces_list = [coord_min]
    
    for i in range(n_coarse):
        cell_left = coarse_faces[i]
        cell_right = coarse_faces[i + 1]
        
        if i in refine_info:
            coord_int = refine_info[i]
            refined = create_log_spacing_around_interface(
                cell_left, cell_right, coord_int, n_refine
            )
            faces_list.extend(refined)
        else:
            faces_list.append(cell_right)
    
    return np.array(faces_list)


# ── mesh ──────────────────────────────────────────────────────────────────────
# Generate refined mesh ONLY at upper channel interfaces (high x)
# This allows comparison within a single simulation:
#   Lower channel (x=1-2): coarse zoning, unresolved interfaces
#   Upper channel (x=3-4): refined zoning, resolved interfaces
# Material interfaces: x = 3, 4 (upper thin channel boundaries only)
#                      z = 3 (upper channel boundary only)
x_edges = generate_refined_faces(
    coord_min=0.0,
    coord_max=5.0,
    interface_locations=[3.0, 4.0],  # Only upper channel x-boundaries
    n_refine=10,
    n_coarse=100,
    refine_width=0.05
)
z_edges = generate_refined_faces(
    coord_min=0.0,
    coord_max=5.0,
    interface_locations=[3.0],  # Only upper channel z-boundary
    n_refine=10,
    n_coarse=100,
    refine_width=0.05
)

NX = len(x_edges) - 1
NZ = len(z_edges) - 1
x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])


# ── run parameters ────────────────────────────────────────────────────────────
NTARGET    = 2000000
NBOUNDARY  = 5000000   # per active boundary side (only 1/5 of each boundary emits)
NMAX       = 20000000
MAX_EVENTS = 1000000    # cap scatter/absorption events per particle

DT_INIT = 1e-3   # ns
DT_MAX  = 5.0    # ns
DT_GROW = 1.1

OUTPUT_TIMES = [1.0,10.,100.0,501.0,700.,1000.0] # [1.0, 100.0, 500.0, 1000.0]   # ns


# ── plotting ──────────────────────────────────────────────────────────────────
def _channel_lines(ax):
    """Draw thin-channel boundary lines on an x–z axes."""
    kw = dict(color="cyan", lw=0.8, ls="--", alpha=0.7)
    ax.axhline(1.0, **kw); ax.axhline(2.0, **kw)   # lower channel x-boundaries
    ax.axhline(3.0, **kw); ax.axhline(4.0, **kw)   # upper channel x-boundaries
    ax.axvline(2.0, **kw); ax.axvline(3.0, **kw)   # z-boundaries


def plot_mesh(x_edges, z_edges):
    """Plot the computational mesh."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot all mesh lines
    for x in x_edges:
        ax.axhline(x, color='gray', lw=0.3, alpha=0.5)
    for z in z_edges:
        ax.axvline(z, color='gray', lw=0.3, alpha=0.5)
    
    # Highlight ALL material interfaces (including unrefined ones)
    # Lower channel (unrefined)
    for x_int in [1.0, 2.0]:
        ax.axhline(x_int, color='orange', lw=1.5, alpha=0.7, ls='--', label='Unrefined interface' if x_int == 1.0 else '')
    ax.axvline(2.0, color='orange', lw=1.5, alpha=0.7, ls='--')
    
    # Upper channel (refined)
    for x_int in [3.0, 4.0]:
        ax.axhline(x_int, color='red', lw=1.5, alpha=0.7, ls='--', label='Refined interface' if x_int == 3.0 else '')
    ax.axvline(3.0, color='blue', lw=1.5, alpha=0.7, ls='--')
    
    ax.set_xlabel('z (cm)', fontsize=12)
    ax.set_ylabel('x (cm)', fontsize=12)
    ax.set_title(f'Computational Mesh ({NX} × {NZ} cells)\nUpper channel refined, lower channel coarse', 
                 fontsize=13, fontweight='bold')
    ax.set_aspect('equal')
    ax.legend(fontsize=10, loc='upper right')
    
    plt.tight_layout()
    filename = 'refined_zoning_imc2d_mesh.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


def plot_material_layout(sigma_arr):
    """One-time plot showing σ_a layout."""
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    im = ax.pcolormesh(z_centers, x_centers, sigma_arr,
                       shading="auto", cmap="RdYlBu_r",
                       norm=matplotlib.colors.LogNorm(vmin=SIG_THIN, vmax=SIG_THICK))
    plt.colorbar(im, ax=ax, label=r"$\sigma_a$ (cm$^{-1}$)")
    _channel_lines(ax)
    ax.set_xlabel("z (cm)"); ax.set_ylabel("x (cm)")
    ax.set_title("Opacity layout")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig("refined_zoning_imc2d_material.png", dpi=150)
    plt.close()
    print("Saved: refined_zoning_imc2d_material.png")


def plot_snapshot(state, t, first_one=False):
    """Save material-T and radiation-T colormaps at time t."""
    T_mat = state.temperature
    T_rad = state.radiation_temperature
    
    # PLOT 1: Material temperature
    fig1, ax1 = plt.subplots(1, 1, figsize=(6, 6))
    if first_one:
        fig1, ax1 = plt.subplots(1, 1, figsize=(6, 6*1.275))
    
    im1 = ax1.pcolormesh(z_centers, x_centers, T_mat, shading='auto', cmap='plasma', vmin=0.0, vmax=0.3)
    ax1.set_xlabel('z (cm)', fontsize=15)
    ax1.set_ylabel('x (cm)', fontsize=15)
    
    if first_one:
        cbar1 = plt.colorbar(im1, ax=ax1, orientation='horizontal', location='top', pad=0.15, label='T (keV)')
    
    ax1.set_aspect('equal')
    
    plt.tight_layout()
    filename1 = f'refined_zoning_imc2d_material_t_{t:.2f}ns.png'
    if first_one:
        show(filename1, close_after=True, cbar_ax=cbar1.ax)
    else:
        show(filename1, close_after=True)
    print(f"Saved: {filename1}")
    
    # PLOT 2: Radiation temperature
    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 6))
    if first_one:
        fig2, ax2 = plt.subplots(1, 1, figsize=(6, 6*1.275))
    
    im2 = ax2.pcolormesh(z_centers, x_centers, T_rad, shading='auto', cmap='plasma', vmin=0.0, vmax=0.3)
    ax2.set_xlabel('z (cm)', fontsize=15)
    ax2.set_ylabel('x (cm)')
    
    if first_one:
        cbar2 = plt.colorbar(im2, ax=ax2, orientation='horizontal', location='top', pad=0.15, label=r'$T_\mathrm{r}$ (keV)')
    
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    filename2 = f'refined_zoning_imc2d_radiation_t_{t:.2f}ns.png'
    if first_one:
        show(filename2, close_after=True, cbar_ax=cbar2.ax)
    else:
        show(filename2, close_after=True)
    print(f"Saved: {filename2}")


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
    ax1.legend(fontsize=10, loc='best', framealpha=0.9)
    ax1.grid(True, which='both', alpha=0.3, linestyle='--')
    ax1.grid(True, which='minor', alpha=0.15, linestyle=':')
    
    plt.tight_layout()
    show('refined_zoning_imc2d_history_material.pdf', close_after=True)
    print("Saved: refined_zoning_imc2d_history_material.pdf")
    
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
    ax2.legend(fontsize=10, loc='best', framealpha=0.9)
    ax2.grid(True, which='both', alpha=0.3, linestyle='--')
    ax2.grid(True, which='minor', alpha=0.15, linestyle=':')
    
    plt.tight_layout()
    show('refined_zoning_imc2d_history_radiation.pdf', close_after=True)
    print("Saved: refined_zoning_imc2d_history_radiation.pdf")


# ── boundary source function ──────────────────────────────────────────────────
def boundary_source_func(x, z, side, current_time):
    """Position and time-dependent boundary source.
    
    Returns temperature (keV) for emission, 0.0 for no emission.
    
    Args:
        x, z: Position coordinates (cm)
        side: Boundary side ('left', 'right', 'bottom', 'top')
        current_time: Current simulation time (ns)
    
    Returns:
        Temperature (keV) for Lambertian emission
    """
    # Turn off source at t >= 500 ns
    if current_time >= T_CUTOFF:
        return 0.0
    
    # Bottom boundary (z=0): source at x ∈ [1.0, 2.0]
    if side == 'bottom':
        if 1.0 <= x <= 2.0:
            return T_BC
        else:
            return 0.0
    
    # Top boundary (z=z_max): source at x ∈ [3.0, 4.0]
    elif side == 'top':
        if 3.0 <= x <= 4.0:
            return T_BC
        else:
            return 0.0
    
    # Left/right boundaries: no emission (reflecting)
    else:
        return 0.0


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("refined_zoning_imc2d.py — non-equilibrium")
    dx_min = np.min(np.diff(x_edges))
    dx_max = np.max(np.diff(x_edges))
    dz_min = np.min(np.diff(z_edges))
    dz_max = np.max(np.diff(z_edges))
    print(f"  Δx range  : [{dx_min:.5f}, {dx_max:.5f}] cm")
    print(f"  Δz range  : [{dz_min:.5f}, {dz_max:.5f}] cm")
    print(f"  REFINEMENT: Upper channel (x=3-4, z>3) only")
    print(f"              Lower channel (x=1-2, z<2) coarse for comparison")
    print(f"  Δz range  : [{dz_min:.5f}, {dz_max:.5f}] cm")
    print(f"  σ_a       : {SIG_THICK} (thick) / {SIG_THIN} (thin) cm⁻¹")
    print(f"  ρc_v      : {CV_THICK} (thick) / {CV_THIN} (thin) GJ/(cm³·keV)")
    print(f"  T_init    : {T_INIT} keV,  T_bc : {T_BC} keV")
    print(f"  T_cutoff  : {T_CUTOFF} ns (source turns off)")
    print(f"  Bottom BC : x ∈ [1.0, 2.0] at T={T_BC} keV")
    print(f"  Top BC    : x ∈ [3.0, 4.0] at T={T_BC} keV")
    print(f"  Ntarget   : {NTARGET},  Nboundary : {NBOUNDARY}")
    print(f"  dt_max    : {DT_MAX} ns,  max_events : {MAX_EVENTS}")
    print(f"  Output    : {OUTPUT_TIMES} ns")
    print("=" * 70)

    # Build material arrays.
    sigma_arr, cv_arr = build_material_arrays(x_centers, z_centers)
    sigma_a_func, eos, inv_eos, cv_func = make_problem_funcs(sigma_arr, cv_arr)

    plot_mesh(x_edges, z_edges)
    plot_material_layout(sigma_arr)

    # Initialise state.
    Tinit  = np.full((NX, NZ), T_INIT)
    Trinit = np.full((NX, NZ), T_INIT)
    source = np.zeros((NX, NZ))

    state = imc2d.init_simulation(
        NTARGET, Tinit, Trinit,
        x_edges, z_edges,
        eos, inv_eos,
        geometry="xy",
    )

    # Boundary: reflecting on x-sides, position/time-dependent source on z-sides.
    # T_boundary values are ignored when boundary_source_func is provided
    T_boundary = (0.0, 0.0, 0.0, 0.0)       # left, right, bottom, top
    reflect    = (True, True, False, False)  # left, right, bottom, top

    # Define fiducial points for tracking
    fiducial_points = {
        'Point1: x=1.5, z=1.95': (1.5, 1.95),
        'Point2: x=1.5, z=2.05': (1.5, 2.05),
        'Point3: x=3.5, z=3.05': (3.5, 3.05),
        'Point4: x=3.5, z=2.95': (3.5, 2.95),
    }
    
    # Find indices for each fiducial point
    fiducial_indices = {}
    print("\nFiducial points:")
    for label, (x_val, z_val) in fiducial_points.items():
        i = np.argmin(np.abs(x_centers - x_val))
        j = np.argmin(np.abs(z_centers - z_val))
        fiducial_indices[label] = (i, j)
        print(f"  {label}: grid point (x={x_centers[i]:.3f}, z={z_centers[j]:.3f})")
    
    # Storage for fiducial history
    times_history = [0.0]
    fiducial_data = {label: {'T_mat': [T_INIT], 'T_rad': [T_INIT]} 
                     for label in fiducial_points.keys()}

    # Time-stepping.
    dt = DT_INIT
    output_queue = sorted(OUTPUT_TIMES)
    step_count = 0
    first_one = True

    while output_queue:
        tout = output_queue[0]

        while state.time < tout - 1e-12:
            step_dt = min(dt, tout - state.time)

            # Create boundary source function with current time closure
            def boundary_source_at_time(x, z, side):
                return boundary_source_func(x, z, side, state.time)

            state, info = imc2d.step(
                state,
                NTARGET,
                NBOUNDARY,
                0,
                NMAX,
                T_boundary,
                step_dt,
                x_edges,
                z_edges,
                sigma_a_func,
                inv_eos,
                cv_func,
                source,
                reflect=reflect,
                geometry="xy",
                max_events_per_particle=MAX_EVENTS,
                boundary_source_func=boundary_source_at_time,
            )
            step_count += 1
            dt = min(dt * DT_GROW, DT_MAX)

            # Track fiducial data
            times_history.append(state.time)
            for label, (i, j) in fiducial_indices.items():
                fiducial_data[label]['T_mat'].append(state.temperature[i, j])
                fiducial_data[label]['T_rad'].append(state.radiation_temperature[i, j])

            if step_count % 10 == 0:
                print(
                    f"  step {step_count:4d}  t = {state.time:.4e} ns  "
                    f"dt = {step_dt:.4e} ns  N = {info['N_particles']:6d}  "
                    f"E_loss = {info['energy_loss']:.3e}"
                )

        print(f"\n→ snapshot at t = {state.time:.4f} ns")
        plot_snapshot(state, state.time, first_one=first_one)
        first_one = False
        output_queue.pop(0)

    # Convert fiducial data to arrays
    times_history = np.array(times_history)
    for label in fiducial_data.keys():
        fiducial_data[label]['T_mat'] = np.array(fiducial_data[label]['T_mat'])
        fiducial_data[label]['T_rad'] = np.array(fiducial_data[label]['T_rad'])
    
    # Plot fiducial history
    print("\nPlotting fiducial point history...")
    plot_fiducial_history(times_history, fiducial_data)

    print("\nDone.")


if __name__ == "__main__":
    main()
