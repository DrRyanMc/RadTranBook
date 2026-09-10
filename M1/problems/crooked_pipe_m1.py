"""
Crooked Pipe Test Problem — 2-D r-z M1 closure version.

Matching DiscreteOrdinates2D/problems/crooked_pipe_rz_sn.py.

Geometry: r ∈ [0, 2] cm, z ∈ [0, 7] cm (cylindrical r-z).
Material:
  Optically thick: σ = 200 cm⁻¹, c_v = 0.5  GJ/(cm³·keV)
  Optically thin:  σ = 0.2  cm⁻¹, c_v = 0.0005 GJ/(cm³·keV)
BCs:
  r = 0:      axis (reflecting)
  r = r_max:  vacuum (Marshak)
  z = 0:      source T = 0.3 keV for r < 0.5 cm (Marshak),
              vacuum (Marshak) for r ≥ 0.5 cm
  z = z_max:  vacuum (Marshak)

Uses M1SolverSCB2D (Simple-Corner-Balance discretisation) by default with
geometry='cylindrical', Fleck–Cummings coupling, and the Levermore /
Kershaw / P1 closures for comparison.  SCB gives sub-cell resolution
without adding numerical dissipation at the sub-cell scale, which is
substantially more accurate than the plain first-order FV M1Solver2D in
the optically thick regions of this problem (σ=200 cm⁻¹).  Pass
``--no-scb`` to fall back to the plain M1Solver2D (faster, ~4x fewer
cells, but more diffusive in the thick regions).

Run from M1/problems/:
    python crooked_pipe_m1.py
    python crooked_pipe_m1.py --Nr 40 --Nz 140 --tfinal 100
    python crooked_pipe_m1.py --Nr 30 --Nz 70 --no-refine --tfinal 100
    python crooked_pipe_m1.py --no-scb --tfinal 100
    python crooked_pipe_m1.py --load crooked_pipe_m1_levermore.npz --prefix out
"""

import sys
import os
import argparse
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── path setup ──────────────────────────────────────────────────────────────
_this_dir   = os.path.dirname(os.path.abspath(__file__))
_m1_dir     = os.path.dirname(_this_dir)          # M1/
_repo_root  = os.path.dirname(_m1_dir)            # RadTranBook/
sys.path.insert(0, _m1_dir)
sys.path.insert(0, _repo_root)

from M1.src.m1_2d import (
    M1Solver2D,
    closure_levermore,
    closure_kershaw,
    closure_minerbo_poly,
    closure_p1,
    C_LIGHT,
    A_RAD,
)
from M1.src.m1_2d_scb import M1SolverSCB2D, _corner_faces

# ===========================================================================
# Constants
# ===========================================================================

T_INIT   = 0.01   # keV — cold background
T_SOURCE = 0.3    # keV — source temperature at z = 0, r < R_SOURCE cm
R_SOURCE = 0.5    # cm  — source radius

SIG_THICK = 200.0
SIG_THIN  =   0.2
CV_THICK  =   0.5
CV_THIN   =   0.0005

# A·c = A_RAD * C_LIGHT  (converts T^4 energy density to photon flux intensity)
AC = A_RAD * C_LIGHT   # GJ/(cm³·keV⁴) × cm/ns  = GJ/(cm²·ns·keV⁴)


# ===========================================================================
# Material geometry (identical to SN/diffusion versions)
# ===========================================================================

def _is_optically_thick(r, z):
    """Crooked-pipe material map — returns True for optically thick region."""
    # lower thin channel: r < 0.5, z ∉ [3, 4]
    if r < 0.5 and (z < 3.0 or z > 4.0):
        return False
    # ascending thin segment (lower)
    if r < 1.5 and (z > 2.5 and z < 3.0):
        return False
    # ascending thin segment (upper)
    if r < 1.5 and (z > 4.0 and z < 4.5):
        return False
    # upper thin channel
    if (1.0 <= r < 1.5) and (2.5 < z < 4.5):
        return False
    return True


def build_cell_materials(r_centers, z_centers):
    """Return sigma and cv arrays at cell centres, shape (Nr, Nz)."""
    Nr = len(r_centers)
    Nz = len(z_centers)
    sigma = np.empty((Nr, Nz))
    cv    = np.empty((Nr, Nz))
    for i in range(Nr):
        for j in range(Nz):
            if _is_optically_thick(r_centers[i], z_centers[j]):
                sigma[i, j] = SIG_THICK
                cv[i, j]    = CV_THICK
            else:
                sigma[i, j] = SIG_THIN
                cv[i, j]    = CV_THIN
    return sigma, cv


# ===========================================================================
# Mesh refinement helpers (identical to the SN version)
# ===========================================================================

def _log_one_sided(z_start, z_end, n_cells, from_left=True, max_ratio=5.0):
    if n_cells <= 1:
        return [z_end]
    width = z_end - z_start
    r = max_ratio ** (1.0 / (n_cells - 1))
    if abs(r - 1.0) < 1e-10:
        widths = [width / n_cells] * n_cells
    else:
        w0 = width * (r - 1.0) / (r**n_cells - 1.0)
        widths = [w0 * r**i for i in range(n_cells)]
    if not from_left:
        widths = widths[::-1]
    faces = []
    pos = z_start
    for w in widths:
        pos += w
        faces.append(pos)
    faces[-1] = z_end
    return faces


def _log_around_interface(z_left, z_right, z_int, n_cells):
    if z_int <= z_left:
        return _log_one_sided(z_left, z_right, n_cells, from_left=True)
    elif z_int >= z_right:
        return _log_one_sided(z_left, z_right, n_cells, from_left=False)
    n_left  = max(1, int(n_cells * (z_int - z_left) / (z_right - z_left)))
    n_right = n_cells - n_left
    return (_log_one_sided(z_left, z_int, n_left,  from_left=False) +
            _log_one_sided(z_int, z_right, n_right, from_left=True))


def generate_refined_faces(coord_min, coord_max, interface_locs,
                            n_refine, n_coarse, refine_width=0.05):
    """Log-refined face array with clustering at material interfaces."""
    coarse = np.linspace(coord_min, coord_max, n_coarse + 1)
    refine_info = {}
    for z_int in sorted(interface_locs):
        for i in range(n_coarse):
            cl, cr = coarse[i], coarse[i + 1]
            if cl <= z_int + refine_width and cr >= z_int - refine_width:
                mid = (cl + cr) / 2
                if i not in refine_info or abs(z_int - mid) < abs(refine_info[i] - mid):
                    refine_info[i] = z_int
    faces = [coord_min]
    for i in range(n_coarse):
        cl, cr = coarse[i], coarse[i + 1]
        if i in refine_info:
            faces.extend(_log_around_interface(cl, cr, refine_info[i], n_refine))
        else:
            faces.append(cr)
    return np.array(faces)


# ===========================================================================
# Problem runner
# ===========================================================================

def setup_and_run(
    Nr=20, Nz=70,
    closure_func=closure_levermore,
    closure_name='levermore',
    output_times=None,
    tfinal=100.0,
    dt_initial=1e-3,
    dt_max=10.0,
    dt_increase_factor=1.1,
    use_refined_mesh=True,
    n_refine=8,
    refine_width=0.05,
    use_scb=True,
    max_nonlinear_iter=3,
    LOUD=False,
    print_stride=5000,
):
    """Set up and run the crooked pipe M1 problem.

    Parameters
    ----------
    Nr, Nz : int
        Coarse cell counts before refinement (or total cells if
        ``use_refined_mesh=False``).
        Default 20×70 gives ~3 min/closure at tfinal=100 ns.
        For publication quality use 40×140 (~25 min/closure).
    closure_func : callable
        M1 Eddington-factor closure χ(f).
    closure_name : str
        Short label for the closure (used in printouts).
    output_times : list or None
        Times at which to save snapshots.
    tfinal : float
        Final time (ns).
    dt_initial : float
        Initial time step (ns).  Same time-step progression as the
        diffusion version (EqDiffusion/problems/crooked_pipe_test.py):
        start at ``dt_initial``, grow geometrically by
        ``dt_increase_factor`` each step, capped at ``dt_max``.  The M1
        solver's symmetric Gauss-Seidel (SGS) scheme (see
        m1_2d.M1Solver2D) is unconditionally stable, so this progression
        is chosen purely for time-accuracy, not stability.
    dt_max : float
        Maximum time step (ns).
    dt_increase_factor : float
        Factor by which dt is multiplied after each step (e.g. 1.1 for a
        10% increase), capped at ``dt_max``.
    use_refined_mesh : bool
        Log-refine the mesh near material interfaces (default True).
    n_refine : int
        Sub-cells per coarse interval in refinement zones.
    refine_width : float
        Half-width (cm) of each refinement zone.
    use_scb : bool
        Use ``M1SolverSCB2D`` (Simple-Corner-Balance discretisation,
        default) instead of the plain first-order FV ``M1Solver2D``.
        SCB internally doubles the resolution (2× in each direction) by
        splitting each cell into 4 corner sub-cells with dissipation-free
        internal coupling, giving much better accuracy in the optically
        thick regions of this problem without needing a finer coarse
        mesh.  Costs ~4x more per time step than ``use_scb=False`` at the
        same ``Nr``/``Nz``.
    max_nonlinear_iter : int
        Forward+backward symmetric Gauss-Seidel (SGS) sweep pairs per step
        (default 3; the SGS scheme is unconditionally stable regardless
        of this value, more sweeps just improve nonlinear-closure
        convergence within the step).
    LOUD : bool
        Verbose output each step.
    print_stride : int
        Print progress every this many steps.

    Returns
    -------
    result : dict
        snapshots, mesh, fiducial histories.
    """
    if output_times is None:
        output_times = [1.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    output_times = sorted(t for t in output_times if t <= tfinal)

    # ── Mesh ────────────────────────────────────────────────────────────────
    if use_refined_mesh:
        r_faces = generate_refined_faces(
            0.0, 2.0, [0.5, 1.0, 1.5], n_refine, Nr, refine_width)
        z_faces = generate_refined_faces(
            0.0, 7.0, [2.5, 3.0, 4.0, 4.5], n_refine, Nz, refine_width)
    else:
        r_faces = np.linspace(0.0, 2.0, Nr + 1)
        z_faces = np.linspace(0.0, 7.0, Nz + 1)

    Nr_act = len(r_faces) - 1
    Nz_act = len(z_faces) - 1
    dr_arr = np.diff(r_faces)
    dz_arr = np.diff(z_faces)
    r_centers = 0.5 * (r_faces[1:] + r_faces[:-1])
    z_centers = 0.5 * (z_faces[1:] + z_faces[:-1])

    # SCB doubles resolution internally (splitting each cell into 4 corner
    # sub-cells), so material arrays must be built on that finer grid.
    if use_scb:
        mat_r_faces, _ = _corner_faces(r_faces)
        mat_z_faces, _ = _corner_faces(z_faces)
    else:
        mat_r_faces, mat_z_faces = r_faces, z_faces
    mat_r_centers = 0.5 * (mat_r_faces[1:] + mat_r_faces[:-1])
    mat_z_centers = 0.5 * (mat_z_faces[1:] + mat_z_faces[:-1])

    # ── Time step (same progression as the diffusion version: start small,
    #    grow geometrically, cap at dt_max; SGS is unconditionally stable so
    #    this controls time-accuracy, not stability) ───────────────────
    dt = dt_initial

    print(f"\n{'='*60}")
    print(f"  Crooked Pipe r-z M1  [{closure_name}]"
          f"  {'[SCB]' if use_scb else '[plain FV]'}")
    mesh_tag = 'refined' if use_refined_mesh else 'uniform'
    print(f"  Coarse: {Nr}×{Nz}  →  Actual ({mesh_tag}): {Nr_act}×{Nz_act}")
    if use_scb:
        print(f"  SCB corner grid: {2*Nr_act}×{2*Nz_act}")
    print(f"  dr: [{dr_arr.min():.4e}, {dr_arr.max():.4e}] cm")
    print(f"  dz: [{dz_arr.min():.4e}, {dz_arr.max():.4e}] cm")
    print(f"  dt: [{dt_initial:.4e}, {dt_max:.4e}] ns  (×{dt_increase_factor} growth)"
          f"   tfinal = {tfinal} ns")
    print(f"{'='*60}")

    # ── Material arrays (built on the actual data grid: doubled corner
    #    grid for SCB, coarse grid otherwise) ─────────────────────────
    sigma_arr, cv_arr = build_cell_materials(mat_r_centers, mat_z_centers)

    def sigma_func(T):
        return sigma_arr

    def scat_func(T):
        return np.zeros_like(T)

    def EOS(T):
        return cv_arr * T

    def invEOS(e):
        return e / cv_arr

    # ── Source BC at z = 0 (mixed source/vacuum, r-dependent) ──────────────
    c = C_LIGHT
    Er_src   = A_RAD * T_SOURCE**4          # energy density of source blackbody
    F2_src   = +0.5 * c * Er_src           # upward half-space (Marshak source)
    source_mask = mat_r_centers < R_SOURCE  # shape matches the data grid

    def bc_z_lo(Er_edge, F1_edge, F2_edge):
        """Callable BC for z=0: source for r<R_SOURCE, vacuum elsewhere."""
        Er_ghost = np.where(source_mask, Er_src,  Er_edge)
        F1_ghost = F1_edge
        F2_ghost = np.where(source_mask, F2_src,  -0.5 * c * Er_edge)
        return Er_ghost, F1_ghost, F2_ghost

    # ── Build solver ─────────────────────────────────────────────────────────
    SolverCls = M1SolverSCB2D if use_scb else M1Solver2D
    solver = SolverCls(
        x1_min=0.0, x1_max=2.0, n1=Nr_act,
        x2_min=0.0, x2_max=7.0, n2=Nz_act,
        geometry='cylindrical',
        sigma_func=sigma_func,
        scat_func=scat_func,
        EOS=EOS,
        invEOS=invEOS,
        dt=dt,
        closure_func=closure_func,
        bc_x1_lo='reflect',   # axis at r = 0
        bc_x1_hi='marshak',   # vacuum at r = 2 cm
        bc_x2_lo='callable',  # mixed source/vacuum at z = 0
        bc_x2_hi='marshak',   # vacuum at z = 7 cm
        bc_x2_lo_state=bc_z_lo,
        max_nonlinear_iter=max_nonlinear_iter,
        x1_faces=r_faces,
        x2_faces=z_faces,
    )
    solver.initialize(T_INIT)

    # ── Fiducial point tracking (against the actual solver grid: doubled
    #    corner grid for SCB, coarse grid otherwise) ─────────────────────
    fid_points = {
        'r=0.0, z=0.25':  (0.0,  0.25),
        'r=0.0, z=2.75':  (0.0,  2.75),
        'r=1.25, z=3.5':  (1.25, 3.5),
        'r=0.0, z=4.25':  (0.0,  4.25),
        'r=0.0, z=6.75':  (0.0,  6.75),
    }
    fid_indices = {}
    for label, (rv, zv) in fid_points.items():
        i = int(np.argmin(np.abs(solver.x1_c - rv)))
        j = int(np.argmin(np.abs(solver.x2_c - zv)))
        fid_indices[label] = (i, j)
        print(f"  Fiducial '{label}': "
              f"grid (r={solver.x1_c[i]:.3f}, z={solver.x2_c[j]:.3f})")

    fid_times = []
    fid_Tmat  = {lab: [] for lab in fid_indices}
    fid_Trad  = {lab: [] for lab in fid_indices}

    # ── Snapshot storage ─────────────────────────────────────────────────────
    snapshots = {}
    output_times_saved = set()

    t = 0.0
    solver.dt = dt_initial
    n_steps = 0
    t_start = time.time()
    out_tol = 1e-9 * max(tfinal, 1.0)

    # ── Time loop: same progression as the diffusion version ────────────────
    #    (EqDiffusion/problems/crooked_pipe_test.py) — start at dt_initial,
    #    shrink to land exactly on output times / tfinal, then grow
    #    geometrically by dt_increase_factor (capped at dt_max) for the
    #    next step.
    while t < tfinal - out_tol:
        n_steps += 1

        dt_current = solver.dt
        hit_output_time = False
        for t_tgt in output_times:
            if t_tgt > t and t + solver.dt > t_tgt:
                solver.dt = t_tgt - t
                hit_output_time = True
                break
        if t + solver.dt > tfinal:
            solver.dt = tfinal - t

        solver.step(verbose=LOUD)
        t += solver.dt

        # Record fiducials
        fid_times.append(float(t))
        for lab, (i, j) in fid_indices.items():
            fid_Tmat[lab].append(float(solver.T[i, j]))
            Er_ij = solver.Er[i, j]
            fid_Trad[lab].append(float((max(Er_ij, 0.0) / A_RAD)**0.25))

        # Save snapshot if we reached an output time
        for t_tgt in output_times:
            if t_tgt not in output_times_saved and abs(t - t_tgt) < out_tol:
                T_rad_snap = (np.maximum(solver.Er, 0.0) / A_RAD)**0.25
                snapshots[t_tgt] = {
                    'T_mat':    solver.T.copy(),
                    'T_rad':    T_rad_snap,
                    'Er':       solver.Er.copy(),
                    'F1':       solver.F1.copy(),
                    'F2':       solver.F2.copy(),
                    't_actual': float(t),
                }
                output_times_saved.add(t_tgt)
                print(f"  Snapshot saved: t = {t:.4f} ns  (target {t_tgt} ns)")

        # Restore the pre-output-snap dt, then grow it for the next step.
        if hit_output_time:
            solver.dt = dt_current
        solver.dt = min(solver.dt * dt_increase_factor, dt_max)

        if n_steps % print_stride == 0:
            elapsed = time.time() - t_start
            frac    = t / tfinal
            eta     = elapsed / frac - elapsed if frac > 0 else 0
            print(f"  step {n_steps:8d}  t = {t:.4f}/{tfinal} ns  "
                  f"dt = {dt_current:.3e} ns  "
                  f"elapsed {elapsed:.1f}s  ETA {eta:.0f}s")

    elapsed = time.time() - t_start
    print(f"\n  Done: {n_steps} steps in {elapsed:.1f} s  ({elapsed/n_steps*1e3:.2f} ms/step)")

    return {
        'snapshots':    snapshots,
        'r_centers':    solver.x1_c,
        'z_centers':    solver.x2_c,
        'r_faces':      solver.x1_faces,
        'z_faces':      solver.x2_faces,
        'Nr': solver.n1, 'Nz': solver.n2,
        'fid_times':    np.asarray(fid_times),
        'fid_Tmat':     {k: np.asarray(v) for k, v in fid_Tmat.items()},
        'fid_Trad':     {k: np.asarray(v) for k, v in fid_Trad.items()},
        'fid_indices':  fid_indices,
        'sigma_arr':    sigma_arr,
        'closure_name': closure_name,
        'use_scb':      use_scb,
    }


# ===========================================================================
# Plotting
# ===========================================================================

def plot_material(result, savefile='crooked_pipe_m1_materials.png'):
    """Plot opacity layout."""
    r_c  = result['r_centers']
    z_c  = result['z_centers']
    sig  = result['sigma_arr']

    fig, ax = plt.subplots(figsize=(8, 3))
    im = ax.pcolormesh(z_c, r_c, sig, shading='auto', cmap='RdYlBu_r',
                       norm=matplotlib.colors.LogNorm(vmin=SIG_THIN, vmax=SIG_THICK))
    ax.set_xlabel('z (cm)'); ax.set_ylabel('r (cm)')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label=r'$\sigma_a$ (cm$^{-1}$)')
    plt.tight_layout()
    plt.savefig(savefile, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {savefile}')


def plot_snapshot(result, t_target, field='T_rad', save_prefix='crooked_pipe_m1',
                  vmax=None, first_one=False):
    """Plot a single r-z colormap snapshot (T_mat or T_rad)."""
    if t_target not in result['snapshots']:
        return

    snap     = result['snapshots'][t_target]
    r_f      = result['r_faces']
    z_f      = result['z_faces']
    data     = snap[field]
    t_actual = snap['t_actual']

    if vmax is None:
        vmax = T_SOURCE

    height = 3.0 * (1.275 if first_one else 1.0)
    fig, ax = plt.subplots(figsize=(8, height))

    im = ax.pcolormesh(z_f, r_f, data, shading='flat',
                       cmap='plasma', vmin=0.0, vmax=vmax)
    ax.set_xlabel('z (cm)')
    ax.set_ylabel('r (cm)')
    ax.set_aspect('equal')

    if first_one:
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal',
                            location='top', pad=0.15)
        label_txt = (r'Radiation $T_r$ (keV)' if field == 'T_rad'
                     else r'Material $T$ (keV)')
        cbar.set_label(label_txt, fontsize=11)

    plt.tight_layout()
    fname = f'{save_prefix}_{field}_t_{t_actual:.3f}ns.png'
    plt.savefig(fname, dpi=650, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {fname}')
    return fname


def plot_cross_sections(result, t_target, save_prefix='crooked_pipe_m1'):
    """Plot radial and axial T_rad cross-sections at t ≈ t_target."""
    if t_target not in result['snapshots']:
        return

    snap     = result['snapshots'][t_target]
    T_rad    = snap['T_rad']
    r_c      = result['r_centers']
    z_c      = result['z_centers']
    t_actual = snap['t_actual']

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = ['C0', 'C1', 'C2', 'C3']

    # Radial profiles at z ≈ 0.5, 2.0, 3.5, 5.0 cm
    ax = axes[0]
    for z_val, col in zip([0.5, 2.0, 3.5, 5.0], colors):
        j = int(np.argmin(np.abs(z_c - z_val)))
        ax.semilogy(r_c, T_rad[:, j], color=col, lw=2, label=f'z = {z_c[j]:.2f}')
    ax.set_xlabel('r (cm)'); ax.set_ylabel(r'$T_r$ (keV)')
    ax.set_title(f'Radial profiles  t = {t_actual:.1f} ns')
    ax.legend(fontsize=9); ax.grid(True, which='both', alpha=0.3)

    # Axial profiles at r ≈ 0.25, 0.75, 1.25, 1.75 cm
    ax = axes[1]
    for r_val, col in zip([0.25, 0.75, 1.25, 1.75], colors):
        i = int(np.argmin(np.abs(r_c - r_val)))
        ax.semilogy(z_c, T_rad[i, :], color=col, lw=2, label=f'r = {r_c[i]:.2f}')
    ax.set_xlabel('z (cm)'); ax.set_ylabel(r'$T_r$ (keV)')
    ax.set_title(f'Axial profiles  t = {t_actual:.1f} ns')
    ax.legend(fontsize=9); ax.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    fname = f'{save_prefix}_cross_sections_t_{t_actual:.1f}ns.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {fname}')


def plot_fiducial_history(result, save_prefix='crooked_pipe_m1'):
    """Plot fiducial-point temperature history (log-log) for T_rad and T_mat."""
    ts       = result['fid_times']
    fid_Trad = result['fid_Trad']
    fid_Tmat = result['fid_Tmat']

    if len(ts) == 0:
        return

    markers = ['o', 's', '^', 'd', 'v']
    colors  = ['C0', 'C1', 'C2', 'C3', 'C4']

    for data_dict, ylabel, tag in [
        (fid_Trad, r'Radiation Temperature $T_r$ (keV)', 'fiducial_history'),
        (fid_Tmat, r'Material Temperature $T$ (keV)',    'fiducial_history_Tmat'),
    ]:
        fig, ax = plt.subplots(figsize=(10, 7))
        for idx, (label, Tv) in enumerate(data_dict.items()):
            ax.loglog(ts, Tv,
                      marker=markers[idx % 5], color=colors[idx % 5],
                      lw=2, ms=6, markevery=max(1, len(ts) // 20),
                      label=label, alpha=0.8)
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, which='both', alpha=0.3, ls='--')
        plt.tight_layout()
        fname = f'{save_prefix}_{tag}.png'
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved: {fname}')


def plot_closure_comparison_fiducials(results_dict, save_prefix='crooked_pipe_m1'):
    """Overlay fiducial histories from multiple closures on one figure."""
    markers  = ['o', 's', '^', 'd', 'v']
    fid_keys = None
    ts_ref   = None

    # collect all fiducial labels from the first result
    for res in results_dict.values():
        fid_keys = list(res['fid_Trad'].keys())
        break

    for field_tag, ylabel in [('fid_Trad', r'Radiation Temperature $T_r$ (keV)'),
                               ('fid_Tmat', r'Material Temperature $T$ (keV)')]:
        fig, axes = plt.subplots(len(fid_keys), 1,
                                 figsize=(10, 3.5 * len(fid_keys)),
                                 sharex=True)
        if len(fid_keys) == 1:
            axes = [axes]

        for ax, fid_label in zip(axes, fid_keys):
            for k_idx, (cname, res) in enumerate(results_dict.items()):
                ts  = res['fid_times']
                Tv  = res[field_tag][fid_label]
                ax.loglog(ts, Tv,
                          marker=markers[k_idx % 5], lw=2,
                          ms=5, markevery=max(1, len(ts)//20),
                          label=cname, alpha=0.85)
            ax.set_ylabel(r'$T$ (keV)', fontsize=9)
            ax.set_title(fid_label, fontsize=9)
            ax.grid(True, which='both', alpha=0.3, ls='--')
            ax.legend(fontsize=8, loc='best')

        axes[-1].set_xlabel('Time (ns)')
        fig.suptitle(f'Crooked Pipe M1 — {ylabel}', y=1.01)
        plt.tight_layout()
        fname = f'{save_prefix}_closure_compare_{field_tag}.png'
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved: {fname}')


# ===========================================================================
# NPZ save / load
# ===========================================================================

def save_npz(result, filename):
    """Save solution, mesh, and fiducial histories to an NPZ file."""
    snap_times  = sorted(result['snapshots'].keys())
    T_rad_snaps = np.stack([result['snapshots'][t]['T_rad'] for t in snap_times])
    T_mat_snaps = np.stack([result['snapshots'][t]['T_mat'] for t in snap_times])
    t_actuals   = np.array([result['snapshots'][t]['t_actual'] for t in snap_times])

    labels       = list(result['fid_Trad'].keys())
    fid_Trad_arr = np.column_stack([result['fid_Trad'][l] for l in labels])
    fid_Tmat_arr = np.column_stack([result['fid_Tmat'][l] for l in labels])

    np.savez_compressed(
        filename,
        r_centers    = result['r_centers'],
        z_centers    = result['z_centers'],
        r_faces      = result['r_faces'],
        z_faces      = result['z_faces'],
        T_rad_snaps  = T_rad_snaps,
        T_mat_snaps  = T_mat_snaps,
        snap_times   = t_actuals,
        fid_times    = result['fid_times'],
        fid_Trad     = fid_Trad_arr,
        fid_Tmat     = fid_Tmat_arr,
        fid_labels   = np.asarray(labels),
        closure_name = np.array(result.get('closure_name', 'unknown')),
        sigma_arr    = result['sigma_arr'],
    )
    print(f'  Saved NPZ: {filename}')


def load_npz(filename):
    """Load a previously saved NPZ result into the result dict format."""
    d = np.load(filename, allow_pickle=True)
    fid_labels = list(d['fid_labels'])
    snap_times = list(d['snap_times'])

    snapshots = {}
    for k, t_act in enumerate(snap_times):
        snapshots[float(d['snap_times'][k])] = {
            'T_rad':    d['T_rad_snaps'][k],
            'T_mat':    d['T_mat_snaps'][k],
            't_actual': float(t_act),
        }

    fid_Trad = {lab: d['fid_Trad'][:, i] for i, lab in enumerate(fid_labels)}
    fid_Tmat = {lab: d['fid_Tmat'][:, i] for i, lab in enumerate(fid_labels)}

    return {
        'snapshots':    snapshots,
        'r_centers':    d['r_centers'],
        'z_centers':    d['z_centers'],
        'r_faces':      d['r_faces'],
        'z_faces':      d['z_faces'],
        'Nr':           len(d['r_centers']),
        'Nz':           len(d['z_centers']),
        'fid_times':    d['fid_times'],
        'fid_Trad':     fid_Trad,
        'fid_Tmat':     fid_Tmat,
        'sigma_arr':    d['sigma_arr'],
        'closure_name': str(d['closure_name']),
    }


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description='Crooked Pipe r-z M1')
    parser.add_argument('--Nr',     type=int,   default=40,
                        help='Coarse radial cells (default 40)')
    parser.add_argument('--Nz',     type=int,   default=140,
                        help='Coarse axial cells (default 140)')
    parser.add_argument('--tfinal', type=float, default=1000.0)
    parser.add_argument('--dt-initial', type=float, default=1e-3,
                        help='Initial time step in ns (default 1e-3, same '
                             'progression as the diffusion version)')
    parser.add_argument('--dt-max',     type=float, default=10.0,
                        help='Maximum time step in ns (default 10.0)')
    parser.add_argument('--dt-increase-factor', type=float, default=1.1,
                        help='Per-step dt growth factor (default 1.1)')
    parser.add_argument('--output-times', type=float, nargs='+', default=None)
    parser.add_argument('--no-refine',    action='store_true',
                        help='Use uniform mesh (skip log-refinement at interfaces)')
    parser.add_argument('--n-refine',     type=int,   default=10,
                        help='Sub-cells per coarse interval in refinement zones (default 8)')
    parser.add_argument('--refine-width', type=float, default=0.05,
                        help='Half-width (cm) of each refinement zone (default 0.05)')
    parser.add_argument('--no-scb', action='store_true',
                        help='Use the plain first-order FV M1Solver2D instead of '
                             'the default M1SolverSCB2D (Simple-Corner-Balance). '
                             'Faster (~4x fewer cells) but more diffusive in the '
                             'optically thick regions.')
    parser.add_argument('--closures', type=str, nargs='+',
                        default=['levermore', 'kershaw', 'p1'],
                        help='Closures to run: levermore kershaw p1')
    parser.add_argument('--load',   type=str, nargs='+', default=None,
                        help='Load NPZ file(s) instead of running. '
                             'Provide one file per closure (same order as --closures).')
    parser.add_argument('--prefix', type=str, default='crooked_pipe_m1')
    parser.add_argument('--loud',   action='store_true')
    parser.add_argument('--print-stride', type=int, default=5)
    args = parser.parse_args()

    closure_map = {
        'levermore': closure_levermore,
        'kershaw':   closure_kershaw,
        'p1':        closure_p1,
        'minerbo_poly': closure_minerbo_poly,
    }

    out_times = args.output_times
    if out_times is None:
        out_times = [t for t in [1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.]
                     if t <= args.tfinal]

    results = {}

    if args.load:
        # ── Load from cached NPZ files ──────────────────────────────────────
        for fname, cname in zip(args.load, args.closures):
            print(f'  Loading {fname} as closure={cname}...')
            results[cname] = load_npz(fname)
            results[cname]['closure_name'] = cname
    else:
        # ── Run each closure ─────────────────────────────────────────────────
        for cname in args.closures:
            if cname not in closure_map:
                print(f'  Unknown closure "{cname}", skipping.')
                continue
            result = setup_and_run(
                Nr=args.Nr, Nz=args.Nz,
                closure_func=closure_map[cname],
                closure_name=cname,
                output_times=out_times,
                tfinal=args.tfinal,
                dt_initial=args.dt_initial,
                dt_max=args.dt_max,
                dt_increase_factor=args.dt_increase_factor,
                use_refined_mesh=(not args.no_refine),
                n_refine=args.n_refine,
                refine_width=args.refine_width,
                use_scb=(not args.no_scb),
                LOUD=args.loud,
                print_stride=args.print_stride,
            )
            results[cname] = result
            # Save NPZ immediately after each run
            npz_fname = f'{args.prefix}_{cname}_{result["Nr"]}x{result["Nz"]}.npz'
            save_npz(result, npz_fname)

    print('\nPost-processing...')

    # ── Material map (use first result) ──────────────────────────────────────
    first_result = next(iter(results.values()))
    plot_material(first_result, savefile=f'{args.prefix}_materials.png')

    # ── Snapshots for each closure ───────────────────────────────────────────
    for cname, result in results.items():
        pfx = f'{args.prefix}_{cname}'
        first_one = True
        for t_tgt in sorted(result['snapshots'].keys()):
            plot_snapshot(result, t_tgt, field='T_rad',
                          save_prefix=pfx, first_one=first_one)
            plot_snapshot(result, t_tgt, field='T_mat',
                          save_prefix=pfx, first_one=False)
            plot_cross_sections(result, t_tgt, save_prefix=pfx)
            first_one = False
        plot_fiducial_history(result, save_prefix=pfx)

    # ── Closure comparison (if multiple closures) ─────────────────────────────
    if len(results) > 1:
        plot_closure_comparison_fiducials(results, save_prefix=args.prefix)


if __name__ == '__main__':
    main()
