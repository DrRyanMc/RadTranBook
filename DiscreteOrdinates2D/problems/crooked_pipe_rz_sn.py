import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

"""
Crooked Pipe Test Problem — 2-D r-z Discrete Ordinates S_N version.

Matching the diffusion version in EqDiffusion/problems/crooked_pipe_test.py.

Geometry: r ∈ [0, 2] cm, z ∈ [0, 7] cm (cylindrical r-z).
Material:
  Optically thick: σ = 200 cm⁻¹, c_v = 0.5 GJ/(cm³·keV)
  Optically thin:  σ = 0.2  cm⁻¹, c_v = 0.0005 GJ/(cm³·keV)
BCs:
  r = 0:      axis (handled automatically)
  r = r_max:  vacuum
  z = 0:      source T = 0.3 keV for r < 0.5 cm, vacuum elsewhere
  z = z_max:  vacuum

Mesh: same log-refined mesh as the diffusion version, with logarithmic
      clustering at the material interfaces z = 2.5, 3.0, 4.0, 4.5 cm
      and r = 0.5, 1.0, 1.5 cm.

Run from DiscreteOrdinates2D/problems directory:
    python crooked_pipe_rz_sn.py
    python crooked_pipe_rz_sn.py --Nr 30 --Nz 70 --N 4 --tfinal 100
    python crooked_pipe_rz_sn.py --no-refine --tfinal 10
"""

import sys
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numba import njit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
_this_dir = os.path.dirname(os.path.abspath(__file__))
_solver_dir = os.path.dirname(_this_dir)   # DiscreteOrdinates2D/
sys.path.insert(0, _solver_dir)

from DiscreteOrdinates2D.src.sn_solver_2d_rz import temp_solve_rz, build_rz_mesh, ac

# ===========================================================================
# Constants
# ===========================================================================

T_INIT   = 0.01   # keV — cold background
T_SOURCE = 0.3    # keV — source temperature at z = 0, r < 0.5 cm
R_SOURCE = 0.5    # cm  — source radius

SIG_THICK = 200.0
SIG_THIN  = 0.2
CV_THICK  = 0.5
CV_THIN   = 0.0005


# ===========================================================================
# Geometry helper (matches diffusion version exactly)
# ===========================================================================

@njit
def _is_optically_thick(r, z):
    """Crooked-pipe material map — same logic as the diffusion version."""
    # lower thin channel: r < 0.5, z ∈ [3, 4)
    if r < 0.5 and (z < 3.0 or z > 4.0):
        return False
    # ascending thin segment (lower)
    if r < 1.5 and (z > 2.5 and z < 3.0):
        return False
    # ascending thin segment (upper)
    if r < 1.5 and (z > 4.0 and z < 4.5):
        return False
    # upper thin channel
    if (r >= 1.0 and r < 1.5) and (z > 2.5 and z < 4.5):
        return False
    return True


# ===========================================================================
# Mesh refinement helpers (identical to the diffusion version)
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
# Position-dependent material arrays
# ===========================================================================

def build_corner_materials(Ir, Iz, dr_arr, dz_arr, r_centers, z_centers):
    """Build (Ir, Iz, 4) opacity and heat-capacity arrays at corner positions."""
    dr_off = np.array([+0.25, -0.25, -0.25, +0.25])
    dz_off = np.array([+0.25, +0.25, -0.25, -0.25])

    sigma_c = np.empty((Ir, Iz, 4))
    cv_c    = np.empty((Ir, Iz, 4))

    for i in range(Ir):
        for j in range(Iz):
            for cc in range(4):
                rc = max(r_centers[i] + dr_off[cc] * dr_arr[i], 0.0)
                zc = z_centers[j] + dz_off[cc] * dz_arr[j]
                if _is_optically_thick(rc, zc):
                    sigma_c[i, j, cc] = SIG_THICK
                    cv_c[i, j, cc]    = CV_THICK
                else:
                    sigma_c[i, j, cc] = SIG_THIN
                    cv_c[i, j, cc]    = CV_THIN

    return sigma_c, cv_c


# ===========================================================================
# Problem runner
# ===========================================================================

def setup_and_run(
    Nr=30, Nz=70,
    N_quad=4, quad_type='product_square',
    use_refined_mesh=True, n_refine=10, refine_width=0.05,
    output_times=None,
    tfinal=100.0,
    dt_min=1e-3, dt_max=5.0,
    K=10, maxits=100, W=3, R=3,
    use_dmd=True, LOUD=False,
    print_stride=50,
):
    """Set up and run the crooked pipe S_N problem.

    Parameters
    ----------
    Nr, Nz : int
        Coarse cell counts before refinement.
    N_quad : int
        S_N quadrature order.
    quad_type : str
        'product_square' (recommended for r-z) or other types.
    use_refined_mesh : bool
        Apply log-refinement near material interfaces.
    n_refine : int
        Sub-cells per coarse cell in refinement zones.
    refine_width : float
        Width (cm) of the refinement zone around each interface.
    output_times : list or None
        Snapshot times (ns).
    tfinal : float
        Final time (ns).
    dt_min, dt_max : float
        Time step bounds.
    K, maxits, W, R : int
        DMD / source-iteration parameters.

    Returns
    -------
    result : dict
    """
    if output_times is None:
        output_times = [1.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    output_times = sorted(t for t in output_times if t <= tfinal)

    # --- Mesh ---
    if use_refined_mesh:
        r_faces = generate_refined_faces(
            0.0, 2.0, [0.5, 1.0, 1.5], n_refine, Nr, refine_width)
        z_faces = generate_refined_faces(
            0.0, 7.0, [2.5, 3.0, 4.0, 4.5], n_refine, Nz, refine_width)
    else:
        r_faces = np.linspace(0.0, 2.0, Nr + 1)
        z_faces = np.linspace(0.0, 7.0, Nz + 1)

    dr_arr = np.diff(r_faces)
    dz_arr = np.diff(z_faces)
    Ir_act = len(dr_arr)
    Iz_act = len(dz_arr)
    r_centers, z_centers, _, _ = build_rz_mesh(r_faces, z_faces)

    print(f"\n{'='*60}")
    print(f"  Crooked Pipe r-z S_N")
    print(f"  Coarse: {Nr}×{Nz}  →  Actual: {Ir_act}×{Iz_act}")
    print(f"  dr: [{dr_arr.min():.4e}, {dr_arr.max():.4e}] cm")
    print(f"  dz: [{dz_arr.min():.4e}, {dz_arr.max():.4e}] cm")
    print(f"  {quad_type} N={N_quad},  tfinal={tfinal} ns")
    print(f"{'='*60}")

    # --- Material properties ---
    sigma_c, cv_c = build_corner_materials(
        Ir_act, Iz_act, dr_arr, dz_arr, r_centers, z_centers)

    def sigma_func(T):
        return sigma_c

    def scat_func(T):
        return np.zeros_like(T)

    def EOS(T):
        return cv_c * T

    def invEOS(e):
        return e / cv_c

    # --- Initial condition ---
    T_init  = np.full((Ir_act, Iz_act, 4), T_INIT)
    phi_init = ac * T_init**4

    # --- Boundary conditions ---
    # r = 0    : axis (handled by on_axis flag, no BC needed)
    # r = r_max: vacuum
    # z = 0    : blackbody source T_SOURCE for r < R_SOURCE; vacuum elsewhere
    # z = z_max: vacuum
    I_src = ac * T_SOURCE**4   # incoming intensity at source (Marshak-consistent)

    def BCs_func(t):
        BCs_zlo = np.zeros((Ir_act, 2))
        for i in range(Ir_act):
            if r_centers[i] < R_SOURCE:
                BCs_zlo[i, 0] = I_src
                BCs_zlo[i, 1] = I_src
        return {
            'rlo': None,
            'rhi': None,
            'zlo': BCs_zlo,
            'zhi': None,
        }

    # --- Fiducial point tracking ---
    fid_points = {
        'r=0.0, z=0.25':  (0.0,  0.25),
        'r=0.0, z=2.75':  (0.0,  2.75),
        'r=1.25, z=3.5':  (1.25, 3.5),
        'r=0.0, z=4.25':  (0.0,  4.25),
        'r=0.0, z=6.75':  (0.0,  6.75),
    }
    fid_indices = {}
    for label, (rv, zv) in fid_points.items():
        i = int(np.argmin(np.abs(r_centers - rv)))
        j = int(np.argmin(np.abs(z_centers - zv)))
        fid_indices[label] = (i, j)
        print(f"  Fiducial '{label}': "
              f"grid (r={r_centers[i]:.3f}, z={z_centers[j]:.3f})")

    fid_times = []
    fid_Tmat  = {lab: [] for lab in fid_indices}
    fid_Trad  = {lab: [] for lab in fid_indices}

    def step_callback(t, phi, T):
        fid_times.append(float(t))
        for lab, (i, j) in fid_indices.items():
            fid_Tmat[lab].append(float(np.mean(T[i, j, :])))
            fid_Trad[lab].append(float((max(np.mean(phi[i, j, :]), 0.0) / ac)**0.25))

    q_ext = np.zeros((Ir_act, Iz_act, 4))

    # --- Run ---
    phis, Ts, iters, ts, its_per_step = temp_solve_rz(
        Ir_act, Iz_act, dr_arr, dz_arr, r_faces, z_faces,
        q_ext, sigma_func, scat_func,
        quad_type, N_quad,
        BCs_func, EOS, invEOS,
        phi_init, T_init,
        dt_min=dt_min, dt_max=dt_max, tfinal=tfinal,
        tolerance=1e-5, Linf_tol=1e-3, maxits=maxits,
        K=K, R=R, W=W,
        reflect_rlo=False,
        reflect_rhi=False,
        reflect_zlo=False,
        reflect_zhi=False,
        use_dmd=use_dmd,
        LOUD=LOUD,
        print_stride=print_stride,
        time_outputs=np.asarray(output_times),
        store_full_history=False,
        step_callback=step_callback,
    )

    print(f"\n  Finished: {iters} sweeps, {len(its_per_step)} steps")

    # --- Assemble snapshots ---
    snapshots = {}
    for t_tgt in output_times:
        idx = int(np.argmin(np.abs(ts - t_tgt)))
        if idx < len(phis):
            phi_s   = phis[idx]
            T_mat_s = Ts[idx]
            phi_cell = np.mean(phi_s, axis=2)
            T_rad_s  = (np.maximum(phi_cell, 0.0) / ac)**0.25
            T_mat_cell = np.mean(T_mat_s, axis=2)
            snapshots[t_tgt] = {
                'T_mat': T_mat_cell,
                'T_rad': T_rad_s,
                't_actual': float(ts[idx]),
            }

    return {
        'snapshots': snapshots,
        'r_centers': r_centers,
        'z_centers': z_centers,
        'r_faces':   r_faces,
        'z_faces':   z_faces,
        'Ir': Ir_act, 'Iz': Iz_act,
        'fid_times': np.asarray(fid_times),
        'fid_Tmat':  {k: np.asarray(v) for k, v in fid_Tmat.items()},
        'fid_Trad':  {k: np.asarray(v) for k, v in fid_Trad.items()},
        'fid_indices': fid_indices,
        'sigma_c': sigma_c,
        'its_per_step': its_per_step,
    }


# ===========================================================================
# Plotting
# ===========================================================================

def plot_material(result, savefile='crooked_pipe_sn_materials.png'):
    """Plot opacity layout."""
    r_c  = result['r_centers']
    z_c  = result['z_centers']
    sig  = np.mean(result['sigma_c'], axis=2)

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


def plot_snapshot(result, t_target, field='T_rad', save_prefix='crooked_pipe_sn',
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
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', location='top', pad=0.15)
        label_txt = (r'Radiation $T_r$ (keV)' if field == 'T_rad'
                     else 'Material $T$ (keV)')
        cbar.set_label(label_txt, fontsize=11)
    else:
        cbar = None

    plt.tight_layout()
    fname = f'{save_prefix}_{field}_t_{t_actual:.3f}ns.png'
    plt.savefig(fname, dpi=650, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {fname}')
    return fname


def plot_cross_sections(result, t_target, save_prefix='crooked_pipe_sn'):
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

    # Radial profiles at z = 0.5, 2.0, 3.5, 5.0 cm
    ax = axes[0]
    for z_val, col in zip([0.5, 2.0, 3.5, 5.0], colors):
        j = int(np.argmin(np.abs(z_c - z_val)))
        ax.semilogy(r_c, T_rad[:, j], color=col, lw=2, label=f'z = {z_c[j]:.2f}')
    ax.set_xlabel('r (cm)'); ax.set_ylabel(r'$T_r$ (keV)')
    ax.set_title(f'Radial profiles  t = {t_actual:.1f} ns')
    ax.legend(fontsize=9); ax.grid(True, which='both', alpha=0.3)

    # Axial profiles at r = 0.25, 0.75, 1.25, 1.75 cm
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


def plot_fiducial_history(result, save_prefix='crooked_pipe_sn'):
    """Plot fiducial-point temperature history (log-log) for both T_rad and T_mat."""
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


# ===========================================================================
# NPZ saving
# ===========================================================================

def save_npz(result, filename):
    """Save solution, mesh, and fiducial histories to an NPZ file."""
    Ir = result['Ir']; Iz = result['Iz']

    snap_times = sorted(result['snapshots'].keys())
    T_rad_snaps = np.stack([result['snapshots'][t]['T_rad'] for t in snap_times])
    T_mat_snaps = np.stack([result['snapshots'][t]['T_mat'] for t in snap_times])
    t_actuals   = np.array([result['snapshots'][t]['t_actual'] for t in snap_times])

    labels = list(result['fid_Trad'].keys())
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
    )
    print(f'  Saved: {filename}')


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description='Crooked Pipe r-z S_N')
    parser.add_argument('--Nr',     type=int,   default=30,
                        help='Coarse radial cells (default 30)')
    parser.add_argument('--Nz',     type=int,   default=70,
                        help='Coarse axial cells (default 70)')
    parser.add_argument('--N',      type=int,   default=4,
                        help='S_N order (default 4)')
    parser.add_argument('--quad',   type=str,   default='product_square')
    parser.add_argument('--tfinal', type=float, default=100.0)
    parser.add_argument('--dt-min', type=float, default=1e-4)
    parser.add_argument('--dt-max', type=float, default=10.0)
    parser.add_argument('--output-times', type=float, nargs='+',
                        default=None)
    parser.add_argument('--no-refine', action='store_true',
                        help='Use uniform mesh instead of log-refined')
    parser.add_argument('--n-refine',  type=int, default=10)
    parser.add_argument('--no-dmd',    action='store_true')
    parser.add_argument('--loud',      action='store_true')
    parser.add_argument('--prefix',    type=str, default='crooked_pipe_sn')
    args = parser.parse_args()

    out_times = args.output_times
    if out_times is None:
        out_times = [t for t in
                     [1.0, 5.0, 10.0, 20.0, 50.0, 100.0,
                      200.0, 500.0, 1000.0]
                     if t <= args.tfinal]

    result = setup_and_run(
        Nr=args.Nr, Nz=args.Nz,
        N_quad=args.N, quad_type=args.quad,
        use_refined_mesh=(not args.no_refine),
        n_refine=args.n_refine,
        output_times=out_times,
        tfinal=args.tfinal,
        dt_min=args.dt_min, dt_max=args.dt_max,
        use_dmd=(not args.no_dmd),
        LOUD=args.loud,
    )

    print('\nPost-processing...')

    # Material layout
    plot_material(result, savefile=f'{args.prefix}_materials.png')

    # Snapshots
    first_one = True
    for t_tgt in sorted(result['snapshots'].keys()):
        plot_snapshot(result, t_tgt, field='T_rad',
                      save_prefix=args.prefix, first_one=first_one)
        plot_snapshot(result, t_tgt, field='T_mat',
                      save_prefix=args.prefix, first_one=False)
        plot_cross_sections(result, t_tgt, save_prefix=args.prefix)
        first_one = False

    # Fiducial history
    plot_fiducial_history(result, save_prefix=args.prefix)

    # Save
    npz_file = (f'{args.prefix}_{result["Ir"]}x{result["Iz"]}.npz')
    save_npz(result, npz_file)


if __name__ == '__main__':
    main()
