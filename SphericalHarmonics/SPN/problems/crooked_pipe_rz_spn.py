"""
Crooked Pipe Test Problem -- 2-D r-z SP_N version.

Same geometry, materials, mesh, and boundary conditions as the P_N version
in SphericalHarmonics/problems/crooked_pipe_rz_pn.py, but solved with the
SP_N equations (smaller moment system, no G matrix in cylindrical geometry).

Key differences from the P_N driver:
  - Uses temp_solve_spn_rz from SphericalHarmonics/SPN/spn_solver_2d_rz.py.
  - Boundary moments array has S = 3(N+1)/2 entries (SPN size) rather than
    (N+1)(N+2)/2 (full PN size).
  - No jacobian CSV files are needed; Jacobians are built analytically.

Run from the SphericalHarmonics/SPN/problems directory, or anywhere:
    python crooked_pipe_rz_spn.py
    python crooked_pipe_rz_spn.py --Nr 40 --Nz 140 --N 3 --filter-local --tfinal 1000
"""

import sys
import os
import argparse
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numba import njit

# ---- Reach solvers ----------------------------------------------------------
_HERE     = os.path.dirname(os.path.abspath(__file__))
_SPN_DIR  = os.path.dirname(_HERE)             # SPN/
_SH_DIR   = os.path.dirname(_SPN_DIR)          # SphericalHarmonics/
_ROOT_DIR = os.path.dirname(_SH_DIR)
_SN_DIR   = os.path.join(_ROOT_DIR, 'DiscreteOrdinates2D')
for _d in (_SPN_DIR, _SH_DIR, _SN_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from SphericalHarmonics.SPN.src.spn_solver_2d_rz import temp_solve_spn_rz
from SphericalHarmonics.SPN.src.spn_solver_2d import ac, build_spn_jacobians
from DiscreteOrdinates2D.src.sn_solver_2d_rz import build_rz_mesh

# ===========================================================================
# Constants (identical to the P_N driver)
# ===========================================================================

T_INIT   = 0.01   # keV
T_SOURCE = 0.3    # keV
R_SOURCE = 0.5    # cm

SIG_THICK = 200.0
SIG_THIN  = 0.2
CV_THICK  = 0.5
CV_THIN   = 0.0005

SQRT4PI = np.sqrt(4.0 * np.pi)


# ===========================================================================
# Geometry helper (shared with P_N driver)
# ===========================================================================

@njit
def _is_optically_thick(r, z):
    if r < 0.5 and (z < 3.0 or z > 4.0):
        return False
    if r < 1.5 and (z > 2.5 and z < 3.0):
        return False
    if r < 1.5 and (z > 4.0 and z < 4.5):
        return False
    if (r >= 1.0 and r < 1.5) and (z > 2.5 and z < 4.5):
        return False
    return True


# ===========================================================================
# Mesh refinement helpers
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
    if z_int >= z_right:
        return _log_one_sided(z_left, z_right, n_cells, from_left=False)
    n_left = max(1, int(n_cells * (z_int - z_left) / (z_right - z_left)))
    n_right = n_cells - n_left
    return (_log_one_sided(z_left, z_int, n_left, from_left=False) +
            _log_one_sided(z_int, z_right, n_right, from_left=True))


def generate_refined_faces(coord_min, coord_max, interface_locs,
                            n_refine, n_coarse, refine_width=0.05):
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
# Material arrays
# ===========================================================================

def build_corner_materials(Ir, Iz, dr_arr, dz_arr, r_centers, z_centers):
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


def build_local_filter_strength(Ir, Iz, dz_arr, z_centers, sigma_c):
    """Laboure-style local filter: 0.01 thick / 5 thin / 50 in thin near crook."""
    dz_off    = np.array([+0.25, +0.25, -0.25, -0.25])
    split_sig = 0.5 * (SIG_THICK + SIG_THIN)
    sigma_filter = np.empty((Ir, Iz, 4), dtype=float)
    for i in range(Ir):
        for j in range(Iz):
            for cc in range(4):
                zc = z_centers[j] + dz_off[cc] * dz_arr[j]
                if (2.5 <= zc <= 3.0) and (sigma_c[i, j, cc] < split_sig):
                    sigma_filter[i, j, cc] = 50.0
                elif sigma_c[i, j, cc] >= split_sig:
                    sigma_filter[i, j, cc] = 0.01
                else:
                    sigma_filter[i, j, cc] = 5.0
    return sigma_filter


# ===========================================================================
# Problem runner
# ===========================================================================

def setup_and_run(
    Nr=30, Nz=70,
    N_spn=3,
    use_refined_mesh=True, n_refine=10, refine_width=0.05,
    output_times=None,
    tfinal=100.0,
    dt_min=1e-4, dt_max=10.0,
    maxits=100, W=3, R=3,
    LOUD=False,
    print_stride=50,
    local_filter=False,
    uniform_filter_strength=None,
    prefix='crooked_pipe_spn',
    stability_eps=1.0,
    restart_state=None,
    checkpoint_every=10,
    restart_file=None,
):
    if output_times is None:
        output_times = [1.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    output_times = sorted(t for t in output_times if t <= tfinal)

    if use_refined_mesh:
        r_faces = generate_refined_faces(0.0, 2.0, [0.5, 1.0, 1.5], n_refine, Nr, refine_width)
        z_faces = generate_refined_faces(0.0, 7.0, [2.5, 3.0, 4.0, 4.5], n_refine, Nz, refine_width)
    else:
        r_faces = np.linspace(0.0, 2.0, Nr + 1)
        z_faces = np.linspace(0.0, 7.0, Nz + 1)

    dr_arr   = np.diff(r_faces)
    dz_arr   = np.diff(z_faces)
    Ir_act   = len(dr_arr)
    Iz_act   = len(dz_arr)
    r_centers, z_centers, _, _ = build_rz_mesh(r_faces, z_faces)

    # SP_N unknown count
    S_spn = 3 * (N_spn + 1) // 2

    print(f"\n{'='*60}")
    print(f"  Crooked Pipe r-z SP_N")
    print(f"  Coarse: {Nr}x{Nz}  ->  Actual: {Ir_act}x{Iz_act}")
    print(f"  dr: [{dr_arr.min():.4e}, {dr_arr.max():.4e}] cm")
    print(f"  dz: [{dz_arr.min():.4e}, {dz_arr.max():.4e}] cm")
    print(f"  SP_{N_spn}  (S={S_spn} unknowns/corner),  tfinal={tfinal} ns")
    print(f"{'='*60}")

    sigma_c, cv_c = build_corner_materials(
        Ir_act, Iz_act, dr_arr, dz_arr, r_centers, z_centers)

    if local_filter and uniform_filter_strength is not None:
        raise ValueError("Choose at most one filter mode.")

    sigma_filter = None
    if local_filter:
        sigma_filter = build_local_filter_strength(Ir_act, Iz_act, dz_arr, z_centers, sigma_c)
        print("  Filter: Lanczos local (0.01 thick, 5 thin, 50 for z in [2.5,3.0] thin)")
        print(f"  sigma_f range: [{sigma_filter.min():.3e}, {sigma_filter.max():.3e}]")
    elif uniform_filter_strength is not None:
        sigma_filter = np.full((Ir_act, Iz_act, 4), float(uniform_filter_strength), dtype=float)
        print(f"  Filter: Lanczos uniform sigma_f={float(uniform_filter_strength):.6g}")
    else:
        print("  Filter: off")

    def sigma_func(T): return sigma_c
    def scat_func(T):  return np.zeros_like(T)
    def EOS(T):        return cv_c * T
    def invEOS(e):     return e / cv_c

    T_init   = np.full((Ir_act, Iz_act, 4), T_INIT)
    phi_init = ac * T_init ** 4

    I_src      = ac * T_SOURCE ** 4
    src_mask_r = r_centers < R_SOURCE
    print(f"  Source BC: {int(np.count_nonzero(src_mask_r))}/{Ir_act} radial rows (r<{R_SOURCE})")

    # Boundary moments — SPN size S_spn (NOT the full PN size)
    def boundary_moments_func(t):
        zlo_bc = np.zeros((Ir_act, S_spn))
        zlo_bc[src_mask_r, 0] = I_src / SQRT4PI  # isotropic source into φ₀
        return {'rlo': None, 'rhi': None, 'zlo': zlo_bc, 'zhi': None}

    fid_points = {
        'r=0.0, z=0.25': (0.0, 0.25),
        'r=0.0, z=2.75': (0.0, 2.75),
        'r=1.25, z=3.5': (1.25, 3.5),
        'r=0.0, z=4.25': (0.0, 4.25),
        'r=0.0, z=6.75': (0.0, 6.75),
    }
    fid_indices = {}
    for label, (rv, zv) in fid_points.items():
        i = int(np.argmin(np.abs(r_centers - rv)))
        j = int(np.argmin(np.abs(z_centers - zv)))
        fid_indices[label] = (i, j)
        print(f"  Fiducial '{label}': grid (r={r_centers[i]:.3f}, z={z_centers[j]:.3f})")

    fid_times = []
    fid_Tmat  = {lab: [] for lab in fid_indices}
    fid_Trad  = {lab: [] for lab in fid_indices}

    def step_callback(t, phi, T):
        fid_times.append(float(t))
        for lab, (i, j) in fid_indices.items():
            fid_Tmat[lab].append(float(np.mean(T[i, j, :])))
            fid_Trad[lab].append(float((max(np.mean(phi[i, j, :]), 0.0) / ac) ** 0.25))

    # ---- Restart / checkpoint -----------------------------------------------
    def _save_checkpoint(checkpoint_path, step_idx, t_now_val):
        if not checkpoint_path:
            return
        payload = {
            'version': 1, 'checkpoint_step': int(step_idx),
            't_done': float(t_now_val), 'dt_start_seg': float(dt_start_seg),
            'phi_curr': phi_curr.copy(), 'T_curr': T_curr.copy(),
            'I_curr': None if I_curr is None else I_curr.copy(),
            'snapshots': snapshots, 'solutions': solutions,
            'fid_times': list(fid_times),
            'fid_Tmat': {k: list(v) for k, v in fid_Tmat.items()},
            'fid_Trad': {k: list(v) for k, v in fid_Trad.items()},
        }
        checkpoint_dir = os.path.dirname(checkpoint_path)
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
        tmp = f'{checkpoint_path}.tmp'
        with open(tmp, 'wb') as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, checkpoint_path)
        print(f'  Saved checkpoint: {checkpoint_path} (step {step_idx})')

    if restart_state is not None:
        phi_curr    = np.asarray(restart_state['phi_curr'], dtype=float).copy()
        T_curr      = np.asarray(restart_state['T_curr'],   dtype=float).copy()
        I_curr      = None if restart_state.get('I_curr') is None else np.asarray(restart_state['I_curr'], dtype=float).copy()
        t_done      = float(restart_state.get('t_done', 0.0))
        dt_start_seg = float(restart_state.get('dt_start_seg', dt_min))
        snapshots   = dict(restart_state.get('snapshots', {}))
        solutions   = dict(restart_state.get('solutions', {}))
        fid_times   = list(restart_state.get('fid_times', []))
        for lab in fid_indices:
            fid_Tmat[lab] = list(restart_state.get('fid_Tmat', {}).get(lab, []))
            fid_Trad[lab] = list(restart_state.get('fid_Trad', {}).get(lab, []))
        checkpoint_step = int(restart_state.get('checkpoint_step', 0))
        print(f"  Resuming from checkpoint at t={t_done:.4e} ns (step {checkpoint_step})")
    else:
        phi_curr = phi_init.copy()
        T_curr   = T_init.copy()
        I_curr   = None
        t_done   = 0.0
        dt_start_seg = float(dt_min)
        snapshots    = {}
        solutions    = {}
        checkpoint_step = 0
        step_callback(0.0, phi_curr, T_curr)

    checkpoint_path = restart_file if (restart_file and checkpoint_every > 0) else None

    def state_callback(t_now, dt_now, phi_now, T_now, I_now, last_hist):
        nonlocal checkpoint_step, phi_curr, T_curr, I_curr, t_done, dt_start_seg
        checkpoint_step += 1
        phi_curr   = np.asarray(phi_now, dtype=float).copy()
        T_curr     = np.asarray(T_now,   dtype=float).copy()
        I_curr     = None if I_now is None else np.asarray(I_now, dtype=float).copy()
        t_done     = float(t_now)
        dt_start_seg = float(dt_now)
        if (checkpoint_path is not None and checkpoint_every > 0
                and checkpoint_step % checkpoint_every == 0):
            _save_checkpoint(checkpoint_path, checkpoint_step, t_done)

    q_ext = np.zeros((Ir_act, Iz_act, 4))

    for t_target in output_times:
        if t_target <= t_done + 1e-14:
            continue

        duration = float(t_target - t_done)
        t_base   = t_done

        def _record_abs(t_local, phi_now, T_now):
            step_callback(t_base + float(t_local), phi_now, T_now)

        phi_curr, T_curr, I_curr, _t, history = temp_solve_spn_rz(
            Ir_act, Iz_act, dr_arr, dz_arr, r_faces, z_faces,
            q_ext, sigma_func, scat_func,
            N_spn, EOS, invEOS,
            phi_curr, T_curr,
            I_init=I_curr,
            dt_start=dt_start_seg, t_end=duration,
            reflect_rlo=False, reflect_rhi=False,
            reflect_zlo=False, reflect_zhi=False,
            tolerance=1e-5, maxits=maxits,
            W=W, n_gs=R,
            loud=LOUD, print_stride=print_stride,
            dt_max=dt_max, T_floor=1e-6,
            boundary_moments=boundary_moments_func(t_base),
            step_callback=_record_abs,
            state_callback=state_callback,
            filter_type='lanczos',
            filter_strength=sigma_filter,
            filter_exp_order=4,
            stability_eps=stability_eps,
        )

        if history:
            dt_last = float(history[-1].get('dt', dt_start_seg))
            dt_start_seg = float(np.clip(dt_last, dt_min, dt_max))

        t_done = float(t_target)
        if checkpoint_path is not None:
            _save_checkpoint(checkpoint_path, checkpoint_step, t_done)

        phi_cell   = np.mean(phi_curr, axis=2)
        T_rad_curr = (np.maximum(phi_cell, 0.0) / ac) ** 0.25
        T_mat_curr = np.mean(T_curr, axis=2)
        snapshots[t_target] = {
            'T_mat': T_mat_curr.copy(),
            'T_rad': T_rad_curr.copy(),
            't_actual': t_done,
        }
        solutions[t_target] = {
            'T': T_curr.copy(), 'phi': phi_curr.copy(), 't_actual': t_done}
        step_iters = int(sum(int(h['sweeps']) for h in history))
        print(f"  t={t_done:.2f} ns: T_max={np.max(T_curr):.4f} keV  step-its={step_iters}")

    return {
        'snapshots': snapshots, 'solutions': solutions,
        'r_centers': r_centers, 'z_centers': z_centers,
        'r_faces': r_faces, 'z_faces': z_faces,
        'Ir': Ir_act, 'Iz': Iz_act,
        'fid_times': np.asarray(fid_times),
        'fid_Tmat': {k: np.asarray(v) for k, v in fid_Tmat.items()},
        'fid_Trad': {k: np.asarray(v) for k, v in fid_Trad.items()},
        'fid_indices': fid_indices,
        'sigma_c': sigma_c,
    }


# ===========================================================================
# Plotting / saving (identical to P_N driver)
# ===========================================================================

def plot_snapshot(result, t_target, field='T_rad', save_prefix='crooked_pipe_spn',
                  vmax=None, first_one=False):
    if t_target not in result['snapshots']:
        return
    snap = result['snapshots'][t_target]
    data = snap[field]
    t_actual = snap['t_actual']
    if vmax is None:
        vmax = T_SOURCE
    height = 3.0 * (1.275 if first_one else 1.0)
    fig, ax = plt.subplots(figsize=(8, height))
    im = ax.pcolormesh(result['z_faces'], result['r_faces'], data,
                       shading='flat', cmap='plasma', vmin=0.0, vmax=vmax)
    ax.set_xlabel('z (cm)'); ax.set_ylabel('r (cm)'); ax.set_aspect('equal')
    if first_one:
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', location='top', pad=0.15)
        cbar.set_label(r'Radiation $T_r$ (keV)' if field == 'T_rad' else 'Material $T$ (keV)', fontsize=11)
    plt.tight_layout()
    fname = f'{save_prefix}_{field}_t_{t_actual:.3f}ns.png'
    plt.savefig(fname, dpi=650, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {fname}')


def plot_fiducial_history(result, save_prefix='crooked_pipe_spn'):
    ts = result['fid_times']
    if len(ts) == 0:
        return
    markers = ['o', 's', '^', 'd', 'v']
    colors  = ['C0', 'C1', 'C2', 'C3', 'C4']
    for data_dict, ylabel, tag in [
        (result['fid_Trad'], r'Radiation Temperature $T_r$ (keV)', 'fiducial_history'),
        (result['fid_Tmat'], r'Material Temperature $T$ (keV)',    'fiducial_history_Tmat'),
    ]:
        fig, ax = plt.subplots(figsize=(10, 7))
        for idx, (label, Tv) in enumerate(data_dict.items()):
            ax.loglog(ts, Tv, marker=markers[idx % 5], color=colors[idx % 5],
                      lw=2, ms=6, markevery=max(1, len(ts) // 20), label=label, alpha=0.8)
        ax.set_xlabel('Time (ns)'); ax.set_ylabel(ylabel)
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, which='both', alpha=0.3, ls='--')
        plt.tight_layout()
        fname = f'{save_prefix}_{tag}.png'
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved: {fname}')


def save_npz(result, filename):
    snap_times = sorted(result['snapshots'].keys())
    T_rad_snaps = np.stack([result['snapshots'][t]['T_rad'] for t in snap_times])
    T_mat_snaps = np.stack([result['snapshots'][t]['T_mat'] for t in snap_times])
    t_actuals   = np.array([result['snapshots'][t]['t_actual'] for t in snap_times])
    labels      = list(result['fid_Trad'].keys())
    np.savez_compressed(
        filename,
        r_centers=result['r_centers'], z_centers=result['z_centers'],
        r_faces=result['r_faces'],     z_faces=result['z_faces'],
        T_rad_snaps=T_rad_snaps, T_mat_snaps=T_mat_snaps, snap_times=t_actuals,
        fid_times=result['fid_times'],
        fid_Trad=np.column_stack([result['fid_Trad'][l] for l in labels]),
        fid_Tmat=np.column_stack([result['fid_Tmat'][l] for l in labels]),
        fid_labels=np.asarray(labels),
    )
    print(f'  Saved: {filename}')


def load_restart_checkpoint(filename):
    with open(filename, 'rb') as fh:
        payload = pickle.load(fh)
    if payload.get('version') != 1:
        raise ValueError(f"Unsupported restart version: {payload.get('version')}")
    return payload


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description='Crooked Pipe r-z SP_N')
    parser.add_argument('--Nr', type=int, default=40)
    parser.add_argument('--Nz', type=int, default=140)
    parser.add_argument('--N',  type=int, default=3, help='SP_N order (must be odd)')
    parser.add_argument('--tfinal', type=float, default=100.0)
    parser.add_argument('--dt-min', type=float, default=1e-4)
    parser.add_argument('--dt-max', type=float, default=10.0)
    parser.add_argument('--output-times', type=float, nargs='+', default=None)
    parser.add_argument('--no-refine', action='store_true')
    parser.add_argument('--n-refine', type=int, default=10)
    parser.add_argument('--refine-width', type=float, default=0.05,
                        help='Width of refinement zone around interfaces (default 0.05).')
    parser.add_argument('--loud', action='store_true')
    parser.add_argument('--prefix', type=str, default='crooked_pipe_spn')
    parser.add_argument('--filter-local', action='store_true',
                        help='Laboure-style local Lanczos filter.')
    parser.add_argument('--filter-uniform', type=float, default=None,
                        help='Uniform Lanczos filter strength.')
    parser.add_argument('--stability-eps', type=float, default=1.0,
                        help='Cross-direction dissipation for cylindrical stability (default 1.0).')
    parser.add_argument('--restart-in', type=str, default=None)
    parser.add_argument('--restart-file', type=str, default=None)
    parser.add_argument('--checkpoint-every', type=int, default=10)
    args = parser.parse_args()

    if args.filter_local and args.filter_uniform is not None:
        parser.error('Use either --filter-local or --filter-uniform.')

    out_times = args.output_times or [
        t for t in [1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
        if t <= args.tfinal]

    restart_state = None
    if args.restart_in is not None:
        restart_state = load_restart_checkpoint(args.restart_in)

    restart_file = args.restart_file or f'{args.prefix}.restart.pkl'

    result = setup_and_run(
        Nr=args.Nr, Nz=args.Nz, N_spn=args.N,
        use_refined_mesh=(not args.no_refine), n_refine=args.n_refine,
        refine_width=args.refine_width,
        output_times=out_times, tfinal=args.tfinal,
        dt_min=args.dt_min, dt_max=args.dt_max,
        LOUD=args.loud, prefix=args.prefix,
        local_filter=args.filter_local,
        uniform_filter_strength=args.filter_uniform,
        stability_eps=args.stability_eps,
        restart_state=restart_state,
        checkpoint_every=args.checkpoint_every,
        restart_file=restart_file,
    )

    print('\nPost-processing...')
    first = True
    for t_tgt in sorted(result['snapshots'].keys()):
        plot_snapshot(result, t_tgt, field='T_rad', save_prefix=args.prefix, first_one=first)
        plot_snapshot(result, t_tgt, field='T_mat', save_prefix=args.prefix, first_one=False)
        first = False
    plot_fiducial_history(result, save_prefix=args.prefix)
    save_npz(result, f'{args.prefix}_{result["Ir"]}x{result["Iz"]}.npz')


if __name__ == '__main__':
    main()
