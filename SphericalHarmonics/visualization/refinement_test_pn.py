"""
2-D Refinement Test - P_N corner-balance solver.

Pn analogue of DiscreteOrdinates2D/problems/refinement_test.py.

Geometry: 2-D Cartesian, domain [0,5]x[0,5] cm

Materials (temperature-independent, position-dependent):
  Thick (default):   sigma_a = 200 cm^-1,  c_v = 0.5  GJ/(cm^3*keV)
  Lower thin channel x in [1,2], y < 2:    sigma_a = 0.2 cm^-1, c_v = 0.05
  Upper thin channel x in [3,4], y > 3:    sigma_a = 0.2 cm^-1, c_v = 0.05

BCs:
  x=0, x=5: reflecting
  y=0, y=5: vacuum in transport operator

Driving source (Pn analogue of incoming blackbody strips):
    bottom and top ghost-cell moments are isotropic with the 00 moment matching
    an isotropic source at T_bc=0.3 keV, switched off at t>=500 ns.

Run from DiscreteOrdinates2D directory:
    python problems/refinement_test_pn.py --Npn 3
    python problems/refinement_test_pn.py --Npn 7
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(os.path.dirname(_here))
_pn_dir = os.path.join(_root, 'SphericalHarmonics')
if _pn_dir not in sys.path:
    sys.path.insert(0, _pn_dir)

from SphericalHarmonics.src.pn_solver_2d import temp_solve_pn_2d, ac

try:
    sys.path.insert(0, os.path.join(_root, 'utils'))
    from plotfuncs import hide_spines as _hide_spines, font as _font
    HAS_PLOTFUNCS = True
except Exception:
    HAS_PLOTFUNCS = False

JACOBIAN_DIR = os.path.join(_pn_dir, 'Jacobians')

# ===========================================================================
# Material constants
# ===========================================================================
SIG_THICK = 200.0   # cm^-1
SIG_THIN = 0.2      # cm^-1
CV_THICK = 0.5      # GJ/(cm^3*keV)
CV_THIN = 0.05      # GJ/(cm^3*keV)

T_INIT = 0.01       # keV
T_BC = 0.3          # keV
T_CUTOFF = 500.0    # ns


# ===========================================================================
# Mesh generation with log-spaced interface refinement
# ===========================================================================

def _log_spacing_one_sided(z_start, z_end, n_cells, from_left=True):
    if n_cells <= 1:
        return [z_end]
    width = z_end - z_start
    max_ratio = 5.0
    r = max_ratio ** (1.0 / (n_cells - 1))
    if abs(r - 1.0) < 1e-10:
        cell_widths = [width / n_cells] * n_cells
    else:
        w0 = width * (r - 1.0) / (r**n_cells - 1.0)
        cell_widths = [w0 * r**i for i in range(n_cells)]
    if not from_left:
        cell_widths = cell_widths[::-1]
    faces = []
    pos = z_start
    for w in cell_widths:
        pos += w
        faces.append(pos)
    faces[-1] = z_end
    return faces


def _log_spacing_around_interface(z_left, z_right, z_int, n_cells):
    if z_int <= z_left:
        return _log_spacing_one_sided(z_left, z_right, n_cells, True)
    if z_int >= z_right:
        return _log_spacing_one_sided(z_left, z_right, n_cells, False)
    n_left = max(1, int(n_cells * (z_int - z_left) / (z_right - z_left)))
    n_right = n_cells - n_left
    return (_log_spacing_one_sided(z_left, z_int, n_left, False) +
            _log_spacing_one_sided(z_int, z_right, n_right, True))


def generate_refined_faces(coord_min, coord_max, interface_locations,
                           n_refine, n_coarse, refine_width=0.05):
    """Face positions with log refinement at material interfaces."""
    coarse_faces = np.linspace(coord_min, coord_max, n_coarse + 1)
    interfaces = sorted(interface_locations)
    refine_info = {}
    for coord_int in interfaces:
        for i in range(n_coarse):
            cl, cr = coarse_faces[i], coarse_faces[i + 1]
            if cl <= coord_int + refine_width and cr >= coord_int - refine_width:
                mid = 0.5 * (cl + cr)
                if i not in refine_info or abs(coord_int - mid) < abs(refine_info[i] - mid):
                    refine_info[i] = coord_int
    faces_list = [coord_min]
    for i in range(n_coarse):
        cl, cr = coarse_faces[i], coarse_faces[i + 1]
        if i in refine_info:
            faces_list.extend(_log_spacing_around_interface(cl, cr, refine_info[i], n_refine))
        else:
            faces_list.append(cr)
    return np.array(faces_list)


def generate_mirror_log_faces(seg_left, seg_right, n_cells, max_ratio=5.0):
    """Return mirror-symmetric log-spaced faces on [seg_left, seg_right]."""
    if n_cells <= 0:
        raise ValueError('n_cells must be positive')

    L = seg_right - seg_left
    if L <= 0.0:
        raise ValueError('segment must satisfy seg_right > seg_left')

    if n_cells == 1:
        return np.array([seg_left, seg_right])

    half_L = 0.5 * L
    r = max_ratio ** (1.0 / max(1, n_cells // 2))

    if abs(r - 1.0) < 1e-12:
        return np.linspace(seg_left, seg_right, n_cells + 1)

    if n_cells % 2 == 0:
        n_side = n_cells // 2
        sum_side = (r**n_side - 1.0) / (r - 1.0)
        w0 = half_L / sum_side
        side = np.array([w0 * r**k for k in range(n_side)])
        widths = np.concatenate([side, side[::-1]])
    else:
        n_side = n_cells // 2
        sum_side = (r**n_side - 1.0) / (r - 1.0) if n_side > 0 else 0.0
        w0 = half_L / (sum_side + 0.5 * r**n_side)
        side = np.array([w0 * r**k for k in range(n_side)]) if n_side > 0 else np.array([])
        w_mid = w0 * r**n_side
        widths = np.concatenate([side, np.array([w_mid]), side[::-1]])

    faces = np.empty(n_cells + 1)
    faces[0] = seg_left
    for i in range(n_cells):
        faces[i + 1] = faces[i] + widths[i]
    faces[-1] = seg_right
    return faces


# ===========================================================================
# Geometry helpers
# ===========================================================================

def is_thin(x, y):
    """True where cells are in a thin channel."""
    lower = (x >= 1.0) & (x <= 2.0) & (y < 2.0)
    upper = (x >= 3.0) & (x <= 4.0) & (y > 3.0)
    return lower | upper


def build_corner_materials(Ix, Iy, dx_arr, dy_arr):
    """Build (Ix, Iy, 4) sigma and cv arrays at corner positions."""
    x_faces = np.cumsum(np.concatenate([[0.0], dx_arr]))
    y_faces = np.cumsum(np.concatenate([[0.0], dy_arr]))
    x_centers = 0.5 * (x_faces[:-1] + x_faces[1:])
    y_centers = 0.5 * (y_faces[:-1] + y_faces[1:])

    sigma_corners = np.full((Ix, Iy, 4), SIG_THICK)
    cv_corners = np.full((Ix, Iy, 4), CV_THICK)

    for i in range(Ix):
        for j in range(Iy):
            for cc in range(4):
                if cc == 0:    # NE
                    xc = x_centers[i] + dx_arr[i] / 4
                    yc = y_centers[j] + dy_arr[j] / 4
                elif cc == 1:  # NW
                    xc = x_centers[i] - dx_arr[i] / 4
                    yc = y_centers[j] + dy_arr[j] / 4
                elif cc == 2:  # SW
                    xc = x_centers[i] - dx_arr[i] / 4
                    yc = y_centers[j] - dy_arr[j] / 4
                else:          # SE
                    xc = x_centers[i] + dx_arr[i] / 4
                    yc = y_centers[j] - dy_arr[j] / 4

                if is_thin(np.array([xc]), np.array([yc]))[0]:
                    sigma_corners[i, j, cc] = SIG_THIN
                    cv_corners[i, j, cc] = CV_THIN

    return sigma_corners, cv_corners


# ===========================================================================
# Problem setup
# ===========================================================================

def _build_boundary_moments_on(Ix, Iy, x_centers, N_pn):
    """Build isotropic ghost-cell moments used while t < T_CUTOFF.

    The ghost cell is taken to be isotropic, with only the 00 moment set.
    The value corresponds to the 00 moment of an isotropic source at T_BC,
    i.e. I_00 = (a c T_BC^4) / sqrt(4 pi).
    """
    S = (N_pn + 1) * (N_pn + 2) // 2
    boundary = {
        'xlo': np.zeros((Iy, S)),
        'xhi': np.zeros((Iy, S)),
        'ylo': np.zeros((Ix, S)),
        'yhi': np.zeros((Ix, S)),
    }
    source_00 = ac * T_BC**4 / np.sqrt(4.0 * np.pi)

    for i in range(Ix):
        if 1.0 <= x_centers[i] <= 2.0:
            boundary['ylo'][i, 0] = source_00
        if 3.0 <= x_centers[i] <= 4.0:
            boundary['yhi'][i, 0] = source_00

    return boundary
def _zero_boundary_moments(Ix, Iy, N_pn):
    S = (N_pn + 1) * (N_pn + 2) // 2
    return {
        'xlo': np.zeros((Iy, S)),
        'xhi': np.zeros((Iy, S)),
        'ylo': np.zeros((Ix, S)),
        'yhi': np.zeros((Ix, S)),
    }


def setup_and_run(Ix=100, Iy=100, N_pn=3,
                  output_times=(1.0, 10.0, 100.0, 501.0, 700.0, 1000.0),
                  LOUD=False,
                  n_refine=10, refine_width=0.05,
                  dt_start=1e-3, dt_max=10.0,
                  tolerance=1e-5, maxits=100,
                  n_gs=1, use_gmres=False):
    """Run the refinement test problem with the P_N solver."""

    Lx, Ly = 5.0, 5.0

    x_edges_base = generate_refined_faces(0.0, Lx, [3.0, 4.0],
                                          n_refine, Ix, refine_width)
    in_upper_channel = ((x_edges_base[:-1] >= 3.0 - 1e-12) &
                        (x_edges_base[1:] <= 4.0 + 1e-12))
    n_cells_upper = int(np.sum(in_upper_channel))
    x_seg = generate_mirror_log_faces(3.0, 4.0, n_cells_upper)

    left = x_edges_base[x_edges_base < 3.0 - 1e-12]
    right = x_edges_base[x_edges_base > 4.0 + 1e-12]
    x_edges = np.concatenate([left, x_seg, right])
    y_edges = generate_refined_faces(0.0, Ly, [3.0],
                                     n_refine, Iy, refine_width)

    dx_arr = np.diff(x_edges)
    dy_arr = np.diff(y_edges)
    Ix_actual = len(dx_arr)
    Iy_actual = len(dy_arr)
    x_faces = x_edges
    y_faces = y_edges
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    sigma_corners, cv_corners = build_corner_materials(Ix_actual, Iy_actual, dx_arr, dy_arr)

    print(f"Refinement Test P_{N_pn}: Ix={Ix_actual} (coarse {Ix}), Iy={Iy_actual} (coarse {Iy})")
    print(f"  dx range: [{dx_arr.min():.5f}, {dx_arr.max():.5f}]")
    print(f"  dy range: [{dy_arr.min():.5f}, {dy_arr.max():.5f}]")
    print(f"  sigma: {SIG_THICK} (thick) / {SIG_THIN} (thin)")
    print(f"  cv: {CV_THICK} (thick) / {CV_THIN} (thin)")
    print(f"  T_init={T_INIT}, T_bc={T_BC}, T_cutoff={T_CUTOFF}")

    def sigma_func(T):
        return sigma_corners

    def scat_func(T):
        return np.zeros_like(T)

    def EOS(T):
        return cv_corners * T

    def invEOS(e):
        return e / cv_corners

    T_curr = np.full((Ix_actual, Iy_actual, 4), T_INIT)
    phi_curr = ac * T_curr**4
    I_curr = None

    bc_on = _build_boundary_moments_on(Ix_actual, Iy_actual, x_centers, N_pn)
    bc_off = _zero_boundary_moments(Ix_actual, Iy_actual, N_pn)

    output_times = np.array(sorted(output_times), dtype=float)

    fid_points = {
        'Point 1 x=1.5, y=1.95': (1.5, 1.95),
        'Point 2 x=1.5, y=2.05': (1.5, 2.05),
        'Point 3 x=3.5, y=3.05': (3.5, 3.05),
        'Point 4 x=3.5, y=2.95': (3.5, 2.95),
    }
    fid_indices = {}
    for label, (xv, yv) in fid_points.items():
        i = int(np.argmin(np.abs(x_centers - xv)))
        j = int(np.argmin(np.abs(y_centers - yv)))
        fid_indices[label] = (i, j)

    fid_times = []
    fid_T_mat = {label: [] for label in fid_indices}
    fid_T_rad = {label: [] for label in fid_indices}

    def _record_fiducials(t_abs, phi_now, T_now):
        fid_times.append(float(t_abs))
        for label, (i, j) in fid_indices.items():
            fid_T_mat[label].append(float(np.mean(T_now[i, j, :])))
            fid_T_rad[label].append(float((np.mean(phi_now[i, j, :]) / ac) ** 0.25))

    solutions = {}
    _record_fiducials(0.0, phi_curr, T_curr)
    total_inner_iterations = 0
    t_done = 0.0

    for t_target in output_times:
        if t_target <= t_done + 1e-14:
            continue

        while t_done < t_target - 1e-14:
            if t_done < T_CUTOFF < t_target:
                t_next = min(t_target, T_CUTOFF)
            else:
                t_next = t_target
            duration = float(t_next - t_done)

            bc_seg = bc_on if t_done < T_CUTOFF else bc_off
            t_base = t_done

            def _step_callback_abs(t_local, phi_now, T_now):
                _record_fiducials(t_base + float(t_local), phi_now, T_now)

            phi_curr, T_curr, I_curr, _t_now, history = temp_solve_pn_2d(
                Ix_actual,
                Iy_actual,
                dx_arr,
                dy_arr,
                np.zeros((Ix_actual, Iy_actual, 4)),
                sigma_func,
                scat_func,
                N_pn,
                JACOBIAN_DIR,
                EOS,
                invEOS,
                phi_curr,
                T_curr,
                I_init=I_curr,
                dt_start=dt_start,
                t_end=duration,
                reflect_xlo=True,
                reflect_xhi=True,
                reflect_ylo=False,
                reflect_yhi=False,
                tolerance=tolerance,
                maxits=maxits,
                W=0,
                n_gs=n_gs,
                use_gmres=use_gmres,
                loud=LOUD,
                print_stride=20,
                dt_max=dt_max,
                T_floor=1e-6,
                step_callback=_step_callback_abs,
                boundary_moments=bc_seg,
            )

            step_iters = int(sum(int(h['sweeps']) for h in history))
            total_inner_iterations += step_iters
            t_done = float(t_next)

        solutions[t_target] = {
            'T': T_curr.copy(),
            'phi': phi_curr.copy(),
            't_actual': float(t_done),
        }
        print(f"  t={t_done:.2f} ns: T_max={np.max(T_curr):.4f} keV")

    print(f"\nTotal source iterations: {total_inner_iterations}, steps: {len(fid_times) - 1}")

    fid_data = {}
    for label in fid_indices:
        fid_data[label] = {
            'T_mat': np.array(fid_T_mat[label]),
            'T_rad': np.array(fid_T_rad[label]),
        }

    return {
        'solutions': solutions,
        'x_centers': x_centers,
        'y_centers': y_centers,
        'x_faces': x_faces,
        'y_faces': y_faces,
        'Ix': Ix_actual,
        'Iy': Iy_actual,
        'ts': np.array(fid_times),
        'fid_data': fid_data,
        'fid_indices': fid_indices,
        'sigma_corners': sigma_corners,
        'N_pn': int(N_pn),
    }


# ===========================================================================
# Plotting
# ===========================================================================

def _channel_lines(ax):
    """Draw thin-channel boundary lines."""
    kw = dict(color='cyan', lw=0.8, ls='--', alpha=0.7)
    ax.axhline(1.0, **kw)
    ax.axhline(2.0, **kw)
    ax.axhline(3.0, **kw)
    ax.axhline(4.0, **kw)
    ax.axvline(2.0, **kw)
    ax.axvline(3.0, **kw)


def plot_material(results, savefile='refinement_test_pn_material.png'):
    """Plot opacity layout."""
    sig = np.mean(results['sigma_corners'], axis=2)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    im = ax.pcolormesh(results['y_centers'], results['x_centers'], sig,
                       shading='auto', cmap='RdYlBu_r',
                       norm=matplotlib.colors.LogNorm(vmin=SIG_THIN, vmax=SIG_THICK))
    plt.colorbar(im, ax=ax, label=r'$\sigma_a$ (cm$^{-1}$)')
    _channel_lines(ax)
    ax.set_xlabel('y (cm)')
    ax.set_ylabel('x (cm)')
    ax.set_title('Opacity layout')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(savefile, dpi=150, bbox_inches='tight')
    print(f'Saved: {savefile}')
    plt.close()


def plot_snapshot(results, t_target, savefile_prefix='refinement_test_pn',
                  first_one=False):
    """Plot material T and radiation T at a given time.

    Matches the style of plot_colormap_at_time from the diffusion code:
    - fixed colour scale [0, T_BC]
    - horizontal colorbar at the top only on the first figure produced
    - figure slightly taller on that first call to accommodate the colorbar
    """
    sol = results['solutions'][t_target]
    x_f = results['x_faces']
    y_f = results['y_faces']

    T_cell = np.mean(sol['T'], axis=2)
    Tr_cell = (np.mean(sol['phi'], axis=2) / ac)**0.25

    is_first = first_one

    for field, label, data in [('material', 'T (keV)', T_cell),
                               ('radiation', r'$T_r$ (keV)', Tr_cell)]:
        height = 6.0 * (1.275 if is_first else 1.0)
        fig, ax = plt.subplots(figsize=(6, height))

        im = ax.pcolormesh(y_f, x_f, data, shading='flat', cmap='plasma',
                           vmin=0.0, vmax=T_BC)
        ax.set_xlabel('y (cm)')
        ax.set_ylabel('x (cm)')
        ax.set_aspect('equal')

        if is_first:
            cbar = plt.colorbar(im, ax=ax, orientation='horizontal',
                                location='top', pad=0.15)
            cbar.set_label(label, fontsize=11)

        plt.tight_layout()
        fname = f'{savefile_prefix}_{field}_t_{t_target:.0f}ns.png'
        plt.savefig(fname, dpi=650, bbox_inches='tight')
        print(f'Saved: {fname}')
        plt.close()


def plot_fiducial_history(results, savefile_prefix='refinement_test_pn'):
    """Plot temperature history at fiducial points."""
    ts = results['ts']
    fid_data = results['fid_data']
    markers = ['o', 's', '^', 'd']
    colors = ['blue', 'red', 'green', 'purple']

    fp = _font if HAS_PLOTFUNCS else None

    for field in ['T_mat', 'T_rad']:
        fig, ax = plt.subplots(figsize=(8, 5))
        for idx, (label, data) in enumerate(fid_data.items()):
            ax.loglog(ts, data[field],
                      marker=markers[idx % 4], color=colors[idx % 4],
                      lw=2, ms=6, markevery=max(1, len(ts) // 20),
                      label=label, alpha=0.8)
        ylabel = 'Material Temperature (keV)' if field == 'T_mat' else 'Radiation Temperature (keV)'
        ax.set_xlabel('Time (ns)', fontproperties=fp, fontsize=12)
        ax.set_ylabel(ylabel, fontproperties=fp, fontsize=12)
        leg = ax.legend(fontsize=9)
        if HAS_PLOTFUNCS:
            for txt in leg.get_texts():
                txt.set_fontproperties(_font)
                txt.set_fontsize(9)
        ax.grid(True, which='both', alpha=0.3, ls='--')
        plt.tight_layout()
        if HAS_PLOTFUNCS:
            _hide_spines()
        fname = f'{savefile_prefix}_history_{field}.png'
        plt.savefig(fname, dpi=650, bbox_inches='tight')
        print(f'Saved: {fname}')
        plt.close()


def save_fiducials_npz(results, savefile_prefix='refinement_test_pn'):
    """Save fiducial-point temperature histories to NPZ."""
    fid_data = results['fid_data']
    fid_labels = list(fid_data.keys())
    save_dict = {
        'fid_times': results['ts'],
        'fid_labels': np.array(fid_labels, dtype=object),
    }
    for j, lbl in enumerate(fid_labels):
        save_dict[f'fid_Tmat_{j}'] = fid_data[lbl]['T_mat']
        save_dict[f'fid_Trad_{j}'] = fid_data[lbl]['T_rad']

    fname = f'{savefile_prefix}_fiducials.npz'
    np.savez_compressed(fname, **save_dict)
    print(f'Saved: {fname}')


def save_snapshots_npz(results, savefile_prefix='refinement_test_pn'):
    """Save snapshots for direct field-by-field comparisons."""
    out = {
        'x_faces': results['x_faces'],
        'y_faces': results['y_faces'],
        'x_centers': results['x_centers'],
        'y_centers': results['y_centers'],
        'ts': results['ts'],
        'N_pn': np.array([results['N_pn']], dtype=int),
    }
    for t_target, sol in results['solutions'].items():
        key = f"{float(t_target):.6f}"
        out[f'T_{key}'] = sol['T']
        out[f'phi_{key}'] = sol['phi']
        out[f't_actual_{key}'] = np.array([sol['t_actual']], dtype=float)

    fname = f'{savefile_prefix}_snapshots.npz'
    np.savez_compressed(fname, **out)
    print(f'Saved: {fname}')


def print_channel_symmetry_diagnostic(results, t_target, channel='upper'):
    """Print x-mirror symmetry diagnostics inside one channel at time t_target."""
    if t_target not in results['solutions']:
        return

    x = results['x_centers']
    y = results['y_centers']
    sol = results['solutions'][t_target]

    T_mat = np.mean(sol['T'], axis=2)
    T_rad = (np.mean(sol['phi'], axis=2) / ac)**0.25

    if channel == 'upper':
        x_lo, x_hi = 3.0, 4.0
        y_mask = y > 3.0
        x_mid = 3.5
    elif channel == 'lower':
        x_lo, x_hi = 1.0, 2.0
        y_mask = y < 2.0
        x_mid = 1.5
    else:
        raise ValueError("channel must be 'upper' or 'lower'")

    ix = np.where((x >= x_lo) & (x <= x_hi))[0]
    jy = np.where(y_mask)[0]
    if ix.size == 0 or jy.size == 0:
        print(f"Symmetry diagnostic ({channel}): no cells in region")
        return

    def _stats(field):
        errs = []
        region_max = 0.0
        for i in ix:
            x_mirror = 2.0 * x_mid - x[i]
            i_m = np.argmin(np.abs(x - x_mirror))
            for j in jy:
                v = field[i, j]
                vm = field[i_m, j]
                errs.append(abs(v - vm))
                if v > region_max:
                    region_max = v
                if vm > region_max:
                    region_max = vm
        errs = np.asarray(errs)
        max_abs = float(np.max(errs))
        mean_abs = float(np.mean(errs))
        rel_max = max_abs / max(1e-14, region_max)
        return max_abs, mean_abs, rel_max

    mT, aT, rT = _stats(T_mat)
    mR, aR, rR = _stats(T_rad)
    t_actual = sol['t_actual']
    print(f"Symmetry diagnostic ({channel}, mirror in x) at t={t_actual:.3f} ns")
    print(f"  T_mat: max|D|={mT:.3e}, mean|D|={aT:.3e}, rel_max={rT:.3e}")
    print(f"  T_rad: max|D|={mR:.3e}, mean|D|={aR:.3e}, rel_max={rR:.3e}")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Refinement Test P_N')
    parser.add_argument('--Ix', type=int, default=100, help='Coarse cells in x')
    parser.add_argument('--Iy', type=int, default=100, help='Coarse cells in y')
    parser.add_argument('--n-refine', type=int, default=10,
                        help='Sub-cells per coarse cell at interfaces')
    parser.add_argument('--Npn', type=int, default=3, help='P_N order (e.g., 3 or 7)')
    parser.add_argument('--output-times', type=float, nargs='+',
                        default=[1.0, 10.0, 100.0, 501.0, 700.0, 1000.0])
    parser.add_argument('--dt-start', type=float, default=1e-3)
    parser.add_argument('--dt-max', type=float, default=10.0)
    parser.add_argument('--tol', type=float, default=1e-5)
    parser.add_argument('--maxits', type=int, default=100)
    parser.add_argument('--n-gs', type=int, default=1)
    parser.add_argument('--gmres', action='store_true',
                        help='Use matrix-free GMRES instead of GS')
    parser.add_argument('--loud', action='store_true')
    parser.add_argument('--save-prefix', type=str, default=None,
                        help='Prefix for output files (default: refinement_test_p{Npn})')
    args = parser.parse_args()

    save_prefix = args.save_prefix if args.save_prefix else f'refinement_test_p{args.Npn}'

    results = setup_and_run(
        Ix=args.Ix,
        Iy=args.Iy,
        N_pn=args.Npn,
        output_times=tuple(args.output_times),
        LOUD=args.loud,
        n_refine=args.n_refine,
        dt_start=args.dt_start,
        dt_max=args.dt_max,
        tolerance=args.tol,
        maxits=args.maxits,
        n_gs=args.n_gs,
        use_gmres=args.gmres,
    )

    plot_material(results, savefile=f'{save_prefix}_material.png')
    first_snap = True
    for t in args.output_times:
        if t in results['solutions']:
            plot_snapshot(results, t, savefile_prefix=save_prefix, first_one=first_snap)
            first_snap = False
            print_channel_symmetry_diagnostic(results, t, channel='upper')
            print_channel_symmetry_diagnostic(results, t, channel='lower')
    plot_fiducial_history(results, savefile_prefix=save_prefix)
    save_fiducials_npz(results, savefile_prefix=save_prefix)
    save_snapshots_npz(results, savefile_prefix=save_prefix)
