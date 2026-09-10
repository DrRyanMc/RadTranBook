import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

"""
2-D Refinement Test — S_N corner-balance solver.

Geometry: 2-D Cartesian, domain [0,5]×[0,5] cm

Materials (temperature-independent, position-dependent):
  Thick (default):   σ_a = 200 cm⁻¹,  c_v = 0.5  GJ/(cm³·keV)
  Lower thin channel x∈[1,2], y<2:  σ_a = 0.2 cm⁻¹, c_v = 0.05
  Upper thin channel x∈[3,4], y>3:  σ_a = 0.2 cm⁻¹, c_v = 0.05

BCs:
  x=0, x=5: reflecting
  y=0: incoming blackbody at T=0.3 keV for x∈[1,2], vacuum elsewhere; off at t≥500 ns
  y=5: incoming blackbody at T=0.3 keV for x∈[3,4], vacuum elsewhere; off at t≥500 ns

Run from the DiscreteOrdinates2D directory:
    python problems/crooked_pipe_sn.py
"""

import sys
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from DiscreteOrdinates2D.src.sn_solver_2d import temp_solve_2d, c, a, ac
from DiscreteOrdinates2D.src.quadratures import get_2d_quadrature

try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from utils.plotfuncs import hide_spines as _hide_spines, font as _font
    HAS_PLOTFUNCS = True
except ImportError:
    HAS_PLOTFUNCS = False

# ===========================================================================
# Material constants
# ===========================================================================
SIG_THICK = 200.0   # cm⁻¹
SIG_THIN  = 0.2     # cm⁻¹
CV_THICK  = 0.5     # GJ/(cm³·keV)
CV_THIN   = 0.05    # GJ/(cm³·keV)

T_INIT = 0.01   # keV
T_BC   = 0.3    # keV
T_CUTOFF = 500.0  # ns


# ===========================================================================
# Mesh generation with log-spaced interface refinement (from IMC code)
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
    elif z_int >= z_right:
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
                mid = (cl + cr) / 2
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
    """Return mirror-symmetric log-spaced faces on [seg_left, seg_right].

    Cell widths are smallest at both ends (interfaces) and largest near the
    segment center, with exact left-right mirror symmetry.
    """
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
        # Continue the geometric progression through the center cell.
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
    x_faces = np.cumsum(np.concatenate([[0], dx_arr]))
    y_faces = np.cumsum(np.concatenate([[0], dy_arr]))
    x_centers = 0.5 * (x_faces[:-1] + x_faces[1:])
    y_centers = 0.5 * (y_faces[:-1] + y_faces[1:])

    sigma_corners = np.full((Ix, Iy, 4), SIG_THICK)
    cv_corners = np.full((Ix, Iy, 4), CV_THICK)

    for i in range(Ix):
        for j in range(Iy):
            for cc in range(4):
                # Corner positions
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

def setup_and_run(Ix=100, Iy=100, N_quad=4, quad_type='level_symmetric',
                  output_times=(1.0, 10.0, 100.0, 501.0, 700.0, 1000.0),
                  LOUD=False, use_dmd=True,
                  n_refine=10, refine_width=0.05):
    """Run the refinement test problem.

    Parameters
    ----------
    Ix, Iy : int
        Number of COARSE cells (before refinement) in x and y.
    n_refine : int
        Number of sub-cells per coarse cell in refinement zones.
    refine_width : float
        Width (cm) around each interface to apply refinement.
    """

    Lx, Ly = 5.0, 5.0

    # Build refined mesh. Keep log refinement near x=3 and x=4 as before, but
    # enforce mirror symmetry inside x in [3,4] for clean left-right checks.
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

    Omega_x, Omega_y, weights = get_2d_quadrature(quad_type, N_quad)
    M = len(Omega_x)

    # Build position-dependent material arrays
    sigma_corners, cv_corners = build_corner_materials(Ix_actual, Iy_actual, dx_arr, dy_arr)

    print(f"Refinement Test S_N: Ix={Ix_actual} (coarse {Ix}), Iy={Iy_actual} (coarse {Iy}), M={M}")
    print(f"  dx range: [{dx_arr.min():.5f}, {dx_arr.max():.5f}]")
    print(f"  dy range: [{dy_arr.min():.5f}, {dy_arr.max():.5f}]")
    print(f"  σ: {SIG_THICK} (thick) / {SIG_THIN} (thin)")
    print(f"  cv: {CV_THICK} (thick) / {CV_THIN} (thin)")
    print(f"  T_init={T_INIT}, T_bc={T_BC}, T_cutoff={T_CUTOFF}")

    # Material functions (position-dependent via closures over corner arrays)
    def sigma_func(T):
        return sigma_corners  # temperature-independent

    def scat_func(T):
        return np.zeros_like(T)

    def EOS(T):
        return cv_corners * T

    def invEOS(e):
        return e / cv_corners

    # Initial conditions
    T_init = np.full((Ix_actual, Iy_actual, 4), T_INIT)
    phi_init = ac * T_init**4
    q_ext = np.zeros((Ix_actual, Iy_actual, 4))

    # Boundary conditions
    I_bc = ac * T_BC**4

    def BCs_func(t):
        BCs_ylo = np.zeros((Ix_actual, M, 2))  # bottom (y=0)
        BCs_yhi = np.zeros((Ix_actual, M, 2))  # top (y=5)

        if t < T_CUTOFF:
            for n in range(M):
                # Bottom: incoming for Ωy > 0 at x∈[1,2]
                if Omega_y[n] > 0:
                    for i in range(Ix_actual):
                        if 1.0 <= x_centers[i] <= 2.0:
                            BCs_ylo[i, n, :] = I_bc

                # Top: incoming for Ωy < 0 at x∈[3,4]
                if Omega_y[n] < 0:
                    for i in range(Ix_actual):
                        if 3.0 <= x_centers[i] <= 4.0:
                            BCs_yhi[i, n, :] = I_bc

        return {
            'xlo': np.zeros((Ix_actual, M, 2)),  # vacuum (zero)
            'xhi': np.zeros((Ix_actual, M, 2)),  # vacuum (zero)
            'ylo': BCs_ylo,
            'yhi': BCs_yhi,
        }

    # Time stepping
    output_times = np.array(sorted(output_times))
    tfinal = output_times[-1]

    phis, Ts, iterations, ts, its_per_step = temp_solve_2d(
        Ix_actual, Iy_actual, dx_arr, dy_arr,
        q_ext, sigma_func, scat_func,
        quad_type, N_quad,
        BCs_func, EOS, invEOS,
        phi_init, T_init,
        dt_min=1e-3, dt_max=10.0, tfinal=tfinal,
        tolerance=1e-5, Linf_tol=1e-3, maxits=100,
        K=100, R=5,W=10,
        reflect_xlo=False, reflect_xhi=False,
        reflect_ylo=False, reflect_yhi=False,
        use_dmd=use_dmd,
        LOUD=LOUD,
        print_stride=20,
        time_outputs=output_times,
    )

    print(f"\nTotal sweeps: {iterations}, steps: {len(ts)-1}")

    # Extract snapshots
    solutions = {}
    for t_target in output_times:
        idx = np.argmin(np.abs(np.array(ts) - t_target))
        solutions[t_target] = {
            'T': Ts[idx],      # (Ix, Iy, 4)
            'phi': phis[idx],  # (Ix, Iy, 4)
            't_actual': ts[idx],
        }
        T_max = np.max(Ts[idx])
        print(f"  t={ts[idx]:.2f} ns: T_max={T_max:.4f} keV")

    # Fiducial point histories
    fid_points = {
        'Point 1 x=1.5, y=1.95': (1.5, 1.95),
        'Point 2 x=1.5, y=2.05': (1.5, 2.05),
        'Point 3 x=3.5, y=3.05': (3.5, 3.05),
        'Point 4 x=3.5, y=2.95': (3.5, 2.95),
    }
    fid_indices = {}
    for label, (xv, yv) in fid_points.items():
        i = np.argmin(np.abs(x_centers - xv))
        j = np.argmin(np.abs(y_centers - yv))
        fid_indices[label] = (i, j)

    fid_data = {}
    for label, (i, j) in fid_indices.items():
        fid_data[label] = {
            'T_mat': np.array([np.mean(Ts[k][i, j, :]) for k in range(len(Ts))]),
            'T_rad': np.array([(np.mean(phis[k][i, j, :]) / ac)**0.25
                               for k in range(len(phis))]),
        }

    return {
        'solutions': solutions, 'x_centers': x_centers, 'y_centers': y_centers,
        'x_faces': x_faces, 'y_faces': y_faces,
        'Ix': Ix_actual, 'Iy': Iy_actual, 'ts': np.array(ts),
        'fid_data': fid_data, 'fid_indices': fid_indices,
        'sigma_corners': sigma_corners,
    }


# ===========================================================================
# Plotting
# ===========================================================================

def _channel_lines(ax):
    """Draw thin-channel boundary lines."""
    kw = dict(color='cyan', lw=0.8, ls='--', alpha=0.7)
    ax.axhline(1.0, **kw); ax.axhline(2.0, **kw)
    ax.axhline(3.0, **kw); ax.axhline(4.0, **kw)
    ax.axvline(2.0, **kw); ax.axvline(3.0, **kw)


def plot_material(results, savefile='refinement_test_sn_material.png'):
    """Plot opacity layout."""
    sig = np.mean(results['sigma_corners'], axis=2)  # avg corners
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    im = ax.pcolormesh(results['y_centers'], results['x_centers'], sig,
                       shading='auto', cmap='RdYlBu_r',
                       norm=matplotlib.colors.LogNorm(vmin=SIG_THIN, vmax=SIG_THICK))
    plt.colorbar(im, ax=ax, label=r'$\sigma_a$ (cm$^{-1}$)')
    _channel_lines(ax)
    ax.set_xlabel('y (cm)'); ax.set_ylabel('x (cm)')
    ax.set_title('Opacity layout')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(savefile, dpi=150, bbox_inches='tight')
    print(f'Saved: {savefile}')
    plt.close()


def plot_snapshot(results, t_target, savefile_prefix='refinement_test_sn',
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

    T_cell  = np.mean(sol['T'],   axis=2)
    Tr_cell = (np.mean(sol['phi'], axis=2) / ac)**0.25

    is_first = first_one   # mutable flag: only the very first panel gets the bar

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
            #is_first = False   # subsequent panels in this call have no colorbar

        plt.tight_layout()
        fname = f'{savefile_prefix}_{field}_t_{t_target:.0f}ns.png'
        plt.savefig(fname, dpi=650, bbox_inches='tight')
        print(f'Saved: {fname}')
        plt.close()


def plot_fiducial_history(results, savefile_prefix='refinement_test_sn'):
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
                      lw=2, ms=6, markevery=max(1, len(ts)//20),
                      label=label, alpha=0.8)
        ylabel = 'Material Temperature (keV)' if field == 'T_mat' else 'Radiation Temperature (keV)'
        ax.set_xlabel('Time (ns)', fontproperties=fp, fontsize=12)
        ax.set_ylabel(ylabel,      fontproperties=fp, fontsize=12)
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


def save_fiducials_npz(results, savefile_prefix='refinement_test_sn'):
    """Save fiducial-point temperature histories to NPZ.

    Layout
    ------
    fid_times      : (n_steps,) time array [ns].
    fid_labels     : (n_fids,) object array of fiducial-point name strings.
    fid_Tmat_{j}   : material temperature history for fiducial j.
    fid_Trad_{j}   : radiation temperature history for fiducial j.

    Loading example
    ---------------
    d = np.load('refinement_test_sn_fiducials.npz', allow_pickle=True)
    labels = list(d['fid_labels'])
    times  = d['fid_times']
    for j, lbl in enumerate(labels):
        T_mat = d[f'fid_Tmat_{j}']
        T_rad = d[f'fid_Trad_{j}']
    """
    fid_data   = results['fid_data']
    fid_labels = list(fid_data.keys())
    save_dict  = {
        'fid_times':  results['ts'],
        'fid_labels': np.array(fid_labels, dtype=object),
    }
    for j, lbl in enumerate(fid_labels):
        save_dict[f'fid_Tmat_{j}'] = fid_data[lbl]['T_mat']
        save_dict[f'fid_Trad_{j}'] = fid_data[lbl]['T_rad']

    fname = f'{savefile_prefix}_fiducials.npz'
    np.savez_compressed(fname, **save_dict)
    print(f'Saved: {fname}')


def print_channel_symmetry_diagnostic(results, t_target, channel='upper'):
    """Print x-mirror symmetry diagnostics inside one channel at time t_target.

    Mirrors about x=1.5 cm in lower channel and x=3.5 cm in upper channel,
    while holding y fixed. Reports max/mean absolute error and max relative
    error for both material and radiation temperatures.
    """
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
    print(f"  T_mat: max|Δ|={mT:.3e}, mean|Δ|={aT:.3e}, rel_max={rT:.3e}")
    print(f"  T_rad: max|Δ|={mR:.3e}, mean|Δ|={aR:.3e}, rel_max={rR:.3e}")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Refinement Test S_N')
    parser.add_argument('--Ix', type=int, default=100, help='Coarse cells in x')
    parser.add_argument('--Iy', type=int, default=100, help='Coarse cells in y')
    parser.add_argument('--n-refine', type=int, default=10,
                        help='Sub-cells per coarse cell at interfaces')
    parser.add_argument('--N', type=int, default=4)
    parser.add_argument('--quad', type=str, default='level_symmetric')
    parser.add_argument('--output-times', type=float, nargs='+',
                        default=[1.0, 10.0, 100.0, 501.0, 700.0, 1000.0])
    parser.add_argument('--no-dmd', action='store_true')
    parser.add_argument('--loud', action='store_true')
    args = parser.parse_args()

    results = setup_and_run(
        Ix=args.Ix, Iy=args.Iy, N_quad=args.N,
        quad_type=args.quad,
        output_times=tuple(args.output_times),
        use_dmd=not args.no_dmd,
        LOUD=args.loud,
        n_refine=args.n_refine,
    )

    plot_material(results)
    first_snap = True
    for t in args.output_times:
        if t in results['solutions']:
            plot_snapshot(results, t, first_one=first_snap)
            first_snap = False
            print_channel_symmetry_diagnostic(results, t, channel='upper')
            print_channel_symmetry_diagnostic(results, t, channel='lower')
    plot_fiducial_history(results)
    save_fiducials_npz(results)
