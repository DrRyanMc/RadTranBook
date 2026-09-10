import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

"""
Marshak Wave 2-D verification tests.

Runs the Marshak wave driven from:
  (a) the x=0 boundary (reflecting on y boundaries and x=Lx)
  (b) the y=0 boundary (reflecting on x boundaries and y=Ly)

Both should produce the same 1-D profile along the propagation direction,
verifying that the 2-D solver handles sweeps and reflecting BCs correctly
in both coordinate directions.

Compares numerical results against the semi-analytic self-similar solution.

Run from the DiscreteOrdinates2D directory:
    python problems/test_marshak_wave_directions.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from DiscreteOrdinates2D.src.sn_solver_2d import temp_solve_2d, c, a, ac
from DiscreteOrdinates2D.src.quadratures import get_2d_quadrature


# ===========================================================================
# Semi-analytic self-similar solutions (Test 1: inhomogeneous Marshak wave)
# ===========================================================================

def semi_analytic_Tr(x, t):
    """Semi-analytic radiation temperature profile."""
    xheat = 0.8332614 * t
    xi0 = 1.2746051
    xi = xi0 * x / xheat
    Tr = np.zeros_like(x)
    mask1 = xi <= xi0 * 0.75
    mask2 = (xi > xi0 * 0.75) & (xi < xi0)
    f = np.zeros_like(x)
    f[mask1] = 1 - 0.75567 * (xi[mask1] / xi0)**2.0416
    f[mask2] = 1.2527 * (1 - xi[mask2] / xi0)**0.55623
    Tr = t**(86.0/57.0) * f
    Tr[xi >= xi0] = 0.0
    return Tr


def semi_analytic_T(x, t):
    """Semi-analytic material temperature profile."""
    xheat = 0.8332614 * t
    xi0 = 1.2746051
    xi = xi0 * x / xheat
    g = np.zeros_like(x)
    mask1 = xi <= xi0 * 0.05
    mask2 = (xi > xi0 * 0.05) & (xi < xi0)
    g[mask1] = -0.15937 * (xi[mask1] / xi0)**0.2194 + (xi[mask1] / xi0)**0.074842
    g[mask2] = (0.63674 + 0.55611 * (xi[mask2] / xi0)**0.56101) * \
               (1 - xi[mask2] / xi0)**0.63964
    T = t**(86.0/57.0) * g
    T[xi >= xi0] = 0.0
    return T


# ===========================================================================
# Material properties (inhomogeneous Marshak wave, Test 1)
# ===========================================================================

def make_material_functions(Ix, Iy, dx_arr, dy_arr, direction='x'):
    """Create material property functions for the Marshak wave.

    Parameters
    ----------
    direction : str
        'x' means wave propagates in x (density varies with x).
        'y' means wave propagates in y (density varies with y).
    """
    # Material parameters
    alpha = 1.5
    beta = 3.4
    u0 = 0.01
    k0 = 0.1
    ks = 40 - k0
    mu_mat = 0.14
    omega = -20.0 / 19.0
    lam = 0.2
    lamp = 0.2

    # Compute corner positions along propagation direction
    if direction == 'x':
        x_faces = np.zeros(Ix + 1)
        for i in range(Ix):
            x_faces[i+1] = x_faces[i] + dx_arr[i]
        x_centers = 0.5 * (x_faces[:-1] + x_faces[1:])

        # Corner x-positions
        pos_corner = np.zeros((Ix, Iy, 4))
        for i in range(Ix):
            dx4 = dx_arr[i] / 4.0
            pos_corner[i, :, 0] = x_centers[i] + dx4  # NE
            pos_corner[i, :, 1] = x_centers[i] - dx4  # NW
            pos_corner[i, :, 2] = x_centers[i] - dx4  # SW
            pos_corner[i, :, 3] = x_centers[i] + dx4  # SE
    else:  # direction == 'y'
        y_faces = np.zeros(Iy + 1)
        for j in range(Iy):
            y_faces[j+1] = y_faces[j] + dy_arr[j]
        y_centers = 0.5 * (y_faces[:-1] + y_faces[1:])

        # Corner y-positions
        pos_corner = np.zeros((Ix, Iy, 4))
        for j in range(Iy):
            dy4 = dy_arr[j] / 4.0
            pos_corner[:, j, 0] = y_centers[j] + dy4  # NE (y+)
            pos_corner[:, j, 1] = y_centers[j] + dy4  # NW (y+)
            pos_corner[:, j, 2] = y_centers[j] - dy4  # SW (y-)
            pos_corner[:, j, 3] = y_centers[j] - dy4  # SE (y-)

    pos_corner = np.maximum(pos_corner, 1e-12)
    rho_corner = pos_corner ** (-omega)

    def EOS(T):
        return u0 * T**beta * rho_corner**(1.0 - mu_mat)

    def invEOS(e):
        return (e / u0 / rho_corner**(1.0 - mu_mat))**(1.0 / beta)

    def sigma_func(T):
        T_safe = np.maximum(T, 1e-30)
        return k0 * T_safe**(-alpha) * rho_corner**(lamp + 1.0)

    def scat_func(T):
        T_safe = np.maximum(T, 1e-30)
        return ks * T_safe**(-alpha) * rho_corner**(lam + 1.0)

    return EOS, invEOS, sigma_func, scat_func


def run_marshak_x(Ix=50, Iy=3, N_quad=4, tfinal=1.0,
                  quad_type='level_symmetric', Lx=1.0, Ly=0.1):
    """Run Marshak wave propagating in x-direction."""
    print(f"\n{'='*60}")
    print(f"  Marshak wave propagating in X direction")
    print(f"  Ix={Ix}, Iy={Iy}, N_quad={N_quad}, quad={quad_type}")
    print(f"{'='*60}")

    dx_arr = np.full(Ix, Lx / Ix)
    dy_arr = np.full(Iy, Ly / Iy)
    Tinit = 1e-3

    Omega_x, Omega_y, weights = get_2d_quadrature(quad_type, N_quad)
    M = len(Omega_x)

    EOS, invEOS, sigma_func, scat_func = make_material_functions(
        Ix, Iy, dx_arr, dy_arr, direction='x')

    T_init = np.full((Ix, Iy, 4), Tinit)
    phi_init = ac * T_init**4
    q_ext = np.zeros((Ix, Iy, 4))

    def BCs_func(t):
        tau = 86.0 / 57.0
        T_bc = 1.0470478 * max(t, 1e-30)**tau
        I_hot = ac * T_bc**4
        BCs_xlo = np.zeros((Iy, M, 2))
        for n in range(M):
            if Omega_x[n] > 0:
                BCs_xlo[:, n, :] = I_hot
        return {'xlo': BCs_xlo, 'xhi': None, 'ylo': None, 'yhi': None}

    phis, Ts, iterations, ts, its_per_step = temp_solve_2d(
        Ix, Iy, dx_arr, dy_arr,
        q_ext, sigma_func, scat_func,
        quad_type, N_quad,
        BCs_func, EOS, invEOS,
        phi_init, T_init,
        dt_min=1e-4, dt_max=0.01, tfinal=tfinal,
        tolerance=1e-5, Linf_tol=1e-3, maxits=20,
        K=30, R=3,
        reflect_xlo=False, reflect_xhi=True,
        reflect_ylo=True, reflect_yhi=True,
        use_dmd=True,
        print_stride=20,
    )

    # Compute x-centers
    x_faces = np.zeros(Ix + 1)
    for i in range(Ix):
        x_faces[i+1] = x_faces[i] + dx_arr[i]
    x_centers = 0.5 * (x_faces[:-1] + x_faces[1:])

    print(f"  Done: {iterations} sweeps, {len(ts)-1} steps")
    return {
        'phis': phis, 'Ts': Ts, 'ts': ts, 'iterations': iterations,
        'prop_centers': x_centers, 'Ix': Ix, 'Iy': Iy,
        'direction': 'x',
    }


def run_marshak_y(Ix=3, Iy=50, N_quad=4, tfinal=1.0,
                  quad_type='level_symmetric', Lx=0.1, Ly=1.0):
    """Run Marshak wave propagating in y-direction."""
    print(f"\n{'='*60}")
    print(f"  Marshak wave propagating in Y direction")
    print(f"  Ix={Ix}, Iy={Iy}, N_quad={N_quad}, quad={quad_type}")
    print(f"{'='*60}")

    dx_arr = np.full(Ix, Lx / Ix)
    dy_arr = np.full(Iy, Ly / Iy)
    Tinit = 1e-3

    Omega_x, Omega_y, weights = get_2d_quadrature(quad_type, N_quad)
    M = len(Omega_x)

    EOS, invEOS, sigma_func, scat_func = make_material_functions(
        Ix, Iy, dx_arr, dy_arr, direction='y')

    T_init = np.full((Ix, Iy, 4), Tinit)
    phi_init = ac * T_init**4
    q_ext = np.zeros((Ix, Iy, 4))

    def BCs_func(t):
        tau = 86.0 / 57.0
        T_bc = 1.0470478 * max(t, 1e-30)**tau
        I_hot = ac * T_bc**4
        BCs_ylo = np.zeros((Ix, M, 2))
        for n in range(M):
            if Omega_y[n] > 0:
                BCs_ylo[:, n, :] = I_hot
        return {'xlo': None, 'xhi': None, 'ylo': BCs_ylo, 'yhi': None}

    phis, Ts, iterations, ts, its_per_step = temp_solve_2d(
        Ix, Iy, dx_arr, dy_arr,
        q_ext, sigma_func, scat_func,
        quad_type, N_quad,
        BCs_func, EOS, invEOS,
        phi_init, T_init,
        dt_min=1e-4, dt_max=0.01, tfinal=tfinal,
        tolerance=1e-5, Linf_tol=1e-3, maxits=20,
        K=30, R=3,
        reflect_xlo=True, reflect_xhi=True,
        reflect_ylo=False, reflect_yhi=True,
        use_dmd=True,
        print_stride=20,
    )

    # Compute y-centers
    y_faces = np.zeros(Iy + 1)
    for j in range(Iy):
        y_faces[j+1] = y_faces[j] + dy_arr[j]
    y_centers = 0.5 * (y_faces[:-1] + y_faces[1:])

    print(f"  Done: {iterations} sweeps, {len(ts)-1} steps")
    return {
        'phis': phis, 'Ts': Ts, 'ts': ts, 'iterations': iterations,
        'prop_centers': y_centers, 'Ix': Ix, 'Iy': Iy,
        'direction': 'y',
    }


def extract_1d_profile(result, time_target):
    """Extract 1-D temperature profile along propagation direction."""
    ts = result['ts']
    Ts = result['Ts']
    direction = result['direction']

    idx = np.argmin(np.abs(ts - time_target))
    T_snap = Ts[idx]  # (Ix, Iy, 4)
    t_actual = ts[idx]

    if direction == 'x':
        # Average over y and corners
        T_1d = np.mean(T_snap, axis=(1, 2))
    else:
        # Average over x and corners
        T_1d = np.mean(T_snap, axis=(0, 2))

    return result['prop_centers'], T_1d, t_actual


def extract_1d_Tr(result, time_target):
    """Extract 1-D radiation temperature profile."""
    ts = result['ts']
    phis = result['phis']
    direction = result['direction']

    idx = np.argmin(np.abs(ts - time_target))
    phi_snap = phis[idx]
    t_actual = ts[idx]

    if direction == 'x':
        phi_1d = np.mean(phi_snap, axis=(1, 2))
    else:
        phi_1d = np.mean(phi_snap, axis=(0, 2))

    Tr_1d = (phi_1d / ac)**0.25
    return result['prop_centers'], Tr_1d, t_actual


def plot_comparison(result_x, result_y, plot_times=(0.3, 0.6, 1.0),
                    savefile='marshak_wave_2d_comparison.png'):
    """Plot comparison of x-direction and y-direction runs with self-similar."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(plot_times)))
    x_fine = np.linspace(1e-6, 1.0, 500)

    # Top left: Material temperature, x-propagation
    ax = axes[0, 0]
    for k, t_tgt in enumerate(plot_times):
        x, T, t_act = extract_1d_profile(result_x, t_tgt)
        ax.plot(x, T, 'o', color=colors[k], ms=3, label=f't={t_act:.2f} (num)')
        T_ref = semi_analytic_T(x_fine, t_act)
        ax.plot(x_fine, T_ref, '-', color=colors[k], alpha=0.7)
    ax.set_xlabel('x (cm)')
    ax.set_ylabel('T (keV)')
    ax.set_title('Material Temp — X propagation')
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)

    # Top right: Material temperature, y-propagation
    ax = axes[0, 1]
    for k, t_tgt in enumerate(plot_times):
        y, T, t_act = extract_1d_profile(result_y, t_tgt)
        ax.plot(y, T, 's', color=colors[k], ms=3, label=f't={t_act:.2f} (num)')
        T_ref = semi_analytic_T(x_fine, t_act)
        ax.plot(x_fine, T_ref, '-', color=colors[k], alpha=0.7)
    ax.set_xlabel('y (cm)')
    ax.set_ylabel('T (keV)')
    ax.set_title('Material Temp — Y propagation')
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)

    # Bottom left: Radiation temperature, x-propagation
    ax = axes[1, 0]
    for k, t_tgt in enumerate(plot_times):
        x, Tr, t_act = extract_1d_Tr(result_x, t_tgt)
        ax.plot(x, Tr, 'o', color=colors[k], ms=3, label=f't={t_act:.2f} (num)')
        Tr_ref = semi_analytic_Tr(x_fine, t_act)
        ax.plot(x_fine, Tr_ref, '-', color=colors[k], alpha=0.7)
    ax.set_xlabel('x (cm)')
    ax.set_ylabel('$T_r$ (keV)')
    ax.set_title('Radiation Temp — X propagation')
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)

    # Bottom right: Radiation temperature, y-propagation
    ax = axes[1, 1]
    for k, t_tgt in enumerate(plot_times):
        y, Tr, t_act = extract_1d_Tr(result_y, t_tgt)
        ax.plot(y, Tr, 's', color=colors[k], ms=3, label=f't={t_act:.2f} (num)')
        Tr_ref = semi_analytic_Tr(x_fine, t_act)
        ax.plot(x_fine, Tr_ref, '-', color=colors[k], alpha=0.7)
    ax.set_xlabel('y (cm)')
    ax.set_ylabel('$T_r$ (keV)')
    ax.set_title('Radiation Temp — Y propagation')
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)

    plt.suptitle('2-D S$_N$ Marshak Wave: X vs Y propagation\n'
                 '(solid lines = self-similar solution)', fontsize=12)
    plt.tight_layout()
    plt.savefig(savefile, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {savefile}")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Marshak wave direction comparison')
    parser.add_argument('--Ncells', type=int, default=50, help='Cells in propagation dir')
    parser.add_argument('--N_quad', type=int, default=4, help='Quadrature order')
    parser.add_argument('--tfinal', type=float, default=1.0, help='Final time (ns)')
    parser.add_argument('--quad', type=str, default='level_symmetric')
    args = parser.parse_args()

    result_x = run_marshak_x(Ix=args.Ncells, Iy=3, N_quad=args.N_quad,
                             tfinal=args.tfinal, quad_type=args.quad)
    result_y = run_marshak_y(Ix=3, Iy=args.Ncells, N_quad=args.N_quad,
                             tfinal=args.tfinal, quad_type=args.quad)

    # Print comparison
    print(f"\n{'='*60}")
    print(f"  Comparison at t = {args.tfinal:.2f} ns")
    print(f"{'='*60}")
    x_x, T_x, t_x = extract_1d_profile(result_x, args.tfinal)
    y_y, T_y, t_y = extract_1d_profile(result_y, args.tfinal)

    # Interpolate y-result onto x-grid for comparison
    T_y_interp = np.interp(x_x, y_y, T_y)
    diff = np.max(np.abs(T_x - T_y_interp))
    rel_diff = diff / (np.max(T_x) + 1e-30)
    print(f"  Max |T_x - T_y| = {diff:.4e} keV")
    print(f"  Relative diff   = {rel_diff:.4e}")

    # Compare with self-similar
    T_ref = semi_analytic_T(x_x, t_x)
    mask = T_ref > 0.01  # only where wave has arrived
    if np.any(mask):
        err_x = np.sqrt(np.mean(((T_x[mask] - T_ref[mask]) / T_ref[mask])**2))
        err_y = np.sqrt(np.mean(((T_y_interp[mask] - T_ref[mask]) / T_ref[mask])**2))
        print(f"  RMS error vs self-similar (x-prop): {err_x:.4e}")
        print(f"  RMS error vs self-similar (y-prop): {err_y:.4e}")

    plot_comparison(result_x, result_y,
                    plot_times=(0.3, 0.6, args.tfinal),
                    savefile='marshak_wave_2d_comparison.png')
