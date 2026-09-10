import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

"""
2-D Cylindrical Zel'dovich Wave — Point source in x-y geometry.

Tests the 2-D S_N corner-balance solver with a radially symmetric problem.
A radiation energy pulse at the origin diffuses outward in an absorbing
medium. The solution should be radially symmetric and can be compared
to the self-similar Zel'dovich wave solution.

Problem setup:
  - Domain: [0, Lx] × [0, Ly] (first quadrant)
  - All boundaries: reflecting (to simulate the full cylinder via symmetry)
  - Material opacity: σ = σ₀ T^{-n}, n=3, σ₀=300 cm⁻¹
  - Heat capacity: c_v = 3×10⁻⁶ GJ/(cm³·keV)
  - Initial condition: energy from self-similar solution at early time t₀
  - Compare radial profiles ρ = √(x² + y²) with N=2 (cylindrical) solution

The 2-D Cartesian solution should match the self-similar solution when
plotted as a function of radius ρ from the origin.

Run from the DiscreteOrdinates2D directory:
    python problems/zeldovich_wave_2d.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from DiscreteOrdinates2D.src.sn_solver_2d import temp_solve_2d, c, a, ac
from DiscreteOrdinates2D.src.quadratures import get_2d_quadrature


# ===========================================================================
# Self-similar Zel'dovich wave solution
# ===========================================================================

# Physical constants
_s = 4        # T^s radiation energy (T^4)
_n = 3.0      # opacity exponent: sigma = sigma0 * T^(-n)
_sigma0 = 300.0   # reference opacity
_cv_vol = 3e-6    # volumetric heat capacity (GJ/(cm^3 keV))
_b = _cv_vol      # per the diffusion parameterization
_ac = a * c       # radiation constant * speed of light
_E0 = 1e-10       # total injected energy (GJ/cm for 2-D, tuned for ~1 cm front at t~1 ns)


def zeldovich_self_similar(r, t, N=2):
    """Compute the Zel'dovich wave self-similar solution.

    Parameters
    ----------
    r : ndarray
        Radial positions.
    t : float
        Time (ns).
    N : int
        Spatial dimension (2 for cylindrical in x-y).

    Returns
    -------
    T : ndarray
        Temperature at each r.
    R_front : float
        Front radius.
    """
    from scipy.special import gamma as Gamma

    m = _s + _n  # = 7
    p = m / _s   # = 7/4

    # Diffusion coefficient
    kappa = (4.0 / (4.0 + _n)) * _ac / (3.0 * _sigma0)
    D = kappa / _b

    # Similarity exponents
    beta = 1.0 / (N * (p - 1.0) + 2.0)
    alpha = N * beta / _s

    # Lambda and A from energy conservation
    lam = (p - 1.0) * beta / (2.0 * p * D)

    # Energy integral: M = E0 / b
    # Integral of T^s over N-D sphere:
    # I_N = S_N * int_0^R r^(N-1) * T^s dr
    # Using u = (r/R)^2, this gives:
    # = S_N * R^N * A^(s/(m-s)) / 2 * B(N/2, s/(m-s) + 1)
    M = _E0 / _b

    # Surface area factor
    if N == 1:
        S_N = 2.0
    elif N == 2:
        S_N = 2.0 * np.pi
    else:
        S_N = 4.0 * np.pi

    # Beta function B(N/2, s/(m-s) + 1) = B(N/2, s/(m-s) + 1)
    from scipy.special import beta as Beta_func
    q = _s / (m - _s)  # = 4/3
    B_val = Beta_func(N / 2.0, q + 1.0)

    # From energy conservation:
    # M = S_N / 2 * (A/lam)^(N/2) * A^q * B_val
    # Solve for A:
    # eta_f = sqrt(A/lam), R = eta_f * t^beta
    # M = S_N/2 * eta_f^N * A^q * B_val
    # M = S_N/2 * (A/lam)^(N/2) * A^q * B_val
    # M = S_N/2 * A^(N/2 + q) / lam^(N/2) * B_val
    exponent = N / 2.0 + q
    A = (M * 2.0 * lam**(N/2.0) / (S_N * B_val))**(1.0 / exponent)

    # Front radius
    eta_f = np.sqrt(A / lam)
    R_front = eta_f * t**beta

    # Temperature profile
    T = np.zeros_like(r, dtype=np.float64)
    mask = r < R_front
    if np.any(mask):
        xi = r[mask] / R_front
        T[mask] = (t**(-alpha) * A**(1.0 / (m - _s)) *
                   (1.0 - xi**2)**(1.0 / (m - _s)))

    return T, R_front


# ===========================================================================
# Material properties
# ===========================================================================

def make_zeldovich_materials(Ix, Iy, dx_arr, dy_arr):
    """Create material property functions for uniform medium."""

    def EOS(T):
        """e = c_v * T (uniform density, linear EOS)."""
        return _cv_vol * T

    def invEOS(e):
        """T from e."""
        return e / _cv_vol

    def sigma_func(T):
        """Absorption opacity: sigma = sigma0 * T^(-n)."""
        T_safe = np.maximum(T, 1e-6)
        return _sigma0 * T_safe**(-_n)

    def scat_func(T):
        """No scattering."""
        return np.zeros_like(T)

    return EOS, invEOS, sigma_func, scat_func


# ===========================================================================
# Problem setup and solver
# ===========================================================================

def run_zeldovich_2d(Ix=40, Iy=40, N_quad=4, tfinal=3.0,
                     quad_type='level_symmetric',
                     Lx=1.5, Ly=1.5,
                     t_init=0.1,
                     dt_min=1e-5, dt_max=1e-2,
                     K=10, maxits=100,W=3,
                     stretch_x=1.0, stretch_y=1.0,
                     LOUD=False, use_dmd=True):
    """Run the 2-D Zel'dovich wave problem.

    Parameters
    ----------
    Ix, Iy : int
        Number of cells.
    N_quad : int
        Quadrature order.
    tfinal : float
        Final physical time (ns).
    t_init : float
        Initial time for self-similar IC (ns).
    Lx, Ly : float
        Domain dimensions (first quadrant).
    stretch_x, stretch_y : float
        Grid stretching (>1 refines near origin).

    Returns
    -------
    results : dict
    """
    print(f"\n{'='*60}")
    print(f"  2-D Zel'dovich Wave (cylindrical symmetry in x-y)")
    print(f"  Ix={Ix}, Iy={Iy}, N_quad={N_quad}, quad={quad_type}")
    print(f"  Domain: [0,{Lx}] x [0,{Ly}], t_init={t_init}, tfinal={tfinal}")
    print(f"{'='*60}")

    # Generate grid (refine near origin where gradients are steep)
    dx_arr = _generate_variable_zones(Ix, Lx, stretch_x)
    dy_arr = _generate_variable_zones(Iy, Ly, stretch_y)

    # Cell centers
    x_faces = np.zeros(Ix + 1)
    for i in range(Ix):
        x_faces[i+1] = x_faces[i] + dx_arr[i]
    x_centers = 0.5 * (x_faces[:-1] + x_faces[1:])

    y_faces = np.zeros(Iy + 1)
    for j in range(Iy):
        y_faces[j+1] = y_faces[j] + dy_arr[j]
    y_centers = 0.5 * (y_faces[:-1] + y_faces[1:])

    # Get quadrature for BC setup
    Omega_x, Omega_y, weights = get_2d_quadrature(quad_type, N_quad)
    M = len(Omega_x)

    # Material functions
    EOS, invEOS, sigma_func, scat_func = make_zeldovich_materials(
        Ix, Iy, dx_arr, dy_arr)

    # Initial condition from self-similar solution at t_init
    # Corner positions
    T_init = np.zeros((Ix, Iy, 4))
    T_floor = 1e-4  # keV

    for i in range(Ix):
        for j in range(Iy):
            for cc in range(4):
                # Corner position
                if cc == 0:  # NE
                    xc = x_centers[i] + dx_arr[i] / 4.0
                    yc = y_centers[j] + dy_arr[j] / 4.0
                elif cc == 1:  # NW
                    xc = x_centers[i] - dx_arr[i] / 4.0
                    yc = y_centers[j] + dy_arr[j] / 4.0
                elif cc == 2:  # SW
                    xc = x_centers[i] - dx_arr[i] / 4.0
                    yc = y_centers[j] - dy_arr[j] / 4.0
                else:  # SE
                    xc = x_centers[i] + dx_arr[i] / 4.0
                    yc = y_centers[j] - dy_arr[j] / 4.0

                rho = np.sqrt(xc**2 + yc**2)
                T_val, _ = zeldovich_self_similar(np.array([rho]), t_init, N=2)
                T_init[i, j, cc] = max(T_val[0], T_floor)

    phi_init = ac * T_init**4
    q_ext = np.zeros((Ix, Iy, 4))

    _, R_front_init = zeldovich_self_similar(np.array([0.0]), t_init, N=2)
    print(f"  Initial front radius: R={R_front_init:.4f} cm")
    print(f"  Initial T_max: {T_init.max():.4f} keV")

    # Boundary conditions: all reflecting (first quadrant with symmetry)
    def BCs_func(t):
        return {'xlo': None, 'xhi': None, 'ylo': None, 'yhi': None}

    # Output times
    time_outputs = np.array([0.1, 0.3, 1.0, tfinal])
    time_outputs = time_outputs[time_outputs <= tfinal]

    # Run solver
    phis, Ts, iterations, ts, its_per_step = temp_solve_2d(
        Ix, Iy, dx_arr, dy_arr,
        q_ext, sigma_func, scat_func,
        quad_type, N_quad,
        BCs_func, EOS, invEOS,
        phi_init, T_init,
        dt_min=dt_min, dt_max=dt_max, tfinal=tfinal - t_init,
        tolerance=1e-7, Linf_tol=1e-4, maxits=maxits,
        K=K, R=3,
        reflect_xlo=True, reflect_xhi=True,
        reflect_ylo=True, reflect_yhi=True,
        use_dmd=use_dmd,
        W=W,
        LOUD=LOUD,
        print_stride=20,
        time_outputs=time_outputs - t_init,
    )

    # Adjust times to physical time
    ts_physical = np.array(ts) + t_init

    print(f"\n  Done: {iterations} sweeps, {len(ts)-1} steps")
    print(f"  Final T_max: {np.max(Ts[-1]):.4f} keV")

    return {
        'phis': phis, 'Ts': Ts, 'ts': ts_physical,
        'iterations': iterations, 'its_per_step': its_per_step,
        'x_centers': x_centers, 'y_centers': y_centers,
        'x_faces': x_faces, 'y_faces': y_faces,
        'dx_arr': dx_arr, 'dy_arr': dy_arr,
        'Ix': Ix, 'Iy': Iy, 'Lx': Lx, 'Ly': Ly,
        't_init': t_init,
    }


def _generate_variable_zones(N_cells, L, stretch):
    """Generate variable zone widths with geometric stretching."""
    if abs(stretch - 1.0) < 1e-10:
        return np.full(N_cells, L / N_cells)
    ratios = stretch ** np.arange(N_cells)
    dx = L * ratios / np.sum(ratios)
    return dx


# ===========================================================================
# Analysis and plotting
# ===========================================================================

def extract_radial_profile(result, time_target):
    """Extract radial profile from 2-D solution at given time."""
    ts = result['ts']
    Ts = result['Ts']
    x_centers = result['x_centers']
    y_centers = result['y_centers']

    idx = np.argmin(np.abs(ts - time_target))
    T_snap = Ts[idx]  # (Ix, Iy, 4)
    t_actual = ts[idx]

    # Average over corners
    T_cell = np.mean(T_snap, axis=2)  # (Ix, Iy)

    # Extract (radius, T) pairs from all cells
    rho_list = []
    T_list = []
    for i in range(len(x_centers)):
        for j in range(len(y_centers)):
            rho = np.sqrt(x_centers[i]**2 + y_centers[j]**2)
            rho_list.append(rho)
            T_list.append(T_cell[i, j])

    rho_arr = np.array(rho_list)
    T_arr = np.array(T_list)

    # Sort by radius and bin-average
    sort_idx = np.argsort(rho_arr)
    rho_sorted = rho_arr[sort_idx]
    T_sorted = T_arr[sort_idx]

    # Bin into radial shells
    n_bins = min(50, len(rho_sorted) // 2)
    rho_bins = np.linspace(rho_sorted.min(), rho_sorted.max(), n_bins + 1)
    rho_avg = []
    T_avg = []
    for k in range(n_bins):
        mask = (rho_sorted >= rho_bins[k]) & (rho_sorted < rho_bins[k+1])
        if np.any(mask):
            rho_avg.append(np.mean(rho_sorted[mask]))
            T_avg.append(np.mean(T_sorted[mask]))

    return np.array(rho_avg), np.array(T_avg), t_actual


def plot_results(result, plot_times=None,
                 savefile='zeldovich_2d_sn.png'):
    """Plot radial profiles and compare with self-similar solution."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if plot_times is None:
        plot_times = [0.1, 0.3, 1.0, result['ts'][-1]]
        plot_times = [t for t in plot_times if t <= result['ts'][-1]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(plot_times)))

    # Left: Material temperature radial profiles
    ax = axes[0]
    for k, t_tgt in enumerate(plot_times):
        rho, T, t_act = extract_radial_profile(result, t_tgt)
        ax.plot(rho, T, 'o', color=colors[k], ms=4, alpha=0.7,
                label=f't={t_act:.2f} ns (num)')

        # Self-similar solution
        rho_fine = np.linspace(0.01, result['Lx'] * 1.2, 300)
        T_ref, R_front = zeldovich_self_similar(rho_fine, t_act, N=2)
        ax.plot(rho_fine, T_ref, '-', color=colors[k], lw=2, alpha=0.8)
        ax.axvline(R_front, color=colors[k], ls=':', alpha=0.4)

    ax.set_xlabel(r'$\rho = \sqrt{x^2 + y^2}$ (cm)')
    ax.set_ylabel('T (keV)')
    ax.set_title('Material Temperature')
    ax.legend(fontsize=8)
    ax.set_xlim(0, result['Lx'])
    ax.grid(True, alpha=0.3)

    # Right: 2-D temperature field at final time
    ax = axes[1]
    T_final = np.mean(result['Ts'][-1], axis=2)  # (Ix, Iy)
    im = ax.pcolormesh(result['x_faces'], result['y_faces'],
                       T_final.T, shading='flat', cmap='hot')
    ax.set_xlabel('x (cm)')
    ax.set_ylabel('y (cm)')
    ax.set_title(f'T field at t={result["ts"][-1]:.2f} ns')
    plt.colorbar(im, ax=ax, label='T (keV)')
    ax.set_aspect('equal')

    # Draw front radius circle
    _, R_front = zeldovich_self_similar(np.array([0.0]), result['ts'][-1], N=2)
    theta = np.linspace(0, np.pi/2, 100)
    ax.plot(R_front * np.cos(theta), R_front * np.sin(theta),
            'c--', lw=2, label=f'R_front={R_front:.2f}')
    ax.legend()

    plt.suptitle("2-D S$_N$ Zel'dovich Wave (cylindrical symmetry)\n"
                 "solid lines = self-similar N=2 solution", fontsize=11)
    plt.tight_layout()
    plt.savefig(savefile, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {savefile}")


def plot_isotropy(result, time_target=None,
                  savefile='zeldovich_2d_isotropy.png'):
    """Check radial isotropy: compare profiles along x-axis, y-axis, diagonal."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if time_target is None:
        time_target = result['ts'][-1]

    ts = result['ts']
    Ts = result['Ts']
    idx = np.argmin(np.abs(ts - time_target))
    T_snap = Ts[idx]
    t_actual = ts[idx]

    T_cell = np.mean(T_snap, axis=2)
    x_centers = result['x_centers']
    y_centers = result['y_centers']

    fig, ax = plt.subplots(figsize=(8, 5))

    # Along x-axis (y=0, take j=0)
    rho_x = x_centers
    T_x = T_cell[:, 0]
    ax.plot(rho_x, T_x, 'b-o', ms=3, label='Along x-axis (y≈0)')

    # Along y-axis (x=0, take i=0)
    rho_y = y_centers
    T_y = T_cell[0, :]
    ax.plot(rho_y, T_y, 'r-s', ms=3, label='Along y-axis (x≈0)')

    # Along diagonal (x=y)
    n_diag = min(len(x_centers), len(y_centers))
    rho_diag = []
    T_diag = []
    for k in range(n_diag):
        i = k
        j_match = np.argmin(np.abs(y_centers - x_centers[i]))
        rho_diag.append(np.sqrt(x_centers[i]**2 + y_centers[j_match]**2))
        T_diag.append(T_cell[i, j_match])
    ax.plot(rho_diag, T_diag, 'g-^', ms=3, label='Along diagonal (x=y)')

    # Self-similar
    rho_fine = np.linspace(0.01, result['Lx'], 300)
    T_ref, R = zeldovich_self_similar(rho_fine, t_actual, N=2)
    ax.plot(rho_fine, T_ref, 'k-', lw=2, alpha=0.7, label='Self-similar (N=2)')
    ax.axvline(R, color='k', ls=':', alpha=0.5)

    ax.set_xlabel(r'$\rho$ (cm)')
    ax.set_ylabel('T (keV)')
    ax.set_title(f"Isotropy check at t={t_actual:.2f} ns")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, result['Lx'])

    plt.tight_layout()
    plt.savefig(savefile, dpi=150, bbox_inches='tight')
    print(f"Isotropy plot saved to {savefile}")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="2-D Zel'dovich Wave (S_N)")
    parser.add_argument('--Ix', type=int, default=30, help='Cells in x')
    parser.add_argument('--Iy', type=int, default=30, help='Cells in y')
    parser.add_argument('--N', type=int, default=4, help='Quadrature order')
    parser.add_argument('--tfinal', type=float, default=1.0, help='Final time')
    parser.add_argument('--quad', type=str, default='level_symmetric')
    parser.add_argument('--stretch', type=float, default=1.0, help='Grid stretch')
    parser.add_argument('--no-dmd', action='store_true')
    parser.add_argument('--loud', action='store_true')
    args = parser.parse_args()

    result = run_zeldovich_2d(
        Ix=args.Ix, Iy=args.Iy, N_quad=args.N,
        tfinal=args.tfinal, quad_type=args.quad,
        stretch_x=args.stretch, stretch_y=args.stretch,
        use_dmd=not args.no_dmd,
        LOUD=args.loud,
    )

    plot_results(result)
    plot_isotropy(result)
