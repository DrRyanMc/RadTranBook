import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

"""
2-D Marshak Wave problem with reflecting boundaries.

This test problem runs a Marshak wave driven from the x=0 boundary into
a 2-D domain [0, Lx] × [0, Ly]. Three of the four boundaries are reflecting
(x_max, y_min, y_max), making the problem effectively 1-D in x. This
verifies the 2-D solver against the known 1-D Marshak wave solution.

The material properties are the same as the 1-D inhomogeneous Marshak wave
(Test 1): power-law density profile ρ(x) = x^{-ω} with absorption and
scattering.

Run from the DiscreteOrdinates2D directory:
    python problems/marshak_wave_2d.py
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from DiscreteOrdinates2D.src.sn_solver_2d import temp_solve_2d, c, a, ac


def setup_and_run(Ix=100, Iy=5, N_quad=4, tfinal=1.0,
                  dt_min=1e-5, dt_max=0.01, K=50, maxits=100,
                  quad_type='level_symmetric',
                  LOUD=0, use_dmd=True, Lx=1.0, Ly=0.2,
                  stretch_x=1.0, stretch_y=1.0):
    """Set up and run the 2-D Marshak wave test.

    Parameters
    ----------
    Ix, Iy : int
        Number of cells in x and y.
    N_quad : int
        Quadrature order.
    tfinal : float
        Final time (ns).
    quad_type : str
        Quadrature type ('level_symmetric', 'equal_weight', 'product_square',
        'product_triangular').
    Lx, Ly : float
        Domain dimensions (cm).
    stretch_x, stretch_y : float
        Geometric stretch factors for variable zoning.
        stretch=1.0 gives uniform zones.
        stretch>1.0 refines near x=0 (or y=0).

    Returns
    -------
    results : dict
    """
    # === Grid generation with variable zones ===
    dx_arr = _generate_variable_zones(Ix, Lx, stretch_x)
    dy_arr = _generate_variable_zones(Iy, Ly, stretch_y)

    # Cell-center coordinates
    x_faces = np.zeros(Ix + 1)
    for i in range(Ix):
        x_faces[i+1] = x_faces[i] + dx_arr[i]
    x_centers = 0.5 * (x_faces[:-1] + x_faces[1:])

    y_faces = np.zeros(Iy + 1)
    for j in range(Iy):
        y_faces[j+1] = y_faces[j] + dy_arr[j]
    y_centers = 0.5 * (y_faces[:-1] + y_faces[1:])

    # Corner positions (for material properties)
    # Corner 0=NE (x+, y+), 1=NW (x-, y+), 2=SW (x-, y-), 3=SE (x+, y-)
    # x-position of corners:
    #   corners 0,3 (right half): x_center + dx/4
    #   corners 1,2 (left half):  x_center - dx/4
    # For the power-law density ρ(x) = x^{-ω}, we use x-position of corners
    x_corner = np.zeros((Ix, Iy, 4))
    for i in range(Ix):
        for j in range(Iy):
            x_mid = x_centers[i]
            dx4 = dx_arr[i] / 4.0
            x_corner[i, j, 0] = x_mid + dx4  # NE
            x_corner[i, j, 1] = x_mid - dx4  # NW
            x_corner[i, j, 2] = x_mid - dx4  # SW
            x_corner[i, j, 3] = x_mid + dx4  # SE

    # Clip to avoid x=0 singularity
    x_corner = np.maximum(x_corner, 1e-12)

    # === Material parameters (same as 1-D Marshak Test 1) ===
    alpha = 1.5          # opacity temperature exponent
    beta = 3.4           # EOS temperature exponent
    u0 = 0.01           # EOS coefficient
    k0 = 0.1            # absorption opacity coefficient
    ks = 40 - k0        # scattering opacity coefficient
    mu_mat = 0.14        # EOS density exponent
    omega = -20.0 / 19.0  # density profile exponent
    lam = 0.2            # scattering density exponent
    lamp = 0.2           # absorption density exponent
    Tinit = 1e-3         # initial temperature (keV)

    # Density at corner positions
    rho_corner = x_corner ** (-omega)

    # === EOS ===
    def EOS(T):
        """e = u0 * T^beta * rho^(1 - mu_mat)"""
        return u0 * T**beta * rho_corner**(1.0 - mu_mat)

    def invEOS(e):
        """T from e"""
        return (e / u0 / rho_corner**(1.0 - mu_mat))**(1.0 / beta)

    # === Opacities ===
    def sigma_func(T):
        """Absorption opacity: sigma_a = k0 * T^(-alpha) * rho^(lamp+1)"""
        T_safe = np.maximum(T, 1e-30)
        return k0 * T_safe**(-alpha) * rho_corner**(lamp + 1.0)

    def scat_func(T):
        """Scattering opacity: sigma_s = ks * T^(-alpha) * rho^(lam+1)"""
        T_safe = np.maximum(T, 1e-30)
        return ks * T_safe**(-alpha) * rho_corner**(lam + 1.0)

    # === Initial conditions ===
    T_init = np.full((Ix, Iy, 4), Tinit)
    phi_init = ac * T_init**4

    # === External source (zero) ===
    q_ext = np.zeros((Ix, Iy, 4))

    # === Boundary conditions ===
    # Left (x=0): incoming blackbody at T_b(t) = 1.0470478 * t^(86/57)
    # Right (x=Lx): reflecting
    # Bottom (y=0): reflecting
    # Top (y=Ly): reflecting
    from quadratures import get_2d_quadrature
    Omega_x, Omega_y, weights = get_2d_quadrature(quad_type, N_quad)
    M = len(Omega_x)

    def BCs_func(t):
        """Boundary conditions at time t."""
        tau = 86.0 / 57.0
        T_bc = 1.0470478 * max(t, 1e-30)**tau
        I_hot = ac * T_bc**4
        I_cold = ac * Tinit**4

        # x_lo: incoming for angles with Omega_x > 0
        BCs_xlo = np.zeros((Iy, M, 2))
        for n in range(M):
            if Omega_x[n] > 0:
                BCs_xlo[:, n, :] = I_hot

        # Other boundaries: zero (reflecting will be handled by solver)
        return {
            'xlo': BCs_xlo,
            'xhi': None,  # reflecting
            'ylo': None,  # reflecting
            'yhi': None,  # reflecting
        }

    # === Run solver ===
    print(f"Running 2-D Marshak wave: Ix={Ix}, Iy={Iy}, N_quad={N_quad}, "
          f"quad={quad_type}, tfinal={tfinal}")
    print(f"  Domain: [{0}, {Lx}] x [{0}, {Ly}]")
    print(f"  dx range: [{dx_arr.min():.4e}, {dx_arr.max():.4e}]")
    print(f"  dy range: [{dy_arr.min():.4e}, {dy_arr.max():.4e}]")

    phis, Ts, iterations, ts, its_per_step = temp_solve_2d(
        Ix, Iy, dx_arr, dy_arr,
        q_ext, sigma_func, scat_func,
        quad_type, N_quad,
        BCs_func, EOS, invEOS,
        phi_init, T_init,
        dt_min=dt_min, dt_max=dt_max, tfinal=tfinal,
        Linf_tol=1e-5, tolerance=1e-8, maxits=maxits,
        LOUD=LOUD, K=K, R=3,
        reflect_xlo=False, reflect_xhi=True,
        reflect_ylo=True, reflect_yhi=True,
        use_dmd=use_dmd,
        print_stride=10,
    )

    print(f"\nDone. Total transport sweeps: {iterations}")
    print(f"  Time steps: {len(ts)-1}")
    print(f"  Avg iterations/step: {iterations / max(len(ts)-1, 1):.1f}")

    return {
        'phis': phis, 'Ts': Ts, 'iterations': iterations,
        'ts': ts, 'its_per_step': its_per_step,
        'x_centers': x_centers, 'y_centers': y_centers,
        'x_faces': x_faces, 'y_faces': y_faces,
        'dx_arr': dx_arr, 'dy_arr': dy_arr,
        'Ix': Ix, 'Iy': Iy, 'Lx': Lx, 'Ly': Ly,
    }


def _generate_variable_zones(N_cells, L, stretch):
    """Generate variable zone widths with geometric stretching.

    Parameters
    ----------
    N_cells : int
        Number of cells.
    L : float
        Total domain length.
    stretch : float
        Geometric ratio between successive cell widths.
        stretch=1.0 → uniform.
        stretch>1.0 → cells grow from left to right (refines left).
        stretch<1.0 → cells shrink from left to right.

    Returns
    -------
    dx : (N_cells,) float64
    """
    if abs(stretch - 1.0) < 1e-10:
        return np.full(N_cells, L / N_cells)

    # Geometric sequence: dx_i = dx_0 * stretch^i
    # Sum = dx_0 * (stretch^N - 1) / (stretch - 1) = L
    ratios = stretch ** np.arange(N_cells)
    dx = L * ratios / np.sum(ratios)
    return dx


# ===========================================================================
# Semi-analytic reference solutions
# ===========================================================================

def semi_analytic_Tr(x, t):
    """Semi-analytic radiation temperature profile for Test 1."""
    xheat = 0.8332614 * t
    xi0 = 1.2746051
    xi = xi0 * x / xheat
    Tr = np.zeros_like(x)
    mask1 = xi <= xi0 * 0.75
    mask2 = (xi > xi0 * 0.75) & (xi < xi0)
    f = np.zeros_like(x)
    f[mask1] = 1 - 0.75567 * (xi[mask1] / xi0)**2.0416
    f[mask2] = 1.2527 * (1 - xi[mask2] / xi0)**0.55623
    Tr = t**(86/57) * f
    Tr[xi >= xi0] = 0.0
    return Tr


def semi_analytic_T(x, t):
    """Semi-analytic material temperature profile for Test 1."""
    xheat = 0.8332614 * t
    xi0 = 1.2746051
    xi = xi0 * x / xheat
    g = np.zeros_like(x)
    mask1 = xi <= xi0 * 0.05
    mask2 = (xi > xi0 * 0.05) & (xi < xi0)
    g[mask1] = -0.15937 * (xi[mask1] / xi0)**0.2194 + (xi[mask1] / xi0)**0.074842
    g[mask2] = (0.63674 + 0.55611 * (xi[mask2] / xi0)**0.56101) * \
               (1 - xi[mask2] / xi0)**0.63964
    T = t**(86/57) * g
    T[xi >= xi0] = 0.0
    return T


# ===========================================================================
# Plotting
# ===========================================================================

def plot_results(results, plot_times=(0.2, 0.6, 1.0), savefile=''):
    """Plot temperature profiles along x (averaged over y) and compare
    with semi-analytic solutions."""
    ts = results['ts']
    Ts = results['Ts']
    x_centers = results['x_centers']
    Ix = results['Ix']
    Iy = results['Iy']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(plot_times)))

    for k, t_target in enumerate(plot_times):
        # Find closest time step
        idx = np.argmin(np.abs(ts - t_target))
        T_snap = Ts[idx]  # (Ix, Iy, 4)
        t_actual = ts[idx]

        # Average over y and corners for 1-D comparison
        T_x = np.mean(T_snap, axis=(1, 2))  # average over y and corners

        # Semi-analytic
        x_fine = np.linspace(1e-6, results['Lx'], 500)
        T_analytic = semi_analytic_T(x_fine, t_actual)
        Tr_analytic = semi_analytic_Tr(x_fine, t_actual)

        # Radiation temperature from phi
        phi_snap = results['phis'][idx]
        phi_x = np.mean(phi_snap, axis=(1, 2))
        Tr_x = (phi_x / ac)**0.25

        ax1.plot(x_centers, T_x, 'o', color=colors[k], ms=3,
                 label=f't={t_actual:.2f} ns (num)')
        ax1.plot(x_fine, T_analytic, '-', color=colors[k], alpha=0.7)

        ax2.plot(x_centers, Tr_x, 's', color=colors[k], ms=3,
                 label=f't={t_actual:.2f} ns (num)')
        ax2.plot(x_fine, Tr_analytic, '-', color=colors[k], alpha=0.7)

    ax1.set_xlabel('x (cm)')
    ax1.set_ylabel('Material Temperature T (keV)')
    ax1.set_title('Material Temperature')
    ax1.legend()
    ax1.set_xlim(0, results['Lx'])

    ax2.set_xlabel('x (cm)')
    ax2.set_ylabel('Radiation Temperature Tr (keV)')
    ax2.set_title('Radiation Temperature')
    ax2.legend()
    ax2.set_xlim(0, results['Lx'])

    plt.tight_layout()
    if savefile:
        plt.savefig(savefile, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {savefile}")
    plt.show()


def plot_2d_temperature(results, time_idx=-1, savefile=''):
    """Plot 2-D temperature field at a given time step."""
    T_snap = results['Ts'][time_idx]
    x_faces = results['x_faces']
    y_faces = results['y_faces']

    # Average corners to get cell-center values
    T_cell = np.mean(T_snap, axis=2)  # (Ix, Iy)

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.pcolormesh(x_faces, y_faces, T_cell.T, shading='flat', cmap='hot')
    ax.set_xlabel('x (cm)')
    ax.set_ylabel('y (cm)')
    ax.set_title(f'Material Temperature at t={results["ts"][time_idx]:.3f} ns')
    plt.colorbar(im, ax=ax, label='T (keV)')
    ax.set_aspect('equal')
    plt.tight_layout()
    if savefile:
        plt.savefig(savefile, dpi=150, bbox_inches='tight')
    plt.show()


# ===========================================================================
# Main
# ===========================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='2-D Marshak Wave')
    parser.add_argument('--Ix', type=int, default=50, help='Cells in x')
    parser.add_argument('--Iy', type=int, default=3, help='Cells in y')
    parser.add_argument('--N', type=int, default=4, help='Quadrature order')
    parser.add_argument('--tfinal', type=float, default=1.0, help='Final time (ns)')
    parser.add_argument('--quad', type=str, default='level_symmetric',
                        help='Quadrature type')
    parser.add_argument('--stretch_x', type=float, default=1.0,
                        help='x-direction stretch factor')
    parser.add_argument('--stretch_y', type=float, default=1.0,
                        help='y-direction stretch factor')
    parser.add_argument('--no-dmd', action='store_true', help='Disable DMD')
    parser.add_argument('--loud', action='store_true', help='Verbose output')
    args = parser.parse_args()

    results = setup_and_run(
        Ix=args.Ix, Iy=args.Iy, N_quad=args.N, tfinal=args.tfinal,
        quad_type=args.quad,
        stretch_x=args.stretch_x, stretch_y=args.stretch_y,
        use_dmd=not args.no_dmd,
        LOUD=args.loud,
    )

    plot_results(results, plot_times=(0.2, 0.5, 1.0))
