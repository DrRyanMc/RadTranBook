import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

"""
2-D Cylindrical Zel'dovich Wave — Convergence study version.

Initializes from the N=2 (cylindrical) self-similar solution at t_start
and runs to t_final.  Compares the numerical RADIATION temperature
  T_rad = (φ / (a c))^{1/4}
against the analytic profile.

Physics:
  σ = 300 T^{-3}  cm^{-1}
  c_v = 10  GJ/(cm³·keV)   (large → f ≈ 1, source iteration converges fast)
  E₀ = 10  GJ/cm           (total injected energy per unit z-length)

The self-similar solution is for:
  Cv·∂T/∂t = ∇·((4ac/(3σ₀))·T^{n+3}·∇T)
with m = n+3 = 6, giving β = 1/(Nm+2) = 1/14 for N=2.

Workflow:
  1. Quick sanity check:   python zeldovich_wave_2d_converge.py
  2. Convergence study:    python zeldovich_wave_2d_converge.py --study
  3. Production run:       python zeldovich_wave_2d_converge.py --Ix 200 --N 8 --tfinal 3.0

Run from the DiscreteOrdinates2D/problems directory.
"""

import sys
import os
import numpy as np
import time as timer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from DiscreteOrdinates2D.src.sn_solver_2d import temp_solve_2d, c, a, ac
from DiscreteOrdinates2D.src.quadratures import get_2d_quadrature


# ===========================================================================
# Physics parameters
# ===========================================================================
# σ₀ = 3000 gives R_front/mfp ≈ 9 (well into diffusion regime).
# Cv = 10 keeps the Fleck factor f ≈ 0.67 for dt = 1e-3 ns,
# so spectral radius ≈ 0.16 and source iteration converges in ~5 sweeps.
# The self-similar solution is for:  Cv·∂T/∂t = ∇·(K₀·T^{m}·∇T)
# with K₀ = 4ac/(3σ₀), m = n+3 = 6 (from radiation flux through T^{-n} medium).

_n = 3.0          # opacity exponent: σ = σ₀·T^{-n}
_sigma0 = 3000.0  # reference opacity (cm⁻¹)
_cv_vol = 100.0    # volumetric heat capacity (GJ/(cm³·keV))
_E0 = 10.0        # total injected energy (GJ/cm, for 2-D cylindrical)

# Derived diffusion constants
_K0 = 4.0 * ac / (3.0 * _sigma0)   # radiation diffusion prefactor
_m_diff = _n + 3                     # diffusion exponent = 6


# ===========================================================================
# Self-similar Zel'dovich wave solution (N=2 cylindrical)
# ===========================================================================

def zeldovich_self_similar(r, t, N=2):
    """Self-similar solution of Cv·∂T/∂t = ∇·(K₀·T^m·∇T) in N dimensions.

    This is the Zel'dovich–Kompaneets–Barenblatt source solution for the
    nonlinear heat equation with diffusivity K₀·T^m, m = n+3.

    Parameters
    ----------
    r : ndarray   – radial positions (cm)
    t : float     – time (ns)
    N : int       – spatial dimension (2 = cylindrical)

    Returns
    -------
    T : ndarray   – temperature (keV)
    R_front : float – front radius (cm)
    """
    from scipy.special import beta as Beta_func
    from math import pi, gamma, sqrt

    m = _m_diff   # = 6
    D = _K0 / _cv_vol  # effective diffusivity prefactor

    # Self-similar exponents for ∂T/∂t = D·∇·(T^m·∇T)
    # β = 1/(Nm + 2),  α = Nβ
    beta = 1.0 / (N * m + 2.0)
    alpha = N * beta

    # Profile: T(r,t) = t^{-α} · [A - B·r²·t^{-2β}]_+^{1/m}
    # with B = m·β / (2·D) from the ODE
    B = m * beta / (2.0 * D)

    # Energy conservation: E₀ = Cv · ∫T dV
    # For N=2 (cylindrical): E₀ = Cv · 2π · ∫₀^R T·r dr
    # = Cv · 2π · (A/B)^{N/2} · A^{1/m} · I_N
    # where I_N = ∫₀^1 (1-ξ²)^{1/m} · ξ^{N-1} dξ · (1/N) ... etc.
    #
    # Working through the integral with substitution u = ξ²:
    # ∫₀^{η_f} (A-Bη²)^{1/m} η^{N-1} dη
    #   = (A^{1/m}/B^{N/2}) · (1/2) · Beta(N/2, 1/m + 1) · A^{N/2}
    #   Actually: let u = Bη²/A, then η = √(Au/B), dη = √(A/(4Bu))du
    #
    # Cleaner: substitute ξ = η/η_f with η_f = √(A/B)
    # ∫₀^1 A^{1/m}(1-ξ²)^{1/m} · (η_f·ξ)^{N-1} · η_f dξ
    # = A^{1/m} · η_f^N · ∫₀^1 (1-ξ²)^{1/m} · ξ^{N-1} dξ
    #
    # The integral ∫₀^1 (1-ξ²)^{1/m} ξ^{N-1} dξ = (1/2)·B(N/2, 1/m+1)
    # via substitution u = ξ².

    beta_int = Beta_func(N / 2.0, 1.0 / m + 1.0)  # = B(1, 7/6) for N=2,m=6

    # Surface area of unit N-sphere
    S_N = 2.0 * pi**(N / 2.0) / gamma(N / 2.0)

    # Total energy: E₀ = Cv · S_N · A^{1/m} · η_f^N · (1/2) · beta_int
    # With η_f^N = (A/B)^{N/2}:
    # E₀ = Cv · S_N/2 · A^{1/m + N/2} · B^{-N/2} · beta_int
    #
    # Solve for A:
    Q = _E0 / _cv_vol  # = ∫T dV
    power = 1.0 / m + N / 2.0
    A = (Q * 2.0 * B**(N / 2.0) / (S_N * beta_int))**(1.0 / power)

    # Front radius
    eta_f = np.sqrt(A / B)
    R_front = eta_f * t**beta

    # Temperature profile
    T = np.zeros_like(r, dtype=np.float64)
    mask = r < R_front
    if np.any(mask):
        xi = r[mask] / R_front
        T[mask] = t**(-alpha) * A**(1.0 / m) * (1.0 - xi**2)**(1.0 / m)

    return T, R_front


# ===========================================================================
# Material properties
# ===========================================================================

def _make_materials(Ix, Iy):
    """Create EOS and opacity functions."""

    def EOS(T):
        return _cv_vol * T

    def invEOS(e):
        return e / _cv_vol

    def sigma_func(T):
        T_safe = np.maximum(T, 1e-6)
        return _sigma0 * T_safe**(-_n)

    def scat_func(T):
        return np.zeros_like(T)

    return EOS, invEOS, sigma_func, scat_func


# ===========================================================================
# Radial profile extraction (local copy)
# ===========================================================================

def extract_radial_profile(result, time_target):
    """Extract radial T_rad = (φ/(ac))^{1/4} profile from 2-D solution."""
    ts = result['ts']
    phis = result['phis']
    x_centers = result['x_centers']
    y_centers = result['y_centers']

    idx = np.argmin(np.abs(ts - time_target))
    phi_snap = phis[idx]        # (Ix, Iy, 4)
    t_actual = ts[idx]

    # Radiation temperature from cell-averaged phi
    phi_cell = np.mean(phi_snap, axis=2)     # (Ix, Iy)
    T_rad = (np.maximum(phi_cell, 0.0) / ac) ** 0.25

    rho_list = []
    T_list = []
    for i in range(len(x_centers)):
        for j in range(len(y_centers)):
            rho = np.sqrt(x_centers[i]**2 + y_centers[j]**2)
            rho_list.append(rho)
            T_list.append(T_rad[i, j])

    rho_arr = np.array(rho_list)
    T_arr = np.array(T_list)

    sort_idx = np.argsort(rho_arr)
    rho_sorted = rho_arr[sort_idx]
    T_sorted = T_arr[sort_idx]

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


# ===========================================================================
# Problem runner
# ===========================================================================

def run_zeldovich_converge(
    Ix=60, Iy=60, N_quad=4,
    quad_type='level_symmetric',
    Lx=1.0, Ly=1.0,
    t_start=0.1, tfinal=1.0,
    dt_min=1e-5, dt_max=1e-3,
    K=10, maxits=100, W=3, R=3,
    stretch=1.0,
    use_dmd=True, LOUD=False,
    print_stride=50,
    output_times=None,
):
    """Run the 2-D Zel'dovich wave initialized from the self-similar solution.

    Parameters
    ----------
    Ix, Iy : int
        Number of cells in each direction.
    N_quad : int
        S_N quadrature order.
    quad_type : str
        Quadrature type ('level_symmetric', 'product_square', etc.).
    Lx, Ly : float
        Domain extent (cm). Should be larger than the expected front position.
    t_start : float
        Initialization time (ns) — self-similar IC evaluated here.
    tfinal : float
        Final simulation time (ns).
    dt_min, dt_max : float
        Time step bounds.
    K, maxits, W, R : int
        DMD/source-iteration parameters.
    stretch : float
        Grid stretching factor (>1 refines near origin).
    output_times : list or None
        Snapshot times (physical). If None, uses sensible defaults.

    Returns
    -------
    result : dict with all data needed for post-processing.
    """
    print(f"\n{'='*60}")
    print(f"  2-D Zel'dovich Wave — Convergence Run")
    print(f"  Grid: {Ix}×{Iy},  S_{N_quad} ({quad_type})")
    print(f"  Domain: [0,{Lx}]×[0,{Ly}] cm")
    print(f"  Time: {t_start:.3f} → {tfinal:.3f} ns")
    print(f"  Grid stretch: {stretch:.2f}")
    print(f"{'='*60}\n")

    # --- Grid generation ---
    dx_arr = _make_grid(Ix, Lx, stretch)
    dy_arr = _make_grid(Iy, Ly, stretch)

    x_faces = np.concatenate([[0.0], np.cumsum(dx_arr)])
    y_faces = np.concatenate([[0.0], np.cumsum(dy_arr)])
    x_centers = 0.5 * (x_faces[:-1] + x_faces[1:])
    y_centers = 0.5 * (y_faces[:-1] + y_faces[1:])

    print(f"  dx range: [{dx_arr.min():.4e}, {dx_arr.max():.4e}] cm")
    print(f"  dy range: [{dy_arr.min():.4e}, {dy_arr.max():.4e}] cm")

    # --- Initial condition from self-similar at t_start ---
    T_init = np.zeros((Ix, Iy, 4))
    T_floor = 1e-4  # keV — cold background

    # Corner offsets (NE, NW, SW, SE)
    cx_off = np.array([+0.25, -0.25, -0.25, +0.25])
    cy_off = np.array([+0.25, -0.25, -0.25, +0.25])
    # Actually: NE=(+,+), NW=(-,+), SW=(-,-), SE=(+,-)
    cy_off = np.array([+0.25, +0.25, -0.25, -0.25])

    for i in range(Ix):
        for j in range(Iy):
            for cc in range(4):
                xc = x_centers[i] + cx_off[cc] * dx_arr[i]
                yc = y_centers[j] + cy_off[cc] * dy_arr[j]
                rho = np.sqrt(xc**2 + yc**2)
                T_val, _ = zeldovich_self_similar(np.array([rho]), t_start, N=2)
                T_init[i, j, cc] = max(T_val[0], T_floor)

    phi_init = ac * T_init**4

    _, R_front_init = zeldovich_self_similar(np.array([0.0]), t_start, N=2)
    print(f"  Self-similar front at t_start: R = {R_front_init:.4f} cm")
    print(f"  T_max(t_start) = {T_init.max():.4f} keV")

    # Check domain is large enough
    _, R_front_final = zeldovich_self_similar(np.array([0.0]), tfinal, N=2)
    if R_front_final > 0.85 * min(Lx, Ly):
        print(f"  WARNING: front at tfinal ({R_front_final:.3f} cm) "
              f"approaches domain boundary ({min(Lx,Ly):.2f} cm)!")

    # --- Material properties ---
    EOS, invEOS, sigma_func, scat_func = _make_materials(Ix, Iy)

    # --- Boundary conditions ---
    # xlo, ylo: reflecting (symmetry planes through origin)
    # xhi, yhi: vacuum (infinite-medium approximation; domain sized to stay ahead of front)
    def BCs_func(t):
        return {'xlo': None, 'xhi': None, 'ylo': None, 'yhi': None}

    # --- Output times ---
    if output_times is None:
        # Sensible defaults: a few snapshots for comparison
        output_times = np.array([0.3, 0.5, 1.0, 2.0, 3.0, 5.0])
        output_times = output_times[(output_times > t_start) &
                                    (output_times <= tfinal)]

    q_ext = np.zeros((Ix, Iy, 4))

    # --- Run the solver ---
    wall_start = timer.perf_counter()

    phis, Ts, iterations, ts, its_per_step = temp_solve_2d(
        Ix, Iy, dx_arr, dy_arr,
        q_ext, sigma_func, scat_func,
        quad_type, N_quad,
        BCs_func, EOS, invEOS,
        phi_init, T_init,
        dt_min=dt_min, dt_max=dt_max,
        tfinal=tfinal - t_start,
        tolerance=1e-7, Linf_tol=1e-4,
        maxits=maxits, K=K, R=R,
        reflect_xlo=True, reflect_xhi=False,
        reflect_ylo=True, reflect_yhi=False,
        use_dmd=use_dmd, W=W,
        LOUD=LOUD,
        print_stride=print_stride,
        time_outputs=output_times - t_start,
    )

    wall_time = timer.perf_counter() - wall_start
    ts_physical = np.array(ts) + t_start

    print(f"\n  Wall time: {wall_time:.1f} s")
    print(f"  Total sweeps: {iterations}")
    print(f"  Steps: {len(ts)-1}")
    T_rad_final = (np.maximum(np.mean(phis[-1], axis=2), 0.0) / ac) ** 0.25
    print(f"  Final T_rad_max: {np.max(T_rad_final):.4f} keV")

    result = {
        'phis': phis, 'Ts': Ts, 'ts': ts_physical,
        'iterations': iterations, 'its_per_step': its_per_step,
        'x_centers': x_centers, 'y_centers': y_centers,
        'x_faces': x_faces, 'y_faces': y_faces,
        'dx_arr': dx_arr, 'dy_arr': dy_arr,
        'Ix': Ix, 'Iy': Iy, 'Lx': Lx, 'Ly': Ly,
        't_start': t_start, 'tfinal': tfinal,
        'N_quad': N_quad, 'quad_type': quad_type,
        'wall_time': wall_time,
    }
    return result


def _make_grid(N, L, stretch):
    """Generate grid with optional geometric stretching toward origin."""
    if abs(stretch - 1.0) < 1e-10:
        return np.full(N, L / N)
    ratios = stretch ** np.arange(N)
    return L * ratios / np.sum(ratios)


# ===========================================================================
# Convergence analysis
# ===========================================================================

def compute_radial_error(result, t_target=None):
    """Compute L2 and Linf error in T_rad vs self-similar at t_target."""
    if t_target is None:
        t_target = result['ts'][-1]

    ts = result['ts']
    idx = np.argmin(np.abs(ts - t_target))
    phi_snap = result['phis'][idx]
    t_act = ts[idx]

    phi_cell = np.mean(phi_snap, axis=2)
    T_rad = (np.maximum(phi_cell, 0.0) / ac) ** 0.25
    x_centers = result['x_centers']
    y_centers = result['y_centers']

    T_ref_all = np.zeros_like(T_rad)
    rho_all = np.zeros_like(T_rad)
    for i in range(len(x_centers)):
        for j in range(len(y_centers)):
            rho = np.sqrt(x_centers[i]**2 + y_centers[j]**2)
            rho_all[i, j] = rho
            T_val, _ = zeldovich_self_similar(np.array([rho]), t_act, N=2)
            T_ref_all[i, j] = T_val[0]

    _, R_front = zeldovich_self_similar(np.array([0.0]), t_act, N=2)
    mask = rho_all < 0.9 * R_front

    if not np.any(mask):
        return np.nan, np.nan, t_act

    diff = np.abs(T_rad[mask] - T_ref_all[mask])
    T_scale = np.max(T_ref_all[mask])

    L2_err = np.sqrt(np.mean((diff / T_scale)**2))
    Linf_err = np.max(diff) / T_scale

    return L2_err, Linf_err, t_act


def run_convergence_study(t_eval=1.0, tfinal=1.0, t_start=0.1):
    """Run a small convergence study varying mesh and quadrature."""
    print("\n" + "="*60)
    print("  CONVERGENCE STUDY")
    print("="*60)

    cases = [
        # (Ix, Iy, N_quad, label)
        (20, 20, 4, '20×20 S4'),
        (40, 40, 4, '40×40 S4'),
        (60, 60, 4, '60×60 S4'),
        (40, 40, 8, '40×40 S8'),
        (60, 60, 8, '60×60 S8'),
    ]

    results_table = []
    all_results = []

    for Ix, Iy, N_quad, label in cases:
        # Domain must accommodate front
        _, R_f = zeldovich_self_similar(np.array([0.0]), tfinal, N=2)
        Lx = max(0.5, R_f * 2.0)
        Ly = Lx

        res = run_zeldovich_converge(
            Ix=Ix, Iy=Iy, N_quad=N_quad,
            Lx=Lx, Ly=Ly,
            t_start=t_start, tfinal=tfinal,
            dt_min=1e-5, dt_max=1e-3,
            print_stride=0,  # suppress per-step output
        )
        all_results.append(res)

        L2, Linf, t_act = compute_radial_error(res, t_eval)
        results_table.append({
            'label': label, 'Ix': Ix, 'N': N_quad,
            'L2': L2, 'Linf': Linf,
            'sweeps': res['iterations'],
            'wall': res['wall_time'],
        })

    # Print table
    print("\n" + "="*60)
    print(f"  Error at t = {t_eval:.2f} ns (relative to T_max)")
    print(f"  {'Case':<12} {'L2 err':>10} {'Linf err':>10} "
          f"{'Sweeps':>8} {'Wall (s)':>9}")
    print("-"*60)
    for row in results_table:
        print(f"  {row['label']:<12} {row['L2']:>10.3e} {row['Linf']:>10.3e} "
              f"{row['sweeps']:>8d} {row['wall']:>9.1f}")
    print("="*60)

    return results_table, all_results


# ===========================================================================
# Publication-quality plotting
# ===========================================================================

def plot_publication(result, savefile='zeldovich_2d_converge.pdf'):
    """Generate publication-quality figure: radial profiles + 2D field."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    rcParams.update({
        'font.size': 10,
        'axes.labelsize': 11,
        'legend.fontsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'text.usetex': False,
    })

    ts = result['ts']
    # Pick plot times: subset of available snapshots
    plot_times = [t for t in [0.3, 0.5, 1.0, 2.0, result['ts'][-1]]
                  if t <= ts[-1] and t >= ts[0]]
    if not plot_times:
        plot_times = [ts[-1]]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(plot_times)))

    # --- Left panel: radial radiation temperature profiles ---
    ax = axes[0]
    for k, t_tgt in enumerate(plot_times):
        rho, T, t_act = extract_radial_profile(result, t_tgt)
        ax.plot(rho, T, 'o', color=colors[k], ms=3, alpha=0.7, mew=0,
                label=f'$t = {t_act:.2f}$ ns')

        # Analytic
        rho_fine = np.linspace(0.005, result['Lx'] * 1.1, 500)
        T_ref, R_front = zeldovich_self_similar(rho_fine, t_act, N=2)
        ax.plot(rho_fine, T_ref, '-', color=colors[k], lw=1.5, alpha=0.9)

    ax.set_xlabel(r'$\rho = \sqrt{x^2 + y^2}$ (cm)')
    ax.set_ylabel(r'$T_{\rm rad}$ (keV)')
    ax.set_xlim(0, result['Lx'] * 0.95)
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_title(r"Radial $T_{\rm rad}$ (symbols = S$_N$, lines = analytic)")
    ax.grid(True, alpha=0.2)

    # --- Right panel: 2-D radiation temperature map ---
    ax = axes[1]
    phi_final = np.mean(result['phis'][-1], axis=2)
    Trad_final = (np.maximum(phi_final, 0.0) / ac) ** 0.25
    im = ax.pcolormesh(result['x_faces'], result['y_faces'],
                       Trad_final.T, shading='flat', cmap='inferno',
                       vmin=0)
    ax.set_xlabel('$x$ (cm)')
    ax.set_ylabel('$y$ (cm)')
    ax.set_title(f"$T_{{\\rm rad}}$ at $t = {ts[-1]:.2f}$ ns")
    cb = plt.colorbar(im, ax=ax)
    cb.set_label(r'$T_{\rm rad}$ (keV)')
    ax.set_aspect('equal')

    # Front circle
    _, R_front = zeldovich_self_similar(np.array([0.0]), ts[-1], N=2)
    theta = np.linspace(0, np.pi / 2, 100)
    ax.plot(R_front * np.cos(theta), R_front * np.sin(theta),
            'c--', lw=1.5, label=f'$R_f = {R_front:.2f}$ cm')
    ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig(savefile, bbox_inches='tight')
    print(f"\n  Figure saved: {savefile}")
    # Also save PNG for quick viewing
    png_file = savefile.replace('.pdf', '.png')
    plt.savefig(png_file, bbox_inches='tight', dpi=150)
    print(f"  PNG copy:    {png_file}")


def plot_convergence(results_table, savefile='zeldovich_2d_convergence.pdf'):
    """Plot convergence data from the study."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))

    # Group by quadrature order
    for N_val in sorted(set(r['N'] for r in results_table)):
        subset = [r for r in results_table if r['N'] == N_val]
        h_arr = [1.0 / r['Ix'] for r in subset]
        L2_arr = [r['L2'] for r in subset]
        ax.loglog(h_arr, L2_arr, 'o-', label=f'S$_{N_val}$', ms=6)

    # Reference slopes
    h_ref = np.array([min(1.0/r['Ix'] for r in results_table),
                      max(1.0/r['Ix'] for r in results_table)])
    ax.loglog(h_ref, 0.5 * (h_ref / h_ref[-1])**2 * max(r['L2'] for r in results_table),
              'k--', alpha=0.4, label=r'$\mathcal{O}(h^2)$')

    ax.set_xlabel('$1/N_x$ (relative mesh size)')
    ax.set_ylabel('Relative $L_2$ error')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    ax.set_title("Convergence: 2-D Zel'dovich wave")
    plt.tight_layout()
    plt.savefig(savefile, bbox_inches='tight')
    plt.savefig(savefile.replace('.pdf', '.png'), bbox_inches='tight', dpi=150)
    print(f"  Convergence plot saved: {savefile}")


# ===========================================================================
# Save/load results
# ===========================================================================

def save_result(result, filename='zeldovich_2d_result.npz'):
    """Save result dict to npz for later comparison."""
    save_dict = {}
    for key, val in result.items():
        if key in ('Ts', 'phis', 'its_per_step'):
            # These are lists — convert to arrays or save individually
            if key == 'its_per_step':
                save_dict[key] = np.array(val)
            else:
                # Save ALL snapshots so we have data at every time step
                save_dict[key] = np.array(val)
        elif isinstance(val, (np.ndarray, float, int)):
            save_dict[key] = val
        elif isinstance(val, str):
            save_dict[key] = np.array(val)

    np.savez_compressed(filename, **save_dict)
    print(f"  Result saved: {filename}")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="2-D Zel'dovich Wave — convergence study version",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Quick test:           python zeldovich_wave_2d_converge.py
  Convergence study:    python zeldovich_wave_2d_converge.py --study
  Finer run:            python zeldovich_wave_2d_converge.py --Ix 100 --N 8 --tfinal 3.0
  Production:           python zeldovich_wave_2d_converge.py --Ix 200 --Iy 200 --N 8 --tfinal 5.0
        """)

    parser.add_argument('--Ix', type=int, default=40, help='Cells in x (default: 40)')
    parser.add_argument('--Iy', type=int, default=None, help='Cells in y (default: same as Ix)')
    parser.add_argument('--N', type=int, default=4, help='Quadrature order (default: 4)')
    parser.add_argument('--tfinal', type=float, default=1.0, help='Final time in ns (default: 1.0)')
    parser.add_argument('--t-start', type=float, default=0.1, help='Start time in ns (default: 0.1)')
    parser.add_argument('--Lx', type=float, default=None, help='Domain size (auto if not given)')
    parser.add_argument('--quad', type=str, default='level_symmetric',
                        choices=['level_symmetric', 'product_square', 'product_triangular'])
    parser.add_argument('--stretch', type=float, default=1.02,
                        help='Grid stretching >1 refines near origin (default: 1.02)')
    parser.add_argument('--dt-max', type=float, default=1e-3, help='Max time step')
    parser.add_argument('--W', type=int, default=3, help='T* outer iterations (default: 3)')
    parser.add_argument('--K', type=int, default=10, help='DMD snapshots')
    parser.add_argument('--maxits', type=int, default=100, help='Max source iterations')
    parser.add_argument('--no-dmd', action='store_true', help='Disable DMD acceleration')
    parser.add_argument('--loud', action='store_true', help='Verbose output')
    parser.add_argument('--study', action='store_true', help='Run convergence study')
    parser.add_argument('--save', type=str, default=None,
                        help='Save result to this npz file')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting')

    args = parser.parse_args()

    if args.study:
        # --- Convergence study mode ---
        table, all_results = run_convergence_study(
            t_eval=min(1.0, args.tfinal),
            tfinal=args.tfinal,
            t_start=args.t_start,
        )
        if not args.no_plot:
            plot_convergence(table)
            # Plot the finest result
            plot_publication(all_results[-1])
    else:
        # --- Single run mode ---
        Iy = args.Iy if args.Iy is not None else args.Ix

        # Auto-size domain to accommodate the front with margin
        if args.Lx is not None:
            Lx = args.Lx
        else:
            _, R_f = zeldovich_self_similar(np.array([0.0]), args.tfinal, N=2)
            Lx = max(0.5, R_f * 2.0)  # 2× front radius gives good margin
        Ly = Lx

        result = run_zeldovich_converge(
            Ix=args.Ix, Iy=Iy, N_quad=args.N,
            quad_type=args.quad,
            Lx=Lx, Ly=Ly,
            t_start=args.t_start, tfinal=args.tfinal,
            dt_min=1e-5, dt_max=args.dt_max,
            K=args.K, maxits=args.maxits, W=args.W,
            stretch=args.stretch,
            use_dmd=not args.no_dmd,
            LOUD=args.loud,
        )

        # Compute and print error
        L2, Linf, t_act = compute_radial_error(result)
        print(f"\n  Error at t={t_act:.3f} ns:  L2={L2:.3e}  Linf={Linf:.3e}")

        if args.save:
            save_result(result, args.save)
        else:
            tag = f"I{args.Ix}_N{args.N}_t{args.tfinal:.1f}"
            save_result(result, f"zeldovich_2d_{tag}.npz")

        if not args.no_plot:
            tag = f"I{args.Ix}_N{args.N}_t{args.tfinal:.1f}"
            plot_publication(result, savefile=f'zeldovich_2d_{tag}.pdf')
