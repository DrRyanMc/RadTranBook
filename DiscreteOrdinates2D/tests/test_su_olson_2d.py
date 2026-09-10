import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

"""
Su-Olson test problem in 2-D S_N transport.

Solves the same Su-Olson problem as the 1-D version but in 2-D Cartesian
geometry with the 2-D corner-balance solver. The problem is effectively 1-D
(source is a slab in x, solution independent of y) so results should match
the 1-D reference data from Su & Olson (1997).

Problem setup:
  - 2-D Cartesian geometry, domain [0, Lx] × [0, Ly]
  - Radiation source Q = a c T_0^4 in x ∈ [0, 0.5] cm for t < 10 tau
  - sigma_a = 1.0 cm^{-1} (constant), no scattering
  - Material energy: e = a T^4  (radiation-dominated EOS)
  - tau = 1 / (c sigma_a) ~ 0.03336 ns
  - Reflecting BC at x = 0  (left boundary)
  - Vacuum BC at x = Lx  (right boundary)
  - Reflecting BCs at y = 0 and y = Ly  (making it effectively 1-D)
  - Output at: 1.0, 3.16228, 10.0  mean free times

Run from the DiscreteOrdinates2D directory:
    python problems/test_su_olson_2d.py
"""

import sys
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from DiscreteOrdinates2D.src.sn_solver_2d import temp_solve_2d, c, a, ac
from DiscreteOrdinates2D.src.quadratures import get_2d_quadrature


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
A_RAD = a
C_LIGHT = c
AC = ac
T_0 = 1.0  # reference temperature (keV)

# ---------------------------------------------------------------------------
# Su-Olson reference data (from Su & Olson 1997, Tables II-III)
# ---------------------------------------------------------------------------
su_olson_x = np.array([
    0.01, 0.1, 0.17783, 0.31623, 0.45, 0.5,
    0.56234, 0.75, 1.0, 1.33352, 1.77828, 3.16228,
    5.62341, 10.0, 17.78279])

su_olson_tau = np.array([
    0.1, 0.31623, 1.0, 3.16228, 10.0, 31.6228, 100.0])

# Transport reference: radiation energy density (normalised)
transport_rad_energy = np.array([
    [0.09531, 0.27526, 0.64308, 1.20052, 2.23575, 0.69020, 0.35720],
    [0.09531, 0.27526, 0.63585, 1.18869, 2.21944, 0.68974, 0.35714],
    [0.09532, 0.27527, 0.61958, 1.16190, 2.18344, 0.68878, 0.35702],
    [0.09529, 0.26262, 0.56187, 1.07175, 2.06448, 0.68569, 0.35664],
    [0.08823, 0.20312, 0.44711, 0.90951, 1.86072, 0.68111, 0.35599],
    [0.04765, 0.13762, 0.35801, 0.79902, 1.73178, 0.67908, 0.35574],
    [0.00375, 0.06277, 0.25374, 0.66678, 1.57496, 0.67619, 0.35538],
    [np.nan, 0.00280, 0.11430, 0.44675, 1.27398, 0.66548, 0.35393],
    [np.nan, np.nan, 0.03648, 0.27540, 0.98782, 0.64691, 0.35141],
    [np.nan, np.nan, 0.00291, 0.14531, 0.70822, 0.61538, 0.34697],
    [np.nan, np.nan, np.nan, 0.05968, 0.45016, 0.56353, 0.33924],
    [np.nan, np.nan, np.nan, 0.00123, 0.09673, 0.36965, 0.30346],
    [np.nan, np.nan, np.nan, np.nan, 0.00375, 0.10830, 0.21382],
    [np.nan, np.nan, np.nan, np.nan, np.nan, 0.00390, 0.07200],
    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.00272],
])

# Transport reference: material energy density (normalised)
transport_mat_energy = np.array([
    [0.00468, 0.04093, 0.27126, 0.94670, 2.11186, 0.70499, 0.35914],
    [0.00468, 0.04093, 0.26839, 0.93712, 2.09585, 0.70452, 0.35908],
    [0.00468, 0.04093, 0.26261, 0.91525, 2.06052, 0.70348, 0.35895],
    [0.00468, 0.04032, 0.23978, 0.84082, 1.94365, 0.70020, 0.35854],
    [0.00455, 0.03314, 0.18826, 0.70286, 1.74291, 0.69532, 0.35793],
    [0.00234, 0.02046, 0.14187, 0.60492, 1.61536, 0.69308, 0.35766],
    [0.00005, 0.00635, 0.08838, 0.48843, 1.46027, 0.68994, 0.35728],
    [np.nan, 0.00005, 0.03014, 0.30656, 1.16591, 0.67850, 0.35581],
    [np.nan, np.nan, 0.00625, 0.17519, 0.88992, 0.65868, 0.35326],
    [np.nan, np.nan, 0.00017, 0.08352, 0.62521, 0.62507, 0.34875],
    [np.nan, np.nan, np.nan, 0.02935, 0.38688, 0.57003, 0.34086],
    [np.nan, np.nan, np.nan, 0.00025, 0.07642, 0.36727, 0.30517],
    [np.nan, np.nan, np.nan, np.nan, 0.00253, 0.10312, 0.21377],
    [np.nan, np.nan, np.nan, np.nan, np.nan, 0.00342, 0.07122],
    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.00261],
])


# ---------------------------------------------------------------------------
# Problem setup & run
# ---------------------------------------------------------------------------
def setup_and_run(Ix=200, Iy=3, N_quad=4, quad_type='level_symmetric',
                  output_tau=(1.0, 3.16228, 10.0),
                  LOUD=False, use_dmd=True, W=10):
    """Run the Su-Olson problem in 2-D.

    Uses reflecting BC at x=0 (the 2-D solver handles this directly, no
    need for the doubled-domain trick used in 1-D).

    Parameters
    ----------
    Ix : int
        Number of cells in x.
    Iy : int
        Number of cells in y.
    N_quad : int
        Quadrature order.
    quad_type : str
        Quadrature type.
    output_tau : tuple of float
        Output times in mean-free-times.

    Returns
    -------
    results : dict
    """
    # --- Problem parameters ---
    sigma_a = 1.0       # cm^{-1}
    Lx = 20.0           # domain length in x (cm)
    if (np.max(output_tau) < 100.0):
        Lx = 6.0       # smaller domain for early-time outputs
    Ly = 0.5            # domain length in y (small, problem is 1-D)
    source_half = 0.5   # source occupies x ∈ [0, 0.5]
    tau_mft = 1.0 / (C_LIGHT * sigma_a)
    source_duration = 10.0 * tau_mft
    Tinit = 0.001       # keV

    # --- Grid ---
    dx_arr = np.full(Ix, Lx / Ix)
    dy_arr = np.full(Iy, Ly / Iy)

    # Cell centers
    x_faces = np.zeros(Ix + 1)
    for i in range(Ix):
        x_faces[i+1] = x_faces[i] + dx_arr[i]
    x_centers = 0.5 * (x_faces[:-1] + x_faces[1:])

    # --- Quadrature ---
    Omega_x, Omega_y, weights = get_2d_quadrature(quad_type, N_quad)
    M = len(Omega_x)

    print(f"Su-Olson 2-D: Ix={Ix}, Iy={Iy}  N_quad={N_quad} ({quad_type}), W={W}, use_dmd={use_dmd}")
    print(f"  sigma_a = {sigma_a}, Lx = {Lx}, Ly = {Ly}")
    print(f"  tau (mfp time) = {tau_mft:.6e} ns")
    print(f"  source duration = {source_duration:.6e} ns (10 tau)")
    print(f"  output times: {output_tau} tau")

    # --- Material functions ---
    # EOS: e = a T^4 (radiation-dominated)
    def EOS(T):
        return A_RAD * np.maximum(T, 0.0)**4

    def invEOS(e):
        return (np.maximum(e, 0.0) / A_RAD)**0.25

    # Opacity: constant
    def sigma_func(T):
        return np.full_like(T, sigma_a)

    # No scattering
    def scat_func(T):
        return np.zeros_like(T)

    # --- External source ---
    # Q = ac * T_0^4 in x < 0.5, isotropic, for t < source_duration
    Q_val = AC * T_0**4
    q_on = np.zeros((Ix, Iy, 4))
    for i in range(Ix):
        if x_centers[i] < source_half:
            q_on[i, :, :] = Q_val
    q_off = np.zeros((Ix, Iy, 4))

    # --- Initial conditions ---
    T_init = np.full((Ix, Iy, 4), Tinit)
    phi_init = AC * T_init**4

    # --- Boundary conditions ---
    # x=0: reflecting (handled by solver flag)
    # x=Lx: vacuum (zero incoming)
    # y=0, y=Ly: reflecting (handled by solver flags)
    def BCs_func(t):
        return {'xlo': None, 'xhi': None, 'ylo': None, 'yhi': None}

    # --- Time stepping parameters ---
    dt_min = 0.005 * tau_mft
    dt_max = 0.1 * tau_mft

    # --- Split into source-on and source-off phases ---
    output_tau = np.array(sorted(output_tau))
    output_times_ns = output_tau * tau_mft
    early_mask = output_times_ns <= source_duration * (1 + 1e-10)
    late_mask = ~early_mask
    early_outputs = output_times_ns[early_mask]
    late_outputs = output_times_ns[late_mask]

    solutions = {}
    total_iterations = 0

    # ---- Phase 1: source ON ----
    phase1_final = source_duration
    if early_outputs.size > 0:
        phase1_final = min(phase1_final, early_outputs[-1])

    print(f"\n--- Phase 1: source ON, 0 -> {phase1_final / tau_mft:.2f} tau ---")
    phis1, Ts1, its1, ts1, its_per1 = temp_solve_2d(
        Ix, Iy, dx_arr, dy_arr,
        q_on, sigma_func, scat_func,
        quad_type, N_quad,
        BCs_func, EOS, invEOS,
        phi_init, T_init,
        dt_min=dt_min, dt_max=dt_max, tfinal=phase1_final,
        tolerance=1e-5, Linf_tol=1e-3, maxits=20,
        K=30, R=3,W=W,
        tau_T=1e-3,
        reflect_xlo=True, reflect_xhi=False,
        reflect_ylo=True, reflect_yhi=True,
        use_dmd=use_dmd,
        LOUD=LOUD,
        print_stride=20,
        time_outputs=early_outputs,
    )
    total_iterations += its1
    print(f"  Phase 1 sweeps: {its1}")

    # Extract early-time snapshots
    for tau_val in output_tau[early_mask]:
        t_ns = tau_val * tau_mft
        tstep = np.argmin(np.abs(t_ns - ts1))
        phi_snap = phis1[tstep]   # (Ix, Iy, 4)
        T_snap = Ts1[tstep]       # (Ix, Iy, 4)
        # Average over y and corners for 1-D comparison
        phi_1d = np.mean(phi_snap, axis=(1, 2))
        T_1d = np.mean(T_snap, axis=(1, 2))
        E_rad_norm = phi_1d / (AC * T_0**4)
        E_mat_norm = T_1d**4 / T_0**4
        solutions[tau_val] = {
            'E_rad': E_rad_norm, 'E_mat': E_mat_norm,
            't_ns': ts1[tstep],
        }
        print(f"  Saved tau = {tau_val:.4f}  (t = {ts1[tstep]:.6e} ns)")

    # ---- Phase 2: source OFF ----
    if late_outputs.size > 0:
        phi2 = phis1[-1].copy()
        T2 = Ts1[-1].copy()
        late_final = late_outputs[-1] - source_duration

        late_outs_rel = late_outputs - source_duration
        print(f"\n--- Phase 2: source OFF, 0 -> {late_final / tau_mft:.2f} tau ---")
        phis2, Ts2, its2, ts2, its_per2 = temp_solve_2d(
            Ix, Iy, dx_arr, dy_arr,
            q_off, sigma_func, scat_func,
            quad_type, N_quad,
            BCs_func, EOS, invEOS,
            phi2, T2,
            dt_min=dt_min, dt_max=dt_max, tfinal=late_final,
            tolerance=1e-5, Linf_tol=1e-3, maxits=20,
            K=30, R=3,W=3,
            tau_T=1e-3,
            reflect_xlo=True, reflect_xhi=False,
            reflect_ylo=True, reflect_yhi=True,
            use_dmd=use_dmd,
            LOUD=LOUD,
            print_stride=20,
            time_outputs=late_outs_rel,
        )
        total_iterations += its2
        print(f"  Phase 2 sweeps: {its2}")

        for tau_val in output_tau[late_mask]:
            t_rel = tau_val * tau_mft - source_duration
            tstep = np.argmin(np.abs(t_rel - ts2))
            phi_snap = phis2[tstep]
            T_snap = Ts2[tstep]
            phi_1d = np.mean(phi_snap, axis=(1, 2))
            T_1d = np.mean(T_snap, axis=(1, 2))
            E_rad_norm = phi_1d / (AC * T_0**4)
            E_mat_norm = T_1d**4 / T_0**4
            solutions[tau_val] = {
                'E_rad': E_rad_norm, 'E_mat': E_mat_norm,
                't_ns': ts2[tstep] + source_duration,
            }
            print(f"  Saved tau = {tau_val:.4f}  (t = {solutions[tau_val]['t_ns']:.6e} ns)")

    print(f"\nTotal transport sweeps: {total_iterations}")
    return {
        'solutions': solutions, 'x': x_centers,
        'Ix': Ix, 'Iy': Iy, 'Lx': Lx, 'tau_mft': tau_mft,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_results(results, savefile='su_olson_2d.png'):
    """Plot 2-D S_N results vs. tabulated transport reference data."""
    solutions = results['solutions']
    x = results['x']

    taus_available = sorted(solutions.keys())
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(taus_available)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for idx, tau_val in enumerate(taus_available):
        sol = solutions[tau_val]
        col = colors[idx]
        label = rf'$\tau={tau_val:.2f}$'

        ax1.plot(x, sol['E_rad'], '-', color=col, lw=1.5, label=label)
        ax2.plot(x, sol['E_mat'], '-', color=col, lw=1.5, label=label)

        # Overlay transport reference
        ti = np.argmin(np.abs(su_olson_tau - tau_val))
        if abs(su_olson_tau[ti] - tau_val) < 0.01 * tau_val:
            ref_rad = transport_rad_energy[:, ti]
            ref_mat = transport_mat_energy[:, ti]
            valid = ~np.isnan(ref_rad)
            ax1.plot(su_olson_x[valid], ref_rad[valid], 's',
                     color=col, ms=5, mec='k', mew=0.5, alpha=0.8)
            valid = ~np.isnan(ref_mat)
            ax2.plot(su_olson_x[valid], ref_mat[valid], 's',
                     color=col, ms=5, mec='k', mew=0.5, alpha=0.8)

    for ax, title, ylabel in [
        (ax1, 'Radiation Energy', r'$E_r / (a\,T_0^4)$'),
        (ax2, 'Material Energy', r'$e / (a\,T_0^4) = (T/T_0)^4$')
    ]:
        ax.set_xlabel('Position x (cm)', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        #ax.set_yscale('log')
        #ax.set_xscale('log')
        #ax.set_xlim(0.05, 12)
        #ax.set_ylim(1e-3, 3)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    # Custom legend entry for reference markers
    ref_handle = Line2D([], [], marker='s', color='gray', ms=5,
                        mec='k', mew=0.5, ls='', label='Su-Olson ref')
    ax1.legend(handles=list(ax1.get_legend_handles_labels()[0]) + [ref_handle],
               fontsize=9, loc='best')

    fig.suptitle(r'Su-Olson Problem — 2-D S$_N$ Corner Balance', fontsize=14,
                 fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(savefile, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to {savefile}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Su-Olson test in 2-D S_N transport')
    parser.add_argument('--Ix', type=int, default=200,
                        help='Cells in x (default: 200)')
    parser.add_argument('--Iy', type=int, default=3,
                        help='Cells in y (default: 3)')
    parser.add_argument('--N', type=int, default=4,
                        help='Quadrature order (default: 4)')
    parser.add_argument('--quad', type=str, default='level_symmetric',
                        help='Quadrature type')
    parser.add_argument('--output-tau', type=float, nargs='+',
                        default=[1.0, 3.16228, 10.0],
                        help='Output times in mean-free-times')
    parser.add_argument('--no-dmd', action='store_true',
                        help='Disable DMD acceleration')
    parser.add_argument('--loud', action='store_true',
                        help='Verbose output')
    parser.add_argument('--W', type=float, default=10,
                        help='number of temperature iterations(default: 10)')
    args = parser.parse_args()

    results = setup_and_run(
        Ix=args.Ix, Iy=args.Iy, N_quad=args.N,
        quad_type=args.quad,
        output_tau=tuple(args.output_tau),
        use_dmd=not args.no_dmd,
        LOUD=args.loud, W=args.W
    )

    plot_results(results)
