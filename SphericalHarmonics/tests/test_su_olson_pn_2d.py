"""
Su-Olson 1-D test problem run with the 2-D P_N (spherical-harmonics) SCB solver.

Problem setup (identical to test_su_olson_2d.py):
  domain:     [0, Lx] × [0, Ly],  Ly = 0.5 cm (small, problem is 1-D)
  source:     Q = a c T_0^4  in x ∈ [0, 0.5] cm  for t < 10 τ
  σ_a = 1 cm⁻¹  (constant),  no scattering
  EOS:        e = a T^4  (radiation-dominated)
  τ = 1/(c σ_a) ≈ 0.03336 ns  (mean-free time)
  BCs:        x=0 reflecting,  x=Lx vacuum,  y=0/Ly reflecting

Validation:  output at τ = 1, 3.16228, 10 and compare to the Su-Olson
             reference tables (Su & Olson 1997, Tables II-III).

Run from the SphericalHarmonics/problems directory:
    python test_su_olson_pn_2d.py
or with options:
    python test_su_olson_pn_2d.py --N 3 --Ix 100 --tau 1.0 3.16228 10.0
"""

import sys
import os
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---- pn_solver_2d lives one level up (SphericalHarmonics/) ----------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from SphericalHarmonics.src.pn_solver_2d import temp_solve_pn_2d, c, a, ac

JACOBIAN_DIR = os.path.join(os.path.dirname(__file__), '..', 'Jacobians')

# ===========================================================================
# Su-Olson reference data  (Su & Olson 1997, Tables II-III)
# ===========================================================================
su_olson_x = np.array([
    0.01, 0.1, 0.17783, 0.31623, 0.45, 0.5,
    0.56234, 0.75, 1.0, 1.33352, 1.77828, 3.16228,
    5.62341, 10.0, 17.78279])

su_olson_tau = np.array([
    0.1, 0.31623, 1.0, 3.16228, 10.0, 31.6228, 100.0])

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

transport_mat_energy = np.array([
    [0.00468, 0.04093, 0.27126, 0.94670, 2.11186, 0.70499, 0.35914],
    [0.00468, 0.04093, 0.26839, 0.93712, 2.09585, 0.70452, 0.35908],
    [0.00468, 0.04093, 0.26261, 0.91525, 2.06052, 0.70348, 0.35895],
    [0.00468, 0.04093, 0.24008, 0.85064, 1.96238, 0.70069, 0.35857],
    [0.00468, 0.03769, 0.19367, 0.72946, 1.79024, 0.69541, 0.35782],
    [0.00461, 0.02565, 0.15584, 0.64419, 1.67183, 0.69316, 0.35755],
    [0.00247, 0.01171, 0.11120, 0.53814, 1.52418, 0.69003, 0.35716],
    [np.nan, 0.00052, 0.05043, 0.36313, 1.23667, 0.67936, 0.35571],
    [np.nan, np.nan, 0.01621, 0.22565, 0.96382, 0.66109, 0.35320],
    [np.nan, np.nan, 0.00130, 0.11978, 0.69363, 0.63011, 0.34878],
    [np.nan, np.nan, np.nan, 0.04930, 0.44231, 0.57882, 0.34109],
    [np.nan, np.nan, np.nan, 0.00102, 0.09526, 0.38153, 0.30567],
    [np.nan, np.nan, np.nan, np.nan, 0.00370, 0.11231, 0.21629],
    [np.nan, np.nan, np.nan, np.nan, np.nan, 0.00406, 0.07299],
    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.00276],
])


# ===========================================================================
# Setup and run
# ===========================================================================

def setup_and_run(N_pn=1, Ix=200, Iy=3,
                  output_tau=(1.0, 3.16228, 10.0),
                  Lx=None,
                  plot=True,
                  save_prefix=None):
    """
    Run the Su-Olson problem with P_N SCB and return solutions at output_tau.

    Parameters
    ----------
    N_pn       : P_N order (1, 3, 5, …).
    Ix         : number of x cells.
    Iy         : number of z cells (3 is fine for 1-D).
    Lx         : domain length in x (default: auto based on max tau).
    output_tau : mean-free-times at which to record the solution.
    plot       : if True, make a comparison plot.
    save_prefix: if not None, save plots as <save_prefix>_tau_<tau>.png.

    Returns
    -------
    solutions : dict  tau → {x, rad_E, mat_E, phi, T}
    """
    sigma_a = 1.0           # cm⁻¹
    T_0     = 1.0           # keV
    if Lx is None:
        Lx = 20.0 if max(output_tau) >= 10.0 else 6.0
    Ly      = 0.5           # small z extent (problem is 1-D)

    tau_mft         = 1.0 / (c * sigma_a)          # mean-free time  (ns)
    source_duration = 10.0 * tau_mft                # source on for 10τ
    t_end_total     = max(output_tau) * tau_mft

    dx_arr = np.full(Ix, Lx / Ix)
    dy_arr = np.full(Iy, Ly / Iy)

    # Cell-centre x positions (using NE corner = centre of NE subcell ≈ x_c)
    x_faces = np.concatenate([[0.0], np.cumsum(dx_arr)])
    x_c     = 0.5 * (x_faces[:-1] + x_faces[1:])   # (Ix,)

    print(f"\nSu-Olson 2-D P_{N_pn}:  Ix={Ix}, Iy={Iy}")
    print(f"  σ_a = {sigma_a},  Lx = {Lx},  Ly = {Ly}")
    print(f"  τ_mft = {tau_mft:.6e} ns,  source ends at {source_duration:.4e} ns")
    print(f"  output at τ = {list(output_tau)}")

    # EOS: e = a T^4
    def EOS(T):
        return a * T ** 4

    def invEOS(e):
        return (np.maximum(e, 0.0) / a) ** 0.25

    def sigma_func(T):
        return np.full_like(T, sigma_a)

    def scat_func(T):
        return np.zeros_like(T)

    # Source: isotropic, Q = ac T_0^4 in x ≤ 0.5 cm
    q_on  = np.zeros((Ix, Iy, 4))
    for i in range(Ix):
        if x_c[i] <= 0.5:
            q_on[i, :, :] = ac * T_0 ** 4

    q_off = np.zeros((Ix, Iy, 4))

    T_floor = 1e-6
    Tinit   = T_floor

    phi_init = np.full((Ix, Iy, 4), a * c * Tinit ** 4)
    T_init   = np.full((Ix, Iy, 4), Tinit)

    # Time-step parameters
    dt_min  = 0.01 * tau_mft
    dt_max  = 0.1   * tau_mft
    dt_start = dt_min

    # Sorted output times
    output_tau  = sorted(output_tau)
    output_ns   = [tv * tau_mft for tv in output_tau]

    # -----------------------------------------------------------------------
    # Time-stepping: run through each output time in sequence
    # -----------------------------------------------------------------------
    solutions = {}
    phi   = phi_init.copy()
    T     = T_init.copy()
    I_cur = None               # will be set after first run

    t_done = 0.0

    def _run_segment(phi, T, I_cur, q, t_start, t_end_abs):
        """Run one time segment; t_end is the absolute target time."""
        duration = t_end_abs - t_start
        assert duration > 1e-15, f"zero-duration segment: {t_start} → {t_end_abs}"
        phi_new, T_new, I_new, _t, _h = temp_solve_pn_2d(
            Ix, Iy, dx_arr, dy_arr,
            q, sigma_func, scat_func,
            N_pn, JACOBIAN_DIR,
            EOS, invEOS,
            phi, T,
            I_init=I_cur,
            dt_start=dt_start,
            t_end=duration,            # ← duration, NOT absolute time
            reflect_xlo=True,  reflect_xhi=False,
            reflect_ylo=True,  reflect_yhi=True,
            tolerance=1e-8,
            W=0,
            n_gs=1,
            dt_max=dt_max,
            T_floor=T_floor,
            print_stride=50,
        )
        return phi_new, T_new, I_new, t_end_abs   # return absolute time reached

    for o_idx, t_target in enumerate(output_ns):
        tau_val = output_tau[o_idx]

        if t_done < source_duration <= t_target:
            # Straddles the source-off transition
            print(f"\n--- P{N_pn}: source ON  [{t_done:.4e} → {source_duration:.4e} ns] ---")
            phi, T, I_cur, t_done = _run_segment(phi, T, I_cur, q_on,
                                                  t_done, source_duration)
            if t_done < t_target:
                print(f"\n--- P{N_pn}: source OFF [{t_done:.4e} → {t_target:.4e} ns] ---")
                phi, T, I_cur, t_done = _run_segment(phi, T, I_cur, q_off,
                                                      t_done, t_target)
        else:
            q_use = q_on if t_done < source_duration else q_off
            label = "ON" if t_done < source_duration else "OFF"
            print(f"\n--- P{N_pn}: source {label} [{t_done:.4e} → {t_target:.4e} ns] ---")
            phi, T, I_cur, t_done = _run_segment(phi, T, I_cur, q_use,
                                                  t_done, t_target)

        # Average phi over y-cells and corners to get 1-D profile
        # phi[i, j, c] → average over j and all 4 corners
        phi_1d = np.mean(phi[:, :, :], axis=(1, 2))       # (Ix,)
        T_1d   = np.mean(T[:, :, :],   axis=(1, 2))       # (Ix,)
        rad_E  = phi_1d / (c)                               # radiation energy density
        mat_E  = a * T_1d ** 4                              # material energy density

        # Normalise by reference energy density
        E0 = a * T_0 ** 4
        rad_E_norm = rad_E / E0
        mat_E_norm = mat_E / E0

        solutions[tau_val] = dict(
            x=x_c, rad_E=rad_E_norm, mat_E=mat_E_norm,
            phi=phi.copy(), T=T.copy(), I=I_cur.copy()
        )
        print(f"\n  Saved τ = {tau_val}  (t = {t_done:.6e} ns)")

    # -----------------------------------------------------------------------
    # Conservation check
    # -----------------------------------------------------------------------
    _print_conservation(solutions, output_tau, dx_arr, q_on, tau_mft,
                        sigma_a=sigma_a, T_0=T_0)

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------
    if plot:
        _plot_solutions(solutions, output_tau, N_pn, save_prefix)

    return solutions


def _print_conservation(solutions, output_tau, dx_arr, q_on, tau_mft, sigma_a=1.0, T_0=1.0):
    """
    Check energy conservation and source normalisation.

    The transport + material energy equations add to:
        d/dt ∫(E_rad + E_mat) dx  =  Q · Δx_src  −  boundary_leakage

    Integrating and normalising by a T₀⁴:

        ∫ (E_rad_norm + E_mat_norm) dx  ≤  c · Δx_src · t_on   [cm]

    where t_on = min(t, 10τ).  The inequality accounts for radiation that
    has escaped through the vacuum boundary.

    If the source had the WRONG sign on √(4π)  (Q·√(4π) instead of Q/√(4π)),
    the deposited energy would be √(4π) ≈ 3.54× too large, and the fraction
    would exceed 354% — immediately visible.

    Parameters
    ----------
    solutions : dict  tau → {x, rad_E, mat_E, ...}  (normalised 1-D profiles)
    dx_arr    : (Ix,) cell widths.
    q_on      : (Ix, Iy, 4) source array used in the run.
    tau_mft   : mean-free time [ns].
    """
    import math
    sqrt4pi = math.sqrt(4.0 * math.pi)

    source_mask  = q_on[:, 0, 0] > 0.0
    source_width = float(np.sum(dx_arr[source_mask]))
    t_source_ns  = 10.0 * tau_mft

    # -------- source magnitude check ----------------------------------------
    Q_val  = float(np.max(q_on))
    Q_exp  = ac * T_0**4
    print("\n=== Source normalisation check ===")
    print(f"  Q in problem  : {Q_val:.5e} GJ/cm³/ns")
    print(f"  Expected ac T₀⁴: {Q_exp:.5e} GJ/cm³/ns  "
          f"({'✓' if abs(Q_val/Q_exp - 1) < 1e-6 else '✗ MISMATCH'})")
    print(f"  P_N 00 source Q̂₀₀ = Q/√(4π) = {Q_val/sqrt4pi:.5e}")
    print(f"  Scalar flux check: Q/σ_a = {Q_val/sigma_a:.5e}  ←  should ≈ ac T₀⁴ = {Q_exp:.5e}")
    print(f"  (If correct, these agree; if source had extra √(4π) they would differ by {sqrt4pi:.3f}×)")

    # -------- conservation balance -------------------------------------------
    print("\n=== Energy conservation (∫·dx normalised by a T₀⁴) ===")
    print(f"  Source width Δx = {source_width:.4f} cm")
    print(f"  Expected E_dep at τ=1: c·Δx·τ_mft = {c*source_width*tau_mft:.4f} cm  "
          f"(= Δx/σ_a = {source_width/sigma_a:.4f} cm)")
    print()
    print(f"  {'τ':>8}  {'∫E_rad dx':>12}  {'∫E_mat dx':>12}  "
          f"{'sum':>10}  {'E_dep':>10}  {'fraction':>10}")
    print("  " + "-"*72)

    for tau_val in sorted(solutions.keys()):
        sol  = solutions[tau_val]
        t_ns = tau_val * tau_mft

        E_rad_int = float(np.dot(sol['rad_E'], dx_arr))
        E_mat_int = float(np.dot(sol['mat_E'], dx_arr))
        E_tot     = E_rad_int + E_mat_int

        t_on  = min(t_ns, t_source_ns)
        E_dep = c * source_width * t_on      # [cm], same normalisation

        pct = 100.0 * E_tot / E_dep if E_dep > 0 else 0.0
        flag = "✓" if pct <= 101.0 else f"✗ ({pct:.0f}% > 100% — energy created!)"
        print(f"  {tau_val:8.4g}  {E_rad_int:12.5f}  {E_mat_int:12.5f}  "
              f"{E_tot:10.5f}  {E_dep:10.5f}  {pct:8.1f}%  {flag}")


def _plot_solutions(solutions, output_tau, N_pn, save_prefix=None):
    tau_to_col = {
        0.1: 0, 0.31623: 1, 1.0: 2, 3.16228: 3,
        10.0: 4, 31.6228: 5, 100.0: 6
    }
    colors = plt.cm.tab10(np.linspace(0, 1, len(output_tau)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax_r, ax_m = axes

    pn_handles_r = []
    pn_handles_m = []

    for k, tau_val in enumerate(sorted(solutions.keys())):
        sol = solutions[tau_val]
        x   = sol['x']
        col = tau_to_col.get(tau_val, None)
        clr = colors[k]

        ln_r, = ax_r.plot(x, np.maximum(sol['rad_E'], 1e-8), color=clr,
                               linewidth=1.8, label=f'P{N_pn} τ={tau_val:.3g}')
        ln_m, = ax_m.plot(x, np.maximum(sol['mat_E'], 1e-8), color=clr,
                               linewidth=1.8, label=f'P{N_pn} τ={tau_val:.3g}')
        pn_handles_r.append(ln_r)
        pn_handles_m.append(ln_m)

        # Reference data
        if col is not None:
            ref_x  = su_olson_x
            ref_re = transport_rad_energy[:, col]
            ref_me = transport_mat_energy[:, col]
            mask_r = ~np.isnan(ref_re)
            mask_m = ~np.isnan(ref_me)
            ax_r.plot(ref_x[mask_r], ref_re[mask_r],
                          'o', color=clr, markersize=4, alpha=0.7)
            ax_m.plot(ref_x[mask_m], ref_me[mask_m],
                          's', color=clr, markersize=4, alpha=0.7)

    from matplotlib.lines import Line2D
    ref_marker = Line2D([0], [0], linestyle='', color='k', marker='o',
                        markersize=4, label='Su-Olson (1997) reference')
    ax_r.legend(handles=pn_handles_r + [ref_marker], fontsize=8, loc='lower left')
    ax_m.legend(handles=pn_handles_m + [ref_marker], fontsize=8, loc='lower left')

    ax_r.set_xlabel('x  (cm)');  ax_r.set_ylabel('E_rad / (a T₀⁴)')
    ax_m.set_xlabel('x  (cm)');  ax_m.set_ylabel('E_mat / (a T₀⁴)')
    ax_r.set_title(f'Radiation energy density   P{N_pn}')
    ax_m.set_title(f'Material energy density    P{N_pn}')
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.suptitle(f'Su-Olson 1-D benchmark — 2-D P{N_pn} SCB solver', fontsize=12)
    fig.tight_layout()

    fname = (f"{save_prefix}_P{N_pn}.png"
             if save_prefix else f"su_olson_pn{N_pn}.png")
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {fname}")
    plt.close(fig)


# ===========================================================================
# CLI
# ===========================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Su-Olson benchmark with 2-D P_N SCB solver.')
    parser.add_argument('--N',   type=int,   default=1,
                        help='P_N order (default: 1)')
    parser.add_argument('--Ix',  type=int,   default=100,
                        help='x cells (default: 100)')
    parser.add_argument('--Iy',  type=int,   default=3,
                        help='z cells (default: 3)')
    #add argument for Lx
    parser.add_argument('--Lx',  type=float, default=None,
                        help='domain length in x (default: auto based on max tau)')
    parser.add_argument('--tau', type=float, nargs='+',
                        default=[1.0, 3.16228, 10.0],
                        help='output mean-free-times (default: 1 3.16 10)')
    parser.add_argument('--no-plot', action='store_true',
                        help='skip plotting')
    parser.add_argument('--save', type=str, default=None,
                        help='prefix for saved plot filenames')
    args = parser.parse_args()

    setup_and_run(
        N_pn=args.N,
        Ix=args.Ix,
        Iy=args.Iy,
        output_tau=args.tau,
        Lx=args.Lx,
        plot=not args.no_plot,
        save_prefix=args.save,
    )
