#!/usr/bin/env python3
"""
Infinite-medium multigroup S_N test (exponential-band opacity).

Mirrors MG_IMC/test_infinite_medium_multigroup_expband.py using 1-D S_N
transport instead of IMC.

Infinite-medium condition is enforced by running a SINGLE spatial cell
(I=1, hx=1 cm) with perfectly reflecting boundaries on both sides.
The ``build_reflecting_BCs`` helper in ``sn_solver`` propagates the outgoing
angular flux from the previous step as the incoming flux for the mirror
direction, so no energy escapes the domain.

Problem setup:
  - Energy range: [1e-4, 20] keV, G log-spaced groups
  - Group opacity: normalised Planck-band form
      sigma_g(T) = sigma0*(exp(-v1/T) - exp(-v2/T)) / (sqrt(T)*norm(v1,v2,T))
  - sigma0 = 10
  - Initial group radiation: non-Planckian Planck spectrum at Trad=0.5 keV
    normalised with Tc=1 keV (same formula as IMC test)
  - Initial material temperature: T_mat0 = 0.4 keV
  - rho*Cv = 0.01 GJ/(cm³·keV)  [linear EOS]

Run from the DiscreteOrdinates directory:
    python problems/test_infinite_medium_multigroup_expband_sn.py
    python problems/test_infinite_medium_multigroup_expband_sn.py --groups 20
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

# ── solver imports ───────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import sn_solver
from sn_solver import build_reflecting_BCs
import mg_sn_solver

# ── project utilities ───────────────────────────────────────────────────────
project_root = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, project_root)
try:
    from utils.plotfuncs import show, hide_spines, font
    HAS_PLOTFUNCS = True
except ImportError:
    HAS_PLOTFUNCS = False

from planck_integrals import Bg as _Bg_scalar, dBgdT as _dBgdT_scalar

# ── Polylog (Li_n) with scipy / mpmath fallback ──────────────────────────────
try:
    from scipy.special import polylog as _scipy_polylog

    def _polylog(n, x):
        return np.asarray(_scipy_polylog(n, x), dtype=float)

except ImportError:
    try:
        from scipy.special import spence as _spence

        def _li2_spence(x):
            return _spence(1.0 - x)

    except ImportError:
        _li2_spence = None

    import mpmath as _mp

    def _polylog(n, x):
        if n == 2 and _li2_spence is not None:
            return np.asarray(_li2_spence(x), dtype=float)
        if np.isscalar(x):
            return float(_mp.polylog(n, x))
        out = np.empty_like(x, dtype=float)
        it = np.nditer(x, flags=['multi_index'])
        for val in it:
            out[it.multi_index] = float(_mp.polylog(n, float(val)))
        return out

# ── Physical constants ───────────────────────────────────────────────────────
C_LIGHT = sn_solver.c    # 29.98 cm/ns
A_RAD   = sn_solver.a    # 0.01372 GJ/(cm³·keV⁴)
AC      = sn_solver.ac   # a·c  [GJ/(cm²·ns·keV⁴)]


# ── Polylog helpers ──────────────────────────────────────────────────────────

def _li2(x):
    return _polylog(2, x)

def _li3(x):
    return _polylog(3, x)

def _li4(x):
    return _polylog(4, x)

def _log1mexp(x):
    """Stable log(1 - exp(-x)) for x > 0."""
    return np.log(-np.expm1(-x))


# ── Analytic initial group radiation energy density ──────────────────────────

def group_energy_density(v1, v2, Tc, Trad, a_rad):
    """Planck group energy density with spectrum at Tc, total energy ∝ Trad⁴."""
    z1 = np.exp(-v1 / Tc)
    z2 = np.exp(-v2 / Tc)
    bracket = (
        6.0 * Tc**3 * (_li4(z1) - _li4(z2))
        + 6.0 * Tc**2 * (v1 * _li3(z1) - v2 * _li3(z2))
        + 3.0 * Tc * (v1**2 * _li2(z1) - v2**2 * _li2(z2))
        - v1**3 * _log1mexp(v1 / Tc)
        + v2**3 * _log1mexp(v2 / Tc)
    )
    return (15.0 * a_rad * Trad**4 / (np.pi**4 * Tc**3)) * bracket


# ── Planck / dBdT wrappers (4π convention to match S_N solver) ───────────────

def _Bg_4pi(El, Eh, T):
    """4π * B_g(T) for group [El, Eh], vectorised over T."""
    T_arr = np.maximum(np.asarray(T, dtype=float), 1e-6)
    result = np.array([_Bg_scalar(El, Eh, t) for t in T_arr.ravel()])
    return (4.0 * np.pi * result).reshape(T_arr.shape)


def _dBdT_4pi(El, Eh, T):
    """4π * dB_g/dT, vectorised over T."""
    T_arr = np.maximum(np.asarray(T, dtype=float), 1e-6)
    result = np.array([_dBgdT_scalar(El, Eh, t) for t in T_arr.ravel()])
    return (4.0 * np.pi * result).reshape(T_arr.shape)


def _make_Bg(El, Eh):
    def Bg(T):
        return _Bg_4pi(El, Eh, T)
    return Bg


def _make_dBdT(El, Eh):
    def dBdT(T):
        return _dBdT_4pi(El, Eh, T)
    return dBdT


# ── Normalised band opacity ──────────────────────────────────────────────────

def make_group_opacity(v1, v2, sigma0=10.0, t_floor=1e-8):
    """Group opacity σ_g(T) for the normalised exponential-band form."""
    def sigma_func(T):
        T_use = np.maximum(np.asarray(T, dtype=float), t_floor)
        z1 = np.exp(-v1 / T_use)
        z2 = np.exp(-v2 / T_use)
        norm_inner = (
            3.0 * T_use * (
                v1**2 * _li2(z1)
                + 2.0 * T_use * (v1 * _li3(z1) + T_use * _li4(z1)
                                 - v2 * _li3(z2) - T_use * _li4(z2))
                - v2**2 * _li2(z2)
            )
            - v1**3 * _log1mexp(v1 / T_use)
            + v2**3 * _log1mexp(v2 / T_use)
        )
        denom_safe = np.where(np.sqrt(T_use) * np.abs(norm_inner) > 1e-300,
                              np.sqrt(T_use) * norm_inner, np.inf)
        return np.maximum(sigma0 * (z1 - z2) / denom_safe, 0.0)
    return sigma_func


# ── Single-step multigroup Fleck-factor transport ────────────────────────────

def _single_mg_step(I, hx, G, sigma_a_funcs, scat_funcs, Bg_funcs, dBdT_funcs,
                    q_ext, N, invEOS, Cv_func,
                    phi_g, psi_g, T_old, e_old,
                    dt, order, K, maxits, R=3,
                    reflect_left=False, reflect_right=False):
    """One time-step of multigroup Fleck-factor S_N transport.

    Uses zero-incoming (vacuum) as the base BCs and applies reflecting BCs
    from ``psi_g`` (previous result) when ``reflect_left`` / ``reflect_right``
    are True.

    Returns
    -------
    phi_g_new, psi_g_new, T_new, e_new, n_iters
    """
    nop1  = order + 1
    icdt  = 1.0 / (C_LIGHT * dt)
    zero_bcs = np.zeros((N, nop1))
    zero_src  = np.zeros((I, N, nop1))

    # Build effective BCs including reflecting contributions from previous psi_g
    def _bc_for_g(g):
        bc = np.zeros((N, nop1))
        if reflect_left or reflect_right:
            bc = build_reflecting_BCs(
                bc, psi_g[g], reflect_left, reflect_right, N, order)
        return bc

    # Fleck factor and per-group coupling weights
    f, alpha_g = mg_sn_solver._compute_fleck_and_alpha(
        T_old, dt, sigma_a_funcs, Bg_funcs, dBdT_funcs, Cv_func, G)

    sigma_ag = [sigma_a_funcs[g](T_old) for g in range(G)]
    scat_g   = [scat_funcs[g](T_old)    for g in range(G)]
    Bg_vals  = [Bg_funcs[g](T_old)      for g in range(G)]
    sum_sigma_B = sum(sigma_ag[g] * Bg_vals[g] for g in range(G))

    sigma_t_g      = []
    fixed_source_g = []
    for g in range(G):
        sigma_t_g.append(sigma_ag[g] + scat_g[g] + icdt)
        emission_fixed = sigma_ag[g] * Bg_vals[g] - alpha_g[g] * sum_sigma_B
        fixed_source_g.append(
            q_ext[g] + emission_fixed[:, None, :] + icdt * psi_g[g])

    # ---- build the scatter+coupling matvec (zero BCs — BCs enter via b) ----
    block = I * nop1
    _sigma_ag = sigma_ag    # capture for closure
    _scat_g   = scat_g
    _alpha_g  = alpha_g
    _sigma_t  = sigma_t_g

    def mv(x_vec):
        result = np.zeros_like(x_vec)
        phi_parts = [x_vec[g * block:(g + 1) * block].reshape((I, nop1))
                     for g in range(G)]
        sum_sp = sum(_sigma_ag[gp] * phi_parts[gp] for gp in range(G))
        for g in range(G):
            src_g = ((_scat_g[g] * phi_parts[g] + _alpha_g[g] * sum_sp)
                     [:, None, :] + zero_src)
            phi_new = sn_solver.single_source_iteration(
                I, hx, src_g, _sigma_t[g], N, zero_bcs, order=order, fix=0)
            result[g * block:(g + 1) * block] = phi_new.ravel()
        return result

    # ---- transport solve with reflecting BC fixed-point iteration ----
    psi_bc_state = [p.copy() for p in psi_g]
    x_sol = np.concatenate([p.ravel() for p in phi_g])
    n_its_total = 0
    reflect_tol = 1e-12

    for _ in range(20):
        def _bc_for_g_iter(g):
            bc = np.zeros((N, nop1))
            if reflect_left or reflect_right:
                bc = build_reflecting_BCs(
                    bc, psi_bc_state[g], reflect_left, reflect_right, N, order)
            return bc

        b_vec = np.zeros(G * block)
        for g in range(G):
            phi_b = sn_solver.single_source_iteration(
                I, hx, fixed_source_g[g], sigma_t_g[g], N, _bc_for_g_iter(g),
                order=order, fix=0)
            b_vec[g * block:(g + 1) * block] = phi_b.ravel()

        x_sol, n_its, _, _, _, _, _ = sn_solver.solver_with_dmd_inc(
            matvec=mv, b=b_vec, K=K, max_its=maxits, steady=1,
            x=x_sol, Rits=R, LOUD=False, order=order,
            L2_tol=1e-12, Linf_tol=1e-10)
        n_its_total += n_its

        phi_g_new = [x_sol[g * block:(g + 1) * block].reshape((I, nop1))
                     for g in range(G)]

        sum_sigma_phi = sum(sigma_ag[gp] * phi_g_new[gp] for gp in range(G))
        psi_g_new = []
        for g in range(G):
            full_src = ((scat_g[g] * phi_g_new[g] + alpha_g[g] * sum_sigma_phi)
                        [:, None, :] + fixed_source_g[g])
            psi_g_new.append(sn_solver.single_source_iteration_psi(
                I, hx, full_src, sigma_t_g[g], N, _bc_for_g_iter(g),
                order=order, fix=0))

        if not (reflect_left or reflect_right):
            break

        bc_change = 0.0
        for g in range(G):
            bc_old = build_reflecting_BCs(
                np.zeros((N, nop1)), psi_bc_state[g],
                reflect_left, reflect_right, N, order)
            bc_new = build_reflecting_BCs(
                np.zeros((N, nop1)), psi_g_new[g],
                reflect_left, reflect_right, N, order)
            bc_change = max(bc_change, np.max(np.abs(bc_new - bc_old)))

        psi_bc_state = [p.copy() for p in psi_g_new]
        if bc_change < reflect_tol:
            break

    # ---- material energy update (Fleck linearisation) ----
    energy_dep = sum(sigma_ag[g] * (phi_g_new[g] - Bg_vals[g]) for g in range(G))
    e_new = e_old + f * dt * energy_dep
    T_new = invEOS(e_new)

    return phi_g_new, psi_g_new, T_new, e_new, n_its_total


# ── Driver ───────────────────────────────────────────────────────────────────

def run_problem(n_groups=50, sigma0=10.0, Tc=1.0, Trad=0.5, Tmat0=0.4,
                rho_cv=0.01, dt_initial=1e-4, dt_max=1e-2, dt_growth=1.1,
                final_time=1.0, order=3, N=8, K=100, maxits=200):
    """Run the infinite-medium S_N expband test.

    Uses a single spatial cell (I=1, hx=1 cm).  After each timestep the
    boundary angular flux is updated to the isotropic Planck equilibrium at
    the current mean material temperature.
    """
    print("=" * 80)
    print("Infinite Medium Multigroup S_N Test (exponential-band opacity)")
    print("=" * 80)

    G    = n_groups
    I    = 1
    hx   = 1.0          # cm
    nop1 = order + 1

    energy_edges = np.logspace(np.log10(1e-4), np.log10(20.0), G + 1)
    MU, _W = np.polynomial.legendre.leggauss(N)  # noqa: F841

    # ---- group physics ----
    sigma_a_funcs = [make_group_opacity(energy_edges[g], energy_edges[g + 1],
                                        sigma0=sigma0)
                     for g in range(G)]
    scat_funcs    = [lambda T: np.zeros_like(T) for _ in range(G)]
    Bg_funcs      = [_make_Bg(energy_edges[g], energy_edges[g + 1])
                     for g in range(G)]
    dBdT_funcs    = [_make_dBdT(energy_edges[g], energy_edges[g + 1])
                     for g in range(G)]
    q_ext         = [np.zeros((I, N, nop1)) for _ in range(G)]

    # ---- EOS ----
    def EOS(T):    return rho_cv * T
    def invEOS(e): return e / rho_cv
    def Cv_func(T): return np.full_like(T, rho_cv)

    # ---- Planck normalisation sanity check ----
    T_chk  = np.array([[0.5, 0.5, 0.5, 0.5]])[:, :nop1]
    Bg_sum = sum(Bg_funcs[g](T_chk) for g in range(G))
    exp_   = AC * T_chk**4
    frac_  = float(np.max(np.abs(Bg_sum - exp_) / (exp_ + 1e-300)))
    print(f"Planck normalisation check: max frac err = {frac_:.2e}")

    # ---- initial conditions ----
    T_old = np.full((I, nop1), Tmat0)
    e_old = EOS(T_old)

    Eg0 = group_energy_density(
        energy_edges[:-1], energy_edges[1:], Tc=Tc, Trad=Trad, a_rad=A_RAD)
    print(f"Initial radiation energy density (total) = {Eg0.sum():.4e} GJ/cm³")
    print(f"  Equilibrium at Trad={Trad} keV → a T^4 = {A_RAD * Trad**4:.4e} GJ/cm³")
    print(f"  Equilibrium at Tmat0={Tmat0} keV → a T^4 = {A_RAD * Tmat0**4:.4e} GJ/cm³")

    # phi_g = c * u_g  (scalar flux from radiation energy density)
    phi_g = [np.full((I, nop1), C_LIGHT * Eg0[g]) for g in range(G)]
    # Equilibrium ψ convention: ψ_n = φ for all n (matches sn benchmark convention;
    # see test_su_olson.py: psi = phi[:,None,:]).
    psi_g = [np.zeros((I, N, nop1)) + phi_g[g][:, None, :]
             for g in range(G)]

    # ---- time loop ----
    t      = 0.0
    dt     = float(dt_initial)
    dt_cap = float(dt_max)
    growth = float(dt_growth)

    times          = [t]
    T_mat_hist     = [Tmat0]
    u_rad_g_init   = Eg0.copy()
    T_rad_init     = float((Eg0.sum() / A_RAD) ** 0.25)
    T_rad_hist     = [T_rad_init]
    group_u_hist   = [Eg0.copy()]   # radiation energy density [GJ/cm³] per group

    total_its = 0
    step_idx  = 0

    print(f"\nG={G}, I={I}, N={N}, order={order}, tfinal={final_time} ns")
    print(f"dt_initial={dt_initial:.2e} ns, dt_max={dt_max:.2e} ns, growth={dt_growth:.3f}")
    print()

    while t < final_time - 1e-14:
        step_idx += 1
        dt_step = min(dt, dt_cap, final_time - t)
        if dt_step <= 0.0:
            break

        # Use true reflecting BCs (psi_g from previous step used inside)
        phi_g, psi_g, T_new, e_new, n_its = _single_mg_step(
            I=I, hx=hx, G=G,
            sigma_a_funcs=sigma_a_funcs, scat_funcs=scat_funcs,
            Bg_funcs=Bg_funcs, dBdT_funcs=dBdT_funcs,
            q_ext=q_ext, N=N,
            invEOS=invEOS, Cv_func=Cv_func,
            phi_g=phi_g, psi_g=psi_g, T_old=T_old, e_old=e_old,
            dt=dt_step, order=order, K=K, maxits=maxits,
            reflect_left=True, reflect_right=True)

        t += dt_step
        total_its += n_its
        T_old  = T_new
        e_old  = e_new

        T_mat_val = float(T_new.mean())
        # u_g = phi_g / c  (radiation energy density per group)
        u_g_arr   = np.array([float(phi_g[g].mean()) / C_LIGHT for g in range(G)])
        T_rad_val = float((u_g_arr.sum() / A_RAD) ** 0.25)

        times.append(t)
        T_mat_hist.append(T_mat_val)
        T_rad_hist.append(T_rad_val)
        group_u_hist.append(u_g_arr)

        if step_idx <= 5 or step_idx % 20 == 0:
            print(f"  step {step_idx:4d}  t={t:.4e} ns  dt={dt_step:.3e} ns  "
                  f"T_mat={T_mat_val:.6f} keV  T_rad={T_rad_val:.6f} keV  "
                  f"its={n_its}")

        dt = min(dt_cap, dt * growth)

    print(f"\nDone: {step_idx} steps, {total_its} total transport sweeps")
    print(f"  Final T_mat = {T_mat_hist[-1]:.6f} keV")
    print(f"  Final T_rad = {T_rad_hist[-1]:.6f} keV")

    return {
        'times':              np.array(times),
        'T_mat':              np.array(T_mat_hist),
        'T_rad':              np.array(T_rad_hist),
        'group_energy_history': np.array(group_u_hist),   # (nsteps+1, G) GJ/cm³
        'energy_edges':       energy_edges,
        'n_groups':           G,
    }


# ── Plots ─────────────────────────────────────────────────────────────────────

def make_plots(data, requested_times=(0.0, 0.01, 0.1, 1.0), savefile=''):
    """Two-panel plot matching the layout of the IMC version."""
    times    = data['times']
    T_mat    = data['T_mat']
    T_rad    = data['T_rad']
    e_edges  = data['energy_edges']
    grp_hist = data['group_energy_history']
    G        = data['n_groups']

    e_mid = np.sqrt(e_edges[:-1] * e_edges[1:])
    dE    = np.diff(e_edges)

    plot_indices = []
    for t_req in requested_times:
        idx = int(np.argmin(np.abs(times - t_req)))
        if idx not in plot_indices:
            plot_indices.append(idx)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.8, 10.2))

    # (a) Temperature vs time
    ax1.plot(times, T_mat, color='tab:blue',   lw=1.6, label='$T$ (material)')
    ax1.plot(times, T_rad, color='tab:orange', lw=1.5, ls='--',
             label=r'$T_r$ (radiation)')
    ax1.set_xlabel('t (ns)')
    ax1.set_ylabel('Temperature (keV)')
    ax1.grid(True, alpha=0.25)
    ax1.legend()

    # (b) Group spectrum at selected times
    styles = ['-', '--', '-.', ':']
    for ii, hidx in enumerate(plot_indices):
        t_label = times[hidx]
        ax2.plot(e_mid, grp_hist[hidx] / dE,
                 ls=styles[ii % len(styles)], lw=1.6,
                 label=f't = {t_label:.2f} ns')
    #ax2.set_xscale('log')
    ax2.set_xlabel(r'Photon Energy, $E_\nu$ (keV)')
    ax2.set_ylabel(r'Spectral energy density  (GJ cm$^{-3}$ keV$^{-1}$)')
    ax2.grid(True, which='both', alpha=0.25)
    ax2.legend()

    plt.tight_layout()

    if savefile and HAS_PLOTFUNCS:
        show(savefile, close_after=True)
    else:
        out = (f'infinite_medium_multigroup_expband_sn_{G}g_plots'
               + ('.pdf' if HAS_PLOTFUNCS else '.png'))
        if HAS_PLOTFUNCS:
            show(out, close_after=True)
        else:
            fig.savefig(out, dpi=150, bbox_inches='tight')
            print(f'Saved: {out}')

    return fig


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Infinite medium multigroup S_N test (expband opacity)')
    parser.add_argument('--groups',     type=int,   default=50,
                        help='Number of energy groups (default: 50)')
    parser.add_argument('--N',          type=int,   default=8,
                        help='Discrete ordinates order (default: 8)')
    parser.add_argument('--order',      type=int,   default=3,
                        help='Bernstein polynomial order (default: 3)')
    parser.add_argument('--K',          type=int,   default=100,
                        help='DMD inner iterations (default: 100)')
    parser.add_argument('--maxits',     type=int,   default=200,
                        help='Max DMD iterations per step (default: 200)')
    parser.add_argument('--dt-initial', type=float, default=1e-4,
                        help='Initial time step in ns (default: 1e-4)')
    parser.add_argument('--dt-max',     type=float, default=1e-2,
                        help='Max time step in ns (default: 1e-2)')
    parser.add_argument('--dt-growth',  type=float, default=1.1,
                        help='dt growth factor per step (default: 1.1)')
    parser.add_argument('--final-time', type=float, default=1.0,
                        help='Final time in ns (default: 1.0)')
    parser.add_argument('--save-fig',   type=str,   default='',
                        help='Save figure to this path')
    args = parser.parse_args()

    data = run_problem(
        n_groups   = args.groups,
        N          = args.N,
        order      = args.order,
        K          = args.K,
        maxits     = args.maxits,
        dt_initial = args.dt_initial,
        dt_max     = args.dt_max,
        dt_growth  = args.dt_growth,
        final_time = args.final_time,
    )

    # auto-save NPZ
    npz_file = f'infinite_medium_multigroup_expband_sn_{args.groups}g.npz'
    np.savez(npz_file, **{k: v for k, v in data.items() if not callable(v)})
    print(f'Saved: {npz_file}')

    make_plots(data,
               requested_times=(0.0, 0.01, 0.1, 1.0),
               savefile=args.save_fig)


if __name__ == '__main__':
    main()
