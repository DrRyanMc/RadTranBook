#!/usr/bin/env python3
"""
Su-Olson Picket Fence test problem for multigroup S_N transport.

Two-group problem with:
  - σ₁ = 2/11 cm⁻¹ (optically thin)
  - σ₂ = 20/11 cm⁻¹ (optically thick)
  - Equal Planck fractions: 4π B_g = 0.5 · a·c·T⁴
  - Source: Q_g = 0.5 · a·c·T_h⁴  in each group for |x − center| < 0.5
  - EOS: e = a T⁴,  C_v = 4 a T³
  - Reflecting BC at x=0 (full-domain symmetric trick)
  - Output at τ = 0.1, 0.3, 1.0  mean-free times
  - Reference data: Su & Olson (1997) Table 4

Run from the DiscreteOrdinates directory:
    python problems/test_su_olson_picket_fence.py
"""

import sys
import os
import argparse
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.interpolate import BPoly
from matplotlib.lines import Line2D
from numba import jit, njit, float64

# Add parent directory so we can import the solvers
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import sn_solver
import mg_sn_solver

# Add project root for shared utilities
project_root = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, project_root)
try:
    from utils.plotfuncs import show, hide_spines, font
    HAS_PLOTFUNCS = True
except ImportError:
    HAS_PLOTFUNCS = False

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
A_RAD = sn_solver.a       # radiation constant  (GJ / cm³ keV⁴)
C_LIGHT = sn_solver.c     # speed of light      (cm / ns)
AC = sn_solver.ac          # a · c
T_h = 1.0                  # reference temperature (keV)

# ---------------------------------------------------------------------------
# Su-Olson Picket Fence transport reference data (Table 4)
# Format: x, τ=0.1, τ=0.3, τ=1.0, τ=3.0
# ---------------------------------------------------------------------------
transport_U1 = np.array([
    [0.00, 0.04956, 0.14632, 0.39890, 0.65095],
    [0.10, 0.04956, 0.14632, 0.39418, 0.64570],
    [0.30, 0.04956, 0.14181, 0.35349, 0.60076],
    [0.45, 0.04578, 0.10753, 0.28118, 0.52264],
    [0.50, 0.02478, 0.07316, 0.23277, 0.47187],
    [0.55, 0.00378, 0.03879, 0.18410, 0.42049],
    [0.75, np.nan, 0.00105, 0.09013, 0.31167],
    [1.00, np.nan, np.nan, 0.03332, 0.22950],
    [1.35, np.nan, np.nan, 0.00250, 0.15410],
    [1.80, np.nan, np.nan, np.nan, 0.09082],
    [2.35, np.nan, np.nan, np.nan, 0.04035],
    [3.15, np.nan, np.nan, np.nan, 0.00316],
])

transport_U2 = np.array([
    [0.00, 0.04585, 0.11858, 0.26676, 0.51786],
    [0.10, 0.04585, 0.11858, 0.26401, 0.51216],
    [0.30, 0.04585, 0.11563, 0.23839, 0.46275],
    [0.45, 0.04254, 0.08956, 0.18496, 0.37580],
    [0.50, 0.02293, 0.05929, 0.14353, 0.32037],
    [0.55, 0.00331, 0.02902, 0.10206, 0.26455],
    [0.75, np.nan, 0.00066, 0.03736, 0.15017],
    [1.00, np.nan, np.nan, 0.01016, 0.07857],
    [1.35, np.nan, np.nan, 0.00054, 0.03199],
    [1.80, np.nan, np.nan, np.nan, 0.00963],
    [2.35, np.nan, np.nan, np.nan, 0.00182],
    [3.15, np.nan, np.nan, np.nan, 0.00004],
])

transport_V = np.array([
    [0.00, 0.00458, 0.03511, 0.23884, 0.81005],
    [0.10, 0.00458, 0.03511, 0.23680, 0.80144],
    [0.30, 0.00458, 0.03489, 0.21640, 0.72556],
    [0.45, 0.00446, 0.02893, 0.16752, 0.58601],
    [0.50, 0.00229, 0.01756, 0.12369, 0.49152],
    [0.55, 0.00012, 0.00617, 0.07986, 0.39651],
    [0.75, np.nan, 0.00002, 0.02270, 0.21608],
    [1.00, np.nan, np.nan, 0.00427, 0.11109],
    [1.35, np.nan, np.nan, 0.00007, 0.04604],
    [1.80, np.nan, np.nan, np.nan, 0.01513],
    [2.35, np.nan, np.nan, np.nan, 0.00336],
    [3.15, np.nan, np.nan, np.nan, 0.00007],
])

# τ values in the reference tables
_ref_taus = [0.1, 0.3, 1.0, 3.0]


def _domain_energy(phi_g, T, hx):
    """Total material + radiation energy in domain (per unit area)."""
    e_mat = A_RAD * np.maximum(T, 0.0)**4
    E_mat = hx * np.sum(np.mean(e_mat, axis=1))
    E_rad = 0.0
    for g in range(len(phi_g)):
        E_rad += hx * np.sum(np.mean(phi_g[g], axis=1) / C_LIGHT)
    return E_mat + E_rad, E_mat, E_rad


def _source_power(q_ext, W, hx):
    """Domain-integrated source power (energy per unit area per ns)."""
    power = 0.0
    for qg in q_ext:
        # Angular quadrature gives scalar source per Bernstein node.
        q_scalar = np.tensordot(qg, W, axes=(1, 0))  # (I, order+1)
        power += hx * np.sum(np.mean(q_scalar, axis=1))
    return float(power)


def _append_conservation_history(history, ts, phis, Ts, source_power, t_offset=0.0):
    """Append step-by-step conservation bookkeeping for one solve phase."""
    if len(ts) < 2:
        return

    for k in range(1, len(ts)):
        t0 = t_offset + float(ts[k - 1])
        t1 = t_offset + float(ts[k])
        dt = t1 - t0

        E0, Em0, Er0 = _domain_energy(phis[k - 1], Ts[k - 1], history['hx'])
        E1, Em1, Er1 = _domain_energy(phis[k], Ts[k], history['hx'])
        Einj = dt * source_power

        history['time'].append(t1)
        history['dt'].append(dt)
        history['E_mat'].append(Em1)
        history['E_rad'].append(Er1)
        history['E_total'].append(E1)
        history['dE_step'].append(E1 - E0)
        history['E_src_step'].append(Einj)
        history['res_step'].append((E1 - E0) - Einj)


def _finalize_conservation_history(history):
    """Compute cumulative source and global conservation residuals."""
    if len(history['E_total']) == 0:
        return

    E0 = history['E0']
    E_src_cum = np.cumsum(np.asarray(history['E_src_step']))
    E_total = np.asarray(history['E_total'])
    E_expected = E0 + E_src_cum
    drift = E_total - E_expected

    history['E_src_cum'] = E_src_cum.tolist()
    history['E_expected'] = E_expected.tolist()
    history['drift'] = drift.tolist()


def print_conservation_diagnostics(conservation):
    """Print a concise energy conservation summary with source accounting."""
    if len(conservation['E_total']) == 0:
        print("\nNo conservation diagnostics available.")
        return

    t = np.asarray(conservation['time'])
    E_mat = np.asarray(conservation['E_mat'])
    E_rad = np.asarray(conservation['E_rad'])
    E_tot = np.asarray(conservation['E_total'])
    E_src = np.asarray(conservation['E_src_step'])
    E_src_cum = np.asarray(conservation['E_src_cum'])
    drift = np.asarray(conservation['drift'])

    E0 = conservation['E0']
    scale_E0 = max(abs(E0), 1e-300)
    scale_src = max(abs(E_src_cum[-1]), 1e-300)
    rel_final_E0 = drift[-1] / scale_E0
    rel_final_src = drift[-1] / scale_src
    rel_max_E0 = np.max(np.abs(drift)) / scale_E0
    rel_max_src = np.max(np.abs(drift)) / scale_src

    print("\n" + "=" * 78)
    print("Energy conservation with explicit source accounting")
    print("=" * 78)
    print(f"  Initial total energy: {E0:.8e}")
    print(f"  Final total energy:   {E_tot[-1]:.8e}")
    print(f"  Cumulative source:    {E_src_cum[-1]:.8e}")
    print(f"  Final drift:          {drift[-1]:+.8e}")
    print(f"  Final drift / E0:     {rel_final_E0:+.3e}")
    print(f"  Final drift / Einj:   {rel_final_src:+.3e}")
    print(f"  Max |drift| / E0:     {rel_max_E0:.3e}")
    print(f"  Max |drift| / Einj:   {rel_max_src:.3e}")

    print("\n  Sample timestep ledger:")
    print("  t(ns)      E_mat        E_rad        E_tot        E_src_step   drift")
    n = len(t)
    sample_ids = sorted(set([0, min(1, n - 1), min(2, n - 1), n - 1]))
    for idx in sample_ids:
        print(f"  {t[idx]:8.3e}  {E_mat[idx]:11.4e}  {E_rad[idx]:11.4e}  "
              f"{E_tot[idx]:11.4e}  {E_src[idx]:11.4e}  {drift[idx]:+11.4e}")


# ---------------------------------------------------------------------------
# Problem setup & run
# ---------------------------------------------------------------------------
def setup_and_run(I=400, order=2, N=8, K=800, maxits=2000, LOUD=0,
                  output_tau=(0.1, 0.3, 1.0)):
    """Run the picket fence problem with multigroup S_N.

    Parameters
    ----------
    I : int
        Number of spatial zones per half-domain (internal grid = 2*I).
    order : int
        Bernstein polynomial order.
    N : int
        Number of discrete ordinates.
    K, maxits : int
        DMD solver parameters.
    LOUD : int
        Verbosity level.
    output_tau : tuple of float
        Output times in mean-free-times.

    Returns
    -------
    results : dict
    """
    # --- problem parameters ---
    G = 2
    sigma_1 = 2.0 / 11.0    # cm⁻¹
    sigma_2 = 20.0 / 11.0   # cm⁻¹
    sigma_avg = (sigma_1 + sigma_2) / 2.0   # = 1.0  (Planck mean)
    tau_mft = 1.0 / (C_LIGHT * sigma_avg)
    source_duration = 10.0 * tau_mft
    Tinit = 0.001
    Q_per_group = 0.5 * AC * T_h**4          # half of gray Su-Olson source

    print(f"Picket fence: σ₁ = {sigma_1:.6f}, σ₂ = {sigma_2:.6f}")
    print(f"Mean free time  τ = {tau_mft:.6e} ns")
    print(f"Source duration    = {source_duration:.6e} ns  (10 τ)")

    # --- geometry (full domain for reflecting BC) ---
    Lx_half = 20.0
    I_full = 2 * I
    Lx_full = 2 * Lx_half
    hx = Lx_full / I_full
    center = Lx_half
    x_comp = np.linspace(hx / 2, Lx_full - hx / 2, I_full)
    x_phys = x_comp[I:] - center  # right half physical coordinates
    source_half = 0.5

    print(f"Full domain cells  = {I_full}  (2 × {I})")

    nop1 = order + 1

    # --- EOS: e = a T⁴ ---
    @njit
    def eos(T):
        return A_RAD * np.maximum(T, 0.0)**4

    @njit
    def invEOS(E):
        return (np.maximum(E, 0.0) / A_RAD)**0.25

    def Cv_func(T):
        return 4.0 * A_RAD * np.maximum(T, 1e-30)**3

    # --- opacities (constant) ---
    def sigma_a_1(T):
        return np.full_like(T, sigma_1)

    def sigma_a_2(T):
        return np.full_like(T, sigma_2)

    def scat_1(T):
        return np.zeros_like(T)

    def scat_2(T):
        return np.zeros_like(T)

    # --- Planck functions: 4π B_g = 0.5 · ac · T⁴ ---
    def Bg_1(T):
        return 0.5 * AC * np.maximum(T, 0.0)**4

    def Bg_2(T):
        return 0.5 * AC * np.maximum(T, 0.0)**4

    def dBdT_1(T):
        return 2.0 * AC * np.maximum(T, 0.0)**3

    def dBdT_2(T):
        return 2.0 * AC * np.maximum(T, 0.0)**3

    sigma_a_funcs = [sigma_a_1, sigma_a_2]
    scat_funcs    = [scat_1, scat_2]
    Bg_funcs      = [Bg_1, Bg_2]
    dBdT_funcs    = [dBdT_1, dBdT_2]

    # --- boundary conditions (vacuum both sides) ---
    @jit(float64[:, :](float64), nopython=True)
    def BCFunc(t):
        return np.zeros((N, nop1))

    # --- initial conditions (full domain) ---
    T = np.ones((I_full, nop1)) * Tinit
    phi_g = [0.5 * AC * T**4 for _ in range(G)]   # B_g(T_init)
    psi_g = [phi_g[g][:, None, :] + np.zeros((I_full, N, nop1))
             for g in range(G)]

    # --- external source (centred, on during phase 1) ---
    q_on = []
    for g in range(G):
        q = np.zeros((I_full, N, nop1))
        for i in range(I_full):
            if abs(x_comp[i] - center) < source_half:
                q[i, :, :] = Q_per_group
        q_on.append(q)
    q_off = [np.zeros((I_full, N, nop1)) for _ in range(G)]

    MU, W = np.polynomial.legendre.leggauss(N)
    W = W / np.sum(W)

    # --- conservation bookkeeping ---
    E0, E0_mat, E0_rad = _domain_energy(phi_g, T, hx)
    conservation = {
        'hx': hx,
        'E0': float(E0),
        'E0_mat': float(E0_mat),
        'E0_rad': float(E0_rad),
        'time': [],
        'dt': [],
        'E_mat': [],
        'E_rad': [],
        'E_total': [],
        'dE_step': [],
        'E_src_step': [],
        'res_step': [],
        'E_src_cum': [],
        'E_expected': [],
        'drift': [],
    }

    # --- time parameters ---
    dt_min = 0.001 * tau_mft
    dt_max = 0.5 * tau_mft

    output_tau = np.array(sorted(output_tau))
    output_times_ns = output_tau * tau_mft
    early_mask = output_times_ns <= source_duration * (1 + 1e-10)
    late_mask = ~early_mask
    early_outputs = output_times_ns[early_mask]
    late_outputs = output_times_ns[late_mask]

    solutions = {}
    total_iterations = 0

    # ---- Phase 1: source ON, t ∈ [0, source_duration] ----
    phase1_final = source_duration
    if early_outputs.size > 0:
        phase1_final = max(phase1_final, early_outputs[-1])

    # Prescribed emission fractions for picket fence
    chi = np.array([0.5, 0.5])

    print(f"\n--- Phase 1: source ON, 0 \u2192 {phase1_final / tau_mft:.2f} \u03c4 ---")
    phis1, Ts1, its1, ts1 = mg_sn_solver.mg_temp_solve_dmd_inc(
        I_full, hx,
        G, sigma_a_funcs, scat_funcs, Bg_funcs, dBdT_funcs,
        q_on, N, BCFunc, eos, invEOS, Cv_func,
        phi_g, psi_g, T,
        dt_min=dt_min, dt_max=dt_max, tfinal=phase1_final,
        order=order, LOUD=LOUD, maxits=maxits, fix=1, K=K, R=3,
        time_outputs=early_outputs if early_outputs.size > 0 else None,
        chi=chi)
    _append_conservation_history(
        conservation, ts1, phis1, Ts1,
        source_power=_source_power(q_on, W, hx),
        t_offset=0.0)
    total_iterations += its1
    print(f"Phase 1 sweeps: {its1}")

    # extract early snapshots (right half only)
    norm = A_RAD * T_h**4
    for tau_val in output_tau[early_mask]:
        t_ns = tau_val * tau_mft
        tstep = np.argmin(np.abs(t_ns - ts1))
        phi_snap = [phis1[tstep][g][I:, :] for g in range(G)]
        T_snap = Ts1[tstep][I:, :]
        U1 = phi_snap[0] / (C_LIGHT * norm)
        U2 = phi_snap[1] / (C_LIGHT * norm)
        V = T_snap**4 / T_h**4
        solutions[tau_val] = {
            'phi_g': phi_snap, 'T': T_snap,
            'U1': U1, 'U2': U2, 'V': V,
            'tstep': tstep, 't_ns': ts1[tstep]
        }
        print(f"  Saved τ = {tau_val:.4f}  (t = {ts1[tstep]:.6e} ns)")

    # ---- Phase 2: source OFF ----
    if late_outputs.size > 0:
        phi_g2 = phis1[-1]
        T2 = Ts1[-1]
        psi_g2 = [phi_g2[g][:, None, :] + np.zeros((I_full, N, nop1))
                  for g in range(G)]
        late_final = late_outputs[-1] - source_duration
        late_outs_rel = late_outputs - source_duration

        print(f"\n--- Phase 2: source OFF, 0 → {late_final / tau_mft:.2f} τ ---")
        phis2, Ts2, its2, ts2 = mg_sn_solver.mg_temp_solve_dmd_inc(
            I_full, hx,
            G, sigma_a_funcs, scat_funcs, Bg_funcs, dBdT_funcs,
            q_off, N, BCFunc, eos, invEOS, Cv_func,
            phi_g2, psi_g2, T2,
            dt_min=dt_min, dt_max=dt_max, tfinal=late_final,
            order=order, LOUD=LOUD, maxits=maxits, fix=1, K=K, R=3,
            time_outputs=late_outs_rel,
            chi=chi)
        _append_conservation_history(
            conservation, ts2, phis2, Ts2,
            source_power=_source_power(q_off, W, hx),
            t_offset=source_duration)
        total_iterations += its2
        print(f"Phase 2 sweeps: {its2}")

        for tau_val in output_tau[late_mask]:
            t_rel = tau_val * tau_mft - source_duration
            tstep = np.argmin(np.abs(t_rel - ts2))
            phi_snap = [phis2[tstep][g][I:, :] for g in range(G)]
            T_snap = Ts2[tstep][I:, :]
            U1 = phi_snap[0] / (C_LIGHT * norm)
            U2 = phi_snap[1] / (C_LIGHT * norm)
            V = T_snap**4 / T_h**4
            solutions[tau_val] = {
                'phi_g': phi_snap, 'T': T_snap,
                'U1': U1, 'U2': U2, 'V': V,
                'tstep': tstep, 't_ns': ts2[tstep] + source_duration
            }
            print(f"  Saved τ = {tau_val:.4f}  (t = {solutions[tau_val]['t_ns']:.6e} ns)")

    _finalize_conservation_history(conservation)
    print_conservation_diagnostics(conservation)

    print(f"\nTotal transport sweeps: {total_iterations}")
    return {
        'solutions': solutions, 'x': x_phys, 'hx': hx, 'Lx': Lx_half,
        'order': order, 'iterations': total_iterations, 'tau_mft': tau_mft,
        'conservation': conservation,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_results(results, savefile=''):
    """Plot multigroup S_N results vs. transport reference data."""
    solutions = results['solutions']
    x = results['x']
    Lx = results['Lx']
    order = results['order']
    nI = len(x)
    hx = results['hx']
    edges = np.linspace(0, Lx, nI + 1)
    xplot = np.linspace(hx / 2, Lx, 2000)

    taus_available = sorted(solutions.keys())
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(taus_available)))

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    ax_U1, ax_U2, ax_V = axes

    for idx, tau_val in enumerate(taus_available):
        sol = solutions[tau_val]
        col = colors[idx]
        label = rf'$\tau={tau_val}$'

        # Bernstein interpolation for smooth curves
        U1_interp = BPoly(sol['U1'].T, edges)(xplot)
        U2_interp = BPoly(sol['U2'].T, edges)(xplot)
        V_interp = BPoly(sol['V'].T, edges)(xplot)

        ax_U1.plot(xplot, U1_interp, '-', color=col, lw=1.5, label=label)
        ax_U2.plot(xplot, U2_interp, '-', color=col, lw=1.5, label=label)
        ax_V.plot(xplot, V_interp, '-', color=col, lw=1.5, label=label)

        # overlay transport reference
        if tau_val in _ref_taus:
            ti = _ref_taus.index(tau_val)
            for ref_data, ax in [(transport_U1, ax_U1),
                                  (transport_U2, ax_U2),
                                  (transport_V, ax_V)]:
                ref_x = ref_data[:, 0]
                ref_y = ref_data[:, ti + 1]
                valid = ~np.isnan(ref_y)
                ax.plot(ref_x[valid], ref_y[valid], 's',
                        color=col, ms=5, mec='k', mew=0.5, alpha=0.8)

    for ax, title, ylabel in [
        (ax_U1, r'Group 1: $\sigma_1 = 2/11$ (thin)',
         r'$U_1 = E_{r,1}/(a T_h^4)$'),
        (ax_U2, r'Group 2: $\sigma_2 = 20/11$ (thick)',
         r'$U_2 = E_{r,2}/(a T_h^4)$'),
        (ax_V, 'Material Temperature',
         r'$V = (T/T_h)^4$')
    ]:
        ax.set_xlabel('Position (mean-free paths)', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlim(0.05, 5.0)
        ax.set_ylim(1e-4, 1e0)
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(fontsize=9)

    ref_handle = Line2D([], [], marker='s', color='gray', ms=5,
                        mec='k', mew=0.5, ls='', label='Transport ref')
    ax_U1.legend(handles=list(ax_U1.get_legend_handles_labels()[0])
                 + [ref_handle], fontsize=9, loc='best')

    fig.suptitle(r'Su-Olson Picket Fence — Multigroup S$_N$', fontsize=14,
                 fontweight='bold', y=1.01)
    plt.tight_layout()

    if savefile:
        plt.savefig(savefile, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {savefile}")
    if HAS_PLOTFUNCS:
        show('su_olson_picket_fence_sn.pdf', close_after=True)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Error table
# ---------------------------------------------------------------------------
def print_error_table(results):
    """Print relative errors at reference points."""
    solutions = results['solutions']
    x = results['x']
    hx = results['hx']
    Lx = results['Lx']
    nI = len(x)
    order = results['order']
    edges = np.linspace(0, Lx, nI + 1)

    print("\n" + "=" * 72)
    print("Relative errors vs. transport reference (Su & Olson 1997, Table 4)")
    print("=" * 72)

    quantities = [
        ('U1', transport_U1, 'U₁'),
        ('U2', transport_U2, 'U₂'),
        ('V',  transport_V,  'V'),
    ]

    for qty_key, ref_table, qty_name in quantities:
        print(f"\n  {qty_name}:")
        for tau_val in sorted(solutions.keys()):
            if tau_val not in _ref_taus:
                continue
            ti = _ref_taus.index(tau_val)
            sol = solutions[tau_val]
            interp = BPoly(sol[qty_key].T, edges)

            ref_x = ref_table[:, 0]
            ref_y = ref_table[:, ti + 1]
            valid = ~np.isnan(ref_y) & (ref_y > 1e-6)
            if not np.any(valid):
                continue
            comp_y = interp(ref_x[valid])
            rel_err = np.abs(comp_y - ref_y[valid]) / ref_y[valid]
            print(f"    τ = {tau_val:5.2f}: "
                  f"max rel err = {rel_err.max():.4f}, "
                  f"mean rel err = {rel_err.mean():.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Su-Olson Picket Fence multigroup S_N test')
    parser.add_argument('--I', type=int, default=400,
                        help='Zones per half-domain (default 400)')
    parser.add_argument('--order', type=int, default=2,
                        help='Bernstein order (default 2)')
    parser.add_argument('--N', type=int, default=8,
                        help='Number of ordinates (default 8)')
    parser.add_argument('--K', type=int, default=800,
                        help='DMD inner iterations (default 800)')
    parser.add_argument('--maxits', type=int, default=2000,
                        help='Max iterations per step (default 2000)')
    parser.add_argument('--no-plot', action='store_true', default=False,
                        help='Skip plotting')
    args = parser.parse_args()

    results = setup_and_run(
        I=args.I, order=args.order, N=args.N, K=args.K, maxits=args.maxits)

    print_error_table(results)

    if not args.no_plot:
        plot_results(results)
