 #!/usr/bin/env python3
"""
Marshak Wave — Multigroup S_N with Power-Law Opacity

S_N transport analogue of the non-equilibrium diffusion Marshak wave
(nonEquilibriumDiffusion/problems/marshak_wave_multigroup_powerlaw.py).

Problem setup:
  - Multigroup opacity: σ_a(T,E) = 10 ρ T^{-1/2} E^{-3}  (cm⁻¹)
  - Group opacity: geometric mean at group boundaries
  - Left boundary: incoming blackbody (Marshak BC) with time-ramped T_bc
  - Right boundary: vacuum (zero incoming)
  - Heat capacity: c_v = 0.05 GJ/(g·keV), ρ = 0.01 g/cm³  (linear EOS)
  - Domain: [0, 7] cm, 140 cells
  - Energy groups: logarithmically spaced from 1e-4 to 10 keV
  - Output times: 1, 2, 5, 10 ns

Run from the DiscreteOrdinates directory:
    python problems/test_marshak_wave_multigroup_powerlaw.py
    python problems/test_marshak_wave_multigroup_powerlaw.py --groups 50
    python problems/test_marshak_wave_multigroup_powerlaw.py --diff-file ../../marshak_wave_multigroup_powerlaw_10g_no_precond_timeBC.npz
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BPoly
from numba import njit

# Parent directory for solver imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import sn_solver
import mg_sn_solver

# Project root for shared utilities
project_root = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, project_root)
try:
    from utils.plotfuncs import show, hide_spines, font
    HAS_PLOTFUNCS = True
except ImportError:
    HAS_PLOTFUNCS = False

# Planck integral library (rational-approximation based, fast & accurate)
from planck_integrals import Bg as _Bg_scalar, dBgdT as _dBgdT_scalar

# ── Physical constants ──────────────────────────────────────────────────
C_LIGHT = sn_solver.c     # 29.98 cm/ns
A_RAD   = sn_solver.a     # 0.01372 GJ/(cm³ keV⁴)
AC      = sn_solver.ac    # a · c
RHO     = 0.01            # g/cm³
CV_MASS = 0.05            # GJ/(g·keV)
CV_VOL  = CV_MASS * RHO   # GJ/(cm³·keV)

# ── Planck group integrals (vectorised wrappers) ───────────────────────
#
# The planck_integrals library returns specific intensity per steradian.
# The S_N solver uses the "4π" forms, so we multiply by 4π.

def _Bg_4pi(E_low, E_high, T):
    r"""Return :math:`4\pi\,B_g(T)` for group *g*, vectorised over *T*."""
    T_arr = np.maximum(np.asarray(T, dtype=float), 1e-6)
    shape = T_arr.shape
    T_flat = T_arr.ravel()
    result = np.array([_Bg_scalar(E_low, E_high, t) for t in T_flat])
    return (4.0 * np.pi * result).reshape(shape)


def _dBgdT_4pi(E_low, E_high, T):
    r"""Return :math:`4\pi\,dB_g/dT`, vectorised over *T*."""
    T_arr = np.maximum(np.asarray(T, dtype=float), 1e-6)
    shape = T_arr.shape
    T_flat = T_arr.ravel()
    result = np.array([_dBgdT_scalar(E_low, E_high, t) for t in T_flat])
    return (4.0 * np.pi * result).reshape(shape)


# ── Power-law opacity ───────────────────────────────────────────────────

def powerlaw_opacity_at_energy(T, E, rho=RHO):
    r"""Continuous opacity  :math:`\sigma_a(T,E) = 10\,\rho\,T^{-1/2}\,E^{-3}`."""
    T_safe = np.maximum(T, 1e-2)
    return np.minimum(10.0 * rho * T_safe**(-0.5) * E**(-3.0), 1e14)


def make_powerlaw_opacity_func(E_low, E_high, rho=RHO):
    """Group opacity: geometric mean at the two energy boundaries."""
    def sigma_a(T):
        return np.sqrt(powerlaw_opacity_at_energy(T, E_low, rho)
                       * powerlaw_opacity_at_energy(T, E_high, rho))
    return sigma_a


# ── Factories for Bg / dBdT closures ───────────────────────────────────

def _make_Bg(El, Eh):
    def Bg(T):
        return _Bg_4pi(El, Eh, T)
    return Bg


def _make_dBdT(El, Eh):
    def dBdT(T):
        return _dBgdT_4pi(El, Eh, T)
    return dBdT


def _domain_energy(phi_g, T, hx):
    """Total material + radiation energy in domain (per unit area)."""
    e_mat = CV_VOL * T
    E_mat = hx * np.sum(np.mean(e_mat, axis=1))
    E_rad = 0.0
    for g in range(len(phi_g)):
        E_rad += hx * np.sum(np.mean(phi_g[g], axis=1) / C_LIGHT)
    return E_mat + E_rad, E_mat, E_rad


def _boundary_incoming_power(BCs, t, MU, W, order):
    """Incoming boundary power into the domain (per unit area per ns)."""
    nop1 = order + 1
    p_in = 0.0
    for bc_g in BCs:
        bc = bc_g(t)

        # Left boundary: incoming directions have mu > 0.
        # In the sweep, that uses boundary coefficient at index nop1-1.
        mask_left = MU > 0.0
        if np.any(mask_left):
            p_in += np.sum(W[mask_left] * MU[mask_left] * bc[mask_left, nop1 - 1])

        # Right boundary: incoming directions have mu < 0.
        # In the sweep, that uses boundary coefficient at index 0.
        mask_right = MU < 0.0
        if np.any(mask_right):
            p_in += np.sum(W[mask_right] * (-MU[mask_right]) * bc[mask_right, 0])

    return float(p_in)


def _build_conservation_history(ts, phi_g_hist, T_hist, hx, BCs, MU, W, order,
                                diagnostics_store=None):
    """Build per-step conservation diagnostics using boundary inflow accounting."""
    if len(ts) < 2:
        return {
            'E0': None,
            'time': [], 'dt': [],
            'E_mat': [], 'E_rad': [], 'E_total': [],
            'E_in_step': [], 'E_in_cum': [],
            'E_out_step': [], 'E_out_cum': [],
            'E_net_step': [], 'E_net_cum': [],
            'E_expected': [], 'drift': [],
        }

    E0, E0_mat, E0_rad = _domain_energy(phi_g_hist[0], T_hist[0], hx)

    hist = {
        'E0': float(E0),
        'E0_mat': float(E0_mat),
        'E0_rad': float(E0_rad),
        'time': [], 'dt': [],
        'E_mat': [], 'E_rad': [], 'E_total': [],
        'E_in_step': [],
        'E_in_cum': [],
        'E_out_step': [],
        'E_out_cum': [],
        'E_net_step': [],
        'E_net_cum': [],
        'E_expected': [],
        'drift': [],
    }

    Ein_cum = 0.0
    Eout_cum = 0.0
    Enet_cum = 0.0
    for k in range(1, len(ts)):
        t0 = float(ts[k - 1])
        t1 = float(ts[k])
        dt = t1 - t0

        if diagnostics_store is not None and (k - 1) < len(diagnostics_store):
            diag = diagnostics_store[k - 1]
            Etot = float(diag['E_total'])
            Emat = float(diag['E_mat'])
            Erad = float(diag['E_rad'])
            Ein = float(diag['boundary_in'])
            Eout = float(diag['boundary_out'])
            Enet = float(diag['net_boundary'])
        else:
            tmid = 0.5 * (t0 + t1)
            Etot, Emat, Erad = _domain_energy(phi_g_hist[k], T_hist[k], hx)
            p_in = _boundary_incoming_power(BCs, tmid, MU, W, order)
            Ein = dt * p_in
            Eout = 0.0
            Enet = Ein

        Ein_cum += Ein
        Eout_cum += Eout
        Enet_cum += Enet

        Eexp = E0 + Enet_cum
        drift = Etot - Eexp

        hist['time'].append(t1)
        hist['dt'].append(dt)
        hist['E_mat'].append(float(Emat))
        hist['E_rad'].append(float(Erad))
        hist['E_total'].append(float(Etot))
        hist['E_in_step'].append(float(Ein))
        hist['E_in_cum'].append(float(Ein_cum))
        hist['E_out_step'].append(float(Eout))
        hist['E_out_cum'].append(float(Eout_cum))
        hist['E_net_step'].append(float(Enet))
        hist['E_net_cum'].append(float(Enet_cum))
        hist['E_expected'].append(float(Eexp))
        hist['drift'].append(float(drift))

    return hist


def print_conservation_diagnostics(conservation):
    """Print open-boundary conservation diagnostics with boundary inflow."""
    if len(conservation['time']) == 0:
        print("\nNo conservation diagnostics available.")
        return

    E0 = conservation['E0']
    E_tot = np.asarray(conservation['E_total'])
    E_in_cum = np.asarray(conservation['E_in_cum'])
    E_out_cum = np.asarray(conservation['E_out_cum'])
    E_net_cum = np.asarray(conservation['E_net_cum'])
    drift = np.asarray(conservation['drift'])

    scale_E0 = max(abs(E0), 1e-300)
    scale_gain = max(abs(E_tot[-1] - E0), 1e-300)
    scale_net = max(abs(E_net_cum[-1]), 1e-300)

    print("\n" + "=" * 78)
    print("Open-boundary energy ledger (accounts for prescribed incoming BC energy)")
    print("=" * 78)
    print(f"  Initial total energy:          {E0:.8e}")
    print(f"  Final total energy:            {E_tot[-1]:.8e}")
    print(f"  Cumulative boundary inflow:    {E_in_cum[-1]:.8e}")
    print(f"  Cumulative boundary outflow:   {E_out_cum[-1]:.8e}")
    print(f"  Cumulative net boundary:       {E_net_cum[-1]:.8e}")
    print(f"  Final residual:                {drift[-1]:+.8e}")
    print(f"  Final residual / (E_final-E0): {drift[-1] / scale_gain:+.3e}")
    print(f"  Final residual / Enet:         {drift[-1] / scale_net:+.3e}")
    print(f"  Final residual / E0:           {drift[-1] / scale_E0:+.3e}")
    print(f"  Max |residual| / Enet:         {np.max(np.abs(drift)) / scale_net:.3e}")

    # For this open system, residual lumps outgoing leakage + numerical error.
    print("  Note: residual = E_tot - (E0 + cumulative net boundary energy).")

    t = np.asarray(conservation['time'])
    E_mat = np.asarray(conservation['E_mat'])
    E_rad = np.asarray(conservation['E_rad'])
    E_in_step = np.asarray(conservation['E_in_step'])
    E_out_step = np.asarray(conservation['E_out_step'])
    E_in_cum_step = np.asarray(conservation['E_in_cum'])
    E_out_cum_step = np.asarray(conservation['E_out_cum'])
    E_net_cum_step = np.asarray(conservation['E_net_cum'])
    n = len(t)
    sample_ids = sorted(set([0, min(1, n - 1), min(2, n - 1), n - 1]))
    print("\n  Sample timestep ledger:")
    print("  t(ns)      E_mat        E_rad        E_tot        E_in(step)  E_out(step) E_in(cum)   E_out(cum)  E_net(cum)  residual")
    for i in sample_ids:
        print(f"  {t[i]:8.3e}  {E_mat[i]:11.4e}  {E_rad[i]:11.4e}  "
              f"{E_tot[i]:11.4e}  {E_in_step[i]:11.4e}  {E_out_step[i]:11.4e}  "
              f"{E_in_cum_step[i]:11.4e}  {E_out_cum_step[i]:11.4e}  {E_net_cum_step[i]:11.4e}  {drift[i]:+11.4e}")


# ── Setup & run ─────────────────────────────────────────────────────────

def setup_and_run(I=140, order=3, N=8, n_groups=10,
                  Lx=7.0,
                  tfinal=10.0, dt_min=1e-4, dt_max=0.01,
                  K=800, maxits=2000, LOUD=0,
                  time_dependent_bc=True, output_times=None,
                  fix=0,
                  fleck_mode='legacy'):
    r"""Run multigroup power-law Marshak wave with S_N transport.

    Parameters
    ----------
    I : int
        Spatial zones.
    order : int
        Bernstein polynomial order.
    N : int
        Number of discrete ordinates.
    n_groups : int
        Energy groups.
    Lx : float
        Domain length (cm).
    tfinal : float
        Final time (ns).
    dt_min, dt_max : float
        Adaptive time-step bounds (ns).
    K, maxits : int
        DMD parameters.
    LOUD : int
        Verbosity.
    time_dependent_bc : bool
        Ramp *T_bc* from 0.05 → 0.25 keV over 5 ns.
    output_times : array-like or None
        Specific output times (ns).  Default ``[1, 2, 5, 10]``.
    fix : int
        Positivity fix-up flag passed to S_N sweeps (0 is conservative).
    fleck_mode : {"legacy", "imc"}
        ``"legacy"`` uses the original dB/dT multigroup linearisation.
        ``"imc"`` uses the IMC multigroup Fleck form with
        :math:`\sigma_P = \sum_g \sigma_g b_g^\star` and effective-scatter
        redistribution proportional to :math:`\sigma_g b_g^\star`.

    Returns
    -------
    results : dict
    """
    if output_times is None:
        output_times = np.array([1.0, 2.0, 5.0, 10.0])
    output_times = np.asarray(output_times, dtype=float)
    output_times = output_times[output_times <= tfinal + 1e-12]

    G = n_groups
    I_full = I
    hx = Lx / I_full
    x = np.linspace(hx / 2, Lx - hx / 2, I_full)
    x_full = x
    nop1 = order + 1

    # Energy group edges (keV) – logarithmically spaced
    energy_edges = np.logspace(np.log10(1e-4), np.log10(10.0), G + 1)

    # Material -----------------------------------------------------------
    T_init = 0.005           # keV
    T_bc_start = 0.05 if time_dependent_bc else 0.25
    T_bc_end   = 0.25
    t_ramp     = 5.0         # ns

    def T_bc_func(t):
        if time_dependent_bc and t < t_ramp:
            return T_bc_start + (T_bc_end - T_bc_start) * (t / t_ramp)
        return T_bc_end

    # EOS:  e = c_v · T  (linear, volumetric)
    @njit
    def eos(T):
        return CV_VOL * T

    @njit
    def invEOS(E):
        return E / CV_VOL

    def Cv_func(T):
        return np.full_like(T, CV_VOL)

    # Group physics ------------------------------------------------------
    sigma_a_funcs = []
    scat_funcs    = []
    Bg_funcs      = []
    dBdT_funcs    = []

    _zero_scat = lambda T: np.zeros_like(T)

    for g in range(G):
        El, Eh = energy_edges[g], energy_edges[g + 1]
        sigma_a_funcs.append(make_powerlaw_opacity_func(El, Eh))
        scat_funcs.append(_zero_scat)
        Bg_funcs.append(_make_Bg(El, Eh))
        dBdT_funcs.append(_make_dBdT(El, Eh))

    # Planck normalisation sanity check
    T_check = np.array([[0.1, 0.2], [0.25, 0.05]])
    Bg_sum = sum(Bg_funcs[g](T_check) for g in range(G))
    expected = AC * T_check**4
    frac_err = np.max(np.abs(Bg_sum - expected) / (expected + 1e-300))
    print(f"Planck normalisation check: max frac. error = {frac_err:.2e}")

    # External source (none — radiation enters via boundary)
    q_ext = [np.zeros((I_full, N, nop1)) for _ in range(G)]

    # Angular quadrature (for boundary conditions)
    MU, _W = np.polynomial.legendre.leggauss(N)

    # Per-group boundary conditions
    # Left:  Marshak BC — incoming blackbody at T_bc (μ > 0 directions)
    # Right: cold incoming at T_init (μ < 0 directions) — essentially vacuum
    def _make_bc_g(g):
        El_, Eh_ = energy_edges[g], energy_edges[g + 1]
        def bc_g(t):
            out = np.zeros((N, nop1))
            Tbc   = T_bc_func(t)
            Bg_in = 4.0 * np.pi * _Bg_scalar(El_, Eh_, max(Tbc, 1e-6))
            Bg_right = 4.0 * np.pi * _Bg_scalar(El_, Eh_, T_init)
            out[MU > 0, :] = Bg_in     # left:  blackbody at T_bc
            out[MU < 0, :] = Bg_right   # right: cold incoming (≈ vacuum)
            return out
        return bc_g

    BCs = [_make_bc_g(g) for g in range(G)]

    # Initial conditions
    T     = np.ones((I_full, nop1)) * T_init
    phi_g = [Bg_funcs[g](T) for g in range(G)]
    psi_g = [phi_g[g][:, None, :] + np.zeros((I_full, N, nop1))
             for g in range(G)]

    # Print summary
    print(f"\nMultigroup Marshak Wave — S_N Transport")
    print(f"  Groups : {G},  Energy : [{energy_edges[0]:.1e}, "
          f"{energy_edges[-1]:.1f}] keV")
    print(f"  Domain : [0, {Lx}] cm")
    print(f"  Cells  : {I_full},  order {order}")
    print(f"  S_{N} ordinates : {N}")
    print(f"  T_init = {T_init} keV")
    print(f"  T_bc   : {'ramped' if time_dependent_bc else 'constant'} "
          f"({T_bc_start} → {T_bc_end} keV over {t_ramp} ns)")
    print(f"  ρ = {RHO} g/cm³,  c_v = {CV_MASS} GJ/(g·keV)")
    print(f"  dt ∈ [{dt_min:.1e}, {dt_max:.1e}] ns,  t_final = {tfinal} ns")
    print(f"  Fleck mode: {fleck_mode}")
    print(f"  Output times : {output_times}")
    for g in range(G):
        El, Eh = energy_edges[g], energy_edges[g + 1]
        sig_test = sigma_a_funcs[g](np.array([[T_bc_end]]))[0, 0]
        Bg_test  = Bg_funcs[g](np.array([[T_bc_end]]))[0, 0]
        print(f"  Group {g:2d} [{El:10.4e}, {Eh:10.4e}] keV: "
              f"σ_a = {sig_test:.3e} cm⁻¹,  4πB_g = {Bg_test:.3e}")

    # Run solver ---------------------------------------------------------
    diagnostics_store = []
    phi_g_hist, T_hist, iterations, ts = mg_sn_solver.mg_temp_solve_dmd_inc(
        I_full, hx,
        G, sigma_a_funcs, scat_funcs, Bg_funcs, dBdT_funcs,
        q_ext, N, BCs, eos, invEOS, Cv_func,
        phi_g, psi_g, T,
        dt_min=dt_min, dt_max=dt_max, tfinal=tfinal,
        order=order, LOUD=LOUD, maxits=maxits, fix=fix, K=K, R=3,
        time_outputs=output_times,
        energy_diagnostics=True,
        diagnostics_stride=10,
        diagnostics_store=diagnostics_store,
        fleck_mode=fleck_mode,
    )

    conservation = _build_conservation_history(
        ts, phi_g_hist, T_hist, hx, BCs, MU, _W / np.sum(_W), order,
        diagnostics_store=diagnostics_store)
    print_conservation_diagnostics(conservation)

    print(f"\nDone.  Total transport sweeps: {iterations}")

    solutions = []
    for t_out in output_times:
        tstep = np.argmin(np.abs(t_out - ts))
        phi_snap = phi_g_hist[tstep]
        T_snap   = T_hist[tstep]
        phi_phys = [phi_snap[g] for g in range(G)]
        T_rad    = (sum(phi_phys[g] for g in range(G)) / AC)**0.25

        solutions.append({
            'time':  ts[tstep],
            'phi_g': [phi_phys[g].copy() for g in range(G)],
            'T':     T_snap.copy(),
            'T_rad': T_rad.copy(),
        })
        print(f"  t = {ts[tstep]:7.4f} ns  |  T_max = {T_snap.max():.6f}  "
              f"T_rad_max = {T_rad.max():.6f} keV")

    return {
        'solutions': solutions, 'x': x, 'hx': hx, 'Lx': Lx,
        'order': order, 'iterations': iterations, 'ts': ts,
        'energy_edges': energy_edges, 'n_groups': G,
        'phi_g_hist': phi_g_hist, 'T_hist': T_hist,
        'conservation': conservation,
        'diagnostics_store': diagnostics_store,
    }


# ── Plotting ────────────────────────────────────────────────────────────

def plot_results(results, diff_npz=None, savefile=''):
    """Plot material and radiation temperature profiles.

    If *diff_npz* is the path to the non-equilibrium diffusion NPZ
    output, overlay those curves for comparison.
    """
    solutions = results['solutions']
    x     = results['x']
    Lx    = results['Lx']
    order = results['order']
    nI    = len(x)
    hx    = results['hx']
    edges = np.linspace(0, Lx, nI + 1)
    xplot = np.linspace(hx / 2, Lx, 2000)

    # Load diffusion reference (optional)
    diff = None
    if diff_npz and os.path.exists(diff_npz):
        diff = np.load(diff_npz)
        print(f"Loaded diffusion reference: {diff_npz}")

    colors = ['C0', 'C1', 'C2', 'C3', 'C4']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for idx, sol in enumerate(solutions):
        col = colors[idx % len(colors)]
        t_ns = sol['time']

        T_interp  = BPoly(sol['T'].T, edges)(xplot)
        Tr_interp = BPoly(sol['T_rad'].T, edges)(xplot)

        ax1.plot(xplot, T_interp, '-',  color=col, lw=1.5,
                 label=f'mat  t = {t_ns:.1f} ns')
        ax1.plot(xplot, Tr_interp, '--', color=col, lw=1.0,
                 label=f'rad  t = {t_ns:.1f} ns')

        ax2.semilogy(xplot, T_interp,  '-',  color=col, lw=1.5)
        ax2.semilogy(xplot, Tr_interp, '--', color=col, lw=1.0)

        # Overlay diffusion if available
        if diff is not None:
            d_times = diff['times']
            d_r     = diff['r']
            tidx    = np.argmin(np.abs(t_ns - d_times))
            ax1.plot(d_r, diff['T_mat'][tidx], ':',  color=col, lw=1.2,
                     alpha=0.7)
            ax1.plot(d_r, diff['T_rad'][tidx], '-.', color=col, lw=1.0,
                     alpha=0.7)

    ax1.set_xlabel('x (cm)')
    ax1.set_ylabel('Temperature (keV)')
    ax1.set_title(r'Multigroup S$_N$ Marshak Wave')
    ax1.legend(fontsize=7, ncol=2)
    ax1.set_xlim(0, Lx)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('x (cm)')
    ax2.set_ylabel('Temperature (keV)')
    ax2.set_title('Log scale')
    ax2.set_xlim(0, Lx)
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    if savefile:
        plt.savefig(savefile, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {savefile}")
    if HAS_PLOTFUNCS:
        show('marshak_wave_powerlaw_sn.pdf', close_after=True)
    else:
        plt.show()


# ── CLI ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Multigroup Marshak Wave — S_N with power-law opacity')
    parser.add_argument('--zones', type=int, default=140,
                        help='Number of spatial zones (default: 140)')
    parser.add_argument('--order', type=int, default=3,
                        help='Bernstein polynomial order (default: 3)')
    parser.add_argument('--N', type=int, default=8,
                        help='Number of discrete ordinates (default: 8)')
    parser.add_argument('--groups', type=int, default=10,
                        help='Number of energy groups (default: 10)')
    parser.add_argument('--Lx', type=float, default=7.0,
                        help='Domain length in cm (default: 7.0)')
    parser.add_argument('--tfinal', type=float, default=10.0,
                        help='Final time in ns (default: 10)')
    parser.add_argument('--dt-min', type=float, default=1e-4,
                        help='Minimum time step (default: 1e-4)')
    parser.add_argument('--dt-max', type=float, default=0.01,
                        help='Maximum time step (default: 0.01)')
    parser.add_argument('--K', type=int, default=800,
                        help='DMD inner iterations (default: 800)')
    parser.add_argument('--maxits', type=int, default=2000,
                        help='Max iterations per step (default: 2000)')
    parser.add_argument('--loud', type=int, default=0,
                        help='Verbosity level (default: 0)')
    parser.add_argument('--fix', type=int, default=0,
                        help='Positivity fix-up (0=off, 1=on; default: 0)')
    parser.add_argument('--fleck-mode', choices=['legacy', 'imc'], default='legacy',
                        help='Multigroup Fleck coupling mode (default: legacy)')
    parser.add_argument('--no-time-bc', action='store_true',
                        help='Use constant T_bc instead of ramp')
    parser.add_argument('--output-times', type=float, nargs='+',
                        default=[1.0, 2.0, 5.0, 10.0],
                        help='Output times in ns (default: 1 2 5 10)')
    parser.add_argument('--save-npz', type=str, default='',
                        help='Output .npz file name')
    parser.add_argument('--diff-file', type=str, default='',
                        help='Diffusion NPZ file for comparison overlay')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip plotting')
    parser.add_argument('--save-fig', type=str, default='',
                        help='Save figure to file')
    args = parser.parse_args()

    results = setup_and_run(
        I=args.zones, order=args.order, N=args.N, n_groups=args.groups,
        Lx=args.Lx,
        tfinal=args.tfinal, dt_min=args.dt_min, dt_max=args.dt_max,
        K=args.K, maxits=args.maxits, LOUD=args.loud,
        time_dependent_bc=not args.no_time_bc,
        output_times=np.array(args.output_times),
        fix=args.fix,
        fleck_mode=args.fleck_mode,
    )

    # Save results
    if args.save_npz:
        npz_name = args.save_npz
    else:
        bc_tag = '_timeBC' if not args.no_time_bc else ''
        npz_name = (f"marshak_wave_powerlaw_sn_{args.groups}g"
                     f"{bc_tag}.npz")

    edges = np.linspace(0, results['Lx'], len(results['x']) + 1)
    x_centers = results['x']
    G = results['n_groups']
    times_out = np.array([s['time'] for s in results['solutions']])
    T_mat  = np.array([BPoly(s['T'].T, edges)(x_centers)
                       for s in results['solutions']])
    T_rad  = np.array([BPoly(s['T_rad'].T, edges)(x_centers)
                       for s in results['solutions']])
    # phi_groups: (n_times, n_groups, n_cells); E_r_groups = phi/c
    phi_groups = np.array([
        [BPoly(s['phi_g'][g].T, edges)(x_centers) for g in range(G)]
        for s in results['solutions']
    ])
    E_r_groups = phi_groups / C_LIGHT
    E_r = E_r_groups.sum(axis=1)

    np.savez_compressed(
        npz_name,
        times=times_out,
        r=x_centers,
        energy_edges=results['energy_edges'],
        T_mat=T_mat,
        T_rad=T_rad,
        phi_groups=phi_groups,
        E_r_groups=E_r_groups,
        E_r=E_r,
    )
    print(f"Results saved to {npz_name}")

    if not args.no_plot or args.save_fig:
        plot_results(results, diff_npz=args.diff_file,
                     savefile=args.save_fig)


if __name__ == '__main__':
    main()
