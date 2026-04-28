#!/usr/bin/env python3
"""
Gaussian IC Equilibrium Test — Multigroup S_N Transport

Starts from an equilibrium Gaussian temperature profile to isolate
transport vs. diffusion physics, independently of boundary condition
treatment.  Both walls are reflecting so there is no incoming radiation.
The wave front does not reach the walls during the short run, making the
result insensitive to BC details.

Problem setup:
  - T(x,0) = max(T_floor, T_max · exp(-(x-x₀)²/(2σ²)))
    T_max = 0.2 keV,  x₀ = 3.5 cm,  σ = 0.8 cm,  T_floor = 0.001 keV
  - Radiation in equilibrium:  ψ_g = 4π B_g(T(x))  (isotropic)
  - Reflecting walls:  no incoming flux, reflect_left = reflect_right = True
  - Opacity:   σ_a(T,E) = 10 ρ T^{-1/2} E^{-3}  (cm⁻¹)
  - ρ = 0.01 g/cm³,  c_v = 0.05 GJ/(g·keV)  (same as Marshak wave tests)
  - Groups: G=10, [1e-4, 10] keV log-spaced
  - Output: 0.05, 0.1, 0.2 ns

Run from the DiscreteOrdinates directory:
    python problems/test_gaussian_ic_sn.py
    python problems/test_gaussian_ic_sn.py --groups 20 --tfinal 0.3
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BPoly
from numba import njit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import sn_solver
import mg_sn_solver

project_root = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, project_root)
try:
    from utils.plotfuncs import show, font
    HAS_PLOTFUNCS = True
except ImportError:
    HAS_PLOTFUNCS = False

from planck_integrals import Bg as _Bg_scalar, dBgdT as _dBgdT_scalar

# ── Physical constants ──────────────────────────────────────────────────
C_LIGHT = sn_solver.c    # 29.98 cm/ns
A_RAD   = sn_solver.a    # 0.01372 GJ/(cm³ keV⁴)
AC      = sn_solver.ac   # a·c
RHO     = 0.01           # g/cm³
CV_MASS = 0.05           # GJ/(g·keV)
CV_VOL  = CV_MASS * RHO  # 5e-4 GJ/(cm³·keV)

# ── Gaussian IC parameters ──────────────────────────────────────────────
T_MAX   = 0.2    # keV   — peak temperature
X_0     = 3.5   # cm    — centre of Gaussian
SIGMA   = 0.8   # cm    — width
T_FLOOR = 0.001  # keV   — cold background


def gaussian_T(x):
    """Cell-centre temperature profile at t=0."""
    return np.maximum(T_FLOOR,
                      T_MAX * np.exp(-0.5 * ((x - X_0) / SIGMA)**2))


# ── Planck wrappers ─────────────────────────────────────────────────────

def _Bg_4pi(E_low, E_high, T):
    T_arr = np.maximum(np.asarray(T, dtype=float), 1e-9)
    shape = T_arr.shape
    result = np.array([_Bg_scalar(E_low, E_high, t) for t in T_arr.ravel()])
    return (4.0 * np.pi * result).reshape(shape)


def _dBgdT_4pi(E_low, E_high, T):
    T_arr = np.maximum(np.asarray(T, dtype=float), 1e-9)
    shape = T_arr.shape
    result = np.array([_dBgdT_scalar(E_low, E_high, t) for t in T_arr.ravel()])
    return (4.0 * np.pi * result).reshape(shape)


def _make_Bg(El, Eh):
    def Bg(T):
        return _Bg_4pi(El, Eh, T)
    return Bg


def _make_dBdT(El, Eh):
    def dBdT(T):
        return _dBgdT_4pi(El, Eh, T)
    return dBdT


# ── Power-law opacity (same as Marshak wave tests) ─────────────────────

def powerlaw_opacity_at_energy(T, E, rho=RHO):
    T_safe = np.maximum(T, 1e-2)
    return np.minimum(10.0 * rho * T_safe**(-0.5) * E**(-3.0), 1e14)


def make_powerlaw_opacity_func(E_low, E_high, rho=RHO):
    def sigma_a(T):
        return np.sqrt(powerlaw_opacity_at_energy(T, E_low, rho)
                       * powerlaw_opacity_at_energy(T, E_high, rho))
    return sigma_a


# ── Main ────────────────────────────────────────────────────────────────

def run(I=140, order=3, N=8, n_groups=10,
        tfinal=0.2, dt_min=1e-4, dt_max=0.005,
        K=800, maxits=2000, LOUD=0, fix=1,
        output_times=None):
    """Run the Gaussian IC equilibrium test with S_N transport."""

    if output_times is None:
        output_times = np.array([0.05, 0.1, 0.2])
    output_times = np.asarray(output_times, dtype=float)
    output_times = output_times[output_times <= tfinal + 1e-12]

    G    = n_groups
    Lx   = 7.0
    hx   = Lx / I
    x    = np.linspace(hx / 2, Lx - hx / 2, I)
    nop1 = order + 1

    energy_edges = np.logspace(np.log10(1e-4), np.log10(10.0), G + 1)

    # EOS
    @njit
    def eos(T):
        return CV_VOL * T

    @njit
    def invEOS(E):
        return E / CV_VOL

    def Cv_func(T):
        return np.full_like(T, CV_VOL)

    # Group physics
    sigma_a_funcs, scat_funcs, Bg_funcs, dBdT_funcs = [], [], [], []
    _zero_scat = lambda T: np.zeros_like(T)
    for g in range(G):
        El, Eh = energy_edges[g], energy_edges[g + 1]
        sigma_a_funcs.append(make_powerlaw_opacity_func(El, Eh))
        scat_funcs.append(_zero_scat)
        Bg_funcs.append(_make_Bg(El, Eh))
        dBdT_funcs.append(_make_dBdT(El, Eh))

    # Planck normalisation check
    T_check = np.array([[0.1, 0.2], [0.25, 0.05]])
    Bg_sum  = sum(Bg_funcs[g](T_check) for g in range(G))
    frac_err = np.max(np.abs(Bg_sum - AC * T_check**4) / (AC * T_check**4))
    print(f"Planck normalisation check: max frac. error = {frac_err:.2e}")

    # External source: none
    q_ext = [np.zeros((I, N, nop1)) for _ in range(G)]

    # Quadrature (for BC array shape; content is all zeros)
    MU, _W = np.polynomial.legendre.leggauss(N)

    # BCs: zero incoming flux at both walls; reflection handled by flags
    def zero_bc(t):
        return np.zeros((N, nop1))

    BCs = [zero_bc for _ in range(G)]

    # ── Initial condition: equilibrium Gaussian ────────────────────────
    # Cell-centre T, tiled over Bernstein nodes (uniform within each cell)
    T_cell  = gaussian_T(x)                           # (I,)
    T_init  = np.tile(T_cell[:, np.newaxis], (1, nop1))  # (I, nop1)

    # Equilibrium scalar flux:  phi_g = 4π B_g(T)
    phi_g_init = [Bg_funcs[g](T_init) for g in range(G)]

    # Isotropic angular flux:  with W normalised to 1, psi_iso = phi_g
    # (phi = sum_n w_n psi_n  → at isotropic: phi = psi_iso · sum(w_n) = psi_iso)
    psi_g_init = [phi_g_init[g][:, np.newaxis, :] * np.ones((I, N, nop1))
                  for g in range(G)]

    # Print setup summary
    print(f"\nGaussian IC Test — Multigroup S_N Transport")
    print(f"  T_max={T_MAX} keV  x₀={X_0} cm  σ={SIGMA} cm  T_floor={T_FLOOR} keV")
    print(f"  Groups : {G},  [{energy_edges[0]:.1e}, {energy_edges[-1]:.1f}] keV")
    print(f"  Cells  : {I},  order {order},  S_{N} ordinates {N}")
    print(f"  BCs    : reflecting (both walls)")
    print(f"  ρ = {RHO} g/cm³,  c_v = {CV_MASS} GJ/(g·keV)")
    print(f"  dt ∈ [{dt_min:.1e}, {dt_max:.1e}] ns,  t_final = {tfinal} ns")
    print(f"  T_cell range: [{T_cell.min():.5f}, {T_cell.max():.5f}] keV")

    # ── Run solver ─────────────────────────────────────────────────────
    phi_g_hist, T_hist, iterations, ts = mg_sn_solver.mg_temp_solve_dmd_inc(
        I, hx,
        G, sigma_a_funcs, scat_funcs, Bg_funcs, dBdT_funcs,
        q_ext, N, BCs, eos, invEOS, Cv_func,
        phi_g_init, psi_g_init, T_init,
        dt_min=dt_min, dt_max=dt_max, tfinal=tfinal,
        order=order, LOUD=LOUD, maxits=maxits, fix=fix, K=K, R=3,
        time_outputs=output_times,
        reflect_left=True,
        reflect_right=True,
    )
    print(f"\nDone.  Total transport sweeps: {iterations}")

    # ── Extract solutions ───────────────────────────────────────────────
    solutions = []
    for t_out in output_times:
        tstep   = np.argmin(np.abs(t_out - np.array(ts)))
        phi_snap = phi_g_hist[tstep]
        T_snap   = T_hist[tstep]
        phi_phys = [phi_snap[g] for g in range(G)]
        T_rad    = (sum(phi_phys[g] for g in range(G)) / AC)**0.25
        solutions.append({
            'time':  ts[tstep],
            'phi_g': [p.copy() for p in phi_phys],
            'T':     T_snap.copy(),
            'T_rad': T_rad.copy(),
        })
        print(f"  t = {ts[tstep]:7.4f} ns  |  T_max = {T_snap.max():.6f}  "
              f"T_rad_max = {T_rad.max():.6f} keV")

    return {'solutions': solutions, 'x': x, 'hx': hx, 'Lx': Lx,
            'order': order, 'ts': ts, 'energy_edges': energy_edges,
            'n_groups': G, 'phi_g_hist': phi_g_hist, 'T_hist': T_hist}


def save_npz(results, npz_name):
    """Save results to compressed NPZ (same format as Marshak wave files)."""
    solutions = results['solutions']
    x_c       = results['x']
    G         = results['n_groups']
    times_out = np.array([s['time'] for s in solutions])

    # For Bernstein polynomial coefficients of order p, the cell integral is
    #   ∫ f dx = hx * mean(coefficients)
    # so the cell-average value equals the mean of the nop1 coefficients.
    # Using BPoly evaluation at the cell centre gives weights [1/8,3/8,3/8,1/8]
    # (for order 3) instead of [1/4,1/4,1/4,1/4], which causes a systematic
    # overestimate of the energy for non-constant within-cell profiles.
    T_mat  = np.array([s['T'].mean(axis=1)     for s in solutions])
    T_rad  = np.array([s['T_rad'].mean(axis=1) for s in solutions])
    phi_groups = np.array([
        [s['phi_g'][g].mean(axis=1) for g in range(G)]
        for s in solutions
    ])
    E_r_groups = phi_groups / sn_solver.c
    E_r        = E_r_groups.sum(axis=1)

    np.savez_compressed(
        npz_name,
        times=times_out, r=x_c,
        energy_edges=results['energy_edges'],
        T_mat=T_mat, T_rad=T_rad,
        phi_groups=phi_groups, E_r_groups=E_r_groups, E_r=E_r,
    )
    print(f"Results saved to {npz_name}")


def plot_results(results, savefile=''):
    solutions = results['solutions']
    x     = results['x']
    Lx    = results['Lx']
    nI    = len(x)
    hx    = results['hx']
    order = results['order']
    edges = np.linspace(0, Lx, nI + 1)
    xplot = np.linspace(hx / 2, Lx, 1000)
    colors = ['C0', 'C1', 'C2', 'C3']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for idx, sol in enumerate(solutions):
        col = colors[idx % len(colors)]
        T_interp  = BPoly(sol['T'].T,     edges)(xplot)
        Tr_interp = BPoly(sol['T_rad'].T, edges)(xplot)
        ax1.plot(xplot, T_interp,  '-',  color=col, lw=1.5, label=f'mat  t={sol["time"]:.2f} ns')
        ax1.plot(xplot, Tr_interp, '--', color=col, lw=1.0, label=f'rad  t={sol["time"]:.2f} ns')
        ax2.semilogy(xplot, T_interp,  '-',  color=col, lw=1.5)
        ax2.semilogy(xplot, Tr_interp, '--', color=col, lw=1.0)

    for ax in (ax1, ax2):
        ax.set_xlabel('x (cm)')
        ax.set_ylabel('Temperature (keV)')
        ax.set_xlim(0, Lx)
        ax.grid(True, alpha=0.3)
    ax1.legend(fontsize=7, ncol=2)
    ax1.set_title(r'Gaussian IC — Multigroup S$_N$')
    ax2.set_title('Log scale')
    plt.tight_layout()
    if savefile:
        plt.savefig(savefile, dpi=150, bbox_inches='tight')
    if HAS_PLOTFUNCS:
        show('gaussian_ic_sn.pdf', close_after=True)
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Gaussian IC equilibrium test — S_N transport')
    parser.add_argument('--zones',    type=int,   default=140)
    parser.add_argument('--order',    type=int,   default=3)
    parser.add_argument('--N',        type=int,   default=8)
    parser.add_argument('--groups',   type=int,   default=10)
    parser.add_argument('--tfinal',   type=float, default=0.2)
    parser.add_argument('--dt-min',   type=float, default=1e-4)
    parser.add_argument('--dt-max',   type=float, default=0.005)
    parser.add_argument('--K',        type=int,   default=800)
    parser.add_argument('--maxits',   type=int,   default=2000)
    parser.add_argument('--fix',      type=int,   default=1,
                        help='Positivity fix-up (0=off, 1=on)')
    parser.add_argument('--loud',     type=int,   default=0)
    parser.add_argument('--output-times', type=float, nargs='+',
                        default=[0.05, 0.1, 0.2])
    parser.add_argument('--save-npz', type=str,   default='')
    parser.add_argument('--no-plot',  action='store_true')
    args = parser.parse_args()

    results = run(
        I=args.zones, order=args.order, N=args.N, n_groups=args.groups,
        tfinal=args.tfinal, dt_min=args.dt_min, dt_max=args.dt_max,
        K=args.K, maxits=args.maxits, LOUD=args.loud, fix=args.fix,
        output_times=np.array(args.output_times),
    )

    npz_name = args.save_npz or f'gaussian_ic_sn_{args.groups}g.npz'
    save_npz(results, npz_name)

    if not args.no_plot:
        plot_results(results, savefile='gaussian_ic_sn.png')
