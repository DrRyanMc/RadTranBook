#!/usr/bin/env python3
"""
ray_effects_spherical.py — Ray Effects in Time: Spherical Two-Material Problem

Demonstrates the difference between IMC (transport) and flux-limited diffusion
(FLD, Levermore–Pomraning limiter) at early times, when radiation streams inward
from a hot thick outer shell into a cold thin inner region.

Material layout
---------------
  Inner region  r ∈ [0, 4] cm   ρ = 0.1 g/cc  σ_a = 10   cm⁻¹   Cv = 0.01 GJ/(cc·keV)
  Outer shell   r ∈ [4, 5] cm   ρ = 10  g/cc  σ_a = 1000 cm⁻¹   Cv = 1.0  GJ/(cc·keV)

  σ_a = 100·ρ cm⁻¹,  Cv_vol = 0.1·ρ GJ/(cc·keV)

Initial conditions
------------------
  T_inner = 0.001 keV (cold),  T_outer = 1.0 keV (hot)
  Radiation starts in equilibrium with the material in each region.

Boundary conditions
-------------------
  r = 0:   reflecting (spherical symmetry)
  r = 5 cm: cold Dirichlet (vacuum approximation)

Physics to observe
------------------
At early times radiation leaks from the hot thick outer shell inward into the cold
thin inner region.  The inward "light cone" front is

    r_front(t) = 4 - c·t

IMC (transport) correctly captures this causality: no signal can reach r < r_front.
Flux-limited diffusion (LP limiter) allows the wave to run ahead of c/√3, placing
the inward front at r = 4 - c/√3·t — the front penetrates faster than transport.

The "ray" physics in the inner region is anisotropic: photons are inward-directed
beams from the thick shell's inner surface.  Diffusion, which assumes an isotropic
radiation field, inherently cannot represent this beam.

Usage
-----
  python ray_effects_spherical.py                        # both, default params
  python ray_effects_spherical.py --imc-only             # IMC only
  python ray_effects_spherical.py --fld-only             # FLD only
  python ray_effects_spherical.py --dt 0.005 --Ntarget 50000
  python ray_effects_spherical.py --output-times 0.01,0.05,0.10,0.20
  python ray_effects_spherical.py --save-fig ray_effects.png
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ── Path setup ─────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, 'nonEquilibriumDiffusion'))
sys.path.insert(0, os.path.join(_ROOT, 'DiscreteOrdinates'))

import IMC1D as imc
from oneDFV import (NonEquilibriumRadiationDiffusionSolver,
                    flux_limiter_levermore_pomraning)
from sn_solver_ld_sphere import temp_solve_sph_ld
from sn_solver import ac as AC_SN   # a·c  [GJ/(cm³·keV⁴) · cm/ns]

# ── Physical constants ──────────────────────────────────────────────────────────
C_LIGHT = 29.98    # cm/ns
A_RAD   = 0.01372  # GJ/(cm³·keV⁴)

# ── Problem geometry ────────────────────────────────────────────────────────────
R_INTERFACE = 2.0   # cm  (inner thin / outer thick boundary)
R_MAX       = 3.0   # cm  (outer domain boundary)
I_INNER     = 40    # cells in inner thin region [0, 2]
I_OUTER     = 10    # cells in outer thick shell [2, 3]
I_TOTAL     = I_INNER + I_OUTER   # 50 cells, dr = 0.1 cm

# ── Material parameters ─────────────────────────────────────────────────────────
RHO_INNER = 10.0   # g/cc
RHO_OUTER = 0.1    # g/cc

SIGMA_COEF  = 100.0  # cm⁻¹ / (g/cc)  →  σ = SIGMA_COEF · ρ
CV_VOL_COEF = 0.1    # GJ/(cc·keV) / (g/cc)  →  Cv_vol = CV_VOL_COEF · ρ

SIGMA_INNER  = SIGMA_COEF  * RHO_INNER   # 1000 cm⁻¹
SIGMA_OUTER  = SIGMA_COEF  * RHO_OUTER   # 10   cm⁻¹
CV_INNER_VOL = CV_VOL_COEF * RHO_INNER   # 1.0  GJ/(cc·keV)
CV_OUTER_VOL = CV_VOL_COEF * RHO_OUTER   # 0.01 GJ/(cc·keV)

T_HOT  = 1.0     # keV  initial temperature of outer shell
T_COLD = 1.0e-3  # keV  initial temperature of inner region

# ── Shared uniform mesh, dr = 0.1 cm ───────────────────────────────────────────
R_FACES  = np.linspace(0.0, R_MAX, I_TOTAL + 1)   # 51 faces
R_CENTERS = 0.5 * (R_FACES[:-1] + R_FACES[1:])    # 50 cell centres

IS_INNER   = R_CENTERS < R_INTERFACE               # True for inner (thin) cells
SIGMA_CELL = np.where(IS_INNER, SIGMA_OUTER,  SIGMA_INNER)   # thin inside, thick outside
CV_CELL    = np.where(IS_INNER, CV_OUTER_VOL, CV_INNER_VOL)
T_INIT     = np.where(IS_INNER, T_COLD,       T_HOT)


# =============================================================================
# IMC MATERIAL FUNCTIONS  (cell-array interface)
# =============================================================================

# IMC mesh: (I_TOTAL, 2) array — each row is [r_inner, r_outer] of a cell
IMC_MESH = np.column_stack([R_FACES[:-1], R_FACES[1:]])


def imc_sigma_a(T):
    """Absorption opacity [cm⁻¹]; T-independent, returns array over cells."""
    return SIGMA_CELL.copy()


def imc_cv(T):
    """Volumetric heat capacity [GJ/(cc·keV)]; returns array over cells."""
    return CV_CELL.copy()


def imc_eos(T):
    """Material energy density [GJ/cc] = Cv_vol · T."""
    return CV_CELL * T


def imc_inv_eos(u):
    """Inverse EOS: T = u / Cv_vol."""
    return u / CV_CELL


# =============================================================================
# FLD MATERIAL FUNCTIONS  (pointwise interface: f(T, r) or f(T, r=None))
# =============================================================================

def fld_sigma(T, r=None):
    """Planck/Rosseland opacity σ [cm⁻¹]; T-independent, space-dependent."""
    if r is None:
        r = 0.0
    rho = RHO_INNER if r < R_INTERFACE else RHO_OUTER
    return SIGMA_COEF * rho


def fld_material_energy(T, r=None):
    """Volumetric material energy density e = Cv_vol · T [GJ/cc]."""
    if r is None:
        r = 0.0
    cv = CV_INNER_VOL if r < R_INTERFACE else CV_OUTER_VOL
    return cv * T


def fld_inv_material_energy(e, r=None):
    """Inverse energy: T = e / Cv_vol."""
    if r is None:
        r = 0.0
    cv = CV_INNER_VOL if r < R_INTERFACE else CV_OUTER_VOL
    return e / cv


def fld_cv(T, r=None):
    """Specific heat [GJ/(g·keV)].

    The FLD solver calls  e = ρ · cv · T  internally when
    material_energy_func is the default, but since we supply a custom
    material_energy_func (= Cv_vol · T), this function is only used via
    the Fleck-factor β = 4aT³/(ρ·cv).  We return Cv_vol directly here so
    that the implicit RHO=1 assumption in get_beta() produces the correct
    volumetric coupling."""
    if r is None:
        r = 0.0
    return CV_INNER_VOL if r < R_INTERFACE else CV_OUTER_VOL


# Boundary conditions ──────────────────────────────────────────────────────────
_PHI_COLD = A_RAD * C_LIGHT * T_COLD**4   # equilibrium φ at T_COLD


def fld_left_bc(phi, x):
    """Left BC (r = 0): reflecting — zero flux by zero-gradient Robin BC."""
    return 0.0, 1.0, 0.0   # A·φ + B·(dφ/dr) = C  →  dφ/dr = 0


def fld_right_bc(phi, x):
    """Right BC (r = R_MAX): cold Dirichlet at T_COLD (vacuum approximation)."""
    return 1.0, 0.0, _PHI_COLD   # φ = φ_cold


# =============================================================================
# S_N MATERIAL FUNCTIONS  (cell-array interface: f(T) → (I, 2))
# =============================================================================

def sn_sigma(T):
    """Absorption opacity (I,2); T-independent, cell-constant."""
    result = np.empty_like(T)
    result[:, 0] = SIGMA_CELL
    result[:, 1] = SIGMA_CELL
    return result


def sn_scat(T):
    """No scattering."""
    return np.zeros_like(T)


def sn_eos(T):
    """Material energy density e = Cv_vol · T [GJ/cc], shape (I,2)."""
    return CV_CELL[:, np.newaxis] * T


def sn_inv_eos(e):
    """Inverse EOS: T = e / Cv_vol, shape (I,2)."""
    return e / CV_CELL[:, np.newaxis]


# =============================================================================
# IMC RUNNER
# =============================================================================

def run_imc(output_times, dt=0.01, Ntarget=20_000, NMax=100_000, verbose=True):
    """Run the spherical IMC simulation; return list of snapshot dicts.

    Each snapshot: {'time': float, 'T': ndarray, 'Tr': ndarray}
    """
    print(f"  Ntarget={Ntarget}, NMax={NMax}, dt={dt} ns")
    print(f"  Mesh: {I_TOTAL} cells, dr={R_FACES[1]-R_FACES[0]:.2f} cm")

    # Initial radiation temperature = material temperature (thermal equilibrium IC)
    Tr_init = T_INIT.copy()

    state = imc.init_simulation(
        Ntarget, T_INIT.copy(), Tr_init, IMC_MESH,
        imc_eos, imc_inv_eos, geometry='spherical')

    snapshots    = []
    current_time = 0.0

    for t_target in sorted(output_times):
        while current_time < t_target - 1e-14:
            step_dt = min(dt, t_target - current_time)
            state, info = imc.step(
                state, Ntarget, 0, 0, NMax,
                (0.0, 0.0), step_dt, IMC_MESH,
                imc_sigma_a, imc_inv_eos, imc_cv,
                np.zeros(I_TOTAL),
                reflect=(True, False),
                geometry='spherical')
            current_time += step_dt
            if verbose:
                print(f"    t={current_time:.5f} ns  N={info['N_particles']:6d}  "
                      f"Eloss={info['energy_loss']:+.3e} GJ  "
                      f"events={info['profiling']['transport_events']['total']}")

        snapshots.append({
            'time': float(current_time),
            'T':    state.temperature.copy(),
            'Tr':   state.radiation_temperature.copy(),
        })
        print(f"  [IMC] t={current_time:.4f} ns → "
              f"T_max={state.temperature.max():.4f} keV  "
              f"Tr_max={state.radiation_temperature.max():.4f} keV")

    return snapshots


# =============================================================================
# FLD RUNNER
# =============================================================================

def run_fld(output_times, dt=0.01, verbose=False):
    """Run the FLD simulation; return (snapshots, r_centers).

    Each snapshot: {'time': float, 'T': ndarray, 'Tr': ndarray}
    """
    print(f"  dt={dt} ns, Levermore-Pomraning flux limiter, implicit Euler")

    solver = NonEquilibriumRadiationDiffusionSolver(
        r_min=0.0, r_max=R_MAX,
        n_cells=I_TOTAL,
        d=2,               # spherical (4πr²) geometry
        dt=dt,
        max_newton_iter=30,
        newton_tol=1e-8,
        rosseland_opacity_func=fld_sigma,
        planck_opacity_func=fld_sigma,
        specific_heat_func=fld_cv,
        material_energy_func=fld_material_energy,
        inverse_material_energy_func=fld_inv_material_energy,
        left_bc_func=fld_left_bc,
        right_bc_func=fld_right_bc,
        flux_limiter_func=flux_limiter_levermore_pomraning,
        theta=1.0,
    )

    # Set initial conditions: equilibrium (φ = acT⁴) in each region
    phi_init = A_RAD * C_LIGHT * T_INIT**4
    solver.phi     = phi_init.copy()
    solver.T       = T_INIT.copy()
    solver.phi_old = phi_init.copy()
    solver.T_old   = T_INIT.copy()

    snapshots    = []
    current_time = 0.0

    for t_target in sorted(output_times):
        while current_time < t_target - 1e-14:
            step_dt = min(dt, t_target - current_time)
            solver.dt = step_dt
            solver.time_step(n_steps=1, verbose=verbose)
            current_time += step_dt

        Er   = solver.phi / C_LIGHT
        T_rad = (np.maximum(Er, 0.0) / A_RAD) ** 0.25
        snapshots.append({
            'time': float(current_time),
            'T':    solver.T.copy(),
            'Tr':   T_rad,
        })
        print(f"  [FLD] t={current_time:.4f} ns → "
              f"T_max={solver.T.max():.4f} keV  "
              f"Tr_max={T_rad.max():.4f} keV")

    return snapshots, solver.r_centers.copy()


# =============================================================================
# S_N RUNNER
# =============================================================================

def run_sn(N, output_times, dt_max=0.01, dt_min=1.0e-4, maxits=200, K=30,
           verbose=False):
    """Run the spherical LD-S_N simulation with N ordinates; return snapshots.

    Each snapshot: {'time': float, 'T': ndarray(I,), 'Tr': ndarray(I,)}
    """
    print(f"  N={N} ordinates, dt_min={dt_min:.0e} ns, dt_max={dt_max:.3g} ns")

    r_left = R_FACES[:-1].copy()
    dr     = np.diff(R_FACES)

    # LD initial condition: both left/right edge dofs equal cell-constant value
    T_ic   = np.column_stack([T_INIT, T_INIT]).astype(np.float64)  # (I, 2)
    phi_ic = AC_SN * T_ic**4                                        # (I, 2)
    psi_ic = np.broadcast_to(phi_ic[:, None, :] / 2,
                             (I_TOTAL, N, 2)).copy()               # (I, N, 2)
    g_ic   = phi_ic / 2.0                                          # (I, 2)

    q_n = np.zeros((I_TOTAL, N, 2))
    q_g = np.zeros((I_TOTAL, 2))

    # Outer BC: vacuum (no incoming radiation from outside)
    _bc_zero = np.zeros((N, 2))
    def BCs(t):
        return _bc_zero, 0.0

    t_outs = np.array(sorted(output_times), dtype=float)
    tfinal = float(t_outs[-1])

    phis, Ts, gs, iterations, ts, _ = temp_solve_sph_ld(
        I_TOTAL, r_left, dr, q_n, q_g,
        sn_sigma, sn_scat, N, BCs, sn_eos, sn_inv_eos,
        phi_ic, psi_ic, T_ic, g_ic,
        dt_min=dt_min, dt_max=dt_max, tfinal=tfinal,
        maxits=maxits, K=K, R=3, LOUD=verbose,
        reflect_outer=False,
        time_outputs=t_outs,
    )

    ts_arr = np.array(ts)
    snapshots = []
    for t_target in sorted(output_times):
        idx     = int(np.argmin(np.abs(ts_arr - t_target)))
        T_arr   = Ts[idx]      # (I, 2)
        phi_arr = phis[idx]    # (I, 2)
        T_ctr   = 0.5 * (T_arr[:, 0]   + T_arr[:, 1])
        phi_ctr = 0.5 * (phi_arr[:, 0] + phi_arr[:, 1])
        Er  = phi_ctr / C_LIGHT
        Tr  = (np.maximum(Er, 0.0) / A_RAD) ** 0.25
        snapshots.append({'time': float(ts_arr[idx]), 'T': T_ctr, 'Tr': Tr})
        print(f"  [S_{N}] t={ts_arr[idx]:.4f} ns → "
              f"T_max={T_ctr.max():.4f} keV  Tr_max={Tr.max():.4f} keV")

    print(f"  Total transport sweeps: {iterations}")
    return snapshots


# =============================================================================
# PLOT
# =============================================================================

# S_N line styles: indexed by position in the list of orders being plotted
_SN_STYLES = [('-.', 2.0), (':', 2.4), ((0, (3, 1, 1, 1)), 1.8)]


def make_plot(imc_snaps, fld_snaps, fld_r, output_times,
             sn_results=None, savefile=None):
    """Two-panel plot: radiation temperature (left) and material temperature (right).

    Parameters
    ----------
    sn_results : dict {N: list_of_snapshots} or None
        Results from run_sn for one or more angular orders.
    """
    t_sorted = sorted(output_times)
    n_times  = len(t_sorted)
    colors   = plt.cm.plasma(np.linspace(0.15, 0.80, n_times))
    sn_orders = sorted(sn_results.keys()) if sn_results else []

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), sharey=False)

    panel_cfg = [
        ('Tr', 'Radiation temperature $T_r$ (keV)', 'upper right'),
        ('T',  'Material temperature $T$ (keV)',    'upper right'),
    ]

    for ax, (key, ylabel, leg_loc) in zip(axes, panel_cfg):
        for idx, t in enumerate(t_sorted):
            c = colors[idx]

            # S_N (plotted first so IMC/FLD overlay on top)
            for si, N in enumerate(sn_orders):
                ls, lw = _SN_STYLES[si % len(_SN_STYLES)]
                ax.plot(R_CENTERS, sn_results[N][idx][key], ls=ls,
                            color=c, lw=lw, alpha=0.80)

            # FLD
            if fld_snaps is not None:
                ax.plot(fld_r, fld_snaps[idx][key], '--',
                            color=c, lw=2.2, alpha=0.85)

            # IMC
            if imc_snaps is not None:
                ax.plot(R_CENTERS, imc_snaps[idx][key], '-',
                            color=c, lw=1.6, alpha=0.95)

        # Inward light-cone markers (thin dotted vertical lines)
        for idx, t in enumerate(t_sorted):
            r_lc = R_INTERFACE - C_LIGHT * t
            if 0.0 < r_lc < R_INTERFACE:
                ax.axvline(r_lc, color=colors[idx], lw=0.9, ls=':',
                           alpha=0.70)

        # Material interface
        ax.axvline(R_INTERFACE, color='k', lw=1.0, ls='--', alpha=0.35,
                   label=f'Interface $r={R_INTERFACE:.0f}$ cm')

        ax.set_xlabel('Radius $r$ (cm)', fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_xlim(0.0, R_MAX)
        ax.grid(True, which='both', alpha=0.25)
        ax.tick_params(labelsize=11)

    # ── Legend ─────────────────────────────────────────────────────────────────
    handles = []
    for idx, t in enumerate(t_sorted):
        handles.append(Line2D([0], [0], color=colors[idx], lw=2,
                               label=f'$t = {t:.3g}$ ns'))

    if imc_snaps is not None:
        handles.append(Line2D([0], [0], color='k', lw=2.0, ls='-',
                               label='IMC (transport)'))
    if fld_snaps is not None:
        handles.append(Line2D([0], [0], color='k', lw=2.2, ls='--',
                               label='FLD (LP limiter)'))
    for si, N in enumerate(sn_orders):
        ls, lw = _SN_STYLES[si % len(_SN_STYLES)]
        handles.append(Line2D([0], [0], color='k', lw=lw, ls=ls,
                               label=f'S$_{{{N}}}$ transport'))
    handles.append(Line2D([0], [0], color='0.5', lw=0.9, ls=':',
                           label='Inward light cone $r = 4 - ct$'))
    handles.append(Line2D([0], [0], color='k', lw=1.0, ls='--', alpha=0.5,
                           label=f'Interface $r = {R_INTERFACE:.0f}$ cm'))

    axes[0].legend(handles=handles, fontsize=9, loc='lower left', ncol=1,
                   framealpha=0.85)

    fig.suptitle(
        'Ray Effects in Time — Spherical Two-Material Problem\n'
        r'Inner ($r < 4$ cm): $\rho=0.1$ g/cc, $\sigma=10$ cm$^{-1}$, '
        r'$C_v=0.01$ GJ/cc/keV, $T_0=0.001$ keV   '
        r'Outer ($4 < r < 5$ cm): $\rho=10$ g/cc, $\sigma=1000$ cm$^{-1}$, '
        r'$C_v=1$ GJ/cc/keV, $T_0=1$ keV',
        fontsize=10)

    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])

    fname = savefile if savefile else 'ray_effects_spherical.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"Plot saved: {fname}")
    plt.close()


# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='Ray effects in time: spherical IMC vs FLD comparison.')
    p.add_argument('--dt', type=float, default=0.01,
                   help='Time step [ns] (default: 0.01)')
    p.add_argument('--Ntarget', type=int, default=20_000,
                   help='Target IMC particle count (default: 20000)')
    p.add_argument('--NMax', type=int, default=100_000,
                   help='Max IMC particle count after combing (default: 100000)')
    p.add_argument('--output-times', type=str, default='0.01,0.05,0.10',
                   help='Comma-separated output times in ns (default: 0.01,0.05,0.10)')
    p.add_argument('--imc-only', action='store_true',
                   help='Run IMC only (skip FLD)')
    p.add_argument('--fld-only', action='store_true',
                   help='Run FLD only (skip IMC)')
    p.add_argument('--verbose-imc', action='store_true',
                   help='Print per-step IMC diagnostics')
    p.add_argument('--save-fig', type=str, default='',
                   help='Filename for the output figure (default: ray_effects_spherical.png)')
    p.add_argument('--no-plot', action='store_true',
                   help='Skip generating the plot')
    p.add_argument('--no-sn', action='store_true',
                   help='Skip S_N transport runs')
    p.add_argument('--sn-orders', type=str, default='2,8',
                   help='Comma-separated S_N angular orders (default: 2,8)')
    p.add_argument('--dt-max-sn', type=float, default=0.0,
                   help='dt_max for S_N solver in ns (default: same as --dt)')
    p.add_argument('--maxits-sn', type=int, default=200,
                   help='Max source iterations per S_N time step (default: 200)')
    return p.parse_args()


if __name__ == '__main__':
    args   = parse_args()
    t_outs = [float(x) for x in args.output_times.split(',')]

    print()
    print('=' * 70)
    print('Ray Effects in Time — Spherical Two-Material Problem')
    print('=' * 70)
    print(f'  Inner thin region r ∈ [0, {R_INTERFACE}] cm:')
    print(f'    ρ = {RHO_OUTER} g/cc,  σ_a = {SIGMA_OUTER:.0f} cm⁻¹,  '
          f'Cv = {CV_OUTER_VOL} GJ/cc/keV,  T₀ = {T_COLD} keV')
    print(f'  Outer thick shell r ∈ [{R_INTERFACE}, {R_MAX}] cm:')
    print(f'    ρ = {RHO_INNER} g/cc,  σ_a = {SIGMA_INNER:.0f} cm⁻¹,  '
          f'Cv = {CV_INNER_VOL} GJ/cc/keV,  T₀ = {T_HOT} keV')
    print()
    print(f'  c = {C_LIGHT:.2f} cm/ns')
    print(f'  Output times: {t_outs} ns   dt = {args.dt} ns')
    print()
    print('  Inward light-cone fronts (r = 4 - c·t):')
    for t in t_outs:
        r_lc  = max(R_INTERFACE - C_LIGHT * t, 0.0)
        r_fld = max(R_INTERFACE - C_LIGHT / 3.0**0.5 * t, 0.0)
        print(f'    t={t:.3f} ns → IMC front ≥ r={r_lc:.3f} cm  '
              f'(FLD front ≈ r={r_fld:.3f} cm at c/√3)')
    print('=' * 70)
    print()

    imc_snaps = fld_snaps = fld_r = None
    sn_results = {}

    if not args.fld_only:
        print('Running IMC simulation ...')
        imc_snaps = run_imc(t_outs, dt=args.dt,
                            Ntarget=args.Ntarget, NMax=args.NMax,
                            verbose=args.verbose_imc)
        print()

    if not args.imc_only:
        print('Running FLD simulation ...')
        fld_snaps, fld_r = run_fld(t_outs, dt=args.dt)
        print()

    if not args.no_sn and not args.imc_only and not args.fld_only:
        sn_orders  = [int(x) for x in args.sn_orders.split(',')]
        dt_max_sn  = args.dt_max_sn if args.dt_max_sn > 0 else args.dt
        for N in sn_orders:
            print(f'Running S_{N} simulation ...')
            sn_results[N] = run_sn(N, t_outs, dt_max=dt_max_sn,
                                   maxits=args.maxits_sn)
            print()

    any_results = (imc_snaps is not None or fld_snaps is not None
                   or bool(sn_results))
    if not args.no_plot and any_results:
        print('Generating comparison plot ...')
        make_plot(imc_snaps, fld_snaps,
                  fld_r if fld_r is not None else R_CENTERS,
                  t_outs, sn_results=sn_results,
                  savefile=args.save_fig or None)
