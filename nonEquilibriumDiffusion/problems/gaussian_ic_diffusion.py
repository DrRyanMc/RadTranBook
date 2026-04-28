#!/usr/bin/env python3
"""
Gaussian IC Equilibrium Test — Multigroup Diffusion

Starts from an equilibrium Gaussian temperature profile to isolate
transport vs. diffusion physics, independently of boundary condition
treatment.  Both walls are reflecting (Neumann, zero flux) so there is
no incoming radiation.  The wave front does not reach the walls during
the short run, making the result insensitive to BC details.

Problem setup:
  - T(x,0) = max(T_floor, T_max · exp(-(x-x₀)²/(2σ²)))
    T_max = 0.2 keV,  x₀ = 3.5 cm,  σ = 0.8 cm,  T_floor = 0.001 keV
  - Radiation in equilibrium:  φ_g = c · a · χ_g(T) · T^4
  - Reflecting walls:  (A=0, B=1, C=0)  ↔  ∂φ/∂n = 0
  - Opacity:   σ_a(T,E) = 10 ρ T^{-1/2} E^{-3}  (cm⁻¹)
  - ρ = 0.01 g/cm³,  c_v = 0.05 GJ/(g·keV)
  - Groups: G=10, [1e-4, 10] keV log-spaced
  - Output: 0.05, 0.1, 0.2 ns

Run from the nonEquilibriumDiffusion directory:
    python problems/gaussian_ic_diffusion.py
    python problems/gaussian_ic_diffusion.py --groups 20 --tfinal 0.3
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from multigroup_diffusion_solver import (
    MultigroupDiffusionSolver1D, C_LIGHT, A_RAD, Bg_multigroup, flux_limiter_larsen
)

# Add project root to path for shared utilities
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
try:
    from utils.plotfuncs import show, hide_spines, font
    HAS_PLOTFUNCS = True
except ImportError:
    HAS_PLOTFUNCS = False

import argparse

MIN_DT_ADJUST = 1e-10  # ns; floor for dt clamping

# ── Physical constants ───────────────────────────────────────────────────
RHO     = 0.01   # g/cm³
CV_MASS = 0.05   # GJ/(g·keV)

# ── Gaussian IC parameters ───────────────────────────────────────────────
T_MAX   = 0.2    # keV   — peak temperature
X_0     = 3.5   # cm    — centre of Gaussian
SIGMA_G = 0.8   # cm    — width
T_FLOOR = 0.001  # keV   — cold background


def gaussian_T(x):
    """Cell-centre temperature profile at t=0."""
    return np.maximum(T_FLOOR,
                      T_MAX * np.exp(-0.5 * ((x - X_0) / SIGMA_G)**2))


# ── Opacity / diffusion factories ───────────────────────────────────────

def powerlaw_opacity_at_energy(T, E, rho=RHO):
    T_safe = np.maximum(T, 1e-2)
    return np.minimum(10.0 * rho * T_safe**(-0.5) * E**(-3.0), 1e14)


def make_powerlaw_opacity_func(E_low, E_high, rho=RHO):
    def opacity_func(T, r):
        return np.sqrt(powerlaw_opacity_at_energy(T, E_low, rho)
                       * powerlaw_opacity_at_energy(T, E_high, rho))
    return opacity_func


def make_powerlaw_diffusion_func(E_low, E_high, rho=RHO):
    opacity_func = make_powerlaw_opacity_func(E_low, E_high, rho)
    def diffusion_func(T, r):
        return C_LIGHT / (3.0 * opacity_func(T, r))
    return diffusion_func


# ── Main simulation ──────────────────────────────────────────────────────

def run_gaussian_ic_diffusion(n_groups=10, dt=0.001,
                               target_times=None, n_cells=140):
    """Run multigroup diffusion Gaussian IC equilibrium test."""

    if target_times is None:
        target_times = [0.05, 0.1, 0.2]
    target_times = sorted(target_times)

    print("=" * 70)
    print(f"Gaussian IC Equilibrium Test — Multigroup Diffusion ({n_groups} groups)")
    print("=" * 70)

    r_min = 0.0
    r_max = 7.0
    energy_edges = np.logspace(np.log10(1e-4), np.log10(10.0), n_groups + 1)

    sigma_funcs = []
    diff_funcs  = []
    for g in range(n_groups):
        El, Eh = energy_edges[g], energy_edges[g + 1]
        sigma_funcs.append(make_powerlaw_opacity_func(El, Eh))
        diff_funcs.append(make_powerlaw_diffusion_func(El, Eh))

    print(f"  T_max={T_MAX} keV  x₀={X_0} cm  σ={SIGMA_G} cm  T_floor={T_FLOOR} keV")
    print(f"  Cells={n_cells},  dt={dt} ns,  targets={target_times}")
    print(f"  ρ={RHO} g/cm³,  c_v={CV_MASS} GJ/(g·keV)")
    print(f"  BCs: reflecting (both walls) — (A=0, B=1, C=0) ∀g")
    print()

    # ── Create solver (None → default reflecting BCs) ──────────────────
    solver = MultigroupDiffusionSolver1D(
        n_groups=n_groups,
        r_min=r_min,
        r_max=r_max,
        n_cells=n_cells,
        energy_edges=energy_edges,
        geometry='planar',
        dt=dt,
        diffusion_coeff_funcs=diff_funcs,
        absorption_coeff_funcs=sigma_funcs,
        left_bc_funcs=None,   # default: reflecting
        right_bc_funcs=None,  # default: reflecting
        flux_limiter_funcs=flux_limiter_larsen,
        rho=RHO,
        cv=CV_MASS,
    )

    r_centers = solver.r_centers

    # ── Initial condition: equilibrium Gaussian ─────────────────────────
    T_init = gaussian_T(r_centers)
    solver.T     = T_init.copy()
    solver.T_old = T_init.copy()

    # Total radiation energy density: E_r = a T^4
    solver.E_r     = A_RAD * T_init**4
    solver.E_r_old = solver.E_r.copy()

    # Zero absorption rate density (equilibrium: κ = 0 initially)
    solver.kappa     = np.zeros(n_cells)
    solver.kappa_old = np.zeros(n_cells)

    # Per-group fractions from Planck spectrum at each cell
    # phi_g ≈ c · a · χ_g(T) · T^4  (per-steradian Bg_multigroup, × c gives flux)
    phi_total = C_LIGHT * solver.E_r   # c · a · T^4  [GJ/(cm²·ns)]
    for i in range(n_cells):
        bg    = Bg_multigroup(energy_edges, T_init[i])   # per-steradian values
        bg_s  = bg.sum()
        if bg_s > 0:
            chi_i = bg / bg_s
        else:
            chi_i = np.ones(n_groups) / n_groups
        for g in range(n_groups):
            solver.phi_g_fraction[g, i] = chi_i[g]
            solver.phi_g_stored[g, i]   = chi_i[g] * phi_total[i]

    # Update absorption coefficients and emission fractions at T_init
    solver.update_absorption_coefficients(T_init)
    solver.update_emission_fractions(T_init, verbose=True)

    solver.t = 0.0
    current_time = 0.0

    print(f"Initial conditions:")
    print(f"  T_init range: [{T_init.min():.5f}, {T_init.max():.5f}] keV")
    print(f"  E_r range   : [{solver.E_r.min():.3e}, {solver.E_r.max():.3e}] GJ/cm³")

    # ── Time-stepping ───────────────────────────────────────────────────
    print(f"\n{'Step':<6} {'t (ns)':<10} {'T_max':<12} {'T_min':<12} "
          f"{'E_r_max':<15} {'Newton':<8} {'GMRES':<8}")
    print("-" * 75)

    solutions  = []
    step_count = 0
    target_idx = 0
    max_steps  = 5000
    dt_nominal = dt

    while (current_time < target_times[-1] + 1e-6 * dt_nominal) and step_count < max_steps:
        # Clamp dt to land exactly on next output time
        dt_saved = solver.dt
        if target_idx < len(target_times):
            dt_to_target = target_times[target_idx] - solver.t
            if 0 < dt_to_target < solver.dt - MIN_DT_ADJUST:
                solver.dt = dt_to_target
                print(f"\n--- Adjusting dt to {solver.dt:.4e} ns "
                      f"for target t = {target_times[target_idx]:.3f} ns ---")

        # Newton step
        info = solver.step(
            max_newton_iter=15,
            newton_tol=1e-6,
            gmres_tol=1e-10,
            gmres_maxiter=400,
            use_preconditioner=False,
            max_relative_change=2.0,
            verbose=False,
        )

        step_count  += 1
        current_time = solver.t

        if step_count % 10 == 0:
            gmres_it = info['gmres_info']['iterations']
            print(f"{step_count:<6} {current_time:<10.4f} {solver.T.max():<12.6f} "
                  f"{solver.T.min():<12.6f} {solver.E_r.max():<15.3e} "
                  f"{info['newton_iter']:<8} {gmres_it:<8}")

        # Save at target times
        if target_idx < len(target_times):
            if np.abs(current_time - target_times[target_idx]) < 0.5 * dt_nominal:
                T   = solver.T.copy()
                E_r = solver.E_r.copy()
                T_rad = (E_r / A_RAD)**0.25

                # Recover per-group fluxes from stored fractions
                phi_total_now = C_LIGHT * E_r
                phi_groups = np.zeros((n_groups, n_cells))
                for g in range(n_groups):
                    phi_groups[g, :] = solver.phi_g_fraction[g, :] * phi_total_now
                E_r_groups = phi_groups / C_LIGHT

                solutions.append({
                    'time':       current_time,
                    'T':          T,
                    'E_r':        E_r,
                    'T_rad':      T_rad,
                    'phi_groups': phi_groups,
                    'E_r_groups': E_r_groups,
                })
                print(f"\n>>> t = {current_time:.4f} ns  |  T_max = {T.max():.6f}  "
                      f"E_r_max = {E_r.max():.3e}")
                target_idx += 1

        # Restore nominal dt then advance
        solver.dt = dt_saved
        solver.advance_time()

    print(f"\nSimulation complete.  Steps: {step_count},  t_final: {current_time:.4f} ns")
    return {'solutions': solutions, 'r': r_centers, 'energy_edges': energy_edges,
            'n_groups': n_groups, 'n_cells': n_cells}


def save_npz(results, npz_name):
    """Save results to compressed NPZ (same format as S_N Gaussian IC file)."""
    sols   = results['solutions']
    r      = results['r']
    edges  = results['energy_edges']
    G      = results['n_groups']
    times  = np.array([s['time'] for s in sols])
    T_mat  = np.array([s['T']          for s in sols])
    T_rad  = np.array([s['T_rad']      for s in sols])
    E_r    = np.array([s['E_r']        for s in sols])
    phi_g  = np.array([s['phi_groups'] for s in sols])
    Er_g   = np.array([s['E_r_groups'] for s in sols])

    np.savez_compressed(
        npz_name,
        times=times, r=r, energy_edges=edges,
        T_mat=T_mat, T_rad=T_rad, E_r=E_r,
        phi_groups=phi_g, E_r_groups=Er_g,
    )
    print(f"Results saved to {npz_name}")


def plot_results(results, savefile=''):
    sols   = results['solutions']
    r      = results['r']
    colors = ['C0', 'C1', 'C2', 'C3']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for idx, sol in enumerate(sols):
        col = colors[idx % len(colors)]
        ax1.plot(r, sol['T'],     '-',  color=col, lw=1.5, label=f'mat  t={sol["time"]:.2f} ns')
        ax1.plot(r, sol['T_rad'], '--', color=col, lw=1.0, label=f'rad  t={sol["time"]:.2f} ns')
        ax2.semilogy(r, sol['T'],     '-',  color=col, lw=1.5)
        ax2.semilogy(r, sol['T_rad'], '--', color=col, lw=1.0)

    for ax in (ax1, ax2):
        ax.set_xlabel('x (cm)')
        ax.set_ylabel('Temperature (keV)')
        ax.set_xlim(results['r'][0], results['r'][-1])
        ax.grid(True, alpha=0.3)
    ax1.legend(fontsize=7, ncol=2)
    ax1.set_title('Gaussian IC — Multigroup Diffusion')
    ax2.set_title('Log scale')
    plt.tight_layout()
    if savefile:
        plt.savefig(savefile, dpi=150, bbox_inches='tight')
    if HAS_PLOTFUNCS:
        show('gaussian_ic_diffusion.pdf', close_after=True)
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Gaussian IC equilibrium test — Multigroup Diffusion')
    parser.add_argument('--groups',   type=int,   default=10)
    parser.add_argument('--dt',       type=float, default=0.001)
    parser.add_argument('--cells',    type=int,   default=140)
    parser.add_argument('--tfinal',   type=float, default=0.2)
    parser.add_argument('--output-times', type=float, nargs='+',
                        default=[0.05, 0.1, 0.2])
    parser.add_argument('--save-npz', type=str, default='')
    parser.add_argument('--no-plot',  action='store_true')
    args = parser.parse_args()

    times_out = [t for t in args.output_times if t <= args.tfinal + 1e-9]

    results = run_gaussian_ic_diffusion(
        n_groups=args.groups,
        dt=args.dt,
        target_times=times_out,
        n_cells=args.cells,
    )

    npz_name = args.save_npz or f'gaussian_ic_diffusion_{args.groups}g.npz'
    save_npz(results, npz_name)

    if not args.no_plot:
        plot_results(results, savefile='gaussian_ic_diffusion.png')
