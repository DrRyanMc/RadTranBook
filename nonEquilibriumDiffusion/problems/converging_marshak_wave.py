#!/usr/bin/env python3
"""
Converging Marshak Wave Test Problem - Spherical Geometry (Test 1)

Spherically convergent radiative heat wave in a uniform gold-like material.

Physical setup:
  Domain:        r ∈ [0, R],  R = 0.001 cm = 10 μm
  Geometry:      Spherical (d = 2)
  Density:       ρ = 19.3 g/cm³
  Opacity:       k_t = 7200 (T/HeV)^{-1.5} (ρ/(g/cc))^{1.2}  cm⁻¹
  Energy density: u = 3.4×10¹³ (T/HeV)^{1.6} (ρ/(g/cc))^{0.86}  erg/cm³
                     = 3.4×10⁻³ (T/HeV)^{1.6} (ρ/(g/cc))^{0.86}  GJ/cm³

Time convention:
  Physical time runs from t_init ≈ -29.626 ns to t_final = -1 ns.
  (The self-similar wave singularity is at t = 0; the problem runs at t < 0.)
  Solver elapsed time τ = t_phys - t_init  runs from 0 to ~28.626 ns.

Boundary conditions:
  r = 0:       Reflecting (symmetry axis)
  r = R:       Time-dependent Dirichlet  T_s(t) = 1.34503465 (-t/ns)^0.0920519
               × W^(5/8)(ξ_R(t)),  where ξ_R(t) = 10/(-t/ns)^0.679501

Self-similar solution (T in HeV):
  T(r,t) = 1.34503465 (-t/ns)^0.0920519 W^(5/8)(ξ(r,t))
  ξ(r,t) = (r/(0.1R)) / (-t/ns)^0.679501

  W(ξ) = (ξ-1)^0.40574 (1.52093 - 0.376185 ξ + 0.0655796 ξ²),   1 ≤ ξ ≤ 2
          (ξ-1)^0.295477 (1.08165 - 0.0271785 ξ + 0.00105539 ξ²), ξ > 2
          0,                                                         ξ ≤ 1

Units note:
  Temperatures in the solver are in keV.  1 HeV = 0.1 keV  (T_HeV = 10 T_keV).
  Material energies in GJ/cm³ (solver convention).
  Plots compare against reference in HeV and 10¹³ erg/cm³ to match test1.py.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from oneDFV import NonEquilibriumRadiationDiffusionSolver

# Add utils to path for plotting
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'utils'))
from plotfuncs import show

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

C_LIGHT = 2.998e1   # cm/ns
A_RAD   = 0.01372   # GJ/(cm³·keV⁴)

# =============================================================================
# PROBLEM PARAMETERS
# =============================================================================

RHO = 19.3          # g/cm³
R   = 0.001         # cm  (outer radius = 10 μm)

# Unit conversion:  1 keV = 10 HeV  →  T_HeV = 10 * T_keV
T_HEV_PER_KEV = 10.0

# Self-similar solution parameters (from the paper)
DELTA          = 0.6795011543956738
T_ANALYTIC_AMP = 1.34503465           # in HeV
T_ANALYTIC_EXP = 0.0920519
XI_WAVE_EXP    = DELTA                # = 0.679501 from the paper

# Physical time span
T_INIT_NS  = -(10.0 ** (1.0 / DELTA))  # ≈ -29.625647 ns  (W(ξ_R) = 0 here)
T_FINAL_NS = -1.0                       # ns

# Output times selected to match test1.py reference
OUTPUT_TIMES_NS = [-22.1223094, -9.44842435, -1.0]

# Precomputed density factors
_RHO_OPACITY_FACTOR = RHO ** 1.2
_RHO_ENERGY_FACTOR  = RHO ** 0.86
_U0_GJ = 3.4e-3 * _RHO_ENERGY_FACTOR   # coefficient for u(T) in GJ/cm³

# =============================================================================
# MATERIAL PROPERTIES  (all T arguments in keV)
# =============================================================================

def test1_opacity(T_keV):
    """
    Rosseland/Planck opacity from Test 1 paper formula  (equal here).

    k_t(T, ρ) = 7200 (T/HeV)^{-1.5} (ρ/(g/cm³))^{1.2}  cm⁻¹

    T argument is in keV;  T_HeV = 10 * T_keV.
    """
    T_HeV = T_keV * T_HEV_PER_KEV
    T_safe = np.maximum(T_HeV, 1e-6)
    return 7200.0 * T_safe**(-1.5) * _RHO_OPACITY_FACTOR


def test1_material_energy(T_keV):
    """
    Material energy density (GJ/cm³).

    u(T, ρ) = 3.4×10¹³ (T/HeV)^{1.6} (ρ/(g/cm³))^{0.86}  erg/cm³
             = 3.4×10⁻³ (T/HeV)^{1.6} (ρ/(g/cm³))^{0.86}  GJ/cm³

    T argument in keV;  T_HeV = 10 * T_keV.
    """
    T_HeV = T_keV * T_HEV_PER_KEV
    return _U0_GJ * T_HeV**1.6


def test1_inverse_material_energy(u_GJ):
    """
    Inverse of material energy: T in keV given u in GJ/cm³.

    u = _U0_GJ * T_HeV^{1.6}  →  T_HeV = (u / _U0_GJ)^{1/1.6} = (u / _U0_GJ)^{0.625}
    T_keV = T_HeV / 10
    """
    T_HeV = (np.maximum(u_GJ, 0.0) / _U0_GJ) ** 0.625
    return T_HeV / T_HEV_PER_KEV


def test1_specific_heat(T_keV):
    """
    Specific heat per unit mass  cv = (1/ρ) du/dT  in GJ/(g·keV).

    du/dT = _U0_GJ * 1.6 * T_HeV^{0.6} * (dT_HeV/dT_keV)
           = _U0_GJ * 1.6 * (10 T_keV)^{0.6} * 10
    """
    T_HeV = T_keV * T_HEV_PER_KEV
    T_safe = np.maximum(T_HeV, 1e-6)
    cv_volumetric = _U0_GJ * 1.6 * T_safe**0.6 * T_HEV_PER_KEV  # GJ/(cm³·keV)
    return cv_volumetric / RHO                                     # GJ/(g·keV)


# =============================================================================
# SELF-SIMILAR SOLUTION
# =============================================================================

def _Wxsi_scalar(xi):
    """W(ξ) similarity profile (scalar)."""
    if xi >= 2.0:
        return (xi - 1.0)**0.295477 * (1.08165 - 0.0271785*xi + 0.00105539*xi**2)
    elif xi > 1.0:
        return (xi - 1.0)**0.40574  * (1.52093 - 0.376185 *xi + 0.0655796 *xi**2)
    else:
        return 0.0

Wxsi = np.vectorize(_Wxsi_scalar)


def xsi_rt(r, t_ns):
    """Similarity coordinate ξ(r, t)."""
    return (r / (R / 10.0)) / (-t_ns) ** DELTA


def T_analytic_HeV(r, t_ns):
    """
    Analytic temperature T(r, t) in HeV.

    T = 1.34503465 (-t/ns)^0.0920519 W^{5/8}(ξ)
    """
    xi = xsi_rt(r, t_ns)
    return T_ANALYTIC_AMP * (-t_ns)**T_ANALYTIC_EXP * Wxsi(xi)**0.625


T_analytic_HeV_vec = np.vectorize(T_analytic_HeV)


def T_analytic_keV(r, t_ns):
    """Analytic temperature in keV."""
    return T_analytic_HeV(r, t_ns) / T_HEV_PER_KEV


def u_analytic_erg_per_1e13(r, t_ns):
    """
    Analytic energy density in units of 10¹³ erg/cm³  (matches test1.py axis).

    urt = 1e-13 * 3.4e13 * T_HeV^{1.6} * rho^{0.86}
        = 3.4 * T_HeV^{1.6} * rho^{0.86}

    (omega=0, mu=0.14 → (rho * r^{-omega})^{1-mu} = rho^{0.86})
    """
    T_HeV = T_analytic_HeV_vec(r, t_ns)
    return 3.4 * T_HeV**1.6 * _RHO_ENERGY_FACTOR


# =============================================================================
# TIME-DEPENDENT BOUNDARY CONDITION
# (mutable container so the closure can access the current physical time)
# =============================================================================

_t_phys_ns_state = [T_INIT_NS]   # updated before each solver step


def _surface_T_keV(t_ns):
    """Surface temperature in keV from the self-similar solution."""
    T_HeV = T_analytic_HeV(R, t_ns)
    return max(T_HeV / T_HEV_PER_KEV, 1e-8)


def left_bc(phi, x):
    """r = 0: reflecting symmetry — zero flux."""
    return 0.0, 1.0, 0.0   # 0·φ + 1·(dφ/dr) = 0


def right_bc(phi, x):
    """r = R: time-dependent Dirichlet at T_s(t)."""
    T_s = _surface_T_keV(_t_phys_ns_state[0])
    phi_bc = C_LIGHT * A_RAD * T_s**4
    return 1.0, 0.0, phi_bc   # 1·φ + 0·(dφ/dr) = phi_bc


# =============================================================================
# PLOTTING
# =============================================================================

def plot_results(solutions, save_prefix='converging_marshak_wave'):
    """
    Produce two PDFs matching the layout of test1.py:
      1) T profiles (HeV) at three output times + analytic
      2) u profiles (10¹³ erg/cm³) at three output times + analytic
    """
    colors = ['tab:blue', 'tab:orange', 'tab:red']
    r_anal = np.linspace(0.0, R, 2000)

    # --- Temperature plot ---
    fig, ax = plt.subplots(figsize=(7, 5))
    for sol, col in zip(solutions, colors):
        t_ns = sol['t_ns']
        r_cm = sol['r_cm']
        T_HeV = sol['T_keV'] * T_HEV_PER_KEV
        t_label = f't = {t_ns:.2f} ns'
        ax.plot(r_cm / 1e-4, T_HeV/10, color=col, lw=2, label=t_label)
        T_ref = T_analytic_HeV_vec(r_anal, t_ns)
        ax.plot(r_anal / 1e-4, T_ref/10, color=col, lw=1.5, ls='--',
                label=f'analytic ({t_label})')
    ax.set_xlabel(r'$r$ ($\mu$m)')
    ax.set_ylabel(r'$T$ (keV)')
    ticks = np.linspace(0, R / 1e-4, 11)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{t:g}' for t in ticks])
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    outfile = f'{save_prefix}_T.pdf'
    show(outfile, close_after=True)
    print(f'Saved: {outfile}')

    # --- Energy density plot ---
    fig, ax = plt.subplots(figsize=(7, 5))
    for sol, col in zip(solutions, colors):
        t_ns = sol['t_ns']
        r_cm = sol['r_cm']
        # convert solver GJ/cm³ → 10¹³ erg/cm³:  1 GJ = 1e16 erg, so × 1e3
        u_sim = test1_material_energy(sol['T_keV']) * 1e3
        t_label = f't = {t_ns:.2f} ns'
        ax.plot(r_cm / 1e-4, u_sim/1e3, color=col, lw=2, label=t_label)
        u_ref = u_analytic_erg_per_1e13(r_anal, t_ns)
        ax.plot(r_anal / 1e-4, u_ref/1e3, color=col, lw=1.5, ls='--',
                label=f'analytic ({t_label})')
    ax.set_xlabel(r'$r$ ($\mu$m]')
    ax.set_ylabel(r'$e(T)$ (GJ/cm$^3$)')
    ticks = np.linspace(0, R / 1e-4, 11)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{t:g}' for t in ticks])
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    outfile = f'{save_prefix}_u.pdf'
    show(outfile, close_after=True)
    print(f'Saved: {outfile}')


# =============================================================================
# MAIN SIMULATION
# =============================================================================

def run(n_cells=200, dt_initial=0.0001, dt_max=0.5, dt_growth=1.05,
        save_prefix='converging_marshak_wave', 
        t_duration = T_FINAL_NS - T_INIT_NS): # ≈ 28.626 ns  (positive)
    """
    Run the converging Marshak wave problem.

    Parameters
    ----------
    n_cells : int
        Number of radial cells in [0, R].
    dt_initial : float
        Initial time step in ns.
    dt_max : float
        Maximum time step in ns.
    dt_growth : float
        Multiplicative growth factor applied after each step.
    save_prefix : str
        Prefix for output PDF filenames.
    """
 

    print("=" * 70)
    print("Converging Marshak Wave  (Test 1 — Spherical, Gold-like)")
    print("=" * 70)
    print(f"  Domain:      r ∈ [0, {R*1e4:.0f} μm]")
    print(f"  Cells:       {n_cells}")
    print(f"  rho:         {RHO} g/cm³")
    print(f"  t_init:      {T_INIT_NS:.6f} ns")
    print(f"  t_final:     {T_FINAL_NS} ns")
    print(f"  Duration:    {t_duration:.6f} ns")
    print(f"  dt_initial:  {dt_initial} ns,  dt_max: {dt_max} ns")
    print("=" * 70)

    # Initial temperature: effectively cold (T_s(t_init) = 0 analytically)
    T_init_keV = 1e-3   # 0.001 HeV — very cold initial material

    # Build solver
    solver = NonEquilibriumRadiationDiffusionSolver(
        r_min=0.0,
        r_max=R,
        n_cells=n_cells,
        d=2,                           # spherical geometry
        dt=dt_initial,
        max_newton_iter=30,
        newton_tol=1e-8,
        rosseland_opacity_func=test1_opacity,
        planck_opacity_func=test1_opacity,
        specific_heat_func=test1_specific_heat,
        material_energy_func=test1_material_energy,
        inverse_material_energy_func=test1_inverse_material_energy,
        left_bc_func=left_bc,
        right_bc_func=right_bc,
        theta=1.0,
    )

    # Uniform cold initial condition
    phi_init = C_LIGHT * A_RAD * T_init_keV**4
    solver.phi = np.full(n_cells, phi_init)
    solver.T   = np.full(n_cells, T_init_keV)

    # Sorted target elapsed times (τ = t_phys - T_INIT_NS)
    output_elapsed = sorted(t - T_INIT_NS for t in OUTPUT_TIMES_NS)
    output_t_ns    = sorted(OUTPUT_TIMES_NS)
    output_saved   = set()

    solutions = []
    phys_elapsed = 0.0
    step = 0

    while phys_elapsed < t_duration - 1e-12:
        step += 1

        # Clip step to not overshoot t_duration or next output time
        dt_step = solver.dt
        for tau_out in output_elapsed:
            if tau_out > phys_elapsed and phys_elapsed + dt_step > tau_out:
                dt_step = tau_out - phys_elapsed
                break
        dt_step = min(dt_step, t_duration - phys_elapsed)
        solver.dt = dt_step

        # Set target physical time for the implicit BC (end-of-step)
        t_end_ns = T_INIT_NS + phys_elapsed + dt_step
        _t_phys_ns_state[0] = t_end_ns

        solver.time_step(n_steps=1, verbose=False)
        phys_elapsed += dt_step

        if step % 50 == 0 or step <= 3:
            t_now = T_INIT_NS + phys_elapsed
            T_surf_HeV = T_analytic_HeV(R, t_now)
            print(f"  step {step:5d}  t_phys = {t_now:8.4f} ns  "
                  f"T_surf = {T_surf_HeV:.4f} HeV  "
                  f"T_center = {solver.T[0]*T_HEV_PER_KEV:.4f} HeV")

        # Save snapshot at output times
        t_now = T_INIT_NS + phys_elapsed
        for t_out in output_t_ns:
            if t_out not in output_saved and abs(t_now - t_out) < 1e-9:
                solutions.append({
                    't_ns':  t_out,
                    'r_cm':  solver.r_centers.copy(),
                    'T_keV': solver.T.copy(),
                })
                output_saved.add(t_out)
                print(f"  >> Snapshot saved at t = {t_out:.4f} ns  "
                      f"(T_max = {solver.T.max()*T_HEV_PER_KEV:.4f} HeV)")

        # Grow time step for next iteration
        solver.dt = min(solver.dt * dt_growth, dt_max)

    print(f"\nTotal steps: {step}")
    print(f"Snapshots saved: {len(solutions)}")

    plot_results(solutions, save_prefix=save_prefix)
    return solver, solutions


if __name__ == "__main__":
    solver, solutions = run(
        n_cells=100,
        dt_initial=0.0001,
        dt_max=0.01,
        dt_growth=1.05,
        t_duration=OUTPUT_TIMES_NS[0]-T_INIT_NS,
    )
