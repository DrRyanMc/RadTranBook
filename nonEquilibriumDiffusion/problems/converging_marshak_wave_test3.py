#!/usr/bin/env python3
"""
Converging Marshak Wave Test Problem 3 - Spherical Geometry

Spherically convergent radiative heat wave in a material with a
power-law decreasing density profile.

Physical setup:
  Domain:        r ∈ [0, R],  R = 0.001 cm = 10 μm
  Geometry:      Spherical (d = 2)
  Density:       ρ(r) = (r/cm)^{-0.45}  g/cm³   (ω = 0.45)
  Opacity:       k_t = 10³ (T/HeV)^{-3.5} (ρ/(g/cm³))^{1.4}  cm⁻¹
  Energy density: u (T,ρ) = 10¹³ (T/HeV)²  (ρ/(g/cm³))^{0.75}  erg/cm³
                           = 10⁻³ (T/HeV)²  (ρ/(g/cm³))^{0.75}  GJ/cm³

Similarity parameters:
  m = 0.3375,  k = 0.63,  n = 2.75,  a = 1.6625,  b = -0.9675
  δ = 1.11575  (similarity exponent)

Time convention:
  Physical time runs from t_init ≈ -7.875084 ns to t_final = -1 ns.
  Solver elapsed time τ = t_phys - t_init  runs from 0 to ~6.875084 ns.

Boundary conditions:
  r = 0:       Reflecting (symmetry axis)
  r = R:       Time-dependent Dirichlet  T_s(t) = 1.1982 (-t/ns)^{0.0276392} W^{1/2}(ξ_R(t))
               where ξ_R(t) = 10 / (-t/ns)^{1.1157536}

Self-similar solution (T in HeV):
  T(r,t) = 1.1982 (-t/ns)^{0.0276392}  W^{1/2}(ξ(r,t))
  ξ(r,t) = (r / 10⁻⁴ cm) / (-t/ns)^{1.1157536}

  W(ξ) = (ξ-1)^{0.357506} (1.9792  - 0.619497 ξ + 0.110644  ξ²),  1 < ξ ≤ 2
          (ξ-1)^{0.210071} (1.27048 - 0.0470724 ξ + 0.00179721 ξ²),  ξ > 2
          0,                                                            ξ ≤ 1

Units note:
  Temperatures in the solver are in keV.  1 HeV = 0.1 keV.
  Material energies in GJ/cm³ (solver convention).
  Plots compare against reference in HeV and 10¹³ erg/cm³ to match test3.py.
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

R   = 0.001         # cm  (outer radius = 10 μm)

# Unit conversion:  1 keV = 10 HeV  →  T_HeV = 10 * T_keV
T_HEV_PER_KEV = 10.0

# Density profile:  ρ(r) = (r/cm)^{-ω}  g/cm³
OMEGA = 0.45

# Opacity exponents:  α=3.5, λ=1.4  →  k_t = 10³ T_HeV^{-3.5} ρ^{1.4}
# In keV units:   k_t = 10³ × 10^{-3.5} × T_keV^{-3.5} × ρ^{1.4}
#               = 10^{-0.5} × T_keV^{-3.5} × ρ^{1.4}
_OPACITY_PREFACTOR = 1e3 * (T_HEV_PER_KEV ** (-3.5))   # = 10^{-0.5}

# Energy density exponents:  β=2, μ=0.25  →  u = 10⁻³ T_HeV² ρ^{0.75}  GJ/cm³
# In keV units:   u = 10⁻³ × (10 T_keV)² × ρ^{0.75}  =  0.1 T_keV² ρ^{0.75}
_ENERGY_PREFACTOR = 1e-3 * (T_HEV_PER_KEV ** 2)        # = 0.1

# Self-similar solution parameters
DELTA          = 1.1157535873060416
T_ANALYTIC_AMP = 1.198199926365715    # in HeV
T_ANALYTIC_EXP = 0.0276392
XI_WAVE_EXP    = DELTA

# Physical time span
T_INIT_NS  = -(10.0 ** (1.0 / DELTA))  # ≈ -7.875084 ns
T_FINAL_NS = -1.0                       # ns

# Output times from test3.py
OUTPUT_TIMES_NS = [-6.591897629554719, -3.926450981261105, -1.0]

# =============================================================================
# MATERIAL PROPERTIES  (all T arguments in keV, r in cm)
# =============================================================================

def rho_r(r):
    """Density profile  ρ(r) = (r/cm)^{-ω}  g/cm³."""
    return r ** (-OMEGA)


def test3_opacity(T_keV, r):
    """
    Opacity  k_t(T, ρ) = 10³ (T/HeV)^{-3.5} (ρ/(g/cm³))^{1.4}  cm⁻¹

    T argument is in keV;  r is position in cm.
    """
    T_safe = np.maximum(T_keV, 1e-8)
    rho = rho_r(np.maximum(r, 1e-30))
    return _OPACITY_PREFACTOR * T_safe ** (-3.5) * rho ** 1.4


def test3_material_energy(T_keV, r):
    """
    Material energy density  u(T, r) = 0.1 T_keV² ρ(r)^{0.75}  GJ/cm³.

    Equivalent to  10¹³ (T/HeV)² (ρ/(g/cm³))^{0.75}  erg/cm³.
    """
    rho = rho_r(np.maximum(r, 1e-30))
    return _ENERGY_PREFACTOR * T_keV ** 2 * rho ** 0.75


def test3_inverse_material_energy(u_GJ, r):
    """
    Inverse of material energy: T in keV given u in GJ/cm³ at position r.

    u = 0.1 T_keV² ρ^{0.75}  →  T_keV = sqrt(u / (0.1 ρ^{0.75}))
    """
    rho = rho_r(np.maximum(r, 1e-30))
    return np.sqrt(np.maximum(u_GJ, 0.0) / (_ENERGY_PREFACTOR * rho ** 0.75))


def test3_specific_heat(T_keV, r):
    """
    Volumetric heat capacity  du/dT  in GJ/(cm³·keV).

    du/dT_keV = 0.1 × 2 × T_keV × ρ^{0.75}  =  0.2 T_keV ρ^{0.75}

    NOTE: Returns the *volumetric* quantity (not per-mass) because the
    default solver density RHO = 1.0; this makes get_beta = 4aT³/(du/dT)
    which is the physically correct linearisation parameter.
    """
    T_safe = np.maximum(T_keV, 1e-8)
    rho = rho_r(np.maximum(r, 1e-30))
    return 2.0 * _ENERGY_PREFACTOR * T_safe * rho ** 0.75


# =============================================================================
# SELF-SIMILAR SOLUTION
# =============================================================================

def _Wxsi_scalar(xi):
    """W(ξ) similarity profile (scalar)."""
    if xi >= 2.0:
        return (xi - 1.0) ** 0.210071 * (1.27048 - 0.0470724 * xi + 0.00179721 * xi ** 2)
    elif xi > 1.0:
        return (xi - 1.0) ** 0.357506 * (1.9792  - 0.619497  * xi + 0.110644  * xi ** 2)
    else:
        return 0.0

Wxsi = np.vectorize(_Wxsi_scalar)


def xsi_rt(r, t_ns):
    """Similarity coordinate  ξ(r, t) = (r / 10⁻⁴ cm) / (-t/ns)^δ."""
    return (r / 1e-4) / (-t_ns) ** DELTA


def T_analytic_HeV(r, t_ns):
    """
    Analytic temperature  T(r, t) = 1.1982 (-t/ns)^{0.0276392} W^{1/2}(ξ)  [HeV].
    """
    xi = xsi_rt(r, t_ns)
    return T_ANALYTIC_AMP * (-t_ns) ** T_ANALYTIC_EXP * Wxsi(xi) ** 0.5

T_analytic_HeV_vec = np.vectorize(T_analytic_HeV)


def T_analytic_keV(r, t_ns):
    """Analytic temperature in keV."""
    return T_analytic_HeV(r, t_ns) / T_HEV_PER_KEV


def u_analytic_per_1e13(r, t_ns):
    """
    Analytic energy density in units of 10¹³ erg/cm³.

    u [×10¹³ erg/cm³] = T_HeV² × ρ(r)^{0.75}
                       = T_HeV² × (r/cm)^{-0.45×0.75}
                       = T_HeV² × r^{-0.3375}
    """
    T_HeV = T_analytic_HeV_vec(r, t_ns)
    rho = rho_r(np.maximum(r, 1e-30))
    return T_HeV ** 2 * rho ** 0.75


# =============================================================================
# TIME-DEPENDENT BOUNDARY CONDITION
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

def plot_results(solutions, save_prefix='converging_marshak_wave_test3'):
    """
    Produce two PDFs matching the layout of test3.py:
      1) T profiles (HeV) at three output times + analytic
      2) u profiles (10¹³ erg/cm³) at three output times + analytic
    """
    colors = ['tab:blue', 'tab:orange', 'tab:red']
    r_anal = np.linspace(1e-8, R, 2000)

    # --- Temperature plot ---
    fig, ax = plt.subplots(figsize=(7, 5))
    for sol, col in zip(solutions, colors):
        t_ns  = sol['t_ns']
        r_cm  = sol['r_cm']
        T_HeV = sol['T_keV'] * T_HEV_PER_KEV
        t_label = f't = {t_ns:.2f} ns'
        ax.plot(r_cm / 1e-4, T_HeV, color=col, lw=2, label=t_label)
        T_ref = T_analytic_HeV_vec(r_anal, t_ns)
        ax.plot(r_anal / 1e-4, T_ref, color=col, lw=1.5, ls='--',
                label=f'analytic ({t_label})')
    ax.set_xlabel(r'$r$ ($\mu$m)')
    ax.set_ylabel(r'$T$ (HeV)')
    ax.set_ylim(ymin=0, ymax=2.0)
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
        # Convert solver u (GJ/cm³) → 10¹³ erg/cm³:  1 GJ = 10¹⁶ erg → × 10³
        rho_cells = rho_r(np.maximum(r_cm, 1e-30))
        u_sim = test3_material_energy(sol['T_keV'], r_cm) * 1e3   # ×10¹³ erg/cm³
        t_label = f't = {t_ns:.2f} ns'
        ax.plot(r_cm / 1e-4, u_sim, color=col, lw=2, label=t_label)
        u_ref = u_analytic_per_1e13(r_anal, t_ns)
        ax.plot(r_anal / 1e-4, u_ref, color=col, lw=1.5, ls='--',
                label=f'analytic ({t_label})')
    ax.set_xlabel(r'$r$ ($\mu$m)')
    ax.set_ylabel(r'$u$ ($10^{13}$ erg/cm$^3$)')
    ax.set_ylim(ymin=0, ymax=45)
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
        save_prefix='converging_marshak_wave_test3',
        t_duration=T_FINAL_NS - T_INIT_NS):   # ≈ 6.875 ns  (positive)
    """
    Run the converging Marshak wave Test 3 problem.

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
    t_duration : float
        Total elapsed solver time in ns (default: full run to t_final = -1 ns).
    """

    print("=" * 70)
    print("Converging Marshak Wave  (Test 3 — Spherical, Power-law density)")
    print("=" * 70)
    print(f"  Domain:      r ∈ [0, {R*1e4:.0f} μm]")
    print(f"  Cells:       {n_cells}")
    print(f"  ρ(r):        (r/cm)^{{-{OMEGA}}}  g/cm³")
    print(f"  t_init:      {T_INIT_NS:.6f} ns")
    print(f"  t_final:     {T_FINAL_NS} ns")
    print(f"  Duration:    {t_duration:.6f} ns")
    print(f"  dt_initial:  {dt_initial} ns,  dt_max: {dt_max} ns")
    print("=" * 70)

    # Initial temperature: effectively cold
    T_init_keV = 1e-4   # very cold initial material

    # Build solver
    solver = NonEquilibriumRadiationDiffusionSolver(
        r_min=0.0,
        r_max=R,
        n_cells=n_cells,
        d=2,                           # spherical geometry
        dt=dt_initial,
        max_newton_iter=30,
        newton_tol=1e-8,
        rosseland_opacity_func=test3_opacity,
        planck_opacity_func=test3_opacity,
        specific_heat_func=test3_specific_heat,
        material_energy_func=test3_material_energy,
        inverse_material_energy_func=test3_inverse_material_energy,
        left_bc_func=left_bc,
        right_bc_func=right_bc,
        theta=1.0,
    )

    # Uniform cold initial condition
    phi_init = C_LIGHT * A_RAD * T_init_keV ** 4
    solver.phi = np.full(n_cells, phi_init)
    solver.T   = np.full(n_cells, T_init_keV)

    # Sorted target elapsed times (τ = t_phys - T_INIT_NS)
    output_elapsed = sorted(t - T_INIT_NS for t in OUTPUT_TIMES_NS)
    output_t_ns    = sorted(OUTPUT_TIMES_NS)
    output_saved   = set()

    solutions    = []
    phys_elapsed = 0.0
    step         = 0

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
                  f"T_center = {solver.T[0]*T_HEV_PER_KEV:.6f} HeV")

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
                print(f"  >> Snapshot saved at t = {t_out:.6f} ns  "
                      f"(T_max = {solver.T.max()*T_HEV_PER_KEV:.4f} HeV)")

        # Grow time step for next iteration
        solver.dt = min(solver.dt * dt_growth, dt_max)

    print(f"\nTotal steps: {step}")
    print(f"Snapshots saved: {len(solutions)}")

    plot_results(solutions, save_prefix=save_prefix)
    return solver, solutions


if __name__ == "__main__":
    solver, solutions = run(
        n_cells=200,
        dt_initial=0.0001,
        dt_max=0.01,
        dt_growth=1.05,
    )
