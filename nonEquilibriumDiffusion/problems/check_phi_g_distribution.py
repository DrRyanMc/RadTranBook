#!/usr/bin/env python3
"""
Check if φ_g distribution at "equilibrium" matches the Planck distribution.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from multigroup_diffusion_solver import MultigroupDiffusionSolver1D, C_LIGHT, A_RAD

# Setup
r_min, r_max = 0.0, 1.0e-3
n_cells = 1
dt = 100.0
rho, cv = 1.0, 0.05

n_groups = 5
energy_edges = np.logspace(np.log10(1e-4), np.log10(5.0), n_groups + 1)
sigma_values = [100.0] * n_groups

sigma_funcs = []
diff_funcs = []
for g in range(n_groups):
    sig = sigma_values[g]
    sigma_funcs.append(lambda T, r, s=sig: s)
    diff_funcs.append(lambda T, r, s=sig: C_LIGHT / (3.0 * s))

from planck_integrals import Bg_multigroup
B_g = Bg_multigroup(energy_edges, 0.05)
chi = B_g / B_g.sum()

def bc(phi, r):
    return 0.0, 1.0, 0.0

left_bcs = [bc] * n_groups
right_bcs = [bc] * n_groups

solver = MultigroupDiffusionSolver1D(
    n_groups=n_groups, r_min=r_min, r_max=r_max, n_cells=n_cells,
    energy_edges=energy_edges, geometry='planar', dt=dt,
    diffusion_coeff_funcs=diff_funcs, absorption_coeff_funcs=sigma_funcs,
    left_bc_funcs=left_bcs, right_bc_funcs=right_bcs,
    emission_fractions=chi, rho=rho, cv=cv
)

solver._debug_update_T = False

T_init = 0.025
T_rad_init = 0.05

solver.T[:] = T_init
solver.T_old[:] = T_init
solver.E_r[:] = A_RAD * T_rad_init**4
solver.E_r_old[:] = solver.E_r.copy()
solver.phi_g_fraction[:, :] = chi[:, np.newaxis]
solver.phi_g_stored[:, :] = chi[:, np.newaxis] * A_RAD * C_LIGHT * T_rad_init**4

# Compute expected equilibrium
E_total_init = T_init * cv * rho + solver.E_r[0]
from scipy.optimize import brentq
T_eq = brentq(lambda T: cv * rho * T + A_RAD * T**4 - E_total_init, 0.001, 1.0)

print("="*80)
print("CHECKING φ_g DISTRIBUTION AT EQUILIBRIUM")
print("="*80)
print(f"Expected equilibrium: T_eq = {T_eq:.10f} keV")
print()

solver.step()

T_mat = solver.T[0]
E_r = solver.E_r[0]
T_rad = (E_r / A_RAD) ** 0.25

print(f"After one large step:")
print(f"  T_mat = {T_mat:.10f} keV")
print(f"  T_rad = {T_rad:.10f} keV")
print(f"  Difference = {abs(T_mat - T_rad):.3e} keV")
print()

# Get φ_g from solver
phi_g_solver = solver.phi_g_stored[:, 0]  # Shape: (n_groups,)
phi_total_solver = phi_g_solver.sum()

print("-"*80)
print("φ_g FROM SOLVER:")
print("-"*80)
print("Group |    φ_g (solver)   | Fraction of total")
print("------+-------------------+------------------")
for g in range(n_groups):
    frac = phi_g_solver[g] / phi_total_solver
    print(f"  {g}   | {phi_g_solver[g]:.6e}  |  {frac*100:5.2f}%")
print(f"TOTAL | {phi_total_solver:.6e}  | 100.00%")
print()

# Compute expected φ_g from Planck distribution at T_mat
# At equilibrium: φ_g = (4π/c) · B_g(T_mat)
from planck_integrals import Bg_multigroup
B_g_Tmat = Bg_multigroup(energy_edges, T_mat)
phi_g_planck = (4.0 * np.pi / C_LIGHT) * B_g_Tmat
phi_total_planck = phi_g_planck.sum()

print("-"*80)
print(f"φ_g FROM PLANCK AT T_mat = {T_mat:.6f} keV:")
print("-"*80)
print("Group |   φ_g (Planck)    | Fraction of total")
print("------+-------------------+------------------")
for g in range(n_groups):
    frac = phi_g_planck[g] / phi_total_planck
    print(f"  {g}   | {phi_g_planck[g]:.6e}  |  {frac*100:5.2f}%")
print(f"TOTAL | {phi_total_planck:.6e}  | 100.00%")
print()

# Compare
print("="*80)
print("COMPARISON")
print("="*80)
print("Group |   Solver / Planck  |  Difference %")
print("------+--------------------+---------------")
for g in range(n_groups):
    ratio = phi_g_solver[g] / phi_g_planck[g]
    diff_pct = (ratio - 1.0) * 100
    print(f"  {g}   |  {ratio:.10f}    |  {diff_pct:+.6f}%")
print(f"TOTAL |  {phi_total_solver/phi_total_planck:.10f}    |  {(phi_total_solver/phi_total_planck - 1)*100:+.6f}%")
print()

# Check if total φ explains the T_rad discrepancy
print("="*80)
print("DIAGNOSIS")
print("="*80)
print(f"Σφ_g (solver) = {phi_total_solver:.15e}")
print(f"Σφ_g (Planck, T_mat) = {phi_total_planck:.15e}")
print(f"Ratio = {phi_total_solver/phi_total_planck:.10f}")
print()

E_r_from_solver_phi = phi_total_solver / C_LIGHT
E_r_from_Tmat = A_RAD * T_mat**4
print(f"E_r from solver φ_g:  {E_r_from_solver_phi:.15e}")
print(f"E_r = a·T_mat⁴:       {E_r_from_Tmat:.15e}")
print(f"E_r from solver.E_r:  {E_r:.15e}")
print()

T_rad_from_phi = (E_r_from_solver_phi / A_RAD) ** 0.25
print(f"T_rad from φ_g: {T_rad_from_phi:.15f} keV")
print(f"T_mat:          {T_mat:.15f} keV")
print(f"Difference:     {abs(T_rad_from_phi - T_mat):.3e} keV")
print()

if abs(phi_total_solver/phi_total_planck - 1.0) > 1e-5:
    print("✗ The φ_g distribution from the solver does NOT match the Planck distribution!")
    print("  This explains why T_rad ≠ T_mat at equilibrium.")
    print()
    print("Possible causes:")
    print("  1. Multigroup discretization error")
    print("  2. Solver iteration not fully converged")
    print("  3. Some approximation in how φ_g is computed")
else:
    print("✓ The φ_g distribution matches the Planck distribution well")
    print("  The issue must be something else...")
