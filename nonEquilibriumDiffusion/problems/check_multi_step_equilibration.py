#!/usr/bin/env python3
"""
Check if multiple large timesteps improve equilibration.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from multigroup_diffusion_solver import MultigroupDiffusionSolver1D, C_LIGHT, A_RAD

# Setup
r_min, r_max = 0.0, 1.0e-3
n_cells = 1
dt = 100.0  # Large timestep
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
E_r_eq = A_RAD * T_eq**4

print("="*80)
print("MULTI-STEP EQUILIBRATION TEST")
print("="*80)
print(f"Expected equilibrium: T_eq = {T_eq:.15f} keV")
print(f"                      E_r_eq = {E_r_eq:.15e} GJ/cm³")
print()

for step in range(1, 11):
    solver.step()
    
    T_mat = solver.T[0]
    E_r = solver.E_r[0]
    T_rad = (E_r / A_RAD) ** 0.25
    E_r_from_Tmat = A_RAD * T_mat**4
    
    T_diff = abs(T_mat - T_rad)
    E_r_discrepancy = abs(E_r - E_r_from_Tmat)
    T_mat_error = abs(T_mat - T_eq)
    E_r_error = abs(E_r - E_r_eq)
    
    print(f"Step {step:2d}:")
    print(f"  T_mat = {T_mat:.15f} keV  (error vs T_eq: {T_mat_error:.3e})")
    print(f"  T_rad = {T_rad:.15f} keV  (diff: {T_diff:.3e})")
    print(f"  E_r = {E_r:.15e} GJ/cm³")
    print(f"  E_r - a·T_mat⁴ = {E_r_discrepancy:.3e} (rel: {E_r_discrepancy/E_r:.2e})")
    print()

print("="*80)
print("CONCLUSION")
print("="*80)
if T_diff < 1e-10:
    print("✓ Temperatures converged to machine precision")
else:
    print(f"⚠ Temperatures still differ by {T_diff:.3e} keV after 10 large steps")
    print(f"  Relative difference: {T_diff/T_mat:.2%}")
