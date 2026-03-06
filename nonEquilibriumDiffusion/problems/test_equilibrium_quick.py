#!/usr/bin/env python3
"""
Track energy conservation and equilibrium convergence over multiple steps
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from multigroup_diffusion_solver import MultigroupDiffusionSolver1D, C_LIGHT, A_RAD

# Simple setup
r_min, r_max = 0.0, 1.0e-3
n_cells = 1
dt = 0.01
rho, cv = 1.0, 0.05

n_groups = 5
energy_edges = np.logspace(np.log10(1e-4), np.log10(5.0), n_groups + 1)

# Constant opacities
sigma_values = [100.0] * n_groups

sigma_funcs = []
diff_funcs = []
for g in range(n_groups):
    sig = sigma_values[g]
    sigma_funcs.append(lambda T, r, s=sig: s)
    diff_funcs.append(lambda T, r, s=sig: C_LIGHT / (3.0 * s))

# Chi
from planck_integrals import Bg_multigroup
B_g = Bg_multigroup(energy_edges, 0.05)
chi = B_g / B_g.sum()

# Reflecting BCs
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
T_rad = 0.05

solver.T[:] = T_init
solver.T_old[:] = T_init
solver.E_r[:] = A_RAD * T_rad**4
solver.E_r_old[:] = solver.E_r.copy()
solver.phi_g_fraction[:, :] = chi[:, np.newaxis]
solver.phi_g_stored[:, :] = chi[:, np.newaxis] * A_RAD * C_LIGHT * T_rad**4

# Compute total energy
def compute_total_energy():
    e_mat = solver.T[0] * cv * rho
    e_rad = solver.E_r[0]
    return e_mat + e_rad

def compute_T_rad():
    return (solver.E_r[0] / A_RAD) ** 0.25

E_total_init = compute_total_energy()
T_rad_init = compute_T_rad()

print(f"INITIAL STATE:")
print(f"  T_mat    = {T_init:.8f} keV")
print(f"  T_rad    = {T_rad_init:.8f} keV")
print(f"  E_mat    = {T_init * cv * rho:.10e} GJ/cm³")
print(f"  E_rad    = {solver.E_r[0]:.10e} GJ/cm³")
print(f"  E_total  = {E_total_init:.10e} GJ/cm³")
print()

# Expected equilibrium
from scipy.optimize import brentq
def energy_balance(T):
    return cv * rho * T + A_RAD * T**4 - E_total_init

T_eq_expected = brentq(energy_balance, 0.001, 1.0)

print(f"EXPECTED EQUILIBRIUM:")
print(f"  T_eq = {T_eq_expected:.8f} keV")
print(f"  E_mat_eq = {cv * rho * T_eq_expected:.10e} GJ/cm³")
print(f"  E_rad_eq = {A_RAD * T_eq_expected**4:.10e} GJ/cm³")
print(f"  E_total = {cv * rho * T_eq_expected + A_RAD * T_eq_expected**4:.10e} GJ/cm³")
print()

print(f"{'='*100}")
print(f"Step     T_mat [keV]  T_rad [keV]  |ΔT|        E_mat [GJ/cm³]  E_rad [GJ/cm³]  E_total [GJ/cm³]  ΔE_total")
print(f"{'-'*100}")

for step in range(20):
    solver.step(verbose=False)
    
    T_mat = solver.T[0]
    T_rad = compute_T_rad()
    E_mat = T_mat * cv * rho
    E_rad = solver.E_r[0]
    E_total = E_mat + E_rad
    delta_E = E_total - E_total_init
    delta_T = abs(T_mat - T_rad)
    
    print(f"{step:4d}     {T_mat:.8f}   {T_rad:.8f}   {delta_T:.3e}    {E_mat:.10e}  {E_rad:.10e}  {E_total:.10e}  {delta_E:.3e}")

print(f"{'='*100}")
print()

T_mat_final = solver.T[0]
T_rad_final = compute_T_rad()
E_total_final = compute_total_energy()

print(f"FINAL STATE (after 20 steps):")
print(f"  T_mat    = {T_mat_final:.8f} keV")
print(f"  T_rad    = {T_rad_final:.8f} keV")
print(f"  |T_mat - T_rad| = {abs(T_mat_final - T_rad_final):.3e} keV")
print()

print(f"COMPARISON TO EXPECTED:")
print(f"  Expected T_eq = {T_eq_expected:.8f} keV")
print(f"  Actual T_mat  = {T_mat_final:.8f} keV")
print(f"  Error = {abs(T_mat_final - T_eq_expected):.3e} keV")
print()

print(f"ENERGY CONSERVATION:")
print(f"  Initial E_total = {E_total_init:.10e} GJ/cm³")
print(f"  Final E_total   = {E_total_final:.10e} GJ/cm³")
print(f"  Error = {abs(E_total_final - E_total_init):.3e} GJ/cm³")
print(f"  Relative error = {abs(E_total_final - E_total_init) / E_total_init:.3e}")
