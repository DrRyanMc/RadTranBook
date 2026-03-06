#!/usr/bin/env python3
"""
Check what's happening to phi_g at pseudo-equilibrium
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
T_rad = 0.05

solver.T[:] = T_init
solver.T_old[:] = T_init
solver.E_r[:] = A_RAD * T_rad**4
solver.E_r_old[:] = solver.E_r.copy()
solver.phi_g_fraction[:, :] = chi[:, np.newaxis]
solver.phi_g_stored[:, :] = chi[:, np.newaxis] * A_RAD * C_LIGHT * T_rad**4

print(f"INITIAL STATE:")
print(f"  T = {T_init} keV")
print(f"  E_r = {solver.E_r[0]:.6e}")
print(f"Expected equilibrium:")
E_total = T_init * cv * rho + solver.E_r[0]
from scipy.optimize import brentq
T_eq = brentq(lambda T: cv * rho * T + A_RAD * T**4 - E_total, 0.001, 1.0)
print(f"  T_eq = {T_eq:.8f} keV")
print(f"  E_r_eq = {A_RAD * T_eq**4:.6e}")
print()

# Take 3 steps and examine phi_g
for step in range(3):
    print(f"{'='*80}")
    print(f"STEP {step}")
    print(f"{'='*80}")
    print(f"Before step:")
    print(f"  T = {solver.T[0]:.8f} keV")
    print(f" E_r = {solver.E_r[0]:.10e}")
    print(f"  φ_g (by group):")
    for g in range(n_groups):
        print(f"    Group {g}: φ = {solver.phi_g_stored[g,0]:.6e}")
    print(f"  Σφ_g = {np.sum(solver.phi_g_stored[:,0]):.6e}")
    print(f" c·E_r = {C_LIGHT * solver.E_r[0]:.6e}")
    print()
    
    solver.step(verbose=False)
    
    print(f"After step:")
    print(f"  T = {solver.T[0]:.8f} keV (ΔT = {solver.T[0] - T_init:.6e})")
    print(f"  E_r = {solver.E_r[0]:.10e}")
    print(f"  φ_g (by group):")
    for g in range(n_groups):
        print(f"    Group {g}: φ = {solver.phi_g_stored[g,0]:.6e}")
    print(f"  Σφ_g = {np.sum(solver.phi_g_stored[:,0]):.6e}")
    print(f"  c·E_r = {C_LIGHT * solver.E_r[0]:.6e}")
    print()
    
    # Check: does Σφ_g = c·E_r?
    sum_phi = np.sum(solver.phi_g_stored[:,0])
    c_Er = C_LIGHT * solver.E_r[0]
    print(f"  Consistency check: Σφ_g vs c·E_r")
    print(f"    Σφ_g   = {sum_phi:.10e}")
    print(f"    c·E_r  = {c_Er:.10e}")
    print(f"    Ratio  = {sum_phi / c_Er:.6f}")
    print()
    
    # What should φ_g be at equilibrium?
    print(f"  At equilibrium (T={T_eq:.6f} keV):")
    B_g_eq = Bg_multigroup(energy_edges, T_eq)
    phi_eq_total = A_RAD * C_LIGHT * T_eq**4
    phi_g_eq = (B_g_eq / B_g_eq.sum()) * phi_eq_total
    print(f"    Expected φ_total = {phi_eq_total:.6e}")
    print(f"    Expected φ_g:")
    for g in range(n_groups):
        print(f"      Group {g}: φ_eq = {phi_g_eq[g]:.6e}")
    print(f"    Actual φ_total = {sum_phi:.6e}")
    print(f"    Ratio (actual/expected) = {sum_phi / phi_eq_total:.6f}")
    print()
