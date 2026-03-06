#!/usr/bin/env python3
"""
Simple single-step test to see what's happening
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from multigroup_diffusion_solver import MultigroupDiffusionSolver1D, C_LIGHT, A_RAD

print("Creating solver...")

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

print("Initializing solver...")

solver = MultigroupDiffusionSolver1D(
    n_groups=n_groups, r_min=r_min, r_max=r_max, n_cells=n_cells,
    energy_edges=energy_edges, geometry='planar', dt=dt,
    diffusion_coeff_funcs=diff_funcs, absorption_coeff_funcs=sigma_funcs,
    left_bc_funcs=left_bcs, right_bc_funcs=right_bcs,
    emission_fractions=chi, rho=rho, cv=cv
)

solver._debug_update_T = False

print("Setting initial conditions...")

T_init = 0.025
T_rad = 0.05

solver.T[:] = T_init
solver.T_old[:] = T_init
solver.E_r[:] = A_RAD * T_rad**4
solver.E_r_old[:] = solver.E_r.copy()
solver.phi_g_fraction[:, :] = chi[:, np.newaxis]
solver.phi_g_stored[:, :] = chi[:, np.newaxis] * A_RAD * C_LIGHT * T_rad**4

print(f"T_init = {T_init}")
print(f"E_r_init = {solver.E_r[0]:.6e}")

print("\nTaking one step...")
import time
t0 = time.time()
solver.step(verbose=True)
t1 = time.time()

print(f"\nStep completed in {t1-t0:.2f} seconds")
print(f"T_final = {solver.T[0]}")
print(f"E_r_final = {solver.E_r[0]:.6e}")
print(f"Delta T = {solver.T[0] - T_init}")
