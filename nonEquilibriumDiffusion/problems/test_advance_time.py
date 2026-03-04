#!/usr/bin/env python3
"""Test if advance_time() causes energy loss."""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from multigroup_diffusion_solver import MultigroupDiffusionSolver1D

C_LIGHT = 2.99792458e1
A_RAD = 0.01372

solver = MultigroupDiffusionSolver1D(
    n_groups=1, r_min=0.0, r_max=1.0, n_cells=1, energy_edges=np.array([0.01, 10.0]),
    geometry='planar', dt=0.01, diffusion_coeff_funcs=[lambda T, r: 1e10],
    absorption_coeff_funcs=[lambda T, r: 5.0], left_bc='neumann', right_bc='neumann',
    left_bc_values=[0.0], right_bc_values=[0.0], rho=1.0, cv=0.01
)

solver.T = np.array([0.4])
solver.E_r = np.array([A_RAD * 1.0**4])

# Initialize "old" values for first timestep
solver.T_old = solver.T.copy()
solver.E_r_old = solver.E_r.copy()

E0 = solver.E_r[0] + 1.0 * 0.01 * solver.T[0]

print(f'Step 0: T={solver.T[0]:.6f}, E_r={solver.E_r[0]:.6e}, E_total={solver.E_r[0] + 1.0*0.01*solver.T[0]:.6e}')

for step in range(1, 3):
    info = solver.step(max_newton_iter=10, newton_tol=1e-8, gmres_tol=1e-6)
    E_total = solver.E_r[0] + 1.0 * 0.01 * solver.T[0]
    print(f'Step {step} (before advance_time): T={solver.T[0]:.6f}, E_r={solver.E_r[0]:.6e}, E_total={E_total:.6e}, ΔE/E0={((E_total-E0)/E0):.6e}')
    solver.advance_time()
    E_total_after = solver.E_r[0] + 1.0 * 0.01 * solver.T[0]
    print(f'Step {step} (after  advance_time): T={solver.T[0]:.6f}, E_r={solver.E_r[0]:.6e}, E_total={E_total_after:.6e}, ΔE/E0={((E_total_after-E0)/E0):.6e}')
    print()
