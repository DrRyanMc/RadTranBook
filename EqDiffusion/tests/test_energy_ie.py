#!/usr/bin/env python3
"""Check energy conservation with implicit Euler"""

import numpy as np
from oneDFV import *

# Setup
sigma_R = 100.0
k_coupling = 1.0

def constant_opacity(Er):
    return sigma_R

def cubic_cv(T):
    return 4.0 * k_coupling * A_RAD * T**3

def linear_material_energy(T):
    return k_coupling * A_RAD * T**4

def left_bc(Er, x):
    return (0.0, 1.0, 0.0)  # Insulating: flux = 0

def right_bc(Er, x):
    return (0.0, 1.0, 0.0)  # Insulating: flux = 0

# Small problem
solver = RadiationDiffusionSolver(
    r_min=0.0, r_max=2.0, n_cells=10, d=0,
    dt=0.001, max_newton_iter=50, newton_tol=1e-6,
    rosseland_opacity_func=constant_opacity,
    specific_heat_func=cubic_cv,
    material_energy_func=linear_material_energy,
    left_bc_func=left_bc,
    right_bc_func=right_bc
)

# Initial condition
Er_background = A_RAD * 0.1**4
amplitude = A_RAD * (1.0**4 - 0.1**4)
def gaussian_Er(r):
    return Er_background + amplitude * np.exp(-(r - 1.0)**2 / (2 * 0.15**2))

solver.set_initial_condition(gaussian_Er)

# Compute total energy
def total_energy(solver):
    Er_sum = np.sum(solver.Er * solver.V_cells)
    e_mat_sum = 0.0
    for i in range(len(solver.Er)):
        T = temperature_from_Er(solver.Er[i])
        e_mat = linear_material_energy(T)
        e_mat_sum += e_mat * solver.V_cells[i]
    return Er_sum + e_mat_sum

E_init = total_energy(solver)
print(f'Initial: E_total = {E_init:.6e}, Er_max = {solver.Er.max():.6e}')

# Run steps and check energy
for step in range(10):
    solver.time_step(n_steps=1, verbose=False)
    E = total_energy(solver)
    ratio = E / E_init
    print(f'Step {step+1}: E_total = {E:.6e} (ratio = {ratio:.9f}), Er_max = {solver.Er.max():.6e}')
