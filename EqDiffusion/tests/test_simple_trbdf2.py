#!/usr/bin/env python3
"""Test TR-BDF2 on small problem"""

import numpy as np
from oneDFV import *

# Setup
sigma_R = 100.0
k_coupling = 1.0e-6

def constant_opacity(Er):
    return sigma_R

def cubic_cv(T):
    return 4.0 * k_coupling * A_RAD * T**3

def linear_material_energy(T):
    return k_coupling * A_RAD * T**4

def left_bc(Er, x):
    return (1.0, 0.0, 0.0)

def right_bc(Er, x):
    return (1.0, 0.0, 0.0)

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
print(f'Initial: Er min={solver.Er.min():.6e}, max={solver.Er.max():.6e}')

# Run 10 TR-BDF2 steps
for step in range(10):
    solver.time_step_trbdf2(n_steps=1, verbose=False)
    print(f'Step {step+1}: Er min={solver.Er.min():.6e}, max={solver.Er.max():.6e}')
