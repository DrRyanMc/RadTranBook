#!/usr/bin/env python3
"""Debug TR-BDF2 with verbose output"""

import numpy as np
from oneDFV import *

# Same setup as test
sigma_R = 100.0
k_coupling = 1.0
r_min, r_max = 0.0, 2.0
n_cells = 200
dt = 0.001
x0 = 1.0
sigma0 = 0.15
T_peak = 1.0
T_background = 0.1

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

# Create solver
solver = RadiationDiffusionSolver(
    r_min=r_min, r_max=r_max, n_cells=n_cells, d=0,
    dt=dt, max_newton_iter=50, newton_tol=1e-6,
    rosseland_opacity_func=constant_opacity,
    specific_heat_func=cubic_cv,
    material_energy_func=linear_material_energy,
    left_bc_func=left_bc,
    right_bc_func=right_bc
)

# Set IC
Er_background = A_RAD * T_background**4
amplitude = A_RAD * (T_peak**4 - T_background**4)
def gaussian_Er(r):
    return Er_background + amplitude * np.exp(-(r - x0)**2 / (2 * sigma0**2))

solver.set_initial_condition(gaussian_Er)
print(f'Initial: Er min={solver.Er.min():.6e}, max={solver.Er.max():.6e}')

# Run one TR-BDF2 step
print('\n=== STEP 1 ===')
solver.time_step_trbdf2(n_steps=1, verbose=True)
print(f'After step 1: Er min={solver.Er.min():.6e}, max={solver.Er.max():.6e}')

# Run another step
print('\n=== STEP 2 ===')
solver.time_step_trbdf2(n_steps=1, verbose=True)
print(f'After step 2: Er min={solver.Er.min():.6e}, max={solver.Er.max():.6e}')
