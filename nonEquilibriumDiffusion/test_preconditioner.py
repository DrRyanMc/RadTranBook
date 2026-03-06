#!/usr/bin/env python3
"""
Simple test of LMFG preconditioner on small problem
"""

import sys
import numpy as np

# Import the solver
from multigroup_diffusion_solver import MultigroupDiffusionSolver1D, C_LIGHT, A_RAD

print("Testing LMFG Preconditioner")
print("="*60)

# Simple 2-group problem
n_groups = 2
n_cells = 20
r_max = 1.0
dt = 0.01

energy_edges = np.array([0.1, 2.0, 50.0])

# Simple constant opacity
sigma_const = 100.0  # cm^-1

def diff_coeff(T, r):
    return 1.0 / (3.0 * sigma_const)

def abs_coeff(T, r):
    return sigma_const / n_groups  # Split equally

# Simple Dirichlet BCs
def left_bc(phi, r):
    return 1.0, 0.0, 1.0  # phi = 1

def right_bc(phi, r):
    return 1.0, 0.0, 0.0  # phi = 0

print(f"Problem setup:")
print(f"  Groups: {n_groups}")
print(f"  Cells: {n_cells}")
print(f"  Opacity: {sigma_const} cm^-1")
print(f"  dt: {dt} ns")

# Create solver
solver = MultigroupDiffusionSolver1D(
    n_groups=n_groups,
    r_min=0.0,
    r_max=r_max,
    n_cells=n_cells,
    geometry='planar',
    dt=dt,
    energy_edges=energy_edges,
    diffusion_coeff_funcs=[diff_coeff] * n_groups,
    absorption_coeff_funcs=[abs_coeff] * n_groups,
    left_bc_funcs=[left_bc] * n_groups,
    right_bc_funcs=[right_bc] * n_groups,
    rho=1.0,
    cv=0.1
)

# Initialize with cold material
solver.T = np.ones(n_cells) * 0.1
solver.T_old = solver.T.copy()
solver.E_r = np.ones(n_cells) * A_RAD * 0.1**4
solver.E_r_old = solver.E_r.copy()
solver.kappa = np.zeros(n_cells)
solver.kappa_old = np.zeros(n_cells)

print("\nTest 1: WITHOUT preconditioner")
print("-" * 60)
info1 = solver.step(
    max_newton_iter=5,
    newton_tol=1e-6,
    gmres_tol=1e-6,
    gmres_maxiter=100,
    use_preconditioner=False,
    verbose=True
)

print(f"\nResult WITHOUT preconditioner:")
print(f"  Converged: {info1['converged']}")
print(f"  Newton iterations: {info1['newton_iter']}")
print(f"  GMRES info: {info1['gmres_info']}")

# Reset for second test
solver.T = np.ones(n_cells) * 0.1
solver.T_old = solver.T.copy()
solver.E_r = np.ones(n_cells) * A_RAD * 0.1**4
solver.E_r_old = solver.E_r.copy()
solver.kappa = np.zeros(n_cells)
solver.kappa_old = np.zeros(n_cells)

print("\n\nTest 2: WITH LMFG preconditioner")
print("-" * 60)
info2 = solver.step(
    max_newton_iter=5,
    newton_tol=1e-6,
    gmres_tol=1e-6,
    gmres_maxiter=100,
    use_preconditioner=True,
    verbose=True
)

print(f"\nResult WITH preconditioner:")
print(f"  Converged: {info2['converged']}")
print(f"  Newton iterations: {info2['newton_iter']}")
print(f"  GMRES info: {info2['gmres_info']}")

print("\n" + "="*60)
print("Preconditioner test completed!")
print("="*60)
