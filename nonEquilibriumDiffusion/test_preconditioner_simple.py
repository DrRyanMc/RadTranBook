#!/usr/bin/env python3
"""
Simple test of LMFG preconditioner on 1-group problem with Neumann BCs
This is the simplest possible case to debug the preconditioner
"""

import sys
import numpy as np

# Import the solver
from multigroup_diffusion_solver import MultigroupDiffusionSolver1D, C_LIGHT, A_RAD

print("Testing LMFG Preconditioner - 1 Group, Neumann BCs")
print("="*70)

# Single group problem
n_groups = 1
n_cells = 30
r_max = 5.0  # cm
dt = 0.1  # ns

energy_edges = np.array([0.0, 10.0])  # Single broad group

# Simple constant opacity (temperature-independent for now)
sigma_a = 100.0  # cm^-1
sigma_r = 100.0  # cm^-1

def diff_coeff(T, r):
    """Constant diffusion coefficient D = 1/(3*sigma_r)"""
    return 1.0 / (3.0 * sigma_r)

def abs_coeff(T, r):
    """Constant absorption coefficient"""
    return sigma_a

# Neumann BCs (zero flux) on both boundaries
def neumann_bc(phi, r):
    """Zero flux: dφ/dr = 0"""
    return 0.0, 1.0, 0.0  # A*phi + B*dphi/dr = C, so 0*phi + 1*dphi/dr = 0

print(f"Problem setup:")
print(f"  Groups: {n_groups}")
print(f"  Cells: {n_cells}")
print(f"  Domain: [0, {r_max}] cm")
print(f"  Absorption opacity: {sigma_a} cm^-1")
print(f"  Rosseland opacity: {sigma_r} cm^-1")
print(f"  Diffusion coeff: {1.0/(3.0*sigma_r):.4f} cm^2/ns")
print(f"  dt: {dt} ns")
print(f"  Boundary conditions: Neumann (zero flux) on both sides")
print()

# Create solver
solver = MultigroupDiffusionSolver1D(
    n_groups=n_groups,
    r_min=0.0,
    r_max=r_max,
    n_cells=n_cells,
    geometry='planar',
    dt=dt,
    energy_edges=energy_edges,
    diffusion_coeff_funcs=[diff_coeff],
    absorption_coeff_funcs=[abs_coeff],
    left_bc_funcs=[neumann_bc],
    right_bc_funcs=[neumann_bc],
    rho=1.0,  # g/cm^3
    cv=0.1    # GJ/(g*keV)
)

# Initialize with a smooth temperature gradient
# T(r) = T_cold + (T_hot - T_cold) * exp(-r/L)
T_hot = 1.0   # keV
T_cold = 0.1  # keV
L = 2.0       # characteristic length
r_centers = solver.r_centers

# Create initial temperature profile
T_init = T_cold + (T_hot - T_cold) * np.exp(-r_centers / L)

print(f"Initial temperature profile:")
print(f"  T(r=0) = {T_init[0]:.4f} keV")
print(f"  T(r={r_max}) = {T_init[-1]:.4f} keV")
print(f"  Profile: T(r) = {T_cold} + {T_hot - T_cold} * exp(-r/{L})")
print()

# Set initial conditions
solver.T = T_init.copy()
solver.T_old = T_init.copy()
solver.E_r = A_RAD * T_init**4
solver.E_r_old = solver.E_r.copy()
solver.kappa = np.zeros(n_cells)
solver.kappa_old = np.zeros(n_cells)

print("="*70)
print("Test 1: WITHOUT preconditioner")
print("="*70)
info1 = solver.step(
    max_newton_iter=10,
    newton_tol=1e-8,
    gmres_tol=1e-8,
    gmres_maxiter=200,
    use_preconditioner=False,
    verbose=True
)

print(f"\n{'='*70}")
print(f"Result WITHOUT preconditioner:")
print(f"{'='*70}")
print(f"  Converged: {info1['converged']}")
print(f"  Newton iterations: {info1['newton_iter']}")
print(f"  GMRES iterations: {info1['gmres_info']['iterations']}")
print(f"  Final T_max: {solver.T.max():.6f} keV")
print(f"  Final T_min: {solver.T.min():.6f} keV")
print(f"  Final E_r_max: {solver.E_r.max():.6e} GJ/cm^3")
print()

# Reset for second test - use same initial conditions
solver.T = T_init.copy()
solver.T_old = T_init.copy()
solver.E_r = A_RAD * T_init**4
solver.E_r_old = solver.E_r.copy()
solver.kappa = np.zeros(n_cells)
solver.kappa_old = np.zeros(n_cells)

print("="*70)
print("Test 2: WITH LMFG preconditioner")
print("="*70)
info2 = solver.step(
    max_newton_iter=10,
    newton_tol=1e-8,
    gmres_tol=1e-8,
    gmres_maxiter=200,
    use_preconditioner=True,
    verbose=True
)

print(f"\n{'='*70}")
print(f"Result WITH preconditioner:")
print(f"{'='*70}")
print(f"  Converged: {info2['converged']}")
print(f"  Newton iterations: {info2['newton_iter']}")
print(f"  GMRES iterations: {info2['gmres_info']['iterations']}")
print(f"  Final T_max: {solver.T.max():.6f} keV")
print(f"  Final T_min: {solver.T.min():.6f} keV")
print(f"  Final E_r_max: {solver.E_r.max():.6e} GJ/cm^3")
print()

print("="*70)
print("Comparison:")
print("="*70)
if info1['converged'] and info2['converged']:
    print(f"✓ Both methods converged")
    print(f"  Newton iterations: {info1['newton_iter']} (no precond) vs {info2['newton_iter']} (with precond)")
    print(f"  GMRES iterations: {info1['gmres_info']['iterations']} (no precond) vs {info2['gmres_info']['iterations']} (with precond)")
elif info1['converged'] and not info2['converged']:
    print(f"✗ Preconditioner FAILED - unpreconditioned worked")
    print(f"  This indicates a bug in the preconditioner implementation")
elif not info1['converged'] and info2['converged']:
    print(f"✓ Preconditioner HELPED - unpreconditioned failed")
else:
    print(f"✗ Both methods failed - problem may be too difficult")

print("="*70)
print("Test completed!")
print("="*70)
