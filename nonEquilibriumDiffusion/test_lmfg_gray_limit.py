#!/usr/bin/env python3
"""
Test #1: Gray-limit exactness test for LMFGK.

Make all groups identical so the multigroup problem is truly gray.
LMFGK should then be near-optimal and GMRES iterations should drop dramatically.
"""

import sys
import numpy as np
sys.path.insert(0, '/Users/ryanmcclarren/Dropbox/Papers/RadTranBook/nonEquilibriumDiffusion')

from diffusion_operator_solver import C_LIGHT, A_RAD
import multigroup_diffusion_solver_patched_lmfgk as m
from multigroup_diffusion_solver_patched_lmfgk import MultigroupDiffusionSolver1D

print("USING SOLVER FILE:", m.__file__)
print("Testing LMFGK Preconditioner - GRAY LIMIT (all groups identical)")
print("="*70)

# Problem setup
n_groups = 10
n_cells = 30
r_min = 0.0
r_max = 5.0  # cm
dt = 0.5     # ns (bigger dt makes coupling stiffer / more meaningful)
geometry = 'planar'

# Energy edges (keV) - still needed for group structure, but unused by sigma in this test
energy_edges = np.array([0.0001,0.000316, 0.001, 0.00316, 0.01, 0.0316, 0.1, 0.316, 1.0, 3.16, 10.0])

# Material properties
rho = 1.0  # g/cm³
cv  = 0.1  # GJ/(g·keV)

# --- GRAY LIMIT: same opacity in every group ---
# Make sigma depend on (T,r) only, NOT on group.
def sigma_gray(T, r):
    # spatial jump to force a real diffusion interface problem
    # (still gray because every group sees same sigma)
    if r < 2.5:
        return 10.0 #/ np.sqrt(T)
    else:
        return 10.0 #/ np.sqrt(T)

sigma_funcs = []
diff_funcs  = []
for g in range(n_groups):
    sigma_funcs.append(lambda T, r, g=g: sigma_gray(T, r))
    diff_funcs.append(lambda T, r, g=g: 1.0 / (3.0 * sigma_gray(T, r)))

print(f"Problem setup:")
print(f"  Groups: {n_groups}")
print(f"  Cells: {n_cells}")
print(f"  Domain: [{r_min}, {r_max}] cm")
print(f"  dt: {dt} ns")
print(f"  Geometry: {geometry}")
print(f"  Boundary conditions: Neumann (zero flux) on both sides")
print()

# Boundary conditions: Neumann
def neumann_bc(phi, r):
    return 0.0, 1.0, 0.0
left_bc_funcs  = [neumann_bc] * n_groups
right_bc_funcs = [neumann_bc] * n_groups

# Emission fractions: equal
chi = np.ones(n_groups) / n_groups

# Initialize solver
solver = MultigroupDiffusionSolver1D(
    n_groups=n_groups,
    r_min=r_min,
    r_max=r_max,
    n_cells=n_cells,
    energy_edges=energy_edges,
    geometry=geometry,
    dt=dt,
    diffusion_coeff_funcs=diff_funcs,
    absorption_coeff_funcs=sigma_funcs,
    left_bc_funcs=left_bc_funcs,
    right_bc_funcs=right_bc_funcs,
    emission_fractions=chi,
    rho=rho,
    cv=cv
)

# Initial temperature profile: smooth exponential
T_cold = 0.1  # keV
T_hot  = 0.5  # keV
L      = 2.0
r_centers = solver.r_centers
T_init = T_cold + (T_hot - T_cold) * np.exp(-((r_centers-L/2)/L)**2)

print("Initial temperature profile:")
print(f"  T(r=0) = {T_init[0]:.4f} keV")
print(f"  T(r={r_max}) = {T_init[-1]:.4f} keV")
print()

def reset_state():
    solver.T      = T_init.copy()
    solver.T_old  = T_init.copy()
    solver.E_r    = A_RAD * T_init**4
    solver.E_r_old= solver.E_r.copy()
    solver.kappa  = np.zeros(n_cells)
    solver.kappa_old = np.zeros(n_cells)

# --- Run WITHOUT preconditioner ---
reset_state()
print("="*70)
print("Test 1: WITHOUT preconditioner")
print("="*70)
info1 = solver.step(
    max_newton_iter=1,
    newton_tol=1e-10,
    gmres_tol=1e-10,
    gmres_maxiter=2000,
    use_preconditioner=False,
    verbose=False   # set True if you want full logs
)
gmres_iter_no_prec = info1['gmres_info']['iterations']

print(f"\n{'='*70}")
print("Result WITHOUT preconditioner:")
print(f"{'='*70}")
print(f"  Converged: {info1['converged']}")
print(f"  Newton iterations: {info1['newton_iter']}")
print(f"  GMRES iterations (last Newton): {gmres_iter_no_prec}")
print(f"  Final T_max: {solver.T.max():.6f} keV")
print(f"  Final T_min: {solver.T.min():.6f} keV")
print()

# --- Run WITH LMFGK preconditioner ---
reset_state()
print("="*70)
print("Test 2: WITH LMFGK preconditioner (gray limit)")
print("="*70)
info2 = solver.step(
    max_newton_iter=1,
    newton_tol=1e-10,
    gmres_tol=1e-10,
    gmres_maxiter=200,
    use_preconditioner=True,
    verbose=True
)
gmres_iter_with_prec = info2['gmres_info']['iterations']

print(f"\n{'='*70}")
print("Result WITH preconditioner:")
print(f"{'='*70}")
print(f"  Converged: {info2['converged']}")
print(f"  Newton iterations: {info2['newton_iter']}")
print(f"  GMRES iterations (last Newton): {gmres_iter_with_prec}")
print(f"  Final T_max: {solver.T.max():.6f} keV")
print(f"  Final T_min: {solver.T.min():.6f} keV")
print()

# Compare
print("="*70)
print("Comparison (GRAY LIMIT):")
print("="*70)
print(f"  Without preconditioner: {gmres_iter_no_prec}")
print(f"  With preconditioner:    {gmres_iter_with_prec}")
reduction = 100.0 * (gmres_iter_no_prec - gmres_iter_with_prec) / max(gmres_iter_no_prec, 1)
print(f"  → {reduction:.1f}% reduction")
print()

# A “sanity expectation” assertion (soft)
if gmres_iter_no_prec >= 8 and gmres_iter_with_prec > 5:
    print("WARNING: In the gray limit, LMFGK usually gives a much bigger iteration reduction.")
    print("         If you see little/no benefit here, it’s a red flag to re-check C/H/m(y).")
else:
    print("Looks consistent: LMFGK is helping strongly in the gray-limit case.")

print("="*70)
print("Test completed!")
print("="*70)