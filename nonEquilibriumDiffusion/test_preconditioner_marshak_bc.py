#!/usr/bin/env python3
"""
Test LMFG preconditioner on a multigroup problem with a Marshak BC.

Same opacity structure as test_preconditioner_multigroup.py but the left
boundary uses a Robin (Marshak) condition with C != 0 (incoming blackbody
flux at T_bc = 0.5 keV), and the right boundary is zero-flux Neumann.
"""

import sys
import numpy as np
sys.path.insert(0, '/Users/ryanmcclarren/Dropbox/Papers/RadTranBook/nonEquilibriumDiffusion')

from multigroup_diffusion_solver import (
    MultigroupDiffusionSolver1D, flux_limiter_larsen, Bg_multigroup
)
from diffusion_operator_solver import C_LIGHT, A_RAD
import multigroup_diffusion_solver as m
print("USING SOLVER FILE:", m.__file__)

print("Testing LMFG Preconditioner - 10 Groups, Marshak BC")
print("="*70)

# Problem setup
n_groups = 10
n_cells = 30
r_min = 0.0
r_max = 5.0   # cm
dt = 0.01      # ns
geometry = 'planar'

# Energy edges (keV) - 10 groups
energy_edges = np.array([0.00001, 0.000316, 0.001, 0.00316, 0.01,
                          0.0316,  0.1,     0.316, 1.0,     3.16, 10.0])

# Material properties
rho = 1.0   # g/cm³
cv  = 0.1   # GJ/(g·keV)

# Power-law opacity: σ_a(T, E) ∝ T^{-1/2} E^{-3}, with a heterogeneous domain
def sigma_func(T, r, E_low, E_high):
    E_mid = 2.0 * E_high * E_low / (E_high + E_low)   # harmonic mean energy
    base = 10.0 / np.sqrt(max(T, 1e-3))
    scale = 1.0 if r < 2.5 else 10.0
    return scale * base * E_mid**(-3.0)

sigma_funcs = []
diff_funcs  = []
for g in range(n_groups):
    sigma_funcs.append(lambda T, r, g=g: sigma_func(T, r, energy_edges[g], energy_edges[g+1]))
    diff_funcs.append( lambda T, r, g=g: C_LIGHT / (3.0 * sigma_func(T, r, energy_edges[g], energy_edges[g+1])))

# -------------------------------------------------------------------------
# Marshak BC on the left: ½φ_g + 2D_g dφ_g/dr = F_in,g
#   F_in,g = χ_g * (a·c·T_bc^4)/2
#   D_g evaluated at T_bc (cold medium, so use a representative temperature)
# -------------------------------------------------------------------------
T_bc = 0.5   # keV — boundary drive temperature

B_g_bc = Bg_multigroup(energy_edges, T_bc)
chi_bc  = B_g_bc / B_g_bc.sum()
F_total = (A_RAD * C_LIGHT * T_bc**4) / 2.0

print(f"Boundary drive: T_bc = {T_bc} keV, F_total = {F_total:.4e} GJ/(cm²·ns)")

left_bc_funcs  = []
right_bc_funcs = []
for g in range(n_groups):
    D_g_bc = C_LIGHT / (3.0 * sigma_func(T_bc, 0.0, energy_edges[g], energy_edges[g+1]))
    F_g    = chi_bc[g] * F_total
    # Robin: A=0.5, B=2D_g, C=F_in,g  (Marshak)
    def make_left(D=D_g_bc, F=F_g):
        def bc(phi, r):
            return 0.5, 2.0 * D, F
        return bc
    left_bc_funcs.append(make_left())

    def neumann_bc(phi, r):
        return 0.0, 1.0, 0.0
    right_bc_funcs.append(neumann_bc)

print(f"Group emission fractions at T_bc:")
for g in range(n_groups):
    print(f"  Group {g:2d} [{energy_edges[g]:.4f}, {energy_edges[g+1]:.4f}] keV: "
          f"χ = {chi_bc[g]:.5f}")

# -------------------------------------------------------------------------
# Solver initialisation
# -------------------------------------------------------------------------
solver = MultigroupDiffusionSolver1D(
    n_groups=n_groups,
    r_min=r_min,
    r_max=r_max,
    n_cells=n_cells,
    energy_edges=energy_edges,
    geometry=geometry,
    dt=dt,
    flux_limiter_funcs=flux_limiter_larsen,
    diffusion_coeff_funcs=diff_funcs,
    absorption_coeff_funcs=sigma_funcs,
    left_bc_funcs=left_bc_funcs,
    right_bc_funcs=right_bc_funcs,
    rho=rho,
    cv=cv
)

# Initial condition: cold everywhere
T_cold = 0.05   # keV
r_centers = solver.r_centers
T_init = T_cold * np.ones(n_cells)

solver.T       = T_init.copy()
solver.T_old   = T_init.copy()
solver.E_r     = A_RAD * T_init**4
solver.E_r_old = solver.E_r.copy()
solver.kappa     = np.zeros(n_cells)
solver.kappa_old = np.zeros(n_cells)

print()
print(f"Initial conditions: T = {T_cold} keV (cold)")
print(f"Boundary conditions: Marshak (T_bc={T_bc} keV) left | zero-flux right")
print()

# -------------------------------------------------------------------------
# Run WITHOUT preconditioner
# -------------------------------------------------------------------------
print("="*70)
print("Test 1: WITHOUT preconditioner")
print("="*70)
info1 = solver.step(
    max_newton_iter=1,
    newton_tol=1e-8,
    gmres_tol=1e-12,
    gmres_maxiter=300,
    use_preconditioner=False,
    verbose=True
)

gmres_iter_no_prec = info1['gmres_info']['iterations']
print(f"\nResult WITHOUT preconditioner:")
print(f"  Converged: {info1['converged']}")
print(f"  GMRES iterations: {gmres_iter_no_prec}")
print(f"  Final T_max: {solver.T.max():.6f} keV")

# -------------------------------------------------------------------------
# Reset and run WITH preconditioner
# -------------------------------------------------------------------------
solver.T       = T_init.copy()
solver.T_old   = T_init.copy()
solver.E_r     = A_RAD * T_init**4
solver.E_r_old = solver.E_r.copy()
solver.kappa     = np.zeros(n_cells)
solver.kappa_old = np.zeros(n_cells)

print()
print("="*70)
print("Test 2: WITH LMFG preconditioner")
print("="*70)
info2 = solver.step(
    max_newton_iter=1,
    newton_tol=1e-8,
    gmres_tol=1e-12,
    gmres_maxiter=300,
    use_preconditioner=True,
    verbose=True
)

gmres_iter_with_prec = info2['gmres_info']['iterations']
print(f"\nResult WITH preconditioner:")
print(f"  Converged: {info2['converged']}")
print(f"  GMRES iterations: {gmres_iter_with_prec}")
print(f"  Final T_max: {solver.T.max():.6f} keV")

# -------------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------------
print()
print("="*70)
print("Comparison:")
print("="*70)
print(f"  GMRES iterations without preconditioner: {gmres_iter_no_prec}")
print(f"  GMRES iterations with    preconditioner: {gmres_iter_with_prec}")
reduction = 100.0 * (gmres_iter_no_prec - gmres_iter_with_prec) / max(gmres_iter_no_prec, 1)
print(f"  → {reduction:.1f}% reduction")
print("="*70)
print("Test completed!")
