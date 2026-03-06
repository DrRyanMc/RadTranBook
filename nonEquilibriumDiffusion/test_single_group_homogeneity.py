#!/usr/bin/env python3
"""Test single diffusion solver homogeneity with exact Marshak BC from multigroup test"""

import numpy as np
from diffusion_operator_solver import DiffusionOperatorSolver2D, C_LIGHT, A_RAD

# Import the Planck function
try:
    from planck_integrals import Bg_multigroup
except:
    print("Warning: planck_integrals not available, using approximation")
    def Bg_multigroup(edges, T):
        return A_RAD * C_LIGHT * T**4 * np.ones(len(edges)-1) / (len(edges)-1)

RHO = 10.0

def powerlaw_opacity_at_energy(T, E, rho=1.0):
    T_safe = 1e-2
    T_use = np.maximum(T, T_safe)
    return np.minimum(100000.0 * rho * (T_use)**(-0.5) * E**(-3.0), 1e14)

def make_powerlaw_diffusion_func(E_low, E_high, rho=1.0):
    def diffusion_func(T, x, y):
        sigma_low = powerlaw_opacity_at_energy(T, E_low, rho)
        sigma_high = powerlaw_opacity_at_energy(T, E_high, rho)
        sigma = np.sqrt(sigma_low * sigma_high)
        return C_LIGHT / (3.0 * sigma)
    return diffusion_func

def make_marshak_bc_func(T_bc, E_low, E_high, rho=1.0):
    D_func = make_powerlaw_diffusion_func(E_low, E_high, rho)
    def marshak_bc(phi, pos, t):
        x, y = pos
        D = D_func(T_bc, x, y)
        B_g = Bg_multigroup(np.array([E_low, E_high]), T_bc)[0]
        F_in = C_LIGHT * B_g
        return 0.5, 2.0 * D, F_in
    return marshak_bc

def bc_zero_flux(phi, pos, t):
    return 0.0, 1.0, 0.0

# Setup - single group
E_low, E_high = 0.05, 0.173  # Middle energy group
T_bc = 0.05

diff_func = make_powerlaw_diffusion_func(E_low, E_high, RHO)
left_bc_func = make_marshak_bc_func(T_bc, E_low, E_high, RHO)

solver = DiffusionOperatorSolver2D(
    x_min=0.0, x_max=6.0, nx_cells=50,
    y_min=0.0, y_max=0.1, ny_cells=3,
    geometry='cartesian',
    diffusion_coeff_func=diff_func,
    absorption_coeff_func=lambda T, x, y: 1.0,
    dt=0.001,
    left_bc_func=left_bc_func,
    right_bc_func=bc_zero_flux,
    bottom_bc_func=bc_zero_flux,
    top_bc_func=bc_zero_flux
)

n_total = solver.n_total
T_uniform = np.full(n_total, T_bc)

# Test homogeneity exactly as multigroup solver does
print("Single Group Homogeneity Test (mimicking multigroup)")
print("="*70)

test_x = np.random.randn(n_total)
alpha = 3.7

# Compute A^{-1}(test_x)
T_2d = test_x.reshape(solver.nx_cells, solver.ny_cells)
rhs_x = test_x.reshape(solver.nx_cells, solver.ny_cells)
T_uniform_2d = T_uniform.reshape(solver.nx_cells, solver.ny_cells)

phi_x_2d = solver.solve(rhs_x, T_uniform_2d)
phi_x = phi_x_2d.flatten()

# Compute A^{-1}(alpha * test_x)
rhs_ax = (alpha * test_x).reshape(solver.nx_cells, solver.ny_cells)
phi_ax_2d = solver.solve(rhs_ax, T_uniform_2d)
phi_ax = phi_ax_2d.flatten()

# Check homogeneity: A^{-1}(αx) vs α·A^{-1}(x)
alpha_phi_x = alpha * phi_x

diff = np.linalg.norm(phi_ax - alpha_phi_x)
norm_ax = np.linalg.norm(phi_ax)
norm_alpha_x = np.linalg.norm(alpha_phi_x)

print(f"||A^{{-1}}(αx)||     = {norm_ax:.6e}")
print(f"||α·A^{{-1}}(x)||   = {norm_alpha_x:.6e}")
print(f"||Difference||      = {diff:.6e}")
print(f"Relative error      = {diff/norm_ax:.6e}")

if diff / norm_ax < 1e-12:
    print("\n✓ Perfect homogeneity achieved!")
else:
    print(f"\n✗ Homogeneity error: {diff/norm_ax:.3e}")
    print("\nDebugging info:")
    print(f"  alpha = {alpha}")
    print(f"  n_total = {n_total}")
    print(f"  T = {T_bc} (uniform)")
    
    # Check boundary conditions
    print("\nBoundary condition check:")
    A_bc, B_bc, C_bc = left_bc_func(0.0, (solver.x_faces[0], solver.y_centers[0]), 0.0)
    print(f"  Left BC: A={A_bc}, B={B_bc:.6e}, C={C_bc:.6e}")
    
    # Check if problem is with caching
    print("\nClearing cache and re-testing...")
    solver._cached_T = None
    solver._cached_A = None
    
    phi_ax_2d_nocache = solver.solve(rhs_ax, T_uniform_2d)
    phi_ax_nocache = phi_ax_2d_nocache.flatten()
    
    diff_nocache = np.linalg.norm(phi_ax_nocache - alpha_phi_x)
    print(f"  After clearing cache: ||difference|| = {diff_nocache:.6e}")
    print(f"  Relative error = {diff_nocache/norm_ax:.6e}")
