#!/usr/bin/env python3
"""Quick homogeneity test"""

import sys
import numpy as np
from multigroup_diffusion_solver_2d import MultigroupDiffusionSolver2D, C_LIGHT, A_RAD, Bg_multigroup

RHO = 10.0

def powerlaw_opacity_at_energy(T, E, rho=1.0):
    T_safe = 1e-2
    T_use = np.maximum(T, T_safe)
    return np.minimum(100000.0 * rho * (T_use)**(-0.5) * E**(-3.0), 1e14)

def make_powerlaw_opacity_func(E_low, E_high, rho=1.0):
    def opacity_func(T, x, y):
        sigma_low = powerlaw_opacity_at_energy(T, E_low, rho)
        sigma_high = powerlaw_opacity_at_energy(T, E_high, rho)
        return np.sqrt(sigma_low * sigma_high)
    return opacity_func

def make_powerlaw_diffusion_func(E_low, E_high, rho=1.0):
    opacity_func = make_powerlaw_opacity_func(E_low, E_high, rho)
    def diffusion_func(T, x, y):
        sigma = opacity_func(T, x, y)
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

# Setup
print("Setting up solver...")
n_groups = 10
energy_edges = np.logspace(np.log10(1e-4), np.log10(25.0), n_groups + 1)

sigma_funcs = []
diff_funcs = []
for g in range(n_groups):
    E_low = energy_edges[g]
    E_high = energy_edges[g+1]
    sigma_funcs.append(make_powerlaw_opacity_func(E_low, E_high, RHO))
    diff_funcs.append(make_powerlaw_diffusion_func(E_low, E_high, RHO))

T_bc = 0.05
left_bc_funcs = []
for g in range(n_groups):
    E_low = energy_edges[g]
    E_high = energy_edges[g+1]
    left_bc_funcs.append(make_marshak_bc_func(T_bc, E_low, E_high, RHO))

boundary_funcs = {
    'left': left_bc_funcs,
    'right': [bc_zero_flux] * n_groups,
    'bottom': [bc_zero_flux] * n_groups,
    'top': [bc_zero_flux] * n_groups
}

solver = MultigroupDiffusionSolver2D(
    n_groups=n_groups,
    x_min=0.0, x_max=6.0, nx_cells=50,
    y_min=0.0, y_max=0.1, ny_cells=3,
    energy_edges=energy_edges,
    geometry='cartesian',
    dt=0.001,
    diffusion_coeff_funcs=diff_funcs,
    absorption_coeff_funcs=sigma_funcs,
    boundary_funcs=boundary_funcs,
    rho=RHO,
    cv=0.05,
    max_newton_iter=5,
    newton_tol=1e-6,
    theta=1.0
)

# Initialize
T_init = 0.05
E_r_init = A_RAD * T_init**4
B_g_init = Bg_multigroup(energy_edges, T_init)
chi_init = B_g_init / B_g_init.sum()
E_r_groups_init = chi_init * E_r_init

solver.T = np.full(solver.n_total, T_init)
solver.T_old = solver.T.copy()
solver.E_r = np.full(solver.n_total, E_r_init)
solver.E_r_old = solver.E_r.copy()

for g in range(n_groups):
    solver.phi_g_stored[g, :] = E_r_groups_init[g] * C_LIGHT

solver.t = 0.0

# Test - just run solver step with verbose on
print("\nRunning one step with verbose=True...")
print("="*80)
info = solver.step(verbose=True, gmres_tol=1e-6, gmres_maxiter=200, use_preconditioner=False)
print("="*80)
print(f"\nResult: Newton converged={info.get('converged', False)}, iterations={info['newton_iterations']}")
