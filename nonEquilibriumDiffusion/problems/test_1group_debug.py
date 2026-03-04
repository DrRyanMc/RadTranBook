#!/usr/bin/env python3
"""
Debug 1-group system - print everything at each step.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multigroup_diffusion_solver import MultigroupDiffusionSolver1D

C_LIGHT = 2.99792458e1  # cm/ns
A_RAD = 0.01372  # GJ/(cm³·keV⁴)

# Setup
n_groups = 1
sigma_a = 5.0
C_v = 0.01
rho = 1.0
energy_edges = np.array([0.01, 10.0])

solver = MultigroupDiffusionSolver1D(
    n_groups=n_groups,
    r_min=0.0,
    r_max=1.0,
    n_cells=1,
    energy_edges=energy_edges,
    geometry='planar',
    dt=0.01,
    diffusion_coeff_funcs=[lambda T, r: 1e10],
    absorption_coeff_funcs=[lambda T, r: sigma_a],
    left_bc='neumann',
    right_bc='neumann',
    left_bc_values=[0.0],
    right_bc_values=[0.0],
    rho=rho,
    cv=C_v
)

# Initial conditions
T_init = 0.4
T_rad_init = 1.0
solver.T = np.array([T_init])
solver.E_r = np.array([A_RAD * T_rad_init**4])

print("Initial state:")
print(f"  T = {solver.T[0]:.6f} keV")
print(f"  E_r = {solver.E_r[0]:.6e} GJ/cm³")
print(f"  e_mat = {rho * C_v * solver.T[0]:.6e} GJ/cm³")
print(f"  E_total = {solver.E_r[0] + rho * C_v * solver.T[0]:.6e} GJ/cm³")
print(f"  φ = {solver.E_r[0] * C_LIGHT:.6e}")

# Store old values for source term
solver.T_old = solver.T.copy()
solver.E_r_old = solver.E_r.copy()
solver.phi_g_fraction = np.array([[1.0]])  # 1 group, fraction = 1

# First Newton iteration
print("\n" + "="*80)
print("First Newton iteration")
print("="*80)

T_star = solver.T_old.copy()
print(f"\nT_★ (initial guess) = {T_star[0]:.6f} keV")

# Update absorption coefficients
solver.update_absorption_coefficients(T_star)
print(f"σ_a = {solver.sigma_a[0, 0]:.6f} cm^-1")

# Computer Fleck factor
solver.fleck_factor = solver.compute_fleck_factor(T_star)
f = solver.fleck_factor[0]
print(f"Fleck factor f = {f:.6f}")

# Compute source term ξ_0
xi_g_list = [solver.compute_source_xi(g, T_star) for g in range(n_groups)]
xi_0 = xi_g_list[0][0]

print(f"\nSource term components:")
# Decompose ξ_0
phi_0_old = solver.E_r_old[0] * C_LIGHT * solver.phi_g_fraction[0, 0]
print(f"  φ_0^n = {phi_0_old:.6e}")
print(f"  (1/cΔt)φ_0^n = {phi_0_old / (C_LIGHT * solver.dt):.6e}")

from planck_integrals import Bg
B_0_star = Bg(energy_edges[0], energy_edges[1], T_star[0])
print(f"  B_0(T_★) = {B_0_star:.6e}")
print(f"  4π·σ_a·B_0(T_★) = {4 * np.pi * sigma_a * B_0_star:.6e}")

e_star = rho * C_v * T_star[0]
e_n = rho * C_v * solver.T_old[0]
Delta_e = e_star - e_n
print(f"  e(T_★) = {e_star:.6e}, e(T^n) = {e_n:.6e}")
print(f"  Δe = {Delta_e:.6e}, Δe/Δt = {Delta_e / solver.dt:.6e}")

chi_0 = 1.0
coupling = 4 * np.pi * sigma_a * B_0_star - Delta_e / solver.dt
print(f"  Coupling term = 4π·σ_a·B_0 - Δe/Δt = {coupling:.6e}")
print(f"  χ_0·(1-f)·coupling = {chi_0 * (1 - f) * coupling:.6e}")

print(f"\nξ_0 = {xi_0:.6e}")

# Now solve for κ
print("\n" + "-"*80)
print("Solving B·κ = RHS for κ...")

# RHS = Σ_g σ*_{a,g}·A_g^{-1}·ξ_g
# For 1 group: RHS = σ_a · A_0^{-1} ξ_0
# We need to solve A_0 φ = ξ_0, then RHS = σ_a φ

phi_from_xi = solver.solvers[0].solve(xi_g_list[0], T_star)
RHS_kappa = sigma_a * phi_from_xi[0]
print(f"  Solved A_0 φ = ξ_0 → φ = {phi_from_xi[0]:.6e}")
print(f"  RHS = σ_a · φ = {RHS_kappa:.6e}")

# Now solve B·κ = RHS
# B = I - σ_a·A_0^{-1}·χ_0·(1-f)
# For single cell, this is a scalar: B·κ = κ - σ_a·φ_from_test·(1-f) where φ_from_test = A_0^{-1}·κ
# This is implicit, so we use GMRES

kappa, gmres_info = solver.solve_for_kappa(T_star, xi_g_list, gmres_tol=1e-6, gmres_maxiter=200, verbose=True)
print(f"  κ (absorption rate) = {kappa[0]:.6e}")
print(f"  GMRES converged: {gmres_info.get('converged', '?')}, iterations: {gmres_info.get('iterations', '?')}")

# Now compute φ_0^{n+1} = A_0^{-1}(χ_0·(1-f)·κ + ξ_0)
rhs_phi = chi_0 * (1 - f) * kappa + xi_g_list[0]
phi_new = solver.solvers[0].solve(rhs_phi, T_star)
E_r_new = phi_new[0] / C_LIGHT

print(f"\nFinal φ_0^{{n+1}} = {phi_new[0]:.6e}")
print(f"E_r^{{n+1}} = φ/c = {E_r_new:.6e} GJ/cm³")

# Update temperature
T_new = solver.update_temperature(kappa, T_star)
e_new = rho * C_v * T_new[0]
print(f"T^{{n+1}} = {T_new[0]:.6f} keV")
print(f"e^{{n+1}} = {e_new:.6e} GJ/cm³")

E_total_new = E_r_new + e_new
E_total_old = solver.E_r_old[0] + rho * C_v * solver.T_old[0]
print(f"\nEnergy budget:")
print(f"  E_total^n = {E_total_old:.6e}")
print(f"  E_total^{{n+1}} = {E_total_new:.6e}")
print(f"  ΔE = {E_total_new - E_total_old:.6e}")
print(f"  ΔE/E_0 = {(E_total_new - E_total_old) / E_total_old:.6f}")
