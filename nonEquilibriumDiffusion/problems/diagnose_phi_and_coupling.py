"""
Diagnose φ_g and material-radiation coupling in steady state.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from multigroup_diffusion_solver import MultigroupDiffusionSolver1D, C_LIGHT, A_RAD, Bg_multigroup

# Physical constants
PLANCK_CONSTANT = 6.62607015e-34
BOLTZMANN_CONSTANT = 1.380649e-23

# Setup matching test_cold_to_hot_incoming_bc.py
r_min, r_max = 0.0, 1.0e-3
n_cells = 1
dt = 0.001
rho, cv = 1.0, 0.05
sigma = 100.0
T_bc = 1.0

# Energy groups (5 groups, log-spaced)
energy_edges = np.logspace(np.log10(1e-4), np.log10(5.0), 6)
n_groups = len(energy_edges) - 1

# Define functions
def D_func(T, r=0):
    return C_LIGHT / (3.0 * sigma)

def sigma_func(T, r=0):
    return sigma

diff_funcs = [D_func] * n_groups
sigma_funcs = [sigma_func] * n_groups

# Boundary conditions
B_g_bc = Bg_multigroup(energy_edges, T_bc)
chi_bc = B_g_bc / B_g_bc.sum()
F_total_bc = (A_RAD * C_LIGHT * T_bc**4) / 2.0
F_g_bc = chi_bc * F_total_bc

def make_incoming_bc(g):
    D = D_func(T_bc)
    F_inc_g = F_g_bc[g]
    def bc_incoming(phi, r):
        A, B, C = 0.5, 2.0 * D, F_inc_g
        return A, B, C
    return bc_incoming

def bc_reflecting(phi, r):
    return 0.0, 1.0, 0.0

left_bcs = [make_incoming_bc(g) for g in range(n_groups)]
right_bcs = [bc_reflecting] * n_groups

# Create solver
T_cold = 0.01
B_g_cold = Bg_multigroup(energy_edges, T_cold)
chi_cold = B_g_cold / B_g_cold.sum()

solver = MultigroupDiffusionSolver1D(
    n_groups=n_groups, r_min=r_min, r_max=r_max, n_cells=n_cells,
    energy_edges=energy_edges, geometry='planar', dt=dt,
    diffusion_coeff_funcs=diff_funcs, absorption_coeff_funcs=sigma_funcs,
    left_bc_funcs=left_bcs, right_bc_funcs=right_bcs,
    emission_fractions=chi_cold, rho=rho, cv=cv
)

# Initialize cold
E_r_cold = A_RAD * T_cold**4
solver.T[:] = T_cold
solver.T_old[:] = T_cold
solver.E_r[:] = E_r_cold
solver.E_r_old[:] = E_r_cold
solver.phi_g_fraction[:, :] = chi_cold[:, np.newaxis]
phi_g_cold = 4.0 * np.pi * B_g_cold
solver.phi_g_stored[:, :] = phi_g_cold[:, np.newaxis]

print("Running to steady state...")
for step in range(100):
    solver.step()
    solver.advance_time()

# Extract steady state values
T_mat_ss = solver.T[0]
E_r_ss = solver.E_r[0]
T_rad_ss = (E_r_ss / A_RAD) ** 0.25
kappa_ss = solver.kappa[0]

print("\n" + "="*80)
print("STEADY STATE DIAGNOSIS")
print("="*80)
print(f"\nTemperatures:")
print(f"  T_mat = {T_mat_ss:.6f} keV")
print(f"  T_rad = {T_rad_ss:.6f} keV")
print(f"  T_bc  = {T_bc:.6f} keV")
print(f"  Difference: |T_mat - T_rad| = {abs(T_mat_ss - T_rad_ss):.6f} keV ({abs(T_mat_ss - T_rad_ss)/T_rad_ss*100:.2f}%)")

print(f"\nRadiation energy:")
print(f"  E_r (steady) = {E_r_ss:.6e} GJ/cm³")
print(f"  E_r (if T_mat equilibrium) = {A_RAD * T_mat_ss**4:.6e} GJ/cm³")

print(f"\nAbsorption rate κ:")
print(f"  κ = {kappa_ss:.6e} GJ/(cm³·ns)")

# Compute φ_g for each group
print(f"\nGroup-by-group analysis:")
print(f"{'Grp':>3} | {'φ_g':>12} | {'E_r,g':>12} | {'σ·φ_g':>12} | {'σ·4π·B_g(T_mat)':>16} | {'Difference':>12}")
print("-"*90)

phi_g_ss = solver.phi_g_stored[:, 0]
sigma_phi_sum = 0.0
sigma_B_sum = 0.0

for g in range(n_groups):
    phi_g = phi_g_ss[g]
    E_r_g = phi_g / C_LIGHT
    sigma_phi = sigma * phi_g
    
    # Compute Planck function at T_mat
    B_g_mat = solver.planck_funcs[g](T_mat_ss)
    sigma_B = sigma * 4.0 * np.pi * B_g_mat
    
    diff = sigma_phi - sigma_B
    
    print(f"{g:3d} | {phi_g:12.6e} | {E_r_g:12.6e} | {sigma_phi:12.6e} | {sigma_B:16.6e} | {diff:12.6e}")
    
    sigma_phi_sum += sigma_phi
    sigma_B_sum += sigma_B

print("-"*90)
print(f"{'SUM':>3} | {'':12} | {E_r_ss:12.6e} | {sigma_phi_sum:12.6e} | {sigma_B_sum:16.6e} | {sigma_phi_sum - sigma_B_sum:12.6e}")

print(f"\nMaterial-radiation coupling term:")
print(f"  κ = Σ_g σ_g·φ_g = {kappa_ss:.6e} GJ/(cm³·ns)")
print(f"  Emission = Σ_g σ_g·4π·B_g(T_mat) = {sigma_B_sum:.6e} GJ/(cm³·ns)")
print(f"  Net = κ - emission = {kappa_ss - sigma_B_sum:.6e} GJ/(cm³·ns)")

# What the coupling SHOULD be for equilibrium
E_r_eq = A_RAD * T_mat_ss**4
coupling_should_be = sigma * C_LIGHT * (E_r_ss - E_r_eq)
print(f"\nFor equilibrium, coupling should be:")
print(f"  σ·c·(E_r - a·T_mat⁴) = {coupling_should_be:.6e} GJ/(cm³·ns)")
print(f"  This would cool material at: dT/dt = {coupling_should_be / (cv * rho):.3e} keV/ns")

# Check the Fleck factor
f = solver.fleck_factor[0]
print(f"\nFleck factor:")
print(f"  f = {f:.6e}")
print(f"  1-f = {1-f:.6e}")

# Energy update per time step
energy_change = kappa_ss - sigma_B_sum
dE_mat = dt * f * energy_change
print(f"\nEnergy change per time step:")
print(f"  dt·f·(κ - emission) = {dE_mat:.6e} GJ/cm³")
print(f"  This changes T_mat by: ΔT ≈ {dE_mat / (cv * rho):.6e} keV")

print(f"\n" + "="*80)
print("CONCLUSION")
print("="*80)
if abs(energy_change) < 1e-8:
    print("✓ κ ≈ emission: The solver has found a self-consistent solution")
    print("  where absorption equals emission.")
    print(f"\n❌ BUT this is NOT physical equilibrium!")
    print(f"  Physical equilibrium requires E_r = a·T_mat⁴")
    print(f"  Current: E_r / (a·T_mat⁴) = {E_r_ss / (A_RAD * T_mat_ss**4):.6f}")
    print(f"\n🔍 The bug is that φ_g is being computed from the diffusion equation")
    print(f"  using T_mat as the material temperature, which creates a spurious")
    print(f"  fixed point where absorption and emission balance, but")
    print(f"  E_r ≠ a·T_mat⁴")
else:
    print(f"⚠ κ - emission = {energy_change:.3e} ≠ 0")
    print(f"  System is not at a fixed point yet.")
