#!/usr/bin/env python3
"""
Test that incoming BB at T_eq MAINTAINS equilibrium when starting at equilibrium.

Start system at: T_mat = T_rad = T_eq
Apply incoming BB at T_eq
Result should stay at: T_mat = T_rad = T_eq
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from multigroup_diffusion_solver import MultigroupDiffusionSolver1D, C_LIGHT, A_RAD

# Setup
r_min, r_max = 0.0, 1.0e-3
n_cells = 1
dt = 100.0  # Large timestep
rho, cv = 1.0, 0.05

n_groups = 5
energy_edges = np.logspace(np.log10(1e-4), np.log10(5.0), n_groups + 1)
sigma_values = [100.0] * n_groups

sigma_funcs = []
diff_funcs = []
for g in range(n_groups):
    sig = sigma_values[g]
    sigma_funcs.append(lambda T, r, s=sig: s)
    diff_funcs.append(lambda T, r, s=sig: C_LIGHT / (3.0 * s))

from planck_integrals import Bg_multigroup

# Start at equilibrium temperature
T_eq = 0.025
E_r_eq = A_RAD * T_eq**4

print("="*80)
print("TEST: INCOMING BB MAINTAINS EQUILIBRIUM")
print("="*80)
print(f"Starting at equilibrium: T_eq = {T_eq:.10f} keV")
print(f"                         E_r_eq = {E_r_eq:.15e} GJ/cm³")
print()

# Compute incoming blackbody flux for each group at T_eq
# Method 1 (Marshak approach): F_g = chi_g * (a*c*T^4)/2
B_g_eq = Bg_multigroup(energy_edges, T_eq)
chi_eq = B_g_eq / B_g_eq.sum()
F_total = (A_RAD * C_LIGHT * T_eq**4) / 2.0
F_inc_g = chi_eq * F_total

print(f"Incoming flux at T_eq:")
print(f"  F_total = {F_total:.6e} GJ/(cm²·ns)")
for g in range(n_groups):
    print(f"  Group {g}: F_g = {F_inc_g[g]:.6e}, fraction = {chi_eq[g]*100:.2f}%")
print()

# ============================================================================
# TEST 1: REFLECTING BCs (baseline)
# ============================================================================
print("-"*80)
print("TEST 1: Reflecting BCs (should maintain equilibrium)")
print("-"*80)

B_g = Bg_multigroup(energy_edges, T_eq)
chi = B_g / B_g.sum()

def bc_reflecting(phi, r):
    """Reflecting BC: dφ/dr = 0 at boundary"""
    return 0.0, 1.0, 0.0

left_bcs_reflecting = [bc_reflecting] * n_groups
right_bcs_reflecting = [bc_reflecting] * n_groups

solver1 = MultigroupDiffusionSolver1D(
    n_groups=n_groups, r_min=r_min, r_max=r_max, n_cells=n_cells,
    energy_edges=energy_edges, geometry='planar', dt=dt,
    diffusion_coeff_funcs=diff_funcs, absorption_coeff_funcs=sigma_funcs,
    left_bc_funcs=left_bcs_reflecting, right_bc_funcs=right_bcs_reflecting,
    emission_fractions=chi, rho=rho, cv=cv
)
solver1._debug_update_T = False

# Initialize at equilibrium
solver1.T[:] = T_eq
solver1.T_old[:] = T_eq
solver1.E_r[:] = E_r_eq
solver1.E_r_old[:] = solver1.E_r.copy()
solver1.phi_g_fraction[:, :] = chi[:, np.newaxis]
# Correct: φ_g = 4π * B_g for isotropic blackbody
phi_g_eq = 4.0 * np.pi * B_g
solver1.phi_g_stored[:, :] = phi_g_eq[:, np.newaxis]

E_total_init = cv * rho * T_eq + E_r_eq

print(f"Initial state:")
print(f"  T_mat = {T_eq:.15f} keV")
print(f"  T_rad = {T_eq:.15f} keV")
print(f"  E_total = {E_total_init:.15e} GJ/cm³")

solver1.step()

T_mat1 = solver1.T[0]
E_r1 = solver1.E_r[0]
T_rad1 = (E_r1 / A_RAD) ** 0.25
E_total1 = cv * rho * T_mat1 + E_r1

print(f"\nAfter one large step:")
print(f"  T_mat = {T_mat1:.15f} keV (change: {T_mat1 - T_eq:+.3e})")
print(f"  T_rad = {T_rad1:.15f} keV (change: {T_rad1 - T_eq:+.3e})")
print(f"  T_mat - T_rad = {T_mat1 - T_rad1:+.3e} keV")
print(f"  E_total = {E_total1:.15e} GJ/cm³")
print(f"  ΔE_total = {E_total1 - E_total_init:+.3e} GJ/cm³")

# ============================================================================
# TEST 2: INCOMING BLACKBODY at T_eq
# ============================================================================
print()
print("-"*80)
print(f"TEST 2: Incoming BB at T_eq = {T_eq:.6f} keV (should maintain equilibrium)")
print("         Using Marshak BC: φ/2 + 2D·(dφ/dr) = φ_inc/2")
print("-"*80)

# Create BC functions for incoming blackbody using Marshak BC
# Marshak BC: 0.5*φ + 2*D*(dφ/dr) = F_in
# In solver form: A*φ + B*(dφ/dr) = C
# With A = 0.5, B = 2*D, C = F_in
def make_incoming_bc(g):
    def bc_incoming(phi, r):
        D = diff_funcs[g](T_eq, r)
        # Marshak BC for incoming blackbody
        return 0.5, 2.0 * D, F_inc_g[g]
    return bc_incoming

left_bcs_incoming = [make_incoming_bc(g) for g in range(n_groups)]
right_bcs_incoming = [bc_reflecting] * n_groups

solver2 = MultigroupDiffusionSolver1D(
    n_groups=n_groups, r_min=r_min, r_max=r_max, n_cells=n_cells,
    energy_edges=energy_edges, geometry='planar', dt=dt,
    diffusion_coeff_funcs=diff_funcs, absorption_coeff_funcs=sigma_funcs,
    left_bc_funcs=left_bcs_incoming, right_bc_funcs=right_bcs_incoming,
    emission_fractions=chi, rho=rho, cv=cv
)
solver2._debug_update_T = False

# Initialize at equilibrium (same as Case 1)
solver2.T[:] = T_eq
solver2.T_old[:] = T_eq
solver2.E_r[:] = E_r_eq
solver2.E_r_old[:] = solver2.E_r.copy()
solver2.phi_g_fraction[:, :] = chi[:, np.newaxis]
# Correct: φ_g = 4π * B_g for isotropic blackbody
phi_g_eq = 4.0 * np.pi * B_g_eq
solver2.phi_g_stored[:, :] = phi_g_eq[:, np.newaxis]

print(f"Initial state:")
print(f"  T_mat = {T_eq:.15f} keV")
print(f"  T_rad = {T_eq:.15f} keV")
print(f"  E_total = {E_total_init:.15e} GJ/cm³")

solver2.step()

T_mat2 = solver2.T[0]
E_r2 = solver2.E_r[0]
T_rad2 = (E_r2 / A_RAD) ** 0.25
E_total2 = cv * rho * T_mat2 + E_r2

print(f"\nAfter one large step:")
print(f"  T_mat = {T_mat2:.15f} keV (change: {T_mat2 - T_eq:+.3e})")
print(f"  T_rad = {T_rad2:.15f} keV (change: {T_rad2 - T_eq:+.3e})")
print(f"  T_mat - T_rad = {T_mat2 - T_rad2:+.3e} keV")
print(f"  E_total = {E_total2:.15e} GJ/cm³")
print(f"  ΔE_total = {E_total2 - E_total_init:+.3e} GJ/cm³")

# ============================================================================
# COMPARISON
# ============================================================================
print()
print("="*80)
print("COMPARISON")
print("="*80)
print()

print("                   Reflecting              Incoming BB")
print(f"T_mat change:    {T_mat1 - T_eq:+.3e} keV        {T_mat2 - T_eq:+.3e} keV")
print(f"T_rad change:    {T_rad1 - T_eq:+.3e} keV        {T_rad2 - T_eq:+.3e} keV")
print(f"T_mat - T_rad:   {T_mat1 - T_rad1:+.3e} keV        {T_mat2 - T_rad2:+.3e} keV")
print(f"ΔE_total:        {E_total1 - E_total_init:+.3e}   {E_total2 - E_total_init:+.3e}")
print()

# ============================================================================
# VERDICT
# ============================================================================
print("="*80)
print("VERDICT")
print("="*80)
print()

tol_T = 1e-6  # keV (relaxed tolerance)
tol_E = 1e-8  # Relative energy tolerance

success = True

print("Case 1 (Reflecting BCs):")
if abs(T_mat1 - T_eq) < tol_T and abs(T_rad1 - T_eq) < tol_T:
    print(f"  ✓ Maintains equilibrium (T changes < {tol_T:.1e} keV)")
else:
    print(f"  ✗ Does NOT maintain equilibrium")
    print(f"    T_mat change: {T_mat1 - T_eq:+.3e}, T_rad change: {T_rad1 - T_eq:+.3e}")
    success = False

if abs(E_total1 - E_total_init) / E_total_init < tol_E:
    print(f"  ✓ Energy conserved (rel change < {tol_E:.1e})")
else:
    print(f"  ✗ Energy NOT conserved (rel change: {abs(E_total1 - E_total_init)/E_total_init:.3e})")
    success = False

print()

print("Case 2 (Incoming BB at T_eq):")
if abs(T_mat2 - T_eq) < tol_T and abs(T_rad2 - T_eq) < tol_T:
    print(f"  ✓ Maintains equilibrium (T changes < {tol_T:.1e} keV)")
else:
    print(f"  ✗ Does NOT maintain equilibrium")
    print(f"    T_mat change: {T_mat2 - T_eq:+.3e}, T_rad change: {T_rad2 - T_eq:+.3e}")
    success = False

# Note: Energy may NOT be conserved with incoming BC (it's an open system)
# But the equilibrium should be maintained
print(f"  Energy change: {E_total2 - E_total_init:+.3e} GJ/cm³")
print(f"  (Open system - energy exchange with boundary is expected)")

print()

if success:
    print("✓✓✓ SUCCESS: Both BC types maintain equilibrium! ✓✓✓")
    print()
    print("This confirms:")
    print("  1. Equilibrium is stable under reflecting BCs (isolated system)")
    print("  2. Incoming BB at T_eq maintains equilibrium (open system)")
    print("  3. Marshak boundary conditions are correctly implemented")
else:
    print("✗✗✗ FAILURE: Equilibrium not maintained ✗✗✗")
    print()
    print("This suggests a problem with either:")
    print("  1. The equilibrium state itself")
    print("  2. The Marshak BC implementation")
    print("  3. The solver time integration")
