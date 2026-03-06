#!/usr/bin/env python3
"""
Test equilibrium with incoming blackbody BCs at equilibrium temperature.

Compare two cases:
1. Reflecting BCs (no incoming radiation)
2. Incoming blackbody at T_eq

Both should give the same equilibrium: T_mat = T_rad = T_eq
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from multigroup_diffusion_solver import MultigroupDiffusionSolver1D, C_LIGHT, A_RAD

# Setup
r_min, r_max = 0.0, 1.0e-3
n_cells = 1
dt = 100.0  # Large timestep for equilibration
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

# Initial conditions (out of equilibrium)
T_init = 0.025
T_rad_init = 0.05

# Compute expected equilibrium temperature
E_r_init = A_RAD * T_rad_init**4
E_total_init = T_init * cv * rho + E_r_init
from scipy.optimize import brentq
T_eq = brentq(lambda T: cv * rho * T + A_RAD * T**4 - E_total_init, 0.001, 1.0)
E_r_eq = A_RAD * T_eq**4

print("="*80)
print("TEST: EQUILIBRIUM WITH INCOMING BLACKBODY BCs")
print("="*80)
print(f"Expected equilibrium: T_eq = {T_eq:.15f} keV")
print(f"                      E_r_eq = {E_r_eq:.15e} GJ/cm³")
print()

# ============================================================================
# TEST 1: REFLECTING BCs (no incoming radiation)
# ============================================================================
print("-"*80)
print("TEST 1: Reflecting BCs (no incoming radiation)")
print("-"*80)

B_g = Bg_multigroup(energy_edges, 0.05)
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

solver1.T[:] = T_init
solver1.T_old[:] = T_init
solver1.E_r[:] = E_r_init
solver1.E_r_old[:] = solver1.E_r.copy()
solver1.phi_g_fraction[:, :] = chi[:, np.newaxis]
solver1.phi_g_stored[:, :] = chi[:, np.newaxis] * A_RAD * C_LIGHT * T_rad_init**4

solver1.step()

T_mat1 = solver1.T[0]
E_r1 = solver1.E_r[0]
T_rad1 = (E_r1 / A_RAD) ** 0.25
E_mat1 = cv * rho * T_mat1

print(f"\nAfter one large step:")
print(f"  T_mat = {T_mat1:.15f} keV")
print(f"  T_rad = {T_rad1:.15f} keV")
print(f"  T_mat - T_rad = {T_mat1 - T_rad1:+.3e} keV")
print(f"  T_mat - T_eq  = {T_mat1 - T_eq:+.3e} keV")
print(f"  E_mat = {E_mat1:.15e} GJ/cm³")
print(f"  E_r   = {E_r1:.15e} GJ/cm³")
print(f"  E_total = {E_mat1 + E_r1:.15e} GJ/cm³")

# ============================================================================
# TEST 2: INCOMING BLACKBODY at T_eq
# ============================================================================
print()
print("-"*80)
print(f"TEST 2: Incoming blackbody at T_eq = {T_eq:.6f} keV")
print("         Using Marshak boundary condition: φ/2 + 2D·(dφ/dr) = φ_inc/2")
print("-"*80)

# Compute incoming blackbody flux for each group at T_eq
B_g_eq = Bg_multigroup(energy_edges, T_eq)
chi_eq = B_g_eq / B_g_eq.sum()
phi_inc_g = (4.0 * np.pi / C_LIGHT) * B_g_eq

print(f"\nIncoming φ_g at T_eq:")
for g in range(n_groups):
    print(f"  Group {g}: φ_inc = {phi_inc_g[g]:.6e}")

# Create BC functions for incoming blackbody using Marshak BC
# For left boundary (incoming from left): Marshak BC for incoming radiation
# Standard form: φ/2 + 2D·(dφ/dr) = φ_inc
# Or in solver form: A·φ + B·(dφ/dr) = C
# With A = 0.5, B = 2D, C = 0.5*φ_inc (for incoming blackbody)
def make_incoming_bc(g):
    def bc_incoming(phi, r):
        # Get diffusion coefficient at this location
        D = diff_funcs[g](T_eq, r)  # Evaluate at equilibrium temperature
        # Marshak BC: 0.5*φ + 2*D*(dφ/dr) = 0.5*φ_inc
        return 0.5, 2.0 * D, 0.5 * phi_inc_g[g]
    return bc_incoming

left_bcs_incoming = [make_incoming_bc(g) for g in range(n_groups)]
right_bcs_incoming = [bc_reflecting] * n_groups  # Keep right BC reflecting

solver2 = MultigroupDiffusionSolver1D(
    n_groups=n_groups, r_min=r_min, r_max=r_max, n_cells=n_cells,
    energy_edges=energy_edges, geometry='planar', dt=dt,
    diffusion_coeff_funcs=diff_funcs, absorption_coeff_funcs=sigma_funcs,
    left_bc_funcs=left_bcs_incoming, right_bc_funcs=right_bcs_incoming,
    emission_fractions=chi, rho=rho, cv=cv
)
solver2._debug_update_T = False

# Same initial conditions
solver2.T[:] = T_init
solver2.T_old[:] = T_init
solver2.E_r[:] = E_r_init
solver2.E_r_old[:] = solver2.E_r.copy()
solver2.phi_g_fraction[:, :] = chi[:, np.newaxis]
solver2.phi_g_stored[:, :] = chi[:, np.newaxis] * A_RAD * C_LIGHT * T_rad_init**4

solver2.step()

T_mat2 = solver2.T[0]
E_r2 = solver2.E_r[0]
T_rad2 = (E_r2 / A_RAD) ** 0.25
E_mat2 = cv * rho * T_mat2

print(f"\nAfter one large step:")
print(f"  T_mat = {T_mat2:.15f} keV")
print(f"  T_rad = {T_rad2:.15f} keV")
print(f"  T_mat - T_rad = {T_mat2 - T_rad2:+.3e} keV")
print(f"  T_mat - T_eq  = {T_mat2 - T_eq:+.3e} keV")
print(f"  E_mat = {E_mat2:.15e} GJ/cm³")
print(f"  E_r   = {E_r2:.15e} GJ/cm³")
print(f"  E_total = {E_mat2 + E_r2:.15e} GJ/cm³")

# ============================================================================
# COMPARISON
# ============================================================================
print()
print("="*80)
print("COMPARISON")
print("="*80)
print()

print("Case 1 (Reflecting):           Case 2 (Incoming BB at T_eq):")
print(f"  T_mat = {T_mat1:.10f} keV      T_mat = {T_mat2:.10f} keV")
print(f"  T_rad = {T_rad1:.10f} keV      T_rad = {T_rad2:.10f} keV")
print(f"  Diff  = {abs(T_mat1-T_rad1):.3e} keV           Diff  = {abs(T_mat2-T_rad2):.3e} keV")
print()

T_diff_cases = abs(T_mat1 - T_mat2)
E_r_diff_cases = abs(E_r1 - E_r2)

print(f"Difference between two cases:")
print(f"  |T_mat(1) - T_mat(2)| = {T_diff_cases:.3e} keV")
print(f"  |E_r(1) - E_r(2)|     = {E_r_diff_cases:.3e} GJ/cm³")
print(f"  Relative E_r diff    = {E_r_diff_cases/E_r1*100:.6f}%")
print()

# ============================================================================
# VERDICT
# ============================================================================
print("="*80)
print("VERDICT")
print("="*80)
print()

tol_T = 1e-8  # keV
tol_E = 1e-10  # Relative

all_good = True

# Check Case 1: Reflecting
print("Case 1 (Reflecting BCs):")
if abs(T_mat1 - T_rad1) < tol_T:
    print(f"  ✓ T_mat ≈ T_rad (diff = {abs(T_mat1-T_rad1):.3e} keV)")
else:
    print(f"  ⚠ T_mat ≠ T_rad (diff = {abs(T_mat1-T_rad1):.3e} keV, tol={tol_T:.1e})")
    all_good = False

if abs(T_mat1 - T_eq) < tol_T:
    print(f"  ✓ T_mat ≈ T_eq (diff = {abs(T_mat1-T_eq):.3e} keV)")
else:
    print(f"  ⚠ T_mat ≠ T_eq (diff = {abs(T_mat1-T_eq):.3e} keV, tol={tol_T:.1e})")
    all_good = False

print()

# Check Case 2: Incoming BB
print("Case 2 (Incoming BB at T_eq):")
if abs(T_mat2 - T_rad2) < tol_T:
    print(f"  ✓ T_mat ≈ T_rad (diff = {abs(T_mat2-T_rad2):.3e} keV)")
else:
    print(f"  ⚠ T_mat ≠ T_rad (diff = {abs(T_mat2-T_rad2):.3e} keV, tol={tol_T:.1e})")
    all_good = False

if abs(T_mat2 - T_eq) < tol_T:
    print(f"  ✓ T_mat ≈ T_eq (diff = {abs(T_mat2-T_eq):.3e} keV)")
else:
    print(f"  ⚠ T_mat ≠ T_eq (diff = {abs(T_mat2-T_eq):.3e} keV, tol={tol_T:.1e})")
    all_good = False

print()

# Check agreement between cases
print("Agreement between cases:")
if T_diff_cases < tol_T:
    print(f"  ✓ Both cases give same T_mat (diff = {T_diff_cases:.3e} keV)")
else:
    print(f"  ✗ Cases give different T_mat (diff = {T_diff_cases:.3e} keV, tol={tol_T:.1e})")
    print(f"    This is WRONG - incoming BB at T_eq should give same equilibrium!")
    all_good = False

if E_r_diff_cases / E_r1 < tol_E:
    print(f"  ✓ Both cases give same E_r (rel diff = {E_r_diff_cases/E_r1:.3e})")
else:
    print(f"  ✗ Cases give different E_r (rel diff = {E_r_diff_cases/E_r1:.3e}, tol={tol_E:.1e})")
    print(f"    This is WRONG - incoming BB at T_eq should give same equilibrium!")
    all_good = False

print()
if all_good:
    print("✓✓✓ SUCCESS: Both BC types give same equilibrium! ✓✓✓")
else:
    print("✗✗✗ FAILURE: BC types give different results ✗✗✗")
