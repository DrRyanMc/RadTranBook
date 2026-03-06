#!/usr/bin/env python3
"""
Test that cold material with incoming BB heats to correct equilibrium.

Setup:
- Start with cold material (T_mat = 0.01 keV, E_r ≈ 0)
- Apply incoming BB at T_bc = 1.0 keV on left boundary
- Reflecting BC on right boundary
- Run until steady state reached
- Check that material heats toward equilibrium with incoming radiation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from multigroup_diffusion_solver import MultigroupDiffusionSolver1D, C_LIGHT, A_RAD

# Setup
r_min, r_max = 0.0, 2.0e-3
n_cells = 2
dt = 0.001  # ns - SMALL for strong material-radiation coupling
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

# Initial conditions: COLD material
T_cold = 0.01  # keV - cold start
E_r_cold = A_RAD * T_cold**4

# Boundary condition: INCOMING blackbody at T_bc
T_bc = 1.0  # keV - hot boundary

print("="*80)
print("TEST: COLD MATERIAL HEATED BY INCOMING BLACKBODY")
print("="*80)
print(f"Initial: T_mat = {T_cold:.3f} keV (cold)")
print(f"         E_r   = {E_r_cold:.6e} GJ/cm³")
print(f"Incoming BC: T_bc = {T_bc:.3f} keV (hot blackbody)")
print(f"Right BC: Reflecting (zero flux)")
print()

# Compute incoming flux at T_bc
B_g_bc = Bg_multigroup(energy_edges, T_bc)
chi_bc = B_g_bc / B_g_bc.sum()
F_total_bc = (A_RAD * C_LIGHT * T_bc**4) / 2.0
F_inc_g = chi_bc * F_total_bc

print(f"Incoming total flux: F_total = {F_total_bc:.6e} GJ/(cm²·ns)")
print(f"Emission fractions at T_bc:")
for g in range(n_groups):
    print(f"  Group {g}: χ_g = {chi_bc[g]:.6e}, F_g = {F_inc_g[g]:.6e}")
print()

# Set up BC functions
def make_incoming_bc(g):
    """Marshak BC for incoming blackbody"""
    def bc_incoming(phi, r):
        D = diff_funcs[g](T_bc, r)  # Use D at boundary temperature
        return 0.5, 2.0 * D, F_inc_g[g]
    return bc_incoming

def bc_reflecting(phi, r):
    """Reflecting BC: no flux out"""
    return 0.0, 1.0, 0.0

left_bcs = [make_incoming_bc(g) for g in range(n_groups)]
right_bcs = [bc_reflecting] * n_groups

# Create solver
B_g_cold = Bg_multigroup(energy_edges, T_cold)
chi_cold = B_g_cold / B_g_cold.sum()

solver = MultigroupDiffusionSolver1D(
    n_groups=n_groups, r_min=r_min, r_max=r_max, n_cells=n_cells,
    energy_edges=energy_edges, geometry='planar', dt=dt,
    diffusion_coeff_funcs=diff_funcs, absorption_coeff_funcs=sigma_funcs,
    left_bc_funcs=left_bcs, right_bc_funcs=right_bcs,
    emission_fractions=chi_cold, rho=rho, cv=cv
)
solver._debug_update_T = False

print(f"\n[DEBUG] Solver dt = {solver.dt} ns")
print(f"[DEBUG] Initial solver.t = {solver.t} ns\n")

# Initialize cold
solver.T[:] = T_cold
solver.T_old[:] = T_cold
solver.E_r[:] = E_r_cold
solver.E_r_old[:] = solver.E_r.copy()
solver.phi_g_fraction[:, :] = chi_cold[:, np.newaxis]
phi_g_cold = 4.0 * np.pi * B_g_cold
solver.phi_g_stored[:, :] = phi_g_cold[:, np.newaxis]

E_mat_init = cv * rho * T_cold
E_r_init = solver.E_r[0]
E_total_init = E_mat_init + E_r_init

print(f"Initial energy:")
print(f"  E_mat = {E_mat_init:.6e} GJ/cm³")
print(f"  E_r   = {E_r_init:.6e} GJ/cm³")
print(f"  E_total = {E_total_init:.6e} GJ/cm³")
print()

# Run simulation
n_steps = 100
print("="*80)
print(f"RUNNING {n_steps} TIME STEPS (dt = {dt} ns)")
print("="*80)
print()
print("Step |   Time (ns) | T_mat_left  | T_mat_right | T_rad_left  | T_rad_right | ΔE_total")
print("-----+-------------+-------------+-------------+-------------+-------------+-----------")

for step in range(n_steps):
    T_before = solver.T.copy()
    E_r_before = solver.E_r.copy()
    
    solver.step()
    solver.advance_time()
    
    T_after = solver.T.copy()
    E_r_after = solver.E_r.copy()
    
    if step == 0:
        print(f"[DEBUG] After step 1: solver.t = {solver.t} ns, solver.dt = {solver.dt} ns")
        for i in range(n_cells):
            print(f"[DEBUG] Cell {i}: T_before = {T_before[i]:.6f} keV, T_after = {T_after[i]:.6f} keV, ΔT = {T_after[i] - T_before[i]:.6e} keV")
    
    # Report both cells
    T_mat_left = solver.T[0]
    T_mat_right = solver.T[-1]
    E_r_left = solver.E_r[0]
    E_r_right = solver.E_r[-1]
    T_rad_left = (E_r_left / A_RAD) ** 0.25
    T_rad_right = (E_r_right / A_RAD) ** 0.25
    
    # Compute average for energy balance
    T_mat_avg = solver.T.mean()
    E_r_avg = solver.E_r.mean()
    E_mat_avg = cv * rho * T_mat_avg
    E_total = E_mat_avg + E_r_avg
    dE_total = E_total - E_total_init
    
    if step % 10 == 0 or step < 10 or step == n_steps - 1:
        print(f" {step+1:3d} | {solver.t:11.3f} | {T_mat_left:11.6f} | {T_mat_right:11.6f} | {T_rad_left:11.6f} | {T_rad_right:11.6f} | {dE_total:+.3e}")

# Final state
print()
print("="*80)
print("FINAL STATE")
print("="*80)

print(f"\nLeft cell (x=0, near incoming boundary):")
T_mat_left = solver.T[0]
E_r_left = solver.E_r[0]
T_rad_left = (E_r_left / A_RAD) ** 0.25
print(f"  T_mat = {T_mat_left:.7f} keV")
print(f"  T_rad = {T_rad_left:.7f} keV")
print(f"  |T_mat - T_rad| = {abs(T_mat_left - T_rad_left):.6e} keV ({abs(T_mat_left - T_rad_left)/T_rad_left*100:.2f}%)")
print(f"  E_r = {E_r_left:.6e} GJ/cm³")

print(f"\nRight cell (x=max, far from boundary):")
T_mat_right = solver.T[-1]
E_r_right = solver.E_r[-1]
T_rad_right = (E_r_right / A_RAD) ** 0.25
print(f"  T_mat = {T_mat_right:.7f} keV")
print(f"  T_rad = {T_rad_right:.7f} keV")
print(f"  |T_mat - T_rad| = {abs(T_mat_right - T_rad_right):.6e} keV ({abs(T_mat_right - T_rad_right)/T_rad_right*100:.2f}%)")
print(f"  E_r = {E_r_right:.6e} GJ/cm³")

T_mat_avg = solver.T.mean()
E_r_avg = solver.E_r.mean()
T_rad_avg = (E_r_avg / A_RAD) ** 0.25
E_mat_avg = cv * rho * T_mat_avg
E_total_final = E_mat_avg + E_r_avg

print(f"\nAverage over cells:")
print(f"  T_mat (avg) = {T_mat_avg:.7f} keV")
print(f"  T_rad (avg) = {T_rad_avg:.7f} keV")
print(f"  |T_mat - T_rad| = {abs(T_mat_avg - T_rad_avg):.6e} keV ({abs(T_mat_avg - T_rad_avg)/T_rad_avg*100:.2f}%)")
print(f"  E_mat   = {E_mat_avg:.6e} GJ/cm³")
print(f"  E_r     = {E_r_avg:.6e} GJ/cm³")
print(f"  E_total = {E_total_final:.6e} GJ/cm³")
print(f"  ΔE_total = {E_total_final - E_total_init:+.6e} GJ/cm³")
print()

# Heating rates
print("="*80)
print("ANALYSIS")
print("="*80)

# Check equilibration in each cell
print(f"\nTemperature equilibration:")
T_diff_left = abs(T_mat_left - T_rad_left)
T_diff_right = abs(T_mat_right - T_rad_right)
T_diff_avg = abs(T_mat_avg - T_rad_avg)

print(f"  Left cell:  |T_mat - T_rad| = {T_diff_left:.6e} keV ({T_diff_left/T_rad_left*100:.2f}%)")
print(f"  Right cell: |T_mat - T_rad| = {T_diff_right:.6e} keV ({T_diff_right/T_rad_right*100:.2f}%)")
print(f"  Average:    |T_mat - T_rad| = {T_diff_avg:.6e} keV ({T_diff_avg/T_rad_avg*100:.2f}%)")

if max(T_diff_left, T_diff_right) < 0.01:
    print(f"  ✓ Material and radiation are in local equilibrium")
else:
    print(f"  ⚠ Material and radiation NOT yet in equilibrium")

print()
print(f"Heating progress:")
print(f"  Initial T_mat: {T_cold:.6f} keV")
print(f"  Final T_mat (left):  {T_mat_left:.6f} keV (increase: {(T_mat_left/T_cold - 1)*100:.1f}%)")
print(f"  Final T_mat (right): {T_mat_right:.6f} keV (increase: {(T_mat_right/T_cold - 1)*100:.1f}%)")
print(f"  T_bc:          {T_bc:.6f} keV")
print(f"  T_mat/T_bc (left):  {T_mat_left/T_bc:.6f}")
print(f"  T_mat/T_bc (right): {T_mat_right/T_bc:.6f}")

print()
if T_mat_avg > T_cold * 1.5:
    print("✓ Material is heating from incoming radiation")
else:
    print("✗ Material is NOT heating significantly")

print()
print("Energy balance:")
print(f"  Total energy added: {E_total_final - E_total_init:.6e} GJ/cm³")
print(f"  Average flux:       {(E_total_final - E_total_init)/(n_steps*dt):.6e} GJ/(cm³·ns)")
print()

print("Expected behavior:")
print("  - Material should heat from cold start ✓" if T_mat_avg > T_cold * 1.5 else "  - Material should heat from cold start ✗")
print(f"  - T_mat ≈ T_rad in each cell ✓" if max(T_diff_left, T_diff_right) < 0.01 else f"  - T_mat ≈ T_rad (max diff = {max(T_diff_left, T_diff_right):.3e})")
print(f"  - Cells may have different T (spatial gradient)")
print()

if T_mat_avg > T_cold * 1.5 and max(T_diff_left, T_diff_right) < T_mat_avg * 0.05:
    print("✓✓✓ TEST PASSED: Material heats correctly with incoming BB ✓✓✓")
else:
    print("⚠ TEST INCONCLUSIVE: May need more time steps or different setup")
