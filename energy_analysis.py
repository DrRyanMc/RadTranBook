#!/usr/bin/env python
"""
Detailed energy balance diagnostic for multigroup S_N transport.
Modifies the solver to output term-by-term energy accounting.
"""

import numpy as np
import sys; sys.path.insert(0, 'DiscreteOrdinates'); sys.path.insert(0, 'DiscreteOrdinates/problems')

# Import and run the simple test case
import test_infinite_medium_multigroup_expband_sn as test_im

# Patch the solver to collect energy diagnostics
import DiscreteOrdinates.mg_sn_solver as msolver
original_solver = msolver.mg_temp_solve_dmd_inc

energy_diagnostics = []

def patched_solver(*args, **kwargs):
    """Wrap the solver to collect energy diagnostics each step."""
    global energy_diagnostics
    energy_diagnostics = []
    
    # Call original but we'll need to access it step-by-step
    # For now, just run the test normally and compute diagnostics after
    return original_solver(*args, **kwargs)

# Just run normally for now
print("Running infinite medium test with diagnostics...")
data = test_im.run_problem(n_groups=10, final_time=0.1)

times = data['times']
T_mat = data['T_mat']
E_mat_hist = 0.01 * T_mat  # CV = 0.01

group_u_hist = data['group_energy_history']
T_rad_hist = data['T_rad']
E_rad_hist = group_u_hist.sum(axis=1)  # sum over groups

print("\nDetailed energy analysis:")
print(f"{'Step':<5} {'Time':<10} {'T_mat':<10} {'E_mat':<12} {'E_rad':<12} {'E_total':<12} {'dE/dt':<12} {'frac_loss':<12}")
print("-" * 110)

for i in range(len(times)):
    E_mat = E_mat_hist[i]
    E_rad = E_rad_hist[i]
    E_total = E_mat + E_rad
    
    if i > 0:
        dE = (E_total - (E_mat_hist[i-1] + E_rad_hist[i-1]))
        dt = times[i] - times[i-1]
        dE_dt = dE / dt if dt > 0 else 0
        frac_loss = dE / (E_mat_hist[i-1] + E_rad_hist[i-1])
    else:
        dE_dt = 0
        frac_loss = 0
    
    print(f"{i:<5} {times[i]:<10.4e} {T_mat[i]:<10.6f} {E_mat:<12.4e} {E_rad:<12.4e} {E_total:<12.4e} {dE_dt:<12.4e} {frac_loss:<12.4e}")

print("\n" + "="*110)
print("\nKey observations:")
print(f"  Initial: E_mat(0) = {E_mat_hist[0]:.4e}, E_rad(0) = {E_rad_hist[0]:.4e}")
print(f"  Final:   E_mat(T) = {E_mat_hist[-1]:.4e}, E_rad(T) = {E_rad_hist[-1]:.4e}")
print(f"  ΔE_mat = {E_mat_hist[-1] - E_mat_hist[0]:.4e}")
print(f"  ΔE_rad = {E_rad_hist[-1] - E_rad_hist[0]:.4e}")
print(f"  Should be: ΔE_mat + ΔE_rad ≈ 0  (reflecting BCs)")
print(f"  Actual sum: {(E_mat_hist[-1] - E_mat_hist[0]) + (E_rad_hist[-1] - E_rad_hist[0]):.4e}")
