#!/usr/bin/env python
"""
Instrument mg_sn_solver to diagnose energy loss by printing f values and energy terms each step.
"""

import numpy as np
import sys; sys.path.insert(0, 'DiscreteOrdinates'); sys.path.insert(0, 'DiscreteOrdinates/problems')

# Patch in diagnostic output
import DiscreteOrdinates.mg_sn_solver as msolver

original_compute_fleck = msolver._compute_fleck_and_alpha

def patched_compute_fleck(T, dt, sigma_a_funcs, Bg_funcs, dBdT_funcs, Cv_func, G):
    """Wrapped version that prints f and alpha diagnostics."""
    f, alpha_g = original_compute_fleck(T, dt, sigma_a_funcs, Bg_funcs, dBdT_funcs, Cv_func, G)
    print(f"    [DIAG] f = {f[0,0]:.10f},  alpha_g[0] = {alpha_g[0][0,0]:.6e}")
    return f, alpha_g

msolver._compute_fleck_and_alpha = patched_compute_fleck

# Now run a simple test
import test_infinite_medium_multigroup_expband_sn as test_im

print("="*80)
print("Running infinite medium test with diagnostic output")
print("="*80)

data = test_im.run_problem(n_groups=10, final_time=0.01)

print("\n" + "="*80)
print("Energy summary:")
CV = 0.01
times = data['times']
T_mat = data['T_mat']
E_mat_hist = CV * T_mat
E_rad_hist = data['group_energy_history'].sum(axis=1)
E_total = E_mat_hist + E_rad_hist

E0 = E_total[0]
print(f"E(t=0) = {E0:.6e}")
print(f"E(t_f) = {E_total[-1]:.6e}")
print(f"ΔE = {E_total[-1] - E0:.6e}")
print(f"ΔE/E0 = {(E_total[-1] - E0)/E0:.6e}")
