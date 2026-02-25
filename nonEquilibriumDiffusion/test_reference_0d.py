#!/usr/bin/env python3
"""Test using the exact BE_update from equilibrationTest.py"""

import numpy as np

C_v = 0.01  # GJ/keV/cm^3
c = 2.99792458e1  # cm/ns
sigma = 10.0  # cm^-1
a_rad = 0.01372  # GJ/cm^3/keV^4

# Initial conditions
T_m_initial = 0.4  # keV
T_r_initial = 1.0  # keV
Er_initial = a_rad * T_r_initial**4

# Time stepping
Delta_t = 0.001  # ns
t_final = 0.01  # ns
n_steps = int(t_final / Delta_t)

print("="*70)
print("Reference 0-D Test (" + "using equilibrationTest.py formula)")
print("="*70)
print(f"C_v = {C_v}, sigma = {sigma}, a_rad = {a_rad}")
print(f"T_m_initial = {T_m_initial}, T_r_initial = {T_r_initial}")
print(f"Delta_t = {Delta_t}, n_steps = {n_steps}")
print()

def BE_update(Tstar, Tn, Ern, max_iters=20):
    """Exact formula from equilibrationTest.py"""
    iteration_count = 0
    converged = False
    while (iteration_count < max_iters) and not(converged):
        beta = 4*a_rad*Tstar**3/C_v
        f = 1/(1 + beta*Delta_t*c*sigma)
        Er_new = (Ern + f*sigma*Delta_t*c*(a_rad*Tstar**4) - (1-f)*(C_v*Tstar-C_v*Tn))/(1+f*Delta_t*c*sigma)
        T_new = (C_v*Tn+f*c*sigma*Delta_t*(Er_new - a_rad*Tstar**4) + (1-f)*(C_v*Tstar-C_v*Tn))/(C_v)
        if np.abs(T_new - Tstar) < 1e-10:
            converged = True
        Tstar = T_new
        iteration_count += 1
    return Er_new, T_new, iteration_count

# Run time evolution
T = T_m_initial
Er = Er_initial
E_total_initial = Er + C_v * T

print(f"Initial total energy: {E_total_initial:.6e} GJ/cm^3")
print()

for step in range(n_steps):
    Er, T, iters = BE_update(T, T, Er, max_iters=50)
    E_total = Er + C_v * T
    T_rad = (Er / a_rad)**0.25
    
    if (step + 1) % max(1, n_steps // 5) == 0:
        rel_change = (E_total - E_total_initial) / E_total_initial
        print(f"Step {step+1}/{n_steps}: T_m = {T:.6f} keV, T_rad = {T_rad:.6f} keV, ΔE/E = {rel_change:.6e}, iters={iters}")

E_total_final = Er + C_v * T
T_rad_final = (Er / a_rad)**0.25

print()
print("="*70)
print("Results")
print("="*70)
print(f"Final: T_m = {T:.6f} keV, T_rad = {T_rad_final:.6f} keV")
print(f"Initial total energy: {E_total_initial:.6e} GJ/cm^3")
print(f"Final total energy:   {E_total_final:.6e} GJ/cm^3")
print(f"Relative change:      {abs(E_total_final - E_total_initial) / E_total_initial:.6e}")
print("="*70)
