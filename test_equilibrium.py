#!/usr/bin/env python
"""
Test energy conservation in equilibrium state: φ = B everywhere.
In equilibrium with no temperature gradients, there should be NO energy transfer
between radiation and material, so material temperature should not change.
This tests whether the energy equation formulation itself is correct.
"""

import numpy as np
import sys; 
sys.path.insert(0, 'DiscreteOrdinates')
sys.path.insert(0, 'DiscreteOrdinates/problems')
sys.path.insert(0, '.')

from RadTranModel import RadTranModel
from DiscreteOrdinates.mg_sn_solver import mg_temp_solve_dmd_inc

# Create test model
RT = RadTranModel(units="CGS-rad")
RT.set_opacity_model("exponential_band", n_groups=10, T_floor=0.1, T_ceil=2.0)
RT.setup([1.0], [1.0])  # single cell, 1 cm

# Constants
I = 1
hx = 1.0
N = 4
order = 3
nop1 = order + 1

# Initialize to a single temperature everywhere (equilibrium state)
T_eq = 0.45  # keV (single uniform temperature)
T_init = np.full((I, nop1), T_eq)

# Get opacity functions
sigma_a_funcs = RT.get_sigma_a_list()
scat_funcs = [lambda T: 0*T for _ in range(RT.ng)]
Bg_funcs = RT.get_Bg_list()
dBdT_funcs = RT.get_dBdT_list()
Cv_func = lambda T: 0.01 * (0*T + 1)

# Initial radiation field: equilibrium Planck (φ = B)
phi_g_init = [Bg_funcs[g](T_init) for g in range(RT.ng)]

# Isotropic angular flux
psi_g_init = [phi_g_init[g][:, np.newaxis, :] * np.ones((I, N, nop1))
              for g in range(RT.ng)]

# External source: none
q_ext = [np.zeros((I, N, nop1)) for _ in range(RT.ng)]

# BCs: zero incoming (no gradient)
def zero_bc(t):
    return np.zeros((N, nop1))

BCs = [zero_bc for _ in range(RT.ng)]

# EOS
a_rad = RT.a_rad
ac = 1.0 * a_rad

def eos(T):
    """T in keV, returns energy density in GJ/cm³."""
    return 0.01 * T  # linear EOS with Cv = 0.01

def invEOS(e):
    """Energy density to temperature."""
    return e / 0.01

print("="*70)
print("EQUILIBRIUM TEST: φ = B everywhere, uniform T = 0.45 keV")
print("="*70)
print("\nExpected behavior:")
print("  - Radiation field is Planck at T_eq everywhere")
print("  - φ - B = 0, so no net energy transfer")
print("  - Material temperature should NOT change")
print("  - Energy should be EXACTLY conserved")
print()

# Run for a short time
output_times = [0.001, 0.01, 0.1]  # ns

phi_g_hist, T_hist, total_its, ts = mg_temp_solve_dmd_inc(
    I, hx,
    RT.ng, sigma_a_funcs, scat_funcs, Bg_funcs, dBdT_funcs,
    q_ext, N, BCs, eos, invEOS, Cv_func,
    phi_g_init, psi_g_init, T_init,
    dt_min=1e-5, dt_max=1e-3, tfinal=0.1,
    order=order, LOUD=False, maxits=100, fix=0, K=30, R=3,
    time_outputs=output_times,
    reflect_left=True,
    reflect_right=True,
)

print(f"\n{'Time (ns)':<12} {'T_mat (keV)':<15} {'ΔT (keV)':<15} {'E_mat':<15} {'E_rad':<15} {'E_total':<15} {'ΔE_frac':<15}")
print("-" * 110)

E_total_0 = 0.01 * T_eq + RT.a_rad * T_eq**4  # initial total energy

for i, t in enumerate(ts):
    T = T_hist[i, 0]
    E_mat = 0.01 * T
    E_rad = RT.a_rad * T**4  # from Planck function
    E_total = E_mat + E_rad
    dT = T - T_eq
    dE_frac = (E_total - E_total_0) / E_total_0
    
    print(f"{t:<12.4e} {T:<15.8f} {dT:<15.4e} {E_mat:<15.4e} {E_rad:<15.4e} {E_total:<15.4e} {dE_frac:<15.4e}")

print("\n" + "="*70)
print("RESULT:")
if abs(T_hist[-1, 0] - T_eq) < 1e-6:
    print("✓ PASS: Temperature unchanged (as expected)")
else:
    print("✗ FAIL: Temperature changed!")
    print(f"   Expected T = {T_eq:.8f} keV")
    print(f"   Got T =      {T_hist[-1, 0]:.8f} keV")
    print(f"   ΔT = {T_hist[-1, 0] - T_eq:.4e} keV")

E_fin = 0.01 * T_hist[-1, 0] + RT.a_rad * T_hist[-1, 0]**4
dE_frac_final = (E_fin - E_total_0) / E_total_0
if abs(dE_frac_final) < 1e-10:
    print("✓ PASS: Energy conserved to machine precision")
else:
    print(f"✗ FAIL: Energy not conserved")
    print(f"   Fractional loss: {dE_frac_final:.4e}")
    print(f"   This indicates an error in the equilibrium formulation")
