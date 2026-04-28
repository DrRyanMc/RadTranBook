#!/usr/bin/env python
"""Diagnostic for energy conservation in multigroup S_N."""

import numpy as np
import sys; sys.path.insert(0, '.')
sys.path.insert(0, 'problems')

# Simplified version of infinite medium test to debug energy conserva conservation

from RadTranModel import RadTranModel
from DiscreteOrdinates.mg_sn_solver import mg_temp_solve_dmd_inc

# Create a model
RT = RadTranModel(units="CGS-rad")
RT.set_opacity_model("exponential_band", n_groups=10, T_floor=0.1, T_ceil=2.0)
RT.setup([1.0], [1.0])  # single cell, unit width

# Initial conditions (near equilibrium, T_rad > T_mat for energy flow)
T_rad_init = 0.5    # keV
T_mat_init = 0.4    # keV
state_init = {'T_rad': np.array([T_rad_init]), 'T_mat': np.array([T_mat_init])}

# Energy scale
Cv = 0.01  # GJ/(cm³·keV)
a = RT.a_rad  # radiation constant
ac = 1.0 * a  # c * a
E_rad_init = a * T_rad_init**4  # GJ/cm³
E_mat_init = Cv * T_mat_init    # GJ/cm³
E_total_init = E_rad_init + E_mat_init

print(f"Initial state:")
print(f"  T_mat = {T_mat_init:.6f} keV, E_mat = {E_mat_init:.6e} GJ/cm³")
print(f"  T_rad = {T_rad_init:.6f} keV, E_rad = {E_rad_init:.6e} GJ/cm³")
print(f"  E_total = {E_total_init:.6e} GJ/cm³")
print()

# Run one short timestep with diagnostic output
I = 1
sigma_a_funcs = RT.get_sigma_a_list()
sigma_s_funcs = [lambda T: 0*T for _ in range(RT.ng)]  # no scattering
Bg_funcs = RT.get_Bg_list()
dBdT_funcs = RT.get_dBdT_list()
Cv_func = lambda T: Cv * (0*T + 1)
chi_func = None

T_mat = np.array([[T_mat_init]])
T_old = T_mat.copy()
e_old = Cv * T_old

# Manual step calculation to see what happens
dt = 1e-4  # ns
dBdT_vals = [f(T_old) for f in dBdT_funcs]
coupling_sum = np.zeros_like(T_old)
for g in range(RT.ng):
    coupling_sum += sigma_a_funcs[g](T_old) * dBdT_vals[g]

f = 1.0 / (1.0 + dt / Cv * coupling_sum)
alpha_g = [sigma_a_funcs[g](T_old) * dBdT_vals[g] * f * dt / Cv for g in range(RT.ng)]
chi = [0*T_old for _ in range(RT.ng)]

print(f"Step diagnostic (dt = {dt}):")
print(f"  f = {f[0,0]:.10f}")
print(f"  coupling_sum = {coupling_sum[0,0]:.6e}")
print()

# Get opacities
sigma_ag = [sigma_a_funcs[g](T_old) for g in range(RT.ng)]
Bg_vals = [Bg_funcs[g](T_old) for g in range(RT.ng)]

# Imagine perfect flux solution: φ = B at each group
phi_g_perfect = [Bg_vals[g].copy() for g in range(RT.ng)]
energy_dep_perfect = 0.0
for g in range(RT.ng):
    energy_dep_perfect += sigma_ag[g][0,0] * (phi_g_perfect[g][0,0] - Bg_vals[g][0,0])

print(f"Perfect flux (φ_g = B_g everywhere):")
print(f"  energy_dep = {energy_dep_perfect:.6e}")
print(f"  e_new (with f) = {e_old[0,0] + f[0,0] * dt * energy_dep_perfect:.6e}")
print(f"  e_new (no f)   = {e_old[0,0] + dt * energy_dep_perfect:.6e}")
print()

# Now imagine a hot radiation field: φ = B + ΔB
Delta_B = np.array([0.1 * Bg_vals[g][0,0] for g in range(RT.ng)])  # 10% hotter than Planck
phi_g_hot = [Bg_vals[g] + Delta_B[g] for g in range(RT.ng)]

energy_dep_hot = 0.0
absorption_g = []
for g in range(RT.ng):
    absorption = sigma_ag[g][0,0] * (Delta_B[g])
    absorption_g.append(absorption)
    energy_dep_hot += absorption

print(f"Hotter radiation (φ_g = B_g + 0.1·B_g):")
for g in range(RT.ng):
    print(f"  g={g}: σ_a = {sigma_ag[g][0,0]:.6e}, absorption = {absorption_g[g]:.6e}")
print(f"  total energy_dep = {energy_dep_hot:.6e}")
print()

# Test: in equilibrium (T_mat = T_rad), what happens?
T_eq = 0.45  # intermediate
T_eq_arr = np.array([[T_eq]])
sigma_ag_eq = [sigma_a_funcs[g](T_eq_arr) for g in range(RT.ng)]
Bg_eq = [Bg_funcs[g](T_eq_arr) for g in range(RT.ng)]
phi_eq = Bg_eq  # in equilibrium

energy_in_eq = 0.0
for g in range(RT.ng):
    energy_in_eq += sigma_ag_eq[g][0,0] * (Bg_eq[g][0,0] - Bg_eq[g][0,0])
print(f"Equilibrium (both at T={T_eq:.6f} keV, φ = B):")
print(f"  energy_dep = {energy_in_eq:.6e} (should be zero)")
print()

# Key question: what is the relation between material temperature update and radiation field?
dBdT_sum = 0.0
for g in range(RT.ng):
    dBdT_sum += sigma_ag[g][0,0] * dBdT_vals[g][0,0]
print(f"Energy coupling terms:")
print(f"  Σ_g σ_a dB/dT = {dBdT_sum:.6e}")
print(f"  dt / Cv * this = {(dt/Cv * dBdT_sum):.6e}")
print(f"  f = 1 / (1 + this) = {f[0,0]:.10f}")
print(f"  (1-f) = {(1-f[0,0]):.10f}")
