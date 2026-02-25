#!/usr/bin/env python3
"""
Debug the T equation solver
"""

import numpy as np

# Constants
C_LIGHT = 29.9792458  # cm/ns
A_RAD = 0.01372       # GJ/(cm³·keV⁴)
RHO = 1.0             # g/cm³

#Material properties
def test_planck_opacity(T):
    return 300.0 * max(T, 0.001)**(-3)

def test_specific_heat(T):
    cv_volumetric = 3e-6
    return cv_volumetric / RHO

def test_material_energy(T):
    cv_volumetric = 3e-6
    return RHO * cv_volumetric * T

def test_inverse_material_energy(e):
    cv_volumetric = 3e-6
    return e / (RHO * cv_volumetric)

def get_beta(T_star):
    cv_star = test_specific_heat(T_star)
    return (4.0 * A_RAD * T_star**3) / (RHO * cv_star)

def get_f_factor(T_star, dt, theta):
    beta = get_beta(T_star)
    sigma_P = test_planck_opacity(T_star)
    return 1.0 / (1.0 + beta * sigma_P * C_LIGHT * theta * dt)

# Cell 1 values from test
dt = 1e-4
theta = 1.0
T_prev_1 = 0.1
T_star_1 = 0.1
phi_prev_1 = 4.1132e-05
phi_np1_1 = 1.33200442

print("="*80)
print("DEBUGGING T EQUATION FOR CELL 1")
print("="*80)
print(f"Inputs:")
print(f"  T_prev = {T_prev_1} keV")
print(f"  T_star = {T_star_1} keV")
print(f"  φ_prev = {phi_prev_1} GJ/cm³")
print(f"  φ_np1 = {phi_np1_1} GJ/cm³")
print(f"  dt = {dt} ns")
print(f"  theta = {theta}")

# Compute phi_tilde
phi_tilde_1 = theta * phi_np1_1 + (1.0 - theta) * phi_prev_1
print(f"\nφ̃ = θ·φ_np1 + (1-θ)·φ_prev = {phi_tilde_1} GJ/cm³")

# Material energies
e_n_1 = test_material_energy(T_prev_1)
e_star_1 = test_material_energy(T_star_1)
Delta_e_1 = e_star_1 - e_n_1
print(f"\nMaterial energies:")
print(f"  e_n = ρ·c_v·T_prev = {e_n_1} GJ/cm³")
print(f"  e_star = ρ·c_v·T_star = {e_star_1} GJ/cm³")
print(f"  Δe = e_star - e_n = {Delta_e_1} GJ/cm³")

# Coupling parameters
sigma_P_1 = test_planck_opacity(T_star_1)
f_1 = get_f_factor(T_star_1, dt, theta)
acT4_star_1 = A_RAD * C_LIGHT * T_star_1**4
beta_1 = get_beta(T_star_1)

print(f"\nCoupling parameters at T_star:")
print(f"  σ_P = {sigma_P_1} cm⁻¹")
print(f"  β = {beta_1}")
print(f"  f = 1/(1 + β·σ_P·c·θ·Δt) = {f_1}")
print(f"  acT★⁴ = {acT4_star_1} GJ/cm³")

# Compute e_np1
term1 = e_n_1
term2 = dt * f_1 * sigma_P_1 * (phi_tilde_1 - acT4_star_1)
term3 = (1.0 - f_1) * Delta_e_1

print(f"\nComputing e_np1 = e_n + dt·f·σ_P·(φ̃ - acT★⁴) + (1-f)·Δe:")
print(f"  Term 1: e_n = {term1} GJ/cm³")
print(f"  Term 2: dt·f·σ_P·(φ̃ - acT★⁴) = {term2} GJ/cm³")
print(f"    Breakdown:")
print(f"      φ̃ - acT★⁴ = {phi_tilde_1 - acT4_star_1} GJ/cm³")  
print(f"      f·σ_P = {f_1 * sigma_P_1}")
print(f"      dt·f·σ_P·(...) = {term2}")
print(f"  Term 3: (1-f)·Δe = {term3} GJ/cm³")

e_np1_1 = term1 + term2 + term3
print(f" e_np1 = {e_np1_1} GJ/cm³")

# Convert to temperature
T_np1_1 = test_inverse_material_energy(e_np1_1)
print(f"\nT_np1 = e_np1 / (ρ·c_v) = {T_np1_1} keV")
print(f"\nExpected: ~0.9 keV")
print(f"Actual from code: 809.6 keV")
print(f"\n*** If we got 809.6 keV, the e_np1 would need to be: {test_material_energy(809.6)} GJ/cm³")
