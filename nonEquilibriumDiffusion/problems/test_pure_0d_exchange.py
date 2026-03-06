#!/usr/bin/env python3
"""
Simplest possible test: 0D radiation-material exchange WITHOUT using the full solver.

Test the basic physics: given E_r and T out of equilibrium, compute the correct dT/dt.

This bypasses all the diffusion,operator, BC, and Newton complexity to test ONLY:
  dT/dt = (Absorption - Emission) / (c_v · ρ)
"""

import numpy as np

# Constants
C_LIGHT = 29.9792458  # cm/ns
A_RAD = 0.01372  # GJ/(cm³·keV⁴)

# Setup
n_groups = 5
energy_edges = np.logspace(np.log10(1e-4), np.log10(5.0), n_groups + 1)

# Material
rho = 1.0  # g/cm³
cv = 0.05 / rho  # GJ/(g·keV)
T_mat = 0.025  # keV

# Radiation (hotter than material)
T_rad_equivalent = 0.05  # keV
E_r_total = A_RAD * T_rad_equivalent**4

# Constant opacities
sigma_values = []
for g in range(n_groups):
    E_mid = np.sqrt(energy_edges[g] * energy_edges[g+1])
    sigma_g = 100.0 * (E_mid**(-2.0))
    sigma_values.append(sigma_g)

# Emission fractions (Planck-weighted)
# For simplicity, use approximate fractions
chi = np.array([8.93e-03, 7.77e-02, 6.21e-01, 2.92e-01, 4.01e-05])
chi /= chi.sum()  # Normalize

print("="*80)
print("PURE 0D RADIATION-MATERIAL EXCHANGE TEST")
print("="*80)
print(f"Material: T = {T_mat} keV, ρ = {rho} g/cm³, c_v = {cv} GJ/(g·keV)")
print(f"Radiation: E_r = {E_r_total:.6e} GJ/cm³ (equivalent to T_rad = {T_rad_equivalent} keV)")
print(f"Number of groups: {n_groups}")
print(f"\nOpacities (cm⁻¹):")
for g, sig in enumerate(sigma_values):
    print(f"  Group {g}: σ = {sig:.3e}, χ = {chi[g]:.6f}")

# Distribute radiation energy by emission fractions
E_r_g = E_r_total * chi

print(f"\n{'-'*80}")
print(f"ENERGY BALANCE CALCULATION")
print(f"{'-'*80}")

# Compute absorption and emission for each group
total_absorption = 0.0
total_emission = 0.0

for g in range(n_groups):
    # Absorption: c · σ_a,g · E_r,g
    absorption_g = C_LIGHT * sigma_values[g] * E_r_g[g]
    
    # Emission: c · σ_a,g · χ_g · a · T_mat^4
    emission_g = C_LIGHT * sigma_values[g] * chi[g] * A_RAD * T_mat**4
    
    total_absorption += absorption_g
    total_emission += emission_g
    
    print(f"Group {g}:")
    print(f"  E_r,g = {E_r_g[g]:.6e} GJ/cm³")
    print(f"  Absorption = c·σ·E_r = {absorption_g:.6e} GJ/(cm³·ns)")
    print(f"  Emission = c·σ·χ·a·T⁴ = {emission_g:.6e} GJ/(cm³·ns)")
    print(f"  Net = {absorption_g - emission_g:.6e} GJ/(cm³·ns)")

net_heating = total_absorption - total_emission
dT_dt = net_heating / (cv * rho)

print(f"\n{'-'*80}")
print(f"RESULTS")
print(f"{'-'*80}")
print(f"Total absorption = {total_absorption:.6e} GJ/(cm³·ns)")
print(f"Total emission = {total_emission:.6e} GJ/(cm³·ns)")
print(f"Net heating = {net_heating:.6e} GJ/(cm³·ns)")
print(f"")
print(f"Material heating rate:")
print(f"  dT/dt = (Absorption - Emission) / (c_v · ρ)")
print(f"        = {net_heating:.6e} / ({cv} · {rho})")
print(f"        = {dT_dt:.6e} keV/ns")
print(f"")
print(f"For dt = 0.01 ns:")
print(f"  ΔT = {dT_dt * 0.01:.6e} keV")
print(f"  T_new = {T_mat + dT_dt * 0.01:.6e} keV")
print(f"")
print(f"This is the CORRECT answer that the solver should reproduce!")
print(f"="*80)

# Also compute what happens to radiation
# In 0D with no boundaries, radiation energy balance is:
#   dE_r/dt = -absorption + emission
dE_r_dt = -total_absorption + total_emission
deltaE_r = dE_r_dt * 0.01

print(f"\nRadiation energy change:")
print(f"  dE_r/dt = {dE_r_dt:.6e} GJ/(cm³·ns)")
print(f"  ΔE_r = {deltaE_r:.6e} GJ/cm³ (for dt=0.01 ns)")
print(f"  E_r_new = {E_r_total + deltaE_r:.6e} GJ/cm³")

# Check energy conservation
delta_e_mat = cv * rho * dT_dt * 0.01  # GJ/cm³
print(f"\n{'-'*80}")
print(f"ENERGY CONSERVATION CHECK")
print(f"{'-'*80}")
print(f"Material energy increase: Δe_mat = {delta_e_mat:.6e} GJ/cm³")
print(f"Radiation energy decrease: -ΔE_r = {-deltaE_r:.6e} GJ/cm³")
print(f"Difference: {delta_e_mat - (-deltaE_r):.6e} GJ/cm³")
print(f"Ratio: {delta_e_mat / (-deltaE_r):.6f} (should be 1.0)")
print(f"="*80)
