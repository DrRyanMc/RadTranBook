#!/usr/bin/env python3
"""Test Zeldovich wave initial condition"""

import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from oneDFV import A_RAD, C_LIGHT, RHO, NonEquilibriumRadiationDiffusionSolver

sys.path.insert(0, str(Path(__file__).parent / 'problems'))
from zeldovich_wave import (equilibrium_temperature, zeldovich_material_energy,
                            zeldovich_rosseland_opacity)

# Setup like the problem
n_cells = 100
r_min, r_max = 0.0, 3.0
d = 0
factor = 1.0  # planar
Etot = factor * 1.0

# Create a dummy solver just to get V_cells
solver = NonEquilibriumRadiationDiffusionSolver(
    r_min=r_min, r_max=r_max, n_cells=n_cells, d=d, dt=1e-5,
    rosseland_opacity_func=zeldovich_rosseland_opacity,
    planck_opacity_func=zeldovich_rosseland_opacity,
    specific_heat_func=lambda T: 3e-6 / RHO,
    material_energy_func=zeldovich_material_energy,
    inverse_material_energy_func=lambda e: e / (RHO * 3e-6),
    left_bc_func=lambda phi,x: (0,1,0),
    right_bc_func=lambda phi,x: (0,1,0)
)

energy_fractions = [0.4, 0.10625, 0.025, 0.0125, 0.00625]

print('Testing equilibrium initial condition:')
print(f'Total energy in pulse: {Etot} GJ')
print(f'Cell width: dx = {(r_max-r_min)/n_cells:.6f} cm')
print(f'Cell 0 volume: V = {solver.V_cells[0]:.6e} cm³')
print()

for i, fraction in enumerate(energy_fractions):
    E_density = fraction * Etot / solver.V_cells[i]
    T_eq = equilibrium_temperature(E_density)
    phi_eq = A_RAD * C_LIGHT * T_eq**4
    e_mat = zeldovich_material_energy(T_eq)
    E_rad = phi_eq / C_LIGHT
    E_total_check = E_rad + e_mat
    sigma_R = zeldovich_rosseland_opacity(T_eq)
    D = C_LIGHT / (3 * sigma_R)
    
    print(f'Cell {i}: fraction = {fraction}')
    print(f'  Target E_density = {E_density:.6e} GJ/cm³')
    print(f'  T_eq = {T_eq:.6f} keV')
    print(f'  phi_eq = {phi_eq:.6e} GJ/cm³')
    print(f'  E_rad = {E_rad:.6e}, E_mat = {e_mat:.6e}, ratio={E_rad/e_mat:.2e}')
    print(f'  E_total (check) = {E_total_check:.6e}')
    print(f'  Error: {abs(E_total_check - E_density)/E_density * 100:.4f}%')
    print(f'  σ_R = {sigma_R:.4e} cm⁻¹, D = {D:.4e} cm²/ns')
    print()
