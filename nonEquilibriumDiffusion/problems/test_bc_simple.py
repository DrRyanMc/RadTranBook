#!/usr/bin/env python3
"""
Very simple test to verify Dirich boundary conditions
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from twoDFV import NonEquilibriumRadiationDiffusionSolver2D

C_LIGHT = 2.998e1
A_RAD = 0.01372

def const_opacity(T):
    return 1.0

def const_heat(T):
    return 1.0

def const_energy(T):
    return T

def inv_const_energy(e):
    return e

# Dirichlet BC at T=1 keV
def bc_hot(phi, pos, boundary='left', geometry='cartesian'):
    T_bc = 1.0
    phi_bc = C_LIGHT * A_RAD * T_bc**4
    return 1.0, 0.0, phi_bc

# Zero flux BC
def bc_cold(phi, pos, boundary='left', geometry='cartesian'):
    return 0.0, 1.0, 0.0

print("Test: X-direction with hot left boundary")
boundary_funcs = {
    'left': bc_hot,
    'right': bc_cold,
    'bottom': bc_cold,
    'top': bc_cold
}

solver = NonEquilibriumRadiationDiffusionSolver2D(
    x_min=0.0, x_max=1.0, nx_cells=5,
    y_min=0.0, y_max=0.2, ny_cells=3,
    geometry='cartesian',
    dt=0.1,  # Larger timestep
    theta=1.0,
    rosseland_opacity_func=const_opacity,
    planck_opacity_func=const_opacity,
    specific_heat_func=const_heat,
    material_energy_func=const_energy,
    inverse_material_energy_func=inv_const_energy,
    boundary_funcs=boundary_funcs
)

T_init = 0.01
phi_init = C_LIGHT * A_RAD * T_init**4
solver.set_initial_condition(phi_init=phi_init, T_init=T_init)

print(f"\nBefore step:")
print(f"φ_2d[:, 1] = {solver.get_phi_2d()[:, 1]}")
print(f"T_2d[:, 1] = {solver.get_T_2d()[:, 1]}")

print(f"\nTaking 100 time steps...")
for istep in range(100):
    solver.time_step(n_steps=1, verbose=False)
    if istep % 20 == 0:
        print(f"Step {istep}: Max T = {solver.T.max():.4f}")

print(f"\nAfter 100 steps:")
print(f"φ_2d[:, 1] = {solver.get_phi_2d()[:, 1]}")
print(f"T_2d[:, 1] = {solver.get_T_2d()[:, 1]}")
print(f"Max T = {solver.T.max():.4f}, should approach 1.0")
