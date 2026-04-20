#!/usr/bin/env python3
"""Quick test to check if CF is depositing energy properly"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import IMC2D_CarterForest as imc2d_cf

# Simple Marshak wave material model
CV_VAL = 0.3
def sigma_a_f(T): return 300.0 * np.maximum(T, 1e-4) ** -3
def eos(T): return CV_VAL * T
def inv_eos(u): return u / CV_VAL
def cv(T): return 0.0 * T + CV_VAL

# Small mesh for quick test
n = 20
x_edges = np.linspace(0.0, 0.1, n + 1)
y_edges = np.linspace(0.0, 0.1, n + 1)

Tinit = np.full((n, n), 1e-4)
Trinit = np.full((n, n), 1e-4)
source = np.zeros((n, n))

print('[CF Energy Deposition Test]')
print('='*60)
print('Initializing simulation...')
state = imc2d_cf.init_simulation(
    5000, Tinit, Trinit, x_edges, y_edges, eos, inv_eos, geometry='xy'
)

cell_vol = (x_edges[1] - x_edges[0]) * (y_edges[1] - y_edges[0])
total_vol = (x_edges[-1] - x_edges[0]) * (y_edges[-1] - y_edges[0])

print(f'\nInitial State:')
print(f'  T_material (avg):    {state.temperature.mean():.6f} keV')
print(f'  T_radiation (avg):   {state.radiation_temperature.mean():.6f} keV')
print(f'  Internal energy:     {np.sum(state.internal_energy) * cell_vol:.6e} GJ')
print(f'  Radiation energy:    {np.sum(state.weights):.6e} GJ')
print(f'  Total energy:        {np.sum(state.internal_energy) * cell_vol + np.sum(state.weights):.6e} GJ')

# Run a single step with left boundary source at 1 keV
T_boundary = (1.0, 0.0, 0.0, 0.0)
reflect = (False, True, True, True)
dt = 0.01

print(f'\nRunning timestep (dt={dt} ns, T_bc={T_boundary[0]} keV)...')
state, info = imc2d_cf.step(
    state, 5000, 2000, 0, 20000, T_boundary, dt, 
    x_edges, y_edges, sigma_a_f, inv_eos, cv, source,
    reflect=reflect, geometry='xy', max_events_per_particle=1000
)

print(f'\nAfter Step:')
print(f'  T_material (avg):    {state.temperature.mean():.6f} keV')
print(f'  T_radiation (avg):   {state.radiation_temperature.mean():.6f} keV')
print(f'  Internal energy:     {info["total_internal_energy"]:.6e} GJ')
print(f'  Radiation energy:    {info["total_radiation_energy"]:.6e} GJ')
print(f'  Total energy:        {info["total_energy"]:.6e} GJ')
print(f'  Boundary emission:   {info["boundary_emission"]:.6e} GJ')
print(f'  Boundary loss:       {info["boundary_loss"]:.6e} GJ')
print(f'  Energy conservation: {info["energy_loss"]:.6e} GJ (should be ~0)')

print(f'\nTemperature Profile (x-direction, middle of domain):')
print(f'  {"x (cm)":<10} {"T_mat (keV)":<15} {"T_rad (keV)":<15}')
print(f'  {"-"*10:<10} {"-"*15:<15} {"-"*15:<15}')
for i in range(min(8, n)):
    x_center = 0.5 * (x_edges[i] + x_edges[i+1])
    T_mat = state.temperature[i, n//2]
    T_rad = state.radiation_temperature[i, n//2]
    print(f'  {x_center:<10.5f} {T_mat:<15.6f} {T_rad:<15.6f}')

print(f'\n{"="*60}')
print('DIAGNOSIS:')
if state.temperature.mean() > 1.1 * Tinit.mean():
    print('✓ Material is heating up - energy deposition working!')
else:
    print('✗ Material NOT heating up - CF energy deposition may be broken!')
    print('  - Check: Are particles being absorbed and truly depositing energy?')
    print('  - Check: Are re-emission times sampled correctly?')
    print('  - Check: Is emitted_energies calculation correct?')
