#!/usr/bin/env python3
"""
Diagnostic: examine angular (mu) distribution of census particles in cells 0-3
from the t=0.90 to t=0.99 snapshots, and also track E_total growth and desired.

Since snapshots don't have particle arrays, we'll:
1. Compare E_total (all cells) across snapshots to get the E_total growth  
2. Estimate desired[cell0] = round(Nmax * E_cell0 / E_total) and ew_target
3. Run a SHORT re-simulation from t=0 with Ntarget=1000 (fast!) to expose the
   mu-distribution change at step 94.

Goal: figure out WHY T_r_si drops at cell 0 while T_r_cen rises.
"""
import sys
import os
import numpy as np

sys.path.insert(0, '/Users/rmcclarr/Dropbox/Papers/RadTranBook')

# ── Part 1: Analyse snapshot E_total and desired estimates ────────────────────
snap_dir = '/Users/rmcclarr/Dropbox/Papers/RadTranBook/results/dilute_spectrum_shell/imc_32g_standard'

Nmax = 4_300_000  # from code (Nmax_final for standard mode)
a_rad = 0.01372   # GJ/(cm^3 keV^4)
c_light = 29.979  # cm/ns

snaps = sorted([f for f in os.listdir(snap_dir) if f.startswith('snapshot_t_0.9')])
snaps = [os.path.join(snap_dir, f) for f in snaps]

print("=" * 72)
print("E_total and desired[cell0] evolution")
print(f"{'t(ns)':>7} {'E_total':>12} {'E_cell0':>12} {'desired_c0':>12} {'ew_tgt_c0':>12} {'T_r_si[0]':>10} {'T_r_cen[0]':>11}")

for f in snaps:
    d = np.load(f)
    t = float(d['time'])
    
    # Total radiation energy: sum over all cells and groups
    # E_rad_by_group shape (n_groups, n_cells), E_rad = E_rad_by_group / volumes
    # But we have E_rad (energy density), and r_edges
    r_edges = d['r_edges']
    r0s = r_edges[:-1]; r1s = r_edges[1:]
    volumes = (4/3)*np.pi*(r1s**3 - r0s**3)
    
    E_rad_by_group = d['E_rad_by_group']  # GJ/cm^3, shape (n_groups, n_cells)
    E_total_rad = np.sum(E_rad_by_group * volumes[np.newaxis, :])  # GJ
    
    # Also add material internal energy (approximate: rho * cv * T_mat * volume)
    # We don't have rho and cv directly... skip material energy for now
    # Just use radiation energy
    
    E_cell0 = np.sum(E_rad_by_group[:, 0]) * volumes[0]  # GJ
    
    desired_c0 = round(Nmax * E_cell0 / E_total_rad) if E_total_rad > 0 else 0
    ew_tgt = E_cell0 / desired_c0 if desired_c0 > 0 else 0
    
    T_r_si = float(d['T_rad'][0])
    T_r_cen = (np.sum(E_rad_by_group[:, 0]) / a_rad) ** 0.25
    
    print(f"  {t:5.3f}  {E_total_rad:12.5e}  {E_cell0:12.5e}  {desired_c0:12d}  "
          f"{ew_tgt:12.3e}  {T_r_si:10.4f}  {T_r_cen:11.4f}")

# ── Part 2: Run INSTRUMENTED simulation with tiny particle count ────────────
print()
print("=" * 72)
print("Running TINY instrumented simulation (Ntarget=200) to expose mu distribution")
print("=" * 72)

# Patch the step() function to collect per-step diagnostics
import random as _random
import math

# Import MG_IMC
from MG_IMC.MG_IMC1D import SimulationState1DMG, init_simulation
import MG_IMC.MG_IMC1D as _imc

from MG_IMC.problems.dilute_spectrum_shell import (
    make_mesh, make_energy_edges, make_sigma_a_funcs, make_eos_functions,
    T_S, T_INIT, T_FINAL, DT_DEFAULT, DUMP_TIMES,
)

mesh, r_centers, rho_per_cell = make_mesh(mode='quick')
n_groups = 32
energy_edges = make_energy_edges(n_groups)
sigma_a_funcs = make_sigma_a_funcs(energy_edges, rho_per_cell)
eos, inv_eos, cv_func = make_eos_functions(rho_per_cell)

n_cells = mesh.shape[0]
source = np.zeros(n_cells)
T_boundary = (T_S, 0.0)
reflect = (False, False)

state = init_simulation(mesh, T_INIT, sigma_a_funcs, energy_edges,
                        eos, inv_eos, cv_func)

# Tiny particle budget: Ntarget=200, Nmax_init=500, growth=50, max=1000
Ntarget   = 200
Nmax_init = 500
Nmax_growth = 50
Nmax_final = 2000

Nmax_current = Nmax_init
dt = DT_DEFAULT

print(f"\n{'step':>5} {'Npart':>8} {'N_cell0':>8} {'N_cell1':>8} "
      f"{'N_c1_muNeg':>12} {'E_c0(GJ)':>12} {'E_c1(GJ)':>12} "
      f"{'Tr_si0':>8} {'Tr_cen0':>8}")

for step_idx in range(101):
    t = state.time
    
    # Call step
    result = _imc.step(
        state=state,
        mesh=mesh,
        sigma_a_funcs=sigma_a_funcs,
        energy_edges=energy_edges,
        eos=eos,
        inv_eos=inv_eos,
        cv_func=cv_func,
        dt=dt,
        Ntarget=Ntarget,
        Nboundary=Ntarget,
        Nsource=0,
        Ntarget_emit=Ntarget,
        source=source,
        T_boundary=T_boundary,
        reflect=reflect,
        Nmax=Nmax_current,
        T_emit_floor=0.025,
        particle_budget_fmin=0.05,
        Nmax_growth=0,  # fixed Nmax for simplicity
        Nmax_final=Nmax_final,
        conserve_comb_energy=True,
    )
    
    # After step: analyze particle state
    w  = state.weights
    ci = state.cell_indices
    mu = state.mus
    
    n_total = len(w)
    
    # Particles in cell 0 and cell 1
    in_c0 = ci == 0
    in_c1 = ci == 1
    
    N_c0 = int(np.sum(in_c0))
    N_c1 = int(np.sum(in_c1))
    N_c1_muNeg = int(np.sum(in_c1 & (mu < 0)))  # inward-moving in cell 1
    
    E_c0 = float(np.sum(w[in_c0]))
    E_c1 = float(np.sum(w[in_c1]))
    
    Tr_si0  = float(state.radiation_temperature[0])
    E_rad_c0_cen = np.sum(state.radiation_energy_by_group[:, 0]) * ((4/3)*np.pi*(mesh[0,1]**3 - mesh[0,0]**3))
    Tr_cen0 = (E_rad_c0_cen / a_rad) ** 0.25
    
    if step_idx >= 88 or (step_idx % 10 == 0):
        print(f"  {step_idx+1:4d}  {n_total:8d}  {N_c0:8d}  {N_c1:8d}  "
              f"  {N_c1_muNeg:8d}    {E_c0:12.4e}  {E_c1:12.4e}  "
              f"  {Tr_si0:6.3f}   {Tr_cen0:6.3f}")
    
    # Grow Nmax
    Nmax_current = min(Nmax_current + Nmax_growth, Nmax_final)

print("\nDone.")
