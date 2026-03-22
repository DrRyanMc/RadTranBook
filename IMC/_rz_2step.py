#!/usr/bin/env python3
"""Run exactly 2 rz steps with the same parameters as run_rz_planar_limit_check defaults."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import IMC2D as imc2d

CV_VAL = 0.3
def sigma_a_f(T): return 300.0 * np.maximum(T, 1e-4)**-3
def eos(T):      return CV_VAL * T
def inv_eos(u):  return u / CV_VAL
def cv(T):       return 0.0*T + CV_VAL

r0, dr, nr, nz = 10.0, 0.2, 10, 60
Lz = 0.2
r_edges = np.linspace(r0, r0+dr, nr+1)
z_edges = np.linspace(0.0, Lz, nz+1)

Tinit  = np.full((nr, nz), 1e-4)
Trinit = np.full((nr, nz), 1e-4)
source = np.zeros((nr, nz))

state = imc2d.init_simulation(24000, Tinit, Trinit, r_edges, z_edges, eos, inv_eos, geometry='rz')
print('init done')

for step_num in range(1, 6):
    t0 = time.perf_counter()
    state, info = imc2d.step(
        state, 24000, 12000, 0, 120000,
        (0.,0.,1.,0.), 0.01, r_edges, z_edges,
        sigma_a_f, inv_eos, cv, source,
        reflect=(True, True, False, True), geometry='rz')
    elapsed = time.perf_counter() - t0
    print(f'step {step_num}: {elapsed:.2f}s  N={info["N_particles"]}  T_max={state.temperature.max():.4f}')
