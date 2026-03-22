#!/usr/bin/env python3
"""Reproduce exactly what MarshakWave2D.py does for rz, with per-step timing."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import IMC1D as imc1d
import IMC2D as imc2d

CV_VAL = 0.3
def sigma_a_f(T): return 300.0 * np.maximum(T, 1e-4)**-3
def eos(T):      return CV_VAL * T
def inv_eos(u):  return u / CV_VAL
def cv(T):       return 0.0*T + CV_VAL

# Warm up the JIT with an xy step, same as MarshakWave2D.py does first
print("=== xy warmup (60x60 grid, 1 step) ===", flush=True)
x_e = np.linspace(0.0, 0.2, 61); y_e = np.linspace(0.0, 0.2, 61)
s_xy = imc2d.init_simulation(20000, np.full((60,60),1e-4), np.full((60,60),1e-4),
                              x_e, y_e, eos, inv_eos, geometry='xy')
t0 = time.perf_counter()
s_xy, _ = imc2d.step(s_xy, 20000, 12000, 0, 80000, (1.,0.,0.,0.), 0.01,
                     x_e, y_e, sigma_a_f, inv_eos, cv, np.zeros((60,60)),
                     reflect=(False,True,True,True), geometry='xy',
                     max_events_per_particle=1_000_000)
print(f"xy warmup done: {time.perf_counter()-t0:.2f}s", flush=True)

# Exact parameters from run_rz_planar_limit_check defaults
r0, dr, nr, nz = 10.0, 0.2, 10, 60
r_edges = np.linspace(r0, r0+dr, nr+1)
z_edges = np.linspace(0.0, 0.2, nz+1)
Tinit  = np.full((nr, nz), 1e-4)
source = np.zeros((nr, nz))

print(f"\n=== rz steps: nr={nr} nz={nz}, Ntarget=24000, Nmax=120000, "
      f"max_events=1_000_000 ===", flush=True)

state = imc2d.init_simulation(24000, Tinit, Tinit.copy(), r_edges, z_edges, eos, inv_eos, geometry='rz')

t_total = 0.0
for step_num in range(1, 21):
    t0 = time.perf_counter()
    state, info = imc2d.step(
        state, 24000, 12000, 0, 120000,
        (0.,0.,1.,0.), 0.01, r_edges, z_edges,
        sigma_a_f, inv_eos, cv, source,
        reflect=(True, True, False, True), geometry='rz',
        max_events_per_particle=1_000_000)
    elapsed = time.perf_counter() - t0
    t_total += elapsed
    print(f"step {step_num:2d}: {elapsed:.2f}s  N={info['N_particles']:6d}  "
          f"T_max={state.temperature.max():.4f}  cumul={t_total:.1f}s", flush=True)

print(f"\nTotal for 20 rz steps: {t_total:.1f}s")
