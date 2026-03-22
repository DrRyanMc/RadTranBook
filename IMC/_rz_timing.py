#!/usr/bin/env python3
"""Time each phase of a 2D step for the MarshakWave2D default rz parameters."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import IMC2D as imc2d

CV_VAL = 0.3
def sigma_a_f(T): return 300.0 * np.maximum(T, 1e-4)**-3
def eos(T):      return CV_VAL * T
def inv_eos(u):  return u / CV_VAL
def cv(T):       return 0.0*T + CV_VAL

# ---- warm up by running one xy step so JIT is compiled ----
print("Warming up with xy step...", flush=True)
x_e = np.linspace(0.0, 0.2, 61)
y_e = np.linspace(0.0, 0.2, 61)
s_xy = imc2d.init_simulation(20000, np.full((60,60),1e-4), np.full((60,60),1e-4),
                              x_e, y_e, eos, inv_eos, geometry='xy')
s_xy, _ = imc2d.step(s_xy, 20000, 12000, 0, 80000, (1.,0.,0.,0.), 0.01,
                     x_e, y_e, sigma_a_f, inv_eos, cv, np.zeros((60,60)),
                     reflect=(False,True,True,True), geometry='xy')
print("xy warmup done", flush=True)

# ---- now run rz with the defaults ----
r0, dr, nr, nz = 10.0, 0.2, 10, 60
r_edges = np.linspace(r0, r0+dr, nr+1)
z_edges = np.linspace(0.0, 0.2, nz+1)
Tinit  = np.full((nr, nz), 1e-4)
Trinit = np.full((nr, nz), 1e-4)
source = np.zeros((nr, nz))

state = imc2d.init_simulation(24000, Tinit, Trinit, r_edges, z_edges, eos, inv_eos, geometry='rz')
print(f"rz init done: {nr}×{nz} grid", flush=True)

for step_num in range(1, 8):
    t0 = time.perf_counter()
    state, info = imc2d.step(
        state, 24000, 12000, 0, 120000,
        (0.,0.,1.,0.), 0.01, r_edges, z_edges,
        sigma_a_f, inv_eos, cv, source,
        reflect=(True, True, False, True), geometry='rz')
    elapsed = time.perf_counter() - t0
    print(f"step {step_num}: {elapsed:.2f}s  N={info['N_particles']:6d}  "
          f"T_max={state.temperature.max():.4f}", flush=True)
