#!/usr/bin/env python3
"""Quick benchmark: xy vs rz with cold-material Marshak physics."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import IMC2D as imc2d

CV_VAL = 0.3
def sigma_a_f(T): return 300.0 * np.maximum(T, 1e-4)**-3
def eos(T): return CV_VAL * T
def inv_eos(u): return u / CV_VAL
def cv(T): return 0.0*T + CV_VAL

L=0.20; NX=60; DT=0.01; NTARGET=10000; NBOUNDARY=10000; NMAX=60000; N_STEPS=5

# --- 2D xy  (x=propagation, single y strip) ---
x_e = np.linspace(0, L, NX+1)
y_e = np.linspace(0, L/NX, 2)
s_xy = imc2d.init_simulation(NTARGET, np.full((NX,1),1e-4), np.full((NX,1),1e-4),
                              x_e, y_e, eos, inv_eos, geometry='xy')
for _ in range(2):  # warm-up / JIT compile
    s_xy, _ = imc2d.step(s_xy, NTARGET, NBOUNDARY, 0, NMAX,
                         (1.,0.,0.,0.), DT, x_e, y_e,
                         sigma_a_f, inv_eos, cv, np.zeros((NX,1)),
                         reflect=(False,True,True,True), geometry='xy')

t0 = time.perf_counter()
for _ in range(N_STEPS):
    s_xy, _ = imc2d.step(s_xy, NTARGET, NBOUNDARY, 0, NMAX,
                         (1.,0.,0.,0.), DT, x_e, y_e,
                         sigma_a_f, inv_eos, cv, np.zeros((NX,1)),
                         reflect=(False,True,True,True), geometry='xy')
t_xy = (time.perf_counter() - t0) / N_STEPS

# --- 2D rz  (z=propagation, single r annulus at large radius) ---
r0 = 10.0
r_e = np.linspace(r0, r0 + L/NX, 2)
z_e = np.linspace(0, L, NX+1)
s_rz = imc2d.init_simulation(NTARGET, np.full((1,NX),1e-4), np.full((1,NX),1e-4),
                              r_e, z_e, eos, inv_eos, geometry='rz')
for _ in range(2):  # warm-up
    s_rz, _ = imc2d.step(s_rz, NTARGET, NBOUNDARY, 0, NMAX,
                         (0.,0.,1.,0.), DT, r_e, z_e,
                         sigma_a_f, inv_eos, cv, np.zeros((1,NX)),
                         reflect=(True,True,False,True), geometry='rz')

t0 = time.perf_counter()
for _ in range(N_STEPS):
    s_rz, _ = imc2d.step(s_rz, NTARGET, NBOUNDARY, 0, NMAX,
                         (0.,0.,1.,0.), DT, r_e, z_e,
                         sigma_a_f, inv_eos, cv, np.zeros((1,NX)),
                         reflect=(True,True,False,True), geometry='rz')
t_rz = (time.perf_counter() - t0) / N_STEPS

print(f"Marshak-wave physics, NX={NX}, Ntarget={NTARGET}, dt={DT} ns, {N_STEPS} steps")
print(f"  xy : {t_xy*1000:.1f} ms/step")
print(f"  rz : {t_rz*1000:.1f} ms/step   ratio: {t_rz/t_xy:.2f}x")
