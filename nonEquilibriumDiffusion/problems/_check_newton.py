"""Quick script to run a few Marshak wave steps with verbose Newton output."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from oneDFV import NonEquilibriumRadiationDiffusionSolver

C_LIGHT = 2.998e1
A_RAD   = 0.01372
RHO     = 1.0

def op(T):      return 300.0 * max(T, 0.01)**(-3)
def cv(T):      return 0.3 / RHO
def mat_e(T):   return RHO * (0.3 / RHO) * T
def inv_e(e):   return e / 0.3
def left_bc(phi, x):  return 1.0, 0.0, C_LIGHT * A_RAD * 1.0**4
def right_bc(phi, x): return 0.0, 1.0, 0.0

solver = NonEquilibriumRadiationDiffusionSolver(
    r_min=0.0, r_max=0.5, n_cells=100, d=0, dt=0.01,
    max_newton_iter=10, newton_tol=1e-8,
    rosseland_opacity_func=op, planck_opacity_func=op,
    specific_heat_func=cv, material_energy_func=mat_e,
    inverse_material_energy_func=inv_e,
    left_bc_func=left_bc, right_bc_func=right_bc, theta=1.0
)

T_init = 0.01
solver.phi = np.full(100, C_LIGHT * A_RAD * T_init**4)
solver.T   = np.full(100, T_init)

solver.time_step(n_steps=5, verbose=True)
