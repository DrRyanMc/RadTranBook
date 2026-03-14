"""Quick correctness check: one step of serial vs parallel solver.

Builds a small synthetic 4-group 8x8 cylindrical problem so we don't depend
on problem-file module-level exports.
"""
import numpy as np
import sys
sys.path.insert(0, '.')

from multigroup_diffusion_solver_2d import MultigroupDiffusionSolver2D
from multigroup_diffusion_solver_2d_parallel import MultigroupDiffusionSolver2DParallel

# ---- tiny synthetic problem ----
N_GROUPS = 4
NX, NY = 8, 8
DT = 1e-3

energy_edges = np.array([0.001, 0.01, 0.1, 1.0, 10.0])

def kappa_g(T, x, y, g=0):  return 10.0
def D_g(T, x, y, g=0):      return 1.0 / (3.0 * 10.0)
def mat_e(T, x, y):          return 0.5 * T**4
def inv_mat_e(e, x, y):      return (2.0 * e)**0.25
def cv(T, x, y):             return 2.0 * T**3

absorption_coeff_funcs = [lambda T, x, y, g=g: kappa_g(T, x, y, g) for g in range(N_GROUPS)]
diffusion_coeff_funcs  = [lambda T, x, y, g=g: D_g(T, x, y, g)     for g in range(N_GROUPS)]

def bc_marshak_g(phi, pos, t): return 0.5, 1.0, 0.5  # simple Marshak-like
def bc_zero(phi, pos, t):      return 0.0, 1.0, 0.0  # zero flux (Neumann)

boundary_funcs = {
    'left':   [bc_marshak_g] * N_GROUPS,
    'right':  [bc_zero]      * N_GROUPS,
    'bottom': [bc_zero]      * N_GROUPS,
    'top':    [bc_zero]      * N_GROUPS,
}

KWARGS = dict(
    n_groups=N_GROUPS,
    x_min=0.0, x_max=1.0, nx_cells=NX,
    y_min=0.0, y_max=1.0, ny_cells=NY,
    energy_edges=energy_edges,
    absorption_coeff_funcs=absorption_coeff_funcs,
    diffusion_coeff_funcs=diffusion_coeff_funcs,
    material_energy_func=mat_e,
    inverse_material_energy_func=inv_mat_e,
    cv=cv,
    boundary_funcs=boundary_funcs,
    geometry='cartesian',
    dt=DT,
)

serial   = MultigroupDiffusionSolver2D(**KWARGS)
parallel = MultigroupDiffusionSolver2DParallel(**KWARGS, n_threads=4)

serial.step()
parallel.step()

T_s  = serial.T
T_p  = parallel.T
Er_s = serial.E_r
Er_p = parallel.E_r

T_err  = np.max(np.abs(T_s  - T_p ) / (np.abs(T_s)  + 1e-30))
Er_err = np.max(np.abs(Er_s - Er_p) / (np.abs(Er_s) + 1e-30))
print(f"Max relative T error:   {T_err:.3e}")
print(f"Max relative E_r error: {Er_err:.3e}")

tol = 1e-8
assert T_err  < tol, f"T mismatch: {T_err}"
assert Er_err < tol, f"Er mismatch: {Er_err}"
print("PASS: parallel results match serial to < 1e-8 relative error")
parallel.close()
