"""
Benchmark the @njit fast-path in assemble_matrix vs np.vectorize path.
Uses constant D and kappa (JIT-friendly) so we can decorate with @njit.
"""
import numpy as np, sys, time
sys.path.insert(0, '.')
from numba import njit
from multigroup_diffusion_solver_2d import MultigroupDiffusionSolver2D

N_GROUPS = 10
NX, NY = 60, 210
energy_edges = np.logspace(np.log10(1e-4), np.log10(25.0), N_GROUPS + 1)

# ---- Plain Python functions (np.vectorize path) ----
def D_py(T, x, y):     return 1.0 / (3.0 * 0.2 * T**(-3.5))
def kappa_py(T, x, y): return 0.2 * T**(-3.5)
def mat_e(T, x, y):    return 0.5 * T**4
def inv_mat_e(e, x, y): return (2.0 * e)**0.25
def cv_fn(T, x, y):    return 2.0 * T**3

# ---- Numba @njit functions (fast-path) ----
@njit(cache=True)
def D_jit(T, x, y):     return 1.0 / (3.0 * 0.2 * T**(-3.5))

@njit(cache=True)
def kappa_jit(T, x, y): return 0.2 * T**(-3.5)

def bc_zero(phi, pos, t): return 0.0, 1.0, 0.0
def bc_in(phi, pos, t):   return 0.5, 1.0, 0.5

def make_solver(D_funcs, kappa_funcs, label):
    bfuncs = {
        'left':   [bc_in]   * N_GROUPS,
        'right':  [bc_zero] * N_GROUPS,
        'bottom': [bc_zero] * N_GROUPS,
        'top':    [bc_zero] * N_GROUPS,
    }
    return MultigroupDiffusionSolver2D(
        n_groups=N_GROUPS,
        x_min=0.0, x_max=1.0, nx_cells=NX,
        y_min=0.0, y_max=1.0, ny_cells=NY,
        energy_edges=energy_edges,
        absorption_coeff_funcs=kappa_funcs,
        diffusion_coeff_funcs=D_funcs,
        material_energy_func=mat_e,
        inverse_material_energy_func=inv_mat_e,
        cv=cv_fn,
        boundary_funcs=bfuncs,
        geometry='cartesian',
        dt=1e-3,
    )

N_STEPS = 2

print("Building solvers...")
solver_py  = make_solver([D_py]  * N_GROUPS, [kappa_py]  * N_GROUPS, "Python")
solver_jit = make_solver([D_jit] * N_GROUPS, [kappa_jit] * N_GROUPS, "JIT")

# check detection
print(f"  Python solver _use_numba_eval: {solver_py.solvers[0]._use_numba_eval}")
print(f"  JIT    solver _use_numba_eval: {solver_jit.solvers[0]._use_numba_eval}")

print("Warm-up (includes JIT compile)...")
solver_py.step()
solver_jit.step()

print(f"Timing {N_STEPS} steps each...")
t0 = time.time()
for _ in range(N_STEPS): solver_py.step()
t_py = time.time() - t0

t0 = time.time()
for _ in range(N_STEPS): solver_jit.step()
t_jit = time.time() - t0

print(f"\nnp.vectorize path   {N_STEPS} steps: {t_py:.2f}s")
print(f"@njit parallel path {N_STEPS} steps: {t_jit:.2f}s  speedup {t_py/t_jit:.1f}x")
