"""Micro-benchmark: assemble_matrix in isolation, vectorize vs JIT parallel."""
import numpy as np, sys, time
sys.path.insert(0, '.')
from numba import njit
from diffusion_operator_solver import DiffusionOperatorSolver2D

NX, NY = 60, 210

def D_py(T, x, y):     return 1.0 / (3.0 * 0.2 * T**(-3.5))
def kappa_py(T, x, y): return 0.2 * T**(-3.5)

@njit(cache=True)
def D_jit(T, x, y):     return 1.0 / (3.0 * 0.2 * T**(-3.5))

@njit(cache=True)
def kappa_jit(T, x, y): return 0.2 * T**(-3.5)

KARGS = dict(x_min=0.0, x_max=1.0, nx_cells=NX,
             y_min=0.0, y_max=1.0, ny_cells=NY,
             geometry='cartesian', dt=1e-3)

s_py  = DiffusionOperatorSolver2D(diffusion_coeff_func=D_py,  absorption_coeff_func=kappa_py,  **KARGS)
s_jit = DiffusionOperatorSolver2D(diffusion_coeff_func=D_jit, absorption_coeff_func=kappa_jit, **KARGS)

T = np.full((NX, NY), 1.0)

# warm-up (includes Numba compile)
for _ in range(3): s_py.assemble_matrix(T)
for _ in range(3): s_jit.assemble_matrix(T)

N = 20
t0 = time.time()
for _ in range(N): s_py.assemble_matrix(T)
t_py = time.time() - t0

t0 = time.time()
for _ in range(N): s_jit.assemble_matrix(T)
t_jit = time.time() - t0

print(f"np.vectorize  {N} calls: {t_py*1e3:.1f} ms  ({t_py/N*1e3:.2f} ms/call)")
print(f"@njit parallel {N} calls: {t_jit*1e3:.1f} ms  ({t_jit/N*1e3:.2f} ms/call)")
print(f"Speedup: {t_py/t_jit:.1f}x")
print(f"_use_numba_eval: py={s_py._use_numba_eval}, jit={s_jit._use_numba_eval}")
