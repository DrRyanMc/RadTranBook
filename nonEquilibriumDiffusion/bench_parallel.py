import numpy as np, sys, time
sys.path.insert(0, '.')
from multigroup_diffusion_solver_2d import MultigroupDiffusionSolver2D
from multigroup_diffusion_solver_2d_parallel import MultigroupDiffusionSolver2DParallel

N_GROUPS = 10
NX, NY = 60, 210           # production grid size
energy_edges = np.logspace(np.log10(1e-4), np.log10(25.0), N_GROUPS + 1)

def kappa_g(T, x, y): return 10.0
def D_g(T, x, y):     return 1.0 / 30.0
def mat_e(T, x, y):   return 0.5 * T**4
def inv_mat_e(e, x, y): return (2.0 * e)**0.25
def cv_fn(T, x, y): return 2.0 * T**3
def bc_zero(phi, pos, t): return 0.0, 1.0, 0.0
def bc_in(phi, pos, t):   return 0.5, 1.0, 0.5

kappa_funcs = [lambda T, x, y: kappa_g(T, x, y)] * N_GROUPS
D_funcs     = [lambda T, x, y: D_g(T, x, y)]     * N_GROUPS
bfuncs = {
    'left':   [bc_in]   * N_GROUPS,
    'right':  [bc_zero] * N_GROUPS,
    'bottom': [bc_zero] * N_GROUPS,
    'top':    [bc_zero] * N_GROUPS,
}

KWARGS = {
    'n_groups': N_GROUPS,
    'x_min': 0.0, 'x_max': 1.0, 'nx_cells': NX,
    'y_min': 0.0, 'y_max': 1.0, 'ny_cells': NY,
    'energy_edges': energy_edges,
    'absorption_coeff_funcs': kappa_funcs,
    'diffusion_coeff_funcs': D_funcs,
    'material_energy_func': mat_e,
    'inverse_material_energy_func': inv_mat_e,
    'cv': cv_fn,
    'boundary_funcs': bfuncs,
    'geometry': 'cartesian',
    'dt': 1e-3,
}

N_STEPS = 2

serial = MultigroupDiffusionSolver2D(**KWARGS)
serial.step()   # warm up
t0 = time.time()
for _ in range(N_STEPS):
    serial.step()
t_serial = time.time() - t0
print(f"Serial        {N_STEPS} steps: {t_serial:.2f}s")

for n_threads in [2, 4, 6, 10]:
    p = MultigroupDiffusionSolver2DParallel(**KWARGS, n_threads=n_threads)
    p.step()
    t0 = time.time()
    for _ in range(N_STEPS):
        p.step()
    t_p = time.time() - t0
    p.close()
    print(f"Parallel ({n_threads:2d}t) {N_STEPS} steps: {t_p:.2f}s  speedup {t_serial/t_p:.1f}x")
