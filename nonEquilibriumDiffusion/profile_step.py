#!/usr/bin/env python3
"""Profile one time step of the crooked pipe at a configurable mesh size.

Usage:
    python profile_step.py               # small grid (20x35, 5 groups)
    python profile_step.py --medium      # medium grid (40x70, 10 groups)
    python profile_step.py --production  # production grid (60x210, 10 groups)
"""
import sys, os, numpy as np, cProfile, pstats, io
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'problems'))

from problems.crooked_pipe_multigroup_noneq import (
    make_powerlaw_opacity_func, make_powerlaw_diffusion_func,
    specific_heat, material_energy, inverse_material_energy,
    bc_left_axis,
)
from multigroup_diffusion_solver_2d_parallel import MultigroupDiffusionSolver2DParallel as Solver
from multigroup_diffusion_solver_2d import A_RAD, C_LIGHT, Bg_multigroup, flux_limiter_larsen

# Grid / group count selection
if '--production' in sys.argv:
    n_groups, NX, NY, n_threads = 10, 60, 210, 4
    label = 'production'
elif '--medium' in sys.argv:
    n_groups, NX, NY, n_threads = 10, 40, 70, 4
    label = 'medium'
else:
    n_groups, NX, NY, n_threads = 5, 20, 35, 2
    label = 'small'

energy_edges = np.logspace(np.log10(0.05), np.log10(10.0), n_groups + 1)

group_diffusion_funcs  = [make_powerlaw_diffusion_func(energy_edges[g], energy_edges[g + 1]) for g in range(n_groups)]
group_absorption_funcs = [make_powerlaw_opacity_func(energy_edges[g], energy_edges[g + 1])  for g in range(n_groups)]


def make_open_bc(g):
    def bc(phi, pos, t, **kw):
        r, z = pos
        D = group_diffusion_funcs[g](0.1, r, z)
        return 0.5, D, 0.0
    return bc


def make_bottom_bc(g):
    def bc(phi, pos, t, **kw):
        r, z = pos
        D = group_diffusion_funcs[g](0.1, r, 0.0)
        if r < 0.5:
            B_groups = Bg_multigroup(energy_edges, 0.3)
            chi = B_groups / np.sum(B_groups)
            return 0.5, D, chi[g] * (A_RAD * C_LIGHT * 0.3**4) / 2.0
        return 0.5, D, 0.0
    return bc


boundary_funcs = {
    'left':   [bc_left_axis] * n_groups,
    'right':  [make_open_bc(g) for g in range(n_groups)],
    'bottom': [make_bottom_bc(g) for g in range(n_groups)],
    'top':    [make_open_bc(g) for g in range(n_groups)],
}

solver = Solver(
    n_groups=n_groups, x_min=0, x_max=2, nx_cells=NX,
    y_min=0, y_max=7, ny_cells=NY,
    energy_edges=energy_edges, geometry='cylindrical', dt=1e-3,
    diffusion_coeff_funcs=group_diffusion_funcs,
    absorption_coeff_funcs=group_absorption_funcs,
    boundary_funcs=boundary_funcs,
    flux_limiter_funcs=flux_limiter_larsen,
    specific_heat_func=specific_heat,
    material_energy_func=material_energy,
    inverse_material_energy_func=inverse_material_energy,
    max_newton_iter=3, newton_tol=1e-6, theta=1.0, n_threads=n_threads,
)
print(f'_use_array_call: {solver.solvers[0]._use_array_call}   '
      f'_use_numba_eval: {solver.solvers[0]._use_numba_eval}')


T_cold = 0.01
solver.T[:] = T_cold
solver.T_old[:] = T_cold
solver.E_r[:] = A_RAD * T_cold**4
solver.E_r_old[:] = A_RAD * T_cold**4
B_cold = Bg_multigroup(energy_edges, T_cold)
chi_cold = B_cold / np.sum(B_cold)
solver.phi_g_fraction[:, :] = chi_cold[:, np.newaxis]
for g in range(n_groups):
    solver.phi_g_stored[g, :] = chi_cold[g] * A_RAD * T_cold**4 * C_LIGHT
solver.t = 0.0

print(f"Mesh: {NX}×{NY} = {NX*NY} cells, {n_groups} groups")

# Warm-up to trigger Numba JIT
print("Warming up…")
solver.step(gmres_tol=1e-6, gmres_maxiter=50)

# Profile
print("Profiling…")
pr = cProfile.Profile()
pr.enable()
solver.step(gmres_tol=1e-6, gmres_maxiter=50)
pr.disable()

buf = io.StringIO()
ps = pstats.Stats(pr, stream=buf).sort_stats('tottime')
ps.print_stats(30)
print(buf.getvalue())
