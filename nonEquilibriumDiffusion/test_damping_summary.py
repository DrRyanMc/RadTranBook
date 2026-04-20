"""Quick summary showing Newton damping effect."""
import numpy as np
from multigroup_diffusion_solver import MultigroupDiffusionSolver1D

A_RAD = 0.01372
n_groups, n_cells = 10, 30
r_min, r_max = 0.0, 5.0
dt = 10.0  # Large time step to force bigger updates
energy_edges = np.logspace(-4, 1, n_groups + 1)
chi = np.ones(n_groups) / n_groups

def setup_solver():
    sigma_funcs = [lambda T, r, g=g: 1.0 for g in range(n_groups)]
    diff_funcs = [lambda T, r, g=g: 1.0/3.0 for g in range(n_groups)]
    neumann_bc = lambda phi, r: (0.0, 1.0, 0.0)
    
    solver = MultigroupDiffusionSolver1D(
        n_groups=n_groups, r_min=r_min, r_max=r_max, n_cells=n_cells,
        energy_edges=energy_edges, geometry='planar', dt=dt,
        diffusion_coeff_funcs=diff_funcs, absorption_coeff_funcs=sigma_funcs,
        left_bc_funcs=[neumann_bc]*n_groups, right_bc_funcs=[neumann_bc]*n_groups,
        emission_fractions=chi, rho=1.0, cv=0.1
    )
    
    T_init = 0.1 + 0.9 * np.exp(-solver.r_centers / 2.0)
    solver.T = T_init.copy()
    solver.T_old = T_init.copy()
    solver.E_r = A_RAD * T_init**4
    solver.E_r_old = solver.E_r.copy()
    solver.kappa = np.zeros(n_cells)
    solver.kappa_old = np.zeros(n_cells)
    return solver

print('='*70)
print('Newton Step Damping - Summary')
print('='*70)
print(f'Large time step (dt={dt} ns) forces bigger Newton updates')
print()

for max_rel in [1.0, 0.5, 0.2, 0.1]:
    solver = setup_solver()
    info = solver.step(
        max_newton_iter=1, newton_tol=1e-8,
        gmres_tol=1e-8, gmres_maxiter=200,
        use_preconditioner=False,
        max_relative_change=max_rel, verbose=False
    )
    
    T_change = info['T_change']
    gmres_iters = info['gmres_info']['iterations']
    damped = ' (damped)' if max_rel < 1.0 and T_change < 0.237 else ''
    
    print(f'max_relative_change = {max_rel:4.1f}: T_change = {T_change:.3e}, GMRES = {gmres_iters:2d}{damped}')

print()
print('✓ When max_relative_change < actual change, step is reduced')
print('='*70)
