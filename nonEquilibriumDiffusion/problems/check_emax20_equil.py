import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from multigroup_diffusion_solver import MultigroupDiffusionSolver1D, C_LIGHT, A_RAD, Bg_multigroup

r_min, r_max = 0.0, 2.0e-3
n_cells = 20
dt = 1.0e-3
n_steps = 300
rho, cv, sigma = 1.0, 0.05, 100.0
n_groups = 5
energy_edges = np.logspace(np.log10(1e-4), np.log10(20.0), n_groups + 1)
T_cold, T_bc = 0.01, 1.0


def D_func(T, r=0.0):
    return C_LIGHT / (3.0 * sigma)


def sigma_func(T, r=0.0):
    return sigma


B_g_bc = Bg_multigroup(energy_edges, T_bc)
chi_bc = B_g_bc / B_g_bc.sum()
F_total_bc = (A_RAD * C_LIGHT * T_bc**4) / 2.0
F_g_bc = chi_bc * F_total_bc


def make_incoming_bc(g):
    D = D_func(T_bc)
    C = F_g_bc[g]

    def bc_in(phi, r):
        return 0.5, 2.0 * D, C

    return bc_in


def bc_reflecting(phi, r):
    return 0.0, 1.0, 0.0


left_bcs = [make_incoming_bc(g) for g in range(n_groups)]
right_bcs = [bc_reflecting] * n_groups

B_cold = Bg_multigroup(energy_edges, T_cold)
chi_cold = B_cold / B_cold.sum()

solver = MultigroupDiffusionSolver1D(
    n_groups=n_groups,
    r_min=r_min,
    r_max=r_max,
    n_cells=n_cells,
    energy_edges=energy_edges,
    geometry='planar',
    dt=dt,
    diffusion_coeff_funcs=[D_func] * n_groups,
    absorption_coeff_funcs=[sigma_func] * n_groups,
    left_bc_funcs=left_bcs,
    right_bc_funcs=right_bcs,
    emission_fractions=chi_cold,
    rho=rho,
    cv=cv,
)
solver._debug_update_T = False

E_r_cold = A_RAD * T_cold**4
solver.T[:] = T_cold
solver.T_old[:] = T_cold
solver.E_r[:] = E_r_cold
solver.E_r_old[:] = E_r_cold
solver.phi_g_fraction[:, :] = chi_cold[:, np.newaxis]
solver.phi_g_stored[:, :] = (4.0 * np.pi * B_cold)[:, np.newaxis]

for _ in range(n_steps):
    solver.step()
    solver.advance_time()

T_mat = solver.T
T_rad = (solver.E_r / A_RAD) ** 0.25
ratio = solver.E_r / (A_RAD * T_mat**4)

print('Emax=20 keV results:')
print(f'max|Tmat-Trad| = {np.max(np.abs(T_mat - T_rad)):.6e} keV')
print(f'mean|Tmat-Trad| = {np.mean(np.abs(T_mat - T_rad)):.6e} keV')
print(f'left  Tm,Tr = {T_mat[0]:.9f}, {T_rad[0]:.9f}')
print(f'mid   Tm,Tr = {T_mat[n_cells//2]:.9f}, {T_rad[n_cells//2]:.9f}')
print(f'ratio E_r/(aT^4): min={ratio.min():.9f}, max={ratio.max():.9f}')
