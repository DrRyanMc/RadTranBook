import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multigroup_diffusion_solver import MultigroupDiffusionSolver1D, C_LIGHT, A_RAD, Bg_multigroup

r_min, r_max = 0.0, 2.0e-3
n_cells, dt, n_steps = 20, 1.0e-3, 300
rho, cv, sigma = 1.0, 0.05, 100.0
n_groups = 5
energy_edges = np.logspace(np.log10(1e-4), np.log10(5.0), n_groups + 1)
T_cold, T_bc = 0.01, 1.0


def D_func(T, r=0.0):
    return C_LIGHT / (3.0 * sigma)


def sigma_func(T, r=0.0):
    return sigma


diff_funcs = [D_func] * n_groups
sigma_funcs = [sigma_func] * n_groups

B_g_bc = Bg_multigroup(energy_edges, T_bc)
chi_bc = B_g_bc / B_g_bc.sum()
F_total_bc = (A_RAD * C_LIGHT * T_bc**4) / 2.0
F_g_bc = chi_bc * F_total_bc


def make_incoming_bc(g):
    D = D_func(T_bc)
    F_inc_g = F_g_bc[g]

    def bc(phi, r):
        return 0.5, 2.0 * D, F_inc_g

    return bc


def bc_reflecting(phi, r):
    return 0.0, 1.0, 0.0


left_bcs = [make_incoming_bc(g) for g in range(n_groups)]
right_bcs = [bc_reflecting] * n_groups

B_g_cold = Bg_multigroup(energy_edges, T_cold)
chi_cold = B_g_cold / B_g_cold.sum()

solver = MultigroupDiffusionSolver1D(
    n_groups=n_groups,
    r_min=r_min,
    r_max=r_max,
    n_cells=n_cells,
    energy_edges=energy_edges,
    geometry="planar",
    dt=dt,
    diffusion_coeff_funcs=diff_funcs,
    absorption_coeff_funcs=sigma_funcs,
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
solver.phi_g_stored[:, :] = (4.0 * np.pi * B_g_cold)[:, np.newaxis]

for _ in range(n_steps):
    solver.step()
    solver.advance_time()

idx = n_cells // 2
T_m = solver.T[idx]

abs_terms = np.zeros(n_groups)
em_terms = np.zeros(n_groups)
for g in range(n_groups):
    abs_terms[g] = solver.sigma_a[g, idx] * solver.phi_g_stored[g, idx]
    B_g_T = solver.planck_funcs[g](T_m)
    em_terms[g] = solver.sigma_a[g, idx] * 4.0 * np.pi * B_g_T

a_frac = abs_terms / abs_terms.sum()
e_frac = em_terms / em_terms.sum()
chi = solver.chi

print("chi (fixed):", np.array2string(chi, precision=6, suppress_small=True))
print("a_g (abs):  ", np.array2string(a_frac, precision=6, suppress_small=True))
print("e_g (emit): ", np.array2string(e_frac, precision=6, suppress_small=True))
print("max|a-e|   =", np.max(np.abs(a_frac - e_frac)))
print("max|chi-e| =", np.max(np.abs(chi - e_frac)))
print("max|chi-a| =", np.max(np.abs(chi - a_frac)))
