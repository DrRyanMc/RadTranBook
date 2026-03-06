"""
Cold-to-hot incoming Marshak BC test with power-law group opacities.

Uses:
    sigma_a(T, E) = 10 * rho * T^(-1/2) * E^(-3)
with group opacities defined by geometric mean at group boundaries.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multigroup_diffusion_solver import (
    MultigroupDiffusionSolver1D,
    C_LIGHT,
    A_RAD,
    Bg_multigroup,
)
from marshak_wave_multigroup_powerlaw import make_powerlaw_opacity_func


# Setup
r_min, r_max = 0.0, 2.0e-3
n_cells = 20
dt = 1.0e-3  # ns
n_steps = 300
rho, cv = 1.0, 0.05

# Groups
n_groups = 5
energy_edges = np.logspace(np.log10(1e-4), np.log10(20.0), n_groups + 1)

# Initial and BC temperatures
T_cold = 0.01
T_bc = 1.0


def make_powerlaw_diffusion_from_opacity(opacity_func):
    def D_func(T, r=0.0):
        sigma = opacity_func(T, r)
        return C_LIGHT / (3.0 * sigma)

    return D_func


# Per-group opacity/diffusion functions from power-law model
sigma_funcs = []
diff_funcs = []
for g in range(n_groups):
    E_low = energy_edges[g]
    E_high = energy_edges[g + 1]
    sigma_g = make_powerlaw_opacity_func(E_low, E_high, rho)
    D_g = make_powerlaw_diffusion_from_opacity(sigma_g)
    sigma_funcs.append(sigma_g)
    diff_funcs.append(D_g)

# Incoming BB BC by group
B_g_bc = Bg_multigroup(energy_edges, T_bc)
chi_bc = B_g_bc / B_g_bc.sum()
F_total_bc = (A_RAD * C_LIGHT * T_bc**4) / 2.0
F_g_bc = chi_bc * F_total_bc


def make_incoming_bc(g):
    D_g = diff_funcs[g](T_bc, 0.0)
    F_inc_g = F_g_bc[g]

    def bc_incoming(phi, r):
        return 0.5, 2.0 * D_g, F_inc_g

    return bc_incoming


def bc_reflecting(phi, r):
    return 0.0, 1.0, 0.0


left_bcs = [make_incoming_bc(g) for g in range(n_groups)]
right_bcs = [bc_reflecting] * n_groups

# Initialize with cold-state emission fractions
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

print("=" * 100)
print(f"RUNNING POWER-LAW OPACITY TEST: n_steps={n_steps}, dt={dt} ns, n_cells={n_cells}")
print("opacity law: sigma_g(T) from geometric-mean boundaries of 10*rho*T^(-1/2)*E^(-3)")
print("=" * 100)
print(" step | t(ns) | T_left | T_mid | T_right | Trad_left | Trad_mid | Trad_right")
print("-" * 100)

for step in range(n_steps):
    solver.step()
    solver.advance_time()

    if step < 5 or (step + 1) % 50 == 0 or step == n_steps - 1:
        i_mid = n_cells // 2
        T_left = solver.T[0]
        T_mid = solver.T[i_mid]
        T_right = solver.T[-1]
        Trad_left = (solver.E_r[0] / A_RAD) ** 0.25
        Trad_mid = (solver.E_r[i_mid] / A_RAD) ** 0.25
        Trad_right = (solver.E_r[-1] / A_RAD) ** 0.25
        print(
            f" {step+1:4d} | {solver.t:5.3f} | {T_left:6.4f} | {T_mid:6.4f} | {T_right:7.4f} |"
            f" {Trad_left:9.4f} | {Trad_mid:8.4f} | {Trad_right:10.4f}"
        )

# Final diagnostics
T_mat = solver.T.copy()
T_rad = (solver.E_r / A_RAD) ** 0.25
T_diff = np.abs(T_mat - T_rad)
ratio_E = solver.E_r / (A_RAD * T_mat**4)

print("\n" + "=" * 100)
print("FINAL DIAGNOSTICS")
print("=" * 100)
print(f"max |T_mat-T_rad| = {T_diff.max():.6e} keV")
print(f"mean|T_mat-T_rad| = {T_diff.mean():.6e} keV")
print(f"E_r/(aT^4): min={ratio_E.min():.6f}, max={ratio_E.max():.6f}, mean={ratio_E.mean():.6f}")

print("\nCell samples (left, mid, right):")
for idx in [0, n_cells // 2, n_cells - 1]:
    print(
        f"  cell {idx:2d}: Tm={T_mat[idx]:.7f}, Tr={T_rad[idx]:.7f}, |dT|={T_diff[idx]:.6e}, "
        f"ratio E_r/(aT^4)={ratio_E[idx]:.6f}"
    )
