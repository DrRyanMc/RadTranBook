#!/usr/bin/env python3
"""
Diagnose high-temperature behavior in marshak_wave_multigroup_powerlaw.py
using a constant boundary drive.

Runs two variants with identical geometry/time stepping:
  1) original_style: mirrors marshak_wave_multigroup_powerlaw choices
  2) test_style: mirrors the validated cold-to-hot test choices
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
    compute_emission_fractions_from_edges,
)
from marshak_wave_multigroup_powerlaw import make_powerlaw_opacity_func


def run_case(case_name: str, use_original_style: bool):
    rho = 1.0
    cv = 0.05 / rho

    n_groups = 10
    energy_edges = np.logspace(np.log10(1e-4), np.log10(25.0), n_groups + 1)

    r_min, r_max = 0.0, 1.0e-1
    n_cells = 50
    dt = 1.0e-2
    n_steps = 600

    T_init = 0.05
    T_bc = 1.0

    sigma_funcs = []
    diff_funcs = []
    for g in range(n_groups):
        E_low = energy_edges[g]
        E_high = energy_edges[g + 1]
        sigma_g = make_powerlaw_opacity_func(E_low, E_high, rho)
        sigma_funcs.append(sigma_g)

        if use_original_style:
            def make_D_orig(sigf):
                def D(T, r=0.0):
                    return 1.0 / (3.0 * sigf(T, r))
                return D

            diff_funcs.append(make_D_orig(sigma_g))
        else:
            def make_D_test(sigf):
                def D(T, r=0.0):
                    return C_LIGHT / (3.0 * sigf(T, r))
                return D

            diff_funcs.append(make_D_test(sigma_g))

    if use_original_style:
        sigma_at_bc = np.array([sigma_funcs[g](T_bc, 0.0) for g in range(n_groups)])
        chi = compute_emission_fractions_from_edges(
            energy_edges, T_ref=T_bc, sigma_a_groups=sigma_at_bc
        )
    else:
        B_g_bc = Bg_multigroup(energy_edges, T_bc)
        chi = B_g_bc / B_g_bc.sum()

    F_total = (A_RAD * C_LIGHT * T_bc**4) / 2.0
    F_g = chi * F_total

    def make_left_bc(g):
        D_g = diff_funcs[g](T_bc, 0.0)
        if use_original_style:
            B_coeff = D_g
        else:
            B_coeff = 2.0 * D_g

        C_coeff = F_g[g]

        def left_bc(phi, r):
            return 0.5, B_coeff, C_coeff

        return left_bc

    def right_bc(phi, r):
        return 0.0, 1.0, 0.0

    left_bcs = [make_left_bc(g) for g in range(n_groups)]
    right_bcs = [right_bc] * n_groups

    B_g_init = Bg_multigroup(energy_edges, T_init)
    chi_init = B_g_init / B_g_init.sum()

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
        emission_fractions=chi_init,
        rho=rho,
        cv=cv,
    )

    solver._debug_update_T = False
    solver.T[:] = T_init
    solver.T_old[:] = T_init
    solver.E_r[:] = A_RAD * T_init**4
    solver.E_r_old[:] = solver.E_r.copy()

    print("\n" + "=" * 110)
    print(f"CASE: {case_name}")
    print(
        f"style={'original' if use_original_style else 'test'}, "
        f"T_bc={T_bc}, dt={dt}, n_steps={n_steps}, n_cells={n_cells}, n_groups={n_groups}"
    )
    print("=" * 110)
    print(" step | t(ns) | T_max | T_left | Trad_left | E_r/(aT^4)_left")
    print("-" * 110)

    for step in range(n_steps):
        solver.step()
        solver.advance_time()

        if step < 5 or (step + 1) % 50 == 0 or step == n_steps - 1:
            T_left = solver.T[0]
            Trad_left = (solver.E_r[0] / A_RAD) ** 0.25
            ratio_left = solver.E_r[0] / (A_RAD * max(T_left, 1e-14) ** 4)
            print(
                f" {step+1:4d} | {solver.t:6.3f} | {solver.T.max():7.4f} | {T_left:7.4f} |"
                f" {Trad_left:9.4f} | {ratio_left:14.6f}"
            )

    T_rad = (solver.E_r / A_RAD) ** 0.25
    dT = np.abs(solver.T - T_rad)
    ratio = solver.E_r / (A_RAD * np.maximum(solver.T, 1e-14) ** 4)

    print("\nFinal diagnostics:")
    print(f"  T_max={solver.T.max():.6f}, T_min={solver.T.min():.6f}")
    print(f"  max|T_mat-T_rad|={dT.max():.6e}, mean|T_mat-T_rad|={dT.mean():.6e}")
    print(f"  E_r/(aT^4): min={ratio.min():.6f}, max={ratio.max():.6f}, mean={ratio.mean():.6f}")


if __name__ == "__main__":
    run_case("original_style_constant_bc", use_original_style=True)
    run_case("test_style_constant_bc", use_original_style=False)
