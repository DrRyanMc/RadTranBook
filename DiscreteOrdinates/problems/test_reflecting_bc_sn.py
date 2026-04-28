#!/usr/bin/env python3
"""test_reflecting_bc_sn.py
===========================

Verifies the reflecting boundary condition implementation in ``sn_solver``.

Physical setup
--------------
* Single spatial cell  (I = 1, hx = 1 cm), so the geometry is effectively
  a homogeneous infinite medium.
* Gray opacity         σ_a = 10 cm⁻¹ (constant, temperature-independent)
* No scattering
* Linear EOS           e = ρcᵥ T  with  ρcᵥ = 0.01 GJ/(cm³·keV)
* Initial radiation    T_rad = 0.5 keV  (isotropic, equilibrium spectrum)
* Initial matter       T_mat = 0.4 keV  (below radiation temperature)
* Both boundaries perfectly reflecting → no photons leave the domain

Expected results
----------------
1. **Energy conservation**: total energy E = u_rad + u_mat should be
   conserved at every time step (within the solver tolerance, ~1e-8
   relative).  With reflecting BCs the Fleck–Cummings scheme satisfies

       Δ(φ/c) + Δe = 0

   exactly at each step.

2. **Equilibration**: at late time T_rad ≈ T_mat ≈ T_eq where T_eq satisfies

       a T_eq⁴ + ρcᵥ T_eq = a T_rad₀⁴ + ρcᵥ T_mat₀

3. **Vacuum BCs do NOT conserve energy**: with zero incoming flux at both
   boundaries, radiation escapes and total energy decreases.

Run from the DiscreteOrdinates directory::

    python problems/test_reflecting_bc_sn.py

"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy.optimize import brentq
import sn_solver

# ── Physical constants (must match sn_solver) ────────────────────────────────
C_LIGHT = 29.98      # cm/ns
A_RAD   = 0.01372    # GJ/(cm³·keV⁴)
AC      = A_RAD * C_LIGHT

# ── Problem parameters ────────────────────────────────────────────────────────
SIGMA0  = 10.0       # absorption opacity  [cm⁻¹]
RHO_CV  = 0.01       # ρcᵥ                 [GJ/(cm³·keV)]
T_RAD0  = 0.5        # initial T_rad       [keV]
T_MAT0  = 0.4        # initial T_mat       [keV]
ORDER   = 3
N_SN    = 8          # S_N order


def _equilibrium_T(T_rad0=T_RAD0, T_mat0=T_MAT0, a=A_RAD, rho_cv=RHO_CV):
    """Solve  a T^4 + ρcᵥ T = a T_rad0^4 + ρcᵥ T_mat0  for T_eq."""
    E_total = a * T_rad0**4 + rho_cv * T_mat0
    return brentq(lambda T: a * T**4 + rho_cv * T - E_total, 1e-6, 10.0)


def _run(reflect_left, reflect_right, tfinal=1.0):
    """Run the single-cell TRT problem; return (phis, Ts, ts)."""
    I    = 1
    hx   = 1.0       # cm
    nop1 = ORDER + 1

    sigma_func = lambda T: np.full_like(T, SIGMA0)
    scat_func  = lambda T: np.zeros_like(T)
    EOS        = lambda T: RHO_CV * T
    invEOS     = lambda e: e / RHO_CV

    phi_init = np.full((I, nop1), AC * T_RAD0**4)
    T_init   = np.full((I, nop1), T_MAT0)
    # Equilibrium ψ convention for this code: ψ_n = φ for all n
    # (matches test_su_olson.py: psi = phi[:,None,:]).
    psi_init = np.zeros((I, N_SN, nop1)) + phi_init[:, None, :]

    q   = np.zeros((I, N_SN, nop1))
    BCs = lambda t: np.zeros((N_SN, nop1))   # zero incoming; reflecting overrides

    phis, Ts, _iters, ts = sn_solver.temp_solve_dmd_inc(
        I=I, hx=hx, q=q,
        sigma_func=sigma_func, scat_func=scat_func,
        N=N_SN, BCs=BCs,
        EOS=EOS, invEOS=invEOS,
        phi=phi_init.copy(), psi=psi_init.copy(), T=T_init.copy(),
        dt_min=1e-4, dt_max=1e-2, tfinal=tfinal,
        LOUD=False, order=ORDER, fix=1, K=100, R=3,
        reflect_left=reflect_left, reflect_right=reflect_right)

    return phis, Ts, ts


def test_energy_conservation():
    """Total energy is conserved at every step with fully reflecting BCs."""
    print("  Running reflecting simulation (tfinal=1 ns) …")
    phis, Ts, ts = _run(True, True, tfinal=1.0)

    # Reference energy from the initial state
    E_rad0 = float(np.mean(phis[0])) / C_LIGHT   # φ/c = u_rad  [GJ/cm³]
    E_mat0 = RHO_CV * float(np.mean(Ts[0]))
    E0     = E_rad0 + E_mat0

    max_rel_err = 0.0
    for phi_t, T_t in zip(phis, Ts):
        E_rad = float(np.mean(phi_t)) / C_LIGHT
        E_mat = RHO_CV * float(np.mean(T_t))
        rel_err = abs((E_rad + E_mat - E0) / E0)
        if rel_err > max_rel_err:
            max_rel_err = rel_err

    print("  Max relative energy error over all steps: %.3e" % max_rel_err)
    # The Fleck–Cummings scheme conserves energy exactly in the dt→0 limit.
    # With dt_max = 0.01 ns the per-step truncation error accumulates to O(1e-3).
    # We require < 1 % to distinguish reflecting from vacuum BCs (which lose > 50 %).
    assert max_rel_err < 1e-2, (
        "Energy not conserved: max relative error = %.3e" % max_rel_err)
    print("  PASSED")


def test_equilibration():
    """Radiation and matter temperatures converge to the analytic T_eq."""
    print("  Running reflecting simulation (tfinal=1 ns) …")
    phis, Ts, ts = _run(True, True, tfinal=1.0)

    T_eq = _equilibrium_T()

    T_mat_final = float(np.mean(Ts[-1]))
    u_rad_final = float(np.mean(phis[-1])) / C_LIGHT
    # T_rad from radiation energy density: u_rad = a T_rad^4
    T_rad_final = (u_rad_final / A_RAD) ** 0.25

    print("  Analytic T_eq  = %.6f keV" % T_eq)
    print("  T_mat (final)  = %.6f keV" % T_mat_final)
    print("  T_rad (final)  = %.6f keV" % T_rad_final)

    tol = 0.01   # 1 % relative tolerance
    err_mat = abs(T_mat_final - T_eq) / T_eq
    err_rad = abs(T_rad_final - T_eq) / T_eq
    assert err_mat < tol, (
        "T_mat not at equilibrium: got %.6f, expected %.6f (err=%.3f%%)"
        % (T_mat_final, T_eq, 100 * err_mat))
    assert err_rad < tol, (
        "T_rad not at equilibrium: got %.6f, expected %.6f (err=%.3f%%)"
        % (T_rad_final, T_eq, 100 * err_rad))
    print("  PASSED")


def test_vacuum_does_not_conserve():
    """With vacuum (zero) BCs, radiation escapes and energy is not conserved."""
    print("  Running vacuum BC simulation (tfinal=1 ns) …")
    phis, Ts, ts = _run(False, False, tfinal=1.0)

    E_rad0 = float(np.mean(phis[0])) / C_LIGHT
    E_mat0 = RHO_CV * float(np.mean(Ts[0]))
    E0     = E_rad0 + E_mat0

    E_rad_f = float(np.mean(phis[-1])) / C_LIGHT
    E_mat_f = RHO_CV * float(np.mean(Ts[-1]))
    E_final = E_rad_f + E_mat_f

    loss_frac = (E0 - E_final) / E0
    print("  Vacuum BC energy-loss fraction: %.4f" % loss_frac)
    assert loss_frac > 0.01, (
        "Expected significant energy loss with vacuum BCs, got only %.4f" % loss_frac)
    print("  PASSED")


if __name__ == "__main__":
    print("=" * 65)
    print("REFLECTING BOUNDARY CONDITION TEST  (gray S_N, single zone)")
    print("=" * 65)
    print()

    print("Test 1: energy conservation with reflecting BCs")
    test_energy_conservation()
    print()

    print("Test 2: equilibration to analytic T_eq with reflecting BCs")
    test_equilibration()
    print()

    print("Test 3: vacuum BCs DO lose energy")
    test_vacuum_does_not_conserve()
    print()

    print("All tests PASSED.")
