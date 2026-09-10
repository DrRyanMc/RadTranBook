#!/usr/bin/env python3
import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

r"""
Method of Manufactured Solutions (MMS) for the multigroup LD-S_N solver
``mg_sn_solver_ld.mg_temp_solve_ld``.

Goal
----
Verify the *full* multigroup, time-dependent, Fleck-coupled transport solver
end-to-end: per-group transport, within-group scattering, the inter-group
emission redistribution (the ``alpha_g Sigma_{g'} sigma_{a,g'} phi_{g'}``
coupling), the material energy update, and the boundary conditions.

Manufactured solution
----------------------
We manufacture a *steady* per-group angular flux that is an isotropic Planck
part plus a linearly-anisotropic current:

    psi_{g,n}(x) = B_g(T(x))  +  mu_n * J_g(x)

With the solver's quadrature (Sum_n W_n = 1, Sum_n W_n mu_n = 0) the scalar
flux is exactly the Planck part,

    phi_g(x) = Sum_n W_n psi_{g,n}(x) = B_g(T(x)),

so the manufactured field is in *radiative equilibrium per group*,
phi_g = B_g(T).  That makes it a genuine steady fixed point of the discrete
time-stepping (the material energy deposition Sum_g sigma_{a,g}(phi_g - B_g)
vanishes), and it causes the Fleck/inter-group coupling terms to cancel
*in the steady residual* so the manufactured source has a clean closed form:

    B_g(T)      = p_g * (a c) * T^4            (group fractions p_g, Sum p_g = 1)
    dB_g/dx     = p_g * (a c) * dTheta/dx,      Theta(x) := T(x)^4
    sigma_t,g   = sigma_a,g + sigma_s,g

    Q_{g,n}(x)  = mu_n ( dB_g/dx + sigma_t,g J_g(x) )  +  mu_n^2 dJ_g/dx

Although the coupling cancels *at the solution*, the solver still assembles and
iterates the full scattering + alpha_g + Fleck operator on the way there, so a
bug in any of that machinery shows up as a failure to reproduce psi_{g,n}.

Derivation (steady per-ordinate transport equation the solver discretises):

    mu dpsi/dx + sigma_t,g psi
        = sigma_s,g phi_g + sigma_a,g B_g
          + alpha_g [ Sum_{g'} sigma_{a,g'}(phi_{g'} - B_{g'}) ] + Q_{g,n}

At equilibrium phi_g = B_g the bracket is zero and the right side is
sigma_t,g B_g.  Substituting psi = B_g + mu J_g and solving for Q gives the
formula above.

Two test modes
--------------
* ``--mode linear``  : Theta(x)=T^4 and J_g(x) are linear in x, so psi is
  linear in x and the LD scheme is *exact*.  The solver must reproduce the
  manufactured solution to machine precision -- a hard correctness check on the
  multigroup assembly.  (Run with the positivity fix-up OFF.)

* ``--mode smooth``  : Theta(x) and J_g(x) are smooth and non-linear, so the
  only error is the O(h^2) LD spatial discretisation.  A mesh-refinement study
  recovers the formal second-order rate.

Run
---
    cd DiscreteOrdinates
    python problems/mms_multigroup_sn.py --mode linear  --G 3 --N 8 --I 64
    python problems/mms_multigroup_sn.py --mode smooth --G 3 --N 8 \
        --I 16 32 64 128 --plot
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_this_dir = os.path.dirname(os.path.abspath(__file__))
_sn_dir = os.path.dirname(_this_dir)        # DiscreteOrdinates/
sys.path.insert(0, _sn_dir)

from DiscreteOrdinates.src.sn_solver import _get_quadrature
from DiscreteOrdinates.src.sn_solver_ld import c, a, ac
from DiscreteOrdinates.src.mg_sn_solver_ld import mg_temp_solve_ld


# ---------------------------------------------------------------------------
# Problem constants (chosen to exercise distinct per-group physics)
# ---------------------------------------------------------------------------
L = 1.0          # slab width (cm)
CV = 0.3         # constant heat capacity  ->  e = CV T,  T = e / CV


def group_fractions(G):
    """Distinct positive emission fractions p_g with Sum_g p_g = 1."""
    raw = 1.0 + np.arange(G, dtype=np.float64)      # 1, 2, 3, ...
    return raw / raw.sum()


def sigma_a_list(G):
    """Distinct per-group absorption cross-sections (1/cm)."""
    return [3.0 / (1.0 + g) for g in range(G)]       # 3.0, 1.5, 1.0, ...


def sigma_s_list(G):
    """Distinct per-group scattering cross-sections (1/cm)."""
    return [0.5 * (1.0 + g) for g in range(G)]       # 0.5, 1.0, 1.5, ...


# ---------------------------------------------------------------------------
# Manufactured profiles
# ---------------------------------------------------------------------------
class Manufactured:
    r"""Holds the manufactured profiles Theta(x)=T^4, T(x), J_g(x) and their
    x-derivatives, plus the cross sections and group fractions."""

    def __init__(self, G, mode):
        self.G = G
        self.mode = mode
        self.p = group_fractions(G)
        self.sa = sigma_a_list(G)
        self.ss = sigma_s_list(G)
        self.st = [self.sa[g] + self.ss[g] for g in range(G)]
        # current amplitudes (kept small so |mu J_g| < B_g -> psi > 0)
        self.Jamp = [0.02 / (1.0 + g) for g in range(G)]

    # -- Theta(x) = T(x)^4 and dTheta/dx -----------------------------------
    def Theta(self, x):
        x = np.asarray(x, dtype=np.float64)
        if self.mode == "linear":
            return 1.0 + 2.0 * (x / L)                       # 1 -> 3
        return 1.0 + 0.4 * np.sin(np.pi * x / L)             # smooth, > 0

    def dTheta(self, x):
        x = np.asarray(x, dtype=np.float64)
        if self.mode == "linear":
            return np.full_like(x, 2.0 / L)
        return 0.4 * (np.pi / L) * np.cos(np.pi * x / L)

    def T(self, x):
        return self.Theta(x) ** 0.25

    # -- per-group current J_g(x) and dJ_g/dx ------------------------------
    def J(self, g, x):
        x = np.asarray(x, dtype=np.float64)
        if self.mode == "linear":
            return self.Jamp[g] * (0.5 + 0.5 * x / L)
        return self.Jamp[g] * (1.0 + 0.3 * np.cos(np.pi * x / L))

    def dJ(self, g, x):
        x = np.asarray(x, dtype=np.float64)
        if self.mode == "linear":
            return np.full_like(x, self.Jamp[g] * 0.5 / L)
        return -self.Jamp[g] * 0.3 * (np.pi / L) * np.sin(np.pi * x / L)

    # -- manufactured Planck emission per group B_g(x) = phi_g(x) ----------
    def B(self, g, x):
        return self.p[g] * ac * self.Theta(x)

    def dB(self, g, x):
        return self.p[g] * ac * self.dTheta(x)


# ---------------------------------------------------------------------------
# Single-mesh solve
# ---------------------------------------------------------------------------
def run_mesh(mms, I, N, tfinal, dt_min, dt_max, K, maxits, fix, LOUD=False):
    """Run the multigroup solver to steady state on an I-cell mesh.

    Returns
    -------
    phi_num : list of G  (I, 2)   final scalar fluxes
    T_num   : (I, 2)              final temperature
    phi_man : list of G  (I, 2)   manufactured B_g at the same edges
    T_man   : (I, 2)              manufactured temperature
    steady_change : float         max relative change over the last step
    """
    G = mms.G
    hx = L / I
    MU, W = _get_quadrature(N)

    # cell-edge coordinates: column 0 = left edge, column 1 = right edge
    xL = np.arange(I, dtype=np.float64) * hx
    xR = xL + hx
    x_edges = np.stack([xL, xR], axis=1)            # (I, 2)

    # -- manufactured external source Q_{g,n}(x) -------------------------------
    q_ext = []
    for g in range(G):
        qe = np.zeros((I, N, 2))
        dBg = mms.dB(g, x_edges)                    # (I, 2)
        Jg = np.stack([mms.J(g, xL), mms.J(g, xR)], axis=1)
        dJg = np.stack([mms.dJ(g, xL), mms.dJ(g, xR)], axis=1)
        st = mms.st[g]
        for n in range(N):
            mu = MU[n]
            qe[:, n, :] = mu * (dBg + st * Jg) + (mu * mu) * dJg
        q_ext.append(qe)

    # -- per-group boundary conditions (steady -> time-independent) ------------
    def make_bc(g):
        BL = mms.B(g, 0.0)
        BR = mms.B(g, L)
        JL = mms.J(g, 0.0)
        JR = mms.J(g, L)
        arr = np.zeros((N, 2))
        for n in range(N):
            mu = MU[n]
            arr[n, 1] = BL + mu * JL          # left wall inflow  (mu > 0)
            arr[n, 0] = BR + mu * JR          # right wall inflow (mu < 0)
        return lambda t, arr=arr: arr

    BCs = [make_bc(g) for g in range(G)]

    # -- material callables ----------------------------------------------------
    sigma_a_funcs = [lambda T, g=g: np.full_like(T, mms.sa[g]) for g in range(G)]
    scat_funcs    = [lambda T, g=g: np.full_like(T, mms.ss[g]) for g in range(G)]
    Bg_funcs      = [lambda T, g=g: mms.p[g] * ac * T ** 4 for g in range(G)]
    dBdT_funcs    = [lambda T, g=g: 4.0 * mms.p[g] * ac * T ** 3 for g in range(G)]
    Cv_func       = lambda T: np.full_like(T, CV)
    EOS           = lambda T: CV * T
    invEOS        = lambda e: e / CV

    # -- initial state: uniform, in equilibrium at T0 --------------------------
    T0 = 1.0
    T_init = np.full((I, 2), T0)
    phi_g = [Bg_funcs[g](T_init) for g in range(G)]
    psi_g = [np.broadcast_to(phi_g[g][:, None, :], (I, N, 2)).copy()
             for g in range(G)]

    phi_hist, T_hist, iters, ts = mg_temp_solve_ld(
        I, hx, G,
        sigma_a_funcs, scat_funcs, Bg_funcs, dBdT_funcs, q_ext,
        N, BCs, EOS, invEOS, Cv_func,
        phi_g, psi_g, T_init,
        dt_min=dt_min, dt_max=dt_max, tfinal=tfinal,
        Linf_tol=1e-12, tolerance=1e-13, maxits=maxits,
        LOUD=LOUD, fix=fix, K=K,
    )

    phi_num = phi_hist[-1]
    T_num = T_hist[-1]

    # steadiness diagnostic: relative change of T over the final step
    T_prev = T_hist[-2]
    denom = np.max(np.abs(T_num)) + 1e-300
    steady_change = np.max(np.abs(T_num - T_prev)) / denom

    # manufactured reference at the same edges
    phi_man = [mms.B(g, x_edges) for g in range(G)]
    T_man = mms.T(x_edges)

    return phi_num, T_num, phi_man, T_man, steady_change


# ---------------------------------------------------------------------------
# Error norms
# ---------------------------------------------------------------------------
def error_norms(phi_num, T_num, phi_man, T_man, hx):
    """Relative L2 and L-infinity errors over the radiation field and T."""
    # radiation field: stack all groups
    num = np.concatenate([p.ravel() for p in phi_num])
    man = np.concatenate([p.ravel() for p in phi_man])
    err = num - man
    # LD cell weight: each of the two edge values carries hx/2
    w = 0.5 * hx
    l2_rad = np.sqrt(np.sum(err ** 2) * w) / (np.sqrt(np.sum(man ** 2) * w) + 1e-300)
    linf_rad = np.max(np.abs(err)) / (np.max(np.abs(man)) + 1e-300)

    Terr = (T_num - T_man).ravel()
    l2_T = np.sqrt(np.sum(Terr ** 2) * w) / (np.sqrt(np.sum(T_man.ravel() ** 2) * w) + 1e-300)
    linf_T = np.max(np.abs(Terr)) / (np.max(np.abs(T_man)) + 1e-300)

    return dict(l2_rad=l2_rad, linf_rad=linf_rad, l2_T=l2_T, linf_T=linf_T)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["linear", "smooth"], default="smooth")
    parser.add_argument("--G", type=int, default=3, help="number of groups")
    parser.add_argument("--N", type=int, default=8, help="number of ordinates")
    parser.add_argument("--I", type=int, nargs="+", default=None,
                        help="cells per mesh (one or more for a refinement study)")
    parser.add_argument("--tfinal", type=float, default=20.0)
    parser.add_argument("--dt-min", type=float, default=1e-3)
    parser.add_argument("--dt-max", type=float, default=1.0)
    parser.add_argument("--K", type=int, default=50)
    parser.add_argument("--maxits", type=int, default=200)
    parser.add_argument("--plot", action="store_true",
                        help="save a log-log convergence figure (smooth mode)")
    parser.add_argument("--loud", action="store_true")
    args = parser.parse_args(argv)

    if args.I is None:
        args.I = [64] if args.mode == "linear" else [16, 32, 64, 128]

    # Both checks run with the positivity fix-up OFF: the manufactured
    # solution is strictly positive, and the limiter would otherwise pollute
    # the linear-exactness and O(h^2) measurements.
    fix = 0

    mms = Manufactured(args.G, args.mode)

    print(f"\nMultigroup MMS  (mode={args.mode}, G={args.G}, N={args.N})")
    print(f"  p_g      = {np.array2string(mms.p, precision=4)}")
    print(f"  sigma_a  = {mms.sa}")
    print(f"  sigma_s  = {mms.ss}")
    print(f"{'I':>6} {'hx':>12} {'L2(rad)':>13} {'Linf(rad)':>13} "
          f"{'L2(T)':>13} {'Linf(T)':>13} {'dT_last':>11}")

    hxs, l2s, linfs = [], [], []
    for I in args.I:
        phi_num, T_num, phi_man, T_man, dT = run_mesh(
            mms, I, args.N, args.tfinal, args.dt_min, args.dt_max,
            args.K, args.maxits, fix, LOUD=args.loud)
        hx = L / I
        e = error_norms(phi_num, T_num, phi_man, T_man, hx)
        hxs.append(hx)
        l2s.append(e["l2_rad"])
        linfs.append(e["linf_rad"])
        print(f"{I:6d} {hx:12.5e} {e['l2_rad']:13.4e} {e['linf_rad']:13.4e} "
              f"{e['l2_T']:13.4e} {e['linf_T']:13.4e} {dT:11.2e}")

    hxs = np.array(hxs)
    l2s = np.array(l2s)
    linfs = np.array(linfs)

    if args.mode == "linear":
        worst = max(l2s.max(), linfs.max())
        print(f"\nLinear-exactness check: worst relative error = {worst:.3e}")
        if worst < 1e-8:
            print("PASS  -- LD reproduces the linear manufactured solution "
                  "to (near) machine precision.")
        else:
            print("WARNING -- error above 1e-8; either not yet steady "
                  "(increase --tfinal) or a solver bug.")
    elif len(hxs) >= 2:
        # fit observed order from successive mesh halvings
        rate_l2 = np.log(l2s[:-1] / l2s[1:]) / np.log(hxs[:-1] / hxs[1:])
        rate_linf = np.log(linfs[:-1] / linfs[1:]) / np.log(hxs[:-1] / hxs[1:])
        print("\nObserved convergence order (radiation field):")
        for k in range(len(rate_l2)):
            print(f"  {args.I[k]:4d} -> {args.I[k+1]:4d}: "
                  f"p_L2 = {rate_l2[k]:.3f},  p_Linf = {rate_linf[k]:.3f}")
        print(f"  mean p_L2 = {rate_l2.mean():.3f}  (expected ~2 for LD)")

        if args.plot:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.loglog(hxs, l2s, "o-", label="L2 (radiation)")
            ax.loglog(hxs, linfs, "s-", label="Linf (radiation)")
            ref = l2s[0] * (hxs / hxs[0]) ** 2
            ax.loglog(hxs, ref, "k--", label=r"$O(h^2)$ reference")
            ax.set_xlabel("cell width  h")
            ax.set_ylabel("relative error")
            ax.set_title(f"Multigroup LD-S$_N$ MMS (G={args.G}, N={args.N})")
            ax.legend()
            ax.grid(True, which="both", ls=":")
            out = os.path.join(_this_dir, "mms_multigroup_sn_convergence.png")
            fig.tight_layout()
            fig.savefig(out, dpi=130)
            print(f"\nSaved convergence figure to {out}")


if __name__ == "__main__":
    main()
