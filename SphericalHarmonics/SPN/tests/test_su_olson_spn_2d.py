#!/usr/bin/env python3
"""
Su-Olson benchmark: SP_N vs P_N equivalence in 1-D slab geometry.

Runs SP1, SP3, SP7 alongside P1, P3, P7 on the Su-Olson fixed-source problem
and verifies that the two methods produce identical results.  In 1-D slab
geometry the SP_N and P_N equations are mathematically equivalent; this test
quantifies the numerical difference.

Outputs
-------
  su_olson_spn_vs_pn.png  – comparison of 1-D radiation profiles
  su_olson_spn_vs_pn_error.png – pointwise relative error SPN vs PN

Run from the SphericalHarmonics/SPN/problems directory, or anywhere:
    python -m SphericalHarmonics.SPN.problems.test_su_olson_spn_2d
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ---- Reach solvers ----------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_SPN_DIR = os.path.dirname(_HERE)               # SPN/
_SH_DIR  = os.path.dirname(_SPN_DIR)            # SphericalHarmonics/
for _d in (_SPN_DIR, _SH_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from SphericalHarmonics.SPN.src.spn_solver_2d import temp_solve_spn_2d, ac
from SphericalHarmonics.src.pn_solver_2d import temp_solve_pn_2d

JACOBIAN_DIR = os.path.join(_SH_DIR, 'Jacobians')

# ---------------------------------------------------------------------------
# Su-Olson problem parameters
# ---------------------------------------------------------------------------
_sigma_a = 1.0    # cm⁻¹
_T_0     = 1.0    # keV
_tau_mft = 1.0 / (29.98 * _sigma_a)  # mean-free time (ns)


def _build_problem(Ix: int, Iy: int, Lx: float, Ly: float):
    dx_arr = np.full(Ix, Lx / Ix)
    dy_arr = np.full(Iy, Ly / Iy)
    x_faces  = np.linspace(0.0, Lx, Ix + 1)
    x_c      = 0.5 * (x_faces[:-1] + x_faces[1:])

    def EOS(T):     return _sigma_a * T ** 4 / (29.98 * _sigma_a)  # a T^4

    def invEOS(e):  return (np.maximum(e, 0.0) / (0.01372)) ** 0.25

    # Re-define with proper radiation constant
    from SphericalHarmonics.src.pn_solver_2d import a, c as clight
    def EOS(T):    return a * T ** 4
    def invEOS(e): return (np.maximum(e, 0.0) / a) ** 0.25

    def sigma_func(T): return np.full_like(T, _sigma_a)
    def scat_func(T):  return np.zeros_like(T)

    # Source Q = ac T₀⁴ in x ≤ 0.5 cm
    q_on = np.zeros((Ix, Iy, 4))
    from SphericalHarmonics.src.pn_solver_2d import ac
    for i in range(Ix):
        if x_c[i] <= 0.5:
            q_on[i, :, :] = ac * _T_0 ** 4

    T_floor = 1e-6
    T_init  = np.full((Ix, Iy, 4), T_floor)
    phi_init = ac * T_init ** 4

    return dx_arr, dy_arr, x_c, x_faces, sigma_func, scat_func, EOS, invEOS, q_on, T_init, phi_init


def run_one(kind: str, N: int, tau: float, Ix: int = 200, Iy: int = 3,
            Lx: float = 15.0, Ly: float = 0.5) -> np.ndarray:
    """Run SPN or PN on the Su-Olson problem and return 1-D normalised rad-energy profile."""
    dx_arr, dy_arr, x_c, x_faces, sigma_func, scat_func, EOS, invEOS, q_on, T_init, phi_init = \
        _build_problem(Ix, Iy, Lx, Ly)

    from SphericalHarmonics.src.pn_solver_2d import a as a_rad, ac
    t_ns = tau * _tau_mft
    t_src = 10.0 * _tau_mft   # source duration
    dt_min  = 0.01 * _tau_mft
    dt_max  = 0.10 * _tau_mft
    q_off   = np.zeros_like(q_on)
    T_floor = 1e-6

    phi, T, I_cur = phi_init.copy(), T_init.copy(), None

    def _run_seg(phi, T, I_cur, q, t_start, t_target):
        dur = t_target - t_start
        kw = dict(
            I_init=I_cur, dt_start=dt_min, t_end=dur,
            reflect_xlo=True, reflect_xhi=False,
            reflect_ylo=True, reflect_yhi=True,
            tolerance=1e-8, W=0, n_gs=1,
            dt_max=dt_max, T_floor=T_floor, print_stride=0)
        if kind == 'spn':
            return temp_solve_spn_2d(
                Ix, Iy, dx_arr, dy_arr, q, sigma_func, scat_func, N,
                EOS, invEOS, phi, T, **kw)
        else:
            return temp_solve_pn_2d(
                Ix, Iy, dx_arr, dy_arr, q, sigma_func, scat_func, N,
                JACOBIAN_DIR, EOS, invEOS, phi, T, **kw)

    t_done = 0.0
    segments = []
    if t_done < t_src <= t_ns:
        segments = [(t_src, q_on), (t_ns, q_off)]
    else:
        segments = [(t_ns, q_on if t_ns <= t_src else q_off)]

    for t_tgt, q_use in segments:
        if t_tgt <= t_done + 1e-15:
            continue
        phi_new, T_new, I_new, _, _ = _run_seg(phi, T, I_cur, q_use, t_done, t_tgt)
        phi, T, I_cur = phi_new, T_new, I_new
        t_done = t_tgt

    E0 = a_rad * _T_0 ** 4
    phi_1d = np.mean(phi[:, :, :], axis=(1, 2))
    return phi_1d / (29.98 * E0)   # = E_rad / E0


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

def compare(orders: list[int] = (1, 3, 7),
            tau: float = 1.0,
            Ix: int = 200,
            out_dir: str = '.') -> None:
    """Run SP_N and P_N for each order and report max relative errors."""
    print(f"\nSu-Olson SP_N vs P_N comparison  τ={tau}  Ix={Ix}\n{'─'*60}")
    os.makedirs(out_dir, exist_ok=True)

    colors = plt.cm.tab10(range(len(orders)))
    fig_comp, ax_comp = plt.subplots(figsize=(8, 5))
    fig_err,  ax_err  = plt.subplots(figsize=(8, 5))

    results: dict[str, dict[int, np.ndarray]] = {'spn': {}, 'pn': {}}
    x_c = None

    for i, N in enumerate(orders):
        for kind in ('pn', 'spn'):
            print(f"  {kind.upper()}{N}  …", end='', flush=True)
            E = run_one(kind, N, tau, Ix=Ix)
            if x_c is None:
                Lx = 15.0; dx = Lx / Ix
                x_c = np.arange(Ix) * dx + 0.5 * dx
            results[kind][N] = E
            print(' done')

        spn = results['spn'][N]
        pn  = results['pn'][N]
        denom = np.maximum(np.abs(pn), 1e-15)
        relerr = np.abs(spn - pn) / denom

        max_err = float(relerr.max())
        l2_err  = float(np.linalg.norm(spn - pn) / (np.linalg.norm(pn) + 1e-30))
        print(f"  N={N}  max-rel-err={max_err:.3e}  L2-rel-err={l2_err:.3e}")

        ax_comp.semilogy(x_c, np.maximum(pn, 1e-8), color=colors[i],
                          ls='-',  lw=2.0, label=f'P$_{N}$')
        ax_comp.semilogy(x_c, np.maximum(spn, 1e-8), color=colors[i],
                          ls='--', lw=1.5, label=f'SP$_{N}$')
        ax_err.semilogy(x_c, relerr + 1e-16, color=colors[i], lw=2,
                         label=f'N={N}')

    ax_comp.set_xlabel('x (cm)')
    ax_comp.set_ylabel(r'$E_r / (a T_0^4)$')
    ax_comp.set_title(rf'Su-Olson  $\tau={tau}$ — solid P$_N$, dashed SP$_N$')
    ax_comp.legend(ncol=2, fontsize=8)
    ax_comp.grid(True, which='both', alpha=0.3, ls='--')
    fig_comp.tight_layout()
    out_comp = os.path.join(out_dir, 'su_olson_spn_vs_pn.png')
    fig_comp.savefig(out_comp, dpi=150, bbox_inches='tight')
    print(f"\n  Saved {out_comp}")
    plt.close(fig_comp)

    ax_err.set_xlabel('x (cm)')
    ax_err.set_ylabel('Pointwise relative error |SP$_N$ − P$_N$| / |P$_N$|')
    ax_err.set_title(rf'SP$_N$ vs P$_N$ pointwise error  τ={tau}')
    ax_err.legend(fontsize=9)
    ax_err.grid(True, which='both', alpha=0.3, ls='--')
    ax_err.set_ylim(1e-16, 1e-5)
    fig_err.tight_layout()
    out_err = os.path.join(out_dir, 'su_olson_spn_vs_pn_error.png')
    fig_err.savefig(out_err, dpi=150, bbox_inches='tight')
    print(f"  Saved {out_err}")
    plt.close(fig_err)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SP_N vs P_N Su-Olson validation.')
    parser.add_argument('--orders', nargs='+', type=int, default=[1, 3, 7],
                        help='SP_N / P_N orders to compare.')
    parser.add_argument('--tau', type=float, default=1.0,
                        help='Output time in mean-free times.')
    parser.add_argument('--Ix', type=int, default=200,
                        help='Number of x cells.')
    parser.add_argument('--out-dir', default='.',
                        help='Output directory for figures.')
    args = parser.parse_args()
    compare(args.orders, args.tau, args.Ix, args.out_dir)
