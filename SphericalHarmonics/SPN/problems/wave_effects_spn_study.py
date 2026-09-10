#!/usr/bin/env python3
"""SP_N vs P_N equivalence study on the 2-D Gaussian hot-spot problem.

On a uniform medium the SP_N equations are mathematically equivalent to
the P_N equations of the same order (Gelbard, 1960).  This script verifies
that equivalence numerically by running SP3 and SP7 alongside P3 and P7 on
the same 4×4 cm hot-spot benchmark and comparing the scalar-flux profiles.

Usage
-----
    python wave_effects_spn_study.py              # run + plot
    python wave_effects_spn_study.py --load       # load cached SPN snapshots
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ---- locate study directory and project root --------------------------------
_HERE   = os.path.dirname(os.path.abspath(__file__))
_SPN    = os.path.dirname(_HERE)                         # SPN/
_SH     = os.path.dirname(_SPN)                          # SphericalHarmonics/
_ROOT   = os.path.dirname(_SH)                          # project root
_WAVE   = os.path.join(_SH, 'docs', 'wave_effects_hotspot')
for _d in (_SPN, _SH, os.path.join(_ROOT, 'utils')):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from SphericalHarmonics.SPN.src.spn_solver_2d import temp_solve_spn_2d, ac
# reuse load_npz / PN snapshot cache from the wave-effects study
sys.path.insert(0, _WAVE)
import generate_wave_effects_study as _study

JAC_DIR = os.path.join(_SH, 'Jacobians')

# ── Problem constants (must match generate_wave_effects_study.py) ─────────────
LX = 4.0; LY = 4.0
IX = 100; IY = 100
DT = 1.0e-3; T_END = 0.05
OUTPUT_TIMES = (0.01, 0.02, 0.03, 0.05)
COMPARE_TIME = T_END
SIGMA = 1.0; CV_VOL = 0.1
T_PEAK = 1.0; T_COLD = 0.01; GAUSS_SIGMA = 0.125
CX = LX / 2.0; CY = LY / 2.0


# ── SPN hot-spot runner ────────────────────────────────────────────────────────

def run_spn(order: int) -> '_study.MethodResult':
    """Run SP_N on the hot-spot problem; return a MethodResult."""
    dx_arr = np.full(IX, LX / IX)
    dy_arr = np.full(IY, LY / IY)
    x_faces = np.linspace(0.0, LX, IX + 1)
    y_faces = np.linspace(0.0, LY, IY + 1)
    x_c = 0.5 * (x_faces[:-1] + x_faces[1:])
    y_c = 0.5 * (y_faces[:-1] + y_faces[1:])

    # Corner initialisation (matches generate_wave_effects_study._build_corner_temperature)
    cx_off = np.array([+0.25, -0.25, -0.25, +0.25])
    cy_off = np.array([+0.25, +0.25, -0.25, -0.25])
    T_init = np.full((IX, IY, 4), T_COLD)
    for i in range(IX):
        for j in range(IY):
            for cc in range(4):
                xq = x_c[i] + cx_off[cc] * dx_arr[i]
                yq = y_c[j] + cy_off[cc] * dy_arr[j]
                r2 = (xq - CX) ** 2 + (yq - CY) ** 2
                T_init[i, j, cc] = (T_COLD +
                    T_PEAK * np.exp(-r2 / (2.0 * GAUSS_SIGMA ** 2)))
    phi_init = ac * T_init ** 4

    sigma_arr = np.full((IX, IY, 4), SIGMA)
    cv_arr    = np.full((IX, IY, 4), CV_VOL)

    def sigma_func(T): return sigma_arr
    def scat_func(T):  return np.zeros_like(T)
    def EOS(T):        return cv_arr * T
    def invEOS(e):     return e / cv_arr

    # Fiducial points (same as generate_wave_effects_study.py)
    fid_labels = _study.FID_LABELS
    fid_indices = _study._fid_indices(x_c, y_c)
    fid_times: list[float] = []
    fid_Trad: dict = {lab: [] for lab in fid_labels}
    fid_Tmat: dict = {lab: [] for lab in fid_labels}
    fid_Erad: dict = {lab: [] for lab in fid_labels}

    phi = phi_init.copy()
    T   = T_init.copy()
    I_cur = None

    for t_tgt in OUTPUT_TIMES:
        phi, T, I_cur, _, _ = temp_solve_spn_2d(
            IX, IY, dx_arr, dy_arr,
            np.zeros((IX, IY, 4)), sigma_func, scat_func, order,
            EOS, invEOS, phi, T,
            I_init=I_cur,
            dt_start=DT, t_end=t_tgt if I_cur is None else t_tgt - float(fid_times[-1]) if fid_times else t_tgt,
            reflect_xlo=False, reflect_xhi=False,
            reflect_ylo=False, reflect_yhi=False,
            tolerance=1e-6, W=0, n_gs=1,
            dt_max=DT, T_floor=1e-6, print_stride=0)
        fid_times.append(float(t_tgt))
        for lab, (i, j) in fid_indices.items():
            phi_ij = float(np.mean(phi[i, j, :]))
            T_ij   = float(np.mean(T[i, j, :]))
            fid_Trad[lab].append(float((max(phi_ij, 0.0) / ac) ** 0.25))
            fid_Tmat[lab].append(T_ij)
            fid_Erad[lab].append(phi_ij / 29.98)

    # Build MethodResult from the final snapshot
    phi_cell = np.mean(phi, axis=2)
    T_mat    = np.mean(T, axis=2)
    T_rad    = (np.maximum(phi_cell, 0.0) / ac) ** 0.25
    E_rad    = phi_cell / 29.98

    return _study._result_from_fields(
        f'SP{order}',
        T_rad, E_rad, T_mat,
        x_faces, y_faces,
        np.asarray(fid_times),
        fid_Trad, fid_Tmat, fid_Erad,
        metadata={'method': f'SP{order}', 't_end': T_END, 'dt': DT})


def _run_segment(order, dx_arr, dy_arr, sigma_func, scat_func, EOS, invEOS,
                 phi, T, I_cur, t_start, t_end):
    """Run one time segment."""
    return temp_solve_spn_2d(
        IX, IY, dx_arr, dy_arr,
        np.zeros((IX, IY, 4)), sigma_func, scat_func, order,
        EOS, invEOS, phi, T,
        I_init=I_cur,
        dt_start=DT, t_end=t_end - t_start,
        reflect_xlo=False, reflect_xhi=False,
        reflect_ylo=False, reflect_yhi=False,
        tolerance=1e-6, W=0, n_gs=1,
        dt_max=DT, T_floor=1e-6, print_stride=0)


def run_spn_hotspot(order: int) -> '_study.MethodResult':
    """Run SP_N on the hot-spot problem (correct segmented version)."""
    dx_arr = np.full(IX, LX / IX)
    dy_arr = np.full(IY, LY / IY)
    x_faces = np.linspace(0.0, LX, IX + 1)
    y_faces = np.linspace(0.0, LY, IY + 1)
    x_c = 0.5 * (x_faces[:-1] + x_faces[1:])
    y_c = 0.5 * (y_faces[:-1] + y_faces[1:])

    cx_off = np.array([+0.25, -0.25, -0.25, +0.25])
    cy_off = np.array([+0.25, +0.25, -0.25, -0.25])
    T_init = np.full((IX, IY, 4), T_COLD)
    for i in range(IX):
        for j in range(IY):
            for cc in range(4):
                xq = x_c[i] + cx_off[cc] * dx_arr[i]
                yq = y_c[j] + cy_off[cc] * dy_arr[j]
                r2 = (xq - CX) ** 2 + (yq - CY) ** 2
                T_init[i, j, cc] = (T_COLD +
                    T_PEAK * np.exp(-r2 / (2.0 * GAUSS_SIGMA ** 2)))
    phi_init = ac * T_init ** 4

    sigma_arr = np.full((IX, IY, 4), SIGMA)
    cv_arr    = np.full((IX, IY, 4), CV_VOL)

    def sigma_func(T): return sigma_arr
    def scat_func(T):  return np.zeros_like(T)
    def EOS(T):        return cv_arr * T
    def invEOS(e):     return e / cv_arr

    fid_indices = _study._fid_indices(x_c, y_c)
    fid_labels  = _study.FID_LABELS
    fid_times: list[float] = []
    fid_Trad = {lab: [] for lab in fid_labels}
    fid_Tmat = {lab: [] for lab in fid_labels}
    fid_Erad = {lab: [] for lab in fid_labels}

    phi = phi_init.copy()
    T   = T_init.copy()
    I_cur = None
    t_done = 0.0

    solutions = {}
    for t_tgt in OUTPUT_TIMES:
        phi, T, I_cur, _, _ = temp_solve_spn_2d(
            IX, IY, dx_arr, dy_arr,
            np.zeros((IX, IY, 4)), sigma_func, scat_func, order,
            EOS, invEOS, phi, T,
            I_init=I_cur,
            dt_start=DT, t_end=float(t_tgt) - t_done,
            reflect_xlo=False, reflect_xhi=False,
            reflect_ylo=False, reflect_yhi=False,
            tolerance=1e-6, W=0, n_gs=1,
            dt_max=DT, T_floor=1e-6, print_stride=50)
        t_done = float(t_tgt)
        fid_times.append(t_done)
        for lab, (i, j) in fid_indices.items():
            phi_ij = float(np.mean(phi[i, j, :]))
            T_ij   = float(np.mean(T[i, j, :]))
            fid_Trad[lab].append(float((max(phi_ij, 0.0) / ac) ** 0.25))
            fid_Tmat[lab].append(T_ij)
            fid_Erad[lab].append(phi_ij / 29.98)
        solutions[t_tgt] = {'phi': phi.copy(), 'T': T.copy(), 't_actual': t_done}

    phi_cell = np.mean(solutions[COMPARE_TIME]['phi'], axis=2)
    T_mat    = np.mean(solutions[COMPARE_TIME]['T'],   axis=2)
    T_rad    = (np.maximum(phi_cell, 0.0) / ac) ** 0.25
    E_rad    = phi_cell / 29.98

    return _study._result_from_fields(
        f'SP{order}',
        T_rad, E_rad, T_mat,
        x_faces, y_faces,
        np.asarray(fid_times),
        fid_Trad, fid_Tmat, fid_Erad,
        metadata={'method': f'SP{order}', 't_end': T_END, 'dt': DT,
                  'spn_order': order})


# ── Comparison and plotting ────────────────────────────────────────────────────

def _safe_rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def compare(orders: list[int], load: bool, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nSP_N vs P_N equivalence on the 2-D hot-spot benchmark")
    print(f"  Compare time:  t = {COMPARE_TIME} ns,  Ix = {IX},  Iy = {IY}")
    print(f"  Uniform medium: σ = {SIGMA} cm⁻¹,  cv = {CV_VOL} GJ/(cm³·keV)")
    print()

    fig_comp, axes_comp = plt.subplots(2, len(orders),
                                        figsize=(5 * len(orders), 9),
                                        gridspec_kw={'wspace': 0.06, 'hspace': 0.06})
    fig_err, ax_err = plt.subplots(figsize=(7, 4.5))

    pn_cache = {
        3:  os.path.join(_WAVE, 'pn3_snapshot.npz'),
        7:  os.path.join(_WAVE, 'pn7_snapshot.npz'),
        15: os.path.join(_WAVE, 'pn15_snapshot.npz'),
    }

    theta = np.linspace(0, 2 * np.pi, 256)
    colors = plt.cm.tab10(range(len(orders)))

    rows: list[dict] = []

    for col_idx, N in enumerate(orders):
        print(f"Order N={N}")

        # Load or run SPN
        spn_cache = os.path.join(out_dir, f'spn{N}_snapshot.npz')
        if load and os.path.exists(spn_cache):
            print(f"  Loading SPN cache: {spn_cache}")
            spn_res = _study.load_npz(spn_cache)
        else:
            print(f"  Running SP{N} …")
            spn_res = run_spn_hotspot(N)
            _study.save_npz(spn_res, spn_cache)

        # Load PN
        if N not in pn_cache or not os.path.exists(pn_cache[N]):
            print(f"  PN{N} cache not found, skipping.")
            continue
        print(f"  Loading PN cache: {pn_cache[N]}")
        pn_res = _study.load_npz(pn_cache[N])

        # Error metrics
        l2_Trad = _safe_rel_l2(spn_res.T_rad, pn_res.T_rad)
        l2_Erad = _safe_rel_l2(spn_res.E_rad, pn_res.E_rad)
        linf    = float(np.max(np.abs(spn_res.T_rad - pn_res.T_rad)))
        rows.append({'N': N, 'L2_Trad': l2_Trad, 'L2_Erad': l2_Erad, 'Linf': linf})
        print(f"  SP{N} vs P{N}: L2_Trad={l2_Trad:.3e}  L2_Erad={l2_Erad:.3e}"
              f"  Linf={linf:.3e}")

        # Centerline comparison
        ax_err.semilogy(pn_res.lineout_x,
                        np.abs(spn_res.centerline_Trad - pn_res.centerline_Trad)
                        / (np.abs(pn_res.centerline_Trad) + 1e-8),
                        color=colors[col_idx], lw=2,
                        label=rf'N={N}')

        # 2-panel heatmaps
        vmax = float(max(pn_res.T_rad.max(), spn_res.T_rad.max()))
        for row_idx, (data, label) in enumerate([
                (pn_res.T_rad, f'P$_{N}$'), (spn_res.T_rad, f'SP$_{N}$')]):
            ax = axes_comp[row_idx, col_idx]
            im = ax.pcolormesh(pn_res.x_faces, pn_res.y_faces, data.T,
                               shading='flat', cmap='plasma', vmin=0.0, vmax=vmax)
            ax.set_aspect('equal')
            ax.plot(CX + GAUSS_SIGMA * np.cos(theta),
                    CY + GAUSS_SIGMA * np.sin(theta), 'w--', lw=0.8, alpha=0.8)
            ax.text(0.03, 0.97, label, transform=ax.transAxes, fontsize=10,
                    color='white', va='top',
                    bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.4, lw=0))
            if row_idx < 1:
                ax.set_xticklabels([])
            if col_idx > 0:
                ax.set_yticklabels([])
        axes_comp[-1, col_idx].set_xlabel('x (cm)')
    axes_comp[0, 0].set_ylabel(r'P$_N$  y (cm)')
    axes_comp[1, 0].set_ylabel(r'SP$_N$  y (cm)')

    cax = fig_comp.add_axes([0.93, 0.1, 0.015, 0.8])
    fig_comp.colorbar(im, cax=cax, label=r'$T_r$ (keV)')
    out_comp = os.path.join(out_dir, 'wave_effects_spn_pn_comparison.png')
    fig_comp.savefig(out_comp, dpi=300, bbox_inches='tight')
    print(f"\nSaved {out_comp}")
    plt.close(fig_comp)

    ax_err.set_xlabel('x along y = 2 cm (cm)')
    ax_err.set_ylabel(r'Pointwise $|SP_N - P_N|/|P_N|$  for $T_r$')
    ax_err.set_title(rf'SP$_N$ vs P$_N$ equivalence  t = {COMPARE_TIME} ns')
    ax_err.legend(fontsize=10)
    ax_err.grid(True, which='both', alpha=0.3, ls='--')
    out_err = os.path.join(out_dir, 'wave_effects_spn_pn_error.png')
    fig_err.tight_layout()
    fig_err.savefig(out_err, dpi=300, bbox_inches='tight')
    print(f"Saved {out_err}")
    plt.close(fig_err)

    # Summary table
    print(f"\n{'N':>4}  {'L2_Trad':>12}  {'L2_Erad':>12}  {'Linf_Trad':>12}")
    for r in rows:
        print(f"  {r['N']:>2}  {r['L2_Trad']:12.3e}  {r['L2_Erad']:12.3e}"
              f"  {r['Linf']:12.3e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SP_N vs P_N wave-effects equivalence study.')
    parser.add_argument('--orders', nargs='+', type=int, default=[3, 7],
                        help='SPN/PN orders to compare.')
    parser.add_argument('--load', action='store_true',
                        help='Load cached SPN snapshots.')
    parser.add_argument('--out-dir', default=_WAVE,
                        help='Output directory (default: wave_effects_hotspot/).')
    args = parser.parse_args()
    compare(args.orders, args.load, args.out_dir)
