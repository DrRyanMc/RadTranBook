#!/usr/bin/env python3
import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

"""
Generate hot-spot ray-effect comparison figures for the LaTeX report.

Six methods at t = T_COMPARE ns:
  1. Equilibrium diffusion
  2. Flux-limited nonequilibrium diffusion
  3. Gray IMC
  4. S_4  equal-weight quadrature
  5. S_4  level-symmetric quadrature
  6. S_4  product-square quadrature

Produces in RESULTS_DIR:
  ray_effects_comparison.pdf / .png  — 3×2 T_rad colour-map panel
  ray_effects_fiducials.pdf  / .png  — fiducial T_rad histories
  <method>_snapshot.npz              — raw snapshot data for each method
"""

import sys, os, importlib.util
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── paths ──────────────────────────────────────────────────────────────────
_here      = os.path.dirname(os.path.abspath(__file__))  # .../problems/
_d2d       = os.path.dirname(_here)                      # .../DiscreteOrdinates2D/
RESULTS_DIR = os.path.join(_d2d, 'results', 'ray_effects_hotspot')
os.makedirs(RESULTS_DIR, exist_ok=True)

# -- shared plot style (Gill Sans / hide_spines) ----------------------------
try:
    sys.path.insert(0, os.path.join(_d2d, '..'))
    from utils.plotfuncs import show as _show, hide_spines as _hide_spines, font as _font
    HAS_PLOTFUNCS = True
except ImportError:
    HAS_PLOTFUNCS = False

# Add solver directories so imports work
sys.path.insert(0, _d2d)
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(_d2d, '..', 'EqDiffusion', 'utils'))
sys.path.insert(0, os.path.join(_d2d, '..', 'IMC'))
sys.path.insert(0, os.path.join(_d2d, '..', 'nonEquilibriumDiffusion'))

# ── shared problem parameters ──────────────────────────────────────────────
T_COMPARE = 5.0          # ns — comparison snapshot
OUTPUT_TIMES = (1.0, 5.0, 10.0)
LX = LY = 5.0
T_PEAK  = 1.0;  T_COLD = 0.01;  GAUSS_SIGMA = 0.125
CX = CY = LX / 2.0
SIGMA = 1.0;  CV_VOL = 0.1

FID_LABELS = ['center', 'x+1 cm', 'diagonal +1 cm', 'y+1 cm']
FID_COORDS = [(CX, CY), (CX+1.0, CY), (CX+0.707, CY+0.707), (CX, CY+1.0)]


# ══════════════════════════════════════════════════════════════════════════
# Helper: standardise result dict
# ══════════════════════════════════════════════════════════════════════════

def make_result(T_rad_2d, T_mat_2d, x_faces, y_faces,
                fid_times, fid_Trad, fid_Tmat, method_label):
    """Wrap a snapshot into the standard comparison dict."""
    return {
        'T_rad': T_rad_2d,    # (Nx, Ny)
        'T_mat': T_mat_2d,
        'x_faces': x_faces,   # (Nx+1,)
        'y_faces': y_faces,
        'fid_times': np.asarray(fid_times),
        'fid_Trad': {l: np.asarray(v) for l, v in fid_Trad.items()},
        'fid_Tmat': {l: np.asarray(v) for l, v in fid_Tmat.items()},
        'label': method_label,
    }


def _fid_indices(x_centers, y_centers):
    idxs = {}
    for label, (xv, yv) in zip(FID_LABELS, FID_COORDS):
        idxs[label] = (int(np.argmin(np.abs(x_centers - xv))),
                       int(np.argmin(np.abs(y_centers - yv))))
    return idxs


# ══════════════════════════════════════════════════════════════════════════
# Method 1 – Equilibrium Diffusion
# ══════════════════════════════════════════════════════════════════════════

def run_eq_diffusion(Nx=100, Ny=100):
    print("\n── Equilibrium Diffusion ──")
    from hot_spot_cooling_eq_diff import setup_and_run
    res = setup_and_run(Nx=Nx, Ny=Ny, dt=0.01, output_times=OUTPUT_TIMES)
    sol  = res['solutions']
    t_k  = min(sol, key=lambda t: abs(t - T_COMPARE))
    T    = sol[t_k]['T']
    xf   = res['x_faces'];  yf = res['y_faces']
    xc   = res['x_centers']; yc = res['y_centers']
    fid  = _fid_indices(xc, yc)
    ts_all = res['ts'];  fd = res['fid_data']
    return make_result(
        T, T, xf, yf,
        ts_all,
        {l: fd[l]['T_rad'] for l in FID_LABELS if l in fd},
        {l: fd[l]['T_mat'] for l in FID_LABELS if l in fd},
        r'Eq. Diffusion')


# ══════════════════════════════════════════════════════════════════════════
# Method 2 – Flux-limited nonequilibrium Diffusion
# ══════════════════════════════════════════════════════════════════════════

def run_fld(Nx=100, Ny=100):
    print("\n── Flux-Limited Diffusion ──")
    # The eq-diffusion run loads EqDiffusion/utils/twoDFV and caches it in
    # sys.modules.  Evict that entry and ensure the nonEquilibriumDiffusion
    # path is first so hot_spot_cooling_noneq_diff gets the right module.
    import sys as _sys
    _neq = os.path.join(_d2d, '..', 'nonEquilibriumDiffusion')
    if _neq not in _sys.path:
        _sys.path.insert(0, _neq)
    for _key in list(_sys.modules.keys()):
        if _key == 'twoDFV' or _key.endswith('.twoDFV'):
            del _sys.modules[_key]
    from hot_spot_cooling_noneq_diff import setup_and_run
    res = setup_and_run(Nx=Nx, Ny=Ny, dt=0.01, output_times=OUTPUT_TIMES)
    sol  = res['solutions']
    t_k  = min(sol, key=lambda t: abs(t - T_COMPARE))
    Tr   = sol[t_k]['Tr'];  T = sol[t_k]['T']
    xf   = res['x_faces'];  yf = res['y_faces']
    xc   = res['x_centers']; yc = res['y_centers']
    ts_all = res['ts'];  fd = res['fid_data']
    return make_result(
        Tr, T, xf, yf,
        ts_all,
        {l: fd[l]['T_rad'] for l in FID_LABELS if l in fd},
        {l: fd[l]['T_mat'] for l in FID_LABELS if l in fd},
        r'FL Diffusion')


# ══════════════════════════════════════════════════════════════════════════
# Method 3 – Gray IMC
# ══════════════════════════════════════════════════════════════════════════

def run_imc(Nx=100, Ny=100, Ntarget=200_000, dt=0.01):
    print("\n── Gray IMC ──")
    from hot_spot_cooling_imc import setup_and_run
    Nmax = 10*Ntarget
    res  = setup_and_run(Nx=Nx, Ny=Ny, Ntarget=Ntarget, Nmax=Nmax,
                         dt=dt, output_times=OUTPUT_TIMES)
    sol  = res['solutions']
    t_k  = min(sol, key=lambda t: abs(t - T_COMPARE))
    Tr   = sol[t_k]['Tr'];  T = sol[t_k]['T']
    xf   = res['x_faces'];  yf = res['y_faces']
    xc   = res['x_centers']; yc = res['y_centers']
    ts_all = res['ts'];  fd = res['fid_data']
    return make_result(
        Tr, T, xf, yf,
        ts_all,
        {l: fd[l]['T_rad'] for l in FID_LABELS if l in fd},
        {l: fd[l]['T_mat'] for l in FID_LABELS if l in fd},
        'IMC')


# ══════════════════════════════════════════════════════════════════════════
# Method 4–6 – S_N with different quadratures
# ══════════════════════════════════════════════════════════════════════════

def run_sn(quad_type='level_symmetric', N_quad=4, Ix=80, Iy=80,
           dt_min=1e-3, dt_max=0.3, label=None):
    print(f"\n── S_N  N={N_quad}  quad={quad_type} ──")
    from hot_spot_cooling import setup_and_run
    res = setup_and_run(
        Ix=Ix, Iy=Iy, N_quad=N_quad, quad_type=quad_type,
        output_times=OUTPUT_TIMES, LOUD=False, use_dmd=False,
        dt_min=dt_min, dt_max=dt_max)

    from sn_solver_2d import ac
    sol  = res['solutions']
    t_k  = min(sol, key=lambda t: abs(t - T_COMPARE))
    phi  = np.mean(sol[t_k]['phi'], axis=2)
    T    = np.mean(sol[t_k]['T'], axis=2)
    Tr   = (np.maximum(phi, 0.0) / ac)**0.25
    xf   = res['x_faces'];  yf = res['y_faces']
    xc   = res['x_centers']; yc = res['y_centers']
    ts_all = res['ts'];  fd = res['fid_data']
    lab  = label or f'$S_{N_quad}$ {quad_type}'
    return make_result(
        Tr, T, xf, yf,
        ts_all,
        {l: fd[l]['T_rad'] for l in FID_LABELS if l in fd},
        {l: fd[l]['T_mat'] for l in FID_LABELS if l in fd},
        lab)


# ══════════════════════════════════════════════════════════════════════════
# Save to NPZ
# ══════════════════════════════════════════════════════════════════════════

def save_npz(result, filename):
    # Only include fiducial labels that were actually recorded.
    present = [l for l in FID_LABELS if l in result['fid_Trad']]
    flat_Trad = np.column_stack(
        [result['fid_Trad'][l] for l in present])
    np.savez_compressed(
        filename,
        T_rad      = result['T_rad'],
        T_mat      = result['T_mat'],
        x_faces    = result['x_faces'],
        y_faces    = result['y_faces'],
        fid_times  = result['fid_times'],
        fid_Trad   = flat_Trad,
        fid_labels = np.array(present),     # matches the columns saved
        method     = np.array(result['label']))
    print(f'  saved {filename}')


def save_fiducials_npz(results_list, filename='ray_effects_fiducials.npz'):
    """Save all fiducial-point T_rad/T_mat histories for all methods.

    Layout
    ------
    method_labels : (n_methods,) object array of method name strings.
    fid_labels    : (n_fids,) object array of fiducial-point name strings.
    fid_times_{i} : time array for method i.
    fid_Trad_{i}_{j} : T_rad vs time for method i, fiducial j.
    fid_Tmat_{i}_{j} : T_mat vs time for method i, fiducial j.

    Loading example
    ---------------
    d = np.load('ray_effects_fiducials.npz', allow_pickle=True)
    methods = list(d['method_labels'])
    fids    = list(d['fid_labels'])
    for i, method in enumerate(methods):
        times = d[f'fid_times_{i}']
        for j, fid in enumerate(fids):
            Trad = d.get(f'fid_Trad_{i}_{j}')
    """
    save_dict = {
        'method_labels': np.array([str(r['label']) for r in results_list],
                                   dtype=object),
        'fid_labels':    np.array(FID_LABELS, dtype=object),
    }
    for i, res in enumerate(results_list):
        save_dict[f'fid_times_{i}'] = np.asarray(res['fid_times'])
        for j, lbl in enumerate(FID_LABELS):
            if lbl in res['fid_Trad']:
                save_dict[f'fid_Trad_{i}_{j}'] = np.asarray(res['fid_Trad'][lbl])
            if lbl in res['fid_Tmat']:
                save_dict[f'fid_Tmat_{i}_{j}'] = np.asarray(res['fid_Tmat'][lbl])
    np.savez_compressed(filename, **save_dict)
    print(f'  saved {filename}')


# ══════════════════════════════════════════════════════════════════════════
# 3×2 comparison figure
# ══════════════════════════════════════════════════════════════════════════

def plot_comparison(results_list, t_ns=T_COMPARE,
                    savefile='ray_effects_comparison'):
    """3×2 panel of T_rad colour maps with a single shared colorbar."""
    assert len(results_list) == 6, "Need exactly 6 results"

    # Shared colour scale from the reference (IMC, index 2)
    ref = results_list[2]
    vmax = float(ref['T_rad'].max())
    vmin = 0.0

    fig, axes = plt.subplots(3, 2, figsize=(7.5, 12),
                             gridspec_kw={'wspace': 0.05, 'hspace': 0.08})

    ims = []
    for ax, res in zip(axes.flat, results_list):
        xf = res['x_faces'];  yf = res['y_faces']
        Tr = res['T_rad']
        im = ax.pcolormesh(xf, yf, Tr.T, shading='flat',
                           cmap='plasma', vmin=vmin, vmax=vmax)
        ims.append(im)
        ax.set_aspect('equal')
        ax.set_xlim(0, LX);  ax.set_ylim(0, LY)

        # Gaussian 1-σ circle for reference
        theta = np.linspace(0, 2*np.pi, 200)
        ax.plot(CX + GAUSS_SIGMA*np.cos(theta), CY + GAUSS_SIGMA*np.sin(theta),
                'w--', lw=0.8, alpha=0.7)

        # Method label in top-left corner
        lbl = res['label']
        if lbl=="Eq.\ Diffusion":
            lbl = "Eq. Diffusion"
        ax.text(0.03, 0.97, lbl,
                transform=ax.transAxes,
                fontsize=10, color='white',
                va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.4, lw=0))

        # Axis labels only on outer edges
        if ax in axes[2, :]:
            ax.set_xlabel('$x$ (cm)', fontsize=10)
        else:
            ax.set_xticklabels([])
        if ax in axes[:, 0]:
            ax.set_ylabel('$y$ (cm)', fontsize=10)
        else:
            ax.set_yticklabels([])

    # Shared colorbar on the right
    cbar_ax = fig.add_axes([0.92, 0.12, 0.015, 0.76])
    cb = fig.colorbar(ims[0], cax=cbar_ax)
    cb.set_label(r'Radiation temperature $T_r$ (keV)', fontsize=11)

    plt.savefig(os.path.join(os.getcwd(), savefile + '.pdf'),
                bbox_inches='tight')
    plt.savefig(os.path.join(os.getcwd(), savefile + '.png'),
                dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {savefile}.pdf / .png  (in {os.getcwd()})')


# ══════════════════════════════════════════════════════════════════════════
# Fiducial history comparison
# ══════════════════════════════════════════════════════════════════════════

def plot_fiducials(results_list, fid_label='center',
                   savefile='ray_effects_fiducials_center'):
    """T_rad vs time at one fiducial point for all 6 methods."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ls_cycle = ['-', '--', '-.', ':', '-', '--']
    mrker_cycle = [None,None,None,None, 'o', 's']
    colors   = plt.cm.tab10(range(len(results_list)))
    for res, ls, clr, mrk in zip(results_list, ls_cycle, colors, mrker_cycle):
        ts = res['fid_times']
        Tv = res['fid_Trad'].get(fid_label)
        if Tv is None or len(ts) == 0:
            continue
        lbl = res['label']
        if lbl=="Eq.\ Diffusion":
            lbl = "Eq. Diffusion"
        #place the marker every 10 points, but at least once
        markevery = max(1, len(ts) // 10)
        ax.semilogy(ts, Tv, ls=ls, lw=2, color=clr, marker=mrk, markevery=markevery, label=lbl if lbl else None)
    ax.axvline(T_COMPARE, color='k', ls=':', lw=1, alpha=0.5,
               label=f'$t={T_COMPARE}$ ns')

    ax.set_yscale('linear')
    fp = _font if HAS_PLOTFUNCS else None
    ax.set_xlabel('Time (ns)',    fontproperties=fp, fontsize=12)
    ax.set_ylabel(r'$T_r$ (keV)', fontproperties=fp, fontsize=12)
    leg = ax.legend(fontsize=9, ncol=2)
    if HAS_PLOTFUNCS:
        for txt in leg.get_texts():
            txt.set_fontproperties(_font)
            txt.set_fontsize(9)
    ax.grid(True, which='both', alpha=0.3, ls='--')
    plt.tight_layout()

    if HAS_PLOTFUNCS:
        _hide_spines()
    plt.savefig(os.path.join(os.getcwd(), savefile + '.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(os.getcwd(), savefile + '.png'), dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {savefile}.pdf / .png  (in {os.getcwd()})')


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Hot-spot ray-effect comparison')
    parser.add_argument('--Nsn',     type=int,   default=4,   help='S_N order')
    parser.add_argument('--Ix',      type=int,   default=80,  help='S_N radial cells')
    parser.add_argument('--Nd',      type=int,   default=100, help='Diffusion cells')
    parser.add_argument('--Nimc',    type=int,   default=500_000)
    parser.add_argument('--load',    action='store_true',
                        help='Load existing NPZ instead of re-running')
    args = parser.parse_args()

    def load_or_run(tag, runner):
        npz = os.path.join(os.getcwd(), f'{tag}_snapshot.npz')
        if args.load and os.path.exists(npz):
            print(f'  Loading {npz}')
            d   = np.load(npz, allow_pickle=True)
            labels  = list(d['fid_labels'])
            fid_Trad = {l: d['fid_Trad'][:, i] for i, l in enumerate(labels)}
            res = make_result(
                d['T_rad'], d['T_mat'],
                d['x_faces'], d['y_faces'],
                d['fid_times'], fid_Trad, fid_Trad,
                str(d['method']))
            return res
        res = runner()
        save_npz(res, npz)
        return res

    results = [
        load_or_run('eq_diff',  lambda: run_eq_diffusion(Nx=args.Nd, Ny=args.Nd)),
        load_or_run('fld',      lambda: run_fld(Nx=args.Nd, Ny=args.Nd)),
        load_or_run('imc',      lambda: run_imc(Nx=args.Nd, Ny=args.Nd, Ntarget=args.Nimc)),
        load_or_run('sn_ew',    lambda: run_sn('equal_weight',     args.Nsn, args.Ix, args.Ix,
                                               label=f'$S_{args.Nsn}$ Equal Weight')),
        load_or_run('sn_ls',    lambda: run_sn('level_symmetric',  args.Nsn, args.Ix, args.Ix,
                                               label=f'$S_{args.Nsn}$ Level-Symmetric')),
        load_or_run('sn_ps',    lambda: run_sn('product_triangular',   args.Nsn, args.Ix, args.Ix,
                                               label=f'$S_{args.Nsn}$ Product-Triangle')),
    ]

    save_fiducials_npz(results,
                       os.path.join(os.getcwd(), 'ray_effects_fiducials.npz'))

    print('\nGenerating figures...')
    plot_comparison(results, T_COMPARE)
    plot_fiducials(results, 'center',          'ray_effects_fiducials_center')
    plot_fiducials(results, 'diagonal +1 cm',  'ray_effects_fiducials_diag')

    print(f'\nAll outputs in {os.getcwd()}')


if __name__ == '__main__':
    main()
