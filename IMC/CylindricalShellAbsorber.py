#!/usr/bin/env python3
"""
Cylindrical shell free-streaming problem with a central absorber (IMC r-z).

Same as CylindricalShellFreeStream.py but a dense absorbing cylinder of
radius 0.1 cm is placed at the axis:
  r < R_ABS  : sigma_a = 100 cm^-1,  cv = 0.3 GJ/(cm^3 keV)
  r >= R_ABS : sigma_a = 1e-8 cm^-1,  cv = 1.0 GJ/(cm^3 keV)  [free-stream]

The inward-going pulse from the shell reaches the absorber around ct ~ 0.9 cm,
deposits its energy rapidly (mean-free-path = 0.01 cm << R_ABS), heats the
absorber, which then slowly re-emits.

Output times: ct = 0.1, 0.25, 0.5, 1.0, 1.25, 1.5, 2.0 cm

Produces:
  cyl_shell_absorber.npz
  cyl_shell_absorber_Tr_profiles.pdf    — radial T_r profiles
  cyl_shell_absorber_Tmat_profiles.pdf  — radial T_mat profiles
  cyl_shell_absorber_xy_ct*.pdf         — 2-D x-y colormaps (T_r, one per time)
  cyl_shell_absorber_xy_grid.pdf        — panel grid of all output times
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

# ── path setup ───────────────────────────────────────────────────────────────
_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_DIR)
for _p in (_DIR, _PROJECT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import IMC2D as imc2d
from utils.plotfuncs import show, font

# ─────────────────────────────────────────────────────────────────────────────
# Physical / problem constants
# ─────────────────────────────────────────────────────────────────────────────
C_LIGHT = 29.98    # cm/ns
A_RAD   = 0.01372  # GJ/(cm^3 keV^4)

R_IN    = 1.00     # cm — inner shell radius
R_OUT   = 1.25     # cm — outer shell radius
R_ABS   = 0.10     # cm — absorber radius

T_SHELL = 1.0      # keV — initial radiation temperature in the shell
T_COLD  = 1e-4     # keV — background temperature everywhere at t=0

SIGMA_FREE = 1e-8  # cm^-1 — effectively transparent
SIGMA_ABS  = 100.0 # cm^-1 — opaque absorber  (mfp = 0.01 cm)

CV_FREE = 1.0      # GJ/(cm^3 keV) — large so T_mat barely moves outside
CV_ABS  = 0.01      # GJ/(cm^3 keV) — absorber specific heat

# ─────────────────────────────────────────────────────────────────────────────
# Mesh
# ─────────────────────────────────────────────────────────────────────────────
DR    = 0.025   # cm — cell width in free-stream region
# Absorber region: want ~4 cells across the 0.1-cm absorber
# Use the same DR throughout (absorber spans 4 cells at dr=0.025).
R_MAX = 4.5     # cm

DZ    = 0.25    # cm — single z-cell (reflecting → infinite-z)

# ─────────────────────────────────────────────────────────────────────────────
# Time stepping
# ─────────────────────────────────────────────────────────────────────────────
CT_VALS = [0.1, 0.25, 0.5, 1.0, 1.25, 1.5, 2.0,5.0,10.0]
T_OUTS  = [ct / C_LIGHT for ct in CT_VALS]

DT_BASE = 0.1 / C_LIGHT   # ns  (c*dt = 0.1 cm)

# ─────────────────────────────────────────────────────────────────────────────
# Particle counts
# ─────────────────────────────────────────────────────────────────────────────
N_IC    = 200_000
NTARGET = 5_000    # higher than free-stream case: absorber emits a lot
NMAX    = 400_000

PREFIX  = 'cyl_shell_absorber'

# ─────────────────────────────────────────────────────────────────────────────
# Spatially-varying material model
# ─────────────────────────────────────────────────────────────────────────────
# These are called with 2-D arrays shaped (nr, nz=1).
# We build the spatial masks once at run-time after the mesh is known.

def make_spatially_varying(r_centers):
    """Return per-cell sigma_a, cv, eos, and inv_eos functions."""
    nr = len(r_centers)
    in_absorber = r_centers < R_ABS   # shape (nr,)

    sigma_arr = np.where(in_absorber, SIGMA_ABS, SIGMA_FREE)   # (nr,)
    cv_arr    = np.where(in_absorber, CV_ABS,    CV_FREE)       # (nr,)

    # Broadcast to (nr, 1) for 2-D IMC calls
    sigma_arr = sigma_arr[:, np.newaxis]
    cv_arr    = cv_arr[:, np.newaxis]

    def sigma_a_func(T):
        # T has shape (nr, 1); return same-shape array
        return np.broadcast_to(sigma_arr, T.shape).copy()

    def eos(T):
        return cv_arr * T

    def inv_eos(u):
        return u / cv_arr

    def cv_func(T):
        return np.broadcast_to(cv_arr, T.shape).copy()

    return sigma_a_func, eos, inv_eos, cv_func


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────
def run(r_edges, z_edges):
    nr = len(r_edges) - 1
    rc = 0.5 * (r_edges[:-1] + r_edges[1:])

    sigma_a_func, eos, inv_eos, cv_func = make_spatially_varying(rc)

    # ── Initial conditions ──────────────────────────────────────────────────
    T_mat  = np.full((nr, 1), T_COLD)
    Tr_mat = np.full((nr, 1), T_COLD)
    shell_mask = (rc >= R_IN) & (rc <= R_OUT)
    Tr_mat[shell_mask, 0] = T_SHELL

    source     = np.zeros((nr, 1))
    T_boundary = (0.0, 0.0, 0.0, 0.0)
    reflect    = (True, False, True, True)   # axis / vacuum / z-periodic / z-periodic

    n_abs = int(np.sum(rc < R_ABS))
    print("=" * 65)
    print("Cylindrical Shell + Central Absorber  (IMC r-z)")
    print("=" * 65)
    print(f"  Shell:    r ∈ [{R_IN}, {R_OUT}] cm,  T_r = {T_SHELL} keV")
    print(f"  Absorber: r < {R_ABS} cm  ({n_abs} cells)")
    print(f"            σ_a = {SIGMA_ABS} cm⁻¹,  mfp = {1/SIGMA_ABS:.3f} cm")
    print(f"            c_v = {CV_ABS} GJ/(cm³·keV)")
    print(f"  Vacuum:   σ_a = {SIGMA_FREE:.1e} cm⁻¹")
    print(f"  Domain:   r ∈ [0, {R_MAX}] cm,  dr = {DR} cm,  {nr} cells")
    print(f"  Output c·t: {CT_VALS} cm")
    print()

    state = imc2d.init_simulation(
        Ntarget    = NTARGET,
        Tinit      = T_mat,
        Tr_init    = Tr_mat,
        edges1     = r_edges,
        edges2     = z_edges,
        eos        = eos,
        inv_eos    = inv_eos,
        Ntarget_ic = N_IC,
        geometry   = 'rz',
    )

    # t = 0 snapshot
    Tr_exact_t0   = np.where(shell_mask, T_SHELL, T_COLD).astype(float)
    Tmat_exact_t0 = np.full(nr, T_COLD)
    tr_snapshots   = {0.0: Tr_exact_t0}
    tmat_snapshots = {0.0: Tmat_exact_t0}

    t_sorted = sorted(T_OUTS)
    oi = 0

    while oi < len(t_sorted):
        target    = t_sorted[oi]
        remaining = target - state.time

        if remaining < 1e-14:
            tr_snapshots[target]   = state.radiation_temperature[:, 0].copy()
            tmat_snapshots[target] = state.temperature[:, 0].copy()
            ct = target * C_LIGHT
            print(f"  ▶ Snapshot at c·t = {ct:.3f} cm  (N = {len(state.weights)})")
            oi += 1
            continue

        dt = min(DT_BASE, remaining)

        state, info = imc2d.step(
            state            = state,
            Ntarget          = NTARGET,
            Nboundary        = 0,
            Nsource          = 0,
            Nmax             = NMAX,
            T_boundary       = T_boundary,
            dt               = dt,
            edges1           = r_edges,
            edges2           = z_edges,
            sigma_a_func     = sigma_a_func,
            inv_eos          = inv_eos,
            cv               = cv_func,
            source           = source,
            reflect          = reflect,
            theta            = 1.0,
            geometry         = 'rz',
            rz_linear_source = False,
        )

        if abs(state.time - target) < 1e-12:
            tr_snapshots[target]   = state.radiation_temperature[:, 0].copy()
            tmat_snapshots[target] = state.temperature[:, 0].copy()
            ct = state.time * C_LIGHT
            print(f"  ▶ Snapshot at c·t = {ct:.3f} cm  (N = {len(state.weights)})")
            oi += 1

    return tr_snapshots, tmat_snapshots


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────
CMAP  = 'inferno'
VMIN  = T_COLD
VMAX  = T_SHELL

_CMAP_P = plt.get_cmap('plasma', len(T_OUTS))
_THETA  = np.linspace(0.0, 2.0 * np.pi, 500)


def _xy_grid_from_1d(Tr_r, r_edges, r_plot_max, N=512):
    xi  = np.linspace(-r_plot_max, r_plot_max, N)
    X, Y = np.meshgrid(xi, xi)
    R   = np.sqrt(X**2 + Y**2)
    idx = np.searchsorted(r_edges, R, side='right') - 1
    idx = np.clip(idx, 0, len(Tr_r) - 1)
    Tr2d = Tr_r[idx].astype(float)
    Tr2d[R > r_edges[-1]] = np.nan
    return X, Y, Tr2d


def plot_profiles(snapshots, r_centers, ylabel, title, outname, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.step(r_centers, snapshots[0.0], where='mid',
            color='k', lw=1.8, ls='--', label=r'$t=0$', zorder=10)

    for ii, t_ns in enumerate(sorted(T_OUTS)):
        snap = snapshots.get(t_ns)
        if snap is None:
            continue
        ct = t_ns * C_LIGHT
        ax.step(r_centers, snap, where='mid',
                color=_CMAP_P(ii), lw=1.5,
                label=rf'$c\,t = {ct:.2f}$ cm')

    ax.set_yscale('log')
    if vmin is not None:
        ax.set_ylim(bottom=vmin)
    if vmax is not None:
        ax.set_ylim(top=vmax)
    ax.set_xlabel('r (cm)', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=11)

    # Mark region boundaries
    for rr, lbl in ((R_ABS, f'absorber\nedge'), (R_IN, 'shell'), (R_OUT, 'shell')):
        ax.axvline(rr, color='gray', ls=':', lw=0.9, alpha=0.6)

    leg = ax.legend(prop=font, fontsize=9, ncol=2, loc='lower right')
    leg.get_frame().set_alpha(None)
    ax.grid(True, alpha=0.25, which='both')
    ax.set_xlim(0.0, min(r_centers[-1], 4.0))  # zoom to region of interest

    plt.tight_layout()
    show(outname, close_after=True)
    print(f'Saved: {outname}')


def plot_xy_colormap(Tr_r, r_edges, t_ns, outname):
    ct = t_ns * C_LIGHT
    r_plot_max = min(r_edges[-1], max(R_OUT + ct + 0.3, R_OUT + 0.5))

    X, Y, Tr2d = _xy_grid_from_1d(Tr_r, r_edges, r_plot_max)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.pcolormesh(
        X, Y, Tr2d,
        norm      = LogNorm(vmin=VMIN, vmax=VMAX),
        cmap      = CMAP,
        shading   = 'auto',
        rasterized= True,
    )
    cbar = fig.colorbar(im, ax=ax, label=r'$T_r$ (keV)')

    for R_ring, ls in ((R_ABS, ':'), (R_IN, '--'), (R_OUT, '--')):
        ax.plot(R_ring * np.cos(_THETA), R_ring * np.sin(_THETA),
                'c', ls=ls, lw=0.9, alpha=0.7)

    title = (r'$T_r$ at $t=0$  (initial condition)'
             if t_ns == 0.0
             else rf'$T_r$ at $ct = {ct:.2f}$ cm')
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('x (cm)', fontsize=11)
    ax.set_ylabel('y (cm)', fontsize=11)
    ax.set_aspect('equal')
    plt.tight_layout()
    show(outname, close_after=True)
    print(f'Saved: {outname}')


def plot_panel_grid(tr_snapshots, r_edges, outname):
    all_ct = [(t_ns, t_ns * C_LIGHT) for t_ns in sorted(T_OUTS)
              if t_ns in tr_snapshots]
    ncols = 4
    nrows = int(np.ceil(len(all_ct) / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 3.5, nrows * 3.3),
                              squeeze=False)

    for idx, (t_ns, ct) in enumerate(all_ct):
        ax  = axes[idx // ncols][idx % ncols]
        Tr  = tr_snapshots[t_ns]
        r_plot_max = min(r_edges[-1], max(R_OUT + ct + 0.4, R_OUT + 0.5))
        X, Y, Tr2d = _xy_grid_from_1d(Tr, r_edges, r_plot_max)

        im = ax.pcolormesh(
            X, Y, Tr2d,
            norm      = LogNorm(vmin=VMIN, vmax=VMAX),
            cmap      = CMAP,
            shading   = 'auto',
            rasterized= True,
        )
        fig.colorbar(im, ax=ax, label=r'$T_r$ (keV)')

        for R_ring, ls in ((R_ABS, ':'), (R_IN, '--'), (R_OUT, '--')):
            ax.plot(R_ring * np.cos(_THETA), R_ring * np.sin(_THETA),
                    'c', ls=ls, lw=0.7, alpha=0.6)

        ax.set_title(rf'$ct = {ct:.2f}$ cm', fontsize=10)
        ax.set_xlabel('x (cm)', fontsize=9)
        ax.set_ylabel('y (cm)', fontsize=9)
        ax.set_aspect('equal')

    for idx in range(len(all_ct), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.tight_layout()
    show(outname, close_after=True)
    print(f'Saved: {outname}')


def plot_absorber_history(tmat_snapshots, tr_snapshots, r_centers, outname):
    """Plot T_mat and T_rad at the absorber center (r≈0) vs ct."""
    abs_idx = np.argmin(r_centers)   # closest to axis
    ct_vals = []
    Tmat_vals = []
    Tr_vals   = []

    all_t = [0.0] + sorted(T_OUTS)
    for t_ns in all_t:
        if t_ns in tmat_snapshots:
            ct_vals.append(t_ns * C_LIGHT)
            Tmat_vals.append(tmat_snapshots[t_ns][abs_idx])
            Tr_vals.append(tr_snapshots[t_ns][abs_idx])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ct_vals, Tmat_vals, 'o-', color='tab:red',  lw=2, label=r'$T_\mathrm{mat}$  at $r\approx 0$')
    ax.plot(ct_vals, Tr_vals,   's-', color='tab:blue', lw=2, label=r'$T_r$ at $r\approx 0$')
    ax.set_xlabel(r'$ct$ (cm)', fontsize=12)
    ax.set_ylabel('Temperature (keV)', fontsize=12)
    ax.set_title('Absorber centre temperature history', fontsize=11)
    ax.set_yscale('log')
    leg = ax.legend(prop=font, fontsize=10)
    leg.get_frame().set_alpha(None)
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    show(outname, close_after=True)
    print(f'Saved: {outname}')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    nr      = int(round(R_MAX / DR))
    r_edges = np.linspace(0.0, R_MAX, nr + 1)
    z_edges = np.array([0.0, DZ])
    rc      = 0.5 * (r_edges[:-1] + r_edges[1:])

    tr_snaps, tmat_snaps = run(r_edges, z_edges)

    # ── Save NPZ ─────────────────────────────────────────────────────────────
    all_t    = [0.0] + sorted(T_OUTS)
    Tr_stack   = np.array([tr_snaps[t]   for t in all_t if t in tr_snaps])
    Tmat_stack = np.array([tmat_snaps[t] for t in all_t if t in tmat_snaps])
    np.savez(
        f'{PREFIX}.npz',
        r_centers  = rc,
        r_edges    = r_edges,
        t_vals     = np.array([t for t in all_t if t in tr_snaps]),
        ct_vals    = np.array([t * C_LIGHT for t in all_t if t in tr_snaps]),
        Tr_snaps   = Tr_stack,
        Tmat_snaps = Tmat_stack,
    )
    print(f'Saved: {PREFIX}.npz')

    # ── Radial profiles ───────────────────────────────────────────────────────
    print("\nPlotting radial T_r profiles...")
    plot_profiles(
        tr_snaps, rc,
        ylabel = r'$T_r$ (keV)',
        title  = 'Cyl-shell + absorber: radiation temperature profiles',
        outname= f'{PREFIX}_Tr_profiles.pdf',
        vmin   = T_COLD * 0.5,
        vmax   = T_SHELL * 3,
    )

    print("Plotting radial T_mat profiles...")
    plot_profiles(
        tmat_snaps, rc,
        ylabel = r'$T_\mathrm{mat}$ (keV)',
        title  = 'Cyl-shell + absorber: material temperature profiles',
        outname= f'{PREFIX}_Tmat_profiles.pdf',
        vmin   = T_COLD * 0.5,
        vmax   = T_SHELL * 3,
    )

    # ── Absorber history ──────────────────────────────────────────────────────
    print("Plotting absorber centre history...")
    plot_absorber_history(tmat_snaps, tr_snaps, rc, f'{PREFIX}_absorber_history.pdf')

    # ── x-y colormaps ─────────────────────────────────────────────────────────
    print("Plotting x-y colormaps...")
    for t_ns in [0.0] + sorted(T_OUTS):
        if t_ns not in tr_snaps:
            continue
        ct  = t_ns * C_LIGHT
        tag = '000' if t_ns == 0.0 else f'{ct:.2f}'.replace('.', 'p')
        plot_xy_colormap(tr_snaps[t_ns], r_edges, t_ns, f'{PREFIX}_xy_ct{tag}.pdf')

    print("Plotting panel grid...")
    plot_panel_grid(tr_snaps, r_edges, f'{PREFIX}_xy_grid.pdf')

    print("\nDone.")
