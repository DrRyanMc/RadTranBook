#!/usr/bin/env python3
"""
Cylindrical shell free-streaming problem using IMC r-z geometry.

An infinitely tall cylindrical shell occupying r in [1.0, 1.25] cm is
initialized to T_r = 1 keV.  The opacity is set to a very small value
(sigma_a = 1e-8 cm^-1) to approximate the free-streaming (vacuum-transport)
limit.  Reflecting BCs at r=0 (axis) and at both z faces (infinite-z limit);
vacuum BC at r_max.

Output times: ct = 0.1, 0.25, 0.5, 1.0, 1.25, 1.5, 2.0 cm
              t  = ct / c   (ns)

Produces:
  cyl_shell_freestream.npz
  cyl_shell_freestream_profiles.pdf    — radial T_r profiles, all output times
  cyl_shell_freestream_xy_ct*.pdf      — 2-D x-y colormaps (one per time)
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

T_SHELL = 1.0      # keV — initial radiation temperature in the shell
T_COLD  = 1e-4     # keV — background temperature

SIGMA_A = 1e-8     # cm^-1 — near-zero, approximating free streaming

# Large cv so material temperature barely changes (sigma≈0 means no coupling)
CV      = 1.0      # GJ/(cm^3 keV)

# ─────────────────────────────────────────────────────────────────────────────
# Mesh
# ─────────────────────────────────────────────────────────────────────────────
DR    = 0.025   # cm — radial cell width (shell spans 10 cells)
# At t_final the outward front has reached R_OUT + 2.0 = 3.25 cm; give margin.
R_MAX = 4.5     # cm — outer boundary (vacuum BC)

# Single z-cell with reflecting top/bottom → effectively an infinite-z problem.
DZ    = 0.25    # cm

# ─────────────────────────────────────────────────────────────────────────────
# Time stepping
# ─────────────────────────────────────────────────────────────────────────────
CT_VALS = [0.1, 0.25, 0.5, 1.0, 1.25, 1.5, 2.0]          # c*t in cm
T_OUTS  = [ct / C_LIGHT for ct in CT_VALS]                  # ns

# Base time step: c * dt = 0.1 cm (4 cells per step).
# Clamped at each output time so we hit them exactly.
DT_BASE = 0.1 / C_LIGHT   # ns

# ─────────────────────────────────────────────────────────────────────────────
# Particle counts
# ─────────────────────────────────────────────────────────────────────────────
N_IC     = 200_000   # initial-condition particles (almost all land in shell)
NTARGET  = 200       # per-step emission (sigma≈0 → near-zero in practice)
NMAX     = 300_000   # population cap after combing

PREFIX   = 'cyl_shell_freestream'

# ─────────────────────────────────────────────────────────────────────────────
# Material model
# ─────────────────────────────────────────────────────────────────────────────
def eos(T):           return CV * T
def inv_eos(u):       return u / CV
def cv_func(T):       return np.zeros_like(T) + CV
def sigma_a_func(T):  return np.zeros_like(T) + SIGMA_A


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────
def run(r_edges, z_edges):
    nr = len(r_edges) - 1
    nz = 1
    rc = 0.5 * (r_edges[:-1] + r_edges[1:])

    # Initial conditions ── hot in shell, cold elsewhere
    T_mat  = np.full((nr, nz), T_COLD)
    Tr_mat = np.full((nr, nz), T_COLD)
    shell_mask = (rc >= R_IN) & (rc <= R_OUT)
    Tr_mat[shell_mask, 0] = T_SHELL

    source     = np.zeros((nr, nz))
    T_boundary = (0.0, 0.0, 0.0, 0.0)   # vacuum on all boundaries
    # reflect order: (rmin=axis, rmax=vacuum, zmin=periodic, zmax=periodic)
    reflect    = (True, False, True, True)

    print("=" * 65)
    print("Cylindrical Shell Free-Streaming  (IMC r-z)")
    print("=" * 65)
    print(f"  Shell: r ∈ [{R_IN}, {R_OUT}] cm,  T_r = {T_SHELL} keV")
    print(f"  Domain: r ∈ [0, {R_MAX}] cm,  dr = {DR} cm,  {nr} cells")
    print(f"  z slice: [0, {DZ}] cm  (reflecting → infinite-z)")
    print(f"  σ_a = {SIGMA_A:.1e} cm⁻¹  (free-streaming limit)")
    print(f"  Output c·t values: {CT_VALS} cm")
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

    # t = 0 snapshot: use exact prescribed profile (not MC-sampled noise)
    Tr_exact_t0 = np.where(shell_mask, T_SHELL, T_COLD).astype(float)
    snapshots = {0.0: Tr_exact_t0}

    t_sorted = sorted(T_OUTS)
    oi = 0   # next output-time index

    while oi < len(t_sorted):
        target    = t_sorted[oi]
        remaining = target - state.time

        if remaining < 1e-14:
            # Already at (or past) this target; record and advance
            snapshots[target] = state.radiation_temperature[:, 0].copy()
            ct = target * C_LIGHT
            n  = len(state.weights)
            print(f"  ▶ Snapshot at c·t = {ct:.3f} cm  (N = {n})")
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
            snapshots[target] = state.radiation_temperature[:, 0].copy()
            ct = state.time * C_LIGHT
            n  = len(state.weights)
            print(f"  ▶ Snapshot at c·t = {ct:.3f} cm  (N = {n})")
            oi += 1

    return snapshots


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────
CMAP = 'inferno'
VMIN = T_COLD
VMAX = T_SHELL


def _xy_grid_from_1d(Tr_r, r_edges, r_plot_max, N=512):
    """Rotate 1-D radial T_r profile into a 2-D x-y grid (azimuthal symmetry)."""
    xi = np.linspace(-r_plot_max, r_plot_max, N)
    X, Y = np.meshgrid(xi, xi)
    R = np.sqrt(X**2 + Y**2)
    # Map each (x,y) → r-cell index
    idx = np.searchsorted(r_edges, R, side='right') - 1
    idx = np.clip(idx, 0, len(Tr_r) - 1)
    Tr2d = Tr_r[idx].astype(float)
    Tr2d[R > r_edges[-1]] = np.nan    # mask outside domain
    return X, Y, Tr2d


def plot_xy_colormap(Tr_r, r_edges, t_ns, outname):
    ct = t_ns * C_LIGHT
    # Zoom the x-y extent to the wavefront region with a small margin
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

    # Dashed circles marking original shell boundaries
    theta_c = np.linspace(0.0, 2.0 * np.pi, 500)
    for R_shell in (R_IN, R_OUT):
        ax.plot(R_shell * np.cos(theta_c), R_shell * np.sin(theta_c),
                'c--', lw=0.9, alpha=0.7)

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


def plot_profiles(snapshots, r_centers, outname):
    fig, ax = plt.subplots(figsize=(9, 5.5))

    cmap_p = plt.get_cmap('plasma', len(T_OUTS))

    # Initial condition (exact block profile)
    ax.step(r_centers, snapshots[0.0], where='mid',
            color='k', lw=1.8, ls='--', label=r'$t=0$', zorder=10)

    for ii, t_ns in enumerate(sorted(T_OUTS)):
        Tr = snapshots.get(t_ns)
        if Tr is None:
            continue
        ct = t_ns * C_LIGHT
        ax.step(r_centers, Tr, where='mid',
                color=cmap_p(ii), lw=1.5,
                label=rf'$c\,t = {ct:.2f}$ cm')

    ax.set_yscale('log')
    ax.set_ylim(T_COLD * 0.5, T_SHELL * 2)
    ax.set_xlabel('r (cm)', fontsize=12)
    ax.set_ylabel(r'$T_r$ (keV)', fontsize=12)
    ax.set_title('Free-streaming cylindrical shell: radiation temperature profiles',
                 fontsize=11)
    ax.axvline(R_IN,  color='gray', ls=':', lw=0.9, alpha=0.6)
    ax.axvline(R_OUT, color='gray', ls=':', lw=0.9, alpha=0.6)

    leg = ax.legend(prop=font, fontsize=9, ncol=2, loc='lower right')
    leg.get_frame().set_alpha(None)
    ax.grid(True, alpha=0.25, which='both')
    ax.set_xlim(0.0, r_centers[-1])

    plt.tight_layout()
    show(outname, close_after=True)
    print(f'Saved: {outname}')


def plot_panel_grid(snapshots, r_edges, outname):
    """4×2 grid of x-y colormaps for every output time on one page."""
    all_ct = [(t_ns, t_ns * C_LIGHT) for t_ns in sorted(T_OUTS) if t_ns in snapshots]
    ncols = 4
    nrows = int(np.ceil(len(all_ct) / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 3.5, nrows * 3.3),
                              squeeze=False)

    theta_c = np.linspace(0.0, 2.0 * np.pi, 400)

    for idx, (t_ns, ct) in enumerate(all_ct):
        ax  = axes[idx // ncols][idx % ncols]
        Tr  = snapshots[t_ns]
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

        for R_shell in (R_IN, R_OUT):
            ax.plot(R_shell * np.cos(theta_c), R_shell * np.sin(theta_c),
                    'c--', lw=0.7, alpha=0.6)

        ax.set_title(rf'$ct = {ct:.2f}$ cm', fontsize=10)
        ax.set_xlabel('x (cm)', fontsize=9)
        ax.set_ylabel('y (cm)', fontsize=9)
        ax.set_aspect('equal')

    # Hide unused axes
    for idx in range(len(all_ct), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

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
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])

    snapshots = run(r_edges, z_edges)

    # ── Save NPZ ────────────────────────────────────────────────────────────
    all_t    = [0.0] + sorted(T_OUTS)
    Tr_stack = np.array([snapshots[t] for t in all_t])
    np.savez(
        f'{PREFIX}.npz',
        r_centers = r_centers,
        r_edges   = r_edges,
        t_vals    = np.array(all_t),
        ct_vals   = np.array([t * C_LIGHT for t in all_t]),
        Tr_snaps  = Tr_stack,
    )
    print(f'Saved: {PREFIX}.npz')

    # ── Radial profile plot ─────────────────────────────────────────────────
    print("\nPlotting radial profiles...")
    plot_profiles(snapshots, r_centers, f'{PREFIX}_profiles.pdf')

    # ── Individual x-y colormaps ────────────────────────────────────────────
    print("Plotting x-y colormaps...")
    for t_ns in [0.0] + sorted(T_OUTS):
        if t_ns not in snapshots:
            continue
        ct  = t_ns * C_LIGHT
        tag = '000' if t_ns == 0.0 else f'{ct:.2f}'.replace('.', 'p')
        outname = f'{PREFIX}_xy_ct{tag}.pdf'
        plot_xy_colormap(snapshots[t_ns], r_edges, t_ns, outname)

    # ── Panel grid (all times on one page) ──────────────────────────────────
    print("Plotting panel grid...")
    plot_panel_grid(snapshots, r_edges, f'{PREFIX}_xy_grid.pdf')

    print("\nDone.")
