#!/usr/bin/env python3
"""
Spherical S_N verification and demonstration examples.

Five problems that probe distinct features of the 1-D spherical LD-S_N solver:

  1  Constant-solution preservation   — exact flat φ(r) = 2 I_0
  2  Origin-regularity test           — smooth φ near r = 0, convergence of ∂φ/∂r
  3  Flux-dip demonstration           — ordinary DD vs. weighted DD near origin
  4  Conservative positivity fixup    — negative intensities at an opacity interface
  5  Zel'dovich wave energy balance   — global energy conservation diagnostic

Usage
-----
    python problems/spherical_sn_examples.py              # run all
    python problems/spherical_sn_examples.py --examples 1 3
    python problems/spherical_sn_examples.py --prefix myrun --examples 1 2 3 4 5

Output: PNG + PDF figures in the current working directory.
"""

import sys
import os
import argparse

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_this_dir = os.path.dirname(os.path.abspath(__file__))
_sn_dir = os.path.dirname(_this_dir)           # DiscreteOrdinates/
_project_root = os.path.dirname(_sn_dir)        # RadTranBook/

sys.path.insert(0, _sn_dir)
sys.path.insert(0, _project_root)

from sn_solver_ld_sphere import (
    single_sweep_phi_sph_ld,
    single_sweep_psi_sph_ld,
    temp_solve_sph_ld,
    _compute_sph_quad_data,
    _compute_geometric_moments,
)
from sn_solver import _get_quadrature, a as A_RAD, c as C_LIGHT, ac as AC

# ---------------------------------------------------------------------------
# Figure style helpers
# ---------------------------------------------------------------------------

def _setup_style():
    """Apply publication-quality matplotlib defaults."""
    for font_name in ["Univers LT Std", "TeX Gyre Heros", "Helvetica",
                      "DejaVu Sans"]:
        matplotlib.rcParams['font.family'] = 'sans-serif'
        matplotlib.rcParams['font.sans-serif'] = [font_name]
        break
    matplotlib.rcParams.update({
        'font.size':           12,
        'axes.spines.top':    False,
        'axes.spines.right':  False,
    })


def _style_ax(ax):
    for spine in ('bottom', 'left'):
        ax.spines[spine].set_linewidth(1.5)
    ax.grid(True, alpha=0.25)


def _save_fig(fig, name):
    fig.savefig(name + '.png', dpi=160, bbox_inches='tight')
    fig.savefig(name + '.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved  {name}.png  and  {name}.pdf")


# ---------------------------------------------------------------------------
# Steady-state source-iteration helper
# ---------------------------------------------------------------------------

def steady_state_si(I, r_left, dr, sigma_t, sigma_s, q_iso, N,
                    bc_outer, bc_g_outer,
                    tol=1e-10, max_its=5000, s_override=None):
    """Source iteration for steady-state spherical S_N.

    Solves  (μ ∂/∂r + σ_t) I_n = σ_s φ/2 + q_iso/2

    Parameters
    ----------
    sigma_t  : (I, 2)  total opacity
    sigma_s  : (I, 2)  scattering opacity
    q_iso    : (I, 2)  external isotropic volumetric source Q(r)
    s_override : (N,) or None — override angular-differencing weights (for
                 ordinary DD pass np.full(N, 0.5))

    Returns
    -------
    phi : (I, 2)
    niters : int
    """
    sigma_hat = sigma_t.copy()          # steady state: no 1/(c Δt) term
    phi = np.zeros((I, 2))

    for it in range(max_its):
        iso_src = sigma_s * phi + q_iso  # total isotropic source (I, 2)
        # Each ordinate receives half of the isotropic source
        source_n = (0.5 * iso_src)[:, np.newaxis, :] * np.ones((1, N, 1))
        source_g = 0.5 * iso_src

        phi_new = single_sweep_phi_sph_ld(
            I, r_left, dr,
            np.ascontiguousarray(source_n, dtype=np.float64),
            np.ascontiguousarray(source_g, dtype=np.float64),
            np.ascontiguousarray(sigma_hat, dtype=np.float64),
            N, bc_outer, float(bc_g_outer),
            s_override=s_override,
        )
        err = (np.max(np.abs(phi_new - phi))
               / (np.max(np.abs(phi_new)) + 1e-30))
        phi = phi_new
        if err < tol:
            return phi, it + 1

    return phi, max_its


# ===========================================================================
# Example 1 — Constant-solution preservation
# ===========================================================================

def example1_constant_preservation(
        save_prefix='ex1_const',
        R=10.0,
        sigma_0=1.0,
        I0=1.0):
    """Verify φ(r) / (2 I_0) = 1 for a uniform isotropic radiation field.

    For a constant isotropic field I_n(r) = I_0 ∀ n, r the scalar flux is
    φ = 2 I_0.  The source that exactly balances absorption is Q_n = σ_0 I_0
    per ordinate.  This test checks that the LD-SN sweep preserves the
    constant solution to machine precision.
    """
    print("\n=== Example 1: Constant-solution preservation ===")
    _setup_style()

    N_vals = [2, 4, 8, 16]
    I_vals = [32, 64, 128]
    colors = plt.cm.tab10(np.linspace(0.0, 0.6, len(N_vals)))

    print(f"  {'N':>4}  {'I':>5}  {'L∞ error':>12}")

    fig, ax = plt.subplots(figsize=(7, 5))

    for ni, N in enumerate(N_vals):
        MU, _ = _get_quadrature(N)

        # Outer BC: I_0 for each inward-moving ordinate, I_0 for the starting
        # direction g (μ = −1 entering from outside).
        bc_outer = np.zeros((N, 2))
        for n in range(N):
            if MU[n] < 0.0:
                bc_outer[n, 0] = I0
        bc_g_outer = I0

        for I in I_vals:
            dr    = np.full(I, R / I)
            r_left = np.arange(I, dtype=float) * R / I

            # For constant I_n = I_0: each ordinate's source = σ_0 I_0
            sigma_hat = np.full((I, 2), sigma_0)
            source_n  = np.full((I, N, 2), sigma_0 * I0)
            source_g  = np.full((I, 2),    sigma_0 * I0)

            phi = single_sweep_phi_sph_ld(
                I, r_left, dr, source_n, source_g,
                sigma_hat, N, bc_outer, bc_g_outer,
            )
            phi_norm = 0.5 * (phi[:, 0] + phi[:, 1]) / (2.0 * I0)
            err      = float(np.max(np.abs(phi_norm - 1.0)))
            print(f"  {N:>4}  {I:>5}  {err:>12.3e}")

        # Plot finest-mesh result for this N
        I = I_vals[-1]
        dr    = np.full(I, R / I)
        r_left = np.arange(I, dtype=float) * R / I
        r_cen  = r_left + 0.5 * dr

        sigma_hat = np.full((I, 2), sigma_0)
        source_n  = np.full((I, N, 2), sigma_0 * I0)
        source_g  = np.full((I, 2),    sigma_0 * I0)

        phi = single_sweep_phi_sph_ld(
            I, r_left, dr, source_n, source_g,
            sigma_hat, N, bc_outer, bc_g_outer,
        )
        phi_norm = 0.5 * (phi[:, 0] + phi[:, 1]) / (2.0 * I0)
        ax.plot(r_cen, phi_norm, lw=1.5, color=colors[ni],
                label=rf'$S_{{{N}}}$, $I={I}$')

    ax.axhline(1.0, color='k', lw=1.0, ls='--', alpha=0.6, label='Exact')
    _style_ax(ax)
    ax.set_xlabel('Radius $r$ (cm)')
    ax.set_ylabel(r'$\phi(r)\;/\;(2 I_0)$')
    ax.set_title('Example 1: Constant-solution preservation')
    ax.legend(frameon=False, fontsize=9)
    ax.set_ylim(0.95, 1.05)
    fig.tight_layout()
    _save_fig(fig, save_prefix)


# ===========================================================================
# Example 2 — Origin-regularity test
# ===========================================================================

def example2_origin_regularity(
        save_prefix='ex2_origin',
        R=5.0,
        sigma_0=2.0,
        q0=1.0,
        rs=0.5,
        N=8,
        scattering_c=0.5):
    """Check that φ is smooth at the origin for a Gaussian source.

    The left-to-right gradient of φ across the first cell,
      D_0 = (φ_{0,R} − φ_{0,L}) / Δr_0,
    should converge to zero as Δr → 0 (origin-regularity condition).
    """
    print("\n=== Example 2: Origin-regularity test ===")
    _setup_style()

    sigma_s0 = scattering_c * sigma_0
    sigma_a0 = (1.0 - scattering_c) * sigma_0   # noqa: F841

    I_vals  = [32, 64, 128, 256]
    D0_vals = []
    dr_vals = []
    colors  = plt.cm.viridis(np.linspace(0.15, 0.85, len(I_vals)))

    print(f"  {'I':>5}  {'niters':>8}  {'|D_0|':>12}  {'Δr':>10}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    MU, _ = _get_quadrature(N)
    bc_outer  = np.zeros((N, 2))   # vacuum outer BC
    bc_g_outer = 0.0

    for ii, I in enumerate(I_vals):
        dr_val = R / I
        dr      = np.full(I, dr_val)
        r_left  = np.arange(I, dtype=float) * dr_val
        r_cen   = r_left + 0.5 * dr_val
        r_l     = r_left
        r_r     = r_left + dr_val

        q_l   = q0 * np.exp(-(r_l / rs) ** 2)
        q_r   = q0 * np.exp(-(r_r / rs) ** 2)
        q_iso = np.column_stack([q_l, q_r])

        sigma_t   = np.full((I, 2), sigma_0)
        sigma_s   = np.full((I, 2), sigma_s0)

        phi, niters = steady_state_si(
            I, r_left, dr, sigma_t, sigma_s, q_iso,
            N, bc_outer, bc_g_outer,
        )

        D0 = abs(phi[0, 1] - phi[0, 0]) / dr_val
        D0_vals.append(D0)
        dr_vals.append(dr_val)
        print(f"  {I:>5}  {niters:>8}  {D0:>12.3e}  {dr_val:>10.5f}")

        # Plot near-origin scalar flux
        phi_cen = 0.5 * (phi[:, 0] + phi[:, 1])
        I_near  = min(25, I // 4)
        axes[0].plot(r_cen[:I_near], phi_cen[:I_near], '-',
                     color=colors[ii], lw=1.5, label=f'$I={I}$')

    # Origin-derivative convergence
    dr_arr = np.array(dr_vals)
    D0_arr = np.array(D0_vals)
    axes[1].loglog(dr_arr, D0_arr, 'o-', color='steelblue', lw=1.5,
                   label=r'$|D_0|$')
    d_ref = np.linspace(dr_arr.min() * 0.8, dr_arr.max() * 1.2, 50)
    for exp, ls, lab in [(1, '--', r'$\Delta r^1$'), (2, ':', r'$\Delta r^2$')]:
        ref = D0_arr[-1] * (d_ref / dr_arr[-1]) ** exp
        axes[1].loglog(d_ref, ref, 'k' + ls, alpha=0.55, label=lab)

    _style_ax(axes[0])
    axes[0].set_xlabel('Radius $r$ (cm)')
    axes[0].set_ylabel(r'$\phi(r)$')
    axes[0].set_title(f'Scalar flux near origin  ($S_{{{N}}}$)')
    axes[0].legend(frameon=False, fontsize=9)

    _style_ax(axes[1])
    axes[1].set_xlabel(r'Cell width $\Delta r$ (cm)')
    axes[1].set_ylabel(r'$|D_0| = |\phi_{0,R} - \phi_{0,L}|\,/\,\Delta r$')
    axes[1].set_title('Origin LD gradient vs mesh width')
    axes[1].legend(frameon=False, fontsize=9)

    fig.tight_layout()
    _save_fig(fig, save_prefix)


# ===========================================================================
# Example 3 — Flux-dip demonstration
# ===========================================================================

def example3_flux_dip(
        save_prefix='ex3_flux_dip',
        R=5.0,
        I=40,
        sigma_t0=2.0,
        c_scatter=0.0,
        q0=1.0,
        rs=1.0):
    """Compare ordinary DD (s_n = 0.5) with weighted DD near the origin.

    In spherical S_N, the angular-differencing coefficient s_n enters the
    reconstruction at each angular-mesh interface.  The weighted scheme sets
    s_n from the Gauss–Legendre ordinate positions so that the first angular
    moment of the transport equation is discretised consistently.  Using
    ordinary DD (s_n = 0.5 for all n) introduces an O(Δμ²) error that
    manifests as an overshoot of the scalar flux near the origin for coarse
    angular order.  The overshoot is largest at r = 0 and decays with N.

    A purely absorbing medium (c = 0) is used so that the effects are
    transport-dominated and clearly visible for low angular order.  The
    reference is S_{32} (N_ref = 16 ordinate pairs) with weighted DD.
    The left panel shows the full near-origin region; the right panel zooms
    into the first few cells to reveal the ordinary-vs-weighted DD difference.
    """
    print("\n=== Example 3: Angular-differencing effects near origin ===")
    _setup_style()

    sigma_s0 = c_scatter * sigma_t0

    dr    = np.full(I, R / I)
    r_left = np.arange(I, dtype=float) * R / I
    r_cen  = r_left + 0.5 * dr
    r_l    = r_left
    r_r    = r_left + dr

    q_l   = q0 * np.exp(-(r_l / rs) ** 2)
    q_r   = q0 * np.exp(-(r_r / rs) ** 2)
    q_iso = np.column_stack([q_l, q_r])

    sigma_t   = np.full((I, 2), sigma_t0)
    sigma_s   = np.full((I, 2), sigma_s0)

    N_ref  = 16          # reference: S32 weighted DD
    N_vals = [2, 4]      # S4, S8

    # ── High-N weighted-DD reference ─────────────────────────────────────────
    bc_ref = np.zeros((N_ref, 2))
    phi_ref, _ = steady_state_si(I, r_left, dr, sigma_t, sigma_s, q_iso,
                                  N_ref, bc_ref, 0.0)
    phi_ref_cen = 0.5 * (phi_ref[:, 0] + phi_ref[:, 1])
    phi0_ref    = max(phi_ref_cen[0], 1e-30)

    # Number of cells to show in main panel and zoom panel
    I_main = min(I, max(10, I // 3))         # first ~33 % of domain
    I_zoom = min(I, 5)                       # first 5 cells for zoom

    col_weighted = plt.cm.Blues(0.7)
    col_ordinary = plt.cm.Reds(0.7)

    # ── Collect data for both N values ────────────────────────────────────────
    data = []
    for ni, N in enumerate(N_vals):
        bc_outer = np.zeros((N, 2))

        phi_w, _ = steady_state_si(I, r_left, dr, sigma_t, sigma_s, q_iso,
                                    N, bc_outer, 0.0)
        phi_w_cen = 0.5 * (phi_w[:, 0] + phi_w[:, 1])

        s_ord = np.full(N, 0.5)
        phi_o, _ = steady_state_si(I, r_left, dr, sigma_t, sigma_s, q_iso,
                                    N, bc_outer, 0.0, s_override=s_ord)
        phi_o_cen = 0.5 * (phi_o[:, 0] + phi_o[:, 1])
        data.append((N, phi_w_cen, phi_o_cen))

    # ── Two-panel figure ──────────────────────────────────────────────────────
    fig, (ax_main, ax_zoom) = plt.subplots(
        1, 2, figsize=(10, 5),
        gridspec_kw={'width_ratios': [2, 1]})

    for ax, I_show in [(ax_main, I_main), (ax_zoom, I_zoom)]:
        ax.plot(r_cen[:I_show], phi_ref_cen[:I_show] / phi0_ref,
                'k-', lw=2.5, zorder=5,
                label=rf'$S_{{{2*N_ref}}}$ wt. (ref)')
        for ni, (N, phi_w_cen, phi_o_cen) in enumerate(data):
            lw = 1.5 + ni * 0.5
            ax.plot(r_cen[:I_show], phi_w_cen[:I_show] / phi0_ref,
                    '-', color=col_weighted, lw=lw, alpha=0.85,
                    label=rf'$S_{{{2*N}}}$ weighted')
            ax.plot(r_cen[:I_show], phi_o_cen[:I_show] / phi0_ref,
                    '--', color=col_ordinary, lw=lw, alpha=0.85,
                    label=rf'$S_{{{2*N}}}$ ordinary')
        _style_ax(ax)
        ax.set_xlabel(r'Radius $r$ (cm)')

    ax_main.set_ylabel(r'$\phi(r)\;/\;\phi_{\mathrm{ref}}(0)$')
    ax_main.legend(frameon=False, fontsize=9)

    # Zoom panel: tight y-axis to reveal small differences
    y_vals = np.concatenate(
        [phi_ref_cen[:I_zoom] / phi0_ref] +
        [np.concatenate([phi_w[:I_zoom], phi_o[:I_zoom]]) / phi0_ref
         for _, phi_w, phi_o in data])
    y_lo, y_hi = y_vals.min(), y_vals.max()
    pad = max(0.01, 0.15 * (y_hi - y_lo))
    ax_zoom.set_ylim(y_lo - pad, y_hi + pad)
    ax_zoom.set_title('Near-origin zoom', fontsize=10)

    fig.suptitle(
        rf'Example 3: Angular-differencing near origin  '
        rf'($\sigma_t={sigma_t0}$, $c={c_scatter}$, $I={I}$)',
        fontsize=12, y=1.01)
    fig.tight_layout()
    _save_fig(fig, save_prefix)


# ===========================================================================
# Example 4 — Conservative positivity fixup
# ===========================================================================

def example4_positivity_fixup(
        save_prefix='ex4_fixup',
        R=5.0,
        I=200,
        N=4,
        r_interface=2.5,
        sigma1=0.5,
        sigma2=100.0):
    """Show the effect of the conservative positivity fixup at an opacity jump.

    Radiation enters from the outer wall and must traverse a high-opacity
    shell (σ₂ ≫ σ₁).  Without the fixup (fix=0) intensities become negative
    near the interface; the fixup (fix=1) clamps them conservatively.

    The test tracks the minimum intensity per SI iteration and the converged
    scalar flux profile.
    """
    print("\n=== Example 4: Conservative positivity fixup ===")
    _setup_style()

    dr_val  = R / I
    dr      = np.full(I, dr_val)
    r_left  = np.arange(I, dtype=float) * dr_val
    r_cen   = r_left + 0.5 * dr_val

    # Two-region opacity (position-dependent, temperature-independent)
    r_l = r_left
    r_r = r_left + dr_val
    sig_l = np.where(r_l < r_interface, sigma1, sigma2)
    sig_r = np.where(r_r < r_interface, sigma1, sigma2)
    sigma_hat = np.column_stack([sig_l, sig_r])   # (I, 2)

    sigma_s = np.zeros((I, 2))   # pure absorption
    q_iso   = np.zeros((I, 2))   # no volumetric source

    # Outer BC: intense isotropic radiation at r = R
    I_bc = 1.0
    MU, _ = _get_quadrature(N)
    bc_outer = np.zeros((N, 2))
    for n in range(N):
        if MU[n] < 0.0:
            bc_outer[n, 0] = I_bc
    bc_g_outer = I_bc

    n_iters = 40
    fix_configs = [(1, '-', 'tab:blue'), (0, '--', 'tab:red')]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    print(f"  {'fix':>5}  {'psi_min (iter 1)':>20}  {'psi_min (converged)':>22}")
    for fix_val, ls, color in fix_configs:
        label       = f'fix={fix_val}'
        psi_min_iters = []
        phi         = np.zeros((I, 2))

        for it in range(n_iters):
            # Pure absorption → scattering source is zero; only fixed source
            source_n = (0.5 * sigma_s * phi)[:, np.newaxis, :] \
                       * np.ones((1, N, 1))
            source_g = 0.5 * sigma_s * phi
            psi_new, phi_new, _ = single_sweep_psi_sph_ld(
                I, r_left, dr,
                np.ascontiguousarray(source_n, dtype=np.float64),
                np.ascontiguousarray(source_g, dtype=np.float64),
                np.ascontiguousarray(sigma_hat, dtype=np.float64),
                N, bc_outer, float(bc_g_outer), fix=fix_val,
            )
            phi = phi_new
            psi_min_iters.append(float(np.min(psi_new)))

        print(f"  {fix_val:>5}  {psi_min_iters[0]:>20.3e}  {psi_min_iters[-1]:>22.3e}")

        axes[0].plot(range(1, n_iters + 1), psi_min_iters,
                     ls, color=color, lw=1.5, label=label)

        phi_cen = 0.5 * (phi[:, 0] + phi[:, 1])
        axes[1].plot(r_cen, phi_cen, ls, color=color, lw=1.5, label=label)

    axes[0].axhline(0.0, color='k', lw=0.8, ls=':', label=r'$I_n = 0$ floor')
    _style_ax(axes[0])
    axes[0].set_xlabel('SI iteration')
    axes[0].set_ylabel(r'$\min_{n,j,s}\;\psi_{n,j,s}$')
    axes[0].set_title('Minimum intensity per iteration')
    axes[0].legend(frameon=False)

    axes[1].axvline(r_interface, color='k', lw=0.8, ls=':', alpha=0.7,
                    label='Interface')
    _style_ax(axes[1])
    axes[1].set_xlabel('Radius $r$ (cm)')
    axes[1].set_ylabel(r'$\phi(r)$')
    axes[1].set_title('Converged scalar flux')
    axes[1].legend(frameon=False, fontsize=9)

    # Conservation diagnostic: r²-weighted angular moments per cell per ordinate
    # For the fix=1 converged state, check that M_l*ψ_l + M_r*ψ_r is non-negative
    # (positivity fixup preserves the r²-weighted mean by construction)
    M_l, M_r = _compute_geometric_moments(r_left, dr)
    print("\n  Conservation check (fix=1): r²-weighted ψ mean near interface:")
    phi_f1 = np.zeros((I, 2))
    for it in range(n_iters):
        source_n = np.zeros((I, N, 2))
        source_g = np.zeros((I, 2))
        psi_fin, phi_f1, _ = single_sweep_psi_sph_ld(
            I, r_left, dr, source_n, source_g,
            np.ascontiguousarray(sigma_hat, dtype=np.float64),
            N, bc_outer, float(bc_g_outer), fix=1,
        )
    j0 = int(r_interface / dr_val) - 2
    for j in range(max(0, j0), min(I, j0 + 5)):
        for n in range(N):
            moment = M_l[j] * psi_fin[j, n, 0] + M_r[j] * psi_fin[j, n, 1]
            print(f"    j={j:3d}, n={n}: M_l ψ_l + M_r ψ_r = {moment:.4e}")

    fig.tight_layout()
    _save_fig(fig, save_prefix)


# ===========================================================================
# Example 5 — Zel'dovich wave energy conservation
# ===========================================================================

def example5_zeldovich_energy(
        save_prefix='ex5_zeld_energy',
        I=200,
        N=8):
    """Run the spherical Zel'dovich wave and evaluate global energy balance.

    Since the outer boundary is reflecting (no flux in or out), the total
    radiation + material energy

        E_tot(t) = 4π ∫₀ᴿ r² [φ(r,t)/c + e(r,t)] dr

    should be conserved.  Any deviation quantifies the numerical energy error
    of the linearised Fleck–Cummings scheme.
    """
    print("\n=== Example 5: Zel'dovich wave energy conservation ===")
    _setup_style()

    # Import Zel'dovich driver from the same directory
    sys.path.insert(0, _this_dir)
    import zeldovich_sph_sn as zel

    output_times = (0.1, 0.3, 1.0, 3.0)
    results = zel.setup_and_run(
        I=I, N=N,
        dt_min=1e-4, dt_max=0.01,
        output_times=output_times,
    )

    r_cen     = results['r_centers']
    r_left_r  = results['r_left']
    dr_r      = results['dr']
    solutions = results['solutions']
    t_vals    = sorted(solutions.keys())

    # ── Energy diagnostic ─────────────────────────────────────────────────
    def E_tot(phi_arr, T_arr):
        """4π ∫ r² [φ/c + c_v T] dr using cell-centered LD values."""
        phi_c = 0.5 * (phi_arr[:, 0] + phi_arr[:, 1])
        T_c   = 0.5 * (T_arr[:, 0]   + T_arr[:, 1])
        E_rad = phi_c / C_LIGHT
        E_mat = zel.CV_VOL * np.maximum(T_c, 0.0)
        return 4.0 * np.pi * float(np.sum(r_cen ** 2 * (E_rad + E_mat) * dr_r))

    E_vals = [E_tot(solutions[t]['phi'], solutions[t]['T']) for t in t_vals]
    E_ref  = E_vals[0]
    eps_E  = [abs(e - E_ref) / (abs(E_ref) + 1e-30) for e in E_vals]

    print(f"\n  {'t (ns)':>8}  {'E_tot (GJ)':>14}  {'|ΔE|/E₀':>12}")
    for t, ev, ep in zip(t_vals, E_vals, eps_E):
        print(f"  {t:>8.2f}  {ev:>14.6e}  {ep:>12.3e}")

    # ── Radiation front location ──────────────────────────────────────────
    front_r = []
    for t in t_vals:
        T_c = 0.5 * (solutions[t]['T'][:, 0] + solutions[t]['T'][:, 1])
        T_peak = T_c.max()
        # front = outermost cell where T > 1% of peak
        mask = T_c > 0.01 * T_peak
        front_r.append(float(r_cen[mask].max()) if mask.any() else 0.0)

    # ── Plots ─────────────────────────────────────────────────────────────
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(t_vals)))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for idx, t_phys in enumerate(t_vals):
        sol   = solutions[t_phys]
        T_c   = 0.5 * (sol['T'][:, 0]   + sol['T'][:, 1])
        phi_c = 0.5 * (sol['phi'][:, 0] + sol['phi'][:, 1])
        axes[0].plot(r_cen, T_c,   '-', color=colors[idx], lw=1.5,
                     label=rf'$t={t_phys:.2g}$ ns')
        axes[1].plot(r_cen, phi_c, '-', color=colors[idx], lw=1.5)

    # Energy error vs. time
    axes[2].semilogy(t_vals, np.maximum(eps_E, 1e-16), 'o-',
                     color='steelblue', lw=1.5)

    for a, ttl, xl, yl in zip(
            axes,
            ['Temperature', 'Scalar flux', 'Relative energy error'],
            [r'$r$ (cm)', r'$r$ (cm)', r'$t$ (ns)'],
            [r'$T$ (keV)',
             r'$\phi$ (GJ cm$^{-2}$ ns$^{-1}$)',
             r'$|\Delta E_{\rm tot}| / E_{\rm tot}(t_0)$']):
        _style_ax(a)
        a.set_xlabel(xl)
        a.set_ylabel(yl)
        a.set_title(ttl)

    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(
        rf'Example 5: Zeldovich wave ($N={N}$, $I={I}$)',
        fontsize=13, y=1.01)
    fig.tight_layout()
    _save_fig(fig, save_prefix)

    print(f"\n  Radiation front at output times:")
    for t, rf in zip(t_vals, front_r):
        print(f"    t={t:.2f} ns:  r_f ≈ {rf:.3f} cm")


# ===========================================================================
# CLI entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Spherical LD-SN verification and demonstration examples.')
    parser.add_argument(
        '--examples', nargs='+', type=int, default=[1, 2, 3, 4, 5],
        metavar='N',
        help='Which example numbers to run (default: 1 2 3 4 5).')
    parser.add_argument(
        '--prefix', default='',
        help='Prepend this string to all output filenames.')
    parser.add_argument(
        '--I', type=int, default=None,
        help='Override number of radial cells for Examples 3/4/5.')
    parser.add_argument(
        '--N', type=int, default=None,
        help='Override number of ordinates for Examples 2/5.')
    args = parser.parse_args()

    pfx = (args.prefix + '_') if args.prefix else ''

    if 1 in args.examples:
        example1_constant_preservation(save_prefix=pfx + 'ex1_const')

    if 2 in args.examples:
        kw = {}
        if args.N:
            kw['N'] = args.N
        example2_origin_regularity(save_prefix=pfx + 'ex2_origin', **kw)

    if 3 in args.examples:
        kw = {}
        if args.I:
            kw['I'] = args.I
        example3_flux_dip(save_prefix=pfx + 'ex3_flux_dip', **kw)

    if 4 in args.examples:
        kw = {}
        if args.I:
            kw['I'] = args.I
        if args.N:
            kw['N'] = args.N
        example4_positivity_fixup(save_prefix=pfx + 'ex4_fixup', **kw)

    if 5 in args.examples:
        kw = {}
        if args.I:
            kw['I'] = args.I
        if args.N:
            kw['N'] = args.N
        example5_zeldovich_energy(save_prefix=pfx + 'ex5_zeld_energy', **kw)


if __name__ == '__main__':
    main()
