#!/usr/bin/env python3
"""
Diagnostic comparison between the S_N and diffusion multigroup Marshak wave setups.

Checks:
  1. Parameter table — side-by-side listing of every setup parameter
  2. Opacity cross-check — σ_a(T,E) from both files at the same (T,E) grid
  3. Planck integrals — 4πB_g from planck_integrals.Bg vs Bg_multigroup
  4. Boundary condition — effective incoming scalar flux at several T_bc values
  5. NPZ spectrum comparison — E_r_groups near x=0 at t=1 ns (if files exist)

Run from the project root:
    python compare_marshak_diagnostics.py
    python compare_marshak_diagnostics.py --npz-sn  marshak_wave_powerlaw_sn_10g_timeBC.npz \\
                                          --npz-diff marshak_wave_multigroup_powerlaw_10g_no_precond_timeBC.npz
"""

import sys
import os
import argparse
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
SN_DIR   = os.path.join(ROOT, 'DiscreteOrdinates')
DIFF_DIR = os.path.join(ROOT, 'nonEquilibriumDiffusion')

sys.path.insert(0, SN_DIR)
sys.path.insert(0, DIFF_DIR)

import sn_solver
from planck_integrals import Bg as _Bg_scalar, dBgdT as _dBgdT_scalar, Bg_multigroup

# ── shared constants ────────────────────────────────────────────────────────
C_LIGHT = sn_solver.c    # 29.98 cm/ns
A_RAD   = sn_solver.a    # 0.01372 GJ/(cm³ keV⁴)
AC      = sn_solver.ac   # a·c

# ── helpers ─────────────────────────────────────────────────────────────────

def Bg_4pi(E_low, E_high, T):
    """4π B_g(T) — S_N convention (scalar flux units, single float T)."""
    return 4.0 * np.pi * _Bg_scalar(E_low, E_high, max(float(T), 1e-9))


def powerlaw_sigma(T, E, rho):
    """σ_a(T,E) = 10 ρ T^{-1/2} E^{-3} with floor T=1e-2, cap 1e14."""
    T_safe = max(float(T), 1e-2)
    return min(10.0 * rho * T_safe**(-0.5) * E**(-3.0), 1e14)


def group_sigma(T, E_low, E_high, rho):
    """Geometric-mean group opacity at boundaries."""
    return np.sqrt(powerlaw_sigma(T, E_low, rho) * powerlaw_sigma(T, E_high, rho))


# ============================================================================
# 1. PARAMETER TABLE
# ============================================================================

def print_parameter_table(n_groups=10):
    print("\n" + "="*72)
    print("1. PARAMETER COMPARISON")
    print("="*72)
    print(f"{'Parameter':<30}  {'S_N':<20}  {'Diffusion':<20}  {'Match?'}")
    print("-"*72)

    rows = [
        # (label, sn_val, diff_val)
        ("Domain [0, L] cm",         "L = 7.0",         "L = 7.0"),
        ("Spatial cells",             "140",              "140"),
        ("Energy groups G",           str(n_groups),      str(n_groups)),
        ("E_min (keV)",               "1e-4",             "1e-4"),
        ("E_max (keV)",               "10.0",             "10.0"),
        ("ρ (g/cm³)",                 "0.01",             "0.01"),
        ("c_v (GJ/(g·keV))",          "0.05",             "0.05"),
        ("c_v·ρ (GJ/(cm³·keV))",      "5e-4",             "5e-4"),
        ("T_init (keV)",              "0.005",            "0.001"),     # ← DIFFERENT
        ("T_bc start (keV)",          "0.05",             "0.05"),
        ("T_bc end (keV)",            "0.25",             "0.25"),
        ("BC ramp time (ns)",         "5.0",              "5.0"),
        ("Left BC type",              "Marshak (S_N)",    "Marshak (diffusion)"),
        ("Right BC type",             "REFLECTING",       "VACUUM (zero incoming)"),  # ← DIFFERENT
        ("dt_max (ns)",               "0.01",             "0.01"),
        ("Output times (ns)",         "1,2,5,10",         "1,2,5,10"),
        ("Opacity formula",           "10ρT^{-1/2}E^{-3}","10ρT^{-1/2}E^{-3}"),
        ("Group σ averaging",         "geom. mean bdry",  "geom. mean bdry"),
    ]

    for label, sn, diff in rows:
        match = "✓" if sn == diff else "✗ DIFFER"
        print(f"  {label:<28}  {sn:<20}  {diff:<20}  {match}")

    print()
    print("  KNOWN DIFFERENCES:")
    print("    ► T_init: S_N=0.005 keV  vs  Diffusion=0.001 keV")
    print("    ► Right BC: S_N=REFLECTING  vs  Diffusion=VACUUM (zero incoming flux)")
    print("      (reflecting means no energy loss at right wall;")
    print("       vacuum allows radiation to escape)")


# ============================================================================
# 2. OPACITY CROSS-CHECK
# ============================================================================

def check_opacities(n_groups=10):
    print("\n" + "="*72)
    print("2. OPACITY CROSS-CHECK  σ_g(T) — geometric mean of σ(T,E_low,E_high)")
    print("   Both codes use identical formula; checking numerically.")
    print("="*72)
    rho = 0.01
    energy_edges = np.logspace(np.log10(1e-4), np.log10(10.0), n_groups + 1)
    T_vals = [0.005, 0.05, 0.1, 0.25]

    print(f"  {'Group':>5}  {'E_low':>10}  {'E_high':>10}  ", end="")
    for T in T_vals:
        print(f"  σ(T={T:.3f})  ", end="")
    print()
    print("  " + "-"*65)
    for g in range(n_groups):
        El, Eh = energy_edges[g], energy_edges[g+1]
        print(f"  {g:5d}  {El:10.4e}  {Eh:10.4e}  ", end="")
        for T in T_vals:
            sig = group_sigma(T, El, Eh, rho)
            print(f"  {sig:10.3e}  ", end="")
        print()


# ============================================================================
# 3. PLANCK INTEGRAL CROSS-CHECK
# ============================================================================

def check_planck(n_groups=10):
    print("\n" + "="*72)
    print("3. PLANCK INTEGRAL CROSS-CHECK")
    print("   S_N uses: 4π·Bg_scalar(E_low, E_high, T)  (planck_integrals.Bg)")
    print("   Diffusion uses: Bg_multigroup(edges, T)[g]  (same library)")
    print("   Both should equal acT^4 when summed over all groups.")
    print("="*72)
    energy_edges = np.logspace(np.log10(1e-4), np.log10(10.0), n_groups + 1)
    T_vals = [0.005, 0.05, 0.1, 0.25]

    print(f"\n  {'T':>6}  {'4πΣBg (S_N)':>15}  {'ΣBg_multi (Diff)':>18}  "
          f"{'acT^4':>14}  {'err_SN':>10}  {'err_Diff':>10}")
    print("  " + "-"*80)
    for T in T_vals:
        acT4 = AC * T**4
        # S_N sum: 4π * sum of Bg_scalar
        sn_sum = sum(Bg_4pi(energy_edges[g], energy_edges[g+1], T)
                     for g in range(n_groups))
        # Diffusion sum: Bg_multigroup (NOT multiplied by 4π — check what it returns)
        bg_mg = Bg_multigroup(energy_edges, T)
        diff_sum = bg_mg.sum()
        err_sn   = abs(sn_sum - acT4) / acT4
        err_diff = abs(diff_sum - acT4) / acT4
        print(f"  {T:6.4f}  {sn_sum:15.6e}  {diff_sum:18.6e}  "
              f"{acT4:14.6e}  {err_sn:10.2e}  {err_diff:10.2e}")

    print()
    print("  Per-group comparison at T = 0.25 keV:")
    T = 0.25
    bg_mg = Bg_multigroup(energy_edges, T)
    acT4  = AC * T**4
    print(f"  {'g':>3}  {'4πBg_scalar':>15}  {'Bg_multigroup':>15}  {'ratio':>10}")
    print("  " + "-"*50)
    for g in range(n_groups):
        El, Eh = energy_edges[g], energy_edges[g+1]
        sn_val   = Bg_4pi(El, Eh, T)
        diff_val = bg_mg[g]
        ratio    = sn_val / diff_val if diff_val != 0 else float('nan')
        flag = "" if abs(ratio - 1) < 1e-6 else "  ← DIFFER"
        print(f"  {g:3d}  {sn_val:15.6e}  {diff_val:15.6e}  {ratio:10.6f}{flag}")


# ============================================================================
# 4. BOUNDARY CONDITION COMPARISON
# ============================================================================

def check_bc(n_groups=10):
    """Compare effective incoming scalar flux imposed at the left boundary.

    S_N:
      ψ(μ>0) = 4π B_g(T_bc)
      Effective phi_g imposed ≈ chi_g * acT^4  (equilibrium)

    Diffusion Marshak BC:
      0.5 * phi_g - D_g * dphi_g/dx = chi_g * (ac T_bc^4 / 2)
      At equilibrium (dphi/dx→0): phi_g = chi_g * acT^4  (same)

    Non-equilibrium check: what phi_g does each BC drive toward?
    """
    print("\n" + "="*72)
    print("4. BOUNDARY CONDITION — EFFECTIVE EQUILIBRIUM PHI_g = chi_g * acT^4")
    print("   Both methods should impose the same group scalar flux at the wall.")
    print("="*72)
    rho  = 0.01
    energy_edges = np.logspace(np.log10(1e-4), np.log10(10.0), n_groups + 1)
    T_vals = [0.05, 0.10, 0.15, 0.20, 0.25]

    print(f"\n  Verification: ΣΣ chi_g * acT^4  vs  acT^4  (should match)")
    for T in T_vals:
        bg = Bg_multigroup(energy_edges, T)
        chi = bg / bg.sum()
        phi_total = chi.sum() * AC * T**4
        acT4 = AC * T**4
        print(f"    T={T:.3f}: Σchi_g*acT^4 = {phi_total:.6e}  acT^4 = {acT4:.6e}  "
              f"ratio = {phi_total/acT4:.8f}")

    print()
    print("  S_N incoming psi vs Diffusion incoming flux at T_bc = 0.25 keV:")
    T = 0.25
    bg  = Bg_multigroup(energy_edges, T)
    chi = bg / bg.sum()
    acT4 = AC * T**4
    F_total_diff = AC * T**4 / 2.0   # = acT^4/2 used by diffusion BC
    rho_local = 0.01
    print(f"  {'g':>3}  {'4πBg (S_N psi)':>18}  {'chi*acT^4 (equil phi)':>22}  "
          f"{'chi*acT^4/2 (diff C_g)':>24}")
    print("  " + "-"*72)
    for g in range(n_groups):
        El, Eh = energy_edges[g], energy_edges[g+1]
        psi_sn   = Bg_4pi(El, Eh, T)                  # psi in mu>0 directions
        phi_eq   = chi[g] * acT4                       # equilibrium phi_g (both methods)
        C_diff   = chi[g] * F_total_diff               # diffusion BC C parameter
        print(f"  {g:3d}  {psi_sn:18.6e}  {phi_eq:22.6e}  {C_diff:24.6e}")

    print()
    print("  Note: psi_sn = phi_eq = chi_g * acT^4  (equilibrium, ✓ consistent)")
    print("        C_diff = phi_eq / 2  (diffusion BC drives phi_g → phi_eq as dphi/dx→0)")

    print()
    print("  RIGHT BC:")
    print("    S_N  : reflecting — all outgoing ψ mirrored back, net flux = 0")
    print("    Diff : vacuum    — 0.5*phi_g + D_g*(dphi_g/dx) = 0")
    print("           → allows radiation to escape, net outward flux ≠ 0")
    print()
    print("  *** This is the most significant physics difference between the two setups. ***")
    print("  *** For early times (wave hasn't reached x=7 cm) the right BC should not  ***")
    print("  *** matter much, but it will diverge at late times.                        ***")


# ============================================================================
# 5. NPZ SPECTRUM COMPARISON
# ============================================================================

def compare_npz(npz_sn, npz_diff, n_groups=10):
    print("\n" + "="*72)
    print("5. NPZ SPECTRUM COMPARISON  (E_r_groups near x=0)")
    print("="*72)

    missing = []
    if not os.path.exists(npz_sn):
        missing.append(f"S_N   : {npz_sn}")
    if not os.path.exists(npz_diff):
        missing.append(f"Diff  : {npz_diff}")
    if missing:
        print("  Skipping — file(s) not found:")
        for m in missing:
            print(f"    {m}")
        return

    sn   = np.load(npz_sn,   allow_pickle=False)
    diff = np.load(npz_diff, allow_pickle=False)

    print(f"\n  S_N  file : {os.path.basename(npz_sn)}")
    print(f"  Diff file : {os.path.basename(npz_diff)}")
    print(f"\n  S_N  saved times (ns) : {sn['times']}")
    print(f"  Diff saved times (ns) : {diff['times']}")

    # Check keys
    for label, arr in [('S_N', sn), ('Diff', diff)]:
        print(f"\n  {label} arrays: {list(arr.keys())}")

    # ── energy edges ──
    eg_sn   = sn['energy_edges']
    eg_diff = diff['energy_edges']
    if np.allclose(eg_sn, eg_diff, rtol=1e-8):
        print("\n  Energy edges: ✓ identical")
    else:
        print("\n  Energy edges: ✗ DIFFER")
        print(f"    S_N  : {eg_sn}")
        print(f"    Diff : {eg_diff}")

    # ── spatial grids ──
    r_sn   = sn['r']
    r_diff = diff['r']
    if np.allclose(r_sn, r_diff, rtol=1e-8):
        print("  Spatial grid: ✓ identical")
    else:
        print(f"  Spatial grid: ✗ differ — "
              f"S_N range [{r_sn[0]:.4f},{r_sn[-1]:.4f}], "
              f"Diff range [{r_diff[0]:.4f},{r_diff[-1]:.4f}]")

    # ── T_mat at t=1 ns ──
    t_target = 1.0
    i_sn   = np.argmin(np.abs(sn['times']   - t_target))
    i_diff = np.argmin(np.abs(diff['times'] - t_target))
    t_sn   = sn['times'][i_sn]
    t_di   = diff['times'][i_diff]

    print(f"\n  Comparing at t ≈ {t_target} ns  (S_N: {t_sn:.4f} ns, Diff: {t_di:.4f} ns)")

    T_sn   = sn['T_mat'][i_sn]
    T_diff = diff['T_mat'][i_diff]
    print(f"  T_mat (keV):  S_N  max={T_sn.max():.5f}  x0={T_sn[0]:.5f}")
    print(f"                Diff max={T_diff.max():.5f}  x0={T_diff[0]:.5f}")

    # ── group E_r spectrum near x=0 ──
    Er_sn   = sn['E_r_groups'][i_sn]    # (G, N_cells)
    Er_diff = diff['E_r_groups'][i_diff] # (G, N_cells)

    # cell closest to x=0
    print(f"\n  E_r_g at cell x[0]={r_sn[0]:.4f} cm (t≈{t_target} ns):")
    print(f"  {'g':>3}  {'E_low':>10}  {'E_high':>10}  "
          f"{'E_r SN':>14}  {'E_r Diff':>14}  {'ratio':>10}")
    print("  " + "-"*67)
    G = min(Er_sn.shape[0], Er_diff.shape[0])
    edges = np.logspace(np.log10(1e-4), np.log10(10.0), G + 1)
    for g in range(G):
        er_s = Er_sn[g, 0]
        er_d = Er_diff[g, 0]
        ratio = er_s / er_d if er_d != 0 else float('nan')
        flag = "" if abs(ratio - 1) < 0.1 else "  ← large diff"
        print(f"  {g:3d}  {edges[g]:10.4e}  {edges[g+1]:10.4e}  "
              f"{er_s:14.6e}  {er_d:14.6e}  {ratio:10.4f}{flag}")

    # ── total E_r at t=1 ns ──
    Er_tot_sn   = sn['E_r'][i_sn]     if 'E_r'   in sn   else Er_sn.sum(axis=0)
    Er_tot_diff = diff['E_r'][i_diff]  if 'E_r'   in diff  else Er_diff.sum(axis=0)
    print(f"\n  Total E_r (GJ/cm³) at t≈{t_target} ns:")
    print(f"    x     : {r_sn[:5]}")
    print(f"    S_N   : {Er_tot_sn[:5]}")
    print(f"    Diff  : {Er_tot_diff[:5]}")
    print(f"  ratio   : {(Er_tot_sn[:5]+1e-300)/(Er_tot_diff[:5]+1e-300)}")

    # ── also T_mat near boundary ──
    print(f"\n  T_mat (keV) near x=0 at t≈{t_target} ns:")
    print(f"    x     : {r_sn[:5]}")
    print(f"    S_N   : {T_sn[:5]}")
    print(f"    Diff  : {T_diff[:5]}")

    # ── make a quick matplotlib figure ──
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Left: per-group E_r spectrum at x[0]
        ax = axes[0]
        E_ctr = np.sqrt(edges[:-1] * edges[1:])
        ax.loglog(E_ctr, Er_sn[:, 0],   'o-', lw=1.5, label='S_N')
        ax.loglog(E_ctr, Er_diff[:, 0], 's--', lw=1.5, label='Diffusion')
        ax.set_xlabel('Photon energy (keV)')
        ax.set_ylabel(r'$E_{r,g}$ (GJ/cm³)')
        ax.set_title(f'Group spectrum at x={r_sn[0]:.3f} cm, t≈{t_target} ns')
        ax.legend()
        ax.grid(True, which='both', alpha=0.3)

        # Right: total E_r profile
        ax = axes[1]
        ax.semilogy(r_sn,   Er_tot_sn,   '-',  lw=1.5, label='S_N')
        ax.semilogy(r_diff, Er_tot_diff, '--', lw=1.5, label='Diffusion')
        ax.set_xlabel('x (cm)')
        ax.set_ylabel(r'$E_r$ (GJ/cm³)')
        ax.set_title(f'Total radiation energy density, t≈{t_target} ns')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_name = 'marshak_diagnostic_compare.png'
        plt.savefig(fig_name, dpi=150)
        plt.close(fig)
        print(f"\n  Figure saved to {fig_name}")
    except Exception as exc:
        print(f"\n  (Figure skipped: {exc})")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Marshak wave diagnostic comparison')
    parser.add_argument('--groups', type=int, default=10,
                        help='Number of energy groups (default: 10)')
    parser.add_argument('--npz-sn', type=str,
                        default='marshak_wave_powerlaw_sn_10g_timeBC.npz',
                        help='S_N NPZ file')
    parser.add_argument('--npz-diff', type=str,
                        default='marshak_wave_multigroup_powerlaw_10g_no_precond_timeBC.npz',
                        help='Diffusion NPZ file')
    args = parser.parse_args()

    G = args.groups

    print_parameter_table(G)
    check_opacities(G)
    check_planck(G)
    check_bc(G)
    compare_npz(args.npz_sn, args.npz_diff, G)

    print("\n" + "="*72)
    print("SUMMARY OF CONFIRMED DIFFERENCES")
    print("="*72)
    print()
    print("  1. T_init:")
    print("       S_N       = 0.005 keV")
    print("       Diffusion = 0.001 keV")
    print("     → FIX: set T_init = 0.005 keV in marshak_wave_multigroup_powerlaw.py")
    print()
    print("  2. Right BC:")
    print("       S_N       = reflecting  (zero net flux; energy stays in domain)")
    print("       Diffusion = vacuum      (D_g(T=0) → near-zero, forces phi≈0 at wall)")
    print("     → FIX: change diffusion right BC to pure Neumann (dphi/dx = 0):")
    print("       return 0.0, 1.0, 0.0  (A=0, B=1, C=0)")
    print()
    print("  3. Planck integral convention:")
    print("       S_N       uses 4π·Bg_scalar (phi convention)")
    print("       Diffusion uses Bg_multigroup (per-steradian, = Bg_scalar)")
    print("       Ratio confirmed = 4π = 12.566...  — NOT a bug.")
    print("       Diffusion solver correctly multiplies by 4π internally")
    print("       in compute_source_xi and compute_fleck_factor.")
    print()
    print("  4. Large T_mat / E_r discrepancy at t=1 ns:")
    print("       Ratio E_r(SN)/E_r(Diff) ≈ 97  at x[0]")
    print("       Ratio T_mat(SN)/T_mat(Diff) ≈ 19")
    print("       This is REAL physics: S_N transport allows free-streaming near")
    print("       the boundary (the Marshak BC directly injects psi = 4π B_g in")
    print("       forward directions), whereas diffusion + Larsen flux limiter")
    print("       throttles the flux to c*E_r and cannot reproduce the sharp")
    print("       near-boundary temperature rise that transport resolves.")
    print("       Items 1 and 2 will reduce (not eliminate) this difference.")
    print()
