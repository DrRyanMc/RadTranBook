#!/usr/bin/env python3
"""
Dilute Spectrum Shell — figure generation script.

Loads the snapshot .npz files written by run_dilute_spectrum_shell.py and
produces six publication-quality figures.

Figure summary
--------------
  1. Problem schematic — radial cartoon with labelled regions.
  2. Temperature profiles — T_r(r), T_mat(r) and free-streaming T_r at t ≈ 1 ns.
  3. Normalised group spectra — at selected radii plus a Planck reference.
  4. Radiation energy density log–log — E_r(r) vs. r⁻² reference slope.
  5. Material temperature in and near the shell at multiple times.
  6. Streaming parameter |F_r|/(c E_r) — total and selected groups.
  7. Spectral map — 2-D pcolormesh of E_g/ΣE_g(r) with T_r and T_c below.

Usage
-----
  python plot_dilute_spectrum_shell.py                    # default standard/32g
  python plot_dilute_spectrum_shell.py --G 64 --mode publication
  python plot_dilute_spectrum_shell.py --results_dir results/dilute_spectrum_shell/imc_32g_standard
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm
import numpy as np

# ---------------------------------------------------------------------------
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(os.path.dirname(_here))  # visualization -> MG_IMC -> RadTranBook
if _root not in sys.path:
    sys.path.insert(0, _root)

from utils.plotfuncs import font_imc, show
font = font_imc  # alias for brevity
# Import analytic reference functions from problem definition
from MG_IMC.problems.dilute_spectrum_shell import (
    R_S, R_1, R_2, R_OUT, T_S, C_LIGHT, A_RAD,
    N_GROUPS_DEFAULT, DUMP_TIMES,
    free_streaming_Tr, free_streaming_Er,
)

# Try to use the planck_integrals library for colour-temperature fits.
try:
    from planck_integrals import Bg_multigroup
    _PLANCK_AVAILABLE = True
except ImportError:
    _PLANCK_AVAILABLE = False


# ===========================================================================
# Loading helpers
# ===========================================================================

def load_snapshots(results_dir):
    """Return a time-sorted list of snapshot dicts from *results_dir*."""
    snapshots = []
    for fname in sorted(os.listdir(results_dir)):
        if fname.startswith("snapshot_t_") and fname.endswith(".npz"):
            full = os.path.join(results_dir, fname)
            d = dict(np.load(full))
            # Ensure scalar time is a Python float
            d["time"] = float(d["time"])
            snapshots.append(d)
    snapshots.sort(key=lambda s: s["time"])
    return snapshots


def pick_snapshot(snapshots, target_t, tol=0.02*100):
    """Return the snapshot whose time is closest to *target_t*.

    Raises RuntimeError if no snapshot is within *tol* ns.
    """
    if not snapshots:
        raise RuntimeError("No snapshots found.")
    idx  = np.argmin([abs(s["time"] - target_t) for s in snapshots])
    best = snapshots[idx]
    if abs(best["time"] - target_t) > tol:
        available = [f"{s['time']:.3f}" for s in snapshots]
        raise RuntimeError(
            f"No snapshot near t = {target_t} ns (tolerance {tol} ns).  "
            f"Available: {available}"
        )
    return best


# ===========================================================================
# Colour-temperature fitting
# ===========================================================================

def _planck_group_integral_fallback(E_low, E_high, T):
    """Planck group integral using 60-point quadrature (fallback)."""
    if T <= 0:
        return 0.0
    E = np.linspace(E_low, E_high, 60)
    B = (2.0 * E**3 / C_LIGHT**2) / (np.exp(np.clip(E / T, 0, 500)) - 1 + 1e-300)
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(B, E))
    return float(np.trapz(B, E))


def _Bg_all(energy_edges, T):
    """Return array of Planck group integrals at temperature T."""
    if _PLANCK_AVAILABLE:
        return Bg_multigroup(energy_edges, T)
    n = len(energy_edges) - 1
    return np.array([_planck_group_integral_fallback(energy_edges[g], energy_edges[g + 1], T)
                     for g in range(n)])


def _peak_nu_from_spec(spec, nu_c):
    """Return the parabolic-interpolation peak frequency of a spectral density.

    Operates in log–log space.  Falls back to the geometric-mean of the peak
    group if the neighbours are missing or the parabola opens upward.

    Parameters
    ----------
    spec : (n_groups,)  — spectral energy density  E_g / ΔE_g  (any units)
    nu_c : (n_groups,)  — group-centre frequencies (keV)
    """
    if np.all(spec <= 0):
        return np.nan
    i = int(np.argmax(spec))
    nu_peak = nu_c[i]
    if 0 < i < len(spec) - 1 and spec[i - 1] > 0 and spec[i + 1] > 0:
        xl, yl = np.log(nu_c[i - 1]), np.log(spec[i - 1])
        xm, ym = np.log(nu_c[i]),     np.log(spec[i])
        xr, yr = np.log(nu_c[i + 1]), np.log(spec[i + 1])
        denom = (xl - xm) * (xl - xr) * (xm - xr)
        a = (xr * (ym - yl) + xm * (yl - yr) + xl * (yr - ym)) / denom
        b = (xr**2 * (yl - ym) + xm**2 * (yr - yl) + xl**2 * (ym - yr)) / denom
        if a < 0:
            x_peak = -b / (2.0 * a)
            if xl <= x_peak <= xr:
                nu_peak = np.exp(x_peak)
    return float(nu_peak)


# Cache: energy_edges tuple → (T_arr, nu_peak_planck_arr)
_PLANCK_PEAK_CACHE: dict = {}


def _get_planck_peak_lookup(energy_edges, T_min=0.05, T_max=15.0, n_pts=300):
    """Build (or retrieve cached) lookup table of  T → ν_peak(B_g/ΔE_g).

    This applies *exactly* the same parabolic peak-finder to the group-averaged
    Planck spectrum as is used for the simulation spectrum, so that inverting the
    lookup gives an unbiased colour temperature on any coarse grid.
    """
    key = tuple(energy_edges)
    if key not in _PLANCK_PEAK_CACHE:
        dE   = energy_edges[1:] - energy_edges[:-1]
        nu_c = np.sqrt(energy_edges[:-1] * energy_edges[1:])
        T_arr = np.logspace(np.log10(T_min), np.log10(T_max), n_pts)
        nu_peaks = np.array([
            _peak_nu_from_spec(_Bg_all(energy_edges, T)
                               / np.where(dE > 0, dE, 1e-300), nu_c)
            for T in T_arr
        ])
        _PLANCK_PEAK_CACHE[key] = (T_arr, nu_peaks)
    return _PLANCK_PEAK_CACHE[key]


def fit_color_temperature(E_g, energy_edges, T_min=0.05, T_max=15.0):
    """Estimate colour temperature by matching the discrete-grid spectral peak.

    Finds T_c such that the group-averaged Planck spectrum B_g(T_c)/ΔE_g,
    processed with the *same* parabolic interpolation, has the same peak
    frequency as E_g/ΔE_g.  This removes the systematic downward bias that
    arises on coarse group grids when Wien's law (ν_peak / 2.821) is applied
    to the discrete peak directly.

    Parameters
    ----------
    E_g : (n_groups,)  — group energy densities (arbitrary units per group)
    energy_edges : (n_groups+1,)
    T_min, T_max : float — clipping bounds (keV)

    Returns
    -------
    T_c : float — colour temperature (keV), or NaN on failure
    """
    # Gray (1-group) case: colour temperature equals radiation temperature.
    # The spectral-peak fitting below is meaningless with a single bin.
    if len(energy_edges) == 2:
        E_tot = float(E_g[0])
        if E_tot <= 0:
            return np.nan
        return float(np.clip((E_tot / A_RAD) ** 0.25, T_min, T_max))

    dE   = energy_edges[1:] - energy_edges[:-1]
    spec = E_g / np.where(dE > 0, dE, 1e-300)

    if np.all(spec <= 0):
        return np.nan

    nu_c        = np.sqrt(energy_edges[:-1] * energy_edges[1:])
    nu_peak_sim = _peak_nu_from_spec(spec, nu_c)

    if not np.isfinite(nu_peak_sim):
        return np.nan

    # Invert the coarse-grid Planck peak-vs-T curve
    T_arr, nu_peak_B = _get_planck_peak_lookup(energy_edges, T_min, T_max)

    if nu_peak_sim <= nu_peak_B[0]:
        return float(T_min)
    if nu_peak_sim >= nu_peak_B[-1]:
        return float(T_max)
    T_c = float(np.interp(nu_peak_sim, nu_peak_B, T_arr))
    return float(np.clip(T_c, T_min, T_max))


# ===========================================================================
# Method-label helper
# ===========================================================================

def _infer_method_label(results_dir):
    """Return a short human-readable label inferred from the results-dir name."""
    tag = os.path.basename(os.path.normpath(results_dir))
    if tag.startswith("imc_gray"):
        return "Gray IMC"
    if tag.startswith("imc"):
        return "MG IMC"
    if "mg_fl" in tag or "gray_fl" in tag:
        prefix = "MG Diff+FL" if "mg" in tag else "Gray Diff+FL"
        parts = tag.split("_")
        ng = next((p.rstrip("g") for p in parts if p.endswith("g") and p[:-1].isdigit()), None)
        return f"{prefix} ({ng}g)" if ng else prefix
    if tag.startswith("mg"):
        parts = tag.split("_")
        ng = next((p.rstrip("g") for p in parts if p.endswith("g") and p[:-1].isdigit()), None)
        return f"MG Diffusion ({ng}g)" if ng else "MG Diffusion"
    if tag.startswith("gray"):
        return "Gray Diffusion"
    return tag  # fallback: use raw tag


# ===========================================================================
# Figure 1 — Problem schematic
# ===========================================================================

def fig_schematic(out_dir):
    fig, ax = plt.subplots(figsize=(7, 2.8))

    regions = [
        (R_S,  R_1,   "#e8f4f8",  "cavity\n($\\rho = 10^{-8} g/cm^3$)","center"),
        (R_1,  R_2,   "#7fa8c9",  " shell \n($\\rho = 2 g/cm^3$)  ","right"),
        (R_2,  R_OUT, "#e8f4f8",  "","center"),
    ]
    for r0, r1, col, lbl, anc in regions:
        ax.axvspan(r0, r1, color=col, alpha=0.85)
        mid = 0.5 * (r0 + r1)
        if (r1==R_2):
            mid = r0
        ax.text(mid, 0.55, lbl, ha=anc, va="center",
                fontproperties=font, fontsize=10)

    # Source circle at r = R_S
    ax.axvline(R_S, color="firebrick", lw=2.5, ls="-")
    ax.text(R_S, 0.92, f" source\n T = {T_S} keV", ha="left", va="top",
            color="firebrick", fontproperties=font, fontsize=9)

    # Labelled boundaries
    for rv, lbl,anc,shift in [(R_1, "{:.1f} cm ".format(R_1),"right",0), 
                        (R_2, " {:.1f} cm".format(R_2),"left",0),
                    (R_OUT, "  {:.1f} cm".format(R_OUT),"left",.5)]:
        ax.axvline(rv, color="k", lw=1.2, ls="--")
        ax.text(rv, 0.08 + shift, lbl, ha=anc, va="bottom", fontproperties=font,
                fontsize=8, rotation=0)

    # Free-streaming sketch arrow
    ax.annotate("", xy=(R_1 - 0.5, 0.75), xytext=(R_S + 0.3, 0.75),
                arrowprops=dict(arrowstyle="->", color="firebrick", lw=1.5))
    ax.text(0.5 * (R_S + R_1), 0.80, "free-stream", ha="center",
            fontproperties=font, fontsize=9, color="firebrick")

    ax.set_xlim(0.0, R_OUT + 0.5)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("r  (cm)", fontproperties=font)
    ax.set_yticks([])
    #ax.set_title("Dilute spectrum shell — problem geometry", fontproperties=font)
    fig.tight_layout()
    show(os.path.join(out_dir, "fig1_schematic.pdf"))
    plt.savefig(os.path.join(out_dir, "fig1_schematic.png"), dpi=300)
    plt.close()


# ===========================================================================
# Figure 2 — Temperature profiles at t ≈ 1 ns
# ===========================================================================

def fig_temperature_profiles(snapshots, out_dir, method_label="", snap_time=1.0):
    """T_r(r), T_mat(r), T_c(r) compared to analytic free-streaming T_r."""
    snap = pick_snapshot(snapshots, snap_time)
    r  = snap["r_centers"]
    Tr = snap["T_rad"]
    Tm = snap["T_mat"]
    t  = snap["time"]

    # Colour temperature at every cell (requires multigroup snapshot)
    Tc = None
    if "E_rad_by_group" in snap and "energy_edges" in snap:
        Eg    = snap["E_rad_by_group"]
        edges = snap["energy_edges"]
        Tc = np.array([fit_color_temperature(Eg[:, i], edges)
                       for i in range(len(r))])

    # Analytic free-streaming curve in the cavity
    r_cav = r[r < R_1]
    Tr_fs = free_streaming_Tr(r_cav)

    lbl_suffix = f" ({method_label})" if method_label else ""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(r, Tr, color="steelblue", lw=1.8, label=r"$T_r$")# + lbl_suffix)
    ax.semilogy(r, Tm, color="darkorange", lw=1.8, ls="--",
                label=r"$T$")# + lbl_suffix)
    if Tc is not None:
        mask = np.isfinite(Tc)
        ax.semilogy(r[mask], Tc[mask], color="seagreen", lw=1.4, ls="-.",
                    label=r"$T_c$")# + lbl_suffix)
    ax.semilogy(r_cav, Tr_fs, "k:", lw=1.4,
                label=r"$T_S\sqrt{R_S/2r}$") # (free-stream)")

    ax.axvline(R_1, color="0.5", lw=1.0, ls="--")
    ax.axvline(R_2, color="0.5", lw=1.0, ls="--")
    ax.text(R_1 + 0.1, ax.get_ylim()[0] * 2, "shell", fontproperties=font,
            fontsize=8, color="0.2")

    ax.set_xlabel("r  (cm)", fontproperties=font)
    ax.set_ylabel("temperature  (keV)", fontproperties=font)
    #ax.set_title(f"temperature profiles,  t = {t:.0f} ns", fontproperties=font)
    ax.legend(prop=font, ncol=2, loc=(0.1, 0.09), fontsize=8, facecolor="white", edgecolor="none", framealpha=1.0)
    fig.tight_layout()
    show(os.path.join(out_dir, "fig2_temperature_profiles.pdf"))
    plt.savefig(os.path.join(out_dir, "fig2_temperature_profiles.png"), dpi=150)
    plt.close()


# ===========================================================================
# Figure 3 — Normalised spectra at selected radii
# ===========================================================================

def fig_spectra(snapshots, out_dir, snap_time=1.0):
    """Normalised group spectra at r ≈ 2, 10, 20, 24 cm plus Planck at T_S."""
    snap = pick_snapshot(snapshots, snap_time)
    r          = snap["r_centers"]
    Eg         = snap["E_rad_by_group"]   # shape (n_groups, n_cells)
    edges      = snap["energy_edges"]
    t          = snap["time"]
    n_groups   = len(edges) - 1

    # Group centre energies for x-axis
    nu_bar = np.sqrt(edges[:-1] * edges[1:])

    # Select spatial cells closest to target radii
    target_radii = [2.0, 15.0, 24.0,28.0]  # cm
    colors       = ["steelblue", "seagreen", "darkorange", "firebrick"]

    fig, ax = plt.subplots(figsize=(8.5, 4.5))

    for r_tgt, col in zip(target_radii, colors):
        idx = int(np.argmin(np.abs(r - r_tgt)))
        r_actual = float(r[idx])
        spec = Eg[:, idx]
        total = np.sum(spec)
        if total <= 0.0:
            continue    
        spec_norm = spec / total
        T_c = fit_color_temperature(spec, edges)
        lbl = (f"{r_tgt:.0f} cm,"
               + (f"  $T_c$ = {T_c:.1f} keV" if np.isfinite(T_c) else ""))
        ax.loglog(nu_bar, spec_norm, lw=1.6, color=col, label=lbl)

    # Reference Planck spectrum at the source temperature T_S
    B_ref = _Bg_all(edges, T_S)
    B_ref_norm = B_ref / np.sum(B_ref)
    ax.loglog(nu_bar, B_ref_norm, "k--", lw=1.4,
                label=f"Planck  T = {T_S} keV")

    ax.set_xlabel(r"${\nu}$  (keV)", fontproperties=font)
    ax.set_ylabel("normalized group energy", fontproperties=font)
    #ax.set_title(f"group spectra at t = {t:.1f} ns", fontproperties=font)
    ax.legend(prop=font, fontsize=7, ncol=3, facecolor="white", edgecolor="none", framealpha=1.0,
              loc="center", bbox_to_anchor=(0.5, -0.25), borderaxespad=0.0)

    fig.subplots_adjust(right=0.74)
    fig.tight_layout()
    show(os.path.join(out_dir, "fig3_spectra.pdf"))
    plt.savefig(os.path.join(out_dir, "fig3_spectra.png"), dpi=150)
    plt.close()


# ===========================================================================
# Figure 4 — Radiation energy density log–log
# ===========================================================================

def fig_Er_loglog(snapshots, out_dir, method_label="", snap_time=1.0):
    """E_r(r) vs. analytic r⁻² free-streaming slope."""
    snap = pick_snapshot(snapshots, snap_time)
    r  = snap["r_centers"]
    Er = snap["E_rad"]
    t  = snap["time"]

    # Cavity only for the reference lines
    mask_cav = r < R_1
    r_cav    = r[mask_cav]
    Er_fs    = free_streaming_Er(r_cav)

    er_label = method_label if method_label else "simulation"
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.loglog(r, Er, color="steelblue", lw=1.8, label=er_label)
    ax.loglog(r_cav, Er_fs, "k--", lw=1.3, label=r"$(R_S/r)^2$ free-stream")

    # r⁻¹ reference slope (for comparison)
    norm_r1 = Er[mask_cav][0] * r_cav[0]
    ax.loglog(r_cav, norm_r1 / r_cav, color="0.6", lw=1.0, ls=":", label=r"$r^{-1}$ slope")

    ax.axvline(R_1, color="0.5", lw=1.0, ls="--")
    ax.axvline(R_2, color="0.5", lw=1.0, ls="--")

    ax.set_xlabel("r  (cm)", fontproperties=font)
    ax.set_ylabel(r"$E_r$  (GJ / cm³)", fontproperties=font)
    #ax.set_title(f"Radiation energy density,  t = {t:.3f} ns", fontproperties=font)
    ax.legend(prop=font)

    show(os.path.join(out_dir, "fig4_Er_loglog.pdf"))
    plt.savefig(os.path.join(out_dir, "fig4_Er_loglog.png"), dpi=150)
    plt.close()


# ===========================================================================
# Figure 5 — Material temperature near the shell at multiple times
# ===========================================================================

def fig_shell_heating(snapshots, out_dir):
    """T_mat(r) in and near the shell [R_1 – 2 cm, R_2 + 1 cm] at several times."""
    target_times = [0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 4.00]
    cmap  = plt.get_cmap("plasma")
    times = [s["time"] for s in snapshots]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    plotted = 0
    for i, t_tgt in enumerate(target_times):
        try:
            snap = pick_snapshot(snapshots, t_tgt, tol=0.1)
        except RuntimeError:
            continue
        r  = snap["r_centers"]
        Tm = snap["T_mat"]
        t  = snap["time"]

        # Zoom in on shell region
        mask = (r > R_1 - 2.0) & (r < R_2 + 1.5)
        color = cmap(i / max(len(target_times) - 1, 1))
        ax.plot(r[mask], Tm[mask], lw=1.8, color=color, label=f"t = {t:.2f} ns")
        plotted += 1

    ax.axvline(R_1, color="0.5", lw=1.0, ls="--")
    ax.axvline(R_2, color="0.5", lw=1.0, ls="--")
    ax.text(R_1 + 0.05, ax.get_ylim()[0], "inner\nshell", fontsize=8,
            va="bottom", ha="left", fontproperties=font, color="0.4")
    ax.text(R_2 + 0.05, ax.get_ylim()[0], "outer\nshell", fontsize=8,
            va="bottom", ha="left", fontproperties=font, color="0.4")

    ax.set_xlabel("r  (cm)", fontproperties=font)
    ax.set_ylabel(r"$T_{\rm mat}$  (keV)", fontproperties=font)
    ax.set_title("Material temperature near the shell", fontproperties=font)
    if plotted > 0:
        ax.legend(prop=font, fontsize=9, ncol=2)

    show(os.path.join(out_dir, "fig5_shell_heating.pdf"))
    plt.savefig(os.path.join(out_dir, "fig5_shell_heating.png"), dpi=150)
    plt.close()


# ===========================================================================
# Figure 6 — Radiation streaming parameter  |F_r| / (c E_r)
# ===========================================================================

def fig_streaming_parameter(snapshots, out_dir, method_label="", snap_time=1.0):
    """Plot the radiation streaming parameter |F_r|/(c E_r) vs. r.

    Ranges from 0 (isotropic / diffusive) to 1 (free-streaming).  The total
    and four representative groups are shown.  Skipped gracefully if the
    snapshot pre-dates the flux tally (no 'F_rad' key).
    """
    snap = pick_snapshot(snapshots, snap_time)

    if "F_rad" not in snap:
        print("  Skipped — snapshot has no F_rad "
              "(re-run simulation to generate flux data)")
        return

    r  = snap["r_centers"]
    Er = snap["E_rad"]
    Fr = snap["F_rad"]
    t  = snap["time"]

    # Total streaming parameter; guard against empty cells
    cEr     = C_LIGHT * np.maximum(Er, 1e-300)
    f_total = np.abs(Fr) / cEr

    lbl_suffix = f" ({method_label})" if method_label else ""

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(r, f_total, color="steelblue", lw=2.0,
            label=r"$|F_r|/(cE_r)$ total")# + lbl_suffix)

    # Per-group lines: peak of 1 keV Planckian + highest-energy group
    if "F_rad_by_group" in snap and "E_rad_by_group" in snap:
        Fg    = snap["F_rad_by_group"]   # (n_groups, n_cells)
        Eg    = snap["E_rad_by_group"]
        edges = snap["energy_edges"]
        n_g   = len(edges) - 1
        dE    = edges[1:] - edges[:-1]
        # Group whose Planck spectral density B_g/ΔE_g peaks for T = T_S
        g_peak = int(np.argmax(_Bg_all(edges, T_S) / np.where(dE > 0, dE, 1e-300)))
        #g_high is group that contains 10 keV, near the high-energy end of the spectrum
        g_high = np.searchsorted(edges, 10.0) - 1
        group_specs = [
            (g_peak, "seagreen",   f"g={g_peak} (Planck peak)"),
            (g_high, "firebrick",  f"g={g_high}"),
        ]
        for g, col, lbl in group_specs:
            cEg = C_LIGHT * np.maximum(Eg[g], 1e-300)
            fg  = np.abs(Fg[g]) / cEg
            lo, hi = edges[g], edges[g + 1]
            ax.plot(r, fg, lw=1.2, alpha=0.85, color=col,
                    label=f"{lbl}  [{lo:.2g}\u2013{hi:.2g} keV]")

    ax.axhline(1.0, color="0.4", lw=1.0, ls="--", label="free-stream limit")
    ax.axvline(R_1, color="0.5", lw=1.0, ls="--")
    ax.axvline(R_2, color="0.5", lw=1.0, ls="--")
    ax.text(R_1 + 0.1, 1.03, "shell", fontproperties=font, fontsize=9, color="0.4")

    ax.set_xlim(r[0], r[-1])
    ax.set_ylim(0.0, 1.15)
    ax.set_xlabel("r  (cm)", fontproperties=font)
    ax.set_ylabel(r"$|F_r|\,/\,(c\,E_r)$", fontproperties=font)
    #ax.set_title(f"Radiation streaming parameter,  t = {t:.3f} ns",
    #             fontproperties=font)
    ax.legend(prop=font, fontsize=9, edgecolor="none", facecolor="white", framealpha=1.0)

    show(os.path.join(out_dir, "fig6_streaming_parameter.pdf"))
    plt.savefig(os.path.join(out_dir, "fig6_streaming_parameter.png"), dpi=150)
    plt.close()


# ===========================================================================
# Figure 7 — 2-D spectral map + T_r / T_c profiles
# ===========================================================================

def fig_spectrum_map(snapshots, out_dir, snap_time=1.0):
    """2-D spectral map (top) and radiation / colour temperatures (bottom).

    Top panel: pcolormesh of E_g / Σ_g E_g as a function of r (x-axis)
    and photon energy ν (y-axis, log scale) with a logarithmic colour scale.

    Bottom panel: T_r(r) and T_c(r) on a semilogy axis, sharing the x-axis
    with the top panel.
    """
    snap = pick_snapshot(snapshots, snap_time)
    r      = snap["r_centers"]
    Eg     = snap["E_rad_by_group"]   # (n_groups, n_cells)
    edges  = snap["energy_edges"]
    Tr     = snap["T_rad"]
    t      = snap["time"]

    n_groups, n_cells = Eg.shape

    # --- Normalise per cell so each column sums to 1 ---
    col_sum = np.sum(Eg, axis=0, keepdims=True)
    col_sum = np.where(col_sum > 0, col_sum, 1.0)
    spec_norm = Eg / col_sum   # (n_groups, n_cells)

    # --- Spatial cell-edge array for pcolormesh ---
    dr = np.diff(r)
    r_edges = np.concatenate([
        [r[0] - 0.5 * dr[0]],
        0.5 * (r[:-1] + r[1:]),
        [r[-1] + 0.5 * dr[-1]],
    ])

    # --- Colour temperature at every cell ---
    Tc = np.array([fit_color_temperature(Eg[:, i], edges) for i in range(n_cells)])

    # --- Colour-scale limits: 1st-percentile floor → maximum ---
    pos_vals = spec_norm[spec_norm > 0]
    vmin = float(np.nanpercentile(pos_vals, 1)) if len(pos_vals) else 1e-6
    vmax = float(np.nanmax(spec_norm))
    vmin = max(vmin, vmax * 1e-5)  # floor at 1e-5 × peak

    # Use a 2x2 GridSpec so both panels share the exact same plotting column,
    # while the colorbar sits in its own narrow column on the right.
    fig = plt.figure(figsize=(8, 6))
    gs = fig.add_gridspec(
        nrows=2,
        ncols=2,
        height_ratios=[2, 1],
        width_ratios=[1.0, 0.03],
        hspace=0.12,
        wspace=0.05,
    )
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    cax = fig.add_subplot(gs[0, 1])

    # --- Top panel: spectral colormap ---
    pcm = ax0.pcolormesh(
        r_edges, edges, spec_norm,
        norm=LogNorm(vmin=vmin, vmax=vmax),
        cmap="viridis", shading="flat",
    )
    cb = fig.colorbar(pcm, cax=cax)
    cb.set_label(r"$E_g\,/\,\sum_g E_g$", fontproperties=font)
    ax0.set_yscale("log")
    ax0.set_ylabel(r"$\nu$  (keV)", fontproperties=font)
    #ax0.set_title(f"Spectral energy distribution,  t = {t:.3f} ns",
    #              fontproperties=font)
    ax0.axvline(R_1, color="w", lw=0.9, ls="--")
    ax0.axvline(R_2, color="w", lw=0.9, ls="--")
    ax0.set_xlim(r_edges[0], r_edges[-1])

    # --- Bottom panel: T_r, T_c, and cavity free-streaming reference ---
    ax1.semilogy(r, Tr, color="steelblue", lw=1.8, label=r"$T_r$")
    mask = np.isfinite(Tc)
    if np.any(mask):
        ax1.semilogy(r[mask], Tc[mask], color="darkorange", lw=1.8, ls="--",
                     label=r"$T_c$")
    # Reuse the Figure 2 analytic cavity trend on this panel.
    r_cav = r[r < R_1]
    if r_cav.size > 0:
        Tr_fs = free_streaming_Tr(r_cav)
        ax1.semilogy(r_cav, Tr_fs, "k:", lw=1.4,
                     label=r"$T_S\sqrt{R_S/2r}$")
    ax1.axvline(R_1, color="0.5", lw=0.9, ls="--")
    ax1.axvline(R_2, color="0.5", lw=0.9, ls="--")
    ax1.set_xlabel("r  (cm)", fontproperties=font)
    ax1.set_ylabel("temperature  (keV)", fontproperties=font)
    ax1.legend(prop=font, fontsize=9, edgecolor="none", facecolor="white", framealpha=1.0,
               bbox_to_anchor=(0.9, 0.5), loc="center")
    fig.tight_layout()
    show(os.path.join(out_dir, "fig7_spectrum_map.png"))
    #plt.savefig(os.path.join(out_dir, "fig7_spectrum_map.png"), dpi=150)
    plt.close()


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate figures for the dilute-spectrum-shell benchmark."
    )
    p.add_argument("--mode", choices=["quick", "standard", "publication"],
                   default="standard")
    p.add_argument("--G", type=int, default=N_GROUPS_DEFAULT, metavar="N_GROUPS")
    p.add_argument("--results_dir", default=None,
                   help="Override the default results directory.")
    p.add_argument("--snap_time", type=float, default=1.0, metavar="T_NS",
                   help="Snapshot time (ns) used for figures 2, 3, and 4 "
                        "(default: 1.0).")
    return p.parse_args()


def main():
    args = parse_args()

    if args.results_dir:
        results_dir = args.results_dir
    else:
        tag = f"imc_{args.G}g_{args.mode}"
        results_dir = os.path.join("results", "dilute_spectrum_shell", tag)

    if not os.path.isdir(results_dir):
        sys.exit(f"Results directory not found: {results_dir}\n"
                 "Run run_dilute_spectrum_shell.py first.")

    out_dir = results_dir  # save figures alongside the data
    print(f"Loading snapshots from: {results_dir}")
    snapshots = load_snapshots(results_dir)
    if not snapshots:
        sys.exit("No snapshot files found.  Run run_dilute_spectrum_shell.py first.")

    print(f"Found {len(snapshots)} snapshot(s):  "
          + ", ".join(f"{s['time']:.3f}" for s in snapshots) + " ns")
    print()

    print("Figure 1: schematic …")
    fig_schematic(out_dir)

    method_label = _infer_method_label(results_dir)
    print(f"Method label: {method_label}")

    print("Figure 2: temperature profiles …")
    try:
        fig_temperature_profiles(snapshots, out_dir, method_label=method_label,
                                 snap_time=args.snap_time)
    except RuntimeError as e:
        print(f"  Skipped — {e}")

    print("Figure 3: group spectra …")
    try:
        fig_spectra(snapshots, out_dir, snap_time=args.snap_time)
    except RuntimeError as e:
        print(f"  Skipped — {e}")

    print("Figure 4: E_r log–log …")
    try:
        fig_Er_loglog(snapshots, out_dir, method_label=method_label,
                      snap_time=args.snap_time)
    except RuntimeError as e:
        print(f"  Skipped — {e}")

    print("Figure 5: shell heating …")
    fig_shell_heating(snapshots, out_dir)

    print("Figure 6: streaming parameter …")
    try:
        fig_streaming_parameter(snapshots, out_dir, method_label=method_label,
                                snap_time=args.snap_time)
    except RuntimeError as e:
        print(f"  Skipped — {e}")

    print("Figure 7: spectral map …")
    try:
        fig_spectrum_map(snapshots, out_dir, snap_time=args.snap_time)
    except RuntimeError as e:
        print(f"  Skipped — {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
