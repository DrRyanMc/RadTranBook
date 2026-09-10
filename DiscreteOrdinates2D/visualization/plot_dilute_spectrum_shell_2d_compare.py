#!/usr/bin/env python3
import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

"""
Compare 2-D dilute-shell radial reductions against 1-D spherical references.

This script overlays radial profiles from three result sets:
- 2-D Cartesian S_N dilute-shell run (radial reductions from snapshot files),
- 1-D spherical S_N run,
- 1-D IMC run.

For each requested time, it generates a 2x2 panel with:
- radiation temperature T_rad(r),
- material temperature T_mat(r),
- radiation energy density E_rad(r),
- streaming factor |F_rad|/(c E_rad).
"""

import argparse
import os
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_do2d = os.path.dirname(_here)
_root = os.path.dirname(_do2d)
if _root not in sys.path:
    sys.path.insert(0, _root)

from MG_IMC.problems.dilute_spectrum_shell import C_LIGHT


def load_snapshots(results_dir: str) -> List[Dict[str, np.ndarray]]:
    snaps: List[Dict[str, np.ndarray]] = []
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    for fname in sorted(os.listdir(results_dir)):
        if not (fname.startswith("snapshot_t_") and fname.endswith(".npz")):
            continue
        full = os.path.join(results_dir, fname)
        d = dict(np.load(full))
        d["time"] = float(d["time"])
        snaps.append(d)

    snaps.sort(key=lambda s: s["time"])
    if not snaps:
        raise RuntimeError(f"No snapshot_t_*.npz files found in: {results_dir}")
    return snaps


def pick_snapshot(snaps: List[Dict[str, np.ndarray]], target_t: float) -> Tuple[Dict[str, np.ndarray], float]:
    times = np.array([s["time"] for s in snaps], dtype=float)
    i = int(np.argmin(np.abs(times - target_t)))
    chosen = snaps[i]
    return chosen, float(times[i] - target_t)


def latest_subdir(base: str, prefix: str) -> str:
    if not os.path.isdir(base):
        raise FileNotFoundError(f"Base directory not found: {base}")
    cands = []
    for name in os.listdir(base):
        full = os.path.join(base, name)
        if os.path.isdir(full) and name.startswith(prefix):
            cands.append(full)
    if not cands:
        raise RuntimeError(f"No subdirectories in {base} with prefix '{prefix}'")
    cands.sort(key=lambda p: os.path.getmtime(p))
    return cands[-1]


def safe_streaming_factor(F_rad: np.ndarray, E_rad: np.ndarray) -> np.ndarray:
    denom = np.maximum(np.abs(C_LIGHT * E_rad), 1e-300)
    return np.abs(F_rad) / denom


def add_vertical_regions(ax: plt.Axes) -> None:
    # Import locally to avoid hard dependency unless plotting is executed.
    from MG_IMC.problems.dilute_spectrum_shell import R_S, R_1, R_2

    ax.axvspan(0.0, R_S, color="#fbe5d6", alpha=0.35, linewidth=0)
    ax.axvspan(R_S, R_1, color="#ddebf7", alpha=0.25, linewidth=0)
    ax.axvspan(R_1, R_2, color="#c6e0b4", alpha=0.25, linewidth=0)


def plot_time_overlay(
    out_dir: str,
    target_t: float,
    s2d: Dict[str, np.ndarray],
    s1d: Dict[str, np.ndarray],
    simc: Dict[str, np.ndarray],
    dt2d: float,
    dt1d: float,
    dtimc: float,
) -> str:
    r2d = np.asarray(s2d["r_centers"], dtype=float)
    r1d = np.asarray(s1d["r_centers"], dtype=float)
    rimc = np.asarray(simc["r_centers"], dtype=float)

    fig, axs = plt.subplots(2, 2, figsize=(11.0, 8.0), sharex=False)
    ax_tr = axs[0, 0]
    ax_tm = axs[0, 1]
    ax_er = axs[1, 0]
    ax_chi = axs[1, 1]

    for ax in [ax_tr, ax_tm, ax_er, ax_chi]:
        add_vertical_regions(ax)

    # T_rad
    ax_tr.plot(r2d, s2d["T_rad"], lw=2.0, label=f"2D S_N (dt={dt2d:+.3e} ns)")
    ax_tr.plot(r1d, s1d["T_rad"], lw=1.8, ls="--", label=f"1D S_N (dt={dt1d:+.3e} ns)")
    ax_tr.plot(rimc, simc["T_rad"], lw=1.8, ls=":", label=f"IMC (dt={dtimc:+.3e} ns)")
    ax_tr.set_ylabel("T_rad [keV]")
    ax_tr.set_title("Radiation Temperature")
    ax_tr.grid(True, alpha=0.25)
    ax_tr.legend(fontsize=8)

    # T_mat
    ax_tm.plot(r2d, s2d["T_mat"], lw=2.0)
    ax_tm.plot(r1d, s1d["T_mat"], lw=1.8, ls="--")
    ax_tm.plot(rimc, simc["T_mat"], lw=1.8, ls=":")
    ax_tm.set_ylabel("T_mat [keV]")
    ax_tm.set_title("Material Temperature")
    ax_tm.grid(True, alpha=0.25)

    # E_rad
    er2d = np.maximum(np.asarray(s2d["E_rad"], dtype=float), 1e-300)
    er1d = np.maximum(np.asarray(s1d["E_rad"], dtype=float), 1e-300)
    erimc = np.maximum(np.asarray(simc["E_rad"], dtype=float), 1e-300)
    ax_er.semilogy(r2d, er2d, lw=2.0)
    ax_er.semilogy(r1d, er1d, lw=1.8, ls="--")
    ax_er.semilogy(rimc, erimc, lw=1.8, ls=":")
    ax_er.set_ylabel("E_rad")
    ax_er.set_xlabel("r [cm]")
    ax_er.set_title("Radiation Energy Density")
    ax_er.grid(True, which="both", alpha=0.25)

    # Streaming factor |F| / (cE)
    chi2d = safe_streaming_factor(np.asarray(s2d["F_rad"]), np.asarray(s2d["E_rad"]))
    chi1d = safe_streaming_factor(np.asarray(s1d["F_rad"]), np.asarray(s1d["E_rad"]))
    chiimc = safe_streaming_factor(np.asarray(simc["F_rad"]), np.asarray(simc["E_rad"]))
    ax_chi.plot(r2d, chi2d, lw=2.0)
    ax_chi.plot(r1d, chi1d, lw=1.8, ls="--")
    ax_chi.plot(rimc, chiimc, lw=1.8, ls=":")
    ax_chi.set_ylim(0.0, 1.05)
    ax_chi.set_ylabel("|F| / (c E)")
    ax_chi.set_xlabel("r [cm]")
    ax_chi.set_title("Streaming Factor")
    ax_chi.grid(True, alpha=0.25)

    fig.suptitle(
        f"Dilute Spectrum Shell Comparison: target t={target_t:.5f} ns | "
        f"used (2D,1D,IMC)=({s2d['time']:.5f},{s1d['time']:.5f},{simc['time']:.5f}) ns",
        fontsize=11,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])

    out_png = os.path.join(out_dir, f"compare_radial_t_{target_t:.5f}ns.png")
    fig.savefig(out_png, dpi=220)
    plt.close(fig)
    return out_png


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dir2d",
        type=str,
        default=None,
        help="2-D results directory. Default: newest under results/dilute_spectrum_shell_2d/sn2d_*",
    )
    p.add_argument(
        "--dir1d",
        type=str,
        default=None,
        help="1-D S_N results directory. Default: results/dilute_spectrum_shell/sn_32g_s8",
    )
    p.add_argument(
        "--dirimc",
        type=str,
        default=None,
        help="IMC results directory. Default: results/dilute_spectrum_shell/imc_32g_standard",
    )
    p.add_argument(
        "--times",
        type=float,
        nargs="+",
        default=None,
        help="Target times [ns] to compare; nearest available snapshots are used. "
        "Default: up to first three nonzero 2-D snapshot times.",
    )
    p.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Output directory for figures. Default: <dir2d>/comparison_against_1d",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    base_2d = os.path.join(_root, "results", "dilute_spectrum_shell_2d")
    default_2d = latest_subdir(base_2d, "sn2d_")
    default_1d = os.path.join(_root, "results", "dilute_spectrum_shell", "sn_32g_s8")
    default_imc = os.path.join(_root, "results", "dilute_spectrum_shell", "imc_32g_standard")

    dir2d = args.dir2d if args.dir2d is not None else default_2d
    dir1d = args.dir1d if args.dir1d is not None else default_1d
    dirimc = args.dirimc if args.dirimc is not None else default_imc

    outdir = args.outdir if args.outdir is not None else os.path.join(dir2d, "comparison_against_1d")
    os.makedirs(outdir, exist_ok=True)

    snaps2d = load_snapshots(dir2d)
    snaps1d = load_snapshots(dir1d)
    snapsimc = load_snapshots(dirimc)

    if args.times is None:
        t2d = [float(s["time"]) for s in snaps2d]
        nz = [t for t in t2d if t > 0.0]
        cmp_times = nz[:3] if nz else t2d[:1]
    else:
        cmp_times = args.times

    print("=" * 72)
    print("Dilute-shell 2D vs 1D comparison")
    print(f"  2D:   {dir2d}")
    print(f"  1D:   {dir1d}")
    print(f"  IMC:  {dirimc}")
    print(f"  Out:  {outdir}")
    print("=" * 72)

    for t in cmp_times:
        s2d, dt2d = pick_snapshot(snaps2d, t)
        s1d, dt1d = pick_snapshot(snaps1d, t)
        simc, dtimc = pick_snapshot(snapsimc, t)

        out_png = plot_time_overlay(outdir, t, s2d, s1d, simc, dt2d, dt1d, dtimc)
        print(
            f"Saved {out_png} | "
            f"used times: 2D={s2d['time']:.5f} (dt={dt2d:+.3e}), "
            f"1D={s1d['time']:.5f} (dt={dt1d:+.3e}), "
            f"IMC={simc['time']:.5f} (dt={dtimc:+.3e}) ns"
        )


if __name__ == "__main__":
    main()
