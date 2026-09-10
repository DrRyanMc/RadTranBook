"""2-D M1 hot-spot wave-effects study.

Runs the same Gaussian hot-spot problem as generate_wave_effects_study.py
using the M1Solver2D with several closures, then compares against the
IMC, S16, and PN cached snapshots from that study.

Usage (from M1/doc/ or any directory)
--------------------------------------
  python ../problems/wave_effects_m1.py              # run all closures
  python ../problems/wave_effects_m1.py --load       # reload cached M1 results

Outputs are written to the current working directory.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
_here        = os.path.dirname(os.path.abspath(__file__))   # M1/problems/
_m1_root     = os.path.dirname(_here)                       # M1/
_proj_root   = os.path.dirname(_m1_root)                    # RadTranBook/
_wave_dir    = os.path.join(_proj_root, "SphericalHarmonics",
                             "docs", "wave_effects_hotspot")
_util_dir    = os.path.join(_proj_root, "utils")

for _p in [_m1_root, _proj_root, _wave_dir, _util_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from M1.src.m1_2d import M1Solver2D
from M1.src.m1_1d import (
    closure_p1,
    closure_levermore,
    closure_minerbo_poly,
    C_LIGHT,
    A_RAD,
)
import generate_wave_effects_study as study
from plotfuncs import show

# ── Problem constants (mirror generate_wave_effects_study.py) ─────────────────
LX         = study.LX         # 4.0 cm
LY         = study.LY         # 4.0 cm
IX         = study.IX         # 100
IY         = study.IY         # 100
T_END      = study.T_END      # 0.05 ns
SIGMA      = study.SIGMA      # 1.0 cm⁻¹
CV_VOL     = study.CV_VOL     # 0.1 GJ/(cm³ keV)
T_COLD     = study.T_COLD     # 0.01 keV
T_PEAK     = study.T_PEAK     # 1.0  keV
GAUSS_SIG  = study.GAUSS_SIGMA  # 0.125 cm

# ── CFL-limited time step for M1 ─────────────────────────────────────────────
dx_m1 = LX / IX                                   # 0.04 cm
dt_m1 = 0.9 * dx_m1 / (2.0 * C_LIGHT)            # CFL = 0.9

print(f"M1 dt = {dt_m1:.4e} ns  "
      f"(CFL = {C_LIGHT * dt_m1 * (1/dx_m1 + 1/dx_m1):.2f})")

# ── M1 closures to compare ────────────────────────────────────────────────────
CLOSURES = {
    "M1-Levermore": closure_levermore,
    "M1-Minerbo":   closure_minerbo_poly,
    #"M1-P1":        closure_p1,
}


# ── run_m1 ────────────────────────────────────────────────────────────────────

def run_m1(closure_func, label: str) -> study.MethodResult:
    """Run the hot-spot problem with M1Solver2D and the given closure."""
    print(f"\n-- {label} --")

    solver = M1Solver2D(
        0.0, LX, IX,
        0.0, LY, IY,
        geometry="cartesian",
        sigma_func=lambda T: np.full_like(T, SIGMA),
        EOS=lambda T: CV_VOL * T,
        invEOS=lambda e: e / CV_VOL,
        dt=dt_m1,
        closure_func=closure_func,
        bc_x1_lo="marshak", bc_x1_hi="marshak",
        bc_x2_lo="marshak", bc_x2_hi="marshak",
        # Use 1 iteration (explicit-in-transport): the multi-iteration Jacobi
        # Picard diverges for sharp hot-spot gradients because the (Er,F)
        # coupling loop has spectral radius > 1 when F starts far from its
        # physical value.  One explicit step is stable for CFL ≤ 1.
        max_nonlinear_iter=1,
        nonlinear_tol=1e-8,
    )

    # Initial conditions: Gaussian hot spot
    X, Y = np.meshgrid(solver.x1_c, solver.x2_c, indexing="ij")
    r2 = (X - LX / 2.0) ** 2 + (Y - LY / 2.0) ** 2
    T_init = T_COLD + T_PEAK * np.exp(-r2 / (2.0 * GAUSS_SIG ** 2))
    solver.initialize(T_init=T_init)

    # Fiducial bookkeeping
    fid_idx = study._fid_indices(solver.x1_c, solver.x2_c)
    times  = [0.0]
    Er_now = solver.Er
    fid_tr = {l: [float((np.maximum(Er_now[i, j], 0.0) / A_RAD) ** 0.25)]
              for l, (i, j) in fid_idx.items()}
    fid_tm = {l: [float(solver.T[i, j])]
              for l, (i, j) in fid_idx.items()}
    fid_er = {l: [float(Er_now[i, j])]
              for l, (i, j) in fid_idx.items()}

    # Time stepping
    t = 0.0
    step_num = 0
    snapshot = None

    while t < T_END - 1e-12:
        dt_use = min(dt_m1, T_END - t)
        solver.dt = dt_use
        solver.step()
        t += dt_use
        step_num += 1

        Er_now = solver.Er
        times.append(t)
        for l, (i, j) in fid_idx.items():
            fid_tr[l].append(float((np.maximum(Er_now[i, j], 0.0) / A_RAD) ** 0.25))
            fid_tm[l].append(float(solver.T[i, j]))
            fid_er[l].append(float(Er_now[i, j]))

        if step_num % 20 == 0 or step_num == 1:
            print(f"  step {step_num:4d}  t={t:.4f} ns  "
                  f"max(T_rad)={float(solver.T_rad.max()):.4f} keV")

        if abs(t - T_END) < 1e-12 * max(T_END, dt_m1):
            snapshot = (solver.T_rad.copy(),
                        solver.Er.copy(),
                        solver.T.copy())

    if snapshot is None:
        raise RuntimeError(f"{label}: snapshot not captured at t={T_END}")

    return study._result_from_fields(
        label, snapshot[0], snapshot[1], snapshot[2],
        solver.x1_faces, solver.x2_faces,
        np.array(times), fid_tr, fid_tm, fid_er,
        metadata={"method": label, "dt": dt_m1, "t_end": T_END,
                  "IX": IX, "IY": IY, "CFL": 0.9},
    )


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _load_or_run_m1(label: str, cache_path: str, do_load: bool,
                    closure_func) -> study.MethodResult:
    if do_load and os.path.exists(cache_path):
        print(f"  loading {cache_path}")
        return study.load_npz(cache_path)
    result = run_m1(closure_func, label)
    study.save_npz(result, cache_path)
    return result


# ── Comparison plots ──────────────────────────────────────────────────────────

def plot_Trad_panels(results: list[study.MethodResult], out_prefix: str) -> None:
    """2-D T_rad pcolormesh for each method."""
    n = len(results)
    ncols = 3 if n >= 3 else n
    nrows = (n + ncols - 1) // ncols
    if (n==4):
        ncols = 2
        nrows = 2
    vmax = float(max(r.T_rad.max() for r in results))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.5 * nrows),
                             gridspec_kw={"wspace": 0.06, "hspace": 0.10},
                             squeeze=False)
    theta = np.linspace(0.0, 2.0 * np.pi, 200)

    for k, (ax, res) in enumerate(zip(axes.flat, results)):
        if res.label.startswith("M1-"):
            label = res.label[3:]  # strip "M1-" prefix
        else:
            label = res.label
        # skip P1
        if label == "P1":
            continue
        im = ax.pcolormesh(res.x1_faces, res.x2_faces, res.T_rad.T,
                           shading="flat", cmap="plasma", vmin=0.0, vmax=vmax)
        ax.set_aspect("equal")
        ax.set_xlim(0.0, LX)
        ax.set_ylim(0.0, LY)
        ax.plot(LX / 2 + GAUSS_SIG * np.cos(theta),
                LY / 2 + GAUSS_SIG * np.sin(theta), "w--", lw=0.8, alpha=0.7)
        ax.text(0.03, 0.97, label, transform=ax.transAxes,
                fontsize=10, color="white", va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.45, lw=0))
        ax.set_xlabel("$x$ (cm)", fontsize=10)
        ax.set_ylabel("$y$ (cm)", fontsize=10)

    for ax in axes.flat[len(results):]:
        ax.axis("off")

    cax = fig.add_axes([0.93, 0.12, 0.016, 0.76])
    fig.colorbar(im, cax=cax).set_label(r"$T_r$ (keV)", fontsize=10)

    plt.suptitle(f"Radiation temperature at $t = {T_END*1000:.1f}$ ps",
                 fontsize=12)
    plt.tight_layout(rect=[0, 0, 0.92, 0.96])
    show(f"{out_prefix}.pdf", close_after=True)
    print(f"Saved {out_prefix}.pdf")


def plot_centerline(results: list[study.MethodResult], out_prefix: str) -> None:
    """Centerline T_rad lineout."""
    styles = ["-", "--", "-.", ":", (0, (3,1,1,1)), (0,(5,2))]
    colors = ["#1f77b4","#d62728","#2ca02c","#9467bd","#8c564b","#e377c2"]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for k, res in enumerate(results):
        ax.plot(res.lineout_x, res.centerline_Trad,
                color=colors[k % len(colors)], ls=styles[k % len(styles)],
                lw=2.0, label=res.label)
    ax.set_xlabel("$x$ (cm)", fontsize=13)
    ax.set_ylabel(r"$T_r$ (keV)", fontsize=13)
    ax.set_xlim(0.0, LX)
    ax.set_ylim(bottom=0.0)
    ax.axvline(LX / 2, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Centerline $T_r$ at $t = {T_END*1000:.1f}$ ps", fontsize=11)
    plt.tight_layout()
    show(f"{out_prefix}.pdf", close_after=True)
    print(f"Saved {out_prefix}.pdf")


def plot_diagonal(results: list[study.MethodResult], out_prefix: str) -> None:
    """Diagonal T_rad lineout."""
    styles = ["-", "--", "-.", ":", (0, (3,1,1,1)), (0,(5,2))]
    colors = ["#1f77b4","#d62728","#2ca02c","#9467bd","#8c564b","#e377c2"]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for k, res in enumerate(results):
        ax.plot(res.diagonal_s, res.diagonal_Trad,
                color=colors[k % len(colors)], ls=styles[k % len(styles)],
                lw=2.0, label=res.label)
    ax.set_xlabel("Distance from centre (cm)", fontsize=13)
    ax.set_ylabel(r"$T_r$ (keV)", fontsize=13)
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Diagonal $T_r$ at $t = {T_END*1000:.1f}$ ps", fontsize=11)
    plt.tight_layout()
    show(f"{out_prefix}.pdf", close_after=True)
    print(f"Saved {out_prefix}.pdf")


def plot_center_history(results: list[study.MethodResult], out_prefix: str) -> None:
    """T_rad time history at the hot-spot centre."""
    styles = ["-", "--", "-.", ":", (0, (3,1,1,1)), (0,(5,2))]
    colors = ["#1f77b4","#d62728","#2ca02c","#9467bd","#8c564b","#e377c2"]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for k, res in enumerate(results):
        ax.plot(res.fid_times * 1000, res.fid_Trad["center"],
                color=colors[k % len(colors)], ls=styles[k % len(styles)],
                lw=2.0, label=res.label)
    ax.set_xlabel("Time (ps)", fontsize=13)
    ax.set_ylabel(r"$T_r$ at centre (keV)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title("Hot-spot centre radiation temperature", fontsize=11)
    plt.tight_layout()
    show(f"{out_prefix}.pdf", close_after=True)
    print(f"Saved {out_prefix}.pdf")


def _print_metrics(results: list[study.MethodResult],
                   reference: study.MethodResult) -> list[dict[str, Any]]:
    def _rl2(a, b):
        n = float(np.linalg.norm(a - b))
        d = float(np.linalg.norm(b))
        return n / d if d > 0 else (0.0 if n == 0 else float("inf"))
    rows = []
    print(f"\nRelative errors vs {reference.label} at t={T_END*1000:.1f} ps:")
    print(f"  {'Method':>18s}  {'L2_Trad':>10s}  {'L2_Erad':>10s}  {'L2_Tmat':>10s}")
    for res in results:
        row = {
            "label": res.label,
            "L2_rel_Trad": _rl2(res.T_rad, reference.T_rad),
            "L2_rel_Erad": _rl2(res.E_rad, reference.E_rad),
            "L2_rel_Tmat": _rl2(res.T_mat, reference.T_mat),
        }
        rows.append(row)
        print(f"  {res.label:>18s}  "
              f"{row['L2_rel_Trad']:10.4e}  "
              f"{row['L2_rel_Erad']:10.4e}  "
              f"{row['L2_rel_Tmat']:10.4e}")
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="2-D M1 wave-effects hot-spot study.")
    parser.add_argument("--load", action="store_true",
                        help="Load cached M1 results instead of re-running.")
    args = parser.parse_args()

    here = os.getcwd()

    # ── Run / load M1 closures ──────────────────────────────────────────────
    m1_results: dict[str, study.MethodResult] = {}
    for label, cl_func in CLOSURES.items():
        cache = os.path.join(here, f"m1_{label.replace(' ','_').lower()}.npz")
        m1_results[label] = _load_or_run_m1(label, cache, args.load, cl_func)

    # ── Load IMC / SN / PN reference snapshots ─────────────────────────────
    ref_methods: dict[str, study.MethodResult] = {}
    for key, fname in [("IMC", "imc_snapshot.npz"),
    #                    ("S16", "sn_s16_snapshot.npz"),
                        ("FP7",  "p7_filt_10p0_snapshot.npz")
    #                    ("P3",  "pn3_snapshot.npz")
                        ]:
        path = os.path.join(_wave_dir, fname)
        if os.path.exists(path):
            print(f"Loading {key} from {path}")
            ref_methods[key] = study.load_npz(path)
        else:
            print(f"WARNING: {path} not found — skipping {key}")

    if "IMC" not in ref_methods:
        print("IMC reference not available; run generate_wave_effects_study.py first.")
        return

    # ── Assemble ordered list for plots ────────────────────────────────────
    ordered = ([ref_methods[k] for k in ("IMC", "FP7")
                if k in ref_methods]
               + list(m1_results.values()))

    # ── Generate figures ────────────────────────────────────────────────────
    plot_Trad_panels(ordered, "m1_wave_Trad_panels")
    plot_centerline(ordered, "m1_wave_centerline")
    plot_diagonal(ordered, "m1_wave_diagonal")
    plot_center_history(list(m1_results.values()), "m1_wave_center_history")

    # ── Metrics vs IMC ─────────────────────────────────────────────────────
    reference = ref_methods["IMC"]
    rows = _print_metrics(ordered, reference)

    # Save metric table as CSV
    import csv
    csv_path = os.path.join(here, "m1_wave_metrics.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["label","L2_rel_Trad",
                                                 "L2_rel_Erad","L2_rel_Tmat"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {csv_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
