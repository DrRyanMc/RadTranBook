#!/usr/bin/env python3
"""
Su-Olson Picket Fence — figure generation script.

Loads the imc_results.npz written by test_su_olson_picket_fence.py and
produces three publication-quality comparison figures (U1, U2, V) against
the reference data from Su & Olson (1997).

Figure summary
--------------
  1. U1 — Group 1 (thin) radiation energy density vs. position and time.
  2. U2 — Group 2 (thick) radiation energy density vs. position and time.
  3. V  — Material energy density vs. position and time.

Usage
-----
  python visualization/plot_su_olson_picket_fence.py
  python visualization/plot_su_olson_picket_fence.py --results path/to/imc_results.npz
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(os.path.dirname(_here))  # visualization -> MG_IMC -> RadTranBook
if _root not in sys.path:
    sys.path.insert(0, _root)

from utils.plotfuncs import font, show  # noqa: F401  (font sets global style)

# ---------------------------------------------------------------------------
_default_results = os.path.normpath(
    os.path.join(_here, "..", "results", "su_olson_picket_fence", "imc_results.npz")
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot Su-Olson picket fence IMC comparison figures"
    )
    p.add_argument(
        "--results",
        default=_default_results,
        metavar="FILE",
        help=f"Path to imc_results.npz  (default: {_default_results})",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.results):
        sys.exit(
            f"Results file not found: {args.results}\n"
            "Run problems/test_su_olson_picket_fence.py first to generate it."
        )

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    data = np.load(args.results, allow_pickle=False)

    x_centers = data["x_centers"]          # (nx,)
    tau_values = data["tau_values"]         # (n_tau,)
    U1_arr     = data["U1"]                 # (n_tau, nx)
    U2_arr     = data["U2"]                 # (n_tau, nx)
    V_arr      = data["V"]                  # (n_tau, nx)
    ref_U1     = data["ref_U1"]             # (n_ref, 5)
    ref_U2     = data["ref_U2"]
    ref_V      = data["ref_V"]
    source_region = float(data["source_region"])
    sigma_1       = float(data["sigma_1"])
    sigma_2       = float(data["sigma_2"])

    # Build a dict keyed by (Python) float tau for convenient lookup
    fields = {
        float(tau): {"U1": U1_arr[i], "U2": U2_arr[i], "V": V_arr[i]}
        for i, tau in enumerate(tau_values)
    }

    # Ordered tau list used for colour/marker indexing
    tau_list   = [0.1, 0.3, 1.0, 3.0]
    colors_tau = ["blue", "green", "red", "purple"]

    plot_specs = [
        (
            "U1", ref_U1,
            r"$U_1$ (Group 1 Radiation)",
            fr"Group 1: $\sigma_1 = {sigma_1:.4f}$ cm$^{{-1}}$ (thin)",
            (1e-3, 1e0),
            "test_su_olson_picket_fence_imc_U1.pdf",
        ),
        (
            "U2", ref_U2,
            r"$U_2$ (Group 2 Radiation)",
            fr"Group 2: $\sigma_2 = {sigma_2:.4f}$ cm$^{{-1}}$ (thick)",
            (1e-4, 1e0),
            "test_su_olson_picket_fence_imc_U2.pdf",
        ),
        (
            "V", ref_V,
            r"$V$ (Material Energy)",
            "Material Energy Density",
            (1e-5, 1e0),
            "test_su_olson_picket_fence_imc_V.pdf",
        ),
    ]

    # ------------------------------------------------------------------
    # Generate figures
    # ------------------------------------------------------------------
    for quantity_key, ref_data, y_label, title, y_limits, output_file in plot_specs:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))

        for idx, tau_val in enumerate(tau_list):
            if tau_val not in fields:
                continue

            color = colors_tau[idx]
            y     = fields[tau_val][quantity_key]

            # IMC line
            mask = (x_centers >= 0.05) & (x_centers <= 5.0) & (y > 0.0)
            ax.plot(
                x_centers[mask], y[mask],
                color=color, linestyle="-", linewidth=2.5, alpha=0.8,
                label=fr"IMC $\tau={tau_val:.1f}$",
            )

            # Reference markers (column order: x, τ=0.1, 0.3, 1.0, 3.0)
            tau_col = tau_list.index(tau_val) + 1
            ref_mask = (
                ~np.isnan(ref_data[:, tau_col])
                & (ref_data[:, 0] >= 0.05)
                & (ref_data[:, tau_col] > 0.0)
            )
            ax.plot(
                ref_data[ref_mask, 0], ref_data[ref_mask, tau_col],
                marker="s", markerfacecolor=color, markeredgecolor="black",
                markersize=6, markeredgewidth=1.5, linestyle="", alpha=0.8,
                label=fr"Ref $\tau={tau_val:.1f}$",
            )

        ax.axvline(x=source_region, color="gray", linestyle="--", alpha=0.3, linewidth=1)
        ax.set_xlabel("Position (mean-free paths)", fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, which="both")
        ax.set_xlim([0.05, 5.0])
        ax.set_ylim(y_limits)

        fig.tight_layout()
        show(output_file, close_after=True)


if __name__ == "__main__":
    main()
