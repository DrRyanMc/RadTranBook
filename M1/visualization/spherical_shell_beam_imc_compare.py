#!/usr/bin/env python3
"""Spherical-shell incoming-beam test with IMC, overlaid with M1 outputs.

This script runs IMC1D in spherical geometry on the same shell setup used by
spherical_incoming_beam_compare.py, with an option to force a collimated
boundary beam (mu = -1 at the outer boundary).

It then overlays IMC profiles with the precomputed M1 dataset stored in
spherical_shell_beam_data.npz.
"""

import os
import sys
import argparse
import random

import numpy as np
import matplotlib.pyplot as plt

_here = os.path.dirname(os.path.abspath(__file__))
_m1_root = os.path.dirname(_here)
_project_root = os.path.dirname(_m1_root)
_imc_root = os.path.join(_project_root, "IMC")

sys.path.insert(0, _project_root)
sys.path.insert(0, _imc_root)

import IMC1D as imc
from utils.plotfuncs import show


def main():
    parser = argparse.ArgumentParser(description="Run spherical-shell IMC beam test and compare to M1.")
    parser.add_argument("--load", action="store_true", help="Load cached IMC data instead of re-running.")
    parser.add_argument("--collimated", action="store_true", default=True,
                        help="Use collimated boundary beam (mu=-1) at outer boundary.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--n-cells", type=int, default=200)
    parser.add_argument("--r-inner", type=float, default=0.2)
    parser.add_argument("--r-outer", type=float, default=1.0)
    parser.add_argument("--sigma-a", type=float, default=1.0)
    parser.add_argument("--cv", type=float, default=10.0)
    parser.add_argument("--t-bc", type=float, default=1.0)
    parser.add_argument("--t-init", type=float, default=1.0e-3)
    parser.add_argument("--n-target", type=int, default=30000,
                        help="IMC internal emission target particles per step.")
    parser.add_argument("--n-boundary", type=int, default=15000,
                        help="IMC boundary source particles per step.")
    parser.add_argument("--n-max", type=int, default=120000,
                        help="IMC max census particles after combing.")
    parser.add_argument("--output-base", type=str, default="spherical_shell_beam_imc")
    parser.add_argument("--ref-data", type=str, default="spherical_shell_beam_data.npz",
                        help="Reference M1 NPZ from spherical_incoming_beam_compare.py")
    args = parser.parse_args()

    # Shared setup
    r_inner = float(args.r_inner)
    r_outer = float(args.r_outer)
    n_cells = int(args.n_cells)
    sigma_a = float(args.sigma_a)
    cv_val = float(args.cv)
    t_bc = float(args.t_bc)
    t_init = float(args.t_init)

    c_light = 29.98
    a_rad = 0.01372
    E_bc = a_rad * t_bc ** 4

    dr = (r_outer - r_inner) / n_cells
    r_edges = np.linspace(r_inner, r_outer, n_cells + 1)
    mesh = np.column_stack([r_edges[:-1], r_edges[1:]])
    r_mid = 0.5 * (mesh[:, 0] + mesh[:, 1])
    volumes = (4.0 / 3.0) * np.pi * (mesh[:, 1]**3 - mesh[:, 0]**3)

    mean_free_time = 1.0 / (c_light * sigma_a)
    dt = 0.01 * mean_free_time
    output_tau = [0.25, 0.5, 1.0, 2.0, 4.0]
    output_times_ns = [tau * mean_free_time for tau in output_tau]
    final_time = output_times_ns[-1]

    npz_path = f"{args.output_base}_data.npz"

    # Load M1 reference for overlays
    ref = np.load(args.ref_data)
    m1 = {}
    for name in ("Kershaw", "Levermore", "Minerbo poly"):
        key = name.replace(" ", "_")
        m1[name] = {tau: ref[f"Er_{key}_tau{tau}"] for tau in output_tau}

    if args.load:
        data = np.load(npz_path)
        imc_by_tau = {tau: data[f"Er_imc_tau{tau}"] for tau in output_tau}
        boundary_cum_by_tau = {
            tau: float(data[f"boundary_cum_tau{tau}"])
            for tau in output_tau
            if f"boundary_cum_tau{tau}" in data
        }
    else:
        np.random.seed(args.seed)
        random.seed(args.seed)

        sigma_a_func = lambda T: np.full_like(T, sigma_a)
        eos = lambda T: cv_val * T
        inv_eos = lambda u: u / cv_val
        cv = lambda T: np.full_like(T, cv_val)

        T0 = np.full(n_cells, t_init)
        Tr0 = np.full(n_cells, t_init)
        source = np.zeros(n_cells)

        # Outer boundary drives inward; inner boundary off.
        T_boundary = (0.0, t_bc)

        # Target incoming power matched to M1/SN beam interpretation:
        # F_in = c * E_bc  over area A = 4*pi*R^2.
        area_outer = 4.0 * np.pi * r_outer * r_outer
        target_power = c_light * E_bc * area_outer
        print(f"Target incoming power (matched): {target_power:.6e} GJ/ns")

        # Optional monkey patch for collimated spherical boundary source.
        original_create_boundary_spherical = imc.create_boundary_spherical
        if args.collimated:
            def _collimated_boundary_spherical(N, T, dt_loc, R, outward=True):
                # Power-matched collimated beam: E_in = (c E_bc A) dt.
                total_emission = target_power * dt_loc
                weights = np.zeros(N) + total_emission / max(N, 1)
                if outward:
                    mus = np.ones(N)
                else:
                    mus = -np.ones(N)
                times = np.random.uniform(0.0, dt_loc, N)
                positions = np.zeros(N) + R
                cell_indices = np.zeros(N, dtype=np.int64)
                return weights, mus, times, positions, cell_indices
            imc.create_boundary_spherical = _collimated_boundary_spherical

        try:
            state = imc.init_simulation(
                args.n_target, T0, Tr0, mesh, eos, inv_eos,
                geometry="spherical"
            )

            imc_by_tau = {}
            boundary_cum_by_tau = {}
            boundary_cum = 0.0
            for t_out, tau in zip(output_times_ns, output_tau):
                while state.time < t_out - 1e-14:
                    step_dt = min(dt, t_out - state.time)
                    state, info = imc.step(
                        state,
                        args.n_target,
                        args.n_boundary,
                        0,
                        args.n_max,
                        T_boundary,
                        step_dt,
                        mesh,
                        sigma_a_func,
                        inv_eos,
                        cv,
                        source,
                        reflect=(False, False),
                        geometry="spherical",
                    )
                    boundary_cum += float(info["boundary_emission"])

                # Use scalar-intensity-based Tr from IMC state and map to Er=aT^4
                Er_imc = a_rad * np.maximum(state.radiation_temperature, 0.0) ** 4
                imc_by_tau[tau] = Er_imc.copy()
                boundary_cum_by_tau[tau] = boundary_cum

                expected_cum = target_power * t_out
                ratio = boundary_cum / expected_cum if expected_cum > 0.0 else np.nan
                print(
                    f"tau={tau:.2f}: boundary injected={boundary_cum:.6e} GJ, "
                    f"expected={expected_cum:.6e} GJ, ratio={ratio:.6f}"
                )

            save_dict = {
                "r_mid": r_mid,
                "E_bc": np.array(E_bc),
                "output_tau": np.array(output_tau),
                "dt": np.array(dt),
                "target_power": np.array(target_power),
            }
            for tau in output_tau:
                save_dict[f"Er_imc_tau{tau}"] = imc_by_tau[tau]
                save_dict[f"boundary_cum_tau{tau}"] = np.array(boundary_cum_by_tau[tau])
            np.savez(npz_path, **save_dict)
            print(f"Saved {npz_path}")
        finally:
            if args.collimated:
                imc.create_boundary_spherical = original_create_boundary_spherical

    # Plot overlays (M1 + IMC only)
    colors = {
        "Kershaw": "#2ca02c",
        "Levermore": "#d62728",
        "Minerbo poly": "#9467bd",
        "IMC": "#1f77b4",
    }

    for tau in output_tau:
        fig, ax = plt.subplots(figsize=(8.0, 5.2))

        ax.plot(r_mid, imc_by_tau[tau] / E_bc, color=colors["IMC"], lw=2.2, label="IMC")

        for name in ("Kershaw", "Levermore", "Minerbo poly"):
            ls = "-." if name == "Kershaw" else ("--" if name == "Levermore" else (0, (3, 1, 1, 1)))
            lbl = "Minerbo" if name == "Minerbo poly" else name
            ax.plot(r_mid, m1[name][tau] / E_bc, color=colors[name], ls=ls, lw=2.0, label=lbl)

        yvals = np.concatenate([
            imc_by_tau[tau] / E_bc,
            m1["Kershaw"][tau] / E_bc,
            m1["Levermore"][tau] / E_bc,
            m1["Minerbo poly"][tau] / E_bc,
        ])
        y_pos = yvals[yvals > 0.0]
        ymin = max(1e-10, 0.7 * float(np.min(y_pos))) if y_pos.size else 1e-10
        ymax = 1.4 * float(np.max(yvals)) if np.max(yvals) > 0.0 else 1.0

        ax.set_yscale("log")
        ax.set_ylim([ymin, ymax])
        ax.set_xlim([r_inner, r_outer])
        ax.set_xlabel("$r$ (cm)")
        ax.set_ylabel(r"$E_r / E_{\rm bc}$")
        ax.set_title(rf"Spherical shell beam comparison at $\tau={tau:.2f}$")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=10)

        outname = f"{args.output_base}_tau{tau:.2f}".replace(".", "p") + ".pdf"
        plt.tight_layout()
        show(outname, close_after=True)
        print(f"Saved {outname}")


if __name__ == "__main__":
    main()
