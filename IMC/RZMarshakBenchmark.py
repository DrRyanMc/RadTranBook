#!/usr/bin/env python3
"""Dedicated cylindrical r-z Marshak-wave benchmark for IMC2D.

This script quantifies how the cylindrical solution approaches the 1D slab
solution as the annulus radius grows (planar limit).
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import MarshakWave2D as mw2d


def run_curvature_scan(r0_values, output_times, dt, dr, nr, nz, Ntarget, Nboundary, Nmax):
    errs = {float(t): [] for t in output_times}

    for r0 in r0_values:
        print(f"\nRunning r-z benchmark at r0={r0:.4f} cm")
        _, _, rz_errs = mw2d.run_rz_planar_limit_check(
            output_times=output_times,
            dt=dt,
            Lz=0.2,
            nz=nz,
            r0=r0,
            dr=dr,
            nr=nr,
            Ntarget=Ntarget,
            Nboundary=Nboundary,
            Nmax=Nmax,
        )
        for t in output_times:
            errs[float(t)].append(rz_errs[float(t)])
            print(f"  t={t:6.3f} ns: rel-L2 error vs 1D = {rz_errs[float(t)]:.4e}")

    return errs


def save_outputs(prefix, r0_values, output_times, errs):
    fig, ax = plt.subplots(figsize=(7, 5))
    for t in output_times:
        ax.plot(r0_values, errs[float(t)], marker="o", lw=2, label=f"t={t:g} ns")

    ax.set_xlabel("inner radius r0 (cm)")
    ax.set_ylabel("relative L2 error vs 1D slab")
    ax.set_title("RZ Marshak Benchmark: Curvature-to-Planar Transition")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig_path = f"{prefix}.png"
    plt.savefig(fig_path, dpi=160)
    plt.close(fig)

    np.savez(
        f"{prefix}.npz",
        r0_values=np.array(r0_values),
        output_times=np.array(output_times),
        errors=np.array([errs[float(t)] for t in output_times]),
    )

    return fig_path


def main():
    parser = argparse.ArgumentParser(description="Cylindrical r-z Marshak benchmark.")
    parser.add_argument("--times", type=float, nargs="+", default=[1.0, 5.0, 10.0])
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--r0-values", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0, 5.0, 10.0])
    parser.add_argument("--dr", type=float, default=0.2)
    parser.add_argument("--nr", type=int, default=10)
    parser.add_argument("--nz", type=int, default=60)
    parser.add_argument("--Ntarget", type=int, default=20000)
    parser.add_argument("--Nboundary", type=int, default=12000)
    parser.add_argument("--Nmax", type=int, default=120000)
    parser.add_argument("--save-prefix", type=str, default="rz_marshak_benchmark")
    args = parser.parse_args()

    output_times = sorted(args.times)
    r0_values = sorted(args.r0_values)

    errs = run_curvature_scan(
        r0_values=r0_values,
        output_times=output_times,
        dt=args.dt,
        dr=args.dr,
        nr=args.nr,
        nz=args.nz,
        Ntarget=args.Ntarget,
        Nboundary=args.Nboundary,
        Nmax=args.Nmax,
    )

    fig_path = save_outputs(args.save_prefix, r0_values, output_times, errs)
    print(f"\nSaved: {fig_path}")
    print(f"Saved: {args.save_prefix}.npz")


if __name__ == "__main__":
    main()
