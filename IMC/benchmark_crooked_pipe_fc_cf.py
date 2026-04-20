#!/usr/bin/env python3
"""Run matched Crooked-pipe FC/CF comparisons and write CSV summary."""

import argparse
import csv
import glob
import os
import subprocess
import time
import numpy as np


def _latest_solution_file(method, mesh_tag, particle_tag, created_after):
    pattern = f"crooked_pipe_imc_solution_{method}_{mesh_tag}_{particle_tag}_*.npz"
    candidates = [p for p in glob.glob(pattern) if os.path.getmtime(p) >= created_after]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def _run_method(method, args):
    cmd = [
        args.python,
        "IMC/CrookedPipe2D.py",
        "--method", method,
        "--nr", str(args.nr),
        "--nz", str(args.nz),
        "--Ntarget", str(args.Ntarget),
        "--Nboundary", str(args.Nboundary),
        "--Nmax", str(args.Nmax),
        "--dt-initial", str(args.dt_initial),
        "--dt-max", str(args.dt_max),
        "--dt-increase-factor", str(args.dt_increase_factor),
        "--output-times", args.output_times,
        "--max-events", str(args.max_events),
    ]
    if args.use_refined_mesh:
        cmd.append("--use-refined-mesh")
        cmd.extend(["--n-refine", str(args.n_refine), "--refine-width", str(args.refine_width)])
    if method == "cf":
        cmd.extend(["--fastpath-threshold", str(args.fastpath_threshold)])

    start = time.perf_counter()
    started_file_time = time.time() - 0.2
    completed = subprocess.run(cmd, check=True)
    wall_s = time.perf_counter() - start

    if completed.returncode != 0:
        raise RuntimeError(f"Run failed for method={method}")

    mesh_tag = "refined" if args.use_refined_mesh else "uniform"
    particle_tag = f"Nb{args.Nboundary}" if args.Nboundary > 0 else f"Nt{args.Ntarget}"
    npz_path = _latest_solution_file(method, mesh_tag, particle_tag, started_file_time)
    if npz_path is None:
        raise RuntimeError(f"Could not locate solution output for method={method}")

    d = np.load(npz_path, allow_pickle=True)
    phase = d["profile_phase_totals_s"]
    events = d["profile_event_totals"]

    result = {
        "case_tag": args.case_tag,
        "mesh_type": "refined" if args.use_refined_mesh else "uniform",
        "nr": int(args.nr),
        "nz": int(args.nz),
        "Ntarget": int(args.Ntarget),
        "Nboundary": int(args.Nboundary),
        "Nmax": int(args.Nmax),
        "method": method,
        "solution_file": npz_path,
        "wall_elapsed_s": wall_s,
        "profile_wall_elapsed_s": float(d["profile_wall_elapsed_s"]),
        "sampling_s": float(phase[0]),
        "transport_s": float(phase[1]),
        "postprocess_s": float(phase[2]),
        "profile_total_s": float(phase[3]),
        "steps_profiled": int(d["profile_step_count"]),
        "events_total": int(events[0]),
        "boundary_crossings": int(events[1]),
        "absorb_continue_events": int(events[2]),
        "census_events": int(events[3]),
        "absorb_capture_events": int(events[4]),
        "weight_floor_kills": int(events[5]),
        "reflections": int(events[6]),
        "event_cap_hits": int(events[7]),
        "transported_particles_total": int(d["profile_transported_particles_total"]),
        "avg_events_per_particle": float(d["profile_avg_events_per_particle"]),
        "fastpath_threshold": float(d["fastpath_threshold"]),
        "max_temperature_keV": float(np.max(d["T_final"])),
    }
    return result


def main():
    p = argparse.ArgumentParser(description="Benchmark Crooked-pipe FC vs CF")
    p.add_argument("--python", type=str, default="/usr/local/bin/python3")
    p.add_argument("--nr", type=int, default=12)
    p.add_argument("--nz", type=int, default=30)
    p.add_argument("--Ntarget", type=int, default=500)
    p.add_argument("--Nboundary", type=int, default=250)
    p.add_argument("--Nmax", type=int, default=4000)
    p.add_argument("--dt-initial", type=float, default=0.001)
    p.add_argument("--dt-max", type=float, default=0.005)
    p.add_argument("--dt-increase-factor", type=float, default=1.2)
    p.add_argument("--output-times", type=str, default="0.01")
    p.add_argument("--max-events", type=int, default=10**6)
    p.add_argument("--use-refined-mesh", action="store_true")
    p.add_argument("--n-refine", type=int, default=10)
    p.add_argument("--refine-width", type=float, default=0.05)
    p.add_argument("--fastpath-threshold", type=float, default=0.0)
    p.add_argument("--csv", type=str, default="IMC/crooked_pipe_fc_cf_compare.csv")
    p.add_argument("--append", action="store_true", help="Append to CSV if it already exists")
    p.add_argument("--case-tag", type=str, default="case", help="Case label written to CSV")
    args = p.parse_args()

    print("Running matched FC/CF Crooked-pipe benchmark...")
    fc = _run_method("fc", args)
    cf = _run_method("cf", args)

    fc_wall = fc["wall_elapsed_s"]
    cf_wall = cf["wall_elapsed_s"]
    wall_ratio_fc_over_cf = fc_wall / cf_wall if cf_wall > 0.0 else float("nan")

    rows = [fc, cf]
    fieldnames = [
        "case_tag",
        "mesh_type",
        "nr",
        "nz",
        "Ntarget",
        "Nboundary",
        "Nmax",
        "method",
        "solution_file",
        "wall_elapsed_s",
        "profile_wall_elapsed_s",
        "sampling_s",
        "transport_s",
        "postprocess_s",
        "profile_total_s",
        "steps_profiled",
        "events_total",
        "boundary_crossings",
        "absorb_continue_events",
        "census_events",
        "absorb_capture_events",
        "weight_floor_kills",
        "reflections",
        "event_cap_hits",
        "transported_particles_total",
        "avg_events_per_particle",
        "fastpath_threshold",
        "max_temperature_keV",
    ]

    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    write_header = True
    mode = "w"
    if args.append and os.path.exists(args.csv):
        write_header = False
        mode = "a"

    with open(args.csv, mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Saved benchmark CSV: {args.csv}")
    print(f"Wall-time ratio FC/CF = {wall_ratio_fc_over_cf:.6f}")


if __name__ == "__main__":
    main()
