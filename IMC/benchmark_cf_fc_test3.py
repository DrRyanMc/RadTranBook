#!/usr/bin/env python3
"""Benchmark FC vs CF for Converging Marshak Wave Test 3 with matched settings.

This script runs both drivers with --profile, loads their *_profile.npz outputs,
and prints a compact comparison table plus optional CSV output.
"""

import argparse
import csv
import os
import subprocess
import sys

import numpy as np


def run_cmd(cmd):
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def load_profile(path):
    d = np.load(path)
    out = {
        "step_count": int(d["step_count"]),
        "phase_totals_s": d["phase_totals_s"],
        "event_totals": d["event_totals"],
        "avg_events_per_particle": float(d["avg_events_per_particle"]),
    }
    return out


def avg_phase(profile):
    return profile["phase_totals_s"] / max(profile["step_count"], 1)


def main():
    p = argparse.ArgumentParser(
        description="Benchmark FC vs CF Test3 with matched settings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--python", default=sys.executable, help="Python executable.")
    p.add_argument("--n-cells", type=int, default=80)
    p.add_argument("--Ntarget", type=int, default=4000)
    p.add_argument("--Nboundary", type=int, default=2000)
    p.add_argument("--NMax", type=int, default=20000)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--final-output-time", type=float, default=-7.86)
    p.add_argument("--numba-threads", type=int, default=None)
    p.add_argument("--event-cap-per-particle", type=int, default=0,
                   help="Passed to CF run only.")
    p.add_argument("--fastpath-threshold", type=float, default=0.0,
                   help="Passed to CF run only.")
    p.add_argument("--prefix", default="bench_test3")
    p.add_argument("--csv", default="", help="Optional CSV output path.")
    args = p.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    fc_script = os.path.join(script_dir, "ConvergingMarshakWaveTest3.py")
    cf_script = os.path.join(script_dir, "ConvergingMarshakWaveTest3_CF.py")

    fc_prefix = f"{args.prefix}_fc"
    cf_prefix = f"{args.prefix}_cf"

    common = [
        "--n-cells", str(args.n_cells),
        "--Ntarget", str(args.Ntarget),
        "--Nboundary", str(args.Nboundary),
        "--NMax", str(args.NMax),
        "--dt", str(args.dt),
        "--final-output-time", str(args.final_output_time),
        "--output-freq", "1",
        "--profile",
    ]

    fc_cmd = [args.python, fc_script, *common,
              "--save-prefix", fc_prefix,
              "--checkpoint-path", f"{fc_prefix}_checkpoint.pkl"]
    cf_cmd = [args.python, cf_script, *common,
              "--save-prefix", cf_prefix,
              "--checkpoint-path", f"{cf_prefix}_checkpoint.pkl",
              "--event-cap-per-particle", str(args.event_cap_per_particle),
              "--fastpath-threshold", str(args.fastpath_threshold)]

    if args.numba_threads is not None:
        fc_cmd += ["--numba-threads", str(args.numba_threads)]
        cf_cmd += ["--numba-threads", str(args.numba_threads)]

    print("Running FC...")
    run_cmd(fc_cmd)
    print("Running CF...")
    run_cmd(cf_cmd)

    fc_prof = load_profile(f"{fc_prefix}_profile.npz")
    cf_prof = load_profile(f"{cf_prefix}_profile.npz")

    fc_avg = avg_phase(fc_prof)
    cf_avg = avg_phase(cf_prof)

    ratio_step = cf_avg[3] / max(fc_avg[3], 1e-30)
    ratio_transport = cf_avg[1] / max(fc_avg[1], 1e-30)
    ratio_events_per_particle = (
        cf_prof["avg_events_per_particle"] / max(fc_prof["avg_events_per_particle"], 1e-30)
    )

    # FC event indices: total,boundary,scatter,census,weight_floor,reflections
    # CF event indices: total,boundary,absorb_continue,census,absorb_capture,weight_floor,reflections,event_cap_hits
    rows = [
        ("avg_step_s", fc_avg[3], cf_avg[3], ratio_step),
        ("avg_sampling_s", fc_avg[0], cf_avg[0], cf_avg[0] / max(fc_avg[0], 1e-30)),
        ("avg_transport_s", fc_avg[1], cf_avg[1], ratio_transport),
        ("avg_post_s", fc_avg[2], cf_avg[2], cf_avg[2] / max(fc_avg[2], 1e-30)),
        ("events_total", int(fc_prof["event_totals"][0]), int(cf_prof["event_totals"][0]),
         int(cf_prof["event_totals"][0]) / max(int(fc_prof["event_totals"][0]), 1)),
        ("events_per_particle", fc_prof["avg_events_per_particle"],
         cf_prof["avg_events_per_particle"], ratio_events_per_particle),
        ("fc_scatter_events", int(fc_prof["event_totals"][2]), None, None),
        ("cf_absorb_continue_events", None, int(cf_prof["event_totals"][2]), None),
        ("cf_absorb_capture_events", None, int(cf_prof["event_totals"][4]), None),
        ("cf_event_cap_hits", None, int(cf_prof["event_totals"][7]), None),
    ]

    print("\nFC vs CF benchmark summary")
    print("metric,fc,cf,cf_over_fc")
    for k, v_fc, v_cf, r in rows:
        print(f"{k},{v_fc},{v_cf},{r}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric", "fc", "cf", "cf_over_fc"])
            for row in rows:
                w.writerow(row)
        print(f"Saved CSV: {args.csv}")


if __name__ == "__main__":
    main()
