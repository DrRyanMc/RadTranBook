#!/usr/bin/env python3
"""
Run a compact incoming-BC regression suite and summarize diagnostics.

Default suite focuses on cold-to-hot behavior:
  - baseline cold-to-hot
  - group-opacity cold-to-hot
  - power-law-opacity cold-to-hot
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time


TESTS = {
    "cold_hot": "test_cold_to_hot_incoming_bc.py",
    "group_opacity": "test_cold_to_hot_incoming_bc_group_opacity.py",
    "powerlaw_opacity": "test_cold_to_hot_incoming_bc_powerlaw_opacity.py",
    "maintain_equilibrium": "test_maintain_equilibrium_with_incoming_bc.py",
    "equilibrium": "test_equilibrium_with_incoming_bc.py",
}

DEFAULT_TESTS = ["cold_hot", "group_opacity", "powerlaw_opacity"]


def parse_diagnostics(output: str) -> dict[str, str]:
    max_dt = "n/a"
    ratio = "n/a"

    m1 = re.search(r"max \|T_mat-T_rad\|\s*=\s*([0-9.eE+-]+)", output)
    if m1:
        max_dt = m1.group(1)

    m2 = re.search(
        r"E_r/\(aT\^4\):\s*min=([0-9.eE+-]+),\s*max=([0-9.eE+-]+),\s*mean=([0-9.eE+-]+)",
        output,
    )
    if m2:
        ratio = f"min={m2.group(1)}, max={m2.group(2)}, mean={m2.group(3)}"

    return {
        "max_dt": max_dt,
        "ratio": ratio,
    }


def run_test(script_path: str, timeout_s: int) -> dict[str, str | int | float]:
    start = time.time()
    try:
        completed = subprocess.run(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        elapsed = time.time() - start
        out = completed.stdout
        diag = parse_diagnostics(out)
        return {
            "exit_code": completed.returncode,
            "elapsed": elapsed,
            "max_dt": diag["max_dt"],
            "ratio": diag["ratio"],
            "output": out,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        elapsed = time.time() - start
        partial = "" if exc.stdout is None else exc.stdout
        return {
            "exit_code": 124,
            "elapsed": elapsed,
            "max_dt": "n/a",
            "ratio": "n/a",
            "output": partial,
            "timed_out": True,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run incoming-BC regression tests")
    parser.add_argument(
        "--tests",
        nargs="+",
        choices=sorted(TESTS.keys()),
        default=DEFAULT_TESTS,
        help="Subset of tests to run",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=240,
        help="Per-test timeout in seconds (default: 240)",
    )
    parser.add_argument(
        "--show-output-on-pass",
        action="store_true",
        help="Print full output for passing tests too",
    )
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))

    print("=" * 110)
    print("Incoming-BC Regression Suite")
    print(f"Python: {sys.executable}")
    print(f"Tests: {', '.join(args.tests)}")
    print("=" * 110)

    results = []
    for name in args.tests:
        script = TESTS[name]
        script_path = os.path.join(here, script)
        print(f"\n>>> Running {name}: {script}")
        result = run_test(script_path, timeout_s=args.timeout)
        results.append((name, script, result))

        status = "PASS" if result["exit_code"] == 0 else "FAIL"
        if result["timed_out"]:
            status = "TIMEOUT"
        print(
            f"    {status} (exit={result['exit_code']}, {result['elapsed']:.1f}s) | "
            f"max|dT|={result['max_dt']} | {result['ratio']}"
        )

        if result["exit_code"] != 0 or args.show_output_on_pass:
            print("    --- output (tail) ---")
            lines = str(result["output"]).splitlines()
            for line in lines[-40:]:
                print(f"    {line}")

    print("\n" + "=" * 110)
    print("Summary")
    print("=" * 110)
    for name, script, result in results:
        status = "PASS" if result["exit_code"] == 0 else "FAIL"
        if result["timed_out"]:
            status = "TIMEOUT"
        print(
            f"{status:8s} {name:20s} {script:45s} "
            f"max|dT|={result['max_dt']:<14s} time={result['elapsed']:.1f}s"
        )

    n_fail = sum(1 for _, _, r in results if int(r["exit_code"]) != 0)
    if n_fail == 0:
        print("\nAll selected incoming-BC regressions passed.")
        return 0

    print(f"\n{n_fail} test(s) failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
