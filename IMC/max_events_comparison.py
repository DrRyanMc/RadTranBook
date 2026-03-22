#!/usr/bin/env python3
"""Compare 2D xy Marshak-wave solutions for different max_events_per_particle.

Runs the same problem (60×60 mesh, T_bc=1 keV, t=1 ns) with a sweep of
max_events values and plots the averaged x-profiles together with the
self-similar solution to show what level of truncation is acceptable.
"""

import sys, os, time as _time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import IMC1D as imc1d
import IMC2D as imc2d

# ── Problem parameters (same as MarshakWave2D defaults) ──────────────────────
CV_VAL   = 0.3
T_BC     = 1.0
L        = 0.2          # cm
N        = 60           # cells per side
DT       = 0.01         # ns
T_FINAL  = 1.0          # ns

NTARGET   = 20000
NBOUNDARY = 12000
NMAX      = 120000

MAX_EVENTS_SWEEP = [10, 30, 100, 300, 1000, 10_000, 1_000_000]

# ── Material functions ────────────────────────────────────────────────────────
def sigma_a_f(T):
    return 300.0 * np.maximum(T, 1e-4)**-3

def eos(T):     return CV_VAL * T
def inv_eos(u): return u / CV_VAL
def cv(T):      return np.zeros_like(T) + CV_VAL


# ── Self-similar solution ─────────────────────────────────────────────────────
def self_similar_T(x_arr, t):
    _a = imc2d.__a; _c = imc2d.__c
    sig0   = 300.0 * T_BC**-3
    xi_max = 1.11305
    omega  = 0.05989
    K = 8.0 * _a * _c / (7.0 * 3.0 * sig0 * 1.0 * CV_VAL)
    xi = x_arr / np.sqrt(K * max(float(t), 1e-30))
    return np.where(
        xi < xi_max,
        T_BC * np.power(np.maximum((1.0 - xi/xi_max)*(1.0 + omega*xi/xi_max), 1e-30), 1/6),
        0.0,
    )


# ── 1D reference run ─────────────────────────────────────────────────────────
def run_1d():
    print("Running 1D reference …", flush=True)
    _t0 = _time.perf_counter()
    x_edges = np.linspace(0.0, L, N + 1)
    mesh    = np.column_stack([x_edges[:-1], x_edges[1:]])
    Tinit   = np.full(N, 1e-4)

    state = imc1d.init_simulation(NTARGET, Tinit, Tinit, mesh, eos, inv_eos,
                                  geometry="slab")
    source = np.zeros(N)
    t = 0.0
    while t < T_FINAL - 1e-12:
        dt  = min(DT, T_FINAL - t)
        state, _ = imc1d.step(
            state, NTARGET, NBOUNDARY, 0, NMAX,
            (T_BC, 0.0), dt, mesh, sigma_a_f, inv_eos, cv, source,
            reflect=(False, True), geometry="slab",
        )
        t += dt
    _elapsed = _time.perf_counter() - _t0
    print(f"  1D done in {_elapsed:.1f}s", flush=True)
    return state.temperature.copy(), _elapsed


# ── 2D run for one max_events value ──────────────────────────────────────────
def run_2d(max_events):
    print(f"Running 2D  max_events={max_events:,} …", flush=True)
    _t0 = _time.perf_counter()
    x_edges = np.linspace(0.0, L, N + 1)
    y_edges = np.linspace(0.0, L, N + 1)
    Tinit   = np.full((N, N), 1e-4)
    source  = np.zeros((N, N))

    state = imc2d.init_simulation(NTARGET, Tinit, Tinit, x_edges, y_edges,
                                  eos, inv_eos, geometry="xy")

    T_boundary = (T_BC, 0.0, 0.0, 0.0)      # left side only
    reflect    = (False, True, True, True)

    n_steps = int(round(T_FINAL / DT))
    t = 0.0
    for step_idx in range(1, n_steps + 1):
        dt = min(DT, T_FINAL - t)
        if dt <= 0:
            break
        _t_step = _time.perf_counter()
        state, _ = imc2d.step(
            state, NTARGET, NBOUNDARY, 0, NMAX,
            T_boundary, dt,
            x_edges, y_edges,
            sigma_a_f, inv_eos, cv, source,
            reflect=reflect,
            geometry="xy",
            max_events_per_particle=max_events,
        )
        t += dt
        _step_wall = _time.perf_counter() - _t_step
        _total     = _time.perf_counter() - _t0
        print(f"  step {step_idx}/{n_steps}  t={t:.3f}ns  step {_step_wall:.1f}s  elapsed {_total:.1f}s",
              flush=True)

    _elapsed = _time.perf_counter() - _t0
    print(f"  2D max_events={max_events:,} done in {_elapsed:.1f}s", flush=True)
    # Average over y to get 1-D profile in x
    return state.temperature.mean(axis=1), _elapsed


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    x_edges  = np.linspace(0.0, L, N + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    T_ss = self_similar_T(x_centers, T_FINAL)

    T1d, t1d = run_1d()

    timings = {}
    results = {}
    for me in MAX_EVENTS_SWEEP:
        np.random.seed(42)          # reproducible comparison
        results[me], timings[me] = run_2d(me)

    # ── Plot profiles ─────────────────────────────────────────────────────────
    cmap   = plt.get_cmap("plasma")
    colors = [cmap(i / (len(MAX_EVENTS_SWEEP) - 1)) for i in range(len(MAX_EVENTS_SWEEP))]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_centers, T_ss, "k:",  lw=2.0, label="self-similar")
    ax.plot(x_centers, T1d, "k--", lw=1.5, label="1D ref")

    for color, me in zip(colors, MAX_EVENTS_SWEEP):
        label = f"max_events={me:,}"
        ax.plot(x_centers, results[me], color=color, lw=1.5, label=label)

    ax.set_xlabel("x (cm)")
    ax.set_ylabel("T (keV)")
    ax.set_title(f"Marshak wave at t = {T_FINAL} ns: effect of max_events_per_particle")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fname = "max_events_comparison_profiles.png"
    plt.savefig(fname, dpi=160)
    print(f"Saved: {fname}")
    plt.close()

    # ── Plot L2 error vs max_events ───────────────────────────────────────────
    ref = results[MAX_EVENTS_SWEEP[-1]]   # highest max_events as reference
    errors_vs_ss = []
    errors_vs_ref = []
    for me in MAX_EVENTS_SWEEP:
        denom_ss = np.linalg.norm(T_ss)
        denom_ref = np.linalg.norm(ref)
        errors_vs_ss.append(np.linalg.norm(results[me] - T_ss)  / max(denom_ss,  1e-30))
        errors_vs_ref.append(np.linalg.norm(results[me] - ref)  / max(denom_ref, 1e-30))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(MAX_EVENTS_SWEEP, errors_vs_ss,  "o-", label="vs self-similar")
    ax.loglog(MAX_EVENTS_SWEEP, errors_vs_ref, "s--", label=f"vs max_events={MAX_EVENTS_SWEEP[-1]:,}")
    ax.set_xlabel("max_events_per_particle")
    ax.set_ylabel("Relative L2 error")
    ax.set_title("Truncation error vs max_events_per_particle")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    fname2 = "max_events_comparison_errors.png"
    plt.savefig(fname2, dpi=160)
    print(f"Saved: {fname2}")
    plt.close()

    # ── Print summary table ───────────────────────────────────────────────────
    print(f"\n  1D reference: {t1d:.1f}s")
    print(f"\n{'max_events':>15}  {'L2 err vs SS':>14}  {'L2 err vs ref':>14}  {'wall time (s)':>14}")
    print("-" * 66)
    for me, e_ss, e_ref in zip(MAX_EVENTS_SWEEP, errors_vs_ss, errors_vs_ref):
        print(f"{me:>15,}  {e_ss:>14.4f}  {e_ref:>14.4f}  {timings[me]:>14.1f}")


if __name__ == "__main__":
    main()
