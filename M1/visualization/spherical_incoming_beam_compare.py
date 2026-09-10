#!/usr/bin/env python3
"""Spherical shell incoming-beam comparison: M1 closures vs spherical S_N.

A 1-D spherical shell r in [r_inner, r_outer] is driven by an incoming beam
at the outer boundary. The same setup is solved with several M1 closures and
with the spherical LD S_N solver for comparison.

Run from M1/doc (or any directory):
  python ../problems/spherical_incoming_beam_compare.py
  python ../problems/spherical_incoming_beam_compare.py --load
"""

import os
import sys
import csv
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_here = os.path.dirname(os.path.abspath(__file__))
_m1_root = os.path.dirname(_here)
_project_root = os.path.dirname(_m1_root)
_sn_root = os.path.join(_project_root, "DiscreteOrdinates")

sys.path.insert(0, _m1_root)
sys.path.insert(0, _project_root)
sys.path.insert(0, _sn_root)

from M1.src.m1_1d import (
    M1Solver1D,
    closure_kershaw,
    closure_levermore,
    closure_minerbo_poly,
    C_LIGHT,
    A_RAD,
)
from DiscreteOrdinates.src.sn_solver import _get_quadrature
from DiscreteOrdinates.src.sn_solver_ld_sphere import temp_solve_sph_ld, c as C_SN, ac as AC_SN
from utils.plotfuncs import show


# ---------------------------------------------------------------------------
# CLI and setup
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Spherical shell incoming-beam M1 vs S_N comparison"
)
parser.add_argument("--load", action="store_true", help="Load cached NPZ data.")
parser.add_argument("--r-inner", type=float, default=0.2, help="Inner radius [cm].")
parser.add_argument("--r-outer", type=float, default=1.0, help="Outer radius [cm].")
parser.add_argument("--n-cells", type=int, default=200, help="Number of radial cells.")
parser.add_argument("--sigma-a", type=float, default=1.0, help="Absorption opacity [cm^-1].")
parser.add_argument("--sn-order", type=int, default=32, help="S_N order for spherical solver.")
parser.add_argument("--output-base", type=str, default="spherical_shell_beam", help="Output filename prefix.")
args = parser.parse_args()

r_inner = float(args.r_inner)
r_outer = float(args.r_outer)
n_cells = int(args.n_cells)
sigma_a = float(args.sigma_a)
N_sn_default = int(args.sn_order)
output_base = args.output_base

if r_inner < 0.0 or r_outer <= r_inner:
    raise ValueError("Require 0 <= r_inner < r_outer.")

T_bc = 1.0
T_init = 1e-3
Cv = 10.0

dr_val = (r_outer - r_inner) / n_cells
x_c = r_inner + dr_val * (0.5 + np.arange(n_cells))

E_bc = A_RAD * T_bc ** 4
mean_free_time = 1.0 / (C_LIGHT * sigma_a)

dt = 0.01 * mean_free_time
output_tau = [0.25, 0.5, 1.0, 2.0, 4.0]
output_times_ns = [tau * mean_free_time for tau in output_tau]
t_final = output_times_ns[-1]
n_steps = int(np.ceil(t_final / dt))

print(f"E_bc            = {E_bc:.4e} GJ/cm^3")
print(f"Domain          = [{r_inner:.3f}, {r_outer:.3f}] cm")
print(f"Mean-free time  = {mean_free_time:.4e} ns")
print(f"dt              = {dt:.4e} ns  (CFL = {C_LIGHT * dt / dr_val:.2f})")
print(f"n_steps         = {n_steps}")
print(f"Output at tau   = {output_tau}")


# ---------------------------------------------------------------------------
# Closures and material helpers
# ---------------------------------------------------------------------------

closures = {
    "Kershaw": closure_kershaw,
    "Levermore": closure_levermore,
    "Minerbo poly": closure_minerbo_poly,
}


def EOS_linear(T):
    return Cv * T


def invEOS_linear(e):
    return e / Cv


def sigma_const(T):
    return np.full_like(T, sigma_a)


def scat_zero(T):
    return np.zeros_like(T)


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------

def run_m1_all_closures():
    m1_results = {}
    # Inward beam at r=r_outer in M1 moments.
    f_bc = -0.999 * C_LIGHT * E_bc

    print("\nRunning M1 closures in spherical geometry (d=2)")
    for name, cl_func in closures.items():
        print(f"  running {name} ...")
        solver = M1Solver1D(
            x_min=r_inner,
            x_max=r_outer,
            n_cells=n_cells,
            d=2,
            sigma_func=sigma_const,
            scat_func=scat_zero,
            EOS=EOS_linear,
            invEOS=invEOS_linear,
            dt=dt,
            closure_func=cl_func,
            # Vacuum-like inner boundary for shell comparison.
            left_bc_mode="incident",
            left_bc_state=(0.0, 0.0),
            right_bc_mode="incident",
            right_bc_state=(E_bc, f_bc),
            max_nonlinear_iter=20,
            nonlinear_tol=1e-8,
        )
        solver.initialize(T_init=T_init)

        snaps = solver.run(n_steps=n_steps, source_func=None, output_times=output_times_ns)

        m1_results[name] = {}
        for tau, t_ns in zip(output_tau, output_times_ns):
            if t_ns in snaps:
                s = snaps[t_ns]
                m1_results[name][tau] = {
                    "x": s["x"],
                    "Er": s["Er"],
                    "F": s["F"],
                }

    return m1_results


def run_spherical_sn():
    N_sn = N_sn_default
    MU, W = _get_quadrature(N_sn)

    # In this spherical LD formulation, mu=-1 is represented by the
    # starting-direction state g. To model a beam entering at mu=-1,
    # inject through bc_g_outer and leave ordinate inflows zero.
    _n_beam = int(np.argmin(MU))
    psi_beam = C_SN * E_bc

    def BCs(_t):
        bc_outer = np.zeros((N_sn, 2))
        # Keep inward ordinate inflow off; beam is supplied via g at mu=-1.
        return bc_outer, float(psi_beam)

    def BCs_inner(_t):
        # Hollow-shell inner vacuum wall: no incoming mu>0 intensity.
        return np.zeros((N_sn, 2))

    r_left = r_inner + np.arange(n_cells, dtype=np.float64) * dr_val
    dr = np.full(n_cells, dr_val, dtype=np.float64)

    phi0 = np.full((n_cells, 2), AC_SN * T_init ** 4)
    psi0 = np.broadcast_to((phi0 / 2.0)[:, None, :], (n_cells, N_sn, 2)).copy()
    T0 = np.full((n_cells, 2), T_init)
    g0 = phi0 / 2.0

    q_n = np.zeros((n_cells, N_sn, 2))
    q_g = np.zeros((n_cells, 2))

    print(f"\nRunning spherical S_N (S{N_sn})")
    phis, _Ts, _gs, iters, ts, _ips = temp_solve_sph_ld(
        I=n_cells,
        r_left=r_left,
        dr=dr,
        q_n=q_n,
        q_g=q_g,
        sigma_func=sigma_const,
        scat_func=scat_zero,
        N=N_sn,
        BCs=BCs,
        EOS=EOS_linear,
        invEOS=invEOS_linear,
        phi=phi0,
        psi=psi0,
        T=T0,
        g_init=g0,
        dt_min=dt,
        dt_max=dt,
        tfinal=t_final,
        Linf_tol=1e-6,
        tolerance=1e-8,
        maxits=200,
        LOUD=False,
        fix=1,
        K=20,
        R=3,
        time_outputs=np.array(output_times_ns),
        reflect_outer=False,
        reflect_inner=False,
        BCs_inner=BCs_inner,
        print_stride=0,
        use_dmd=True,
    )

    print(f"  S_N total sweeps: {iters}")

    ts_arr = np.asarray(ts)
    sn_results = {}
    for tau, t_ns in zip(output_tau, output_times_ns):
        idx = int(np.argmin(np.abs(ts_arr[1:] - t_ns)))
        phi_snap = phis[idx + 1]
        Er_snap = 0.5 * (phi_snap[:, 0] + phi_snap[:, 1]) / C_SN
        sn_results[tau] = {"x": x_c.copy(), "Er": Er_snap}

    return N_sn, sn_results


def compute_metrics(m1_results, sn_results):
    rows = []
    for tau in output_tau:
        ref = sn_results[tau]["Er"]
        ref_norm = np.linalg.norm(ref)
        for name in closures:
            if tau not in m1_results[name]:
                continue
            val = m1_results[name][tau]["Er"]
            diff = val - ref
            l2_rel = float(np.linalg.norm(diff) / ref_norm) if ref_norm > 0.0 else np.nan
            linf_rel = float(np.max(np.abs(diff)) / (np.max(np.abs(ref)) + 1e-30))
            rows.append({
                "tau": float(tau),
                "method": name,
                "l2_rel_Er": l2_rel,
                "linf_rel_Er": linf_rel,
            })
    return rows


def save_metrics_csv(rows, csv_path):
    cols = ["tau", "method", "l2_rel_Er", "linf_rel_Er"]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {csv_path}")


# ---------------------------------------------------------------------------
# Load or run
# ---------------------------------------------------------------------------

npz_path = f"{output_base}_data.npz"
csv_path = f"{output_base}_metrics.csv"

if args.load:
    print(f"Loading data from {npz_path} ...")
    data = np.load(npz_path)
    x_c = data["x"]
    E_bc = float(data["E_bc"])
    output_tau = list(data["output_tau"])
    N_sn = int(data["N_sn"])

    sn_results = {}
    for tau in output_tau:
        sn_results[tau] = {"x": x_c, "Er": data[f"Er_sn_tau{tau}"]}

    m1_results = {}
    for name in closures:
        key = name.replace(" ", "_")
        m1_results[name] = {}
        for tau in output_tau:
            arr_key = f"Er_{key}_tau{tau}"
            f_key = f"F_{key}_tau{tau}"
            if arr_key in data:
                m1_results[name][tau] = {
                    "x": x_c,
                    "Er": data[arr_key],
                    "F": data[f_key] if f_key in data else np.zeros_like(x_c),
                }
    print("Done.")
else:
    m1_results = run_m1_all_closures()
    N_sn, sn_results = run_spherical_sn()

    save_dict = {
        "x": x_c,
        "E_bc": E_bc,
        "output_tau": np.array(output_tau),
        "N_sn": np.array(N_sn),
        "r_inner": np.array(r_inner),
        "r_outer": np.array(r_outer),
        "sigma_a": np.array(sigma_a),
    }
    for tau in output_tau:
        save_dict[f"Er_sn_tau{tau}"] = sn_results[tau]["Er"]
        for name, snaps in m1_results.items():
            if tau in snaps:
                key = name.replace(" ", "_")
                save_dict[f"Er_{key}_tau{tau}"] = snaps[tau]["Er"]
                save_dict[f"F_{key}_tau{tau}"] = snaps[tau]["F"]
    np.savez(npz_path, **save_dict)
    print(f"Saved {npz_path}")


# ---------------------------------------------------------------------------
# Metrics and plots
# ---------------------------------------------------------------------------

rows = compute_metrics(m1_results, sn_results)
save_metrics_csv(rows, csv_path)

closure_colors = {
    "Kershaw": "#2ca02c",
    "Levermore": "#d62728",
    "Minerbo poly": "#9467bd",
}
closure_styles = {
    "Kershaw": ("-.", r"Kershaw"),
    "Levermore": ("--", r"Levermore"),
    "Minerbo poly": ((0, (3, 1, 1, 1)), r"Minerbo"),
}

legend_handles = [
    Line2D(
        [0], [0], marker="s", color="w", markerfacecolor="k",
        markeredgecolor="k", markersize=5, linestyle="-",
        label=rf"S$_{{\!{N_sn}}}$ spherical"
    )
] + [
    Line2D(
        [0], [0], color=closure_colors[n], linestyle=closure_styles[n][0],
        lw=2, label=closure_styles[n][1]
    )
    for n in closures
]

for tau in output_tau:
    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    ax.plot(
        sn_results[tau]["x"], sn_results[tau]["Er"] / E_bc,
        "ks", ms=4, ls="-", zorder=9, markevery=range(0, n_cells, 10),
    )

    for name, snaps in m1_results.items():
        if tau in snaps:
            ax.plot(
                snaps[tau]["x"], snaps[tau]["Er"] / E_bc,
                color=closure_colors[name], linestyle=closure_styles[name][0],
                lw=2.0,
            )

    yvals = [sn_results[tau]["Er"] / E_bc]
    for name, snaps in m1_results.items():
        if tau in snaps:
            yvals.append(snaps[tau]["Er"] / E_bc)
    ycat = np.concatenate(yvals)
    ycat_pos = ycat[ycat > 0.0]
    y_min = max(1e-8, 0.8 * float(np.min(ycat_pos))) if ycat_pos.size else 1e-8
    y_max = 1.2 * float(np.max(ycat)) if np.max(ycat) > 0.0 else 1.0

    ax.set_ylim([y_min, y_max])
    ax.set_xlim([r_inner, r_outer])
    ax.set_yscale("log")
    ax.set_xlabel("$r$ (cm)", fontsize=13)
    ax.set_ylabel(r"$E_r\,/\,E_{\rm bc}$", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(handles=legend_handles, fontsize=9, loc="best")
    ax.set_title(rf"Incoming beam at $r={r_outer:.1f}$ cm, shell optical time $\tau={tau:.2f}$")

    tau_str = f"{tau:.2f}".replace(".", "p")
    outname = f"{output_base}_tau{tau_str}.pdf"
    plt.tight_layout()
    show(outname, close_after=True)
    print(f"Saved {outname}")

print("\nDone.")
