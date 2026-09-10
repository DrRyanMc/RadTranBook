"""Two-beam interaction problem: M1 closures vs S_N transport.

Two opposing beams enter from opposite boundaries of a 2 cm (2 mfp) absorbing
slab.  Because M1 represents the radiation field with only its first two angular
moments, it cannot distinguish two beams travelling in opposite directions; when
the beams overlap strongly the moment equations force a nearly isotropic state,
producing a characteristic ``jump'' from beam-like to isotropic.

Problem setup
-------------
  Domain:   x ∈ [0, 2] cm  (σ_a = 1 cm⁻¹ → 2 mean-free paths)
  Material: linear EOS  e = Cᵥ T  with Cᵥ = 10 GJ/(cm³ keV)  (cold, stays cold)
  Initial:  T = 10⁻³ keV throughout,  Eᵣ = aT⁴,  F = 0
  Left BC:  E_r = aT_bc⁴,  F = +c E_r   (fully right-streaming beam)
  Right BC: E_r = aT_bc⁴,  F = −c E_r   (fully  left-streaming beam)
  T_bc = 1 keV

The analytic steady-state transport solution is the superposition of two
exponentially attenuated beams:
    E_r(x) = E_bc (e^{−x} + e^{−(2−x)}).

Outputs
-------
Saved in the directory from which the script is run:
  two_beam_m1_ss.pdf   – E_r / E_bc vs x for all M1 closures + SN
  two_beam_f_ss.pdf    – reduced flux |F|/(c E_r) vs x (steady state)
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_here        = os.path.dirname(os.path.abspath(__file__))
_m1_root     = os.path.dirname(_here)              # M1/
_project_root = os.path.dirname(_m1_root)           # RadTranBook/
_sn_root     = os.path.join(_project_root, "DiscreteOrdinates")

sys.path.insert(0, _m1_root)
sys.path.insert(0, _project_root)
sys.path.insert(0, _sn_root)

from M1.src.m1_1d import (
    M1Solver1D,
    closure_p1,
    closure_kershaw,
    closure_levermore,
    closure_minerbo_poly,
    closure_minerbo_rational,
    C_LIGHT,
    A_RAD,
)
from utils.plotfuncs import show
from DiscreteOrdinates.src.sn_solver import _get_quadrature
from DiscreteOrdinates.src.sn_solver_ld import temp_solve_ld

# ---------------------------------------------------------------------------
# Problem parameters
# ---------------------------------------------------------------------------

L        = 2.0          # cm
sigma_a  = 1.0          # cm⁻¹  →  domain is 2 mean-free paths
T_bc     = 1.0          # keV  (beam drive temperature)
T_init   = 1e-3         # keV  (initial cold state)
Cv       = 10.0         # GJ/(cm³ keV)  — large so material barely heats
E_bc     = A_RAD * T_bc ** 4          # GJ/cm³  beam energy density
n_cells  = 200
hx       = L / n_cells
x_c      = hx / 2 + hx * np.arange(n_cells)   # cell centres

mean_free_time = 1.0 / (C_LIGHT * sigma_a)    # τ₀ = 1/(cσ)  [ns]

print(f"E_bc            = {E_bc:.4e} GJ/cm³")
print(f"Mean-free time  = {mean_free_time:.4e} ns")
print(f"Light-crossing  = {L/C_LIGHT:.4e} ns")

# ---------------------------------------------------------------------------
# Time discretisation — CFL ≈ 1 for good temporal accuracy
# ---------------------------------------------------------------------------

dt = 0.01 * mean_free_time     # c·dt/hx = 0.01*(1/σ)/(L/n) = 1 exactly

output_tau      = [0.5, 1.0, 2.0, 5.0]   # output in units of τ₀
output_times_ns = [tau * mean_free_time for tau in output_tau]
t_final         = output_times_ns[-1]
n_steps         = int(np.ceil(t_final / dt))

print(f"dt              = {dt:.4e} ns  (CFL = {C_LIGHT*dt/hx:.2f})")
print(f"n_steps         = {n_steps}")
print(f"Output at τ     = {output_tau}")

# ---------------------------------------------------------------------------
# Analytic solution at time τ (purely absorbing, sharp wavefront)
#
#   E_r(x, τ) = E_bc · e^{−σx}   ·  H(τ − σx)        right-going beam
#             + E_bc · e^{−σ(L−x)} ·  H(τ − σ(L−x))   left-going beam
#
# At τ = 1 the two fronts (each 1 mfp wide) exactly meet at x = 1.
# For τ ≥ σL = 2 the full steady-state superposition is established.
# ---------------------------------------------------------------------------

def analytic_Er(tau):
    ct_mfp = tau           # c·t in units of mfp  (σ = 1)
    Er = np.zeros(n_cells)
    mask_R = x_c * sigma_a <= ct_mfp
    Er[mask_R] += E_bc * np.exp(-sigma_a * x_c[mask_R])
    mask_L = (L - x_c) * sigma_a <= ct_mfp
    Er[mask_L] += E_bc * np.exp(-sigma_a * (L - x_c[mask_L]))
    return Er

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Two-beam M1 vs S_N comparison")
parser.add_argument("--load", action="store_true",
                    help="load results from two_beam_data.npz instead of re-running")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Closures (needed for both computation and plotting)
# ---------------------------------------------------------------------------

closures = {
    #"P1":               closure_p1,
    "Kershaw":          closure_kershaw,
    "Levermore":        closure_levermore,
    "Minerbo poly":     closure_minerbo_poly,
    #"Minerbo rational": closure_minerbo_rational,
}

# ---------------------------------------------------------------------------
# M1 run — all closures, time snapshots via run()
# ---------------------------------------------------------------------------

def EOS_linear(T):
    return Cv * T

def invEOS_linear(e):
    return e / Cv

def sigma_const(T):
    return np.full_like(T, sigma_a)

# 99.9% free-streaming to stay inside the realizability boundary.
_f_bc = 0.999 * C_LIGHT * E_bc

if args.load:
    # ── Load pre-computed results ────────────────────────────────────────────
    npz_path = "two_beam_data.npz"
    print(f"Loading data from {npz_path} ...")
    data = np.load(npz_path)
    x_c        = data["x"]
    E_bc       = float(data["E_bc"])
    output_tau = list(data["output_tau"])

    sn_results = {}
    for tau in output_tau:
        sn_results[tau] = data[f"Er_sn_tau{tau}"]

    m1_results = {}
    for name in closures:
        key = name.replace(" ", "_")
        m1_results[name] = {}
        for tau in output_tau:
            arr_key = f"Er_{key}_tau{tau}"
            if arr_key in data:
                m1_results[name][tau] = {"x": x_c, "Er": data[arr_key]}
    print("Done.")

else:
    # ── Run M1 ───────────────────────────────────────────────────────────────
    m1_results = {}   # m1_results[closure_name][tau] = {"x", "Er", "F"}

    print(f"\nM1: dt = {dt:.4e} ns, n_steps = {n_steps}")

    for name, cl_func in closures.items():
        print(f"  running {name} ...")
        solver = M1Solver1D(
            x_min=0.0, x_max=L, n_cells=n_cells, d=0,
            sigma_func=sigma_const,
            EOS=EOS_linear, invEOS=invEOS_linear,
            dt=dt,
            closure_func=cl_func,
            left_bc_mode="incident",
            right_bc_mode="incident",
            left_bc_state=(E_bc,  _f_bc),    # right-going beam from left
            right_bc_state=(E_bc, -_f_bc),   # left-going  beam from right
            max_nonlinear_iter=10,
            nonlinear_tol=1e-6,
        )
        solver.initialize(T_init=T_init)

        snaps_ns = solver.run(n_steps=n_steps, source_func=None,
                              output_times=output_times_ns)

        m1_results[name] = {}
        for tau, t_ns in zip(output_tau, output_times_ns):
            if t_ns in snaps_ns:
                s = snaps_ns[t_ns]
                m1_results[name][tau] = {"x": s["x"], "Er": s["Er"], "F": s["F"]}

    # ── Run S_N ──────────────────────────────────────────────────────────────
    N_sn = 32
    MU, W = _get_quadrature(N_sn)

    psi_bc_left  = E_bc / W[N_sn - 1]
    psi_bc_right = E_bc / W[0]

    def sn_BCs(t):
        bcs = np.zeros((N_sn, 2))
        bcs[N_sn - 1, 1] = psi_bc_left
        bcs[0,         0] = psi_bc_right
        return bcs

    phi0 = np.full((n_cells, 2), A_RAD * T_init ** 4)
    psi0 = np.zeros((n_cells, N_sn, 2))
    T0   = np.full((n_cells, 2), T_init)
    q_sn = np.zeros((n_cells, N_sn, 2))

    print(f"\nS_N (S{N_sn}): dt = {dt:.4e} ns, t_final = {t_final:.4e} ns")

    phis_sn, _, iters_sn, ts_sn, _ = temp_solve_ld(
        I=n_cells, hx=hx, q=q_sn,
        sigma_func=lambda T: np.full_like(T, sigma_a),
        scat_func=lambda T: np.zeros_like(T),
        N=N_sn, BCs=sn_BCs,
        EOS=lambda T: Cv * T, invEOS=lambda e: e / Cv,
        phi=phi0, psi=psi0, T=T0,
        dt_min=dt, dt_max=dt,
        tfinal=t_final,
        time_outputs=np.array(output_times_ns),
        LOUD=False, fix=1,
        reflect_left=False, reflect_right=False,
        use_dmd=True, K=20, R=3,
        tolerance=1e-8, Linf_tol=1e-5, maxits=200,
    )
    print(f"  S_N total sweeps: {iters_sn}")

    sn_results = {}
    for tau, t_ns in zip(output_tau, output_times_ns):
        idx = int(np.argmin(np.abs(np.array(ts_sn) - t_ns)))
        sn_results[tau] = 0.5 * (phis_sn[idx][:, 0] + phis_sn[idx][:, 1])

    # ── Save data ────────────────────────────────────────────────────────────
    save_dict = {"x": x_c, "E_bc": E_bc, "output_tau": np.array(output_tau)}
    for tau in output_tau:
        save_dict[f"Er_sn_tau{tau}"] = sn_results[tau]
        for name, snaps in m1_results.items():
            if tau in snaps:
                key = name.replace(" ", "_")
                save_dict[f"Er_{key}_tau{tau}"] = snaps[tau]["Er"]
    np.savez("two_beam_data.npz", **save_dict)
    print("Saved two_beam_data.npz")

# N_sn is used in the legend label; define it if we loaded instead of running.
if args.load:
    N_sn = 32

# ---------------------------------------------------------------------------
# Plot — one single-panel figure per output time
# ---------------------------------------------------------------------------

closure_colors = {
    "P1":               "#1f77b4",
    "Kershaw":          "#2ca02c",
    "Levermore":        "#d62728",
    "Minerbo poly":     "#9467bd",
    #"Minerbo rational": "#8c564b",
}
closure_styles = {
    "P1":               (":", r"P$_1$"),
    "Kershaw":          ("-.", r"Kershaw"),
    "Levermore":        ("--", r"Levermore"),
    "Minerbo poly":     ((0, (3, 1, 1, 1)), r"Minerbo"),
    #"Minerbo rational": (0, (3, 1, 1, 1)),
}

legend_handles = [
    Line2D([0], [0], marker="s", color="w", markerfacecolor="k",
           markeredgecolor="k", markersize=5, linestyle="-", label=f"S$_{{\\!{N_sn}}}$"),
] + [
    Line2D([0], [0], color=closure_colors[n], linestyle=closure_styles[n][0],
           lw=2, label=closure_styles[n][1])
    for n in closures
]

for tau in output_tau:
    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    ax.plot(x_c, sn_results[tau] / E_bc, "ks", ms=4, ls="-", zorder=9, markevery=range(0, n_cells, 10))

    for name, snaps in m1_results.items():
        if tau in snaps:
            ax.plot(snaps[tau]["x"], snaps[tau]["Er"] / E_bc,
                    color=closure_colors[name], linestyle=closure_styles[name][0],
                    lw=2.0)

    #ax.set_yscale("log")
    ax.set_ylim([1e-3, 2.0])
    ax.set_xlim([0, L])
    ax.set_xlabel("$z$ (cm = mfp)", fontsize=13)
    ax.set_ylabel(r"$E_r\,/\,E_{\rm bc}$", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(handles=legend_handles, fontsize=9, loc="best")

    tau_str = f"{tau:.1f}".replace(".", "p")
    outname = f"two_beam_tau{tau_str}.pdf"
    plt.tight_layout()
    show(outname, close_after=True)
    print(f"Saved {outname}")

print("\nDone.")


