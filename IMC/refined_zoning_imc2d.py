#!/usr/bin/env python3
"""2D IMC simulation of the refined_zoning_noneq diffusion problem.

Geometry: 2D Cartesian, xy-mode in IMC2D (x = x-direction, y = z-direction).
Domain:   x ∈ [0, 5] cm,  z ∈ [0, 5] cm

Materials (temperature-independent, position-dependent):
  Optically thick (default):    σ_a = 200 cm⁻¹,  ρc_v = 0.5  GJ/(cm³·keV)
  Lower thin channel x∈[1,2], z<2:  σ_a = 0.2  cm⁻¹,  ρc_v = 0.05 GJ/(cm³·keV)
  Upper thin channel x∈[3,4], z>3:  σ_a = 0.2  cm⁻¹,  ρc_v = 0.05 GJ/(cm³·keV)

Initial condition: T = 0.01 keV everywhere (cold).
Boundary conditions:
  Left (x=0) / Right (x=5): reflecting (vacuum outside treated as perfect mirror)
  Bottom (z=0) / Top (z=5):  Lambertian source at T_bc = 0.3 keV
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import IMC2D as imc2d

__a = imc2d.__a
__c = imc2d.__c

# ── material constants ────────────────────────────────────────────────────────
SIG_THICK = 200.0   # cm⁻¹
SIG_THIN  =   0.2   # cm⁻¹
CV_THICK  =   0.5   # GJ/(cm³·keV)
CV_THIN   =   0.05  # GJ/(cm³·keV)

T_INIT = 0.01   # keV  (uniform cold start)
T_BC   = 0.3    # keV  (boundary temperature, bottom and top)


# ── geometry ──────────────────────────────────────────────────────────────────
def is_optically_thin(x_arr, z_arr):
    """Return bool array: True where cells are in a thin channel."""
    lower = (x_arr >= 1.0) & (x_arr <= 2.0) & (z_arr <  2.0)
    upper = (x_arr >= 3.0) & (x_arr <= 4.0) & (z_arr >  3.0)
    return lower | upper


def build_material_arrays(x_centers, z_centers):
    """Pre-build 2-D σ_a and ρc_v arrays (shape nx × nz)."""
    X, Z = np.meshgrid(x_centers, z_centers, indexing="ij")
    thin = is_optically_thin(X, Z)
    sigma = np.where(thin, SIG_THIN,  SIG_THICK)
    cv    = np.where(thin, CV_THIN,   CV_THICK)
    return sigma, cv


def make_problem_funcs(sigma_arr, cv_arr):
    """Return (sigma_a_func, eos, inv_eos, cv_func) closures for IMC2D.

    All functions accept a 2-D temperature or energy array and return a 2-D
    array of the same shape.  Because the material properties are assumed
    temperature-independent here, the position map is pre-built once.
    """
    def sigma_a_func(T):
        return sigma_arr               # shape (nx, nz), ignores T

    def eos(T):
        return cv_arr * T              # internal energy density e = ρc_v T

    def inv_eos(u):
        return u / cv_arr              # T = e / ρc_v

    def cv_func(T):
        return cv_arr                  # ignores T

    return sigma_a_func, eos, inv_eos, cv_func


# ── mesh ──────────────────────────────────────────────────────────────────────
# Uniform 50×50 grid; thin channels (1 cm wide) resolve to 10 cells each.
NX, NZ = 50, 50
x_edges = np.linspace(0.0, 5.0, NX + 1)
z_edges = np.linspace(0.0, 5.0, NZ + 1)
x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])


# ── run parameters ────────────────────────────────────────────────────────────
NTARGET    = 20000
NBOUNDARY  = 12000   # per active boundary side (two sides inject)
NMAX       = 200000
MAX_EVENTS = 1000    # cap scatter/absorption events per particle

DT_INIT = 1e-3   # ns
DT_MAX  = 5.0    # ns
DT_GROW = 1.1

OUTPUT_TIMES = [1.0, 10.0, 100.0]   # ns


# ── plotting ──────────────────────────────────────────────────────────────────
def _channel_lines(ax):
    """Draw thin-channel boundary lines on an x–z axes."""
    kw = dict(color="cyan", lw=0.8, ls="--", alpha=0.7)
    ax.axhline(1.0, **kw); ax.axhline(2.0, **kw)   # lower channel x-boundaries
    ax.axhline(3.0, **kw); ax.axhline(4.0, **kw)   # upper channel x-boundaries
    ax.axvline(2.0, **kw); ax.axvline(3.0, **kw)   # z-boundaries


def plot_material_layout(sigma_arr):
    """One-time plot showing σ_a layout."""
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    im = ax.pcolormesh(z_centers, x_centers, sigma_arr,
                       shading="auto", cmap="RdYlBu_r",
                       norm=matplotlib.colors.LogNorm(vmin=SIG_THIN, vmax=SIG_THICK))
    plt.colorbar(im, ax=ax, label=r"$\sigma_a$ (cm$^{-1}$)")
    _channel_lines(ax)
    ax.set_xlabel("z (cm)"); ax.set_ylabel("x (cm)")
    ax.set_title("Opacity layout")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig("refined_zoning_imc2d_material.png", dpi=150)
    plt.close()
    print("Saved: refined_zoning_imc2d_material.png")


def plot_snapshot(state, t):
    """Save material-T and radiation-T colormaps at time t."""
    T_mat = state.temperature
    T_rad = state.radiation_temperature

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, T2d, label in zip(
        axes,
        [T_mat, T_rad],
        ["Material $T$ (keV)", "Radiation $T_r$ (keV)"],
    ):
        im = ax.pcolormesh(z_centers, x_centers, T2d,
                           shading="auto", cmap="plasma",
                           vmin=0.0, vmax=T_BC)
        plt.colorbar(im, ax=ax, label=label)
        _channel_lines(ax)
        ax.set_xlabel("z (cm)"); ax.set_ylabel("x (cm)")
        ax.set_title(f"{label}  t = {t:.2f} ns")
        ax.set_aspect("equal")

    plt.tight_layout()
    fname = f"refined_zoning_imc2d_t{t:.2f}ns.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved: {fname}")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("refined_zoning_imc2d.py — non-equilibrium 2-D IMC")
    print(f"  Mesh      : {NX} × {NZ}  (5 cm × 5 cm)")
    print(f"  σ_a       : {SIG_THICK} (thick) / {SIG_THIN} (thin) cm⁻¹")
    print(f"  ρc_v      : {CV_THICK} (thick) / {CV_THIN} (thin) GJ/(cm³·keV)")
    print(f"  T_init    : {T_INIT} keV,  T_bc : {T_BC} keV")
    print(f"  Ntarget   : {NTARGET},  Nboundary : {NBOUNDARY}")
    print(f"  dt_max    : {DT_MAX} ns,  max_events : {MAX_EVENTS}")
    print(f"  Output    : {OUTPUT_TIMES} ns")
    print("=" * 70)

    # Build material arrays.
    sigma_arr, cv_arr = build_material_arrays(x_centers, z_centers)
    sigma_a_func, eos, inv_eos, cv_func = make_problem_funcs(sigma_arr, cv_arr)

    plot_material_layout(sigma_arr)

    # Initialise state.
    Tinit  = np.full((NX, NZ), T_INIT)
    Trinit = np.full((NX, NZ), T_INIT)
    source = np.zeros((NX, NZ))

    state = imc2d.init_simulation(
        NTARGET, Tinit, Trinit,
        x_edges, z_edges,
        eos, inv_eos,
        geometry="xy",
    )

    # Boundary: reflecting on x-sides, Lambertian T_BC on z-sides (bottom/top).
    T_boundary = (0.0, 0.0, T_BC, T_BC)       # left, right, bottom, top
    reflect    = (True, True, False, False)    # left, right, bottom, top

    # Time-stepping.
    dt = DT_INIT
    output_queue = sorted(OUTPUT_TIMES)
    step_count = 0

    while output_queue:
        tout = output_queue[0]

        while state.time < tout - 1e-12:
            step_dt = min(dt, tout - state.time)

            state, info = imc2d.step(
                state,
                NTARGET,
                NBOUNDARY,
                0,
                NMAX,
                T_boundary,
                step_dt,
                x_edges,
                z_edges,
                sigma_a_func,
                inv_eos,
                cv_func,
                source,
                reflect=reflect,
                geometry="xy",
                max_events_per_particle=MAX_EVENTS,
            )
            step_count += 1
            dt = min(dt * DT_GROW, DT_MAX)

            if step_count % 10 == 0:
                print(
                    f"  step {step_count:4d}  t = {state.time:.4e} ns  "
                    f"dt = {step_dt:.4e} ns  N = {info['N_particles']:6d}  "
                    f"E_loss = {info['energy_loss']:.3e}"
                )

        print(f"\n→ snapshot at t = {state.time:.4f} ns")
        plot_snapshot(state, state.time)
        output_queue.pop(0)

    print("\nDone.")


if __name__ == "__main__":
    main()
