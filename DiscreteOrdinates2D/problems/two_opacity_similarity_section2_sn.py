#!/usr/bin/env python3
import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

"""Two-opacity similarity problem, Section 2, using 2-D S_N transport.

The geometry and material parameters match
``EqDiffusion/problems/two_opacity_similarity_section2_eqdiff.py``:

* ``y < 0`` has kappa = 2400 cm^-1 and ``y > 0`` has kappa = 4800 cm^-1;
* both regions have Cv = 300 GJ/(cm^3 keV);
* x = 0 is driven at T_S = 1 keV, x = x_max is vacuum, and y = +/-L
  are reflecting.

Run from the repository root, for example::

    python DiscreteOrdinates2D/problems/two_opacity_similarity_section2_sn.py \
        --ix 160 --iy 80 --n-quad 4 --dt-max 50
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from DiscreteOrdinates2D.src.quadratures import get_2d_quadrature
from DiscreteOrdinates2D.src.sn_solver_2d import a as A_RAD, ac as AC, c as C_LIGHT, temp_solve_2d


N_OPACITY = 0
BETA = 0.5
GAMMA = 0.5
KAPPA0_1 = 2400.0
KAPPA0_2 = 4800.0
CV = 300.0
T_SOURCE = 1.0
T_INIT = 0.01
T_FLOOR = 0.001
ETA_MAX = 21.0
XI_MAX = 1.231
DEFAULT_L = 2.0


def similarity_diffusion_scale():
    """Return K = 1/A1**2 in cm**2/ns for the equilibrium scaling."""
    return 8.0 * A_RAD * C_LIGHT / (3.0 * 4.0 * KAPPA0_1 * CV)


def similarity_A1():
    return 1.0 / np.sqrt(similarity_diffusion_scale())


def derive_similarity_domain(L=DEFAULT_L):
    A1 = similarity_A1()
    t_final = (A1 * L / ETA_MAX)**2
    x_max = 1.1 * (XI_MAX / ETA_MAX) * L
    return A1, x_max, t_final


def _corner_region(y_faces):
    """Return region masks for the four corners of every x-y cell."""
    y_centers = 0.5 * (y_faces[:-1] + y_faces[1:])
    # Corner order in sn_solver_2d: NE, NW, SW, SE.
    y_corner = np.empty((len(y_centers), 4))
    y_corner[:, 0] = y_centers + 0.25 * np.diff(y_faces)
    y_corner[:, 1] = y_centers + 0.25 * np.diff(y_faces)
    y_corner[:, 2] = y_centers - 0.25 * np.diff(y_faces)
    y_corner[:, 3] = y_centers - 0.25 * np.diff(y_faces)
    return y_corner < 0.0


def _save_snapshot(x, y, temperature, time_ns, path):
    X, Y = np.meshgrid(x, y, indexing="ij")
    fig, ax = plt.subplots(figsize=(9, 4.2))
    image = ax.pcolormesh(X, Y, temperature, shading="auto", cmap="plasma")
    ax.axhline(0.0, color="white", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set(xlabel="x (cm)", ylabel="y (cm)",
           title=f"2-D S_N Section 2: t = {time_ns:.1f} ns")
    fig.colorbar(image, ax=ax, label="T (keV)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(ix=160, iy=80, n_quad=4, dt_min=1e-3, dt_max=50.0,
        L=DEFAULT_L, quad_type="level_symmetric", output_times=None,
        maxits=100, use_dmd=True, output_dir="."):
    """Run the 2-D S_N version and return solver histories and metadata."""
    A1, x_max, t_final = derive_similarity_domain(L)
    x_faces = np.linspace(0.0, x_max, ix + 1)
    y_faces = np.linspace(-L, L, iy + 1)
    dx = np.diff(x_faces)
    dy = np.diff(y_faces)
    x = 0.5 * (x_faces[:-1] + x_faces[1:])
    y = 0.5 * (y_faces[:-1] + y_faces[1:])
    region1 = _corner_region(y_faces)[None, :, :]

    def EOS(T):
        return CV * T

    def invEOS(energy):
        return energy / CV

    def sigma_func(T):
        kappa = np.where(region1, KAPPA0_1, KAPPA0_2)
        return np.broadcast_to(kappa, T.shape).copy()

    def scat_func(T):
        return np.zeros_like(T)

    T_init = np.full((ix, iy, 4), T_INIT)
    phi_init = AC * T_init**4
    q_ext = np.zeros_like(T_init)
    Omega_x, Omega_y, _ = get_2d_quadrature(quad_type, n_quad)
    incoming = np.zeros((iy, len(Omega_x), 2))
    for n, mu_x in enumerate(Omega_x):
        if mu_x > 0.0:
            incoming[:, n, :] = AC * T_SOURCE**4

    def BCs_func(_time):
        return {"xlo": incoming, "xhi": None, "ylo": None, "yhi": None}

    if output_times is None:
        output_times = np.array([0.1, 0.25, 0.5, 1.0]) * t_final
    else:
        output_times = np.asarray(output_times, dtype=float)

    print("Two-opacity similarity, Section 2, 2-D S_N")
    print(f"  beta={BETA:.3f}, gamma={GAMMA:.3f}, N_quad={n_quad}")
    print(f"  domain: [0, {x_max:.6f}] x [-{L}, {L}], cells={ix} x {iy}")
    print(f"  A1={A1:.6f} cm^-1 ns^1/2, t_final={t_final:.3f} ns")

    phis, temperatures, iterations, times, its_per_step = temp_solve_2d(
        ix, iy, dx, dy, q_ext, sigma_func, scat_func,
        quad_type, n_quad, BCs_func, EOS, invEOS,
        phi_init, T_init, dt_min=dt_min, dt_max=dt_max,
        tfinal=t_final, time_outputs=output_times,
        Linf_tol=1e-5, tolerance=1e-8, maxits=maxits, K=50, R=3,
        reflect_xlo=False, reflect_xhi=False,
        reflect_ylo=True, reflect_yhi=True,
        use_dmd=use_dmd, print_stride=10, store_full_history=False,
    )

    os.makedirs(output_dir, exist_ok=True)
    selected = [int(np.argmin(np.abs(times - target))) for target in output_times]
    for target, index in zip(output_times, selected):
        _save_snapshot(x, y, temperatures[index].mean(axis=2), times[index],
                       os.path.join(output_dir, f"T_{times[index]:.0f}ns.png"))

    final_index = selected[-1]
    np.savez(os.path.join(output_dir, f"two_opacity_sect2_sn_{ix}x{iy}.npz"),
             x_centers=x, y_centers=y, phi=phis[final_index],
             T=temperatures[final_index], times=times,
             t_final=t_final, L=L, x_max=x_max, eta_max=ETA_MAX,
             xi_max=XI_MAX, A1=A1, beta=BETA, gamma=GAMMA, n=N_OPACITY,
             kappa0_1=KAPPA0_1, kappa0_2=KAPPA0_2, cv=CV,
             iterations=iterations)
    return {"phis": phis, "temperatures": temperatures, "times": times,
            "iterations": iterations, "its_per_step": its_per_step,
            "x": x, "y": y, "A1": A1, "t_final": t_final}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ix", type=int, default=160)
    parser.add_argument("--iy", type=int, default=80)
    parser.add_argument("--n-quad", type=int, default=4)
    parser.add_argument("--dt-min", type=float, default=1e-3)
    parser.add_argument("--dt-max", type=float, default=50.0)
    parser.add_argument("--L", type=float, default=DEFAULT_L)
    parser.add_argument("--maxits", type=int, default=100)
    parser.add_argument("--no-dmd", action="store_true")
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()
    run(ix=args.ix, iy=args.iy, n_quad=args.n_quad,
        dt_min=args.dt_min, dt_max=args.dt_max, L=args.L,
        maxits=args.maxits, use_dmd=not args.no_dmd,
        output_dir=args.output_dir)