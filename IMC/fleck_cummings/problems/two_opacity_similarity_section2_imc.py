#!/usr/bin/env python3
"""Two-opacity Section 2 similarity problem using 2-D gray IMC.

This is the Fleck-Cummings IMC counterpart to the equilibrium-diffusion
driver.  The material interface is at y=0, with kappa=2400 cm^-1 below and
kappa=4800 cm^-1 above the interface.  Both materials have
Cv=300 GJ/(cm^3 keV).  A 1 keV source enters through x=0; x=x_max is vacuum,
and the y boundaries are reflecting.
"""

import argparse
import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from IMC.fleck_cummings.src import IMC2D as imc2d


N_OPACITY = 0
BETA = 0.5
GAMMA = 0.5
KAPPA0_1 = 2400.0
KAPPA0_2 = 4800.0
CV = 300.0
T_SOURCE = 1.0
T_INIT = 0.01
ETA_MAX = 21.0
XI_MAX = 1.231
DEFAULT_L = 2.0


def similarity_diffusion_scale():
    """Return K=1/A1^2 in cm^2/ns for the equilibrium similarity scale."""
    return 8.0 * imc2d.__a * imc2d.__c / (3.0 * 4.0 * KAPPA0_1 * CV)


def similarity_A1():
    return 1.0 / np.sqrt(similarity_diffusion_scale())


def derive_similarity_domain(L=DEFAULT_L):
    A1 = similarity_A1()
    t_final = (A1 * L / ETA_MAX) ** 2
    x_max = 1.1 * XI_MAX * L / ETA_MAX
    return A1, x_max, t_final


def create_log_spacing_one_sided(start, end, n_cells, from_left=True):
    if n_cells <= 1:
        return [end]
    width = end - start
    growth = 5.0 ** (1.0 / (n_cells - 1))
    first_width = width * (growth - 1.0) / (growth ** n_cells - 1.0)
    cell_widths = [first_width * growth ** index for index in range(n_cells)]
    if not from_left:
        cell_widths.reverse()
    faces = []
    position = start
    for cell_width in cell_widths:
        position += cell_width
        faces.append(position)
    faces[-1] = end
    return faces


def create_log_spacing_around_interface(left, right, interface, n_cells):
    if interface <= left:
        return create_log_spacing_one_sided(left, right, n_cells, from_left=True)
    if interface >= right:
        return create_log_spacing_one_sided(left, right, n_cells, from_left=False)
    n_left = max(1, int(n_cells * (interface - left) / (right - left)))
    n_right = max(1, n_cells - n_left)
    return (
        create_log_spacing_one_sided(left, interface, n_left, from_left=False)
        + create_log_spacing_one_sided(interface, right, n_right, from_left=True)
    )


def generate_refined_faces(domain_min, domain_max, interface_locations,
                           n_refine, n_coarse, refine_width):
    coarse_faces = np.linspace(domain_min, domain_max, n_coarse + 1)
    refine_info = {}
    for interface in sorted(interface_locations):
        for cell_index, (left, right) in enumerate(zip(coarse_faces[:-1], coarse_faces[1:])):
            if left <= interface + refine_width and right >= interface - refine_width:
                center = 0.5 * (left + right)
                if (cell_index not in refine_info
                        or abs(interface - center) < abs(refine_info[cell_index] - center)):
                    refine_info[cell_index] = interface

    face_list = [domain_min]
    for cell_index, (left, right) in enumerate(zip(coarse_faces[:-1], coarse_faces[1:])):
        if cell_index in refine_info:
            face_list.extend(create_log_spacing_around_interface(
                left, right, refine_info[cell_index], n_refine))
        else:
            face_list.append(right)
    return np.asarray(face_list)


def _save_snapshot(x, y, temperature, time_ns, output_dir):
    X, Y = np.meshgrid(x, y, indexing="ij")
    figure, axis = plt.subplots(figsize=(9, 4.2))
    image = axis.pcolormesh(X, Y, temperature, shading="auto", cmap="plasma")
    axis.axhline(0.0, color="white", linestyle="--", linewidth=1.0, alpha=0.7)
    axis.set(
        xlabel="x (cm)",
        ylabel="y (cm)",
        title=f"2-D IMC Section 2: t = {time_ns:.1f} ns",
    )
    figure.colorbar(image, ax=axis, label="T (keV)")
    figure.tight_layout()
    figure.savefig(output_dir / f"two_opacity_sect2_imc_T_{time_ns:.0f}ns.png", dpi=160)
    plt.close(figure)


def _save_similarity_profiles(snapshots, A1, L, output_dir):
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(snapshots)))
    for axis, title, y_target in zip(
        axes, ("Region 1 (y < 0)", "Region 2 (y > 0)"), (-0.5 * L, 0.5 * L)
    ):
        for time_ns, x, y, temperature in snapshots:
            j = int(np.argmin(np.abs(y - y_target)))
            axis.plot(x * A1 / np.sqrt(time_ns), temperature[:, j] / T_SOURCE,
                      color=colors[len(axis.lines)], linewidth=1.8,
                      label=f"t = {time_ns:.0f} ns")
        axis.set(xlabel=r"$\xi=xA_1/\sqrt{t}$", ylabel=r"$T/T_S$", title=title,
                 xlim=(0.0, 1.5), ylim=(0.0, 1.05))
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    figure.suptitle("Two-opacity similarity profiles: gray IMC", fontweight="bold")
    figure.tight_layout()
    figure.savefig(output_dir / "two_opacity_sect2_imc_similarity.png", dpi=160)
    plt.close(figure)


def run(ix=160, iy=60, dt_initial=0.1, dt_max=50.0, dt_increase_factor=1.1,
        L=DEFAULT_L, use_refined_mesh=True,
        n_refine=20, refine_width=0.05, n_target=20_000,
        n_boundary=12_000, n_max=120_000, output_times=None, output_dir="."):
    """Run the Section 2 two-opacity similarity case with gray IMC."""
    if dt_initial <= 0.0 or dt_max <= 0.0:
        raise ValueError("dt_initial and dt_max must be positive")
    if dt_initial > dt_max:
        raise ValueError("dt_initial must not exceed dt_max")
    if dt_increase_factor < 1.0:
        raise ValueError("dt_increase_factor must be at least one")

    A1, x_max, t_final = derive_similarity_domain(L)
    x_faces = np.linspace(0.0, x_max, ix + 1)
    if use_refined_mesh:
        y_faces = generate_refined_faces(-L, L, [0.0], n_refine, iy, refine_width)
    else:
        y_faces = np.linspace(-L, L, iy + 1)
    x = 0.5 * (x_faces[:-1] + x_faces[1:])
    y = 0.5 * (y_faces[:-1] + y_faces[1:])

    region1 = y < 0.0
    kappa_by_y = np.where(region1, KAPPA0_1, KAPPA0_2)

    def eos(temperature):
        return CV * temperature

    def inv_eos(energy):
        return energy / CV

    def cv(temperature):
        return np.full_like(temperature, CV)

    def sigma_a(temperature):
        return np.broadcast_to(kappa_by_y[None, :], temperature.shape).copy()

    if output_times is None:
        output_times = np.array([0.1, 0.25, 0.5, 1.0]) * t_final
    else:
        output_times = np.asarray(output_times, dtype=float)
    output_times = np.sort(output_times)
    if np.any(output_times <= 0.0) or output_times[-1] > t_final + 1e-12:
        raise ValueError("output_times must be positive and no later than t_final")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    shape = (len(x), len(y))
    state = imc2d.init_simulation(
        n_target,
        np.full(shape, T_INIT),
        np.full(shape, T_INIT),
        x_faces,
        y_faces,
        eos,
        inv_eos,
        geometry="xy",
    )

    print("Two-opacity similarity, Section 2, 2-D gray IMC")
    print(f"  beta={BETA:.3f}, gamma={GAMMA:.3f}")
    print(f"  domain: [0, {x_max:.6f}] x [-{L}, {L}], cells={len(x)} x {len(y)}")
    print(f"  A1={A1:.6f} cm^-1 ns^1/2, t_final={t_final:.3f} ns")
    print(f"  time steps: initial={dt_initial:g} ns, max={dt_max:g} ns, "
          f"growth={dt_increase_factor:g}")
    if use_refined_mesh:
        print(f"  refined at y=0: min dy={np.min(np.diff(y_faces)):.3e} cm")

    snapshots = []
    source = np.zeros(shape)
    current_dt = dt_initial
    for target_time in output_times:
        while state.time < target_time - 1e-12:
            step_dt = min(current_dt, target_time - state.time)
            state, info = imc2d.step(
                state, n_target, n_boundary, 0, n_max,
                (T_SOURCE, 0.0, 0.0, 0.0), step_dt,
                x_faces, y_faces, sigma_a, inv_eos, cv, source,
                reflect=(False, False, True, True), geometry="xy",
            )
            current_dt = min(current_dt * dt_increase_factor, dt_max)
        snapshots.append((state.time, x.copy(), y.copy(), state.temperature.copy()))
        _save_snapshot(x, y, state.temperature, state.time, output_path)
        midline = state.temperature[:, int(np.argmin(np.abs(y)))]
        active = np.flatnonzero(midline > 5.0 * T_INIT)
        x_front = x[active[-1]] if active.size else 0.0
        print(f"  t={state.time:9.1f} ns | Tmax={state.temperature.max():.4f} keV "
              f"| xi_front={x_front * A1 / np.sqrt(state.time):.3f} "
              f"| particles={info['N_particles']}")

    _save_similarity_profiles(snapshots, A1, L, output_path)
    np.savez(
        output_path / f"two_opacity_sect2_imc_{ix}x{len(y)}.npz",
        x_centers=x, y_centers=y, T=state.temperature,
        radiation_temperature=state.radiation_temperature,
        t_final=t_final, L=L, x_max=x_max, eta_max=ETA_MAX, xi_max=XI_MAX,
        A1=A1, beta=BETA, gamma=GAMMA, n=N_OPACITY,
        kappa0_1=KAPPA0_1, kappa0_2=KAPPA0_2, cv=CV,
    )
    return {"state": state, "snapshots": snapshots, "x": x, "y": y,
            "x_faces": x_faces, "y_faces": y_faces, "A1": A1,
            "t_final": t_final}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ix", type=int, default=160)
    parser.add_argument("--iy", type=int, default=60)
    parser.add_argument("--dt-initial", type=float, default=0.1)
    parser.add_argument("--dt-max", type=float, default=50.0)
    parser.add_argument("--dt-increase-factor", type=float, default=1.1)
    parser.add_argument("--L", type=float, default=DEFAULT_L)
    parser.add_argument("--n-target", type=int, default=20_000)
    parser.add_argument("--n-boundary", type=int, default=12_000)
    parser.add_argument("--n-max", type=int, default=120_000)
    parser.add_argument("--no-refine", action="store_true")
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()
    run(ix=args.ix, iy=args.iy, dt_initial=args.dt_initial, dt_max=args.dt_max,
        dt_increase_factor=args.dt_increase_factor, L=args.L,
        use_refined_mesh=not args.no_refine, n_target=args.n_target,
        n_boundary=args.n_boundary, n_max=args.n_max, output_dir=args.output_dir)