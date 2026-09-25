#!/usr/bin/env python3
"""Run Gentile-Morel moving-material IMC validation problems.

The shared driver currently implements the moving-equilibrium problem.  Face
flux and Fleck-factor stability subcommands will use the same preset, metadata,
and artifact conventions as they are added.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
import sys
import time

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from MG_IMC.fleck_cummings.src.MG_IMC1DMoving import (  # noqa: E402
    A_RAD,
    C_LIGHT,
    init_simulation,
    seed_moving_imc_random,
    step,
)


@dataclass(frozen=True)
class EquilibriumConfig:
    preset: str
    n_cells: int
    particles: int
    steps: int
    velocities: tuple[float, ...]
    seeds: tuple[int, ...]


EQUILIBRIUM_PRESETS = {
    "reduced": EquilibriumConfig(
        preset="reduced",
        n_cells=20,
        particles=5_000,
        steps=5,
        velocities=(0.0, 0.3, 0.6, 0.9, 0.99),
        seeds=(8101, 8102, 8103),
    ),
    "paper": EquilibriumConfig(
        preset="paper",
        n_cells=100,
        particles=1_000_000,
        steps=10,
        velocities=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999),
        seeds=(8101,),
    ),
}


def _parse_csv_values(text, converter):
    values = tuple(converter(value.strip()) for value in text.split(",") if value.strip())
    if not values:
        raise argparse.ArgumentTypeError("at least one comma-separated value is required")
    return values


def equilibrium_config_from_args(args):
    config = EQUILIBRIUM_PRESETS[args.preset]
    updates = {}
    for name in ("n_cells", "particles", "steps"):
        value = getattr(args, name)
        if value is not None:
            updates[name] = value
    if args.velocities is not None:
        updates["velocities"] = args.velocities
    if args.seeds is not None:
        updates["seeds"] = args.seeds
    config = replace(config, **updates)
    if config.n_cells <= 0 or config.particles <= 0 or config.steps <= 0:
        raise ValueError("n_cells, particles, and steps must be positive")
    if any(beta < 0.0 or beta >= 1.0 for beta in config.velocities):
        raise ValueError("equilibrium velocities must satisfy 0 <= beta < 1")
    if any(seed < 0 or seed >= 2**32 for seed in config.seeds):
        raise ValueError("seeds must lie in [0, 2**32)")
    return config


def run_equilibrium(config, output_path=None):
    """Run the normalized moving-equilibrium sweep and return result arrays."""
    mesh_edges = np.linspace(0.0, 10.0, config.n_cells + 1)
    mesh = np.column_stack((mesh_edges[:-1], mesh_edges[1:]))
    widths = np.diff(mesh_edges)
    middle = config.n_cells // 2
    energy_edges = np.array([0.0, 100.0])
    temperature = np.ones(config.n_cells)
    dt = 1.0 / C_LIGHT
    volumetric_cv = A_RAD * 1.0e51
    opacity_functions = [lambda value: np.full_like(value, 100.0)]

    def eos(value):
        return volumetric_cv * value

    def inv_eos(value):
        return value / volumetric_cv

    def cv(value):
        return np.full_like(value, volumetric_cv)

    shape = (len(config.seeds), len(config.velocities))
    census = np.empty(shape)
    path_length = np.empty(shape)
    fluid_temperature = np.empty(shape)
    maximum_energy_residual = np.empty(shape)
    runtime_seconds = np.empty(shape)

    for seed_index, seed in enumerate(config.seeds):
        for velocity_index, beta in enumerate(config.velocities):
            seed_moving_imc_random(seed)
            velocity = np.zeros((config.n_cells, 3))
            velocity[:, 0] = beta * C_LIGHT
            state = init_simulation(
                config.particles,
                temperature,
                temperature,
                mesh,
                energy_edges,
                velocity,
                eos=eos,
                inv_eos=inv_eos,
            )
            residuals = []
            started = time.perf_counter()
            for _ in range(config.steps):
                state, info = step(
                    state,
                    target=config.particles,
                    dt=dt,
                    mesh=mesh,
                    energy_edges=energy_edges,
                    sigma_a_funcs=opacity_functions,
                    inv_eos=inv_eos,
                    cv=cv,
                    population_target=config.particles,
                )
                residuals.append(abs(info["exchange_energy_residual"]))
            runtime_seconds[seed_index, velocity_index] = time.perf_counter() - started

            census[seed_index, velocity_index] = (
                state.radiation_energy_lab[middle] / widths[middle] / A_RAD
            )
            path_length[seed_index, velocity_index] = (
                np.sum(info["track_length_energy_lab_by_fluid_group"][:, middle])
                / (C_LIGHT * dt * widths[middle] * A_RAD)
            )
            fluid_temperature[seed_index, velocity_index] = (
                state.radiation_temperature[middle]
            )
            maximum_energy_residual[seed_index, velocity_index] = max(residuals)

    velocities = np.asarray(config.velocities)
    gamma = 1.0 / np.sqrt(1.0 - velocities**2)
    analytic = gamma**2 * (1.0 + velocities**2 / 3.0)
    result = {
        "problem": np.array("moving_equilibrium"),
        "preset": np.array(config.preset),
        "n_cells": np.int64(config.n_cells),
        "particles_per_step": np.int64(config.particles),
        "steps": np.int64(config.steps),
        "seeds": np.asarray(config.seeds, dtype=np.int64),
        "velocities_beta": velocities,
        "dt_code": np.float64(dt),
        "middle_cell": np.int64(middle),
        "analytic_lab_energy_density_normalized": analytic,
        "census_lab_energy_density_normalized": census,
        "path_lab_energy_density_normalized": path_length,
        "fluid_radiation_temperature": fluid_temperature,
        "maximum_absolute_exchange_energy_residual": maximum_energy_residual,
        "runtime_seconds": runtime_seconds,
    }
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, **result)
    return result


def _print_equilibrium_summary(result, output_path):
    analytic = result["analytic_lab_energy_density_normalized"]
    census = result["census_lab_energy_density_normalized"]
    path = result["path_lab_energy_density_normalized"]
    temperature = result["fluid_radiation_temperature"]
    print("beta    analytic      census mean   path mean     Tr,F mean")
    for index, beta in enumerate(result["velocities_beta"]):
        print(
            f"{beta:5.3f}  {analytic[index]:12.6g}  "
            f"{np.mean(census[:, index]):12.6g}  "
            f"{np.mean(path[:, index]):12.6g}  "
            f"{np.mean(temperature[:, index]):10.6g}"
        )
    census_error = np.max(np.abs(np.mean(census, axis=0) / analytic - 1.0))
    path_error = np.max(np.abs(np.mean(path, axis=0) / analytic - 1.0))
    temperature_error = np.max(np.abs(np.mean(temperature, axis=0) - 1.0))
    print(f"maximum mean census fractional error: {census_error:.6e}")
    print(f"maximum mean path fractional error:   {path_error:.6e}")
    print(f"maximum mean fluid-T absolute error:  {temperature_error:.6e}")
    print(f"artifact: {output_path}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run published moving-material IMC validation problems."
    )
    subparsers = parser.add_subparsers(dest="problem", required=True)
    equilibrium = subparsers.add_parser("equilibrium", help="Run the moving-equilibrium sweep.")
    equilibrium.add_argument("--preset", choices=tuple(EQUILIBRIUM_PRESETS), default="reduced")
    equilibrium.add_argument("--n-cells", dest="n_cells", type=int)
    equilibrium.add_argument("--particles", type=int)
    equilibrium.add_argument("--steps", type=int)
    equilibrium.add_argument(
        "--velocities", type=lambda text: _parse_csv_values(text, float)
    )
    equilibrium.add_argument("--seeds", type=lambda text: _parse_csv_values(text, int))
    equilibrium.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.problem == "equilibrium":
        config = equilibrium_config_from_args(args)
        output = args.output
        if output is None:
            output = (
                REPOSITORY_ROOT
                / "results"
                / "moving_material_published"
                / "equilibrium"
                / config.preset
                / "results.npz"
            )
        result = run_equilibrium(config, output)
        _print_equilibrium_summary(result, output)
        return 0
    raise ValueError(f"unsupported problem {args.problem}")


if __name__ == "__main__":
    raise SystemExit(main())
