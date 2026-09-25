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


def _atomic_savez(output_path, result):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.npz")
    np.savez_compressed(temporary, **result)
    temporary.replace(output)


def _validate_equilibrium_resume(saved, config):
    expected = {
        "problem": "moving_equilibrium",
        "n_cells": config.n_cells,
        "particles_per_step": config.particles,
        "steps": config.steps,
    }
    for key, value in expected.items():
        if key not in saved or saved[key].item() != value:
            raise ValueError(f"resume artifact {key} does not match requested configuration")
    if not np.array_equal(saved["velocities_beta"], config.velocities):
        raise ValueError("resume artifact velocities do not match requested configuration")
    if not np.array_equal(saved["seeds"], config.seeds):
        raise ValueError("resume artifact seeds do not match requested configuration")


def run_equilibrium(config, output_path=None, resume=False):
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
    cell_shape = shape + (config.n_cells,)
    census = np.full(shape, np.nan)
    path_length = np.full(shape, np.nan)
    fluid_temperature = np.full(shape, np.nan)
    census_by_cell = np.full(cell_shape, np.nan)
    path_length_by_cell = np.full(cell_shape, np.nan)
    fluid_temperature_by_cell = np.full(cell_shape, np.nan)
    maximum_energy_residual = np.full(shape, np.nan)
    runtime_seconds = np.full(shape, np.nan)
    completed_cases = np.zeros(shape, dtype=bool)

    velocities = np.asarray(config.velocities)
    gamma = 1.0 / np.sqrt(1.0 - velocities**2)
    analytic = gamma**2 * (1.0 + velocities**2 / 3.0)

    def assemble_result():
        return {
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
            "census_lab_energy_density_normalized_by_cell": census_by_cell,
            "path_lab_energy_density_normalized_by_cell": path_length_by_cell,
            "fluid_radiation_temperature_by_cell": fluid_temperature_by_cell,
            "maximum_absolute_exchange_energy_residual": maximum_energy_residual,
            "runtime_seconds": runtime_seconds,
            "completed_cases": completed_cases,
        }

    output = Path(output_path) if output_path is not None else None
    if resume:
        if output is None:
            raise ValueError("resume requires an output path")
        if output.exists():
            with np.load(output) as saved_file:
                saved = {key: saved_file[key] for key in saved_file.files}
            _validate_equilibrium_resume(saved, config)
            for key, array in (
                ("census_lab_energy_density_normalized", census),
                ("path_lab_energy_density_normalized", path_length),
                ("fluid_radiation_temperature", fluid_temperature),
                ("census_lab_energy_density_normalized_by_cell", census_by_cell),
                ("path_lab_energy_density_normalized_by_cell", path_length_by_cell),
                ("fluid_radiation_temperature_by_cell", fluid_temperature_by_cell),
                ("maximum_absolute_exchange_energy_residual", maximum_energy_residual),
                ("runtime_seconds", runtime_seconds),
            ):
                array[...] = saved[key]
            completed_cases[...] = saved["completed_cases"]

    for seed_index, seed in enumerate(config.seeds):
        for velocity_index, beta in enumerate(config.velocities):
            if completed_cases[seed_index, velocity_index]:
                print(f"skipping completed equilibrium case seed={seed}, beta={beta:g}", flush=True)
                continue
            print(f"running equilibrium case seed={seed}, beta={beta:g}", flush=True)
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

            census_cells = state.radiation_energy_lab / widths / A_RAD
            path_length_cells = (
                np.sum(info["track_length_energy_lab_by_fluid_group"], axis=0)
                / (C_LIGHT * dt * widths * A_RAD)
            )
            census_by_cell[seed_index, velocity_index] = census_cells
            path_length_by_cell[seed_index, velocity_index] = path_length_cells
            fluid_temperature_by_cell[seed_index, velocity_index] = (
                state.radiation_temperature
            )
            census[seed_index, velocity_index] = census_cells[middle]
            path_length[seed_index, velocity_index] = path_length_cells[middle]
            fluid_temperature[seed_index, velocity_index] = state.radiation_temperature[middle]
            maximum_energy_residual[seed_index, velocity_index] = max(residuals)
            completed_cases[seed_index, velocity_index] = True
            if output is not None:
                _atomic_savez(output, assemble_result())

    result = assemble_result()
    if output is not None:
        _atomic_savez(output, result)
    return result


def run_equilibrium_convergence(config, particle_counts, output_path=None):
    """Run a particle-count sweep and collect cellwise convergence diagnostics."""
    counts = np.asarray(tuple(particle_counts), dtype=np.int64)
    if counts.ndim != 1 or counts.size < 2 or np.any(counts <= 0):
        raise ValueError("particle_counts must contain at least two positive values")
    if np.any(np.diff(counts) <= 0):
        raise ValueError("particle_counts must be strictly increasing")

    runs = []
    for particles in counts:
        print(f"running equilibrium convergence case with {particles:,} particles per step", flush=True)
        runs.append(run_equilibrium(replace(config, particles=int(particles))))

    stacked_keys = (
        "census_lab_energy_density_normalized",
        "path_lab_energy_density_normalized",
        "fluid_radiation_temperature",
        "census_lab_energy_density_normalized_by_cell",
        "path_lab_energy_density_normalized_by_cell",
        "fluid_radiation_temperature_by_cell",
        "maximum_absolute_exchange_energy_residual",
        "runtime_seconds",
    )
    result = {
        key: value
        for key, value in runs[0].items()
        if key not in stacked_keys and key != "particles_per_step"
    }
    result["problem"] = np.array("moving_equilibrium_convergence")
    result["particle_counts"] = counts
    for key in stacked_keys:
        result[key] = np.stack([run[key] for run in runs])

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, **result)
    return result


def equilibrium_convergence_metrics(result):
    """Return central-cell error statistics and particle-count noise slopes."""
    counts = result["particle_counts"]
    analytic = result["analytic_lab_energy_density_normalized"]
    n_cells = int(result["n_cells"])
    central = slice(n_cells // 4, n_cells - n_cells // 4)
    census = result["census_lab_energy_density_normalized_by_cell"][..., central]
    path = result["path_lab_energy_density_normalized_by_cell"][..., central]
    temperature = result["fluid_radiation_temperature_by_cell"][..., central]

    census_relative = census / analytic[None, None, :, None] - 1.0
    path_relative = path / analytic[None, None, :, None] - 1.0
    census_rms = np.sqrt(np.mean(census_relative**2, axis=(1, 3)))
    path_rms = np.sqrt(np.mean(path_relative**2, axis=(1, 3)))
    temperature_rms = np.sqrt(np.mean((temperature - 1.0) ** 2, axis=(1, 3)))

    census_noise = np.std(census_relative, axis=(1, 3), ddof=1)
    slopes = np.array(
        [
            np.polyfit(np.log(counts), np.log(census_noise[:, velocity]), 1)[0]
            for velocity in range(analytic.size)
        ]
    )
    return {
        "census_relative_rms_by_velocity": census_rms,
        "path_relative_rms_by_velocity": path_rms,
        "fluid_temperature_rms_by_velocity": temperature_rms,
        "census_relative_noise_by_velocity": census_noise,
        "census_noise_slope_by_velocity": slopes,
    }


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


def _print_equilibrium_convergence_summary(result, output_path):
    metrics = equilibrium_convergence_metrics(result)
    print("particles   max census RMS   max path RMS   max fluid-T RMS")
    for index, particles in enumerate(result["particle_counts"]):
        print(
            f"{particles:9d}   "
            f"{np.max(metrics['census_relative_rms_by_velocity'][index]):14.6e}   "
            f"{np.max(metrics['path_relative_rms_by_velocity'][index]):12.6e}   "
            f"{np.max(metrics['fluid_temperature_rms_by_velocity'][index]):15.6e}"
        )
    slopes = metrics["census_noise_slope_by_velocity"]
    print("census noise slopes by velocity: " + ", ".join(f"{value:.3f}" for value in slopes))
    print(f"median census noise slope: {np.median(slopes):.3f}")
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
    equilibrium.add_argument(
        "--resume",
        action="store_true",
        help="Resume completed seed/velocity cases from the output artifact.",
    )
    equilibrium.add_argument("--seeds", type=lambda text: _parse_csv_values(text, int))
    equilibrium.add_argument(
        "--particle-counts",
        type=lambda text: _parse_csv_values(text, int),
        help="Run a convergence sweep over increasing particle counts.",
    )
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
        if args.particle_counts is None:
            result = run_equilibrium(config, output, resume=args.resume)
            _print_equilibrium_summary(result, output)
        else:
            if args.output is None:
                output = (
                    REPOSITORY_ROOT
                    / "results"
                    / "moving_material_published"
                    / "equilibrium"
                    / "convergence"
                    / "results.npz"
                )
            result = run_equilibrium_convergence(config, args.particle_counts, output)
            _print_equilibrium_convergence_summary(result, output)
        return 0
    raise ValueError(f"unsupported problem {args.problem}")


if __name__ == "__main__":
    raise SystemExit(main())
