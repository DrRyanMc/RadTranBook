"""Moving-material multigroup IMC in one-dimensional slab geometry.

This module provides the moving state/frame contract, equilibrium census,
material volume source, and slab transport with lab-frame energy-momentum
exchange tallies.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np

try:
    from .MG_IMC1D import (
        _compute_Bg_1d,
        _sample_planck_spectrum_mg,
        _seed_numba_random,
    )
    from .material_motion import (
        C_LIGHT,
        fluid_group_from_lab_photon,
        lab_energy_momentum_to_fluid,
        lab_groups_from_energies,
        precompute_velocity_factors,
        sample_fluid_directions,
        transform_fluid_particles_to_lab,
        validate_material_velocity,
    )
except ImportError:
    from MG_IMC1D import _compute_Bg_1d, _sample_planck_spectrum_mg, _seed_numba_random
    from material_motion import (
        C_LIGHT,
        fluid_group_from_lab_photon,
        lab_energy_momentum_to_fluid,
        lab_groups_from_energies,
        precompute_velocity_factors,
        sample_fluid_directions,
        transform_fluid_particles_to_lab,
        validate_material_velocity,
    )


A_RAD = 0.01372  # GJ / cm^3 / keV^4


def seed_moving_imc_random(seed):
    """Seed every random stream used by the moving-material solver."""
    if (
        isinstance(seed, (bool, np.bool_))
        or not isinstance(seed, (int, np.integer))
        or seed < 0
        or seed >= 2**32
    ):
        raise ValueError("seed must be an integer in [0, 2**32)")
    integer_seed = int(seed)
    np.random.seed(integer_seed)
    _seed_numba_random(integer_seed)


@dataclass
class SimulationState1DMovingMG:
    """State shared by moving-material slab IMC time steps.

    Particle quantities are lab-frame values.  Material temperature and
    internal energy are fluid-frame thermodynamic quantities.  The cached lab
    group is diagnostic only; transport must derive its fluid interaction
    group from continuous photon energy on every segment.
    """

    weights: np.ndarray
    dir_x: np.ndarray
    dir_y: np.ndarray
    dir_z: np.ndarray
    photon_energies_lab: np.ndarray
    lab_groups: np.ndarray
    positions: np.ndarray
    times: np.ndarray
    cell_indices: np.ndarray

    internal_energy: np.ndarray
    temperature: np.ndarray
    material_velocity: np.ndarray

    radiation_energy_lab: np.ndarray
    radiation_momentum_lab: np.ndarray
    radiation_energy_fluid: np.ndarray
    radiation_momentum_fluid: np.ndarray
    radiation_temperature: np.ndarray

    time: float
    previous_total_energy: float
    count: int = 0


@dataclass
class MovingParticleBatch:
    """A lab-frame particle batch plus exact or sampled cell moments."""

    weights: np.ndarray
    dir_x: np.ndarray
    dir_y: np.ndarray
    dir_z: np.ndarray
    photon_energies_lab: np.ndarray
    lab_groups: np.ndarray
    positions: np.ndarray
    times: np.ndarray
    cell_indices: np.ndarray
    energy_lab_by_cell: np.ndarray
    momentum_lab_by_cell: np.ndarray


@dataclass
class MovingTransportResult:
    """Tallies and event counts from one lab-frame transport interval."""

    material_energy_exchange_lab: np.ndarray
    material_momentum_exchange_lab: np.ndarray
    track_length_energy_lab_by_fluid_group: np.ndarray
    boundary_energy_loss_lab: np.ndarray
    boundary_momentum_loss_lab: np.ndarray
    wall_momentum_exchange_lab: np.ndarray
    event_counts: np.ndarray


@dataclass
class FaceAngularTable:
    """Numerical inverse-CDF table for the moving blackbody face law."""

    mu_grid: np.ndarray
    mu_cdf: np.ndarray
    phi_grid: np.ndarray
    phi_cdf_by_mu: np.ndarray
    beta_tangent: float
    beta_normal: float
    gamma: float
    flux_factor: float
    normalization_integral: float


@dataclass
class PopulationControlResult:
    """Energy-conserving census resampling diagnostics."""

    particle_count_before: int
    particle_count_after: int
    energy_before: float
    energy_after: float
    energy_change: float
    momentum_change_lab: np.ndarray
    momentum_change_lab_by_cell: np.ndarray


def _validate_mesh(mesh):
    mesh_array = np.asarray(mesh, dtype=np.float64)
    if mesh_array.ndim != 2 or mesh_array.shape[1] != 2:
        raise ValueError("mesh must have shape (n_cells, 2)")
    if not np.all(np.isfinite(mesh_array)):
        raise ValueError("mesh must contain only finite values")
    if np.any(mesh_array[:, 1] <= mesh_array[:, 0]):
        raise ValueError("every mesh cell must have positive width")
    if len(mesh_array) > 1 and not np.allclose(mesh_array[:-1, 1], mesh_array[1:, 0]):
        raise ValueError("mesh cells must be contiguous")
    return np.ascontiguousarray(mesh_array)


def _validate_energy_edges(energy_edges):
    edges = np.asarray(energy_edges, dtype=np.float64)
    if (
        edges.ndim != 1
        or len(edges) < 2
        or not np.all(np.isfinite(edges))
        or np.any(np.diff(edges) <= 0.0)
    ):
        raise ValueError("energy_edges must be a finite, strictly increasing 1-D array")
    return np.ascontiguousarray(edges)


def _validate_sampling_controls(target, temperature_floor):
    if isinstance(target, (bool, np.bool_)) or not isinstance(target, (int, np.integer)):
        raise ValueError("target must be a nonnegative integer")
    if target < 0:
        raise ValueError("target must be a nonnegative integer")
    if not np.isfinite(temperature_floor) or temperature_floor < 0.0:
        raise ValueError("temperature_floor must be finite and nonnegative")


def create_state_from_particles(
    weights,
    directions_lab,
    photon_energies_lab,
    positions,
    times,
    cell_indices,
    temperature,
    material_velocity,
    mesh,
    energy_edges,
    eos,
    time=0.0,
):
    """Build a validated moving-material state from lab-frame particles.

    This explicit constructor supports kinematics and transport tests.  Use
    :func:`init_simulation` when the starting census is moving equilibrium.
    """
    mesh_array = _validate_mesh(mesh)
    n_cells = len(mesh_array)
    velocity = validate_material_velocity(
        material_velocity, n_cells=n_cells, c=C_LIGHT
    )

    particle_weights = np.asarray(weights, dtype=np.float64)
    directions = np.asarray(directions_lab, dtype=np.float64)
    photon_energies = np.asarray(photon_energies_lab, dtype=np.float64)
    particle_positions = np.asarray(positions, dtype=np.float64)
    particle_times = np.asarray(times, dtype=np.float64)
    cells = np.asarray(cell_indices, dtype=np.int64)
    material_temperature = np.asarray(temperature, dtype=np.float64)

    if particle_weights.ndim != 1:
        raise ValueError("weights must be one-dimensional")
    n_particles = len(particle_weights)
    if directions.shape != (n_particles, 3):
        raise ValueError("directions_lab must have shape (n_particles, 3)")
    for name, values in (
        ("photon_energies_lab", photon_energies),
        ("positions", particle_positions),
        ("times", particle_times),
        ("cell_indices", cells),
    ):
        if values.ndim != 1 or len(values) != n_particles:
            raise ValueError(f"{name} must be one-dimensional with n_particles entries")
    if material_temperature.shape != (n_cells,):
        raise ValueError("temperature must have one entry per cell")
    for name, values in (
        ("weights", particle_weights),
        ("photon_energies_lab", photon_energies),
        ("positions", particle_positions),
        ("times", particle_times),
        ("temperature", material_temperature),
    ):
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain only finite values")
    if not np.isfinite(time):
        raise ValueError("time must be finite")
    if np.any(material_temperature < 0.0):
        raise ValueError("temperature must be nonnegative")
    if not np.all(np.isfinite(directions)):
        raise ValueError("directions_lab must contain only finite values")
    direction_norms = np.sqrt(np.sum(directions * directions, axis=1))
    if not np.allclose(direction_norms, 1.0, rtol=1.0e-12, atol=1.0e-12):
        raise ValueError("every lab-frame direction must be a unit vector")
    if np.any(particle_weights < 0.0) or np.any(photon_energies <= 0.0):
        raise ValueError("particle weights must be nonnegative and photon energies positive")
    if np.any(cells < 0) or np.any(cells >= n_cells):
        raise ValueError("cell_indices contain an out-of-range cell")
    if n_particles:
        cell_left = mesh_array[cells, 0]
        cell_right = mesh_array[cells, 1]
        if np.any(particle_positions < cell_left) or np.any(particle_positions > cell_right):
            raise ValueError("particle position lies outside its indexed cell")

    internal_energy = np.asarray(eos(material_temperature), dtype=np.float64)
    if internal_energy.shape != (n_cells,):
        raise ValueError("eos(temperature) must return one internal-energy value per cell")
    if not np.all(np.isfinite(internal_energy)):
        raise ValueError("eos(temperature) must return only finite values")

    edges = _validate_energy_edges(energy_edges)
    lab_groups = lab_groups_from_energies(photon_energies, edges)
    radiation_energy_lab = np.bincount(
        cells, weights=particle_weights, minlength=n_cells
    ).astype(np.float64)
    radiation_momentum_lab = np.zeros((n_cells, 3), dtype=np.float64)
    for component in range(3):
        radiation_momentum_lab[:, component] = np.bincount(
            cells,
            weights=particle_weights * directions[:, component] / C_LIGHT,
            minlength=n_cells,
        )

    radiation_energy_fluid, radiation_momentum_fluid = lab_energy_momentum_to_fluid(
        radiation_energy_lab, radiation_momentum_lab, velocity, c=C_LIGHT
    )
    _, gamma = precompute_velocity_factors(velocity, n_cells=n_cells, c=C_LIGHT)
    cell_widths = mesh_array[:, 1] - mesh_array[:, 0]
    fluid_volumes = gamma * cell_widths
    radiation_temperature = np.maximum(
        radiation_energy_fluid / fluid_volumes / A_RAD, 0.0
    ) ** 0.25

    previous_total_energy = float(
        np.sum(internal_energy * cell_widths) + np.sum(particle_weights)
    )

    return SimulationState1DMovingMG(
        weights=np.ascontiguousarray(particle_weights.copy()),
        dir_x=np.ascontiguousarray(directions[:, 0].copy()),
        dir_y=np.ascontiguousarray(directions[:, 1].copy()),
        dir_z=np.ascontiguousarray(directions[:, 2].copy()),
        photon_energies_lab=np.ascontiguousarray(photon_energies.copy()),
        lab_groups=np.ascontiguousarray(lab_groups),
        positions=np.ascontiguousarray(particle_positions.copy()),
        times=np.ascontiguousarray(particle_times.copy()),
        cell_indices=np.ascontiguousarray(cells.copy()),
        internal_energy=np.ascontiguousarray(internal_energy.copy()),
        temperature=np.ascontiguousarray(material_temperature.copy()),
        material_velocity=velocity,
        radiation_energy_lab=radiation_energy_lab,
        radiation_momentum_lab=radiation_momentum_lab,
        radiation_energy_fluid=radiation_energy_fluid,
        radiation_momentum_fluid=radiation_momentum_fluid,
        radiation_temperature=radiation_temperature,
        time=float(time),
        previous_total_energy=previous_total_energy,
    )


def _particle_counts_by_cell(target, energy_by_cell):
    counts = np.zeros(len(energy_by_cell), dtype=np.int64)
    total = float(np.sum(energy_by_cell))
    if target <= 0 or total <= 0.0:
        return counts
    active = energy_by_cell > 0.0
    counts[active] = np.ceil(target * energy_by_cell[active] / total).astype(np.int64)
    return counts


def _sample_opacity_weighted_planck(n, temperature, energy_edges, opacity_by_group, b_fraction):
    """Sample Planck photon energies with piecewise-constant opacity weighting."""
    if n <= 0:
        return np.empty(0), np.empty(0, dtype=np.int32)
    opacity = np.asarray(opacity_by_group, dtype=np.float64)
    maximum_opacity = float(np.max(opacity))
    weighted_mean = float(np.sum(opacity * b_fraction))
    if maximum_opacity <= 0.0 or weighted_mean <= 0.0:
        raise ValueError("opacity-weighted Planck sampling needs positive opacity")
    expected_acceptance = weighted_mean / maximum_opacity
    maximum_draws = max(10_000, int(math.ceil(20.0 * n / expected_acceptance)))

    energies = np.empty(n, dtype=np.float64)
    groups = np.empty(n, dtype=np.int32)
    filled = 0
    draws = 0
    while filled < n:
        remaining = n - filled
        batch_size = max(128, int(math.ceil(1.25 * remaining / expected_acceptance)))
        batch_size = min(batch_size, maximum_draws - draws)
        if batch_size <= 0:
            raise RuntimeError("opacity-weighted Planck rejection sampler did not converge")
        candidate_energy, candidate_group = _sample_planck_spectrum_mg(
            batch_size, float(temperature), energy_edges
        )
        acceptance = opacity[candidate_group] / maximum_opacity
        keep = np.random.random(batch_size) < acceptance
        accepted_energy = candidate_energy[keep]
        accepted_group = candidate_group[keep]
        take = min(remaining, len(accepted_energy))
        if take:
            energies[filled:filled + take] = accepted_energy[:take]
            groups[filled:filled + take] = accepted_group[:take]
            filled += take
        draws += batch_size
    return energies, groups


def _assemble_batch(parts, n_cells, energy_edges):
    if not parts:
        empty = np.empty(0, dtype=np.float64)
        return MovingParticleBatch(
            empty, empty, empty, empty, empty, np.empty(0, dtype=np.int32),
            empty, empty, np.empty(0, dtype=np.int64), np.zeros(n_cells),
            np.zeros((n_cells, 3)),
        )

    weights = np.concatenate([part[0] for part in parts])
    directions = np.concatenate([part[1] for part in parts], axis=0)
    energies = np.concatenate([part[2] for part in parts])
    positions = np.concatenate([part[3] for part in parts])
    times = np.concatenate([part[4] for part in parts])
    cells = np.concatenate([part[5] for part in parts])
    energy_by_cell = np.bincount(cells, weights=weights, minlength=n_cells)
    momentum_by_cell = np.zeros((n_cells, 3), dtype=np.float64)
    for component in range(3):
        momentum_by_cell[:, component] = np.bincount(
            cells,
            weights=weights * directions[:, component] / C_LIGHT,
            minlength=n_cells,
        )
    return MovingParticleBatch(
        weights=np.ascontiguousarray(weights),
        dir_x=np.ascontiguousarray(directions[:, 0]),
        dir_y=np.ascontiguousarray(directions[:, 1]),
        dir_z=np.ascontiguousarray(directions[:, 2]),
        photon_energies_lab=np.ascontiguousarray(energies),
        lab_groups=np.ascontiguousarray(lab_groups_from_energies(energies, energy_edges)),
        positions=np.ascontiguousarray(positions),
        times=np.ascontiguousarray(times),
        cell_indices=np.ascontiguousarray(cells),
        energy_lab_by_cell=energy_by_cell,
        momentum_lab_by_cell=momentum_by_cell,
    )


def _append_particle_batch(state, batch):
    """Append a source batch to the lab-frame particle arrays in ``state``."""
    state.weights = np.concatenate((state.weights, batch.weights))
    state.dir_x = np.concatenate((state.dir_x, batch.dir_x))
    state.dir_y = np.concatenate((state.dir_y, batch.dir_y))
    state.dir_z = np.concatenate((state.dir_z, batch.dir_z))
    state.photon_energies_lab = np.concatenate(
        (state.photon_energies_lab, batch.photon_energies_lab)
    )
    state.lab_groups = np.concatenate((state.lab_groups, batch.lab_groups))
    state.positions = np.concatenate((state.positions, batch.positions))
    state.times = np.concatenate((state.times, batch.times))
    state.cell_indices = np.concatenate((state.cell_indices, batch.cell_indices))


def _radiation_moments(weights, directions, cells, n_cells):
    energy = np.bincount(cells, weights=weights, minlength=n_cells).astype(np.float64)
    momentum = np.zeros((n_cells, 3), dtype=np.float64)
    for component in range(3):
        momentum[:, component] = np.bincount(
            cells,
            weights=weights * directions[:, component] / C_LIGHT,
            minlength=n_cells,
        )
    return energy, momentum


def moving_face_flux_factor(beta_tangent, beta_normal):
    """Return the dimensionless factor in the moving face-flux formula."""
    beta_tangent = float(beta_tangent)
    beta_normal = float(beta_normal)
    if not np.isfinite(beta_tangent) or not np.isfinite(beta_normal):
        raise ValueError("face velocity components must be finite")
    if beta_tangent < 0.0 or beta_tangent**2 + beta_normal**2 >= 1.0:
        raise ValueError("face velocity components must satisfy beta_t >= 0 and |beta| < 1")
    transverse_root = math.sqrt(1.0 - beta_tangent**2)
    scaled_normal = beta_normal / transverse_root
    # Factored form of 1 + 8x/3 + 2x^2 - x^4/3 avoids cancellation at x -> -1.
    factor = (
        transverse_root * (1.0 + scaled_normal) ** 3
        * (3.0 - scaled_normal) / 3.0
    )
    if factor <= 0.0:
        raise ValueError("moving face flux factor is not positive")
    return factor


def build_moving_face_angular_table(
    beta_tangent,
    beta_normal,
    mu_points=1025,
    phi_points=2049,
):
    """Tabulate the normalized face density ``mu / D_L**4``.

    The returned table omits the analytic normalization factor from its stored
    CDFs but records the numerical integral for quadrature-convergence tests.
    """
    if (
        isinstance(mu_points, (bool, np.bool_))
        or isinstance(phi_points, (bool, np.bool_))
        or not isinstance(mu_points, (int, np.integer))
        or not isinstance(phi_points, (int, np.integer))
        or mu_points < 3
        or phi_points < 3
        or phi_points % 2 == 0
    ):
        raise ValueError("mu_points >= 3 and odd phi_points >= 3 are required")
    beta_tangent = float(beta_tangent)
    beta_normal = float(beta_normal)
    flux_factor = moving_face_flux_factor(beta_tangent, beta_normal)
    beta_squared = beta_tangent**2 + beta_normal**2
    gamma = 1.0 / math.sqrt(1.0 - beta_squared)

    mu_parameter = np.linspace(0.0, 1.0, int(mu_points))
    mu_left = mu_parameter**4
    mu_right = (1.0 - mu_parameter)**4
    mu_grid = mu_left / (mu_left + mu_right)
    phi_grid = np.linspace(-np.pi, np.pi, int(phi_points))
    transverse = np.sqrt(np.maximum(0.0, 1.0 - mu_grid**2))
    doppler_lab = (
        1.0
        - beta_normal * mu_grid[:, None]
        - beta_tangent * transverse[:, None] * np.cos(phi_grid)[None, :]
    )
    angular_factor = doppler_lab**-4
    dphi = phi_grid[1] - phi_grid[0]
    phi_increments = 0.5 * (
        angular_factor[:, :-1] + angular_factor[:, 1:]
    ) * dphi
    phi_cdf = np.zeros_like(angular_factor)
    phi_cdf[:, 1:] = np.cumsum(phi_increments, axis=1)
    phi_integral = phi_cdf[:, -1].copy()
    phi_cdf /= phi_integral[:, None]

    mu_density = mu_grid * phi_integral
    mu_cdf = np.zeros_like(mu_grid)
    mu_cdf[1:] = np.cumsum(
        0.5 * (mu_density[:-1] + mu_density[1:]) * np.diff(mu_grid)
    )
    normalization = float(mu_cdf[-1])
    mu_cdf /= normalization
    return FaceAngularTable(
        mu_grid=mu_grid,
        mu_cdf=mu_cdf,
        phi_grid=phi_grid,
        phi_cdf_by_mu=phi_cdf,
        beta_tangent=beta_tangent,
        beta_normal=beta_normal,
        gamma=gamma,
        flux_factor=flux_factor,
        normalization_integral=normalization,
    )


@lru_cache(maxsize=4)
def _get_moving_face_angular_table(
    beta_tangent,
    beta_normal,
    mu_points=1025,
    phi_points=2049,
):
    return build_moving_face_angular_table(
        float(beta_tangent),
        float(beta_normal),
        int(mu_points),
        int(phi_points),
    )


def _sample_face_local_angles(table, count):
    mu = np.interp(np.random.random(count), table.mu_cdf, table.mu_grid)
    row = np.searchsorted(table.mu_grid, mu)
    row = np.clip(row, 1, len(table.mu_grid) - 1)
    choose_left = (
        mu - table.mu_grid[row - 1] <= table.mu_grid[row] - mu
    )
    row[choose_left] -= 1
    phi = np.empty(count, dtype=np.float64)
    phi_random = np.random.random(count)
    for row_index in np.unique(row):
        selected = row == row_index
        phi[selected] = np.interp(
            phi_random[selected],
            table.phi_cdf_by_mu[row_index],
            table.phi_grid,
        )
    return mu, phi


def sample_moving_equilibrium_particles(
    target,
    radiation_temperature,
    mesh,
    energy_edges,
    material_velocity,
    temperature_floor=0.0,
):
    """Sample an equal-lab-weight moving blackbody census in each slab cell."""
    _validate_sampling_controls(target, temperature_floor)
    mesh_array = _validate_mesh(mesh)
    edges = _validate_energy_edges(energy_edges)
    n_cells = len(mesh_array)
    velocity = validate_material_velocity(material_velocity, n_cells=n_cells, c=C_LIGHT)
    beta, gamma = precompute_velocity_factors(velocity, n_cells=n_cells, c=C_LIGHT)
    temperature = np.asarray(radiation_temperature, dtype=np.float64)
    if temperature.shape != (n_cells,) or not np.all(np.isfinite(temperature)):
        raise ValueError("radiation_temperature must be finite with one value per cell")
    if np.any(temperature < 0.0):
        raise ValueError("radiation_temperature must be nonnegative")

    widths = mesh_array[:, 1] - mesh_array[:, 0]
    beta_squared = np.sum(beta * beta, axis=1)
    energy_by_cell = (
        A_RAD * temperature**4 * gamma**2 * (1.0 + beta_squared / 3.0) * widths
    )
    if temperature_floor > 0.0:
        energy_by_cell[temperature < temperature_floor] = 0.0
    counts = _particle_counts_by_cell(target, energy_by_cell)

    parts = []
    for cell, count in enumerate(counts):
        if count <= 0:
            continue
        energies_fluid, _ = _sample_planck_spectrum_mg(
            int(count), float(temperature[cell]), edges
        )
        directions_fluid = sample_fluid_directions(int(count), beta[cell], doppler_power=2)
        energies_lab, directions_lab = transform_fluid_particles_to_lab(
            energies_fluid, directions_fluid, beta[cell]
        )
        weights = np.full(int(count), energy_by_cell[cell] / count)
        positions = np.random.uniform(mesh_array[cell, 0], mesh_array[cell, 1], int(count))
        parts.append((
            weights,
            directions_lab,
            energies_lab,
            positions,
            np.zeros(int(count)),
            np.full(int(count), cell, dtype=np.int64),
        ))
    return _assemble_batch(parts, n_cells, edges)


def sample_moving_volume_source(
    target,
    temperature,
    dt,
    mesh,
    energy_edges,
    sigma_a_fleck,
    material_velocity,
    temperature_floor=0.0,
):
    """Sample equal-lab-weight thermal emission from moving slab material."""
    _validate_sampling_controls(target, temperature_floor)
    mesh_array = _validate_mesh(mesh)
    edges = _validate_energy_edges(energy_edges)
    n_cells = len(mesh_array)
    n_groups = len(edges) - 1
    velocity = validate_material_velocity(material_velocity, n_cells=n_cells, c=C_LIGHT)
    beta, gamma = precompute_velocity_factors(velocity, n_cells=n_cells, c=C_LIGHT)
    material_temperature = np.asarray(temperature, dtype=np.float64)
    opacity = np.asarray(sigma_a_fleck, dtype=np.float64)
    if material_temperature.shape != (n_cells,):
        raise ValueError("temperature must have one value per cell")
    if opacity.shape != (n_groups, n_cells):
        raise ValueError("sigma_a_fleck must have shape (n_groups, n_cells)")
    if not np.all(np.isfinite(material_temperature)) or not np.all(np.isfinite(opacity)):
        raise ValueError("temperature and sigma_a_fleck must be finite")
    if np.any(material_temperature < 0.0):
        raise ValueError("temperature must be nonnegative")
    if not np.isfinite(dt) or np.any(opacity < 0.0) or dt < 0.0:
        raise ValueError("sigma_a_fleck must be nonnegative and dt finite/nonnegative")

    widths = mesh_array[:, 1] - mesh_array[:, 0]
    b_group = _compute_Bg_1d(edges, material_temperature)
    b_total = np.sum(b_group, axis=0)
    b_fraction = b_group / np.maximum(b_total[None, :], 1.0e-300)
    emitted_fluid_by_group = (
        A_RAD * C_LIGHT * material_temperature[None, :]**4
        * opacity * b_fraction * dt * widths[None, :]
    )
    if temperature_floor > 0.0:
        emitted_fluid_by_group[:, material_temperature < temperature_floor] = 0.0
    energy_by_cell = gamma * np.sum(emitted_fluid_by_group, axis=0)
    counts = _particle_counts_by_cell(target, energy_by_cell)

    parts = []
    for cell, count in enumerate(counts):
        if count <= 0:
            continue
        energies_fluid, _ = _sample_opacity_weighted_planck(
            int(count), material_temperature[cell], edges,
            opacity[:, cell], b_fraction[:, cell],
        )
        directions_fluid = sample_fluid_directions(int(count), beta[cell], doppler_power=1)
        energies_lab, directions_lab = transform_fluid_particles_to_lab(
            energies_fluid, directions_fluid, beta[cell]
        )
        weights = np.full(int(count), energy_by_cell[cell] / count)
        positions = np.random.uniform(mesh_array[cell, 0], mesh_array[cell, 1], int(count))
        times = np.random.uniform(0.0, dt, int(count))
        parts.append((
            weights,
            directions_lab,
            energies_lab,
            positions,
            times,
            np.full(int(count), cell, dtype=np.int64),
        ))
    return _assemble_batch(parts, n_cells, edges)


def sample_moving_face_source(
    target,
    temperature,
    dt,
    mesh,
    energy_edges,
    material_velocity,
    side,
    mu_points=1025,
    phi_points=2049,
):
    """Sample equal-lab-weight blackbody particles entering one slab face.

    ``side`` is 0 for the left face and 1 for the right face.  The boundary
    velocity is taken from the adjacent material cell, while the face remains
    stationary in the lab frame.
    """
    _validate_sampling_controls(target, 0.0)
    mesh_array = _validate_mesh(mesh)
    edges = _validate_energy_edges(energy_edges)
    n_cells = len(mesh_array)
    velocity = validate_material_velocity(
        material_velocity, n_cells=n_cells, c=C_LIGHT
    )
    if side not in (0, 1):
        raise ValueError("side must be 0 (left) or 1 (right)")
    if not np.isfinite(temperature) or temperature < 0.0:
        raise ValueError("boundary temperature must be finite and nonnegative")
    if not np.isfinite(dt) or dt < 0.0:
        raise ValueError("dt must be finite and nonnegative")
    if target == 0 or temperature == 0.0 or dt == 0.0:
        return _assemble_batch([], n_cells, edges)

    cell = 0 if side == 0 else n_cells - 1
    inward_normal = np.array([1.0, 0.0, 0.0])
    face_position = mesh_array[0, 0]
    if side == 1:
        inward_normal[0] = -1.0
        face_position = mesh_array[-1, 1]

    beta = velocity[cell] / C_LIGHT
    beta_normal = float(np.dot(beta, inward_normal))
    beta_tangent_vector = beta - beta_normal * inward_normal
    beta_tangent = float(np.linalg.norm(beta_tangent_vector))
    if beta_tangent > 1.0e-15:
        tangent_1 = beta_tangent_vector / beta_tangent
    else:
        tangent_1 = np.array([0.0, 1.0, 0.0])
    tangent_2 = np.cross(inward_normal, tangent_1)

    table = _get_moving_face_angular_table(
        beta_tangent,
        beta_normal,
        mu_points=mu_points,
        phi_points=phi_points,
    )
    mu, phi = _sample_face_local_angles(table, target)
    transverse = np.sqrt(np.maximum(0.0, 1.0 - mu**2))
    directions = (
        mu[:, None] * inward_normal[None, :]
        + transverse[:, None]
        * (
            np.cos(phi)[:, None] * tangent_1[None, :]
            + np.sin(phi)[:, None] * tangent_2[None, :]
        )
    )
    doppler_lab = 1.0 - directions @ beta
    energies_fluid, _ = _sample_planck_spectrum_mg(target, temperature, edges)
    energies_lab = energies_fluid / (table.gamma * doppler_lab)
    total_energy = (
        A_RAD * C_LIGHT * temperature**4 * 0.25
        * table.gamma**2 * table.flux_factor * dt
    )
    parts = [(
        np.full(target, total_energy / target),
        directions,
        energies_lab,
        np.full(target, face_position),
        np.random.uniform(0.0, dt, target),
        np.full(target, cell, dtype=np.int64),
    )]
    return _assemble_batch(parts, n_cells, edges)


def _refresh_state_radiation(state, mesh_array):
    """Recompute cellwise lab/fluid radiation moments from the census."""
    n_cells = len(mesh_array)
    directions = np.column_stack((state.dir_x, state.dir_y, state.dir_z))
    energy_lab, momentum_lab = _radiation_moments(
        state.weights, directions, state.cell_indices, n_cells
    )
    energy_fluid, momentum_fluid = lab_energy_momentum_to_fluid(
        energy_lab, momentum_lab, state.material_velocity, c=C_LIGHT
    )
    _, gamma = precompute_velocity_factors(
        state.material_velocity, n_cells=n_cells, c=C_LIGHT
    )
    widths = mesh_array[:, 1] - mesh_array[:, 0]
    state.radiation_energy_lab = energy_lab
    state.radiation_momentum_lab = momentum_lab
    state.radiation_energy_fluid = energy_fluid
    state.radiation_momentum_fluid = momentum_fluid
    state.radiation_temperature = np.maximum(
        energy_fluid / (gamma * widths) / A_RAD, 0.0
    ) ** 0.25


def _population_counts_by_cell(target, energy_by_cell):
    active_cells = np.flatnonzero(energy_by_cell > 0.0)
    if len(active_cells) == 0:
        return np.zeros(len(energy_by_cell), dtype=np.int64)
    if target < len(active_cells):
        raise ValueError(
            "population target must be at least the number of energetic cells"
        )
    counts = np.zeros(len(energy_by_cell), dtype=np.int64)
    counts[active_cells] = 1
    remaining = target - len(active_cells)
    if remaining == 0:
        return counts
    active_energy = energy_by_cell[active_cells]
    shares = remaining * active_energy / np.sum(active_energy)
    additions = np.floor(shares).astype(np.int64)
    counts[active_cells] += additions
    leftover = remaining - int(np.sum(additions))
    if leftover:
        order = np.argsort(-(shares - additions), kind="stable")
        counts[active_cells[order[:leftover]]] += 1
    return counts


def population_control(state, target, mesh, energy_edges):
    """Systematically resample complete particle records within each cell.

    Cell radiation energy is retained exactly.  Momentum is not forced; its
    stochastic change is returned explicitly for conservation diagnostics.
    """
    if (
        isinstance(target, (bool, np.bool_))
        or not isinstance(target, (int, np.integer))
        or target <= 0
    ):
        raise ValueError("population target must be a positive integer")
    mesh_array = _validate_mesh(mesh)
    edges = _validate_energy_edges(energy_edges)
    n_cells = len(mesh_array)
    count_before = len(state.weights)
    energy_before = float(np.sum(state.weights))
    momentum_before_by_cell = state.radiation_momentum_lab.copy()
    momentum_before = np.sum(momentum_before_by_cell, axis=0)

    positive = state.weights > 0.0
    positive_count = int(np.count_nonzero(positive))
    if positive_count <= target:
        if positive_count != count_before:
            state.weights = np.ascontiguousarray(state.weights[positive])
            state.dir_x = np.ascontiguousarray(state.dir_x[positive])
            state.dir_y = np.ascontiguousarray(state.dir_y[positive])
            state.dir_z = np.ascontiguousarray(state.dir_z[positive])
            state.photon_energies_lab = np.ascontiguousarray(
                state.photon_energies_lab[positive]
            )
            state.positions = np.ascontiguousarray(state.positions[positive])
            state.times = np.ascontiguousarray(state.times[positive])
            state.cell_indices = np.ascontiguousarray(state.cell_indices[positive])
            state.lab_groups = np.ascontiguousarray(
                lab_groups_from_energies(state.photon_energies_lab, edges)
            )
            _refresh_state_radiation(state, mesh_array)
        momentum_after_by_cell = state.radiation_momentum_lab.copy()
        momentum_change_by_cell = momentum_after_by_cell - momentum_before_by_cell
        return PopulationControlResult(
            particle_count_before=count_before,
            particle_count_after=positive_count,
            energy_before=energy_before,
            energy_after=float(np.sum(state.weights)),
            energy_change=float(np.sum(state.weights)) - energy_before,
            momentum_change_lab=np.sum(momentum_change_by_cell, axis=0),
            momentum_change_lab_by_cell=momentum_change_by_cell,
        )

    cell_energy = np.bincount(
        state.cell_indices[positive],
        weights=state.weights[positive],
        minlength=n_cells,
    ).astype(np.float64)
    counts = _population_counts_by_cell(target, cell_energy)
    selected_parts = []
    new_weights = []
    for cell, new_count in enumerate(counts):
        if new_count <= 0:
            continue
        candidates = np.flatnonzero(positive & (state.cell_indices == cell))
        candidate_weights = state.weights[candidates]
        cumulative = np.cumsum(candidate_weights)
        cumulative /= cumulative[-1]
        points = (np.random.random() + np.arange(new_count)) / new_count
        selected = candidates[np.searchsorted(cumulative, points, side="right")]
        weights = np.full(new_count, cell_energy[cell] / new_count)
        weights[-1] += cell_energy[cell] - float(np.cumsum(weights)[-1])
        selected_parts.append(selected)
        new_weights.append(weights)

    if selected_parts:
        selected = np.concatenate(selected_parts)
        weights = np.concatenate(new_weights)
    else:
        selected = np.empty(0, dtype=np.int64)
        weights = np.empty(0, dtype=np.float64)
    state.weights = np.ascontiguousarray(weights)
    state.dir_x = np.ascontiguousarray(state.dir_x[selected])
    state.dir_y = np.ascontiguousarray(state.dir_y[selected])
    state.dir_z = np.ascontiguousarray(state.dir_z[selected])
    state.photon_energies_lab = np.ascontiguousarray(
        state.photon_energies_lab[selected]
    )
    state.positions = np.ascontiguousarray(state.positions[selected])
    state.times = np.ascontiguousarray(state.times[selected])
    state.cell_indices = np.ascontiguousarray(state.cell_indices[selected])
    state.lab_groups = np.ascontiguousarray(
        lab_groups_from_energies(state.photon_energies_lab, edges)
    )
    _refresh_state_radiation(state, mesh_array)

    energy_after = float(np.sum(state.weights))
    momentum_after_by_cell = state.radiation_momentum_lab.copy()
    momentum_change_by_cell = momentum_after_by_cell - momentum_before_by_cell
    return PopulationControlResult(
        particle_count_before=count_before,
        particle_count_after=len(state.weights),
        energy_before=energy_before,
        energy_after=energy_after,
        energy_change=energy_after - energy_before,
        momentum_change_lab=(
            np.sum(momentum_after_by_cell, axis=0) - momentum_before
        ),
        momentum_change_lab_by_cell=momentum_change_by_cell,
    )


def transport_particles(
    state,
    dt,
    mesh,
    energy_edges,
    sigma_a_true,
    fleck_factors,
    reflect=(False, False),
    max_events_per_particle=1_000_000,
):
    """Transport the state's census through one fixed-material time interval.

    Opacity and group selection are evaluated in the local fluid frame for
    every segment.  Packet weight is lab energy: implicit capture attenuates
    it continuously, while effective scattering changes photon energy and
    direction but leaves packet weight unchanged.
    """
    mesh_array = _validate_mesh(mesh)
    edges = _validate_energy_edges(energy_edges)
    n_cells = len(mesh_array)
    n_groups = len(edges) - 1
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    if len(reflect) != 2:
        raise ValueError("reflect must contain left and right boundary flags")
    if (
        isinstance(max_events_per_particle, (bool, np.bool_))
        or not isinstance(max_events_per_particle, (int, np.integer))
        or max_events_per_particle <= 0
    ):
        raise ValueError("max_events_per_particle must be a positive integer")

    opacity = np.asarray(sigma_a_true, dtype=np.float64)
    fleck = np.asarray(fleck_factors, dtype=np.float64)
    if opacity.shape != (n_groups, n_cells):
        raise ValueError("sigma_a_true must have shape (n_groups, n_cells)")
    if fleck.shape != (n_cells,):
        raise ValueError("fleck_factors must have one value per cell")
    if not np.all(np.isfinite(opacity)) or np.any(opacity < 0.0):
        raise ValueError("sigma_a_true must be finite and nonnegative")
    if not np.all(np.isfinite(fleck)) or np.any(fleck < 0.0) or np.any(fleck > 1.0):
        raise ValueError("fleck_factors must be finite and lie in [0, 1]")
    velocity = validate_material_velocity(
        state.material_velocity, n_cells=n_cells, c=C_LIGHT
    )
    beta, gamma = precompute_velocity_factors(velocity, n_cells=n_cells, c=C_LIGHT)
    if np.any(state.times < 0.0) or np.any(state.times > dt):
        raise ValueError("particle birth times must lie in [0, dt]")

    b_group = _compute_Bg_1d(edges, state.temperature)
    b_fraction = b_group / np.maximum(np.sum(b_group, axis=0)[None, :], 1.0e-300)
    material_energy = np.zeros(n_cells, dtype=np.float64)
    material_momentum = np.zeros((n_cells, 3), dtype=np.float64)
    track_energy = np.zeros((n_groups, n_cells), dtype=np.float64)
    boundary_energy = np.zeros(2, dtype=np.float64)
    boundary_momentum = np.zeros((2, 3), dtype=np.float64)
    wall_momentum = np.zeros((2, 3), dtype=np.float64)
    # segments, boundary crossings, effective scatters, census events, reflections
    event_counts = np.zeros(5, dtype=np.int64)

    alive = np.ones(len(state.weights), dtype=bool)
    for particle in range(len(state.weights)):
        distance_to_census = (dt - state.times[particle]) * C_LIGHT
        events = 0
        while distance_to_census > 1.0e-14 and alive[particle]:
            events += 1
            if events > max_events_per_particle:
                raise RuntimeError(
                    f"particle {particle} exceeded max_events_per_particle"
                )
            event_counts[0] += 1
            cell = int(state.cell_indices[particle])
            if cell < 0 or cell >= n_cells:
                raise RuntimeError("live particle has an out-of-range cell index")

            direction = np.array(
                [state.dir_x[particle], state.dir_y[particle], state.dir_z[particle]]
            )
            group, _, doppler_lab = fluid_group_from_lab_photon(
                state.photon_energies_lab[particle],
                direction[0], direction[1], direction[2],
                beta[cell, 0], beta[cell, 1], beta[cell, 2],
                edges,
            )
            opacity_lab = gamma[cell] * doppler_lab * opacity[group, cell]
            sigma_capture = fleck[cell] * opacity_lab
            sigma_effective_scatter = (1.0 - fleck[cell]) * opacity_lab

            if direction[0] > 0.0:
                distance_to_boundary = (
                    mesh_array[cell, 1] - state.positions[particle]
                ) / direction[0]
                boundary_delta = 1
            elif direction[0] < 0.0:
                distance_to_boundary = (
                    mesh_array[cell, 0] - state.positions[particle]
                ) / direction[0]
                boundary_delta = -1
            else:
                distance_to_boundary = np.inf
                boundary_delta = 0

            if sigma_effective_scatter > 0.0:
                distance_to_scatter = -math.log(
                    max(np.random.random(), 1.0e-300)
                ) / sigma_effective_scatter
            else:
                distance_to_scatter = np.inf

            census_distance_before = distance_to_census
            distance = min(
                distance_to_boundary, distance_to_scatter, census_distance_before
            )
            if not np.isfinite(distance) or distance < -1.0e-12:
                raise RuntimeError("transport produced an invalid event distance")
            distance = max(distance, 0.0)

            weight_before = state.weights[particle]
            if sigma_capture > 0.0:
                attenuation = math.exp(-sigma_capture * distance)
                integrated_weight = weight_before * (1.0 - attenuation) / sigma_capture
            else:
                attenuation = 1.0
                integrated_weight = weight_before * distance
            deposited = weight_before * (1.0 - attenuation)
            state.weights[particle] = weight_before * attenuation
            material_energy[cell] += deposited
            material_momentum[cell] += deposited * direction / C_LIGHT
            track_energy[group, cell] += integrated_weight

            state.positions[particle] += direction[0] * distance
            distance_to_census = max(0.0, distance_to_census - distance)

            if (
                distance_to_boundary <= distance_to_scatter
                and distance_to_boundary <= census_distance_before
            ):
                event_counts[1] += 1
                state.positions[particle] = mesh_array[
                    cell, 1 if boundary_delta > 0 else 0
                ]
                next_cell = cell + boundary_delta
                if 0 <= next_cell < n_cells:
                    state.cell_indices[particle] = next_cell
                    continue

                side = 0 if next_cell < 0 else 1
                if reflect[side]:
                    outgoing = direction.copy()
                    outgoing[0] *= -1.0
                    wall_momentum[side] += (
                        state.weights[particle] * (direction - outgoing) / C_LIGHT
                    )
                    state.dir_x[particle] = outgoing[0]
                    state.cell_indices[particle] = cell
                    event_counts[4] += 1
                else:
                    boundary_energy[side] += state.weights[particle]
                    boundary_momentum[side] += (
                        state.weights[particle] * direction / C_LIGHT
                    )
                    alive[particle] = False
                continue

            if distance_to_scatter <= census_distance_before:
                event_counts[2] += 1
                incoming = direction
                energy_fluid, _ = _sample_opacity_weighted_planck(
                    1,
                    state.temperature[cell],
                    edges,
                    opacity[:, cell],
                    b_fraction[:, cell],
                )
                direction_fluid = sample_fluid_directions(
                    1, beta[cell], doppler_power=1
                )
                energy_lab, direction_lab = transform_fluid_particles_to_lab(
                    energy_fluid, direction_fluid, beta[cell]
                )
                outgoing = direction_lab[0]
                material_momentum[cell] += (
                    state.weights[particle] * (incoming - outgoing) / C_LIGHT
                )
                state.photon_energies_lab[particle] = energy_lab[0]
                state.dir_x[particle] = outgoing[0]
                state.dir_y[particle] = outgoing[1]
                state.dir_z[particle] = outgoing[2]
                continue

            event_counts[3] += 1

    keep = alive
    state.weights = np.ascontiguousarray(state.weights[keep])
    state.dir_x = np.ascontiguousarray(state.dir_x[keep])
    state.dir_y = np.ascontiguousarray(state.dir_y[keep])
    state.dir_z = np.ascontiguousarray(state.dir_z[keep])
    state.photon_energies_lab = np.ascontiguousarray(
        state.photon_energies_lab[keep]
    )
    state.positions = np.ascontiguousarray(state.positions[keep])
    state.times = np.zeros(np.count_nonzero(keep), dtype=np.float64)
    state.cell_indices = np.ascontiguousarray(state.cell_indices[keep])
    state.lab_groups = np.ascontiguousarray(
        lab_groups_from_energies(state.photon_energies_lab, edges)
    )
    _refresh_state_radiation(state, mesh_array)

    return MovingTransportResult(
        material_energy_exchange_lab=material_energy,
        material_momentum_exchange_lab=material_momentum,
        track_length_energy_lab_by_fluid_group=track_energy,
        boundary_energy_loss_lab=boundary_energy,
        boundary_momentum_loss_lab=boundary_momentum,
        wall_momentum_exchange_lab=wall_momentum,
        event_counts=event_counts,
    )


def _evaluate_group_opacities(sigma_a_funcs, temperature, n_groups):
    if len(sigma_a_funcs) != n_groups:
        raise ValueError("sigma_a_funcs must contain one callable per energy group")
    opacity = np.empty((n_groups, len(temperature)), dtype=np.float64)
    for group, opacity_function in enumerate(sigma_a_funcs):
        values = np.asarray(opacity_function(temperature), dtype=np.float64)
        if values.ndim == 0:
            opacity[group] = values
        elif values.shape == temperature.shape:
            opacity[group] = values
        else:
            raise ValueError("each opacity function must return a scalar or one value per cell")
    if not np.all(np.isfinite(opacity)) or np.any(opacity < 0.0):
        raise ValueError("opacity functions must return finite, nonnegative values")
    return opacity


def step(
    state,
    target,
    dt,
    mesh,
    energy_edges,
    sigma_a_funcs,
    inv_eos,
    cv,
    theta=1.0,
    reflect=(False, False),
    temperature_floor=0.0,
    max_events_per_particle=1_000_000,
    material_velocity=None,
    boundary_target=0,
    boundary_temperature=(0.0, 0.0),
    face_mu_points=1025,
    face_phi_points=2049,
    population_target=0,
):
    """Advance the moving-material solver by one source/transport step.

    This step combines moving face and volume sources with implicit-capture
    and effective-scattering tallies, updates fluid internal energy at fixed
    velocity, and reports separate lab energy and momentum residuals.
    """
    _validate_sampling_controls(target, temperature_floor)
    mesh_array = _validate_mesh(mesh)
    edges = _validate_energy_edges(energy_edges)
    n_cells = len(mesh_array)
    n_groups = len(edges) - 1
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    if not np.isfinite(theta) or theta < 0.0 or theta > 1.0:
        raise ValueError("theta must be finite and lie in [0, 1]")
    if (
        isinstance(population_target, (bool, np.bool_))
        or not isinstance(population_target, (int, np.integer))
        or population_target < 0
    ):
        raise ValueError("population_target must be a nonnegative integer")
    if material_velocity is not None:
        state.material_velocity = validate_material_velocity(
            material_velocity, n_cells=n_cells, c=C_LIGHT
        )
    if len(boundary_temperature) != 2:
        raise ValueError("boundary_temperature must contain left and right values")
    if isinstance(boundary_target, (int, np.integer)) and not isinstance(
        boundary_target, (bool, np.bool_)
    ):
        boundary_targets = (int(boundary_target), int(boundary_target))
    else:
        try:
            boundary_target_length = len(boundary_target)
        except TypeError as error:
            raise ValueError(
                "boundary_target must be an integer or a two-entry sequence"
            ) from error
        if boundary_target_length != 2:
            raise ValueError("boundary_target must be an integer or a two-entry sequence")
        boundary_targets = tuple(boundary_target)
    for boundary_count in boundary_targets:
        _validate_sampling_controls(boundary_count, 0.0)

    old_radiation_momentum = np.sum(state.radiation_momentum_lab, axis=0).copy()
    old_total_energy = float(state.previous_total_energy)
    opacity = _evaluate_group_opacities(
        sigma_a_funcs, state.temperature, n_groups
    )
    heat_capacity = np.asarray(cv(state.temperature), dtype=np.float64)
    if heat_capacity.ndim == 0:
        heat_capacity = np.full(n_cells, float(heat_capacity))
    if heat_capacity.shape != (n_cells,):
        raise ValueError("cv(temperature) must return a scalar or one value per cell")
    if not np.all(np.isfinite(heat_capacity)) or np.any(heat_capacity <= 0.0):
        raise ValueError("cv(temperature) must be finite and positive")

    _, gamma = precompute_velocity_factors(
        state.material_velocity, n_cells=n_cells, c=C_LIGHT
    )
    b_group = _compute_Bg_1d(edges, state.temperature)
    b_fraction = b_group / np.maximum(np.sum(b_group, axis=0)[None, :], 1.0e-300)
    planck_opacity = np.sum(opacity * b_fraction, axis=0)
    fleck_beta = 4.0 * A_RAD * state.temperature**3 / heat_capacity
    fleck = 1.0 / (
        1.0 + theta * fleck_beta * gamma * C_LIGHT * planck_opacity * dt
    )

    boundary_energy_injection = np.zeros(2, dtype=np.float64)
    boundary_momentum_injection = np.zeros((2, 3), dtype=np.float64)
    for side in (0, 1):
        boundary_value = boundary_temperature[side]
        boundary_value = (
            boundary_value(state.time) if callable(boundary_value) else boundary_value
        )
        boundary_source = sample_moving_face_source(
            boundary_targets[side],
            boundary_value,
            dt,
            mesh_array,
            edges,
            state.material_velocity,
            side,
            mu_points=face_mu_points,
            phi_points=face_phi_points,
        )
        boundary_energy_injection[side] = np.sum(boundary_source.weights)
        boundary_momentum_injection[side] = np.sum(
            boundary_source.momentum_lab_by_cell, axis=0
        )
        _append_particle_batch(state, boundary_source)

    source = sample_moving_volume_source(
        target,
        state.temperature,
        dt,
        mesh_array,
        edges,
        fleck[None, :] * opacity,
        state.material_velocity,
        temperature_floor=temperature_floor,
    )
    source_energy_exchange = -source.energy_lab_by_cell
    source_momentum_exchange = -source.momentum_lab_by_cell
    _append_particle_batch(state, source)

    transport = transport_particles(
        state,
        dt,
        mesh_array,
        edges,
        opacity,
        fleck,
        reflect=reflect,
        max_events_per_particle=max_events_per_particle,
    )
    material_energy_exchange = (
        source_energy_exchange + transport.material_energy_exchange_lab
    )
    material_momentum_exchange = (
        source_momentum_exchange + transport.material_momentum_exchange_lab
    )

    widths = mesh_array[:, 1] - mesh_array[:, 0]
    state.internal_energy = state.internal_energy + material_energy_exchange / widths
    state.temperature = np.asarray(inv_eos(state.internal_energy), dtype=np.float64)
    if state.temperature.shape != (n_cells,):
        raise ValueError("inv_eos(internal_energy) must return one value per cell")
    if not np.all(np.isfinite(state.temperature)) or np.any(state.temperature < 0.0):
        raise ValueError("inv_eos(internal_energy) returned an invalid temperature")

    population_result = PopulationControlResult(
        particle_count_before=len(state.weights),
        particle_count_after=len(state.weights),
        energy_before=float(np.sum(state.weights)),
        energy_after=float(np.sum(state.weights)),
        energy_change=0.0,
        momentum_change_lab=np.zeros(3),
        momentum_change_lab_by_cell=np.zeros((n_cells, 3)),
    )
    if population_target > 0:
        population_result = population_control(
            state, population_target, mesh_array, edges
        )
    state.time += dt
    state.count += 1

    total_internal_energy = float(np.sum(state.internal_energy * widths))
    total_radiation_energy = float(np.sum(state.weights))
    total_energy = total_internal_energy + total_radiation_energy
    boundary_energy_loss = float(np.sum(transport.boundary_energy_loss_lab))
    energy_residual_including_population_control = (
        total_energy
        - old_total_energy
        - np.sum(boundary_energy_injection)
        + boundary_energy_loss
    )
    energy_residual = (
        energy_residual_including_population_control
        - population_result.energy_change
    )
    radiation_momentum = np.sum(state.radiation_momentum_lab, axis=0)
    momentum_residual_including_population_control = (
        radiation_momentum
        - old_radiation_momentum
        + np.sum(material_momentum_exchange, axis=0)
        + np.sum(transport.boundary_momentum_loss_lab, axis=0)
        + np.sum(transport.wall_momentum_exchange_lab, axis=0)
        - np.sum(boundary_momentum_injection, axis=0)
    )
    momentum_residual = (
        momentum_residual_including_population_control
        - population_result.momentum_change_lab
    )
    state.previous_total_energy = total_energy

    info = {
        "time": state.time,
        "temperature": state.temperature.copy(),
        "radiation_temperature": state.radiation_temperature.copy(),
        "N_particles": len(state.weights),
        "fleck_factors": fleck,
        "planck_opacity": planck_opacity,
        "material_energy_exchange_lab": material_energy_exchange,
        "material_momentum_exchange_lab": material_momentum_exchange,
        "source_energy_exchange_lab": source_energy_exchange,
        "source_momentum_exchange_lab": source_momentum_exchange,
        "transport_energy_exchange_lab": transport.material_energy_exchange_lab,
        "transport_momentum_exchange_lab": transport.material_momentum_exchange_lab,
        "boundary_energy_loss_lab": transport.boundary_energy_loss_lab,
        "boundary_momentum_loss_lab": transport.boundary_momentum_loss_lab,
        "boundary_energy_injection_lab": boundary_energy_injection,
        "boundary_momentum_injection_lab": boundary_momentum_injection,
        "wall_momentum_exchange_lab": transport.wall_momentum_exchange_lab,
        "track_length_energy_lab_by_fluid_group": (
            transport.track_length_energy_lab_by_fluid_group
        ),
        "event_counts": transport.event_counts,
        "event_count_labels": (
            "segments",
            "boundary_crossings",
            "effective_scatters",
            "census_events",
            "reflections",
        ),
        "population_control": population_result,
        "total_internal_energy": total_internal_energy,
        "total_radiation_energy": total_radiation_energy,
        "total_energy": total_energy,
        "energy_residual": energy_residual,
        "energy_residual_including_population_control": (
            energy_residual_including_population_control
        ),
        "momentum_residual_lab": momentum_residual,
        "momentum_residual_lab_including_population_control": (
            momentum_residual_including_population_control
        ),
        "radiation_energy_lab_by_cell": state.radiation_energy_lab.copy(),
        "radiation_momentum_lab_by_cell": state.radiation_momentum_lab.copy(),
        "radiation_energy_fluid_by_cell": state.radiation_energy_fluid.copy(),
        "radiation_momentum_fluid_by_cell": state.radiation_momentum_fluid.copy(),
        "radiation_energy_lab_global": float(np.sum(state.radiation_energy_lab)),
        "radiation_momentum_lab_global": radiation_momentum.copy(),
        "radiation_energy_fluid_sum": float(np.sum(state.radiation_energy_fluid)),
        "radiation_momentum_fluid_sum": np.sum(
            state.radiation_momentum_fluid, axis=0
        ),
    }
    return state, info


def init_simulation(
    target,
    material_temperature,
    radiation_temperature,
    mesh,
    energy_edges,
    material_velocity,
    eos,
    inv_eos=None,
    target_initial=None,
    temperature_floor=0.0,
):
    """Initialize a moving-material equilibrium census and material state."""
    temperature = np.asarray(material_temperature, dtype=np.float64)
    internal_energy = np.asarray(eos(temperature), dtype=np.float64)
    if inv_eos is not None and not np.allclose(inv_eos(internal_energy), temperature):
        raise ValueError("inv_eos(eos(material_temperature)) did not recover temperature")
    initial_target = target if target_initial is None else target_initial
    particles = sample_moving_equilibrium_particles(
        initial_target,
        radiation_temperature,
        mesh,
        energy_edges,
        material_velocity,
        temperature_floor=temperature_floor,
    )
    directions = np.column_stack((particles.dir_x, particles.dir_y, particles.dir_z))
    return create_state_from_particles(
        particles.weights,
        directions,
        particles.photon_energies_lab,
        particles.positions,
        particles.times,
        particles.cell_indices,
        temperature,
        material_velocity,
        mesh,
        energy_edges,
        eos,
    )
