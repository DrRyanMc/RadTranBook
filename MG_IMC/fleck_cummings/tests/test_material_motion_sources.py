"""Phase 2 tests for moving equilibrium and material volume sources."""

from __future__ import annotations

import numpy as np
import pytest

from MG_IMC.fleck_cummings.src.MG_IMC1DMoving import (
    A_RAD,
    C_LIGHT,
    _compute_Bg_1d,
    init_simulation,
    sample_moving_equilibrium_particles,
    sample_moving_volume_source,
)


def _directions(batch):
    return np.column_stack((batch.dir_x, batch.dir_y, batch.dir_z))


def _six_sigma_momentum_bound(total_energy, particle_count):
    # Each unit-direction component has variance no greater than one.
    return 6.0 * total_energy / (C_LIGHT * np.sqrt(particle_count))


@pytest.mark.parametrize("seed", [1201, 1204, 1207])
def test_stationary_volume_source_energy_groups_and_isotropy(seed):
    np.random.seed(seed)
    mesh = np.array([[0.0, 1.0]])
    energy_edges = np.array([0.0, 1.0, 3.0, 10.0])
    temperature = np.array([1.0])
    opacity = np.array([[1.0], [2.0], [4.0]])
    dt = 0.2
    target = 30_000

    particles = sample_moving_volume_source(
        target,
        temperature,
        dt,
        mesh,
        energy_edges,
        opacity,
        np.zeros((1, 3)),
    )

    b_group = _compute_Bg_1d(energy_edges, temperature)[:, 0]
    b_fraction = b_group / np.sum(b_group)
    expected_group_probability = opacity[:, 0] * b_fraction
    expected_group_probability /= np.sum(expected_group_probability)
    expected_energy = (
        A_RAD * C_LIGHT * temperature[0] ** 4 * dt
        * np.sum(opacity[:, 0] * b_fraction)
    )

    observed_group_probability = (
        np.bincount(particles.lab_groups, minlength=3) / target
    )
    binomial_six_sigma = 6.0 * np.sqrt(
        expected_group_probability * (1.0 - expected_group_probability) / target
    )
    assert len(particles.weights) == target
    assert np.isclose(np.sum(particles.weights), expected_energy, rtol=2.0e-15)
    assert np.all(
        np.abs(observed_group_probability - expected_group_probability)
        <= binomial_six_sigma
    )
    assert np.allclose(np.linalg.norm(_directions(particles), axis=1), 1.0)
    assert np.all((particles.positions >= 0.0) & (particles.positions <= 1.0))
    assert np.all((particles.times >= 0.0) & (particles.times <= dt))
    assert np.all(
        np.abs(particles.momentum_lab_by_cell[0])
        <= _six_sigma_momentum_bound(expected_energy, target)
    )


@pytest.mark.parametrize("seed", [1203, 1206, 1209, 1212])
def test_moving_volume_source_recovers_lorentz_energy_and_momentum(seed):
    np.random.seed(seed)
    target = 60_000
    dt = 0.2
    beta = np.array([0.2, -0.3, 0.45])
    gamma = 1.0 / np.sqrt(1.0 - np.dot(beta, beta))
    fluid_energy = A_RAD * C_LIGHT * 2.0 * dt
    expected_lab_energy = gamma * fluid_energy
    expected_lab_momentum = expected_lab_energy * beta / C_LIGHT

    particles = sample_moving_volume_source(
        target,
        np.array([1.0]),
        dt,
        np.array([[0.0, 1.0]]),
        np.array([0.0, 100.0]),
        np.array([[2.0]]),
        beta[None, :] * C_LIGHT,
    )

    assert np.isclose(np.sum(particles.weights), expected_lab_energy, rtol=2.0e-15)
    assert np.all(
        np.abs(particles.momentum_lab_by_cell[0] - expected_lab_momentum)
        <= _six_sigma_momentum_bound(expected_lab_energy, target)
    )
    assert np.allclose(np.linalg.norm(_directions(particles), axis=1), 1.0)


@pytest.mark.parametrize("seed", [1202, 1205, 1208, 1211])
def test_moving_equilibrium_recovers_analytic_moments_and_temperature(seed):
    np.random.seed(seed)
    target = 60_000
    radiation_temperature = 1.3
    beta = np.array([0.2, -0.3, 0.45])
    beta_squared = np.dot(beta, beta)
    gamma = 1.0 / np.sqrt(1.0 - beta_squared)
    fluid_energy_density = A_RAD * radiation_temperature**4
    expected_lab_energy = (
        fluid_energy_density * gamma**2 * (1.0 + beta_squared / 3.0)
    )
    expected_lab_momentum = (
        (4.0 / 3.0) * fluid_energy_density * gamma**2 * beta / C_LIGHT
    )

    particles = sample_moving_equilibrium_particles(
        target,
        np.array([radiation_temperature]),
        np.array([[0.0, 1.0]]),
        np.array([0.0, 100.0]),
        beta[None, :] * C_LIGHT,
    )

    assert np.isclose(np.sum(particles.weights), expected_lab_energy, rtol=2.0e-15)
    assert np.all(
        np.abs(particles.momentum_lab_by_cell[0] - expected_lab_momentum)
        <= _six_sigma_momentum_bound(expected_lab_energy, target)
    )

    np.random.seed(seed)
    state = init_simulation(
        target,
        np.array([1.0]),
        np.array([radiation_temperature]),
        np.array([[0.0, 1.0]]),
        np.array([0.0, 100.0]),
        beta[None, :] * C_LIGHT,
        eos=lambda value: 2.0 * value,
        inv_eos=lambda value: 0.5 * value,
    )
    expected_fluid_energy = gamma * fluid_energy_density
    lab_momentum_bound = _six_sigma_momentum_bound(expected_lab_energy, target)
    fluid_energy_bound = gamma * C_LIGHT * np.sum(np.abs(beta)) * lab_momentum_bound
    assert abs(state.radiation_energy_fluid[0] - expected_fluid_energy) <= fluid_energy_bound
    assert np.isclose(
        state.radiation_temperature[0],
        (state.radiation_energy_fluid[0] / (gamma * A_RAD)) ** 0.25,
    )
    temperature_bound = (
        radiation_temperature * fluid_energy_bound / (4.0 * expected_fluid_energy)
    )
    assert abs(state.radiation_temperature[0] - radiation_temperature) <= temperature_bound


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"target": 1.5}, "target"),
        ({"radiation_temperature": np.array([-1.0])}, "nonnegative"),
        ({"energy_edges": np.array([0.0, np.inf])}, "energy_edges"),
    ],
)
def test_equilibrium_source_rejects_invalid_inputs(kwargs, message):
    arguments = {
        "target": 10,
        "radiation_temperature": np.array([1.0]),
        "mesh": np.array([[0.0, 1.0]]),
        "energy_edges": np.array([0.0, 10.0]),
        "material_velocity": np.zeros((1, 3)),
    }
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=message):
        sample_moving_equilibrium_particles(**arguments)


def test_volume_source_rejects_nonfinite_timestep():
    with pytest.raises(ValueError, match="dt"):
        sample_moving_volume_source(
            10,
            np.array([1.0]),
            np.inf,
            np.array([[0.0, 1.0]]),
            np.array([0.0, 10.0]),
            np.ones((1, 1)),
            np.zeros((1, 3)),
        )
