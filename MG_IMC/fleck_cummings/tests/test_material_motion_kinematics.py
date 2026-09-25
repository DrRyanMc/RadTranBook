"""Phase 1 tests for moving-material IMC frame and state contracts."""

from __future__ import annotations

import math

import numpy as np
import pytest

from MG_IMC.fleck_cummings.src.MG_IMC1DMoving import create_state_from_particles
from MG_IMC.fleck_cummings.src.material_motion import (
    C_LIGHT,
    fluid_group_from_lab_photon,
    fluid_to_lab_photon,
    lab_to_fluid_photon,
    precompute_velocity_factors,
    validate_material_velocity,
)


def _random_unit_vectors(rng, count):
    directions = rng.normal(size=(count, 3))
    return directions / np.linalg.norm(directions, axis=1)[:, None]


def test_zero_velocity_is_identity():
    energy = 7.25
    direction = np.array([0.2, -0.4, math.sqrt(0.8)])
    transformed = lab_to_fluid_photon(energy, *direction, 0.0, 0.0, 0.0)
    assert transformed[0] == energy
    assert np.array_equal(np.asarray(transformed[1:4]), direction)
    assert transformed[4] == 1.0


def test_lab_fluid_round_trip_for_general_velocities():
    rng = np.random.default_rng(90210)
    directions = _random_unit_vectors(rng, 512)
    beta_directions = _random_unit_vectors(rng, 512)
    beta_magnitudes = rng.uniform(0.0, 0.99, size=512)
    betas = beta_directions * beta_magnitudes[:, None]
    energies = rng.uniform(1.0e-3, 100.0, size=512)

    for energy, direction, beta in zip(energies, directions, betas):
        fluid = lab_to_fluid_photon(energy, *direction, *beta)
        lab = fluid_to_lab_photon(fluid[0], *fluid[1:4], *beta)
        assert np.isclose(lab[0], energy, rtol=2.0e-13, atol=1.0e-13)
        assert np.allclose(lab[1:4], direction, rtol=2.0e-13, atol=2.0e-13)
        assert np.isclose(np.linalg.norm(fluid[1:4]), 1.0, atol=2.0e-13)
        assert fluid[4] > 0.0


def test_transverse_velocity_aberration():
    beta = 0.9
    gamma = 1.0 / math.sqrt(1.0 - beta * beta)
    fluid = lab_to_fluid_photon(2.0, 1.0, 0.0, 0.0, 0.0, 0.0, beta)
    assert np.isclose(fluid[0], 2.0 * gamma)
    assert np.allclose(fluid[1:4], [1.0 / gamma, 0.0, -beta])


def test_fluid_group_changes_across_velocity_discontinuity():
    edges = np.array([0.0, 3.0, 5.0, 10.0])
    energy_lab = 4.0
    direction = (1.0, 0.0, 0.0)

    stationary = fluid_group_from_lab_photon(
        energy_lab, *direction, 0.0, 0.0, 0.0, edges
    )
    moving = fluid_group_from_lab_photon(
        energy_lab, *direction, 0.5, 0.0, 0.0, edges
    )

    assert stationary[0] == 1
    assert moving[0] == 0
    assert energy_lab == 4.0
    assert direction == (1.0, 0.0, 0.0)
    assert np.isclose(moving[1], 4.0 / math.sqrt(3.0))


def test_material_velocity_validation_and_precomputation():
    velocity = np.array([[0.0, 0.0, 0.0], [0.5 * C_LIGHT, 0.0, 0.0]])
    beta, gamma = precompute_velocity_factors(velocity, n_cells=2)
    assert np.allclose(beta[1], [0.5, 0.0, 0.0])
    assert np.allclose(gamma, [1.0, 2.0 / math.sqrt(3.0)])

    with pytest.raises(ValueError, match=r"shape"):
        validate_material_velocity(np.zeros((2, 2)))
    with pytest.raises(ValueError, match=r"\|v\| < c"):
        validate_material_velocity(np.array([[C_LIGHT, 0.0, 0.0]]))
    with pytest.raises(ValueError, match=r"finite"):
        validate_material_velocity(np.array([[np.nan, 0.0, 0.0]]))


def test_state_contract_uses_lab_energy_and_momentum():
    mesh = np.array([[0.0, 1.0], [1.0, 3.0]])
    weights = np.array([2.0, 3.0])
    directions = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    state = create_state_from_particles(
        weights=weights,
        directions_lab=directions,
        photon_energies_lab=np.array([2.0, 7.0]),
        positions=np.array([0.25, 2.0]),
        times=np.zeros(2),
        cell_indices=np.array([0, 1]),
        temperature=np.array([1.0, 2.0]),
        material_velocity=np.array([[0.0, 0.0, 0.0], [0.0, 0.1 * C_LIGHT, 0.0]]),
        mesh=mesh,
        energy_edges=np.array([0.0, 5.0, 10.0]),
        eos=lambda temperature: 4.0 * temperature,
    )

    assert np.array_equal(state.lab_groups, [0, 1])
    assert np.array_equal(state.radiation_energy_lab, weights)
    assert np.allclose(
        state.radiation_momentum_lab,
        [[2.0 / C_LIGHT, 0.0, 0.0], [0.0, -3.0 / C_LIGHT, 0.0]],
    )
    expected_material = 4.0 * 1.0 * 1.0 + 4.0 * 2.0 * 2.0
    assert np.isclose(state.previous_total_energy, expected_material + np.sum(weights))


def test_state_rejects_non_unit_direction():
    with pytest.raises(ValueError, match=r"unit vector"):
        create_state_from_particles(
            weights=np.array([1.0]),
            directions_lab=np.array([[1.0, 1.0, 0.0]]),
            photon_energies_lab=np.array([1.0]),
            positions=np.array([0.5]),
            times=np.array([0.0]),
            cell_indices=np.array([0]),
            temperature=np.array([1.0]),
            material_velocity=np.zeros((1, 3)),
            mesh=np.array([[0.0, 1.0]]),
            energy_edges=np.array([0.0, 2.0]),
            eos=lambda temperature: temperature,
        )


def test_state_rejects_non_finite_particle_data():
    with pytest.raises(ValueError, match=r"photon_energies_lab.*finite"):
        create_state_from_particles(
            weights=np.array([1.0]),
            directions_lab=np.array([[1.0, 0.0, 0.0]]),
            photon_energies_lab=np.array([np.inf]),
            positions=np.array([0.5]),
            times=np.array([0.0]),
            cell_indices=np.array([0]),
            temperature=np.array([1.0]),
            material_velocity=np.zeros((1, 3)),
            mesh=np.array([[0.0, 1.0]]),
            energy_edges=np.array([0.0, 2.0]),
            eos=lambda temperature: temperature,
        )
