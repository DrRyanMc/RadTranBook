"""Phase 3 tests for moving-material slab transport and exchange tallies."""

from __future__ import annotations

import numpy as np

from MG_IMC.fleck_cummings.src.MG_IMC1DMoving import (
    A_RAD,
    C_LIGHT,
    create_state_from_particles,
    step,
    transport_particles,
)


def _state(
    mesh,
    energy_edges,
    directions,
    photon_energies,
    positions,
    cells,
    weights=None,
    temperature=None,
    material_velocity=None,
):
    particle_count = len(photon_energies)
    cell_count = len(mesh)
    if weights is None:
        weights = np.ones(particle_count)
    if temperature is None:
        temperature = np.ones(cell_count)
    if material_velocity is None:
        material_velocity = np.zeros((cell_count, 3))
    return create_state_from_particles(
        weights=np.asarray(weights),
        directions_lab=np.asarray(directions),
        photon_energies_lab=np.asarray(photon_energies),
        positions=np.asarray(positions),
        times=np.zeros(particle_count),
        cell_indices=np.asarray(cells),
        temperature=np.asarray(temperature),
        material_velocity=np.asarray(material_velocity),
        mesh=np.asarray(mesh),
        energy_edges=np.asarray(energy_edges),
        eos=lambda value: 100.0 * value,
    )


def test_one_segment_capture_uses_doppler_transformed_opacity_and_momentum():
    beta = 0.3
    direction = np.array([0.6, 0.8, 0.0])
    dt = 0.01
    energy_edges = np.array([0.0, 3.0, 5.0, 100.0])
    state = _state(
        mesh=np.array([[0.0, 100.0]]),
        energy_edges=energy_edges,
        directions=np.array([direction]),
        photon_energies=np.array([4.0]),
        positions=np.array([10.0]),
        cells=np.array([0]),
        material_velocity=np.array([[beta * C_LIGHT, 0.0, 0.0]]),
    )

    result = transport_particles(
        state,
        dt,
        np.array([[0.0, 100.0]]),
        energy_edges,
        sigma_a_true=np.array([[1.0], [2.0], [3.0]]),
        fleck_factors=np.ones(1),
    )

    gamma = 1.0 / np.sqrt(1.0 - beta**2)
    doppler_lab = 1.0 - beta * direction[0]
    sigma_lab = gamma * doppler_lab * 2.0
    expected_weight = np.exp(-sigma_lab * C_LIGHT * dt)
    expected_deposit = 1.0 - expected_weight
    assert np.isclose(state.weights[0], expected_weight, rtol=2.0e-14)
    assert np.isclose(result.material_energy_exchange_lab[0], expected_deposit)
    assert np.allclose(
        result.material_momentum_exchange_lab[0],
        expected_deposit * direction / C_LIGHT,
    )
    assert np.array_equal(result.event_counts, [1, 0, 0, 1, 0])


def test_crossing_velocity_discontinuity_recomputes_fluid_group():
    mesh = np.array([[0.0, 1.0], [1.0, 3.0]])
    energy_edges = np.array([0.0, 3.0, 5.0, 100.0])
    state = _state(
        mesh=mesh,
        energy_edges=energy_edges,
        directions=np.array([[1.0, 0.0, 0.0]]),
        photon_energies=np.array([4.0]),
        positions=np.array([0.5]),
        cells=np.array([0]),
        material_velocity=np.array(
            [[0.0, 0.0, 0.0], [0.5 * C_LIGHT, 0.0, 0.0]]
        ),
    )

    result = transport_particles(
        state,
        0.05,
        mesh,
        energy_edges,
        sigma_a_true=np.zeros((3, 2)),
        fleck_factors=np.ones(2),
    )

    expected_second_segment = C_LIGHT * 0.05 - 0.5
    assert np.isclose(result.track_length_energy_lab_by_fluid_group[1, 0], 0.5)
    assert np.isclose(
        result.track_length_energy_lab_by_fluid_group[0, 1],
        expected_second_segment,
    )
    assert np.count_nonzero(result.track_length_energy_lab_by_fluid_group) == 2
    assert state.photon_energies_lab[0] == 4.0
    assert np.array_equal([state.dir_x[0], state.dir_y[0], state.dir_z[0]], [1, 0, 0])
    assert state.cell_indices[0] == 1


def test_effective_scattering_preserves_energy_and_tallies_momentum_change():
    np.random.seed(42)
    mesh = np.array([[-100.0, 100.0]])
    energy_edges = np.array([0.0, 100.0])
    initial_direction = np.array([1.0, 0.0, 0.0])
    state = _state(
        mesh=mesh,
        energy_edges=energy_edges,
        directions=np.array([initial_direction]),
        photon_energies=np.array([4.0]),
        positions=np.array([0.0]),
        cells=np.array([0]),
    )

    result = transport_particles(
        state,
        0.02,
        mesh,
        energy_edges,
        sigma_a_true=np.array([[5.0]]),
        fleck_factors=np.zeros(1),
    )

    final_direction = np.array([state.dir_x[0], state.dir_y[0], state.dir_z[0]])
    assert result.event_counts[2] == 4
    assert state.weights[0] == 1.0
    assert result.material_energy_exchange_lab[0] == 0.0
    assert np.allclose(
        result.material_momentum_exchange_lab[0],
        (initial_direction - final_direction) / C_LIGHT,
        rtol=2.0e-14,
        atol=2.0e-16,
    )


def test_vacuum_escape_records_boundary_energy_and_momentum():
    mesh = np.array([[0.0, 1.0]])
    energy_edges = np.array([0.0, 10.0])
    state = _state(
        mesh=mesh,
        energy_edges=energy_edges,
        directions=np.array([[1.0, 0.0, 0.0]]),
        photon_energies=np.array([2.0]),
        positions=np.array([0.9]),
        cells=np.array([0]),
        weights=np.array([2.0]),
    )

    result = transport_particles(
        state,
        0.01,
        mesh,
        energy_edges,
        sigma_a_true=np.zeros((1, 1)),
        fleck_factors=np.ones(1),
    )

    assert len(state.weights) == 0
    assert np.array_equal(result.boundary_energy_loss_lab, [0.0, 2.0])
    assert np.allclose(
        result.boundary_momentum_loss_lab[1], [2.0 / C_LIGHT, 0.0, 0.0]
    )
    assert np.all(result.material_energy_exchange_lab == 0.0)


def test_closed_step_combines_source_transport_and_closes_energy_momentum():
    np.random.seed(9)
    mesh = np.array([[0.0, 1.0]])
    energy_edges = np.array([0.0, 100.0])
    beta = np.array([0.0, 0.6, 0.0])
    dt = 0.005
    state = _state(
        mesh=mesh,
        energy_edges=energy_edges,
        directions=np.array([[1.0, 0.0, 0.0]]),
        photon_energies=np.array([2.0]),
        positions=np.array([0.5]),
        cells=np.array([0]),
        weights=np.array([0.1]),
        material_velocity=beta[None, :] * C_LIGHT,
    )

    state, info = step(
        state,
        target=2_000,
        dt=dt,
        mesh=mesh,
        energy_edges=energy_edges,
        sigma_a_funcs=[lambda temperature: np.full_like(temperature, 2.0)],
        inv_eos=lambda internal_energy: internal_energy / 100.0,
        cv=lambda temperature: np.full_like(temperature, 100.0),
        reflect=(True, True),
    )

    gamma = 1.0 / np.sqrt(1.0 - np.dot(beta, beta))
    expected_fleck = 1.0 / (
        1.0 + 4.0 * A_RAD * gamma * C_LIGHT * 2.0 * dt / 100.0
    )
    assert np.isclose(info["fleck_factors"][0], expected_fleck)
    assert np.allclose(
        info["material_energy_exchange_lab"],
        info["source_energy_exchange_lab"] + info["transport_energy_exchange_lab"],
    )
    assert np.allclose(
        info["material_momentum_exchange_lab"],
        info["source_momentum_exchange_lab"]
        + info["transport_momentum_exchange_lab"],
    )
    assert abs(info["energy_residual"]) < 5.0e-13
    assert np.linalg.norm(info["momentum_residual_lab"], ord=np.inf) < 5.0e-15
    assert np.all(info["boundary_energy_loss_lab"] == 0.0)
    assert state.time == dt
    assert state.count == 1
