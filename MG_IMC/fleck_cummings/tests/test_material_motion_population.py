"""Phase 5 tests for moving-particle population control and diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from MG_IMC.fleck_cummings.src.MG_IMC1DMoving import (
    C_LIGHT,
    create_state_from_particles,
    population_control,
    seed_moving_imc_random,
    step,
)


def _population_state(count=100):
    index = np.arange(count)
    angle = 0.071 * index
    polar = -0.8 + 1.6 * (index + 0.5) / count
    transverse = np.sqrt(1.0 - polar**2)
    directions = np.column_stack(
        (polar, transverse * np.cos(angle), transverse * np.sin(angle))
    )
    cells = (index >= count // 2).astype(np.int64)
    positions = np.where(
        cells == 0,
        0.01 + 0.98 * index / max(count, 1),
        1.01 + 0.98 * (index - count // 2) / max(count, 1),
    )
    positions = np.minimum(positions, np.where(cells == 0, 0.99, 1.99))
    return create_state_from_particles(
        weights=0.2 + (index % 11) / 13.0,
        directions_lab=directions,
        photon_energies_lab=0.17 + 0.113 * index,
        positions=positions,
        times=np.zeros(count),
        cell_indices=cells,
        temperature=np.array([1.0, 1.1]),
        material_velocity=np.array(
            [[0.1 * C_LIGHT, 0.0, 0.0], [0.0, -0.2 * C_LIGHT, 0.0]]
        ),
        mesh=np.array([[0.0, 1.0], [1.0, 2.0]]),
        energy_edges=np.array([0.0, 2.0, 5.0, 20.0]),
        eos=lambda temperature: 20.0 * temperature,
    )


def _particle_records(state):
    return set(
        zip(
            state.dir_x,
            state.dir_y,
            state.dir_z,
            state.photon_energies_lab,
            state.positions,
            state.times,
            state.cell_indices,
        )
    )


def test_population_control_preserves_cell_energy_and_complete_particle_records():
    seed_moving_imc_random(5101)
    state = _population_state()
    records_before = _particle_records(state)
    energy_by_cell_before = state.radiation_energy_lab.copy()
    momentum_by_cell_before = state.radiation_momentum_lab.copy()

    result = population_control(
        state,
        target=20,
        mesh=np.array([[0.0, 1.0], [1.0, 2.0]]),
        energy_edges=np.array([0.0, 2.0, 5.0, 20.0]),
    )

    assert len(state.weights) == 20
    assert _particle_records(state) <= records_before
    assert np.allclose(state.radiation_energy_lab, energy_by_cell_before, rtol=0.0, atol=2e-15)
    assert result.energy_change == pytest.approx(0.0, abs=2e-15)
    assert np.allclose(
        result.momentum_change_lab_by_cell,
        state.radiation_momentum_lab - momentum_by_cell_before,
    )
    assert np.allclose(
        result.momentum_change_lab,
        np.sum(result.momentum_change_lab_by_cell, axis=0),
    )
    assert result.particle_count_before == 100
    assert result.particle_count_after == 20


def test_population_control_does_not_expand_an_under_target_census():
    state = _population_state(count=12)
    records_before = _particle_records(state)
    weights_before = state.weights.copy()

    result = population_control(
        state,
        target=20,
        mesh=np.array([[0.0, 1.0], [1.0, 2.0]]),
        energy_edges=np.array([0.0, 2.0, 5.0, 20.0]),
    )

    assert result.particle_count_before == result.particle_count_after == 12
    assert np.array_equal(state.weights, weights_before)
    assert _particle_records(state) == records_before
    assert np.all(result.momentum_change_lab == 0.0)


def test_population_target_must_represent_every_energetic_cell():
    state = _population_state(count=12)
    with pytest.raises(ValueError, match="energetic cells"):
        population_control(
            state,
            target=1,
            mesh=np.array([[0.0, 1.0], [1.0, 2.0]]),
            energy_edges=np.array([0.0, 2.0, 5.0, 20.0]),
        )


def test_step_population_control_reports_physics_and_sampling_residuals_separately():
    seed_moving_imc_random(5102)
    mesh = np.array([[0.0, 1.0]])
    energy_edges = np.array([0.0, 100.0])
    initial_count = 200
    directions = np.zeros((initial_count, 3))
    directions[:, 0] = 1.0
    state = create_state_from_particles(
        weights=np.full(initial_count, 0.001),
        directions_lab=directions,
        photon_energies_lab=np.linspace(0.5, 3.5, initial_count),
        positions=np.linspace(0.1, 0.9, initial_count),
        times=np.zeros(initial_count),
        cell_indices=np.zeros(initial_count, dtype=np.int64),
        temperature=np.array([1.0]),
        material_velocity=np.array([[0.0, 0.4 * C_LIGHT, 0.0]]),
        mesh=mesh,
        energy_edges=energy_edges,
        eos=lambda temperature: 100.0 * temperature,
    )

    state, info = step(
        state,
        target=1_000,
        dt=0.004,
        mesh=mesh,
        energy_edges=energy_edges,
        sigma_a_funcs=[lambda temperature: np.full_like(temperature, 1.5)],
        inv_eos=lambda internal_energy: internal_energy / 100.0,
        cv=lambda temperature: np.full_like(temperature, 100.0),
        reflect=(True, True),
        population_target=250,
    )

    population = info["population_control"]
    assert population.particle_count_before > 250
    assert population.particle_count_after == len(state.weights) == 250
    assert abs(population.energy_change) < 5.0e-15
    assert abs(info["energy_residual"]) < 5.0e-13
    assert np.linalg.norm(info["momentum_residual_lab"], ord=np.inf) < 5.0e-14
    assert np.allclose(
        info["momentum_residual_lab_including_population_control"]
        - population.momentum_change_lab,
        info["momentum_residual_lab"],
    )
    assert np.isclose(
        info["radiation_energy_lab_global"],
        np.sum(info["radiation_energy_lab_by_cell"]),
    )
    assert np.allclose(
        info["radiation_momentum_lab_global"],
        np.sum(info["radiation_momentum_lab_by_cell"], axis=0),
    )
    assert info["radiation_momentum_fluid_by_cell"].shape == (1, 3)
    assert info["event_count_labels"][2] == "effective_scatters"
