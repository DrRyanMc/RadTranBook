"""Phase 6 zero-velocity regression against the stationary IMC kernels."""

from __future__ import annotations

from statistics import NormalDist

import numpy as np

from IMC.fleck_cummings.src.IMC1D import (
    SimulationState as GraySimulationState,
    create_boundary as gray_create_boundary,
    emitted_particles as gray_emitted_particles,
    equilibrium_sample as gray_equilibrium_sample,
    move_particles as gray_move_particles,
    step as gray_step,
)
from MG_IMC.fleck_cummings.src.MG_IMC1D import (
    _emitted_particles_spherical_mg,
)
from MG_IMC.fleck_cummings.src.MG_IMC1DMoving import (
    C_LIGHT,
    create_state_from_particles,
    sample_moving_equilibrium_particles,
    sample_moving_face_source,
    sample_moving_volume_source,
    seed_moving_imc_random,
    step as moving_step,
    transport_particles,
)


def test_zero_velocity_transport_matches_gray_slab_kernel():
    mesh = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
    weights = np.array([1.0, 0.7, 1.3, 0.4, 0.9])
    mu = np.array([0.8, -0.6, 0.35, -0.9, 0.95])
    positions = np.array([0.2, 1.8, 1.2, 2.8, 2.6])
    cells = np.array([0, 1, 1, 2, 2], dtype=np.int64)
    times = np.array([0.0, 0.01, 0.0, 0.005, 0.02])
    opacity = np.array([0.4, 1.2, 0.1])
    dt = 0.04

    gray_weights = weights.copy()
    gray_mu = mu.copy()
    gray_times = times.copy()
    gray_positions = positions.copy()
    gray_cells = cells.copy()
    gray_stats = np.zeros(6, dtype=np.int64)
    gray_deposit, gray_scalar_intensity = gray_move_particles(
        gray_weights,
        gray_mu,
        gray_times,
        gray_positions,
        gray_cells,
        mesh,
        opacity,
        np.zeros_like(opacity),
        dt,
        (False, False),
        gray_stats,
        0.0,
    )

    directions = np.column_stack((mu, np.sqrt(1.0 - mu**2), np.zeros(len(mu))))
    moving_state = create_state_from_particles(
        weights=weights,
        directions_lab=directions,
        photon_energies_lab=np.full(len(weights), 2.0),
        positions=positions,
        times=times,
        cell_indices=cells,
        temperature=np.ones(3),
        material_velocity=np.zeros((3, 3)),
        mesh=mesh,
        energy_edges=np.array([0.0, 10.0]),
        eos=lambda temperature: 10.0 * temperature,
    )
    moving = transport_particles(
        moving_state,
        dt,
        mesh,
        np.array([0.0, 10.0]),
        opacity[None, :],
        np.ones(3),
        reflect=(False, False),
    )

    gray_alive = (gray_cells >= 0) & (gray_cells < len(mesh))
    assert np.allclose(moving_state.weights, gray_weights[gray_alive], rtol=2e-14)
    assert np.allclose(moving_state.dir_x, gray_mu[gray_alive], rtol=0.0, atol=0.0)
    assert np.allclose(moving_state.positions, gray_positions[gray_alive], atol=2e-14)
    assert np.array_equal(moving_state.cell_indices, gray_cells[gray_alive])
    assert np.allclose(
        moving.material_energy_exchange_lab, gray_deposit, rtol=2e-14
    )
    assert np.allclose(
        moving.track_length_energy_lab_by_fluid_group[0],
        gray_scalar_intensity * dt * np.diff(mesh, axis=1)[:, 0],
        rtol=2e-14,
    )
    assert np.isclose(
        np.sum(moving.boundary_energy_loss_lab),
        np.sum(gray_weights[~gray_alive]),
        rtol=2e-14,
    )
    assert np.array_equal(moving.event_counts[:4], gray_stats[:4])

    initial_radiation = np.sum(weights)
    moving_balance = (
        np.sum(moving_state.weights)
        + np.sum(moving.material_energy_exchange_lab)
        + np.sum(moving.boundary_energy_loss_lab)
    )
    gray_balance = (
        np.sum(gray_weights[gray_alive])
        + np.sum(gray_deposit)
        + np.sum(gray_weights[~gray_alive])
    )
    assert np.isclose(moving_balance, initial_radiation, atol=2e-14)
    assert np.isclose(gray_balance, initial_radiation, atol=2e-14)


def test_zero_velocity_source_energies_match_gray_slab_samplers():
    mesh = np.array([[0.0, 0.5], [0.5, 1.5], [1.5, 3.0]])
    temperature = np.array([0.8, 1.0, 1.2])
    opacity = np.array([0.4, 0.7, 1.1])
    dt = 0.03
    target = 6_000
    edges = np.array([0.0, 100.0])

    seed_moving_imc_random(7101)
    gray_equilibrium = gray_equilibrium_sample(target, temperature, mesh)
    seed_moving_imc_random(7101)
    moving_equilibrium = sample_moving_equilibrium_particles(
        target, temperature, mesh, edges, np.zeros((3, 3))
    )
    gray_equilibrium_by_cell = np.bincount(
        gray_equilibrium[4], weights=gray_equilibrium[0], minlength=3
    )
    assert np.allclose(
        moving_equilibrium.energy_lab_by_cell,
        gray_equilibrium_by_cell,
        rtol=2e-15,
    )

    seed_moving_imc_random(7102)
    gray_volume = gray_emitted_particles(
        target, temperature, dt, mesh, opacity
    )
    seed_moving_imc_random(7102)
    moving_volume = sample_moving_volume_source(
        target,
        temperature,
        dt,
        mesh,
        edges,
        opacity[None, :],
        np.zeros((3, 3)),
    )
    assert np.allclose(
        moving_volume.energy_lab_by_cell, gray_volume[5], rtol=2e-15
    )

    seed_moving_imc_random(7103)
    gray_face = gray_create_boundary(target, 1.1, dt)
    seed_moving_imc_random(7103)
    moving_face = sample_moving_face_source(
        target,
        temperature=1.1,
        dt=dt,
        mesh=mesh,
        energy_edges=edges,
        material_velocity=np.zeros((3, 3)),
        side=0,
    )
    assert np.isclose(
        np.sum(moving_face.weights), np.sum(gray_face[0]), rtol=2e-15
    )


def test_zero_velocity_full_step_matches_gray_slab_solver(capsys):
    mesh = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
    weights = np.array([1.0, 0.7, 1.3, 0.4, 0.9])
    mu = np.array([0.8, -0.6, 0.35, -0.9, 0.95])
    positions = np.array([0.2, 1.8, 1.2, 2.8, 2.6])
    cells = np.array([0, 1, 1, 2, 2], dtype=np.int64)
    times = np.array([0.0, 0.01, 0.0, 0.005, 0.02])
    opacity = np.array([0.4, 1.2, 0.1])
    temperature = np.ones(3)
    internal_energy = 10.0 * temperature
    dt = 0.04
    previous_total = np.sum(internal_energy) + np.sum(weights)

    gray_state = GraySimulationState(
        weights=weights.copy(),
        mus=mu.copy(),
        times=times.copy(),
        positions=positions.copy(),
        cell_indices=cells.copy(),
        internal_energy=internal_energy.copy(),
        temperature=temperature.copy(),
        radiation_temperature=np.zeros(3),
        time=0.0,
        previous_total_energy=previous_total,
    )
    gray_state, gray_info = gray_step(
        gray_state,
        Ntarget=0,
        Nboundary=0,
        Nsource=0,
        NMax=0,
        T_boundary=(0.0, 0.0),
        dt=dt,
        mesh=mesh,
        sigma_a_func=lambda value: opacity,
        inv_eos=lambda value: value / 10.0,
        cv=lambda value: np.full_like(value, 10.0),
        source=np.zeros(3),
        theta=0.0,
        use_scalar_intensity_Tr=False,
        T_emit_floor=2.0,
        geometry="slab",
    )
    capsys.readouterr()

    directions = np.column_stack((mu, np.sqrt(1.0 - mu**2), np.zeros(len(mu))))
    moving_state = create_state_from_particles(
        weights=weights,
        directions_lab=directions,
        photon_energies_lab=np.full(len(weights), 2.0),
        positions=positions,
        times=times,
        cell_indices=cells,
        temperature=temperature,
        material_velocity=np.zeros((3, 3)),
        mesh=mesh,
        energy_edges=np.array([0.0, 10.0]),
        eos=lambda value: 10.0 * value,
    )
    moving_state, moving_info = moving_step(
        moving_state,
        target=0,
        dt=dt,
        mesh=mesh,
        energy_edges=np.array([0.0, 10.0]),
        sigma_a_funcs=[lambda value: opacity],
        inv_eos=lambda value: value / 10.0,
        cv=lambda value: np.full_like(value, 10.0),
        theta=0.0,
        temperature_floor=2.0,
    )

    assert np.allclose(moving_state.weights, gray_state.weights, rtol=2e-14)
    assert np.allclose(moving_state.dir_x, gray_state.mus, rtol=0.0, atol=0.0)
    assert np.allclose(moving_state.positions, gray_state.positions, atol=2e-14)
    assert np.array_equal(moving_state.cell_indices, gray_state.cell_indices)
    assert np.allclose(moving_state.temperature, gray_state.temperature, rtol=2e-14)
    assert np.allclose(
        moving_state.radiation_temperature,
        gray_state.radiation_temperature,
        rtol=2e-14,
    )
    assert np.isclose(
        np.sum(moving_state.weights), gray_info["total_radiation_energy"], rtol=2e-14
    )
    assert np.isclose(
        np.sum(moving_info["boundary_energy_loss_lab"]),
        gray_info["boundary_loss"],
        rtol=2e-14,
    )
    assert abs(moving_info["energy_residual"]) < 2e-14
    assert abs(gray_info["energy_loss"]) < 2e-14


def test_zero_velocity_multigroup_source_matches_stationary_group_energies():
    # Choose a spherical cell with unit volume so only the source law is compared.
    unit_volume_radius = (3.0 / (4.0 * np.pi)) ** (1.0 / 3.0)
    stationary_mesh = np.array([[0.0, unit_volume_radius]])
    slab_mesh = np.array([[0.0, 1.0]])
    edges = np.array([0.0, 1.0, 3.0, 10.0])
    temperature = np.array([1.0])
    opacity = np.array([[1.0], [2.0], [4.0]])
    target = 12_000
    dt = 0.2

    stationary = _emitted_particles_spherical_mg(
        target, temperature, dt, stationary_mesh, opacity, edges
    )
    stationary_group_energy = stationary[6][:, 0]
    stationary_sampled_energy = np.bincount(
        stationary[5], weights=stationary[0], minlength=3
    )
    assert np.allclose(
        stationary_sampled_energy, stationary_group_energy, rtol=2e-15
    )

    group_fractions = []
    for seed in range(7201, 7213):
        seed_moving_imc_random(seed)
        moving = sample_moving_volume_source(
            target,
            temperature,
            dt,
            slab_mesh,
            edges,
            opacity,
            np.zeros((1, 3)),
        )
        assert np.isclose(
            np.sum(moving.weights), np.sum(stationary_group_energy), rtol=2e-15
        )
        group_fractions.append(
            np.bincount(moving.lab_groups, minlength=3) / target
        )

    expected = stationary_group_energy / np.sum(stationary_group_energy)
    observed = np.mean(group_fractions, axis=0)
    standard_error = np.sqrt(expected * (1.0 - expected) / (12 * target))
    critical = NormalDist().inv_cdf(1.0 - 1.0e-3 / (2.0 * len(expected)))
    assert np.all(np.abs(observed - expected) / standard_error <= critical)
