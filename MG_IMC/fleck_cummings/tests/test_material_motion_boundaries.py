"""Phase 4 tests for moving blackbody face sources and boundary accounting."""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

from MG_IMC.fleck_cummings.src.MG_IMC1DMoving import (
    A_RAD,
    C_LIGHT,
    build_moving_face_angular_table,
    create_state_from_particles,
    moving_face_flux_factor,
    sample_moving_face_source,
    step,
)


def _directions(batch):
    return np.column_stack((batch.dir_x, batch.dir_y, batch.dir_z))


def _face_direction_moments(beta_tangent, beta_normal):
    mu_node, mu_weight = leggauss(180)
    phi_node, phi_weight = leggauss(240)
    mu = 0.5 * (mu_node + 1.0)
    mu_weight *= 0.5
    phi = np.pi * phi_node
    phi_weight *= np.pi
    transverse = np.sqrt(1.0 - mu**2)
    doppler = (
        1.0
        - beta_normal * mu[:, None]
        - beta_tangent * transverse[:, None] * np.cos(phi)[None, :]
    )
    quadrature_weight = (
        mu[:, None] / doppler**4 * mu_weight[:, None] * phi_weight[None, :]
    )
    local_direction = np.stack(
        np.broadcast_arrays(
            mu[:, None],
            transverse[:, None] * np.cos(phi)[None, :],
            transverse[:, None] * np.sin(phi)[None, :],
        ),
        axis=-1,
    )
    normalization = np.sum(quadrature_weight)
    mean = np.sum(
        local_direction * quadrature_weight[:, :, None], axis=(0, 1)
    ) / normalization
    second = np.sum(
        local_direction**2 * quadrature_weight[:, :, None], axis=(0, 1)
    ) / normalization
    return mean, second - mean**2


@pytest.mark.parametrize("seed", [4101, 4102, 4103])
def test_stationary_face_recovers_cosine_law_and_blackbody_flux(seed):
    np.random.seed(seed)
    target = 30_000
    dt = 0.1
    particles = sample_moving_face_source(
        target,
        temperature=1.0,
        dt=dt,
        mesh=np.array([[0.0, 1.0]]),
        energy_edges=np.array([0.0, 100.0]),
        material_velocity=np.zeros((1, 3)),
        side=0,
    )
    directions = _directions(particles)
    expected_energy = A_RAD * C_LIGHT * dt / 4.0
    mu_standard_error = math.sqrt(1.0 / 18.0 / target)

    assert np.isclose(np.sum(particles.weights), expected_energy, rtol=2.0e-15)
    assert abs(np.mean(directions[:, 0]) - 2.0 / 3.0) <= 6.0 * mu_standard_error
    assert abs(np.mean(directions[:, 1])) <= 6.0 / np.sqrt(target)
    assert abs(np.mean(directions[:, 2])) <= 6.0 / np.sqrt(target)
    assert np.all(directions[:, 0] >= 0.0)
    assert np.allclose(np.linalg.norm(directions, axis=1), 1.0)


@pytest.mark.parametrize(
    ("beta_tangent", "beta_normal"),
    [
        (0.0, 0.0),
        (0.3, 0.4),
        (0.0, 0.9),
        (0.9, 0.0),
        (0.0, 0.999),
        (0.0, -0.999),
        (0.6, math.sqrt(0.999**2 - 0.6**2)),
    ],
)
def test_face_table_normalizes_published_density(beta_tangent, beta_normal):
    table = build_moving_face_angular_table(beta_tangent, beta_normal)
    analytic_integral = (
        np.pi * table.gamma**6
        * moving_face_flux_factor(beta_tangent, beta_normal)
    )
    assert np.isclose(
        table.normalization_integral,
        analytic_integral,
        rtol=2.5e-4,
    )


def test_moving_face_angular_moments_match_independent_quadrature():
    np.random.seed(4110)
    target = 60_000
    beta_normal = 0.4
    beta_tangent = 0.3
    beta = np.array([beta_normal, beta_tangent, 0.0])
    particles = sample_moving_face_source(
        target,
        temperature=1.2,
        dt=0.08,
        mesh=np.array([[0.0, 1.0]]),
        energy_edges=np.array([0.0, 100.0]),
        material_velocity=beta[None, :] * C_LIGHT,
        side=0,
    )

    expected_mean, expected_variance = _face_direction_moments(
        beta_tangent, beta_normal
    )
    observed_mean = np.mean(_directions(particles), axis=0)
    six_sigma = 6.0 * np.sqrt(expected_variance / target) + 5.0e-4
    gamma = 1.0 / np.sqrt(1.0 - np.dot(beta, beta))
    expected_energy = (
        A_RAD * C_LIGHT * 1.2**4 * 0.25 * gamma**2
        * moving_face_flux_factor(beta_tangent, beta_normal) * 0.08
    )
    assert np.all(np.abs(observed_mean - expected_mean) <= six_sigma)
    assert np.isclose(np.sum(particles.weights), expected_energy, rtol=2.0e-15)


def test_right_face_uses_inward_normal_and_adjacent_cell_velocity():
    np.random.seed(4111)
    beta = np.array([0.2, 0.3, 0.0])
    particles = sample_moving_face_source(
        20_000,
        temperature=1.0,
        dt=0.04,
        mesh=np.array([[0.0, 0.5], [0.5, 1.0]]),
        energy_edges=np.array([0.0, 100.0]),
        material_velocity=np.array([[0.0, 0.0, 0.0], beta * C_LIGHT]),
        side=1,
    )
    directions = _directions(particles)
    gamma = 1.0 / np.sqrt(1.0 - np.dot(beta, beta))
    expected_energy = (
        A_RAD * C_LIGHT * 0.25 * gamma**2
        * moving_face_flux_factor(0.3, -0.2) * 0.04
    )
    assert np.all(directions[:, 0] <= 0.0)
    assert np.all(particles.positions == 1.0)
    assert np.all(particles.cell_indices == 1)
    assert np.isclose(np.sum(particles.weights), expected_energy, rtol=2.0e-15)


def test_step_accounts_for_face_energy_and_momentum_injection():
    np.random.seed(4112)
    mesh = np.array([[0.0, 1.0]])
    energy_edges = np.array([0.0, 100.0])
    state = create_state_from_particles(
        weights=np.empty(0),
        directions_lab=np.empty((0, 3)),
        photon_energies_lab=np.empty(0),
        positions=np.empty(0),
        times=np.empty(0),
        cell_indices=np.empty(0, dtype=np.int64),
        temperature=np.array([1.0]),
        material_velocity=np.zeros((1, 3)),
        mesh=mesh,
        energy_edges=energy_edges,
        eos=lambda temperature: 100.0 * temperature,
    )

    state, info = step(
        state,
        target=0,
        dt=0.003,
        mesh=mesh,
        energy_edges=energy_edges,
        sigma_a_funcs=[lambda temperature: np.zeros_like(temperature)],
        inv_eos=lambda internal_energy: internal_energy / 100.0,
        cv=lambda temperature: np.full_like(temperature, 100.0),
        boundary_target=(4_000, 0),
        boundary_temperature=(1.0, 0.0),
    )

    expected_injection = A_RAD * C_LIGHT * 0.003 / 4.0
    assert np.isclose(info["boundary_energy_injection_lab"][0], expected_injection)
    assert info["boundary_energy_injection_lab"][1] == 0.0
    assert np.all(info["boundary_energy_loss_lab"] == 0.0)
    assert abs(info["energy_residual"]) < 5.0e-14
    assert np.linalg.norm(info["momentum_residual_lab"], ord=np.inf) < 5.0e-16
    assert np.allclose(
        np.sum(state.radiation_momentum_lab, axis=0),
        np.sum(info["boundary_momentum_injection_lab"], axis=0),
    )
