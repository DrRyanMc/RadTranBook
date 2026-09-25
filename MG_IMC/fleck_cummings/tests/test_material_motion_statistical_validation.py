"""Phase 6 repeated-seed validation of moving thermal source samplers."""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
from numpy.polynomial.legendre import leggauss

from MG_IMC.fleck_cummings.src.MG_IMC1DMoving import (
    A_RAD,
    C_LIGHT,
    _compute_Bg_1d,
    sample_moving_equilibrium_particles,
    sample_moving_face_source,
    sample_moving_volume_source,
    seed_moving_imc_random,
)


SEEDS = tuple(range(6201, 6213))
FAMILY_ALPHA = 1.0e-3


def _directions(batch):
    return np.column_stack((batch.dir_x, batch.dir_y, batch.dir_z))


def _simultaneous_normal_limit(metric_count):
    """Bonferroni two-sided normal critical value for one metric family."""
    return NormalDist().inv_cdf(1.0 - FAMILY_ALPHA / (2.0 * metric_count))


def _assert_pooled_means(observations, expected, variances):
    observations = np.asarray(observations)
    expected = np.asarray(expected)
    variances = np.asarray(variances)
    sample_count = observations.shape[0] * observations.shape[1]
    observed = np.mean(observations, axis=(0, 1))
    standard_error = np.sqrt(variances / sample_count)
    z_score = np.abs(observed - expected) / standard_error
    assert np.all(z_score <= _simultaneous_normal_limit(expected.size))


def _face_direction_moments(beta_tangent, beta_normal):
    """Independent Gauss-Legendre moments for density mu / D_L**4."""
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
    weight = mu[:, None] / doppler**4 * mu_weight[:, None] * phi_weight[None, :]
    direction = np.stack(
        np.broadcast_arrays(
            mu[:, None],
            transverse[:, None] * np.cos(phi)[None, :],
            transverse[:, None] * np.sin(phi)[None, :],
        ),
        axis=-1,
    )
    normalization = np.sum(weight)
    mean = np.sum(direction * weight[:, :, None], axis=(0, 1)) / normalization
    second = np.sum(direction**2 * weight[:, :, None], axis=(0, 1)) / normalization
    return mean, second - mean**2


def _stationary_volume_samples(target):
    batches = []
    group_fractions = []
    for seed in SEEDS:
        seed_moving_imc_random(seed)
        batch = sample_moving_volume_source(
            target,
            np.array([1.0]),
            0.2,
            np.array([[0.0, 1.0]]),
            np.array([0.0, 1.0, 3.0, 10.0]),
            np.array([[1.0], [2.0], [4.0]]),
            np.zeros((1, 3)),
        )
        batches.append(_directions(batch))
        group_fractions.append(
            np.bincount(batch.lab_groups, minlength=3) / target
        )
    return np.asarray(batches), np.asarray(group_fractions)


def _stationary_face_samples(target):
    batches = []
    for seed in SEEDS:
        seed_moving_imc_random(seed)
        batch = sample_moving_face_source(
            target,
            temperature=1.0,
            dt=0.1,
            mesh=np.array([[0.0, 1.0]]),
            energy_edges=np.array([0.0, 100.0]),
            material_velocity=np.zeros((1, 3)),
            side=0,
        )
        batches.append(_directions(batch))
    return np.asarray(batches)


def test_seed_contract_replays_python_and_numba_draws():
    def draw_batch():
        return sample_moving_volume_source(
            2_000,
            np.array([1.0]),
            0.2,
            np.array([[0.0, 1.0]]),
            np.array([0.0, 1.0, 3.0, 10.0]),
            np.array([[1.0], [2.0], [4.0]]),
            np.array([[0.2 * C_LIGHT, -0.1 * C_LIGHT, 0.0]]),
        )

    seed_moving_imc_random(6199)
    first = draw_batch()
    seed_moving_imc_random(6199)
    second = draw_batch()

    for name in (
        "weights",
        "dir_x",
        "dir_y",
        "dir_z",
        "photon_energies_lab",
        "lab_groups",
        "positions",
        "times",
        "cell_indices",
    ):
        assert np.array_equal(getattr(first, name), getattr(second, name))


def test_stationary_volume_ensemble_is_isotropic_and_has_expected_groups():
    target = 12_000
    directions, group_fractions = _stationary_volume_samples(target)
    _assert_pooled_means(directions, np.zeros(3), np.full(3, 1.0 / 3.0))

    edges = np.array([0.0, 1.0, 3.0, 10.0])
    opacity = np.array([1.0, 2.0, 4.0])
    planck_fraction = _compute_Bg_1d(edges, np.array([1.0]))[:, 0]
    probability = opacity * planck_fraction
    probability /= np.sum(probability)
    observed_probability = np.mean(group_fractions, axis=0)
    standard_error = np.sqrt(
        probability * (1.0 - probability) / (len(SEEDS) * target)
    )
    z_score = np.abs(observed_probability - probability) / standard_error
    assert np.all(z_score <= _simultaneous_normal_limit(len(probability)))


def test_moving_volume_ensemble_recovers_energy_and_momentum():
    target = 12_000
    beta = np.array([0.2, -0.3, 0.45])
    gamma = 1.0 / np.sqrt(1.0 - np.dot(beta, beta))
    expected_energy = gamma * A_RAD * C_LIGHT * 2.0 * 0.2
    directions = []
    for seed in SEEDS:
        seed_moving_imc_random(seed)
        batch = sample_moving_volume_source(
            target,
            np.array([1.0]),
            0.2,
            np.array([[0.0, 1.0]]),
            np.array([0.0, 100.0]),
            np.array([[2.0]]),
            beta[None, :] * C_LIGHT,
        )
        assert np.isclose(np.sum(batch.weights), expected_energy, rtol=2.0e-15)
        directions.append(_directions(batch))
    directions = np.asarray(directions)
    pooled_variance = np.var(directions.reshape(-1, 3), axis=0, ddof=1)
    _assert_pooled_means(directions, beta, pooled_variance)


def test_moving_equilibrium_ensemble_recovers_stress_energy_momentum():
    target = 12_000
    beta = np.array([0.2, -0.3, 0.45])
    beta_squared = np.dot(beta, beta)
    gamma = 1.0 / np.sqrt(1.0 - beta_squared)
    temperature = 1.3
    expected_energy = (
        A_RAD * temperature**4 * gamma**2 * (1.0 + beta_squared / 3.0)
    )
    expected_direction = 4.0 * beta / (3.0 + beta_squared)
    directions = []
    for seed in SEEDS:
        seed_moving_imc_random(seed)
        batch = sample_moving_equilibrium_particles(
            target,
            np.array([temperature]),
            np.array([[0.0, 1.0]]),
            np.array([0.0, 100.0]),
            beta[None, :] * C_LIGHT,
        )
        assert np.isclose(np.sum(batch.weights), expected_energy, rtol=2.0e-15)
        directions.append(_directions(batch))
    directions = np.asarray(directions)
    pooled_variance = np.var(directions.reshape(-1, 3), axis=0, ddof=1)
    _assert_pooled_means(directions, expected_direction, pooled_variance)


def test_stationary_face_ensemble_recovers_cosine_law():
    target = 12_000
    directions = _stationary_face_samples(target)
    _assert_pooled_means(
        directions,
        np.array([2.0 / 3.0, 0.0, 0.0]),
        np.array([1.0 / 18.0, 1.0 / 4.0, 1.0 / 4.0]),
    )


def test_moving_face_ensemble_matches_independent_quadrature():
    target = 12_000
    beta_normal = 0.4
    beta_tangent = 0.3
    expected_mean, expected_variance = _face_direction_moments(
        beta_tangent, beta_normal
    )
    directions = []
    for seed in SEEDS:
        seed_moving_imc_random(seed)
        batch = sample_moving_face_source(
            target,
            temperature=1.2,
            dt=0.08,
            mesh=np.array([[0.0, 1.0]]),
            energy_edges=np.array([0.0, 100.0]),
            material_velocity=np.array(
                [[beta_normal * C_LIGHT, beta_tangent * C_LIGHT, 0.0]]
            ),
            side=0,
        )
        directions.append(_directions(batch))
    _assert_pooled_means(np.asarray(directions), expected_mean, expected_variance)


def test_volume_and_face_errors_decrease_at_four_times_the_sample_count():
    small_count = 3_000
    large_count = 12_000
    volume_small, _ = _stationary_volume_samples(small_count)
    volume_large, _ = _stationary_volume_samples(large_count)
    face_small = _stationary_face_samples(small_count)
    face_large = _stationary_face_samples(large_count)

    volume_expected = np.zeros(3)
    volume_scale = np.sqrt(np.full(3, 1.0 / 3.0))
    face_expected = np.array([2.0 / 3.0, 0.0, 0.0])
    face_scale = np.sqrt(np.array([1.0 / 18.0, 1.0 / 4.0, 1.0 / 4.0]))

    def normalized_rmse(samples, expected, scale):
        seed_means = np.mean(samples, axis=1)
        return np.sqrt(np.mean(((seed_means - expected) / scale) ** 2))

    volume_ratio = normalized_rmse(
        volume_small, volume_expected, volume_scale
    ) / normalized_rmse(volume_large, volume_expected, volume_scale)
    face_ratio = normalized_rmse(
        face_small, face_expected, face_scale
    ) / normalized_rmse(face_large, face_expected, face_scale)

    # Independent Monte Carlo means should approach the expected ratio of two.
    assert 1.4 <= volume_ratio <= 2.8
    assert 1.4 <= face_ratio <= 2.8
