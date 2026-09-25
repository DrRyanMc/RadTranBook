"""Phase 6 deterministic validation of moving-material frame contracts."""

from __future__ import annotations

from decimal import Decimal, getcontext

import numpy as np
import pytest

from MG_IMC.fleck_cummings.src.MG_IMC1DMoving import (
    C_LIGHT,
    create_state_from_particles,
    moving_face_flux_factor,
    transport_particles,
)
from MG_IMC.fleck_cummings.src.material_motion import (
    fluid_to_lab_photon,
    lab_energy_momentum_to_fluid,
    lab_to_fluid_photon,
)


def test_ultrarelativistic_round_trip_and_doppler_reciprocity():
    beta_direction = np.array([1.0, -2.0, 3.0])
    beta_direction /= np.linalg.norm(beta_direction)
    betas = (
        np.zeros(3),
        np.array([0.8, 0.0, 0.0]),
        np.array([0.0, 0.99, 0.0]),
        0.999999 * beta_direction,
    )
    directions = (
        np.array([1.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.2, -0.4, np.sqrt(0.8)]),
    )
    energy = 7.3

    for beta in betas:
        gamma = 1.0 / np.sqrt(1.0 - np.dot(beta, beta))
        for direction in directions:
            fluid = lab_to_fluid_photon(energy, *direction, *beta)
            lab = fluid_to_lab_photon(fluid[0], *fluid[1:4], *beta)
            assert fluid[4] > 0.0
            assert lab[4] > 0.0
            assert np.isclose(gamma**2 * fluid[4] * lab[4], 1.0, atol=3.0e-10)
            assert np.isclose(np.linalg.norm(fluid[1:4]), 1.0, atol=5.0e-13)
            assert np.isclose(lab[0], energy, rtol=3.0e-10, atol=2.0e-9)
            assert np.allclose(lab[1:4], direction, rtol=0.0, atol=5.0e-10)


@pytest.mark.parametrize("speed", [0.0, 0.6, 0.99])
def test_equilibrium_cell_moments_transform_analytically(speed):
    direction = np.array([1.0, -2.0, 0.5])
    direction /= np.linalg.norm(direction)
    beta = speed * direction
    gamma = 1.0 / np.sqrt(1.0 - speed**2)
    fluid_energy_density = 2.3
    lab_width = 1.7
    energy_lab = np.array([
        fluid_energy_density * gamma**2 * (1.0 + speed**2 / 3.0) * lab_width
    ])
    momentum_lab = np.array([
        (4.0 / 3.0) * fluid_energy_density * gamma**2
        * beta * lab_width / C_LIGHT
    ])

    energy_fluid, momentum_fluid = lab_energy_momentum_to_fluid(
        energy_lab,
        momentum_lab,
        beta[None, :] * C_LIGHT,
    )

    assert np.isclose(
        energy_fluid[0], fluid_energy_density * gamma * lab_width, rtol=3.0e-14
    )
    assert np.allclose(
        momentum_fluid[0],
        fluid_energy_density * gamma * beta * lab_width / (3.0 * C_LIGHT),
        rtol=3.0e-13,
        atol=2.0e-15,
    )


def test_specular_reflection_has_exact_wall_momentum_balance():
    direction = np.array([0.6, 0.8, 0.0])
    state = create_state_from_particles(
        weights=np.array([2.0]),
        directions_lab=np.array([direction]),
        photon_energies_lab=np.array([4.0]),
        positions=np.array([0.9]),
        times=np.array([0.0]),
        cell_indices=np.array([0]),
        temperature=np.array([1.0]),
        material_velocity=np.zeros((1, 3)),
        mesh=np.array([[0.0, 1.0]]),
        energy_edges=np.array([0.0, 10.0]),
        eos=lambda temperature: 10.0 * temperature,
    )
    initial_momentum = 2.0 * direction / C_LIGHT

    result = transport_particles(
        state,
        dt=0.01,
        mesh=np.array([[0.0, 1.0]]),
        energy_edges=np.array([0.0, 10.0]),
        sigma_a_true=np.zeros((1, 1)),
        fleck_factors=np.ones(1),
        reflect=(False, True),
    )

    final_direction = np.array([state.dir_x[0], state.dir_y[0], state.dir_z[0]])
    final_momentum = 2.0 * final_direction / C_LIGHT
    assert np.array_equal(final_direction, [-0.6, 0.8, 0.0])
    assert state.weights[0] == 2.0
    assert np.array_equal(result.event_counts, [2, 1, 0, 1, 1])
    assert np.all(result.material_energy_exchange_lab == 0.0)
    assert np.all(result.material_momentum_exchange_lab == 0.0)
    assert np.allclose(
        initial_momentum,
        final_momentum + np.sum(result.wall_momentum_exchange_lab, axis=0),
        rtol=0.0,
        atol=2.0e-17,
    )


@pytest.mark.parametrize("beta_normal", [0.0, 0.9, -0.9, 0.999999, -0.999999])
def test_normal_face_flux_factor_matches_high_precision_factored_form(beta_normal):
    getcontext().prec = 50
    beta_decimal = Decimal.from_float(beta_normal)
    expected = (
        (Decimal(1) + beta_decimal) ** 3
        * (Decimal(3) - beta_decimal)
        / Decimal(3)
    )
    observed = moving_face_flux_factor(0.0, beta_normal)
    assert observed > 0.0
    assert np.isclose(observed, float(expected), rtol=3.0e-15, atol=0.0)


@pytest.mark.parametrize("beta_tangent", [0.0, 0.3, 0.9, 0.999999])
def test_tangential_face_flux_factor_is_inverse_gamma(beta_tangent):
    assert np.isclose(
        moving_face_flux_factor(beta_tangent, 0.0),
        np.sqrt(1.0 - beta_tangent**2),
        rtol=2.0e-15,
        atol=0.0,
    )
