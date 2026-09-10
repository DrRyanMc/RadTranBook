import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

"""Regression checks for the 1-D multigroup Kompaneets frequency split."""

import os
import sys

import numpy as np


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import DiscreteOrdinates.src.mg_sn_solver
from DiscreteOrdinates.src.kompaneets import (
    _densmore_linear_number_update,
    operator_split_kompaneets_update,
)


C_LIGHT = 29.9792458
A_RAD = 0.01372


def _energy_grid(groups=32):
    edges = np.geomspace(1.0e-3, 200.0, groups + 1)
    means = (
        0.75 * (edges[1:]**4 - edges[:-1]**4)
        / (edges[1:]**3 - edges[:-1]**3)
    )
    return edges, means


def run_equilibrium_test():
    edges, means = _energy_grid()
    temperature = 1.5
    widths = np.diff(edges)
    number = 1.0e-8 * means**2 * np.exp(-means / temperature) * widths
    phi = C_LIGHT * means * number
    phi_star = [np.array([[value]]) for value in phi]
    psi_star = [np.full((1, 2, 1), value) for value in phi]
    result = operator_split_kompaneets_update(
        phi_star, psi_star, np.array([[temperature]]),
        np.array([[temperature]]), 1.0, C_LIGHT, lambda energy: energy,
        edges, means, 1.0, return_diagnostics=True,
    )
    assert np.allclose(np.asarray(result[0]).ravel(), phi, rtol=1.0e-12, atol=1.0e-18)
    assert result[-1]["photon_number_error"] < 1.0e-18
    assert result[-1]["energy_error"] < 1.0e-14
    print("Kompaneets fitted-Wien equilibrium passed")


def run_densmore_equations_test():
    edges, means = _energy_grid(5)
    widths = np.diff(edges)
    temperature = 2.5
    opacity = 0.7
    timestep = 0.03
    number_star = np.array([0.3, 1.1, 0.7, 0.2, 0.05])[:, None, None]
    actual = _densmore_linear_number_update(
        number_star, np.array([[temperature]]), timestep, C_LIGHT,
        edges, means, opacity, 511.0,
    ).ravel()
    zero_occupation = _densmore_linear_number_update(
        number_star, np.array([[temperature]]), timestep, C_LIGHT,
        edges, means, opacity, 511.0, occupation=np.zeros_like(number_star),
    ).ravel()
    assert np.array_equal(actual, zero_occupation)

    def equation_20_flux(energy_density):
        exponentials = np.exp(means / temperature)
        flux = np.zeros(means.size + 1)
        flux[1:-1] = edges[1:-1]**4 / 511.0 * (
            exponentials[1:] * energy_density[1:] / means[1:]**3
            - exponentials[:-1] * energy_density[:-1] / means[:-1]**3
        ) / (exponentials[1:] - exponentials[:-1])
        return flux

    matrix = np.eye(means.size)
    for column in range(means.size):
        basis = np.zeros(means.size)
        basis[column] = 1.0
        flux = equation_20_flux(basis)
        equation_19_rate = (
            opacity * C_LIGHT * means / widths
            * (flux[1:] - flux[:-1])
        )
        matrix[:, column] -= timestep * equation_19_rate
    energy_density_star = number_star.ravel() * means / widths
    energy_density_expected = np.linalg.solve(matrix, energy_density_star)
    expected = energy_density_expected * widths / means
    assert np.allclose(actual, expected, rtol=2.0e-14, atol=1.0e-14)
    print("Densmore Eqs. 19--20 coefficient regression passed")


def run_conservation_test():
    edges, means = _energy_grid(12)
    rng = np.random.default_rng(91827)
    phi_values = 1.0e-3 + rng.random(means.size)
    phi_star = [np.array([[value]]) for value in phi_values]
    psi_star = [np.full((1, 2, 1), value) for value in phi_values]
    heat_capacity = 20.0
    temperature = np.array([[2.0]])
    material_energy = heat_capacity * temperature
    result = operator_split_kompaneets_update(
        phi_star, psi_star, material_energy, temperature, 0.2, C_LIGHT,
        lambda energy: energy / heat_capacity, edges, means, 1.0,
        tolerance=1.0e-11, max_iterations=100, return_diagnostics=True,
    )
    phi_new, psi_new, energy_new, temperature_new, _, diagnostics = result
    number_before = np.sum(phi_values / (C_LIGHT * means))
    number_after = np.sum(np.array([value[0, 0] for value in phi_new]) / (C_LIGHT * means))
    total_before = material_energy + np.sum(phi_values) / C_LIGHT
    total_after = energy_new + sum(phi_new) / C_LIGHT
    assert diagnostics["converged"]
    assert abs(number_after - number_before) < 1.0e-12
    assert np.allclose(total_after, total_before, atol=1.0e-12)
    assert all(np.all(value >= 0.0) for value in phi_new)
    assert all(np.all(value >= 0.0) for value in psi_new)
    assert np.all(temperature_new > 0.0)
    print("Kompaneets photon and total-energy conservation passed")


def run_stimulated_equilibrium_test():
    edges, means = _energy_grid(128)
    temperature = 2.0
    fugacity = 0.4
    scalar_capacity = (
        A_RAD * C_LIGHT * 15.0 / np.pi**4
        * (edges[1:]**4 - edges[:-1]**4) / 4.0
    )
    occupation = 1.0 / (
        np.exp(np.minimum(means / temperature, 700.0)) / fugacity - 1.0
    )
    phi_values = scalar_capacity * occupation
    phi_star = [np.array([[value]]) for value in phi_values]
    psi_star = [np.full((1, 2, 1), value) for value in phi_values]
    result = operator_split_kompaneets_update(
        phi_star, psi_star, np.array([[0.0]]), np.array([[temperature]]),
        0.1, C_LIGHT, lambda energy: np.full((1, 1), temperature),
        edges, means, 1.0, induced_scattering=True,
        group_scalar_capacity=scalar_capacity, tolerance=1.0e-12,
        max_iterations=100, relaxation=0.5, return_diagnostics=True,
    )
    final = np.array([value.item() for value in result[0]])
    assert np.allclose(final, phi_values, rtol=1.0e-11, atol=1.0e-18)
    assert result[-1]["photon_number_error"] < 1.0e-16
    assert result[-1]["energy_error"] < 1.0e-14
    print("Stimulated Kompaneets Bose--Einstein equilibrium passed")


def run_solver_integration_test():
    groups, cells, order, ordinates = 4, 1, 1, 2
    nodes = order + 1
    edges, means = _energy_grid(groups)
    initial_values = np.array([0.1, 0.5, 1.0, 0.2])
    phi = [np.full((cells, nodes), value) for value in initial_values]
    psi = [np.full((cells, ordinates, nodes), value) for value in initial_values]
    temperature = np.ones((cells, nodes))

    def zero(values):
        return np.zeros_like(values)

    def eos(values):
        return 100.0 * values

    def inv_eos(values):
        return values / 100.0

    def heat_capacity(values):
        return np.full_like(values, 100.0)

    sources = [np.zeros((cells, ordinates, nodes)) for _ in range(groups)]

    def boundary(_time):
        return np.zeros((ordinates, nodes))

    histories, temperatures, _, times = mg_sn_solver.mg_temp_solve_dmd_inc(
        cells, 1.0, groups, [zero] * groups, [zero] * groups,
        [zero] * groups, [zero] * groups, sources, ordinates, boundary,
        eos, inv_eos, heat_capacity, phi, psi, temperature,
        dt_min=1.0e-4, dt_max=1.0e-4, tfinal=1.0e-4,
        tolerance=1.0e-9, Linf_tol=1.0e-8, maxits=20,
        order=order, K=5, R=1, reflect_left=True, reflect_right=True,
        kompaneets_options={
            "energy_edges_kev": edges,
            "group_mean_energy_kev": means,
            "thomson_opacity": 1.0,
            "tolerance": 1.0e-10,
            "max_iterations": 50,
        },
    )
    assert times[-1] == 1.0e-4
    assert np.isfinite(np.asarray(histories[-1])).all()
    assert np.isfinite(np.asarray(temperatures[-1])).all()
    final_values = np.array([value[0, 0] for value in histories[-1]])
    assert np.max(np.abs(final_values - initial_values)) > 1.0e-12
    print("1-D S_N Kompaneets integration passed")


if __name__ == "__main__":
    run_equilibrium_test()
    run_densmore_equations_test()
    run_conservation_test()
    run_stimulated_equilibrium_test()
    run_solver_integration_test()
