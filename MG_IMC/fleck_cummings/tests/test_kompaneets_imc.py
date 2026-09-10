"""Regression checks for the implicit Kompaneets IMC census update.

Run from the repository root with
``python -m MG_IMC.fleck_cummings.tests.test_kompaneets_imc``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

import random
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from MG_IMC.fleck_cummings.src.MG_IMC1D import init_simulation, step
from MG_IMC.fleck_cummings.src.kompaneets_imc import implicit_kompaneets_census_update


C_LIGHT = 29.98
A_RAD = 0.01372


def _zero_opacities(groups):
    return [lambda values: np.zeros_like(values) for _ in range(groups)]


def run_census_conservation_test():
    np.random.seed(8401)
    edges = np.linspace(0.0, 100.0, 101)
    groups = np.array([1, 5, 9, 14, 22, 40, 60, 85], dtype=np.int32)
    weights = np.linspace(1.0e-5, 2.0e-5, groups.size)
    cells = np.zeros(groups.size, dtype=np.int64)
    energies = 0.5 * (edges[groups] + edges[groups + 1])
    material = np.array([0.10])
    temperature = np.array([10.0])
    before = material.sum() + weights.sum()
    result = implicit_kompaneets_census_update(
        weights, cells, groups, energies, np.ones(1), material, temperature,
        1.0, edges, 1.0, C_LIGHT, lambda value: value / 0.01,
        tolerance=1.0e-8, max_iterations=100,
    )
    new_weights, new_groups, new_energies, new_material, new_temperature, diagnostics = result
    after = new_material.sum() + new_weights.sum()
    assert abs(after - before) < 1.0e-12
    assert diagnostics["energy_error"] < 1.0e-12
    assert diagnostics["photon_number_error"] < 1.0e-12
    assert np.all(new_weights >= 0.0)
    assert np.all((new_groups >= 0) & (new_groups < 100))
    assert np.all((new_energies >= edges[new_groups]) & (new_energies < edges[new_groups + 1]))
    assert np.all(new_temperature > 0.0)
    print("IMC census Kompaneets conservation passed")


def run_1d_split_integration_test():
    np.random.seed(8402)
    random.seed(8402)
    group_count = 20
    edges = np.linspace(0.0, 100.0, group_count + 1)
    mesh = np.array([[0.0, 1.0]])
    heat_capacity = 0.01
    eos = lambda temperature: heat_capacity * temperature
    inv_eos = lambda energy: energy / heat_capacity
    state = init_simulation(
        2000, np.array([10.0]), np.array([1.0]), mesh, edges, eos, inv_eos,
        Ntarget_ic=2000,
    )
    before = state.previous_total_energy
    state, info = step(
        state, 0, 0, 0, 0, (0.0, 0.0), 1.0, mesh, edges,
        _zero_opacities(group_count), inv_eos,
        lambda values: np.full_like(values, heat_capacity), np.zeros(1),
        reflect=(True, True), use_scalar_intensity_Tr=False,
        kompaneets_options={
            "thomson_opacity": 1.0,
            "tolerance": 1.0e-8,
            "max_iterations": 100,
        },
    )
    assert info["kompaneets"] is not None
    assert abs(info["total_energy"] - before) < 1.0e-11
    assert abs(info["energy_residual"]) < 1.0e-11
    assert state.temperature[0] < 10.0
    print("1-D IMC Kompaneets operator split passed")


if __name__ == "__main__":
    run_census_conservation_test()
    run_1d_split_integration_test()
