"""Tests for the shared published-problem driver."""

from __future__ import annotations

import numpy as np

from MG_IMC.fleck_cummings.problems.run_moving_material_published import (
    EquilibriumConfig,
    run_equilibrium,
)


def test_reduced_equilibrium_driver_writes_self_describing_artifact(tmp_path):
    output = tmp_path / "equilibrium.npz"
    config = EquilibriumConfig(
        preset="test",
        n_cells=4,
        particles=400,
        steps=2,
        velocities=(0.0, 0.6),
        seeds=(91, 92),
    )
    result = run_equilibrium(config, output)

    assert output.exists()
    assert result["census_lab_energy_density_normalized"].shape == (2, 2)
    assert result["path_lab_energy_density_normalized"].shape == (2, 2)
    assert result["fluid_radiation_temperature"].shape == (2, 2)
    assert np.all(np.isfinite(result["runtime_seconds"]))
    assert np.all(result["maximum_absolute_exchange_energy_residual"] < 2.0e-12)
    with np.load(output) as saved:
        assert saved["problem"] == "moving_equilibrium"
        assert saved["preset"] == "test"
        assert np.array_equal(saved["velocities_beta"], [0.0, 0.6])
        assert np.array_equal(saved["seeds"], [91, 92])
