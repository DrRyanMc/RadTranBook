import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

"""Regression checks for the production thermal Compton collision kernel."""

import sys
from pathlib import Path

import numpy as np
from numba import njit


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "MG_IMC"))

from MG_IMC.fleck_cummings.src.compton_kernel import _kn_total_ratio, compton_scatter_3d
from MG_IMC.fleck_cummings.src.MG_IMC2D import _comb_mg


@njit
def _sample_candidates(energies, temperature, seed):
    np.random.seed(seed)
    fractional_deposit = 0.0
    null_collisions = 0
    maximum_norm_error = 0.0
    for energy in energies:
        _, out_x, out_y, out_z, deposit = compton_scatter_3d(
            energy, 0.0, 0.0, 1.0, temperature,
        )
        fractional_deposit += deposit / energy
        if deposit == 0.0:
            null_collisions += 1
        norm_error = abs(out_x**2 + out_y**2 + out_z**2 - 1.0)
        maximum_norm_error = max(maximum_norm_error, norm_error)
    count = energies.size
    return (
        fractional_deposit / count,
        null_collisions / count,
        maximum_norm_error,
    )


def _represented_planck_energies(temperature, count=300_000):
    with np.load(ROOT / "MG_IMC/data/compton_micro_32g_50t_conservative.npz") as data:
        edges = np.asarray(data["energy_edges_kev"])
    nodes, weights = np.polynomial.legendre.leggauss(64)
    group_energy = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        energy = 0.5 * (upper - lower) * nodes + 0.5 * (upper + lower)
        group_energy.append(
            0.5 * (upper - lower) * np.sum(
                weights * energy**3 / np.expm1(energy / temperature)
            )
        )
    probabilities = np.asarray(group_energy) / np.sum(group_energy)
    group_counts = np.floor(count * probabilities).astype(int)
    group_counts[np.argmax(probabilities)] += count - np.sum(group_counts)
    centers = np.sqrt(edges[:-1] * edges[1:])
    return np.repeat(centers, group_counts)


def _test_comb_preserves_photon_energy():
    weights = np.array([1.0, 2.0, 3.0])
    energies = np.array([1.23, 2.34, 7.89])
    cell_indices = np.zeros(3, dtype=np.int32)
    groups = np.array([0, 0, 1], dtype=np.int32)
    values = np.zeros(3)
    unchanged = _comb_mg(
        weights, energies, cell_indices, cell_indices, groups,
        values, values, values, values, values, 3, 1, 1, 2,
    )
    assert np.array_equal(unchanged[0], weights)
    assert np.array_equal(unchanged[9], energies)

    np.random.seed(91827)
    combined = _comb_mg(
        weights, energies, cell_indices, cell_indices, groups,
        values, values, values, values, values, 2, 1, 1, 2,
    )
    assert np.all(np.isin(combined[9], energies))
    assert np.isclose(np.sum(combined[10]), np.sum(weights) - np.sum(combined[0]))


def run_test():
    _test_comb_preserves_photon_energy()
    monoenergetic = np.full(300_000, 40.0)
    _, null_fraction, norm_error = _sample_candidates(monoenergetic, 0.0, 12345)
    expected_null_fraction = 1.0 - _kn_total_ratio(40.0 / 510.99895)
    assert abs(null_fraction - expected_null_fraction) < 2.0e-3
    assert norm_error < 1.0e-12

    planck_energies = _represented_planck_energies(10.0)
    heating, _, _ = _sample_candidates(planck_energies, 8.5, 24680)
    cooling, _, _ = _sample_candidates(planck_energies, 10.0, 13579)
    assert heating > 2.0e-3
    assert cooling < -1.0e-3

    print("Production Compton kernel passed")
    print(f"  40-keV null fraction : {null_fraction:.6f}")
    print(f"  expected             : {expected_null_fraction:.6f}")
    print(f"  Planck heating at 8.5: {heating:.6e}")
    print(f"  Planck cooling at 10 : {cooling:.6e}")


if __name__ == "__main__":
    run_test()