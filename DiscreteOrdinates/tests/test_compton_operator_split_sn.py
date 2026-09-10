import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

"""Conservation check for the multigroup S_N Compton split update.

This is a manufactured two-group Compton table, not a replacement for CSK
data.  Group 0 loses energy to group 1, with the unrecovered energy deposited
in the material.  It exercises the same in-/out-scattering bookkeeping used
for physical Compton tables.

Run from ``DiscreteOrdinates`` with::

    python problems/test_compton_operator_split_sn.py
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from DiscreteOrdinates.src.compton import ComptonMatrices, operator_split_compton_update


# Kept local so this conservation test does not need to JIT-compile the legacy
# transport sweep before exercising the purely local Compton update.
C_LIGHT = 29.9792458  # cm / ns


def run_test():
    # One spatial cell with order zero and two discrete directions.  The
    # quadrature weights used by this S_N code sum to one, hence psi = phi for
    # an isotropic state.
    phi_star = [np.array([[4.0]]), np.array([[0.0]])]
    psi_star = [
        np.full((1, 2, 1), 4.0),
        np.zeros((1, 2, 1)),
    ]
    e_star = np.array([[1.0]])
    T_star = e_star.copy()
    dt = 1.0e-2

    # source -> destination.  A group-0 photon has 0.5 cm^-1 total loss
    # opacity, while only 0.25 cm^-1 of its incident energy reaches group 1.
    # The difference is the Compton heating opacity.
    matrices = ComptonMatrices(
        in_scatter=np.array([[0.0, 0.25], [0.0, 0.0]]),
        out_scatter=np.array([[0.0, 0.50], [0.0, 0.0]]),
        # Exercise the same net-out-scattering path used by the
        # Planck-linearized operator.
        planck_net_out=np.array([0.50, 0.0]),
    )

    def matrix_func(_temperature):
        return matrices

    def inv_eos(energy):
        return energy

    def zero_planck(temperature):
        return np.zeros_like(temperature)

    phi_new, psi_new, e_new, T_new, iterations = operator_split_compton_update(
        phi_star, psi_star, e_star, T_star, dt, C_LIGHT, inv_eos,
        [zero_planck, zero_planck], matrix_func, induced_mode="planck",
    )

    energy_before = e_star + sum(phi_star) / C_LIGHT
    energy_after = e_new + sum(phi_new) / C_LIGHT
    error = float(np.max(np.abs(energy_after - energy_before)))
    assert error < 1.0e-12, f"Compton energy conservation error: {error:.3e}"
    assert float(phi_new[0][0, 0]) < float(phi_star[0][0, 0])
    assert float(phi_new[1][0, 0]) > 0.0
    assert float(e_new[0, 0]) > float(e_star[0, 0])
    assert np.all(psi_new[0] >= 0.0) and np.all(psi_new[1] >= 0.0)
    assert np.allclose(e_new, T_new)

    print("Compton split update passed")
    print(f"  nonlinear iterations : {iterations}")
    print(f"  phi*                 : {[float(p[0, 0]) for p in phi_star]}")
    print(f"  phi^(n+1)            : {[float(p[0, 0]) for p in phi_new]}")
    print(f"  material energy gain : {float(e_new[0, 0] - e_star[0, 0]):.8e}")
    print(f"  total-energy error   : {error:.3e}")


def run_nonlinear_test():
    """Exercise the unlinearized induced-scattering iteration and closure."""
    # These manufactured coefficients have a positive, nonzero induced term.
    # The initial state is an exact discrete fixed point of the nonlinear
    # equation, so the test detects accidentally evaluating induced terms with
    # a Planck function or with the old-time intensity.
    phi_star = [np.array([[1.0]]), np.array([[2.0]])]
    psi_star = [
        np.full((1, 2, 1), 1.0),
        np.full((1, 2, 1), 2.0),
    ]
    e_star = np.array([[1.0]])
    T_star = e_star.copy()
    matrices = ComptonMatrices(
        in_scatter=np.array([[0.0, 1.0], [0.25, 0.0]]),
        out_scatter=np.array([[0.0, 0.50], [0.40, 0.0]]),
        out_induced=np.array([[0.0, 0.10], [0.20, 0.0]]),
        in_induced=np.array([[0.0, 0.10], [0.10, 0.0]]),
    )

    def matrix_func(_temperature):
        return matrices

    def inv_eos(energy):
        return energy

    def zero_planck(temperature):
        return np.zeros_like(temperature)

    phi_new, _, e_new, T_new, iterations = operator_split_compton_update(
        phi_star, psi_star, e_star, T_star, 1.0e-2, C_LIGHT, inv_eos,
        [zero_planck, zero_planck], matrix_func, induced_mode="nonlinear",
        tolerance=1.0e-12,
    )
    energy_before = e_star + sum(phi_star) / C_LIGHT
    energy_after = e_new + sum(phi_new) / C_LIGHT
    error = float(np.max(np.abs(energy_after - energy_before)))
    assert error < 1.0e-12, f"nonlinear Compton energy error: {error:.3e}"
    assert np.allclose(phi_new[0], phi_star[0], atol=1.0e-12)
    assert np.allclose(phi_new[1], phi_star[1], atol=1.0e-12)
    assert np.allclose(e_new, e_star, atol=1.0e-12)
    assert np.allclose(T_new, T_star, atol=1.0e-12)
    print("Nonlinear Compton split update passed")
    print(f"  nonlinear iterations : {iterations}")
    print(f"  total-energy error   : {error:.3e}")


def run_positivity_limiter_test():
    """Ensure an indefinite lagged group solve is limited, not propagated."""
    phi_star = [np.array([[1.0]]), np.array([[0.0]])]
    psi_star = [
        np.full((1, 2, 1), 1.0),
        np.zeros((1, 2, 1)),
    ]
    e_star = np.array([[1.0]])
    T_star = e_star.copy()

    # With c*dt=1, this manufactured in-scattering matrix gives a raw linear
    # solution with negative components.  It represents the loss of the
    # M-matrix property that can occur in a strongly lagged induced update.
    matrices = ComptonMatrices(
        in_scatter=np.array([[0.0, 2.0], [2.0, 0.0]]),
        out_scatter=np.zeros((2, 2)),
    )

    def matrix_func(_temperature):
        return matrices

    def inv_eos(energy):
        return energy

    def zero_planck(temperature):
        return np.zeros_like(temperature)

    phi_new, psi_new, e_new, _, _, diagnostics = operator_split_compton_update(
        phi_star, psi_star, e_star, T_star, 1.0 / C_LIGHT, C_LIGHT, inv_eos,
        [zero_planck, zero_planck], matrix_func, induced_mode="wien",
        max_iterations=1, on_nonconvergence="accept", return_diagnostics=True,
    )
    energy_before = e_star + sum(phi_star) / C_LIGHT
    energy_after = e_new + sum(phi_new) / C_LIGHT
    error = float(np.max(np.abs(energy_after - energy_before)))
    assert diagnostics["positivity_limited_cells"] == 1
    assert diagnostics["minimum_positivity_damping"] == 0.0
    assert all(np.all(phi >= 0.0) for phi in phi_new)
    assert all(np.all(psi >= 0.0) for psi in psi_new)
    assert error < 1.0e-12, f"limited Compton energy error: {error:.3e}"
    print("Compton positivity limiter passed")
    print(f"  total-energy error   : {error:.3e}")


def run_compact_conservative_test():
    """Match compact conservative data against all four expanded matrices."""
    phi_star = [np.array([[1.0, 0.5]]), np.array([[0.25, 0.75]])]
    psi_star = [values[:, None, :] for values in phi_star]
    e_star = np.array([[2.0, 3.0]])
    T_star = e_star.copy()
    out_scatter = np.array([
        [[[0.0, 0.0]], [[0.1, 0.2]]],
        [[[0.4, 0.3]], [[0.0, 0.0]]],
    ])
    group_energy = np.array([1.0, 2.0])
    group_capacity = np.array([3.0, 5.0])
    energy_ratio = group_energy[None, :, None, None] / group_energy[:, None, None, None]
    in_scatter = out_scatter * energy_ratio

    compact = ComptonMatrices(
        in_scatter=None,
        out_scatter=out_scatter,
        group_mean_energy=group_energy,
        group_scalar_capacity=group_capacity,
    )
    expanded = ComptonMatrices(
        in_scatter=in_scatter,
        out_scatter=out_scatter,
        out_induced=out_scatter / group_capacity[None, :, None, None],
        in_induced=in_scatter / group_capacity[None, :, None, None],
    )

    def inv_eos(energy):
        return energy

    def zero_planck(temperature):
        return np.zeros_like(temperature)

    arguments = (
        phi_star, psi_star, e_star, T_star, 1.0e-3, C_LIGHT, inv_eos,
        [zero_planck, zero_planck],
    )
    compact_result = operator_split_compton_update(
        *arguments, lambda _temperature: compact, induced_mode="nonlinear",
        max_iterations=1, on_nonconvergence="accept",
    )
    expanded_result = operator_split_compton_update(
        *arguments, lambda _temperature: expanded, induced_mode="nonlinear",
        max_iterations=1, on_nonconvergence="accept",
    )
    for compact_values, expanded_values in zip(compact_result[:4], expanded_result[:4]):
        assert np.array_equal(np.asarray(compact_values), np.asarray(expanded_values))
    print("Compact conservative Compton matrices passed")


def run_picard_backtracking_test():
    """Keep a divergent nonlinear candidate in the positive-energy domain."""
    phi_star = [np.array([[1.0]]), np.array([[0.0]])]
    psi_star = [np.array([[[1.0]]]), np.array([[[0.0]]])]
    e_star = np.array([[1.0]])
    matrices = ComptonMatrices(
        in_scatter=np.array([[0.0, 100.0], [0.0, 0.0]]),
        out_scatter=np.zeros((2, 2)),
    )

    def inv_eos(energy):
        return energy

    def zero_planck(temperature):
        return np.zeros_like(temperature)

    phi_new, _, e_new, T_new, _, diagnostics = operator_split_compton_update(
        phi_star, psi_star, e_star, e_star, 1.0 / C_LIGHT, C_LIGHT, inv_eos,
        [zero_planck, zero_planck], lambda _temperature: matrices,
        induced_mode="nonlinear", max_iterations=1,
        on_nonconvergence="accept", return_diagnostics=True,
    )
    energy_before = e_star + sum(phi_star) / C_LIGHT
    energy_after = e_new + sum(phi_new) / C_LIGHT
    assert diagnostics["minimum_picard_relaxation"] < 1.0
    assert np.all(np.isfinite(T_new)) and np.all(T_new > 0.0)
    assert np.allclose(energy_after, energy_before, atol=1.0e-12)
    print("Compton Picard backtracking passed")


def run_planck_lagged_test():
    """Match the explicit linear mode to one frozen Planck iteration."""
    phi_star = [np.array([[1.0]]), np.array([[0.25]])]
    psi_star = [np.array([[[1.0]]]), np.array([[[0.25]]])]
    e_star = np.array([[2.0]])
    T_star = e_star.copy()
    matrices = ComptonMatrices(
        in_scatter=np.array([[0.0, 0.3], [0.2, 0.0]]),
        out_scatter=np.array([[0.0, 0.4], [0.35, 0.0]]),
        out_induced=np.array([[0.0, 0.04], [0.03, 0.0]]),
        in_induced=np.array([[0.0, 0.02], [0.01, 0.0]]),
    )

    def inv_eos(energy):
        return energy

    def planck_low(temperature):
        return 0.1 * temperature

    def planck_high(temperature):
        return 0.2 * temperature

    arguments = (
        phi_star, psi_star, e_star, T_star, 1.0e-3, C_LIGHT, inv_eos,
        [planck_low, planck_high], lambda _temperature: matrices,
    )
    lagged = operator_split_compton_update(
        *arguments, induced_mode="planck-lagged", return_diagnostics=True,
    )
    legacy = operator_split_compton_update(
        *arguments, induced_mode="planck", max_iterations=1,
        on_nonconvergence="accept", return_diagnostics=True,
    )
    for lagged_values, legacy_values in zip(lagged[:4], legacy[:4]):
        assert np.array_equal(np.asarray(lagged_values), np.asarray(legacy_values))
    diagnostics = lagged[-1]
    assert lagged[4] == 1
    assert diagnostics["converged"]
    assert diagnostics["residual"] == 0.0
    assert diagnostics["lagged_temperature_change"] > 0.0
    energy_before = e_star + sum(phi_star) / C_LIGHT
    energy_after = lagged[2] + sum(lagged[0]) / C_LIGHT
    assert np.allclose(energy_after, energy_before, atol=1.0e-12)
    print("Frozen-Planck linear Compton update passed")


if __name__ == "__main__":
    run_test()
    run_nonlinear_test()
    run_positivity_limiter_test()
    run_compact_conservative_test()
    run_picard_backtracking_test()
    run_planck_lagged_test()
