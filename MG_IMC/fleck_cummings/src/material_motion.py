"""Special-relativistic helpers for moving-material IMC.

Particle energy and direction passed to these routines are lab-frame values.
Material velocity is represented by beta = v / c.  The helpers are kept small
and Numba-compatible so source and transport kernels can share one convention.
"""

from __future__ import annotations

import math

import numpy as np

try:
    from numba import jit
except Exception:
    def jit(*jit_args, **jit_kwargs):
        if len(jit_args) == 1 and callable(jit_args[0]) and not jit_kwargs:
            return jit_args[0]

        def _decorator(func):
            return func

        return _decorator


C_LIGHT = 29.98  # cm / ns


def validate_material_velocity(material_velocity, n_cells=None, c=C_LIGHT):
    """Validate and copy a cellwise material velocity array.

    Parameters
    ----------
    material_velocity : array-like, shape (n_cells, 3)
        Lab-frame material velocity in cm/ns.
    n_cells : int or None
        Required cell count when known.
    c : float
        Speed of light in the same distance/time units as the velocity.
    """
    velocity = np.asarray(material_velocity, dtype=np.float64)
    if velocity.ndim != 2 or velocity.shape[1] != 3:
        raise ValueError("material_velocity must have shape (n_cells, 3)")
    if n_cells is not None and velocity.shape[0] != n_cells:
        raise ValueError(
            f"material_velocity has {velocity.shape[0]} cells; expected {n_cells}"
        )
    if not np.all(np.isfinite(velocity)):
        raise ValueError("material_velocity must contain only finite values")
    if not np.isfinite(c) or c <= 0.0:
        raise ValueError("c must be finite and positive")

    speed_squared = np.sum(velocity * velocity, axis=1)
    if np.any(speed_squared >= c * c):
        raise ValueError("every material speed must satisfy |v| < c")
    return np.ascontiguousarray(velocity)


def precompute_velocity_factors(material_velocity, n_cells=None, c=C_LIGHT):
    """Return contiguous cellwise beta vectors and Lorentz factors."""
    velocity = validate_material_velocity(material_velocity, n_cells=n_cells, c=c)
    beta = np.ascontiguousarray(velocity / c)
    beta_squared = np.sum(beta * beta, axis=1)
    gamma = np.ascontiguousarray(1.0 / np.sqrt(1.0 - beta_squared))
    return beta, gamma


@jit(nopython=True, cache=True)
def _gamma_from_beta(beta_x, beta_y, beta_z):
    beta_squared = beta_x * beta_x + beta_y * beta_y + beta_z * beta_z
    return 1.0 / math.sqrt(1.0 - beta_squared), beta_squared


@jit(nopython=True, cache=True)
def lab_to_fluid_photon(
    energy_lab,
    omega_x_lab,
    omega_y_lab,
    omega_z_lab,
    beta_x,
    beta_y,
    beta_z,
):
    """Transform photon energy and direction from the lab to fluid frame.

    Returns ``(energy_fluid, omega_x, omega_y, omega_z, doppler_lab)`` where
    ``doppler_lab = 1 - omega_lab . beta``.  Inputs must satisfy |beta| < 1
    and the direction must be a unit vector.
    """
    gamma, beta_squared = _gamma_from_beta(beta_x, beta_y, beta_z)
    beta_dot_omega = (
        beta_x * omega_x_lab
        + beta_y * omega_y_lab
        + beta_z * omega_z_lab
    )
    doppler_lab = 1.0 - beta_dot_omega
    energy_fluid = gamma * doppler_lab * energy_lab

    if beta_squared < 1.0e-30:
        return (
            energy_fluid,
            omega_x_lab,
            omega_y_lab,
            omega_z_lab,
            doppler_lab,
        )

    coefficient = ((gamma - 1.0) * beta_dot_omega / beta_squared) - gamma
    denominator = gamma * doppler_lab
    omega_x_fluid = (omega_x_lab + coefficient * beta_x) / denominator
    omega_y_fluid = (omega_y_lab + coefficient * beta_y) / denominator
    omega_z_fluid = (omega_z_lab + coefficient * beta_z) / denominator
    return (
        energy_fluid,
        omega_x_fluid,
        omega_y_fluid,
        omega_z_fluid,
        doppler_lab,
    )


@jit(nopython=True, cache=True)
def fluid_to_lab_photon(
    energy_fluid,
    omega_x_fluid,
    omega_y_fluid,
    omega_z_fluid,
    beta_x,
    beta_y,
    beta_z,
):
    """Transform photon energy and direction from the fluid to lab frame.

    Returns ``(energy_lab, omega_x, omega_y, omega_z, doppler_fluid)`` where
    ``doppler_fluid = 1 + omega_fluid . beta``.
    """
    gamma, beta_squared = _gamma_from_beta(beta_x, beta_y, beta_z)
    beta_dot_omega = (
        beta_x * omega_x_fluid
        + beta_y * omega_y_fluid
        + beta_z * omega_z_fluid
    )
    doppler_fluid = 1.0 + beta_dot_omega
    energy_lab = gamma * doppler_fluid * energy_fluid

    if beta_squared < 1.0e-30:
        return (
            energy_lab,
            omega_x_fluid,
            omega_y_fluid,
            omega_z_fluid,
            doppler_fluid,
        )

    coefficient = ((gamma - 1.0) * beta_dot_omega / beta_squared) + gamma
    denominator = gamma * doppler_fluid
    omega_x_lab = (omega_x_fluid + coefficient * beta_x) / denominator
    omega_y_lab = (omega_y_fluid + coefficient * beta_y) / denominator
    omega_z_lab = (omega_z_fluid + coefficient * beta_z) / denominator
    return (
        energy_lab,
        omega_x_lab,
        omega_y_lab,
        omega_z_lab,
        doppler_fluid,
    )


@jit(nopython=True, cache=True)
def energy_group_index(energy, energy_edges):
    """Return the group containing energy, clamped to the end groups."""
    group = np.searchsorted(energy_edges, energy, side="right") - 1
    if group < 0:
        return 0
    upper_group = len(energy_edges) - 2
    if group > upper_group:
        return upper_group
    return group


@jit(nopython=True, cache=True)
def fluid_group_from_lab_photon(
    energy_lab,
    omega_x_lab,
    omega_y_lab,
    omega_z_lab,
    beta_x,
    beta_y,
    beta_z,
    energy_edges,
):
    """Transform a lab photon and return its fluid-frame opacity group.

    The returned tuple is ``(group, energy_fluid, doppler_lab)``.  The lab
    photon is not mutated; callers recompute this value on every path segment.
    """
    transformed = lab_to_fluid_photon(
        energy_lab,
        omega_x_lab,
        omega_y_lab,
        omega_z_lab,
        beta_x,
        beta_y,
        beta_z,
    )
    energy_fluid = transformed[0]
    return energy_group_index(energy_fluid, energy_edges), energy_fluid, transformed[4]


def lab_groups_from_energies(photon_energies_lab, energy_edges):
    """Vectorized diagnostic grouping of continuous lab-frame energies."""
    energies = np.asarray(photon_energies_lab, dtype=np.float64)
    edges = np.asarray(energy_edges, dtype=np.float64)
    if edges.ndim != 1 or len(edges) < 2 or np.any(np.diff(edges) <= 0.0):
        raise ValueError("energy_edges must be a strictly increasing 1-D array")
    groups = np.searchsorted(edges, energies, side="right") - 1
    return np.clip(groups, 0, len(edges) - 2).astype(np.int32)


def sample_fluid_directions(n, beta_vector, doppler_power):
    """Sample fluid-frame directions proportional to ``D_F**doppler_power``.

    ``D_F = 1 + omega_fluid . beta``.  Volume emission uses power 1;
    equal-lab-weight equilibrium radiation uses power 2.  Rejection from an
    isotropic direction is efficient for both supported powers at |beta| < 1.
    """
    if n < 0:
        raise ValueError("n must be nonnegative")
    if doppler_power not in (1, 2):
        raise ValueError("doppler_power must be 1 or 2")
    beta = np.asarray(beta_vector, dtype=np.float64)
    if beta.shape != (3,) or not np.all(np.isfinite(beta)):
        raise ValueError("beta_vector must be a finite three-vector")
    beta_magnitude = float(np.linalg.norm(beta))
    if beta_magnitude >= 1.0:
        raise ValueError("beta_vector must satisfy |beta| < 1")
    if n == 0:
        return np.empty((0, 3), dtype=np.float64)

    directions = np.empty((n, 3), dtype=np.float64)
    filled = 0
    maximum = (1.0 + beta_magnitude) ** doppler_power
    while filled < n:
        remaining = n - filled
        candidates = max(64, 2 * remaining)
        mu = np.random.uniform(-1.0, 1.0, candidates)
        accept_probability = (1.0 + beta_magnitude * mu) ** doppler_power / maximum
        accepted = mu[np.random.random(candidates) < accept_probability]
        take = min(remaining, len(accepted))
        if take:
            directions[filled:filled + take, 2] = accepted[:take]
            filled += take

    mu = directions[:, 2].copy()
    phi = np.random.uniform(0.0, 2.0 * np.pi, n)
    transverse = np.sqrt(np.maximum(0.0, 1.0 - mu * mu))

    if beta_magnitude < 1.0e-15:
        directions[:, 0] = transverse * np.cos(phi)
        directions[:, 1] = transverse * np.sin(phi)
        return directions

    axis = beta / beta_magnitude
    reference = np.array([0.0, 0.0, 1.0])
    if abs(axis[2]) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    basis_1 = np.cross(reference, axis)
    basis_1 /= np.linalg.norm(basis_1)
    basis_2 = np.cross(axis, basis_1)
    directions[:] = (
        mu[:, None] * axis[None, :]
        + transverse[:, None]
        * (
            np.cos(phi)[:, None] * basis_1[None, :]
            + np.sin(phi)[:, None] * basis_2[None, :]
        )
    )
    return directions


@jit(nopython=True, cache=True)
def transform_fluid_particles_to_lab(energies_fluid, directions_fluid, beta_vector):
    """Vectorized Numba loop for fluid-to-lab photon transformation."""
    count = len(energies_fluid)
    energies_lab = np.empty(count)
    directions_lab = np.empty((count, 3))
    for particle in range(count):
        transformed = fluid_to_lab_photon(
            energies_fluid[particle],
            directions_fluid[particle, 0],
            directions_fluid[particle, 1],
            directions_fluid[particle, 2],
            beta_vector[0],
            beta_vector[1],
            beta_vector[2],
        )
        energies_lab[particle] = transformed[0]
        directions_lab[particle, 0] = transformed[1]
        directions_lab[particle, 1] = transformed[2]
        directions_lab[particle, 2] = transformed[3]
    return energies_lab, directions_lab


def lab_energy_momentum_to_fluid(energy_lab, momentum_lab, material_velocity, c=C_LIGHT):
    """Transform cell-integrated radiation energy and momentum to fluid frames."""
    energy = np.asarray(energy_lab, dtype=np.float64)
    momentum = np.asarray(momentum_lab, dtype=np.float64)
    if energy.ndim != 1 or momentum.shape != (len(energy), 3):
        raise ValueError("energy_lab and momentum_lab must have shapes (I,) and (I, 3)")
    velocity = validate_material_velocity(material_velocity, n_cells=len(energy), c=c)
    beta, gamma = precompute_velocity_factors(velocity, n_cells=len(energy), c=c)
    beta_dot_momentum = np.sum(beta * momentum, axis=1)
    energy_fluid = gamma * (energy - c * beta_dot_momentum)

    momentum_fluid = np.empty_like(momentum)
    beta_squared = np.sum(beta * beta, axis=1)
    for cell in range(len(energy)):
        if beta_squared[cell] < 1.0e-30:
            momentum_fluid[cell] = momentum[cell]
            continue
        coefficient = (
            (gamma[cell] - 1.0)
            * beta_dot_momentum[cell]
            / beta_squared[cell]
            - gamma[cell] * energy[cell] / c
        )
        momentum_fluid[cell] = momentum[cell] + coefficient * beta[cell]
    return energy_fluid, momentum_fluid
