"""Numba-compatible thermal Compton collision kernel for MG-IMC.

Transport packets retain a continuous photon energy alongside their group
index.  The electron velocity is sampled from the Maxwell--Juttner
distribution, and scattering uses relativistic Klein--Nishina recoil and
Lorentz transforms before remapping the outgoing photon to a group.
"""

import math
import numpy as np

try:
    from numba import jit
except Exception:
    def jit(*jit_args, **jit_kwargs):
        if len(jit_args) == 1 and callable(jit_args[0]) and not jit_kwargs:
            return jit_args[0]
        def decorator(func):
            return func
        return decorator


ELECTRON_REST_KEV = 510.99895


@jit(nopython=True, cache=True)
def _kn_differential_ratio(epsilon, mu):
    ratio = 1.0 / (1.0 + epsilon * (1.0 - mu))
    return 0.5 * ratio * ratio * (1.0 / ratio + ratio - (1.0 - mu * mu))


@jit(nopython=True, cache=True)
def _kn_total_ratio(epsilon):
    if epsilon < 1.0e-3:
        value = 1.0 - 2.0 * epsilon + 5.2 * epsilon * epsilon
    else:
        log_term = math.log1p(2.0 * epsilon)
        value = 0.75 * (
            (1.0 + epsilon) / epsilon**3
            * (2.0 * epsilon * (1.0 + epsilon) / (1.0 + 2.0 * epsilon)
               - log_term)
            + log_term / (2.0 * epsilon)
            - (1.0 + 3.0 * epsilon) / (1.0 + 2.0 * epsilon)**2
        )
    return min(1.0, max(0.0, value))


@jit(nopython=True, cache=True)
def _sample_kn_cosine(epsilon):
    while True:
        mu = 2.0 * np_random() - 1.0
        if np_random() < _kn_differential_ratio(epsilon, mu):
            return mu


@jit(nopython=True, cache=True)
def np_random():
    """Small wrapper so the kernel has a single Numba RNG entry point."""
    return np.random.random()


@jit(nopython=True, cache=True)
def _unit_perpendicular(dx, dy, dz):
    if abs(dz) < 0.9:
        ax, ay, az = 0.0, 0.0, 1.0
    else:
        ax, ay, az = 1.0, 0.0, 0.0
    px = dy * az - dz * ay
    py = dz * ax - dx * az
    pz = dx * ay - dy * ax
    norm = math.sqrt(px * px + py * py + pz * pz)
    return px / norm, py / norm, pz / norm


@jit(nopython=True, cache=True)
def compton_scatter_3d(energy_kev, dx, dy, dz, electron_temperature_kev):
    """Process one Thomson-majorant candidate collision.

    The returned deposit is positive when the photon loses energy and negative
    when it gains energy from the electrons.  A rejected total
    Klein--Nishina test returns the unchanged photon as a null collision.
    """
    theta = max(electron_temperature_kev, 0.0) / ELECTRON_REST_KEV
    direction_norm = math.sqrt(dx * dx + dy * dy + dz * dz)
    if direction_norm <= 1.0e-30:
        dx, dy, dz = 0.0, 0.0, 1.0
    else:
        dx /= direction_norm
        dy /= direction_norm
        dz /= direction_norm

    # Draw a Maxwell--Juttner speed from the exact chi-square mixture, then
    # condition the electron direction on relative flux at that fixed speed.
    while True:
        w3 = 0.5 * math.sqrt(math.pi)
        w4 = math.sqrt(theta / 2.0)
        w5 = 0.75 * theta * math.sqrt(math.pi)
        w6 = 2.0 * theta * math.sqrt(theta / 2.0)
        choice = np_random() * (w3 + w4 + w5 + w6)
        if choice < w3:
            degrees = 3.0
        elif choice < w3 + w4:
            degrees = 4.0
        elif choice < w3 + w4 + w5:
            degrees = 5.0
        else:
            degrees = 6.0
        y = math.sqrt(0.5 * np.random.gamma(0.5 * degrees, 2.0))
        acceptance = math.sqrt(1.0 + 0.5 * theta * y * y) / (
            1.0 + math.sqrt(theta / 2.0) * y
        )
        if np_random() < acceptance:
            break
    gamma = 1.0 + theta * y * y
    beta2 = 1.0 - 1.0 / (gamma * gamma)
    beta = math.sqrt(beta2)
    while True:
        electron_mu = 2.0 * np_random() - 1.0
        if np_random() < (1.0 - beta * electron_mu) / (1.0 + beta):
            break
    electron_phi = 2.0 * math.pi * np_random()
    px, py, pz = _unit_perpendicular(dx, dy, dz)
    qx = dy * pz - dz * py
    qy = dz * px - dx * pz
    qz = dx * py - dy * px
    electron_transverse = math.sqrt(max(0.0, 1.0 - electron_mu * electron_mu))
    bx = beta * (
        electron_mu * dx
        + electron_transverse * (math.cos(electron_phi) * px + math.sin(electron_phi) * qx)
    )
    by = beta * (
        electron_mu * dy
        + electron_transverse * (math.cos(electron_phi) * py + math.sin(electron_phi) * qy)
    )
    bz = beta * (
        electron_mu * dz
        + electron_transverse * (math.cos(electron_phi) * pz + math.sin(electron_phi) * qz)
    )

    dot = bx * dx + by * dy + bz * dz
    energy0 = gamma * energy_kev * (1.0 - dot)
    coeff = ((gamma - 1.0) * dot / beta2 - gamma) if beta2 > 1e-30 else 0.0
    p0x = dx + coeff * bx
    p0y = dy + coeff * by
    p0z = dz + coeff * bz
    p0norm = math.sqrt(p0x * p0x + p0y * p0y + p0z * p0z)
    p0x /= p0norm
    p0y /= p0norm
    p0z /= p0norm

    epsilon0 = energy0 / ELECTRON_REST_KEV
    if np_random() >= _kn_total_ratio(epsilon0):
        return energy_kev, dx, dy, dz, 0.0

    mu0 = _sample_kn_cosine(epsilon0)
    phi = 2.0 * math.pi * np_random()
    px, py, pz = _unit_perpendicular(p0x, p0y, p0z)
    qx = p0y * pz - p0z * py
    qy = p0z * px - p0x * pz
    qz = p0x * py - p0y * px
    transverse = math.sqrt(max(0.0, 1.0 - mu0 * mu0))
    cphi = math.cos(phi)
    sphi = math.sin(phi)
    out0x = mu0 * p0x + transverse * (cphi * px + sphi * qx)
    out0y = mu0 * p0y + transverse * (cphi * py + sphi * qy)
    out0z = mu0 * p0z + transverse * (cphi * pz + sphi * qz)
    energy0_out = energy0 / (1.0 + epsilon0 * (1.0 - mu0))

    dot_out = bx * out0x + by * out0y + bz * out0z
    energy_out = gamma * energy0_out * (1.0 + dot_out)
    coeff_out = ((gamma - 1.0) * dot_out / beta2 + gamma) if beta2 > 1e-30 else 1.0
    poutx = out0x + coeff_out * bx
    pouty = out0y + coeff_out * by
    poutz = out0z + coeff_out * bz
    poutnorm = math.sqrt(poutx * poutx + pouty * pouty + poutz * poutz)
    return (energy_out, poutx / poutnorm, pouty / poutnorm,
            poutz / poutnorm, energy_kev - energy_out)


@jit(nopython=True, cache=True)
def _normal_random():
    u1 = max(np_random(), 1e-300)
    u2 = np_random()
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
