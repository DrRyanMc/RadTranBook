"""MG_IMC2D.py - Multigroup Implicit Monte Carlo in 2D Cartesian (xy) and cylindrical (r-z).

This module extends IMC2D.py to support multigroup radiation transport. Each particle
carries a group index and a frequency within that group. The implementation follows
the multigroup IMC equations with the Fleck factor:

f = 1 / (1 + β c σ_P Δt)

where σ_P is the Planck-weighted opacity, and the transport equation is:

(1/c ∂/∂t + Ω·∇ + σ_g) I_g = (f σ_g b_g★ c a T_n^4) / 4π 
                                + ((1-f) σ_g b_g★ / σ_P) Σ_g' σ_g' φ_g' / 4π + Q_g / 4π

Key multigroup features:
- Energy groups defined by energy_edges array
- Particles carry group index and frequency
- Initial conditions and boundary sources: frequency sampled from mixture of Gammas
  (equations 10.23-10.27 in textbook)
- Material emission and effective scatter: group sampled from piecewise constant 
  distribution proportional to σ_a,g b_g★ (equation 10.18)
- Group-dependent opacities σ_a,g(T)
- Planck function integrals B_g(T) computed via external library

Units consistent with IMC2D:
- distance: cm
- time: ns
- temperature: keV
- energy: GJ
- frequency: keV (photon energy)
"""

from dataclasses import dataclass
import numpy as np
import random
import time as _time

try:
    from numba import jit, prange, get_thread_id, get_num_threads
except Exception:
    # Fallback path when numba/llvmlite are unavailable.
    def jit(*jit_args, **jit_kwargs):
        def decorator(func):
            if jit_kwargs.get("cache"):
                return func
            if jit_kwargs.get("parallel"):
                return func
            return func
        return decorator

    def prange(*args):
        return range(*args)

    def get_thread_id():
        return 0

    def get_num_threads():
        return 1

# Import Planck integral library for multigroup calculations
try:
    from planck_integrals import Bg, dBgdT, Bg_multigroup, dBgdT_multigroup
    _PLANCK_AVAILABLE = True
except ImportError:
    print("Warning: planck_integrals module not found. Using gray approximations.")
    _PLANCK_AVAILABLE = False
    
    def Bg(E_low, E_high, T):
        """Gray approximation: B = σT⁴/(4π)"""
        C_LIGHT = 29.98  # cm/ns
        A_RAD = 0.01372  # GJ/(cm³·keV⁴)
        return (C_LIGHT * A_RAD / 4) * T**4 / (4 * np.pi)
    
    def dBgdT(E_low, E_high, T):
        """Gray approximation: dB/dT = 4σT³/(4π)"""
        C_LIGHT = 29.98
        A_RAD = 0.01372
        return (C_LIGHT * A_RAD) * T**3 / (4 * np.pi)
    
    def Bg_multigroup(energy_edges, T):
        """Gray approximation for all groups"""
        C_LIGHT = 29.98
        A_RAD = 0.01372
        n_groups = len(energy_edges) - 1
        total_B = (C_LIGHT * A_RAD / 4) * T**4 / (4 * np.pi)
        return np.full(n_groups, total_B / n_groups)
    
    def dBgdT_multigroup(energy_edges, T):
        """Gray approximation derivatives for all groups"""
        C_LIGHT = 29.98
        A_RAD = 0.01372
        n_groups = len(energy_edges) - 1
        total_dB = (C_LIGHT * A_RAD) * T**3 / (4 * np.pi)
        return np.full(n_groups, total_dB / n_groups)

__c = 29.98
__a = 0.01372

_PLANCK_N_MAX = 200
_PLANCK_N_VALUES = np.arange(1, _PLANCK_N_MAX + 1, dtype=np.float64)
_PLANCK_MIXTURE_CDF = np.cumsum((1.0 / _PLANCK_N_VALUES**4) / np.sum(1.0 / _PLANCK_N_VALUES**4))

_GEOM_XY = 0
_GEOM_RZ = 1

_CROSS_NONE = 0
_CROSS_I_PLUS = 1
_CROSS_I_MINUS = 2
_CROSS_J_PLUS = 3
_CROSS_J_MINUS = 4

_EVT_CENSUS = 0
_EVT_BOUNDARY = 1
_EVT_SCATTER = 2


@dataclass
class SimulationState2DMG:
    """Mutable 2D multigroup IMC state passed between timesteps."""

    weights: np.ndarray
    dir1: np.ndarray
    dir2: np.ndarray
    times: np.ndarray
    pos1: np.ndarray
    pos2: np.ndarray
    cell_i: np.ndarray
    cell_j: np.ndarray
    groups: np.ndarray  # Group index for each particle

    internal_energy: np.ndarray
    temperature: np.ndarray
    radiation_temperature: np.ndarray
    radiation_energy_by_group: np.ndarray  # (n_groups, nx, ny) or (nx, ny, n_groups)

    time: float
    previous_total_energy: float
    count: int = 0


def _shape_from_edges(edges1, edges2):
    return len(edges1) - 1, len(edges2) - 1


def _cell_volumes_xy(x_edges, y_edges):
    dx = np.diff(x_edges)
    dy = np.diff(y_edges)
    return dx[:, None] * dy[None, :]


def _cell_volumes_rz(r_edges, z_edges):
    dr2 = r_edges[1:] ** 2 - r_edges[:-1] ** 2
    dz = np.diff(z_edges)
    # Axisymmetric full 3D cell volume after revolution.
    return np.pi * dr2[:, None] * dz[None, :]


def _cell_volumes(edges1, edges2, geometry):
    if geometry == "xy":
        return _cell_volumes_xy(edges1, edges2)
    if geometry == "rz":
        return _cell_volumes_rz(edges1, edges2)
    raise ValueError(f"Unknown geometry: {geometry}")


def _locate_indices(pos1, pos2, edges1, edges2):
    i = np.searchsorted(edges1, pos1, side="right") - 1
    j = np.searchsorted(edges2, pos2, side="right") - 1
    return i, j


def _flatten_index(i, j, nx):
    return i + nx * j


def _unflatten_index(idx, nx):
    j = idx // nx
    i = idx - nx * j
    return i, j


@jit(nopython=True, cache=True)
def _sample_isotropic_xy(n):
    """Sample isotropic directions for 2D-in-space, 3D-in-angle transport."""
    uz  = np.random.uniform(-1.0, 1.0, n)
    phi = np.random.uniform(0.0, 2.0 * np.pi, n)
    r_xy = np.sqrt(np.maximum(0.0, 1.0 - uz * uz))
    ux = r_xy * np.cos(phi)
    uy = r_xy * np.sin(phi)
    return ux, uy


@jit(nopython=True, cache=True)
def _sample_isotropic_rz(n):
    """Sample axisymmetric direction pair (mu_perp, eta)."""
    eta = np.random.uniform(-1.0, 1.0, n)
    mu_perp = np.cos(2.0 * np.pi * np.random.uniform(0.0, 1.0, n))
    return mu_perp, eta


@jit(nopython=True, cache=True)
def _sample_planck_spectrum_mixture_of_gammas_jit(n, T, energy_edges_low, energy_edges_high, cdf):
    """Numba-jitted Planck spectrum sampling via mixture of Gammas.
    
    Much faster than the Python version. Returns frequencies and group indices.
    """
    if n <= 0:
        return np.zeros(n), np.zeros(n, dtype=np.int32)
    
    if T <= 0.0:
        return np.full(n, energy_edges_low[0]), np.zeros(n, dtype=np.int32)
    
    frequencies = np.zeros(n)
    groups = np.zeros(n, dtype=np.int32)
    n_groups = len(energy_edges_low)
    
    n_max = len(cdf)
    
    # Sample particles
    for p in range(n):
        # Sample n_s from CDF (rejection search)
        xi = np.random.random()
        n_s = float(n_max)
        for i in range(n_max):
            if xi <= cdf[i]:
                n_s = float(i + 1)
                break
        
        # Sample x from Gamma(k=4, r=n_s) via product of exponentials
        r1 = -np.log(np.random.random() + 1e-300)
        r2 = -np.log(np.random.random() + 1e-300)
        r3 = -np.log(np.random.random() + 1e-300)
        r4 = -np.log(np.random.random() + 1e-300)
        x = (r1 + r2 + r3 + r4) / n_s
        
        # Convert to frequency
        frequencies[p] = T * x
        
        # Determine group via search
        freq = frequencies[p]
        if freq < energy_edges_low[0]:
            g = 0
        elif freq >= energy_edges_high[n_groups - 1]:
            g = n_groups - 1
        else:
            g = n_groups - 1
            for i in range(n_groups):
                if freq >= energy_edges_low[i] and freq < energy_edges_high[i]:
                    g = i
                    break
        groups[p] = g
    
    return frequencies, groups


@jit(nopython=True, cache=True)
def _sample_group_piecewise_constant_jit(n, cdf_normalized):
    """Numba-jitted group sampling from normalized CDF."""
    if n <= 0:
        return np.zeros(n, dtype=np.int32)
    
    groups = np.zeros(n, dtype=np.int32)
    n_groups = len(cdf_normalized)
    
    for p in range(n):
        xi = np.random.random()
        # Search through CDF
        g = n_groups - 1
        for i in range(n_groups):
            if xi <= cdf_normalized[i]:
                g = i
                break
        groups[p] = g
    
    return groups


def _sample_planck_spectrum_mixture_of_gammas(n, T, energy_edges):
    """Sample frequency from Planck spectrum using mixture of Gammas.
    
    This implements equations 10.23-10.27 from the textbook for sampling
    from the Planck distribution p(ν) = (15/π^4) x^3 e^(-x) where x = ν/T.
    
    Parameters
    ----------
    n : int
        Number of samples
    T : float
        Temperature (keV)
    energy_edges : array
        Energy group edges (keV)
    
    Returns
    -------
    frequencies : array
        Sampled frequencies (keV)
    groups : array
        Group indices for sampled frequencies
    """
    if n <= 0:
        return np.array([]), np.array([], dtype=np.int32)
    
    n_groups = len(energy_edges) - 1
    edges_low = energy_edges[:-1].astype(np.float64)
    edges_high = energy_edges[1:].astype(np.float64)
    
    return _sample_planck_spectrum_mixture_of_gammas_jit(
        n,
        float(T),
        edges_low,
        edges_high,
        _PLANCK_MIXTURE_CDF,
    )


def _sample_group_piecewise_constant(n, probabilities):
    """Sample group from piecewise constant distribution.
    
    This implements equation 10.18 for sampling group g from emission/effective scatter
    with probability P_g = σ_a,g b_g★ / σ_P.
    
    Parameters
    ----------
    n : int
        Number of samples
    probabilities : array
        Probability for each group (must sum to 1)
    
    Returns
    -------
    groups : array (int)
        Sampled group indices
    """
    if n <= 0:
        return np.array([], dtype=np.int32)
    
    # Ensure probabilities are normalized
    probabilities = np.asarray(probabilities, dtype=np.float64)
    prob_sum = np.sum(probabilities)
    if prob_sum > 0.0:
        probabilities = probabilities / prob_sum
    else:
        probabilities = np.ones_like(probabilities) / len(probabilities)
    
    # Build CDF
    cdf = np.cumsum(probabilities).astype(np.float64)
    cdf[-1] = 1.0
    
    return _sample_group_piecewise_constant_jit(n, cdf)


def _compute_Bg_multigroup_grid(energy_edges, temperature):
    """Compute Planck integrals B_g for all groups and all cells efficiently.
    
    Returns array of shape (n_groups, nx, ny) with B_g values.
    """
    n_groups = len(energy_edges) - 1
    nx, ny = temperature.shape
    B_g = np.zeros((n_groups, nx, ny))
    
    # Vectorized: call Bg wrapper for each group only, not each cell
    for g in range(n_groups):
        E_low = energy_edges[g]
        E_high = energy_edges[g + 1]
        # This calls the external Bg function to handle vectorization
        # If Bg is not vectorized, the loop below will be executed
        try:
            B_vals = Bg(E_low, E_high, temperature)
            if np.isscalar(B_vals):
                B_g[g, :, :] = B_vals
            elif B_vals.shape == temperature.shape:
                B_g[g, :, :] = B_vals
            else:
                # Fallback: call element-wise
                for i in range(nx):
                    for j in range(ny):
                        B_g[g, i, j] = Bg(E_low, E_high, temperature[i, j])
        except:
            # Fallback: call element-wise if vectorization fails
            for i in range(nx):
                for j in range(ny):
                    B_g[g, i, j] = Bg(E_low, E_high, temperature[i, j])
    
    return B_g


def _boundary_temperature_value(Tb, t):
    return Tb(t) if callable(Tb) else Tb


def _sample_boundary_xy(n, side, T, dt, x_edges, y_edges, energy_edges, boundary_source_func=None):
    """Half-Lambertian boundary source for Cartesian geometry with multigroup.
    
    Samples particles from boundary with frequencies from Planck spectrum.
    """
    if n <= 0:
        return None
    
    if T <= 0.0 and boundary_source_func is None:
        return None

    x0 = x_edges[0]
    x1 = x_edges[-1]
    y0 = y_edges[0]
    y1 = y_edges[-1]

    if side in ("left", "right"):
        area = y1 - y0
    else:
        area = x1 - x0

    # Original uniform sampling
    total_emission = __a * __c * T**4 / 4.0 * area * dt
    weights = np.full(n, total_emission / n)
    times = np.random.uniform(0.0, dt, n)

    # 3D Lambertian
    mu_n = np.sqrt(np.random.uniform(0.0, 1.0, n))
    mu_t = np.sqrt(np.maximum(0.0, 1.0 - mu_n * mu_n))
    phi  = np.random.uniform(0.0, 2.0 * np.pi, n)

    if side == "left":
        x = np.full(n, x0)
        y = np.random.uniform(y0, y1, n)
        ux = mu_n
        uy = mu_t * np.cos(phi)
    elif side == "right":
        x = np.full(n, x1)
        y = np.random.uniform(y0, y1, n)
        ux = -mu_n
        uy = mu_t * np.cos(phi)
    elif side == "bottom":
        x = np.random.uniform(x0, x1, n)
        y = np.full(n, y0)
        ux = mu_t * np.cos(phi)
        uy = mu_n
    elif side == "top":
        x = np.random.uniform(x0, x1, n)
        y = np.full(n, y1)
        ux = mu_t * np.cos(phi)
        uy = -mu_n
    else:
        raise ValueError(f"Unknown side: {side}")

    # Sample frequencies from Planck spectrum (mixture of gammas)
    frequencies, groups = _sample_planck_spectrum_mixture_of_gammas(n, T, energy_edges)

    return weights, ux, uy, times, x, y, groups


def _sample_boundary_rz(n, side, T, dt, r_edges, z_edges, energy_edges, boundary_source_func=None):
    """Boundary source for cylindrical r-z geometry with multigroup."""
    if n <= 0:
        return None
    if T <= 0.0 and boundary_source_func is None:
        return None

    r0 = r_edges[0]
    r1 = r_edges[-1]
    z0 = z_edges[0]
    z1 = z_edges[-1]

    if side in ("rmin", "rmax"):
        r_side = r0 if side == "rmin" else r1
        area = 2.0 * np.pi * r_side * (z1 - z0)
    else:
        area = np.pi * (r1**2 - r0**2)

    total_emission = __a * __c * T**4 / 4.0 * area * dt
    weights = np.full(n, total_emission / n)
    times = np.random.uniform(0.0, dt, n)

    eta = np.random.uniform(-1.0, 1.0, n)
    mu_abs = np.sqrt(np.random.uniform(0.0, 1.0, n))

    if side == "rmin":
        r = np.full(n, r0)
        z = np.random.uniform(z0, z1, n)
        mu_perp = mu_abs
    elif side == "rmax":
        r = np.full(n, r1)
        z = np.random.uniform(z0, z1, n)
        mu_perp = -mu_abs
    elif side == "zmin":
        r_2 = r0**2 + np.random.uniform(0.0, 1.0, n) * (r1**2 - r0**2)
        r = np.sqrt(r_2)
        z = np.full(n, z0)
        mu_perp = np.cos(2.0 * np.pi * np.random.uniform(0.0, 1.0, n))
        eta = mu_abs
    elif side == "zmax":
        r_2 = r0**2 + np.random.uniform(0.0, 1.0, n) * (r1**2 - r0**2)
        r = np.sqrt(r_2)
        z = np.full(n, z1)
        mu_perp = np.cos(2.0 * np.pi * np.random.uniform(0.0, 1.0, n))
        eta = -mu_abs
    else:
        raise ValueError(f"Unknown side: {side}")

    # Sample frequencies from Planck spectrum
    frequencies, groups = _sample_planck_spectrum_mixture_of_gammas(n, T, energy_edges)

    return weights, mu_perp, eta, times, r, z, groups


@jit(nopython=True, cache=True)
def _move_particle_xy(weight, ux, uy, x, y, i, j, x_edges, y_edges, sigma_a, sigma_s, distance_to_census):
    """Move one Cartesian particle to next event (boundary/scatter/census)."""
    x_l = x_edges[i]
    x_r = x_edges[i + 1]
    y_l = y_edges[j]
    y_r = y_edges[j + 1]

    sx = 1e30
    sy = 1e30

    if ux > 0.0:
        sx = (x_r - x) / (ux + 1e-300)
    elif ux < 0.0:
        sx = (x_l - x) / (ux - 1e-300)

    if uy > 0.0:
        sy = (y_r - y) / (uy + 1e-300)
    elif uy < 0.0:
        sy = (y_l - y) / (uy - 1e-300)

    if sigma_s > 1e-12:
        s_scat = -np.log(np.random.uniform(0.0, 1.0)) / sigma_s
    else:
        s_scat = 1e30

    s_min = min(sx, sy, s_scat, distance_to_census)
    
    # Stabilization for extremely small distances
    if s_min < 1e-14:
        s_min = max(s_min, sx, sy, distance_to_census)

    x_new = x + ux * s_min
    y_new = y + uy * s_min

    # Absorption
    if sigma_a > 1e-12:
        w_new = weight * np.exp(-sigma_a * s_min)
        deposited = weight - w_new
    else:
        w_new = weight
        deposited = 0.0

    # Determine event type
    crossing = _CROSS_NONE
    evt = _EVT_CENSUS

    if s_min == distance_to_census:
        evt = _EVT_CENSUS
    elif s_min == s_scat:
        evt = _EVT_SCATTER
    elif s_min == sx:
        if ux > 0.0:
            crossing = _CROSS_I_PLUS
        else:
            crossing = _CROSS_I_MINUS
        evt = _EVT_BOUNDARY
    elif s_min == sy:
        if uy > 0.0:
            crossing = _CROSS_J_PLUS
        else:
            crossing = _CROSS_J_MINUS
        evt = _EVT_BOUNDARY

    return x_new, y_new, w_new, deposited, s_min, evt, crossing


@jit(nopython=True, cache=True)
def _distance_to_radial_boundary_rz(r, mu_perp, eta, R):
    """Distance to cross radial boundary at radius R in cylindrical geometry."""
    if abs(mu_perp) < 1e-12:
        return 1e30
    
    sin2 = 1.0 - mu_perp * mu_perp
    if sin2 < 0.0:
        sin2 = 0.0
    
    a = sin2
    b = 2.0 * r * mu_perp
    c = r * r - R * R
    
    if abs(a) < 1e-14:
        if abs(b) > 1e-14:
            s = -c / b
            return s if s > 0.0 else 1e30
        return 1e30
    
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return 1e30
    
    sqrt_disc = np.sqrt(disc)
    s1 = (-b + sqrt_disc) / (2.0 * a)
    s2 = (-b - sqrt_disc) / (2.0 * a)
    
    s_min = 1e30
    if s1 > 1e-14:
        s_min = min(s_min, s1)
    if s2 > 1e-14:
        s_min = min(s_min, s2)
    
    return s_min

@jit(nopython=True, cache=True)

def _move_particle_rz(weight, mu_perp, eta, r, z, i, j, r_edges, z_edges, sigma_a, sigma_s, distance_to_census):
    """Move one cylindrical particle to next event."""
    r_l = r_edges[i]
    r_r = r_edges[i + 1]
    z_l = z_edges[j]
    z_r = z_edges[j + 1]

    # Distance to radial boundaries
    sr_l = _distance_to_radial_boundary_rz(r, mu_perp, eta, r_l)
    sr_r = _distance_to_radial_boundary_rz(r, mu_perp, eta, r_r)
    sr = min(sr_l, sr_r)

    # Distance to axial boundaries
    sz = 1e30
    if abs(eta) > 1e-12:
        if eta > 0.0:
            sz = (z_r - z) / (eta + 1e-300)
        else:
            sz = (z_l - z) / (eta - 1e-300)

    # Distance to scatter
    if sigma_s > 1e-12:
        s_scat = -np.log(np.random.uniform(0.0, 1.0)) / sigma_s
    else:
        s_scat = 1e30

    s_min = min(sr, sz, s_scat, distance_to_census)
    
    if s_min < 1e-14:
        s_min = max(s_min, sr, sz, distance_to_census)

    # New position
    sin_theta = np.sqrt(max(0.0, 1.0 - mu_perp * mu_perp))
    r_new = np.sqrt(max(0.0, r * r + 2.0 * r * mu_perp * s_min + sin_theta * sin_theta * s_min * s_min))
    z_new = z + eta * s_min

    # Absorption
    if sigma_a > 1e-12:
        w_new = weight * np.exp(-sigma_a * s_min)
        deposited = weight - w_new
    else:
        w_new = weight
        deposited = 0.0

    # Determine event
    crossing = _CROSS_NONE
    evt = _EVT_CENSUS

    if s_min == distance_to_census:
        evt = _EVT_CENSUS
    elif s_min == s_scat:
        evt = _EVT_SCATTER
    elif s_min == sr:
        if abs(sr_l - sr) < 1e-14:
            crossing = _CROSS_I_MINUS
        else:
            crossing = _CROSS_I_PLUS
        evt = _EVT_BOUNDARY
    elif s_min == sz:
        if eta > 0.0:
            crossing = _CROSS_J_PLUS
        else:
            crossing = _CROSS_J_MINUS
        evt = _EVT_BOUNDARY

    return r_new, z_new, w_new, deposited, s_min, evt, crossing


@jit(nopython=True, cache=True)
def _transport_particles_2d_mg(
    weights,
    dir1,
    dir2,
    times,
    pos1,
    pos2,
    cell_i,
    cell_j,
    groups,
    edges1,
    edges2,
    sigma_a,  # (n_groups, nx, ny)
    sigma_s,  # (n_groups, nx, ny)
    volumes,
    dt,
    reflect,
    max_events_per_particle,
    geometry_code,
    weight_floor,
):
    """Transport multigroup particles with JIT compilation.
    
    This is the core transport kernel for multigroup IMC.
    """
    n = len(weights)
    nx = len(edges1) - 1
    ny = len(edges2) - 1
    n_groups = sigma_a.shape[0]

    # Output arrays
    dep_cell = np.zeros((n_groups, nx, ny))
    si_cell = np.zeros((n_groups, nx, ny))
    
    # Statistics
    n_events = 0
    n_boundary_cross = 0
    n_abs_continue = 0
    n_census = 0
    n_abs_capture = 0
    n_weight_floor_kills = 0
    n_reflect = 0
    n_event_cap = 0

    boundary_loss = 0.0

    reflect_l, reflect_r, reflect_b, reflect_t = reflect

    for p in range(n):
        w = weights[p]
        d1 = dir1[p]
        d2 = dir2[p]
        t = times[p]
        x = pos1[p]
        y = pos2[p]
        i = cell_i[p]
        j = cell_j[p]
        g = groups[p]

        if w <= weight_floor:
            continue

        for evt_count in range(max_events_per_particle):
            if i < 0 or i >= nx or j < 0 or j >= ny:
                boundary_loss += w
                n_boundary_cross += 1
                break

            if w <= weight_floor:
                n_weight_floor_kills += 1
                break

            distance_to_census = __c * (dt - t)

            # Get opacities for this group and cell
            sig_a = sigma_a[g, i, j]
            sig_s = sigma_s[g, i, j]

            if geometry_code == _GEOM_XY:
                x_new, y_new, w_new, deposited, s_min, evt, crossing = _move_particle_xy(
                    w, d1, d2, x, y, i, j, edges1, edges2, sig_a, sig_s, distance_to_census
                )
            else:
                x_new, y_new, w_new, deposited, s_min, evt, crossing = _move_particle_rz(
                    w, d1, d2, x, y, i, j, edges1, edges2, sig_a, sig_s, distance_to_census
                )

            # Record deposition and scalar intensity
            dep_cell[g, i, j] += deposited / volumes[i, j]
            # Gray-style path-length estimator per group:
            #   I_g ~= deposited / (sigma_a * dt * volume)
            # with the sigma_a -> 0 limit using w * s_min.
            if sig_a > 1e-12:
                deposited_intensity = deposited / sig_a
            else:
                deposited_intensity = w * s_min
            si_cell[g, i, j] += deposited_intensity / (dt * volumes[i, j])

            x = x_new
            y = y_new
            w = w_new
            t = t + s_min / __c

            n_events += 1

            if evt == _EVT_CENSUS:
                n_census += 1
                break
            elif evt == _EVT_SCATTER:
                # Effective scattering - direction change only
                if geometry_code == _GEOM_XY:
                    d1, d2 = _sample_isotropic_xy(1)
                    d1 = d1[0]
                    d2 = d2[0]
                else:
                    d1, d2 = _sample_isotropic_rz(1)
                    d1 = d1[0]
                    d2 = d2[0]
                n_abs_continue += 1
            elif evt == _EVT_BOUNDARY:
                n_boundary_cross += 1

                # Move to neighboring cell first. Most boundary events are
                # interior cell crossings, not domain exits.
                if crossing == _CROSS_I_PLUS:
                    i += 1
                elif crossing == _CROSS_I_MINUS:
                    i -= 1
                elif crossing == _CROSS_J_PLUS:
                    j += 1
                elif crossing == _CROSS_J_MINUS:
                    j -= 1

                # Reflect or lose only if particle actually exited the domain.
                if i < 0:
                    if reflect_l:
                        d1 = -d1
                        g = n_groups - 1
                        n_reflect += 1
                    else:
                        boundary_loss += w
                        w = 0.0
                        break
                elif i >= nx:
                    if reflect_r:
                        d1 = -d1
                        i = nx - 1
                        n_reflect += 1
                    else:
                        boundary_loss += w
                        w = 0.0
                        break
                elif j < 0:
                    if reflect_b:
                        d2 = -d2
                        j = 0
                        n_reflect += 1
                    else:
                        boundary_loss += w
                        w = 0.0
                        break
                elif j >= ny:
                    if reflect_t:
                        d2 = -d2
                        j = ny - 1
                        n_reflect += 1
                    else:
                        boundary_loss += w
                        w = 0.0
                        break

            if evt_count == max_events_per_particle - 1:
                n_event_cap += 1

        # Write updated values back to arrays
        weights[p] = w
        dir1[p] = d1
        dir2[p] = d2
        times[p] = t
        pos1[p] = x
        pos2[p] = y
        cell_i[p] = i
        cell_j[p] = j

    stats = np.array([
        float(n_events),
        float(n_boundary_cross),
        float(n_abs_continue),
        float(n_census),
        float(n_abs_capture),
        float(n_weight_floor_kills),
        float(n_reflect),
        float(n_event_cap),
    ])

    return dep_cell, si_cell, boundary_loss, stats


def _equilibrium_sample_xy_mg(N, Tr, x_edges, y_edges, energy_edges):
    """Sample particles in equilibrium for XY geometry with multigroup."""
    nx = len(x_edges) - 1
    ny = len(y_edges) - 1
    n_groups = len(energy_edges) - 1

    # Total equilibrium energy
    volumes = _cell_volumes_xy(x_edges, y_edges)
    E_eq = __a * np.sum(Tr**4 * volumes)

    if N <= 0 or E_eq <= 0.0:
        return (
            np.array([]), np.array([]), np.array([]),
            np.array([]), np.array([]), np.array([]),
            np.array([], dtype=np.int32)
        )

    weights = np.full(N, E_eq / N)

    # Sample positions uniformly by volume
    vol_flat = volumes.flatten()
    probs = vol_flat / np.sum(vol_flat)
    flat_indices = np.random.choice(len(vol_flat), size=N, p=probs)
    
    cell_i = flat_indices % nx
    cell_j = flat_indices // nx

    # Sample within cells
    x = x_edges[cell_i] + np.random.uniform(0.0, 1.0, N) * np.diff(x_edges)[cell_i]
    y = y_edges[cell_j] + np.random.uniform(0.0, 1.0, N) * np.diff(y_edges)[cell_j]

    # Isotropic directions
    ux, uy = _sample_isotropic_xy(N)

    # Times uniformly in [0, dt) - here we use 0
    times = np.zeros(N)

    # Sample groups and frequencies from local Planck spectrum
    T_local = Tr[cell_i, cell_j]
    frequencies = np.zeros(N)
    groups = np.zeros(N, dtype=np.int32)
    
    for p in range(N):
        freq, grp = _sample_planck_spectrum_mixture_of_gammas(1, T_local[p], energy_edges)
        if len(freq) > 0:
            frequencies[p] = freq[0]
            groups[p] = grp[0]

    return weights, ux, uy, times, x, y, groups


def _equilibrium_sample_rz_mg(N, Tr, r_edges, z_edges, energy_edges):
    """Sample particles in equilibrium for RZ geometry with multigroup."""
    nr = len(r_edges) - 1
    nz = len(z_edges) - 1
    n_groups = len(energy_edges) - 1

    volumes = _cell_volumes_rz(r_edges, z_edges)
    E_eq = __a * np.sum(Tr**4 * volumes)

    if N <= 0 or E_eq <= 0.0:
        return (
            np.array([]), np.array([]), np.array([]),
            np.array([]), np.array([]), np.array([]),
            np.array([], dtype=np.int32)
        )

    weights = np.full(N, E_eq / N)

    vol_flat = volumes.flatten()
    probs = vol_flat / np.sum(vol_flat)
    flat_indices = np.random.choice(len(vol_flat), size=N, p=probs)
    
    cell_i = flat_indices % nr
    cell_j = flat_indices // nr

    # Sample r uniformly in volume (r^2 weighting)
    r_l = r_edges[cell_i]
    r_r = r_edges[cell_i + 1]
    r2 = r_l**2 + np.random.uniform(0.0, 1.0, N) * (r_r**2 - r_l**2)
    r = np.sqrt(r2)

    z = z_edges[cell_j] + np.random.uniform(0.0, 1.0, N) * np.diff(z_edges)[cell_j]

    mu_perp, eta = _sample_isotropic_rz(N)
    times = np.zeros(N)

    # Sample groups from local Planck
    T_local = Tr[cell_i, cell_j]
    groups = np.zeros(N, dtype=np.int32)
    
    for p in range(N):
        _, grp = _sample_planck_spectrum_mixture_of_gammas(1, T_local[p], energy_edges)
        if len(grp) > 0:
            groups[p] = grp[0]

    return weights, mu_perp, eta, times, r, z, groups


def _sample_source_xy_mg(N, source, dt, x_edges, y_edges, energy_edges, temperature):
    """Sample external source particles for XY geometry with multigroup.
    
    source can be:
    - scalar: uniform in space and gray
    - (nx, ny): spatially varying, gray
    - (n_groups, nx, ny): spatially varying, multigroup
    """
    nx = len(x_edges) - 1
    ny = len(y_edges) - 1
    n_groups = len(energy_edges) - 1

    source = np.asarray(source)
    
    if source.ndim == 0:
        # Scalar - uniform gray
        source_mg = np.zeros((n_groups, nx, ny))
        source_mg[0, :, :] = source  # Put all in lowest group
    elif source.ndim == 2:
        # (nx, ny) - spatial gray, distribute to groups
        source_mg = np.zeros((n_groups, nx, ny))
        source_mg[0, :, :] = source
    elif source.ndim == 3:
        # Already multigroup
        source_mg = source
    else:
        raise ValueError(f"source shape {source.shape} not supported")

    # source_mg has units of energy/(volume*time), so multiply by volumes to get energy/time
    volumes = _cell_volumes_xy(x_edges, y_edges)
    # source_mg has units of energy/(volume*time), so multiply by volumes to get energy/time
    volumes = _cell_volumes_xy(x_edges, y_edges)
    total_source = np.sum(source_mg * volumes[np.newaxis, :, :]) * dt
    if N <= 0 or total_source <= 0.0:
        return (
            np.array([]), np.array([]), np.array([]),
            np.array([]), np.array([]), np.array([]),
            np.array([], dtype=np.int32), np.array([], dtype=np.int32)
        )

    # Sample by group then by space (use source*volume for proper weighting)
    source_by_group = np.sum(source_mg * volumes[np.newaxis, :, :], axis=(1, 2)) * dt
    group_probs = source_by_group / (total_source + 1e-300)
    
    groups_sampled = _sample_group_piecewise_constant(N, group_probs)
    
    # For each particle, sample spatial location within its group
    cell_i = np.zeros(N, dtype=np.int32)
    cell_j = np.zeros(N, dtype=np.int32)
    
    for g in range(n_groups):
        mask = (groups_sampled == g)
        n_g = np.sum(mask)
        if n_g > 0:
            # Weight cells by source*volume, not just source
            source_g_flat = (source_mg[g, :, :] * volumes).flatten()
            if np.sum(source_g_flat) > 0:
                probs_g = source_g_flat / np.sum(source_g_flat)
                flat_idx = np.random.choice(len(source_g_flat), size=n_g, p=probs_g)
                cell_i[mask] = flat_idx % nx
                cell_j[mask] = flat_idx // nx

    weights = np.full(N, total_source / N)
    
    x = x_edges[cell_i] + np.random.uniform(0.0, 1.0, N) * np.diff(x_edges)[cell_i]
    y = y_edges[cell_j] + np.random.uniform(0.0, 1.0, N) * np.diff(y_edges)[cell_j]
    
    ux, uy = _sample_isotropic_xy(N)
    times = np.random.uniform(0.0, dt, N)

    return weights, ux, uy, times, x, y, cell_i, cell_j, groups_sampled


def _sample_source_rz_mg(N, source, dt, r_edges, z_edges, energy_edges, temperature):
    """Sample external source particles for RZ geometry with multigroup."""
    nr = len(r_edges) - 1
    nz = len(z_edges) - 1
    n_groups = len(energy_edges) - 1

    source = np.asarray(source)
    
    if source.ndim == 0:
        source_mg = np.zeros((n_groups, nr, nz))
        source_mg[0, :, :] = source
    elif source.ndim == 2:
        source_mg = np.zeros((n_groups, nr, nz))
        source_mg[0, :, :] = source
    elif source.ndim == 3:
        source_mg = source
    else:
        raise ValueError(f"source shape {source.shape} not supported")

    # source_mg has units of energy/(volume*time), so multiply by volumes to get energy/time
    volumes = _cell_volumes_rz(r_edges, z_edges)
    total_source = np.sum(source_mg * volumes[np.newaxis, :, :]) * dt
    if N <= 0 or total_source <= 0.0:
        return (
            np.array([]), np.array([]), np.array([]),
            np.array([]), np.array([]), np.array([]),
            np.array([], dtype=np.int32), np.array([], dtype=np.int32),
            np.array([], dtype=np.int32)
        )

    # Sample by group then by space (use source*volume for proper weighting)
    source_by_group = np.sum(source_mg * volumes[np.newaxis, :, :], axis=(1, 2)) * dt
    group_probs = source_by_group / (total_source + 1e-300)
    groups_sampled = _sample_group_piecewise_constant(N, group_probs)
    
    cell_i = np.zeros(N, dtype=np.int32)
    cell_j = np.zeros(N, dtype=np.int32)
    
    for g in range(n_groups):
        mask = (groups_sampled == g)
        n_g = np.sum(mask)
        if n_g > 0:
            # Weight cells by source*volume, not just source
            source_g_flat = (source_mg[g, :, :] * volumes).flatten()
            if np.sum(source_g_flat) > 0:
                probs_g = source_g_flat / np.sum(source_g_flat)
                flat_idx = np.random.choice(len(source_g_flat), size=n_g, p=probs_g)
                cell_i[mask] = flat_idx % nr
                cell_j[mask] = flat_idx // nr

    weights = np.full(N, total_source / N)
    
    # Sample r with r^2 weighting
    r_l = r_edges[cell_i]
    r_r = r_edges[cell_i + 1]
    r2 = r_l**2 + np.random.uniform(0.0, 1.0, N) * (r_r**2 - r_l**2)
    r = np.sqrt(r2)
    
    z = z_edges[cell_j] + np.random.uniform(0.0, 1.0, N) * np.diff(z_edges)[cell_j]
    
    mu_perp, eta = _sample_isotropic_rz(N)
    times = np.random.uniform(0.0, dt, N)

    return weights, mu_perp, eta, times, r, z, cell_i, cell_j, groups_sampled


@jit(nopython=True, cache=True)
def _sample_groups_for_emission_jit(Ntarget, cell_i, cell_j, emission_by_group):
    """Numba-jitted vectorized group sampling for emission particles.
    
    For each particle at (cell_i[p], cell_j[p]), sample group from local distribution.
    Much faster than calling _sample_group_piecewise_constant in a loop.
    """
    g_sampled = np.zeros(Ntarget, dtype=np.int32)
    n_groups = emission_by_group.shape[0]
    
    for p in range(Ntarget):
        i_p = cell_i[p]
        j_p = cell_j[p]
        
        # Get local probabilities
        probs = emission_by_group[:, i_p, j_p]
        total_prob = np.sum(probs)
        
        if total_prob > 0.0:
            # Sample group from unnormalized distribution
            xi = np.random.random() * total_prob
            cumsum = 0.0
            for g in range(n_groups):
                cumsum += probs[g]
                if xi <= cumsum:
                    g_sampled[p] = g
                    break
        else:
            g_sampled[p] = 0
    
    return g_sampled


def _comb_mg(weights, cell_i, cell_j, groups, dir1, dir2, times, pos1, pos2, Nmax, nx, ny, n_groups):
    """Per-cell-group stochastic comb to cap total particle count.
    
    Ensures at least 1 particle per (cell_i, cell_j, group) bin to avoid losing
    energy in any region or group. Stochastically splits/merges particles within
    each bin to target Nmax total.
    """
    # Keep all positive-weight particles. For robustness, clip indices into
    # valid cell bounds rather than discarding out-of-range particles.
    alive = (weights > 0.0)
    weights = weights[alive]
    cell_i = np.clip(cell_i[alive], 0, nx - 1)
    cell_j = np.clip(cell_j[alive], 0, ny - 1)
    groups = groups[alive]
    dir1 = dir1[alive]
    dir2 = dir2[alive]
    times = times[alive]
    pos1 = pos1[alive]
    pos2 = pos2[alive]
    
    if len(weights) == 0:
        return weights, cell_i, cell_j, groups, dir1, dir2, times, pos1, pos2, np.zeros((n_groups, nx, ny))
    
    # Compute energy per (cell_i, cell_j, group) bin before combing
    # bin_id = cell_i + cell_j * nx + group * nx * ny
    n_bins = nx * ny * n_groups
    bin_id = cell_i + cell_j * nx + groups * (nx * ny)
    bin_id = bin_id.astype(np.int32)
    
    ecen = np.bincount(bin_id, weights=weights, minlength=n_bins)
    E = np.sum(ecen)
    if E <= 0.0:
        return weights, cell_i, cell_j, groups, dir1, dir2, times, pos1, pos2, np.zeros((n_groups, nx, ny))
    
    # Desired number of particles per bin: at least 1 if bin has energy
    desired = np.where(ecen > 0.0, np.maximum(1, np.round(Nmax * ecen / E).astype(int)), 0)
    ew = np.zeros_like(ecen)
    nonzero_desired = desired > 0
    ew[nonzero_desired] = ecen[nonzero_desired] / desired[nonzero_desired]
    
    nw = []
    ni = []
    nj = []
    ng = []
    nd1 = []
    nd2 = []
    nt = []
    np1 = []
    np2 = []
    
    for k in range(len(weights)):
        b = int(bin_id[k])
        w_target = ew[b]
        if w_target <= 0.0:
            continue
        n = int(weights[k] / w_target + random.random())
        for _ in range(n):
            nw.append(w_target)
            ni.append(cell_i[k])
            nj.append(cell_j[k])
            ng.append(groups[k])
            nd1.append(dir1[k])
            nd2.append(dir2[k])
            nt.append(times[k])
            np1.append(pos1[k])
            np2.append(pos2[k])
    
    nw = np.array(nw)
    ni = np.array(ni, dtype=np.int32)
    nj = np.array(nj, dtype=np.int32)
    ng = np.array(ng, dtype=np.int32)
    nd1 = np.array(nd1)
    nd2 = np.array(nd2)
    nt = np.array(nt)
    np1 = np.array(np1)
    np2 = np.array(np2)
    
    # Compute pre- and post-comb radiation energy by group/cell.
    # The returned array is the comb-induced discrepancy: pre - post.
    rad_energy_before = np.zeros((n_groups, nx, ny))
    for g in range(n_groups):
        mask = (groups == g)
        if np.any(mask):
            flat_indices = cell_i[mask] + cell_j[mask] * nx
            rad_by_cell = np.bincount(flat_indices, weights=weights[mask], minlength=nx * ny).reshape(nx, ny)
            rad_energy_before[g, :, :] = rad_by_cell

    rad_energy_after = np.zeros((n_groups, nx, ny))
    if len(nw) > 0:
        for g in range(n_groups):
            mask = (ng == g)
            if np.any(mask):
                flat_indices = ni[mask] + nj[mask] * nx
                rad_by_cell = np.bincount(flat_indices, weights=nw[mask], minlength=nx * ny).reshape(nx, ny)
                rad_energy_after[g, :, :] = rad_by_cell

    comb_energy_discrepancy = rad_energy_before - rad_energy_after

    return nw, ni, nj, ng, nd1, nd2, nt, np1, np2, comb_energy_discrepancy


def init_simulation(
    Ntarget,
    Tinit,
    Tr_init,
    edges1,
    edges2,
    energy_edges,
    eos,
    inv_eos,
    Ntarget_ic=None,
    geometry="xy",
):
    """Initialize multigroup particle arrays and material state for 2D IMC.
    
    Parameters
    ----------
    Ntarget : int
        Target number of particles for material emission
    Tinit : array (nx, ny)
        Initial material temperature (keV)
    Tr_init : array (nx, ny)
        Initial radiation temperature (keV)
    edges1 : array
        x or r edges
    edges2 : array
        y or z edges
    energy_edges : array
        Energy group edges (keV)
    eos : callable
        Material energy as function of temperature
    inv_eos : callable
        Temperature as function of energy
    Ntarget_ic : int, optional
        Number of initial condition particles
    geometry : str
        'xy' or 'rz'
    
    Returns
    -------
    state : SimulationState2DMG
        Initial state
    """
    nx, ny = _shape_from_edges(edges1, edges2)
    n_groups = len(energy_edges) - 1
    volumes = _cell_volumes(edges1, edges2, geometry)

    internal_energy = eos(Tinit)
    temperature = Tinit.copy()
    assert np.allclose(inv_eos(internal_energy), Tinit), "Inverse EOS failed"

    N_ic = Ntarget if Ntarget_ic is None else Ntarget_ic
    if geometry == "xy":
        p = _equilibrium_sample_xy_mg(N_ic, Tr_init, edges1, edges2, energy_edges)
    elif geometry == "rz":
        p = _equilibrium_sample_rz_mg(N_ic, Tr_init, edges1, edges2, energy_edges)
    else:
        raise ValueError(f"Unknown geometry: {geometry}")

    weights, dir1, dir2, times, pos1, pos2, groups = p

    # Locate particles in cells
    cell_i, cell_j = _locate_indices(pos1, pos2, edges1, edges2)

    # Compute radiation energy by group
    radiation_energy_by_group = np.zeros((n_groups, nx, ny))
    for g in range(n_groups):
        mask = (groups == g)
        if np.any(mask):
            flat = _flatten_index(cell_i[mask], cell_j[mask], nx)
            rad_cell_g = np.bincount(flat, weights=weights[mask], minlength=nx * ny).reshape(nx, ny)
            radiation_energy_by_group[g, :, :] = rad_cell_g / volumes

    # Overall radiation temperature
    total_rad = np.sum(radiation_energy_by_group, axis=0)
    radiation_temperature = (total_rad / __a) ** 0.25

    total_internal = float(np.sum(internal_energy * volumes))
    total_rad = float(np.sum(weights))
    previous_total = total_internal + total_rad

    print(
        "Time",
        "N",
        "Total Energy",
        "Total Internal Energy",
        "Total Radiation Energy",
        "Boundary Emission",
        "Lost Energy",
        sep="\t",
    )
    print("=" * 111)
    print(
        "{:.6f}".format(0.0),
        len(weights),
        "{:.6f}".format(previous_total),
        "{:.6f}".format(total_internal),
        "{:.6f}".format(total_rad),
        "{:.6f}".format(0.0),
        "{:.6f}".format(0.0),
        sep="\t",
    )

    return SimulationState2DMG(
        weights=weights,
        dir1=dir1,
        dir2=dir2,
        times=times,
        pos1=pos1,
        pos2=pos2,
        cell_i=cell_i,
        cell_j=cell_j,
        groups=groups,
        internal_energy=internal_energy,
        temperature=temperature,
        radiation_temperature=radiation_temperature,
        radiation_energy_by_group=radiation_energy_by_group,
        time=0.0,
        previous_total_energy=previous_total,
        count=0,
    )


def step(
    state,
    Ntarget,
    Nboundary,
    Nsource,
    Nmax,
    T_boundary,
    dt,
    edges1,
    edges2,
    energy_edges,
    sigma_a_funcs,  # List of callables: sigma_a_g(T) for each group
    inv_eos,
    cv,
    source,
    reflect=(False, False, False, False),
    theta=1.0,
    use_scalar_intensity_Tr=True,
    conserve_comb_energy=False,
    geometry="xy",
    max_events_per_particle=1_000_000,
    boundary_source_func=None,
    emission_fractions=None,
    _timing=False,
):
    """Advance one 2D multigroup IMC step.
    
    Parameters
    ----------
    state : SimulationState2DMG
        Current state
    sigma_a_funcs : list of callables
        List of absorption opacity functions, one per group
        Each function takes temperature array and returns opacity array
    ... (other parameters similar to IMC2D.step)
    
    Returns
    -------
    state : SimulationState2DMG
        Updated state
    info : dict
        Step information
    """
    nx, ny = _shape_from_edges(edges1, edges2)
    n_groups = len(energy_edges) - 1
    volumes = _cell_volumes(edges1, edges2, geometry)

    weights = state.weights
    dir1 = state.dir1
    dir2 = state.dir2
    times = state.times
    pos1 = state.pos1
    pos2 = state.pos2
    cell_i = state.cell_i
    cell_j = state.cell_j
    groups = state.groups
    internal_energy = state.internal_energy
    temperature = state.temperature

    t_step_start = _time.perf_counter()

    # Compute group-dependent opacities and Fleck factors
    sigma_a = np.zeros((n_groups, nx, ny))
    for g in range(n_groups):
        sigma_a[g, :, :] = sigma_a_funcs[g](temperature)

    # Compute Planck-weighted opacity for Fleck factor (vectorized)
    B_g = _compute_Bg_multigroup_grid(energy_edges, temperature)  # (n_groups, nx, ny)

    sigma_P = np.sum(sigma_a * B_g, axis=0) / (np.sum(B_g, axis=0) + 1e-300)

    # Fleck factor (scalar, independent of group)
    beta = 4.0 * __a * temperature**3 / cv(temperature)
    f = 1.0 / (1.0 + theta * beta * sigma_P * __c * dt)
    f = np.clip(f, 0.0, 1.0)

    # Effective scattering and absorption by group
    sigma_s = np.zeros((n_groups, nx, ny))
    for g in range(n_groups):
        sigma_s[g, :, :] = sigma_a[g, :, :] * (1.0 - f)
        sigma_a[g, :, :] = sigma_a[g, :, :] * f

    # Boundary injection
    b_left = _boundary_temperature_value(T_boundary[0], state.time)
    b_right = _boundary_temperature_value(T_boundary[1], state.time)
    b_bottom = _boundary_temperature_value(T_boundary[2], state.time)
    b_top = _boundary_temperature_value(T_boundary[3], state.time)

    boundary_emission = 0.0

    if Nboundary > 0:
        if geometry == "xy":
            for side, Tb in (
                ("left", b_left),
                ("right", b_right),
                ("bottom", b_bottom),
                ("top", b_top),
            ):
                s = _sample_boundary_xy(Nboundary, side, Tb, dt, edges1, edges2, energy_edges, boundary_source_func)
                if s is None:
                    continue
                w, d1, d2, t, p1, p2, g = s
                ci, cj = _locate_indices(p1, p2, edges1, edges2)
                weights = np.concatenate((weights, w))
                dir1 = np.concatenate((dir1, d1))
                dir2 = np.concatenate((dir2, d2))
                times = np.concatenate((times, t))
                pos1 = np.concatenate((pos1, p1))
                pos2 = np.concatenate((pos2, p2))
                cell_i = np.concatenate((cell_i, ci))
                cell_j = np.concatenate((cell_j, cj))
                groups = np.concatenate((groups, g))
                boundary_emission += float(np.sum(w))
        else:
            for side, Tb in (
                ("rmin", b_left),
                ("rmax", b_right),
                ("zmin", b_bottom),
                ("zmax", b_top),
            ):
                s = _sample_boundary_rz(Nboundary, side, Tb, dt, edges1, edges2, energy_edges, boundary_source_func)
                if s is None:
                    continue
                w, d1, d2, t, p1, p2, g = s
                ci, cj = _locate_indices(p1, p2, edges1, edges2)
                valid = (ci >= 0) & (ci < nx) & (cj >= 0) & (cj < ny)
                if np.any(valid):
                    weights = np.concatenate((weights, w[valid]))
                    dir1 = np.concatenate((dir1, d1[valid]))
                    dir2 = np.concatenate((dir2, d2[valid]))
                    times = np.concatenate((times, t[valid]))
                    pos1 = np.concatenate((pos1, p1[valid]))
                    pos2 = np.concatenate((pos2, p2[valid]))
                    cell_i = np.concatenate((cell_i, ci[valid]))
                    cell_j = np.concatenate((cell_j, cj[valid]))
                    groups = np.concatenate((groups, g[valid]))
                    boundary_emission += float(np.sum(w[valid]))

    # Fixed source
    source_emission = 0.0
    if Nsource > 0 and np.max(source) > 0.0:
        if geometry == "xy":
            s = _sample_source_xy_mg(Nsource, source, dt, edges1, edges2, energy_edges, temperature)
        else:
            s = _sample_source_rz_mg(Nsource, source, dt, edges1, edges2, energy_edges, temperature)
        
        if len(s[0]) > 0:
            w, d1, d2, t, p1, p2, ci, cj, g = s
            weights = np.concatenate((weights, w))
            dir1 = np.concatenate((dir1, d1))
            dir2 = np.concatenate((dir2, d2))
            times = np.concatenate((times, t))
            pos1 = np.concatenate((pos1, p1))
            pos2 = np.concatenate((pos2, p2))
            cell_i = np.concatenate((cell_i, ci))
            cell_j = np.concatenate((cell_j, cj))
            groups = np.concatenate((groups, g))
            source_emission = float(np.sum(w))

    # Material emission: sample groups from piecewise constant distribution
    # proportional to σ_a,g b_g★ (equation 10.18)
    emission_by_group = np.zeros((n_groups, nx, ny))
    
    if Ntarget > 0:
        # Compute emission probabilities by group
        if emission_fractions is not None:
            # Use custom emission fractions (e.g., for picket fence problem)
            b_star = np.zeros((n_groups, nx, ny))
            for g in range(n_groups):
                b_star[g, :, :] = emission_fractions[g]
        else:
            # Reuse B_g computed above for Fleck factors to avoid duplicate work.
            b_star = B_g.copy()
            
            # Normalize b_star so it sums to 1 across groups
            # This ensures we emit the full acT^4 energy even when groups don't cover full spectrum
            b_sum = np.sum(b_star, axis=0)  # Sum over groups
            b_star = b_star / b_sum[None, :, :]  # Normalize

        # Emission rate by group: σ_a,g f b_g★ c a T^4 Δt V
        # Note: sigma_a has already been modified by Fleck factor (f) at line 1280
        emission_by_group = sigma_a * b_star * __a * __c * temperature[None, :, :]**4 * dt * volumes[None, :, :]

        
        # Sample particles: first by position weighted by total emission, then by group
        total_emission_per_cell = np.sum(emission_by_group, axis=0)
        E_emit = float(np.sum(total_emission_per_cell))
        
        if E_emit > 0.0:
            # Sample cell locations
            emission_flat = total_emission_per_cell.flatten()
            probs_cell = emission_flat / E_emit
            flat_indices = np.random.choice(len(emission_flat), size=Ntarget, p=probs_cell)
            
            ci = flat_indices % nx
            cj = flat_indices // nx
            
            # Vectorized group sampling using JIT-compiled function
            g_sampled = _sample_groups_for_emission_jit(Ntarget, ci, cj, emission_by_group)
            
            # Sample positions within cells
            if geometry == "xy":
                p1 = edges1[ci] + np.random.uniform(0.0, 1.0, Ntarget) * np.diff(edges1)[ci]
                p2 = edges2[cj] + np.random.uniform(0.0, 1.0, Ntarget) * np.diff(edges2)[cj]
                d1, d2 = _sample_isotropic_xy(Ntarget)
            else:
                r_l = edges1[ci]
                r_r = edges1[ci + 1]
                r2 = r_l**2 + np.random.uniform(0.0, 1.0, Ntarget) * (r_r**2 - r_l**2)
                p1 = np.sqrt(r2)
                p2 = edges2[cj] + np.random.uniform(0.0, 1.0, Ntarget) * np.diff(edges2)[cj]
                d1, d2 = _sample_isotropic_rz(Ntarget)
            
            t = np.random.uniform(0.0, dt, Ntarget)
            w = np.full(Ntarget, E_emit / Ntarget)
            
            weights = np.concatenate((weights, w))
            dir1 = np.concatenate((dir1, d1))
            dir2 = np.concatenate((dir2, d2))
            times = np.concatenate((times, t))
            pos1 = np.concatenate((pos1, p1))
            pos2 = np.concatenate((pos2, p2))
            cell_i = np.concatenate((cell_i, ci))
            cell_j = np.concatenate((cell_j, cj))
            groups = np.concatenate((groups, g_sampled))

    # Transport particles
    t_transport_start = _time.perf_counter()
    n_particles_transported = len(weights)
    geometry_code = _GEOM_XY if geometry == "xy" else _GEOM_RZ
    weight_floor = 1e-10 * float(np.sum(weights)) / max(len(weights), 1)
    
    if state.count == 0:
        print(f"[MG_IMC2D] Using {get_num_threads()} threads for transport")
    
    dep_cell, si_cell, boundary_loss, stats = _transport_particles_2d_mg(
        weights,
        dir1,
        dir2,
        times,
        pos1,
        pos2,
        cell_i,
        cell_j,
        groups,
        edges1,
        edges2,
        sigma_a,
        sigma_s,
        volumes,
        dt,
        reflect,
        max_events_per_particle,
        geometry_code,
        weight_floor,
    )
    t_post_start = _time.perf_counter()

    # Material/radiation update
    total_deposited = np.sum(dep_cell, axis=0)
    total_emitted = np.sum(emission_by_group if Ntarget > 0 else 0.0, axis=0) / volumes
    
    internal_energy = internal_energy + total_deposited - total_emitted
    temperature = inv_eos(internal_energy)

    # Update radiation energy by group
    radiation_energy_by_group = np.zeros((n_groups, nx, ny))
    for g in range(n_groups):
        if use_scalar_intensity_Tr:
            # si_cell stores scalar intensity by group. Convert to group energy
            # density via E_g = I_g / c, then Tr from sum_g(E_g).
            radiation_energy_by_group[g, :, :] = si_cell[g, :, :] / __c
        else:
            mask = (
                (groups == g)
                & (weights > 0.0)
                & (cell_i >= 0)
                & (cell_i < nx)
                & (cell_j >= 0)
                & (cell_j < ny)
            )
            if np.any(mask):
                flat = _flatten_index(cell_i[mask], cell_j[mask], nx)
                rad_cell_g = np.bincount(flat, weights=weights[mask], minlength=nx * ny).reshape(nx, ny)
                radiation_energy_by_group[g, :, :] = rad_cell_g / volumes

    # Combing
    (
        weights,
        cell_i,
        cell_j,
        groups,
        dir1,
        dir2,
        times,
        pos1,
        pos2,
        comb_disc,
    ) = _comb_mg(weights, cell_i, cell_j, groups, dir1, dir2, times, pos1, pos2, Nmax, nx, ny, n_groups)

    if conserve_comb_energy:
        total_comb_disc = np.sum(comb_disc, axis=0)
        internal_energy = internal_energy + total_comb_disc / volumes
        temperature = inv_eos(internal_energy)

    # Rebuild radiation group energies from post-comb particle weights so
    # state and diagnostics are consistent with the final particle population.
    radiation_energy_by_group = np.zeros((n_groups, nx, ny))
    for g in range(n_groups):
        mask = (
            (groups == g)
            & (weights > 0.0)
            & (cell_i >= 0)
            & (cell_i < nx)
            & (cell_j >= 0)
            & (cell_j < ny)
        )
        if np.any(mask):
            flat = _flatten_index(cell_i[mask], cell_j[mask], nx)
            rad_cell_g = np.bincount(flat, weights=weights[mask], minlength=nx * ny).reshape(nx, ny)
            radiation_energy_by_group[g, :, :] = rad_cell_g / volumes

    total_rad_energy = np.sum(radiation_energy_by_group, axis=0)
    radiation_temperature = (total_rad_energy / __a) ** 0.25

    times = np.zeros_like(times)

    total_internal = float(np.sum(internal_energy * volumes))
    total_rad = float(np.sum(weights))
    total_energy = total_internal + total_rad
    energy_loss = (
        total_energy
        - state.previous_total_energy
        - boundary_emission
        + boundary_loss
        - source_emission
    )

    state.weights = weights
    state.dir1 = dir1
    state.dir2 = dir2
    state.times = times
    state.pos1 = pos1
    state.pos2 = pos2
    state.cell_i = cell_i
    state.cell_j = cell_j
    state.groups = groups
    state.internal_energy = internal_energy
    state.temperature = temperature
    state.radiation_temperature = radiation_temperature
    state.radiation_energy_by_group = radiation_energy_by_group
    state.time += dt
    state.previous_total_energy = total_energy
    state.count += 1

    t_end = _time.perf_counter()
    events_total = int(stats[0])
    n_transported = max(int(n_particles_transported), 1)

    info = {
        "time": state.time,
        "temperature": temperature,
        "radiation_temperature": radiation_temperature,
        "radiation_energy_by_group": radiation_energy_by_group,
        "N_particles": len(weights),
        "total_energy": total_energy,
        "total_internal_energy": total_internal,
        "total_radiation_energy": total_rad,
        "boundary_emission": boundary_emission,
        "boundary_loss": boundary_loss,
        "source_emission": source_emission,
        "energy_loss": energy_loss,
        "profiling": {
            "phase_times_s": {
                "sampling": t_transport_start - t_step_start,
                "transport": t_post_start - t_transport_start,
                "postprocess": t_end - t_post_start,
                "total": t_end - t_step_start,
            },
            "transport_events": {
                "total": events_total,
                "boundary_crossings": int(stats[1]),
                "absorption_continue_events": int(stats[2]),
                "census_events": int(stats[3]),
                "absorption_capture_events": int(stats[4]),
                "weight_floor_kills": int(stats[5]),
                "reflections": int(stats[6]),
                "event_cap_hits": int(stats[7]),
                "avg_events_per_particle": events_total / n_transported,
                "n_particles_transported": int(n_transported),
            },
        },
    }

    return state, info


def run_simulation(
    Ntarget,
    Nboundary,
    Nsource,
    Nmax,
    Tinit,
    Tr_init,
    T_boundary,
    dt,
    edges1,
    edges2,
    energy_edges,
    sigma_a_funcs,
    eos,
    inv_eos,
    cv,
    source,
    final_time,
    reflect=(False, False, False, False),
    output_freq=1,
    theta=1.0,
    use_scalar_intensity_Tr=True,
    Ntarget_ic=None,
    conserve_comb_energy=False,
    geometry="xy",
    max_events_per_particle=1_000_000,
    emission_fractions=None,
):
    """Run full multigroup IMC simulation.
    
    Parameters
    ----------
    energy_edges : array
        Energy group boundaries (keV), length n_groups + 1
    sigma_a_funcs : list of callables
        Absorption opacity functions for each group
    emission_fractions : array or None, optional
        Custom emission fractions (must sum to 1.0). If None, uses Planck integrals.
        For picket fence problems, use np.array([0.5, 0.5]) for equal emission.
    ... (other parameters similar to IMC2D.run_simulation)
    
    Returns
    -------
    history : list of info dicts
        Simulation history
    state : SimulationState2DMG
        Final state
    """
    state = init_simulation(
        Ntarget,
        Tinit,
        Tr_init,
        edges1,
        edges2,
        energy_edges,
        eos,
        inv_eos,
        Ntarget_ic=Ntarget_ic,
        geometry=geometry,
    )

    history = []
    t = 0.0
    step_count = 0

    time_tol = max(1e-15, 1e-12 * max(final_time, 1.0))
    while t < final_time - time_tol:
        dt_step = min(dt, final_time - t)
        if dt_step <= time_tol:
            break
        
        state, info = step(
            state,
            Ntarget,
            Nboundary,
            Nsource,
            Nmax,
            T_boundary,
            dt_step,
            edges1,
            edges2,
            energy_edges,
            sigma_a_funcs,
            inv_eos,
            cv,
            source,
            reflect=reflect,
            theta=theta,
            use_scalar_intensity_Tr=use_scalar_intensity_Tr,
            conserve_comb_energy=conserve_comb_energy,
            geometry=geometry,
            max_events_per_particle=max_events_per_particle,
            emission_fractions=emission_fractions,
        )

        t = state.time
        step_count += 1

        if step_count % output_freq == 0 or (final_time - t) < time_tol:
            history.append(info)
            print(
                "{:.6f}".format(t),
                info["N_particles"],
                "{:.6f}".format(info["total_energy"]),
                "{:.6f}".format(info["total_internal_energy"]),
                "{:.6f}".format(info["total_radiation_energy"]),
                "{:.6f}".format(info["boundary_emission"]),
                "{:.6f}".format(info["energy_loss"]),
                sep="\t",
            )

    return history, state


if __name__ == "__main__":
    print("Multigroup IMC 2D - Core module loaded successfully")
    print(f"Planck integrals available: {_PLANCK_AVAILABLE}")
