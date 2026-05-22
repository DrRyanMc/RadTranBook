"""MG_IMC1D.py — Multigroup Implicit Monte Carlo in 1-D spherical geometry.

Extends the gray 1-D spherical IMC (IMC1D.py) to support multigroup radiation
transport.  Each MC particle carries a group index g in addition to the usual
(weight, mu, r, t) state.  The multigroup IMC equations implemented here are:

    f = 1 / (1 + β c σ_P Δt),    σ_P = Σ_g σ_a,g B_g / B_total

    Emission per group per cell:
        E_{g,i} = a c T⁴ f σ_{a,g} b_{g}★ Δt V_i
    where  b_{g}★ = B_g(T) / B_total(T)

Tracking is identical to the gray spherical case (Box 9.2):
    r' = sqrt(r² + 2rμs + s²),   μ' = (rμ + s) / r'
    s± = −rμ ± sqrt(R² − b²),    b² = r²(1 − μ²)

The cell volumes are  V_i = (4/3)π(r₁³ − r₀³).

Units (same as IMC1D / MG_IMC2D):
  distance: cm,  time: ns,  temperature: keV,  energy: GJ

Public API
----------
    init_simulation(...)  → SimulationState1DMG
    step(state, ...)      → (state, info)
    run_simulation(...)   → (time_values, Tr_history, T_history, state, history)
"""

import math
import random
import time as _time
from dataclasses import dataclass
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Optional Numba JIT support
# ---------------------------------------------------------------------------
try:
    from numba import jit, prange, get_thread_id, get_num_threads
    _NUMBA_AVAILABLE = True
except Exception:
    _NUMBA_AVAILABLE = False

    def jit(*jit_args, **jit_kwargs):
        if len(jit_args) == 1 and callable(jit_args[0]) and not jit_kwargs:
            return jit_args[0]
        def _decorator(func):
            return func
        return _decorator

    def prange(*args):
        return range(*args)

    def get_thread_id():
        return 0

    def get_num_threads():
        return 1

# ---------------------------------------------------------------------------
# Optional Planck integral library
# ---------------------------------------------------------------------------
try:
    from planck_integrals import Bg, dBgdT, Bg_multigroup, dBgdT_multigroup
    _PLANCK_AVAILABLE = True
except ImportError:
    _PLANCK_AVAILABLE = False

    def Bg(E_low, E_high, T):
        """Gray approximation: total B = (c a / 4) T⁴ / (4π), split equally."""
        C_LIGHT = 29.98
        A_RAD = 0.01372
        return (C_LIGHT * A_RAD / 4.0) * T**4

    def dBgdT(E_low, E_high, T):
        C_LIGHT = 29.98
        A_RAD = 0.01372
        return (C_LIGHT * A_RAD) * T**3

    def Bg_multigroup(energy_edges, T):
        C_LIGHT = 29.98
        A_RAD = 0.01372
        n_groups = len(energy_edges) - 1
        total_B = (C_LIGHT * A_RAD / 4.0) * T**4
        return np.full(n_groups, total_B / n_groups)

    def dBgdT_multigroup(energy_edges, T):
        C_LIGHT = 29.98
        A_RAD = 0.01372
        n_groups = len(energy_edges) - 1
        total_dB = (C_LIGHT * A_RAD) * T**3
        return np.full(n_groups, total_dB / n_groups)

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
__c = 29.98    # speed of light  (cm / ns)
__a = 0.01372  # radiation constant a = 4σ/c  (GJ / cm³ / keV⁴)

# ---------------------------------------------------------------------------
# Planck sampling: mixture-of-Gammas coefficients
# ---------------------------------------------------------------------------
_PLANCK_N_MAX = 200
_PLANCK_N_VALUES = np.arange(1, _PLANCK_N_MAX + 1, dtype=np.float64)
_PLANCK_MIXTURE_CDF = np.cumsum(
    (1.0 / _PLANCK_N_VALUES**4) / np.sum(1.0 / _PLANCK_N_VALUES**4)
)


# ---------------------------------------------------------------------------
# Simulation state
# ---------------------------------------------------------------------------
@dataclass
class SimulationState1DMG:
    """Mutable 1-D spherical multigroup IMC state passed between time steps."""
    weights:               np.ndarray  # particle weights
    mus:                   np.ndarray  # direction cosine μ = r̂·Ω̂
    times:                 np.ndarray  # birth time within step
    positions:             np.ndarray  # radial position r (cm)
    cell_indices:          np.ndarray  # cell index (int)
    groups:                np.ndarray  # energy group index (int)

    internal_energy:       np.ndarray  # u_i  (GJ / cm³)
    temperature:           np.ndarray  # T_i  (keV)
    radiation_temperature: np.ndarray  # Tr_i (keV) from scalar intensity

    # Shape (n_groups, n_cells) — energy density per group per cell
    radiation_energy_by_group: np.ndarray
    # Shape (n_groups, n_cells) — radial flux  F_g = c·Σ(w_k μ_k)/V  (GJ/cm²/ns)
    radiation_flux_by_group: np.ndarray

    time:                  float
    previous_total_energy: float
    count:                 int = 0
    radiation_energy_by_group_postcomb: Optional[np.ndarray] = None


# ===========================================================================
# SECTION 1 — PLANCK SPECTRUM SAMPLING (mixture-of-Gammas)
# ===========================================================================

@jit(nopython=True, cache=True)
def _sample_planck_spectrum_mg_jit(n, T, energy_edges_low, energy_edges_high, cdf):
    """Numba-compiled Planck sampler (mixture of Gamma(4, n_s) distributions).

    Returns (frequencies, groups) where frequencies are in keV and groups are
    integer indices into the energy group structure.

    Algorithm (equations 10.23–10.27):
      1. Sample n_s from the mixture weights  w_n ∝ 1/n⁴.
      2. Sample x ~ Gamma(4, 1/n_s) = sum of 4 Exp(n_s) r.v.s.
      3. Convert to frequency: ν = T * x.
      4. Assign to group by searching the energy edges.
    """
    if n <= 0:
        return np.zeros(n), np.zeros(n, dtype=np.int32)
    if T <= 0.0:
        return np.full(n, energy_edges_low[0]), np.zeros(n, dtype=np.int32)

    frequencies = np.zeros(n)
    groups = np.zeros(n, dtype=np.int32)
    n_groups = len(energy_edges_low)
    n_max = len(cdf)

    for p in range(n):
        # --- Step 1: sample n_s from mixture CDF ---
        xi = np.random.random()
        n_s = float(n_max)
        for k in range(n_max):
            if xi <= cdf[k]:
                n_s = float(k + 1)
                break

        # --- Step 2: Gamma(4, n_s) via sum of 4 exponentials ---
        r1 = -np.log(np.random.random() + 1e-300)
        r2 = -np.log(np.random.random() + 1e-300)
        r3 = -np.log(np.random.random() + 1e-300)
        r4 = -np.log(np.random.random() + 1e-300)
        x = (r1 + r2 + r3 + r4) / n_s

        # --- Step 3: frequency ---
        freq = T * x
        frequencies[p] = freq

        # --- Step 4: assign group ---
        g = n_groups - 1
        for i in range(n_groups):
            if freq >= energy_edges_low[i] and freq < energy_edges_high[i]:
                g = i
                break
        groups[p] = g

    return frequencies, groups


def _sample_planck_spectrum_mg(n, T, energy_edges):
    """Python wrapper for the Planck spectrum sampler."""
    if n <= 0:
        return np.array([]), np.array([], dtype=np.int32)
    edges_low  = energy_edges[:-1].astype(np.float64)
    edges_high = energy_edges[1:].astype(np.float64)
    return _sample_planck_spectrum_mg_jit(
        n, float(T), edges_low, edges_high, _PLANCK_MIXTURE_CDF
    )


@jit(nopython=True, cache=True)
def _sample_group_cdf_jit(n, cdf_normalized):
    """Sample group indices from a pre-built CDF (equation 10.18)."""
    if n <= 0:
        return np.zeros(n, dtype=np.int32)
    groups = np.zeros(n, dtype=np.int32)
    n_g = len(cdf_normalized)
    for p in range(n):
        xi = np.random.random()
        g = n_g - 1
        for k in range(n_g):
            if xi <= cdf_normalized[k]:
                g = k
                break
        groups[p] = g
    return groups


def _sample_group_piecewise_constant(n, probabilities):
    """Sample groups proportional to `probabilities` (must be non-negative)."""
    if n <= 0:
        return np.array([], dtype=np.int32)
    probs = np.asarray(probabilities, dtype=np.float64)
    s = probs.sum()
    probs = probs / s if s > 0.0 else np.ones_like(probs) / len(probs)
    cdf = np.cumsum(probs)
    cdf[-1] = 1.0
    return _sample_group_cdf_jit(n, cdf)


# ===========================================================================
# SECTION 2 — PLANCK INTEGRALS ON 1-D GRID
# ===========================================================================

def _compute_Bg_1d(energy_edges, temperature):
    """Return B_g array of shape (n_groups, n_cells).

    B_g[g, i] = integral of B(ν, T_i) from energy_edges[g] to energy_edges[g+1].
    """
    n_groups = len(energy_edges) - 1
    n_cells  = len(temperature)
    B_g = np.zeros((n_groups, n_cells))
    for g in range(n_groups):
        E_lo = energy_edges[g]
        E_hi = energy_edges[g + 1]
        try:
            vals = Bg(E_lo, E_hi, temperature)
            if np.isscalar(vals):
                B_g[g, :] = vals
            elif vals.shape == temperature.shape:
                B_g[g, :] = vals
            else:
                for i in range(n_cells):
                    B_g[g, i] = Bg(E_lo, E_hi, temperature[i])
        except Exception:
            for i in range(n_cells):
                B_g[g, i] = Bg(E_lo, E_hi, temperature[i])
    return B_g


# ===========================================================================
# SECTION 3 — SPHERICAL PARTICLE TRANSPORT (group-aware)
# ===========================================================================
# The single-particle move kernel is identical to the gray version from
# IMC1D.py.  We copy it here (unchanged) so MG_IMC1D is self-contained and
# avoids a cross-module dependency on Numba-cached functions.

@jit(nopython=True, cache=True)
def _move_particle_spherical(weight, mu, r, r_inner, r_outer,
                              sigma_a, sigma_s, distance_to_census):
    """Move one spherical-geometry particle to the next event.

    Identical to IMC1D.move_particle_spherical — see that module for full
    documentation (Box 9.2 tracking equations).

    Returns
    -------
    (weight, mu_new, r_new, new_location,
     deposited_weight, deposited_intensity, distance_to_census, event_code)

    new_location : -1 crossed inner shell, 0 scatter/census, +1 outer shell
    event_code   : 1 boundary, 2 scatter, 3 census
    """
    b2 = r * r * (1.0 - mu * mu)          # impact parameter squared

    # Distance to outer shell
    disc_outer = r_outer * r_outer - b2
    dist_outer = -r * mu + math.sqrt(disc_outer)   # always positive

    # Distance to inner shell (only relevant if moving inward)
    disc_inner = r_inner * r_inner - b2
    if mu < 0.0 and disc_inner > 0.0:
        dist_inner = -r * mu - math.sqrt(disc_inner)
    else:
        dist_inner = 1e30

    distance_to_boundary = min(dist_outer, dist_inner)
    hit_outer = (dist_outer <= dist_inner)

    dr = r_outer - r_inner
    if sigma_s > 1e-10:
        distance_to_scatter = -math.log(1.0 - random.random()) / sigma_s
    else:
        distance_to_scatter = 1e30
    if sigma_s * dr > 10000.0:
        distance_to_scatter = -math.log(1.0 - random.random()) / (10000.0 / dr)

    s = min(distance_to_boundary, distance_to_scatter, distance_to_census)
    assert s > 0.0, "Non-positive distance in spherical transport"

    # --- Box 9.2 streaming update ---
    r_new2 = r * r + 2.0 * r * s * mu + s * s
    r_new  = math.sqrt(r_new2) if r_new2 > 0.0 else 0.0
    if r_new > 1e-15:
        mu_new = (r * mu + s) / r_new
    else:
        mu_new = 1.0
    mu_new = max(-1.0, min(1.0, mu_new))

    new_location = 0
    if math.fabs(distance_to_boundary - s) < 1e-10:
        if hit_outer:
            r_new  = r_outer
            mu_new = (r * mu + s) / r_outer if r_outer > 1e-15 else 1.0
            mu_new = max(-1.0, min(1.0, mu_new))
            new_location = 1
        else:
            r_new  = r_inner
            mu_new = (r * mu + s) / r_inner if r_inner > 1e-15 else -1.0
            mu_new = max(-1.0, min(1.0, mu_new))
            new_location = -1
    elif s == distance_to_scatter:
        mu_new = random.uniform(-1.0, 1.0)

    assert r_new >= r_inner and r_new <= r_outer, "Particle left spherical cell"

    # --- Energy deposition, normalised by cell volume ---
    cell_vol = (4.0 / 3.0) * math.pi * (r_outer**3 - r_inner**3)
    distance_to_census -= s
    weight_factor = math.exp(-sigma_a * s)
    if sigma_a > 1e-10:
        deposited_intensity = weight * (1.0 - weight_factor) / sigma_a / cell_vol
    else:
        deposited_intensity = weight * s / cell_vol
    deposited_weight = deposited_intensity * sigma_a
    weight *= weight_factor

    if math.fabs(distance_to_boundary - s) < 1e-10:
        event_code = 1
    elif s == distance_to_scatter:
        event_code = 2
    else:
        event_code = 3

    return (weight, mu_new, r_new, new_location,
            deposited_weight, deposited_intensity, distance_to_census, event_code)


@jit(nopython=True, cache=True)
def _move_sph_slice_mg(weights, mus, times, positions, cell_indices, groups,
                       mesh, sigma_a, sigma_s, n_groups, dt, refl,
                       weight_floor, dep, si, stats, boundary_loss_by_group):
    """Serial multigroup transport kernel for a contiguous particle slice.

    Parameters
    ----------
    sigma_a, sigma_s : shape (n_groups, n_cells) — Fleck-modified opacities
    dep              : shape (n_groups, n_cells) — accumulates deposited energy density
    si               : shape (n_groups, n_cells) — accumulates scalar intensity
    stats            : shape (6,) — event counters
    boundary_loss_by_group : shape (n_groups,)

    stats layout: [total_events, boundary_crossings, scatters, census,
                   weight_floor_kills, reflections]
    """
    N = len(weights)
    n_cells = mesh.shape[0]

    for i in range(N):
        distance_to_census = (dt - times[i]) * __c
        g = int(groups[i])

        while distance_to_census > 1e-15:
            loc = int(cell_indices[i])
            if loc < 0 or loc >= n_cells:
                break

            stats[0] += 1

            out = _move_particle_spherical(
                weights[i], mus[i], positions[i],
                mesh[loc, 0], mesh[loc, 1],
                sigma_a[g, loc], sigma_s[g, loc],
                distance_to_census
            )

            weights[i]   = out[0]
            mus[i]       = out[1]
            positions[i] = out[2]

            if out[3] == 1:
                cell_indices[i] += 1
            elif out[3] == -1:
                cell_indices[i] -= 1

            dep[g, loc] += out[4]
            si[g, loc]  += out[5] / dt
            distance_to_census = out[6]

            evt = out[7]
            if evt == 1:
                stats[1] += 1
            elif evt == 2:
                stats[2] += 1
            else:
                stats[3] += 1

            # --- Boundary handling ---
            if cell_indices[i] >= n_cells:
                if refl[1]:
                    mus[i] = -mus[i]
                    cell_indices[i] = n_cells - 1
                    stats[5] += 1
                else:
                    boundary_loss_by_group[g] += weights[i]
                    weights[i] = 0.0
                    distance_to_census = 0.0
                    break
            elif cell_indices[i] < 0:
                if refl[0]:
                    mus[i] = -mus[i]
                    cell_indices[i] = 0
                    stats[5] += 1
                else:
                    boundary_loss_by_group[g] += weights[i]
                    weights[i] = 0.0
                    distance_to_census = 0.0
                    break

            # --- Weight floor kill: deposit remaining energy locally ---
            if weights[i] < weight_floor and weights[i] > 0.0:
                stats[4] += 1
                loc2 = int(cell_indices[i])
                if 0 <= loc2 < n_cells:
                    r0 = mesh[loc2, 0]
                    r1 = mesh[loc2, 1]
                    cell_vol = (4.0 / 3.0) * math.pi * (r1**3 - r0**3)
                    dep[g, loc2] += weights[i] / cell_vol
                    sa = sigma_a[g, loc2]
                    if sa > 1e-10:
                        si[g, loc2] += (weights[i] / dt / sa / cell_vol)
                weights[i] = 0.0
                distance_to_census = 0.0


@jit(nopython=True, parallel=True)
def _move_particles_spherical_mg(weights, mus, times, positions, cell_indices, groups,
                                  mesh, sigma_a, sigma_s, n_groups, dt, refl,
                                  weight_floor, stats_accum):
    """Parallel multigroup spherical transport — splits particles into thread chunks.

    Avoids Numba's CFG restriction on prange + nested while by calling the
    serial _move_sph_slice_mg kernel from a simple prange loop.

    Returns
    -------
    dep : (n_groups, n_cells)
    si  : (n_groups, n_cells)
    boundary_loss_by_group : (n_groups,)
    """
    N = len(weights)
    n_cells = mesh.shape[0]
    n_threads = get_num_threads()

    dep_threads  = np.zeros((n_threads, n_groups, n_cells))
    si_threads   = np.zeros((n_threads, n_groups, n_cells))
    stats_threads = np.zeros((n_threads, 6), dtype=np.int64)
    bloss_threads = np.zeros((n_threads, n_groups))

    chunk = (N + n_threads - 1) // n_threads
    for t in prange(n_threads):
        start = t * chunk
        end   = min(start + chunk, N)
        if start < end:
            _move_sph_slice_mg(
                weights[start:end], mus[start:end], times[start:end],
                positions[start:end], cell_indices[start:end], groups[start:end],
                mesh, sigma_a, sigma_s, n_groups, dt, refl, weight_floor,
                dep_threads[t], si_threads[t], stats_threads[t],
                bloss_threads[t]
            )

    stats_sum = stats_threads.sum(axis=0)
    for j in range(len(stats_sum)):
        stats_accum[j] += stats_sum[j]

    return (dep_threads.sum(axis=0),
            si_threads.sum(axis=0),
            bloss_threads.sum(axis=0))


# ===========================================================================
# SECTION 4 — PARTICLE SAMPLING
# ===========================================================================

def _equilibrium_sample_spherical_mg(N, Tr, mesh, energy_edges, T_emit_floor=0.0):
    """Sample N initial-condition particles from equilibrium radiation field.

    The total equilibrium energy  E = a T⁴ V  is split among groups using
    the Planck spectrum (mixture-of-Gammas) evaluated at the local cell
    temperature Tr[i].
    """
    I = mesh.shape[0]
    r0s = mesh[:, 0]
    r1s = mesh[:, 1]
    volumes = (4.0 / 3.0) * np.pi * (r1s**3 - r0s**3)

    energy_per_zone = __a * Tr**4 * volumes
    if T_emit_floor > 0.0:
        energy_per_zone[Tr < T_emit_floor] = 0.0
    total_E = np.sum(energy_per_zone)
    if N <= 0 or total_E <= 0.0:
        empty = np.array([])
        emptyi = np.array([], dtype=np.int64)
        emptyg = np.array([], dtype=np.int32)
        return empty, empty, empty, empty, emptyi, emptyg

    # Number of particles per zone (at least 1 if zone has energy, 0 for cold cells)
    n_per_zone = np.where(energy_per_zone > 0,
                          np.maximum(1, np.round(N * energy_per_zone / total_E).astype(int)),
                          0)
    N_total = int(np.sum(n_per_zone))

    weights      = np.empty(N_total)
    mus          = np.empty(N_total)
    times        = np.zeros(N_total)
    positions    = np.empty(N_total)
    cell_indices = np.empty(N_total, dtype=np.int64)
    groups       = np.empty(N_total, dtype=np.int32)

    offset = 0
    for i in range(I):
        n  = int(n_per_zone[i])
        if n > 0:
            r0 = r0s[i]; r1 = r1s[i]
            xis = np.random.uniform(0.0, 1.0, n)
            r_samples = (r0**3 + (r1**3 - r0**3) * xis) ** (1.0 / 3.0)
            weights[offset:offset+n]      = energy_per_zone[i] / n
            mus[offset:offset+n]          = np.random.uniform(-1.0, 1.0, n)
            positions[offset:offset+n]    = r_samples
            cell_indices[offset:offset+n] = i
            # Sample groups from Planck spectrum at local temperature
            _, grps = _sample_planck_spectrum_mg(n, float(Tr[i]), energy_edges)
            groups[offset:offset+n] = grps
            offset += n

    return (weights[:offset], mus[:offset], times[:offset],
            positions[:offset], cell_indices[:offset], groups[:offset])


def _emitted_particles_spherical_mg(Ntarget, temperature, dt, mesh,
                                     sigma_a_fleck, energy_edges,
                                     T_emit_floor=0.0):
    """Sample emission particles for multigroup spherical IMC.

    Parameters
    ----------
    sigma_a_fleck : ndarray shape (n_groups, n_cells)
        Fleck-modified opacity  f * σ_a,g(T)  for each group.
    energy_edges  : ndarray length n_groups+1

    Returns
    -------
    weights, mus, times, positions, cell_indices, groups, emitted_energies_by_group

    emitted_energies_by_group : shape (n_groups, n_cells) — total energy emitted
        per group per cell (used to subtract from internal energy).
    """
    I = mesh.shape[0]
    n_groups = len(energy_edges) - 1
    r0s = mesh[:, 0]
    r1s = mesh[:, 1]
    volumes = (4.0 / 3.0) * np.pi * (r1s**3 - r0s**3)

    # Planck fractions: B_g / B_total for each cell
    B_g = _compute_Bg_1d(energy_edges, temperature)          # (n_groups, n_cells)
    B_total = np.sum(B_g, axis=0) + 1e-300                  # (n_cells,)
    b_star   = B_g / B_total[np.newaxis, :]                 # (n_groups, n_cells)

    # Emission energy per group per cell:
    #   E_{g,i} = a c T_i^4 * sigma_a_fleck[g,i] * b_star[g,i] * dt * V_i
    T4 = temperature**4
    emitted_by_group = (__a * __c * T4[np.newaxis, :] *
                        sigma_a_fleck * b_star * dt * volumes[np.newaxis, :])  # (n_groups, n_cells)

    if T_emit_floor > 0.0:
        cold_mask = temperature < T_emit_floor
        emitted_by_group[:, cold_mask] = 0.0

    total_emission = np.sum(emitted_by_group)
    if Ntarget <= 0 or total_emission <= 0.0:
        empty  = np.array([])
        emptyi = np.array([], dtype=np.int64)
        emptyg = np.array([], dtype=np.int32)
        return (empty, empty, empty, empty, emptyi, emptyg,
                np.zeros((n_groups, I)))

    # Allocate particles proportionally to emission per (group, cell)
    flat_emit = emitted_by_group.ravel()  # length n_groups * n_cells
    n_flat = np.maximum(1, np.round(Ntarget * flat_emit / total_emission).astype(int))
    n_flat[flat_emit <= 0.0] = 0
    N_total = int(np.sum(n_flat))

    weights      = np.empty(N_total)
    mus          = np.empty(N_total)
    times        = np.empty(N_total)
    positions    = np.empty(N_total)
    cell_indices = np.empty(N_total, dtype=np.int64)
    groups_arr   = np.empty(N_total, dtype=np.int32)

    offset = 0
    for g in range(n_groups):
        for i in range(I):
            n = int(n_flat[g * I + i])
            if n <= 0:
                continue
            E_gi = emitted_by_group[g, i]
            r0 = r0s[i]; r1 = r1s[i]
            # Sample positions uniformly in volume (correct for spherical)
            xis = np.random.uniform(0.0, 1.0, n)
            r_samples = (r0**3 + (r1**3 - r0**3) * xis) ** (1.0 / 3.0)
            weights[offset:offset+n]      = E_gi / n
            mus[offset:offset+n]          = np.random.uniform(-1.0, 1.0, n)
            times[offset:offset+n]        = np.random.uniform(0.0, dt, n)
            positions[offset:offset+n]    = r_samples
            cell_indices[offset:offset+n] = i
            groups_arr[offset:offset+n]   = g
            offset += n

    return (weights[:offset], mus[:offset], times[:offset],
            positions[:offset], cell_indices[:offset], groups_arr[:offset],
            emitted_by_group)


def _create_boundary_spherical_mg(N, T, dt, R, energy_edges, outward=True):
    """Half-Lambertian boundary source on a sphere of radius R with multigroup.

    Total emission = (a c T⁴ / 4) × 4π R² × dt.

    Parameters
    ----------
    outward : True  → μ > 0  (inner boundary, particles going outward)
              False → μ < 0  (outer boundary, particles going inward)
    """
    if N <= 0 or T <= 0.0:
        return None
    total_emission = __a * __c * T**4 / 4.0 * dt * 4.0 * math.pi * R * R
    weights      = np.full(N, total_emission / N)
    raw_mus      = np.sqrt(np.random.uniform(0.0, 1.0, N))
    mus          = raw_mus if outward else -raw_mus
    times        = np.random.uniform(0.0, dt, N)
    positions    = np.full(N, R)
    cell_indices = np.zeros(N, dtype=np.int64)
    # cell_indices are a placeholder — the caller always sets the correct
    # cell (0 for inner boundary, I-1 for outer boundary) before appending.

    _, groups = _sample_planck_spectrum_mg(N, float(T), energy_edges)
    return weights, mus, times, positions, cell_indices, groups


def _sample_source_spherical_mg(N, source, dt, mesh, energy_edges, temperature):
    """Sample fixed-source particles for spherical MG IMC.

    Parameters
    ----------
    source : (n_cells,) power per unit volume, or (n_groups, n_cells)
    """
    I = mesh.shape[0]
    n_groups = len(energy_edges) - 1
    source = np.asarray(source)
    if source.ndim == 1:
        # Gray source: put all in the group determined by local temperature
        source_mg = np.zeros((n_groups, I))
        source_mg[0, :] = source   # lowest group placeholder; will be split below
    elif source.ndim == 2:
        source_mg = source.copy()
    else:
        raise ValueError(f"source shape {source.shape} not supported")

    r0s = mesh[:, 0]; r1s = mesh[:, 1]
    volumes = (4.0 / 3.0) * np.pi * (r1s**3 - r0s**3)

    source_energy = source_mg * volumes[np.newaxis, :] * dt  # (n_groups, n_cells)
    total_source  = np.sum(source_energy)
    if N <= 0 or total_source <= 0.0:
        empty  = np.array([])
        emptyi = np.array([], dtype=np.int64)
        emptyg = np.array([], dtype=np.int32)
        return empty, empty, empty, empty, emptyi, emptyg

    flat_e = source_energy.ravel()
    n_flat = np.maximum(0, np.round(N * flat_e / total_source).astype(int))
    n_flat[flat_e <= 0.0] = 0
    N_total = int(np.sum(n_flat))

    weights      = np.empty(N_total)
    mus          = np.empty(N_total)
    times        = np.empty(N_total)
    positions    = np.empty(N_total)
    cell_indices = np.empty(N_total, dtype=np.int64)
    groups_arr   = np.empty(N_total, dtype=np.int32)

    offset = 0
    for g in range(n_groups):
        for i in range(I):
            n = int(n_flat[g * I + i])
            if n <= 0:
                continue
            r0 = r0s[i]; r1 = r1s[i]
            xis = np.random.uniform(0.0, 1.0, n)
            r_samples = (r0**3 + (r1**3 - r0**3) * xis) ** (1.0 / 3.0)
            weights[offset:offset+n]      = source_energy[g, i] / n
            mus[offset:offset+n]          = np.random.uniform(-1.0, 1.0, n)
            times[offset:offset+n]        = np.random.uniform(0.0, dt, n)
            positions[offset:offset+n]    = r_samples
            cell_indices[offset:offset+n] = i
            groups_arr[offset:offset+n]   = g
            offset += n

    return (weights[:offset], mus[:offset], times[:offset],
            positions[:offset], cell_indices[:offset], groups_arr[:offset])


# ===========================================================================
# SECTION 5 — PER-(CELL, GROUP) STOCHASTIC COMB
# ===========================================================================

def _comb_mg_1d(weights, cell_indices, groups, mus, times, positions,
                Nmax, n_cells, n_groups, conserve_rad_energy=False):
    """Per-(cell, group) stochastic comb for 1-D multigroup IMC.

    Guarantees ≥1 particle per occupied (cell, group) bin.  Stochastically
    merges/splits particles to keep the total near Nmax.

    Parameters
    ----------
    conserve_rad_energy : bool
        When True, rescale each bin's surviving-particle weights so that
        post-comb energy exactly equals pre-comb energy.  The returned
        comb_energy_discrepancy will be zero in this case.

    Returns
    -------
    (weights, cell_indices, groups, mus, times, positions,
     comb_energy_discrepancy)

    comb_energy_discrepancy : shape (n_groups, n_cells)
        Energy removed from the radiation field by combing (pre − post).
        Zero when conserve_rad_energy=True.
    """
    alive = weights > 0.0
    if not np.any(alive):
        empty  = np.array([])
        emptyi = np.array([], dtype=np.int64)
        emptyg = np.array([], dtype=np.int32)
        return (empty, emptyi, emptyg, empty, empty, empty,
                np.zeros((n_groups, n_cells)))

    weights      = weights[alive]
    cell_indices = np.clip(cell_indices[alive], 0, n_cells - 1)
    groups       = groups[alive]
    mus          = mus[alive]
    times        = times[alive]
    positions    = positions[alive]

    # bin_id = cell + group * n_cells
    n_bins = n_cells * n_groups
    bin_id = cell_indices + groups * n_cells

    ecen = np.bincount(bin_id, weights=weights, minlength=n_bins)
    E    = np.sum(ecen)
    if E <= 0.0:
        return (weights, cell_indices, groups, mus, times, positions,
                np.zeros((n_groups, n_cells)))

    desired = np.where(ecen > 0.0,
                       np.maximum(1, np.round(Nmax * ecen / E).astype(int)),
                       0)
    ew = np.zeros(n_bins)
    mask = desired > 0
    ew[mask] = ecen[mask] / desired[mask]

    ew_per_particle = ew[bin_id]
    valid = ew_per_particle > 0.0

    # Stochastic number of output particles per input particle (vectorised)
    numcomb = np.where(
        valid,
        (weights / np.where(valid, ew_per_particle, 1.0)
         + np.random.random(len(weights))).astype(np.int64),
        np.int64(0),
    )

    nw  = np.repeat(ew_per_particle, numcomb)
    nc  = np.repeat(cell_indices,    numcomb).astype(np.int64)
    ng  = np.repeat(groups,          numcomb).astype(np.int32)
    nm  = np.repeat(mus,             numcomb)
    nt  = np.repeat(times,           numcomb)
    npos = np.repeat(positions,      numcomb)

    # Energy discrepancy before – after combing
    rad_before = (np.bincount(bin_id, weights=weights, minlength=n_bins)
                  .reshape(n_groups, n_cells))
    if len(nw) > 0:
        new_bin_id = nc + ng * n_cells
        if conserve_rad_energy:
            # Rescale surviving-particle weights so post-comb energy per bin
            # exactly equals pre-comb energy.  Because desired ≥ 1 for every
            # occupied bin, at least one particle always survives, so
            # post_ecen > 0 whenever ecen > 0.
            post_ecen = np.bincount(new_bin_id, weights=nw, minlength=n_bins)
            scale     = np.where(post_ecen > 0.0,
                                 ecen / (post_ecen + 1e-300),
                                 1.0)
            nw        = nw * scale[new_bin_id]
            rad_after = rad_before          # energy conserved by construction
        else:
            rad_after = (np.bincount(new_bin_id, weights=nw, minlength=n_bins)
                         .reshape(n_groups, n_cells))
    else:
        rad_after  = np.zeros((n_groups, n_cells))

    return (nw, nc, ng,
            nm, nt, npos,
            rad_before - rad_after)


# ===========================================================================
# SECTION 6 — HIGH-LEVEL API
# ===========================================================================

def init_simulation(Ntarget, Tinit, Tr_init, mesh, energy_edges,
                    eos, inv_eos, Ntarget_ic=None, T_emit_floor=0.0):
    """Initialise a multigroup spherical IMC simulation.

    Parameters
    ----------
    Ntarget     : int   — target total number of particles per step
    Tinit       : (I,) — initial material temperature (keV)
    Tr_init     : (I,) — initial radiation temperature (keV)
    mesh        : (I, 2) — [[r_inner, r_outer], ...] cell boundaries (cm)
    energy_edges : (n_groups+1,) — group energy boundaries (keV)
    eos         : callable T → u  (GJ / cm³)
    inv_eos     : callable u → T
    Ntarget_ic  : int or None — number of IC particles (defaults to Ntarget)

    Returns
    -------
    SimulationState1DMG
    """
    I = mesh.shape[0]
    n_groups = len(energy_edges) - 1
    volumes  = (4.0 / 3.0) * np.pi * (mesh[:, 1]**3 - mesh[:, 0]**3)

    internal_energy = eos(Tinit)
    temperature     = Tinit.copy()
    assert np.allclose(inv_eos(internal_energy), Tinit), "Inverse EOS failed"

    N_ic = Ntarget if Ntarget_ic is None else Ntarget_ic
    (weights, mus, times, positions,
     cell_indices, groups) = _equilibrium_sample_spherical_mg(N_ic, Tr_init, mesh, energy_edges,
                                                              T_emit_floor=T_emit_floor)

    # Radiation energy by group
    if len(weights) > 0:
        valid = (cell_indices >= 0) & (cell_indices < I)
        bin_id = cell_indices[valid] + groups[valid].astype(np.int64) * I
        rad_by_group_flat = np.bincount(bin_id, weights=weights[valid],
                                        minlength=n_groups * I)
        radiation_energy_by_group = rad_by_group_flat.reshape(n_groups, I)
    else:
        radiation_energy_by_group = np.zeros((n_groups, I))

    total_rad_energy   = np.sum(radiation_energy_by_group, axis=0)
    radiation_temperature = (total_rad_energy / volumes / __a) ** 0.25

    total_internal = float(np.sum(internal_energy * volumes))
    total_rad      = float(np.sum(weights))
    prev_total     = total_internal + total_rad

    print("Time", "N", "Total Energy", "Total Internal",
          "Total Radiation", "Boundary Emission", "Energy Residual", sep="\t")
    print("=" * 120)
    print(f"{0.0:.6e}", len(weights),
          f"{prev_total:.6e}", f"{total_internal:.6e}", f"{total_rad:.6e}",
          f"{0.0:.6e}", f"{0.0:.6e}", sep="\t")

    return SimulationState1DMG(
        weights=weights, mus=mus, times=times, positions=positions,
        cell_indices=cell_indices, groups=groups,
        internal_energy=internal_energy, temperature=temperature,
        radiation_temperature=radiation_temperature,
        radiation_energy_by_group=radiation_energy_by_group,
        radiation_flux_by_group=np.zeros((n_groups, I)),
        time=0.0, previous_total_energy=prev_total, count=0,
    )


def step(state, Ntarget, Nboundary, Nsource, Nmax,
         T_boundary, dt, mesh, energy_edges,
         sigma_a_funcs, inv_eos, cv, source,
         reflect=(False, False), theta=1.0,
         use_scalar_intensity_Tr=True,
         conserve_comb_energy=True,
         max_events_per_particle=1_000_000,
         T_emit_floor=0.0,
         particle_budget_fmin=0.0,
         Nmax_growth=0,
         Nmax_final=None):
    """Advance the multigroup spherical IMC simulation by one time step.

    Parameters
    ----------
    state         : SimulationState1DMG
    Ntarget       : int  — total particle budget per step when particle_budget_fmin > 0;
                          otherwise emission particles per step only
    Nboundary     : int  — boundary particles per side (ignored when particle_budget_fmin > 0)
    Nsource       : int  — source particles per step (ignored when particle_budget_fmin > 0)
    Nmax          : int  — particle comb target/threshold.
                          > 0 : always comb to Nmax after transport.
                          = 0 : combing disabled (no-comb mode).
                          < 0 : threshold mode — comb only when the census
                                count reaches |Nmax|; target is also |Nmax|.
    T_boundary    : (T_inner, T_outer)  — float or callable(t)
    dt            : float  — time step (ns)
    mesh          : (I, 2) — cell boundaries
    energy_edges  : (n_groups+1,) — group energy edges (keV)
    sigma_a_funcs : list of n_groups callables  sigma_a_g(T) → (I,) array
    inv_eos       : callable u → T
    cv            : callable T → heat capacity (GJ / cm³ / keV)
    source        : (I,) or (n_groups, I) power per unit volume
    reflect       : (reflect_inner, reflect_outer) booleans
    theta         : float — IMC theta parameter (1 = fully implicit)
    use_scalar_intensity_Tr : bool — use path-length tally for Tr
    conserve_comb_energy    : bool or str — how to handle the stochastic comb
                              energy discrepancy:
                              True / ``"material"`` — deposit into matter (default);
                              ``"radiation"``       — redistribute to surviving photons
                                                     (no effect on T_mat);
                              False / ``"none"``    — discard (original behaviour).
    T_emit_floor            : float — suppress emission from cells below this T (keV)
    particle_budget_fmin    : float — if > 0, Ntarget is the total budget shared among all
                              source types; each non-zero source gets at least
                              fmin * Ntarget particles.
    Nmax_growth   : int  — amount to grow Nmax after this step (0 = fixed)
    Nmax_final    : int or None — cap on Nmax growth; None = unlimited

    Returns
    -------
    (state, info)  — info["Nmax_next"] carries the Nmax to use on the next step
    """
    t0 = _time.perf_counter()
    I = mesh.shape[0]
    n_groups = len(energy_edges) - 1
    volumes  = (4.0 / 3.0) * np.pi * (mesh[:, 1]**3 - mesh[:, 0]**3)

    weights      = state.weights.copy()
    mus          = state.mus.copy()
    times        = state.times.copy()
    positions    = state.positions.copy()
    cell_indices = state.cell_indices.copy()
    groups       = state.groups.copy()
    internal_energy = state.internal_energy.copy()
    temperature     = state.temperature.copy()

    # -----------------------------------------------------------------------
    # 1. Compute multigroup opacities and Fleck factors
    # -----------------------------------------------------------------------
    sigma_a_true = np.zeros((n_groups, I))
    for g in range(n_groups):
        sigma_a_true[g, :] = sigma_a_funcs[g](temperature)

    # Planck-weighted opacity for Fleck factor:  σ_P = Σ_g σ_a,g B_g / B_total
    B_g    = _compute_Bg_1d(energy_edges, temperature)     # (n_groups, I)
    B_total = np.sum(B_g, axis=0) + 1e-300                 # (I,)
    sigma_P = np.sum(sigma_a_true * B_g, axis=0) / B_total # (I,)

    beta = 4.0 * __a * temperature**3 / cv(temperature)
    f    = 1.0 / (1.0 + theta * beta * sigma_P * __c * dt)
    f    = np.clip(f, 0.0, 1.0)

    # Fleck-modified opacities
    sigma_a_f = sigma_a_true * f[np.newaxis, :]         # (n_groups, I) effective absorption
    sigma_s_f = sigma_a_true * (1.0 - f)[np.newaxis, :] # (n_groups, I) effective scatter

    # -----------------------------------------------------------------------
    # 1b. Particle budget allocation
    # -----------------------------------------------------------------------
    T_inner = T_boundary[0](state.time) if callable(T_boundary[0]) else T_boundary[0]
    T_outer = T_boundary[1](state.time) if callable(T_boundary[1]) else T_boundary[1]

    if particle_budget_fmin > 0.0:
        # Pre-compute energy from each source type for proportional allocation.
        r_inner_bc = float(mesh[0, 0])
        r_outer_bc = float(mesh[-1, 1])
        E_bc_inner = (__a * __c * T_inner**4 / 4.0 * dt * 4.0 * math.pi * r_inner_bc**2
                      if T_inner > 0.0 else 0.0)
        E_bc_outer = (__a * __c * T_outer**4 / 4.0 * dt * 4.0 * math.pi * r_outer_bc**2
                      if T_outer > 0.0 else 0.0)
        src_arr_b = np.asarray(source)
        if src_arr_b.ndim == 1:
            E_src_b = float(np.sum(np.maximum(src_arr_b, 0.0) * volumes * dt))
        else:
            E_src_b = float(np.sum(np.maximum(src_arr_b, 0.0) * volumes[np.newaxis, :] * dt))
        # Emission energy: Planck-weighted Fleck opacity already computed above
        b_star_b    = B_g / B_total[np.newaxis, :]
        sigma_eff_b = np.sum(sigma_a_f * b_star_b, axis=0)  # (I,)
        emit_e_b    = __a * __c * temperature**4 * sigma_eff_b * dt * volumes
        if T_emit_floor > 0.0:
            emit_e_b[temperature < T_emit_floor] = 0.0
        E_emit_b = float(np.sum(emit_e_b))
        E_all_b  = E_bc_inner + E_bc_outer + E_src_b + E_emit_b
        N_total_b = int(Ntarget)
        N_floor_b = max(1, int(round(particle_budget_fmin * N_total_b)))
        if E_all_b > 0.0:
            _N_inner = (max(int(round(N_total_b * E_bc_inner / E_all_b)), N_floor_b)
                        if E_bc_inner > 0.0 else 0)
            _N_outer = (max(int(round(N_total_b * E_bc_outer / E_all_b)), N_floor_b)
                        if E_bc_outer > 0.0 else 0)
            _N_src   = (max(int(round(N_total_b * E_src_b   / E_all_b)), N_floor_b)
                        if E_src_b   > 0.0 else 0)
            _N_emit  = (max(int(round(N_total_b * E_emit_b  / E_all_b)), N_floor_b)
                        if E_emit_b  > 0.0 else 0)
        else:
            _N_inner = N_total_b   # put everything into inner boundary
            _N_outer = 0
            _N_src   = 0
            _N_emit  = 0
    else:
        _N_inner = int(Nboundary)
        _N_outer = int(Nboundary)
        _N_src   = int(Nsource)
        _N_emit  = int(Ntarget)

    # -----------------------------------------------------------------------
    # 2. Boundary particles
    # -----------------------------------------------------------------------
    boundary_emission = 0.0
    N_boundary        = 0

    if _N_inner > 0 and T_inner > 0.0:
        bs = _create_boundary_spherical_mg(_N_inner, T_inner, dt,
                                           mesh[0, 0], energy_edges, outward=True)
        if bs is not None:
            bw, bm, bt, bp, bc, bg = bs
            bc[:] = 0  # inner boundary → cell 0
            weights      = np.concatenate([weights,      bw])
            mus          = np.concatenate([mus,          bm])
            times        = np.concatenate([times,        bt])
            positions    = np.concatenate([positions,    bp])
            cell_indices = np.concatenate([cell_indices, bc])
            groups       = np.concatenate([groups,       bg])
            boundary_emission += float(np.sum(bw))
            N_boundary        += len(bw)

    if _N_outer > 0 and T_outer > 0.0:
        bs = _create_boundary_spherical_mg(_N_outer, T_outer, dt,
                                           mesh[-1, 1], energy_edges, outward=False)
        if bs is not None:
            bw, bm, bt, bp, bc, bg = bs
            bc[:] = I - 1  # outer boundary → last cell
            weights      = np.concatenate([weights,      bw])
            mus          = np.concatenate([mus,          bm])
            times        = np.concatenate([times,        bt])
            positions    = np.concatenate([positions,    bp])
            cell_indices = np.concatenate([cell_indices, bc])
            groups       = np.concatenate([groups,       bg])
            boundary_emission += float(np.sum(bw))
            N_boundary        += len(bw)

    # -----------------------------------------------------------------------
    # 3. Fixed source
    # -----------------------------------------------------------------------
    source_emission = 0.0
    source_arr = np.asarray(source)
    if _N_src > 0 and np.max(source_arr) > 0.0:
        sp = _sample_source_spherical_mg(_N_src, source_arr, dt, mesh,
                                          energy_edges, temperature)
        if len(sp[0]) > 0:
            sw, sm, st, spos, sc, sg = sp
            weights      = np.concatenate([weights,      sw])
            mus          = np.concatenate([mus,          sm])
            times        = np.concatenate([times,        st])
            positions    = np.concatenate([positions,    spos])
            cell_indices = np.concatenate([cell_indices, sc])
            groups       = np.concatenate([groups,       sg.astype(np.int32)])
            source_emission = float(np.sum(sw))

    # -----------------------------------------------------------------------
    # 4. Material emission
    # -----------------------------------------------------------------------
    (ew, em, et, ep, ec, eg, emitted_by_group) = _emitted_particles_spherical_mg(
        _N_emit, temperature, dt, mesh, sigma_a_f, energy_edges,
        T_emit_floor=T_emit_floor,
    )
    if len(ew) > 0:
        weights      = np.concatenate([weights,      ew])
        mus          = np.concatenate([mus,          em])
        times        = np.concatenate([times,        et])
        positions    = np.concatenate([positions,    ep])
        cell_indices = np.concatenate([cell_indices, ec])
        groups       = np.concatenate([groups,       eg.astype(np.int32)])

    # -----------------------------------------------------------------------
    # 5. Transport
    # -----------------------------------------------------------------------
    t_transport_start = _time.perf_counter()
    n_transported = len(weights)
    weight_floor = 1e-14 * float(np.sum(weights)) / max(n_transported, 1)
    transport_stats = np.zeros(6, dtype=np.int64)

    # Convert mesh to contiguous C array for Numba
    mesh_nb = np.ascontiguousarray(mesh, dtype=np.float64)

    dep, si, boundary_loss_by_group = _move_particles_spherical_mg(
        weights, mus, times, positions, cell_indices, groups,
        mesh_nb,
        np.ascontiguousarray(sigma_a_f, dtype=np.float64),
        np.ascontiguousarray(sigma_s_f, dtype=np.float64),
        n_groups, dt, reflect, weight_floor, transport_stats
    )
    t_post_start = _time.perf_counter()

    boundary_loss = float(np.sum(boundary_loss_by_group))

    # -----------------------------------------------------------------------
    # 6. Material energy update
    # -----------------------------------------------------------------------
    # dep[g, i] has units of energy density (GJ/cm³) deposited in group g, cell i
    total_deposited    = np.sum(dep, axis=0)   # (I,) sum over groups
    total_emitted_dens = np.sum(emitted_by_group, axis=0) / volumes  # (I,) GJ/cm³

    internal_energy = internal_energy + total_deposited - total_emitted_dens
    temperature     = inv_eos(internal_energy)

    # -----------------------------------------------------------------------
    # 7. Radiation temperature from path-length tally
    # -----------------------------------------------------------------------
    if use_scalar_intensity_Tr:
        # si[g, i] = scalar intensity (energy/area/time/sr) from path-length
        # φ_g = 4π ∫ I_g dΩ ≈ 4π * si[g,i] → E_rad,g / V = a Tr^4 summed over g
        total_si = np.sum(si, axis=0)    # (I,)
        radiation_temperature_new = (total_si / __a / __c) ** 0.25
    else:
        # Bin surviving particles
        valid = (cell_indices >= 0) & (cell_indices < I) & (weights > 0.0)
        if np.any(valid):
            rad_flat = np.bincount(
                cell_indices[valid].astype(np.int64),
                weights=weights[valid], minlength=I
            )
            radiation_temperature_new = (rad_flat / volumes / __a) ** 0.25
        else:
            radiation_temperature_new = np.zeros(I)

    # -----------------------------------------------------------------------
    # 8. Remove escaped particles
    # -----------------------------------------------------------------------
    escaped = (cell_indices < 0) | (cell_indices >= I)
    # Particles that escaped were already accounted for in boundary_loss_by_group
    # Zero their weights (transport kernel sets w=0 for escaped, so this is safe)
    keep = ~escaped | (weights <= 0.0)
    keep = np.ones(len(weights), dtype=bool)
    # Actually the transport kernel already zeros weights of escaped particles
    # and accumulates them in boundary_loss_by_group. We just need to remove zeros.
    keep = weights > 0.0
    weights      = weights[keep]
    mus          = mus[keep]
    times        = times[keep]
    positions    = positions[keep]
    cell_indices = cell_indices[keep]
    groups       = groups[keep]

    # -----------------------------------------------------------------------
    # 9. Per-(cell, group) combing
    #    Nmax > 0 : always comb to Nmax.
    #    Nmax = 0 : disabled.
    #    Nmax < 0 : threshold mode — only comb once N >= |Nmax|; target = |Nmax|.
    # -----------------------------------------------------------------------
    comb_disc = np.zeros((n_groups, I))
    _Nmax_target = abs(Nmax) if Nmax < 0 else Nmax
    _do_comb = (Nmax > 0) or (Nmax < 0 and len(weights) >= _Nmax_target)
    if _do_comb:
        _conserve_rad_comb = (conserve_comb_energy == "radiation")
        (weights, cell_indices, groups, mus, times, positions,
         comb_disc) = _comb_mg_1d(
            weights, cell_indices, groups, mus, times, positions,
            _Nmax_target, I, n_groups,
            conserve_rad_energy=_conserve_rad_comb,
        )

        if (conserve_comb_energy is True or conserve_comb_energy == "material") \
                and np.any(comb_disc != 0.0):
            # comb_disc[g, i] is energy removed from radiation → deposit into matter
            internal_energy = internal_energy + np.sum(comb_disc, axis=0) / volumes
            temperature     = inv_eos(internal_energy)

    times = np.zeros_like(times)

    # -----------------------------------------------------------------------
    # 10. Radiation energy and flux by group (post-comb, for diagnostics)
    # -----------------------------------------------------------------------
    if len(weights) > 0:
        valid = (cell_indices >= 0) & (cell_indices < I) & (weights > 0.0)
        if np.any(valid):
            bin_id = cell_indices[valid] + groups[valid].astype(np.int64) * I
            rad_flat  = np.bincount(bin_id, weights=weights[valid],
                                    minlength=n_groups * I)
            radiation_energy_by_group = rad_flat.reshape(n_groups, I)
            # Radial flux tally: F_g = c * Σ(w_k μ_k) / V_i  [GJ / cm² / ns]
            flux_flat = np.bincount(bin_id,
                                    weights=weights[valid] * mus[valid],
                                    minlength=n_groups * I)
            radiation_flux_by_group = (__c * flux_flat.reshape(n_groups, I)
                                       / volumes[np.newaxis, :])
        else:
            radiation_energy_by_group = np.zeros((n_groups, I))
            radiation_flux_by_group   = np.zeros((n_groups, I))
    else:
        radiation_energy_by_group = np.zeros((n_groups, I))
        radiation_flux_by_group   = np.zeros((n_groups, I))

    # -----------------------------------------------------------------------
    # 11. Diagnostics
    # -----------------------------------------------------------------------
    total_internal = float(np.sum(internal_energy * volumes))
    total_rad      = float(np.sum(weights))
    total_energy   = total_internal + total_rad
    dE_system      = total_energy - state.previous_total_energy
    energy_residual = dE_system - boundary_emission + boundary_loss - source_emission

    # Update state
    state.weights      = weights
    state.mus          = mus
    state.times        = times
    state.positions    = positions
    state.cell_indices = cell_indices
    state.groups       = groups
    state.internal_energy       = internal_energy
    state.temperature           = temperature
    state.radiation_temperature     = radiation_temperature_new
    state.radiation_energy_by_group = radiation_energy_by_group
    state.radiation_flux_by_group   = radiation_flux_by_group
    state.time              += dt
    state.previous_total_energy = total_energy
    state.count             += 1

    t_end = _time.perf_counter()
    n_tp = max(n_transported, 1)

    info = {
        "time":                   state.time,
        "temperature":            temperature.copy(),
        "radiation_temperature":  radiation_temperature_new.copy(),
        "radiation_energy_by_group": radiation_energy_by_group.copy(),
        "radiation_flux_by_group":   radiation_flux_by_group.copy(),
        "N_particles":            len(weights),
        "total_energy":           total_energy,
        "total_internal_energy":  total_internal,
        "total_radiation_energy": total_rad,
        "boundary_emission":      boundary_emission,
        "N_boundary":             N_boundary,
        "boundary_loss":          boundary_loss,
        "boundary_loss_by_group": boundary_loss_by_group.copy(),
        "source_emission":        source_emission,
        "energy_residual":        energy_residual,
        "Nmax_next":              (min(Nmax + Nmax_growth, Nmax_final)
                                   if Nmax_final is not None
                                   else Nmax + Nmax_growth)
                                  if Nmax_growth > 0 else Nmax,
        "profiling": {
            "phase_times_s": {
                "sampling":    t_transport_start - t0,
                "transport":   t_post_start - t_transport_start,
                "postprocess": t_end - t_post_start,
                "total":       t_end - t0,
            },
            "transport_events": {
                "total":                   int(transport_stats[0]),
                "boundary_crossings":      int(transport_stats[1]),
                "scatter_events":          int(transport_stats[2]),
                "census_events":           int(transport_stats[3]),
                "weight_floor_kills":      int(transport_stats[4]),
                "reflections":             int(transport_stats[5]),
                "avg_events_per_particle": int(transport_stats[0]) / n_tp,
                "n_particles_transported": int(n_transported),
            },
        },
    }

    return state, info


def run_simulation(Ntarget, Nboundary, Nsource, Nmax,
                   Tinit, Tr_init, T_boundary, dt, mesh, energy_edges,
                   sigma_a_funcs, eos, inv_eos, cv, source, final_time,
                   reflect=(False, False), output_freq=1, theta=1.0,
                   use_scalar_intensity_Tr=True, Ntarget_ic=None,
                   conserve_comb_energy=True,
                   Nmax_growth=0, Nmax_final=None):
    """Run the full multigroup spherical IMC simulation from t=0 to final_time.

    Parameters
    ----------
    (same as step() plus eos, Ntarget_ic, output_freq)

    Returns
    -------
    time_values      : (n_steps+1,) array
    Tr_history       : (n_steps+1, I) array of radiation temperature
    T_history        : (n_steps+1, I) array of material temperature
    state            : final SimulationState1DMG
    history          : list of info dicts
    """
    state = init_simulation(Ntarget, Tinit, Tr_init, mesh, energy_edges,
                             eos, inv_eos, Ntarget_ic=Ntarget_ic)

    time_values = [0.0]
    Tr_history  = [state.radiation_temperature.copy()]
    T_history   = [state.temperature.copy()]
    history     = []

    time_tol = max(1e-15, 1e-12 * max(final_time, 1.0))
    while state.time < final_time - time_tol:
        step_dt = min(dt, final_time - state.time)
        state, info = step(
            state, Ntarget, Nboundary, Nsource, Nmax,
            T_boundary, step_dt, mesh, energy_edges,
            sigma_a_funcs, inv_eos, cv, source, reflect,
            theta=theta,
            use_scalar_intensity_Tr=use_scalar_intensity_Tr,
            conserve_comb_energy=conserve_comb_energy,
            Nmax_growth=Nmax_growth,
            Nmax_final=Nmax_final,
        )
        Nmax = info["Nmax_next"]
        history.append(info)

        if (state.count - 1) % output_freq == 0:
            time_values.append(info["time"])
            Tr_history.append(state.radiation_temperature.copy())
            T_history.append(state.temperature.copy())
            print(f"{info['time']:.6e}", info["N_particles"],
                  f"{info['total_energy']:.6e}",
                  f"{info['total_internal_energy']:.6e}",
                  f"{info['total_radiation_energy']:.6e}",
                  f"{info['boundary_emission']:.6e}",
                  f"{info['energy_residual']:.3e}", sep="\t")

    return (np.array(time_values), np.array(Tr_history), np.array(T_history),
            state, history)


# ===========================================================================
# QUICK SMOKE TEST
# ===========================================================================
if __name__ == "__main__":
    print("MG_IMC1D — multigroup 1-D spherical IMC")
    print(f"Planck integrals available: {_PLANCK_AVAILABLE}")
    print(f"Numba available:            {_NUMBA_AVAILABLE}")

    # Single-cell equilibrium test: T_mat = T_rad = 0.5 keV, should stay constant
    I = 4; R = 1.0
    mesh = np.array([[i * R / I, (i + 1) * R / I] for i in range(I)])
    T0 = np.full(I, 0.5)
    energy_edges = np.array([0.0, 0.5, 2.0, 10.0])   # 3 groups
    sigma0 = 100.0
    sigma_a_funcs = [lambda T, s=sigma0: np.full_like(T, s) for _ in range(3)]
    cv_val = 0.1
    eos     = lambda T: cv_val * T
    inv_eos = lambda u: u / cv_val
    cv_f    = lambda T: np.full_like(T, cv_val)
    source  = np.zeros(I)

    state = init_simulation(1000, T0, T0, mesh, energy_edges, eos, inv_eos, Ntarget_ic=500)
    state, info = step(state, 1000, 0, 0, 5000, (0.0, 0.0), 0.01, mesh, energy_edges,
                       sigma_a_funcs, inv_eos, cv_f, source, reflect=(True, False))
    print("After 1 step:", info["time"], "Tr =", info["radiation_temperature"])
    print("Energy residual:", info["energy_residual"])
