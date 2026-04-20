"""IMC1D.py — Implicit Monte Carlo for 1-D geometry (slab or spherical).

The module is a drop-in replacement for IMCSlab.py with an added ``geometry``
keyword accepted by the three public entry-points (``init_simulation``,
``step``, ``run_simulation``).  The default is ``geometry='slab'``, which
reproduces the original IMCSlab behaviour exactly.

With ``geometry='spherical'`` the mesh is interpreted as radial shells:
``mesh[i] = [r_inner, r_outer]``.  The key differences are:

* **Tracking** (``move_particle_spherical``): straight-line propagation in 3-D
  space expressed in the (r, µ) phase space using the formulae from Box 9.2:

      r' = sqrt(r² + 2 r s µ + s²),   µ' = (r µ + s) / r'

  Distance to each shell boundary uses the impact parameter b = r√(1−µ²):

      s± = −rµ ± √(R² − b²)   (shortest positive root)

* **Volume** measure: ``V_i = (4/3)π(r_outer³ − r_inner³)``.  All deposited
  energies and internal-energy densities are normalised by this volume so that
  ``total_energy = Σ u_i V_i`` is the physical total energy.

* **Equilibrium / emission sampling** uses volume-weighted position sampling:

      r = (r₀³ + (r₁³ − r₀³) ξ)^(1/3),   ξ ~ U([0, 1])

  for the uniform part, plus a linear-in-T⁴ tilt (Box 9.2 "Linear in r
  Sourcing") solved via Newton's method on the quartic CDF.

Notes on Box 9.2
----------------
* The sign of M is printed as (T̂₀⁴ − T̂₁⁴)/Δ in the textbook, but the correct
  definition consistent with T⁴(r) = a + M(r − r₀) is
  M = (T̂₁⁴ − T̂₀⁴)/Δ (subscripts are swapped in the printed formula).
* The equation for the linear sourcing is degree-4 in r (a quartic), not a
  quadratic as stated in the textbook.  We solve it numerically.
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import random
import time
from dataclasses import dataclass
try:
    from numba import jit, prange, get_thread_id, get_num_threads
except Exception:
    # Fallback path for environments where numba/llvmlite are unavailable.
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

__c = 29.98   # cm/ns
__a = 0.01372  # radiation constant

# =============================================================================
# SLAB-GEOMETRY — identical to IMCSlab.py
# =============================================================================

@jit(nopython=True, cache=True)
def move_particle(weight, mu, position, cell_l, cell_r, sigma_a, sigma_s, distance_to_census):
    """Move one slab-geometry particle to the next event (boundary / scatter / census)."""
    dx = cell_r - cell_l
    if mu > 0:
        distance_to_boundary = (cell_r - position) / mu
    else:
        distance_to_boundary = (cell_l - position) / mu
    if sigma_s > 1e-10:
        distance_to_scatter = -math.log(1 - random.random()) / sigma_s
    else:
        distance_to_scatter = 1e30
    if sigma_s * dx > 10000:
        distance_to_scatter = -math.log(1 - random.random()) / (10000 / dx)

    distance_to_next_event = min(distance_to_boundary, distance_to_scatter, distance_to_census)
    assert distance_to_next_event > 0, "Non-positive distance in slab move_particle"
    dx = cell_r - cell_l
    new_location = 0
    position = position + mu * distance_to_next_event

    if math.fabs(distance_to_boundary - distance_to_next_event) < 1e-10:
        if mu > 0:
            assert np.isclose(position, cell_r), "Position not close to right boundary"
            position = cell_r
            new_location = 1
        else:
            assert np.isclose(position, cell_l), "Position not close to left boundary"
            position = cell_l
            new_location = -1
    elif distance_to_next_event == distance_to_scatter:
        mu = random.uniform(-1, 1)
    else:
        True  # census

    if position < cell_l or position > cell_r:
        print("Particle not in cell - after move"); print("position:", position)
        print("cell:", cell_l, cell_r); print("weight:", weight)
    assert position >= cell_l and position <= cell_r, "Particle not in cell"

    distance_to_census = distance_to_census - distance_to_next_event
    weight_factor = math.exp(-sigma_a * distance_to_next_event)
    if sigma_a > 1e-10:
        deposited_intensity = weight * (1 - weight_factor) / sigma_a / dx
    else:
        deposited_intensity = weight * distance_to_next_event / dx
    deposited_weight = deposited_intensity * sigma_a
    weight = weight * weight_factor
    if math.fabs(distance_to_boundary - distance_to_next_event) < 1e-10:
        event_code = 1  # boundary crossing
    elif distance_to_next_event == distance_to_scatter:
        event_code = 2  # scatter
    else:
        event_code = 3  # census
    return (weight, mu, position, new_location, deposited_weight,
            deposited_intensity, distance_to_census, event_code)


@jit(nopython=True, parallel=True)
def move_particles(weights, mus, times, positions, cell_indices, mesh, sigma_a, sigma_s,
                   dt, refl, stats_accum, weight_floor=0.0):
    """Transport all slab particles for one time step using numba thread parallelism."""
    N = len(weights)
    n_cells = len(sigma_a)
    n_threads = get_num_threads()
    dep_threads = np.zeros((n_threads, n_cells))
    si_threads  = np.zeros((n_threads, n_cells))
    # stats columns: [0]=events, [1]=boundary, [2]=scatter,
    # [3]=census, [4]=weight_floor_kill, [5]=reflections
    stats_threads = np.zeros((n_threads, 6), dtype=np.int64)

    for i in prange(N):
        tid = get_thread_id()
        distance_to_census = (dt - times[i]) * __c
        while distance_to_census > 0:
            loc = int(cell_indices[i])
            stats_threads[tid, 0] += 1
            if positions[i] < mesh[loc][0] or positions[i] > mesh[loc][1]:
                print("Particle not in cell"); print("position:", positions[i])
                print("cell:", mesh[loc][0], mesh[loc][1]); print("index:", loc, i)
            output = move_particle(weights[i], mus[i], positions[i],
                                   mesh[loc][0], mesh[loc][1],
                                   sigma_a[loc], sigma_s[loc], distance_to_census)
            weights[i]   = output[0]
            mus[i]       = output[1]
            positions[i] = output[2]
            if output[3] == 1:
                cell_indices[i] += 1
            elif output[3] == -1:
                cell_indices[i] -= 1
            dep_threads[tid, loc] += output[4]
            si_threads[tid, loc]  += output[5] / dt
            distance_to_census = output[6]
            if output[7] == 1:
                stats_threads[tid, 1] += 1
            elif output[7] == 2:
                stats_threads[tid, 2] += 1
            else:
                stats_threads[tid, 3] += 1
            if cell_indices[i] == len(mesh):
                if refl[1]:
                    mus[i] = -mus[i]
                    cell_indices[i] -= 1
                    stats_threads[tid, 5] += 1
                else:
                    distance_to_census = 0.0
            elif cell_indices[i] == -1:
                if refl[0]:
                    mus[i] = -mus[i]
                    cell_indices[i] += 1
                    stats_threads[tid, 5] += 1
                else:
                    distance_to_census = 0.0
            if weights[i] < weight_floor:
                stats_threads[tid, 4] += 1
                loc = int(cell_indices[i])
                if 0 < loc < len(sigma_a) - 1:
                    dep_threads[tid, loc] += weights[i] / (mesh[loc][1] - mesh[loc][0])
                    si_threads[tid, loc]  += weights[i] / dt / sigma_a[loc] / (mesh[loc][1] - mesh[loc][0])
                weights[i] = 0.0
                distance_to_census = 0.0

    stats_sum = stats_threads.sum(axis=0)
    for j in range(len(stats_sum)):
        stats_accum[j] += stats_sum[j]
    return dep_threads.sum(axis=0), si_threads.sum(axis=0)


@jit(nopython=True, cache=True)
def sample_linear_density(dx, s, T, N, adjust_slope=True):
    """Sample N positions ~ s*x + (T − s*dx/2) in [0, dx] (slab linear-T tilt)."""
    if math.fabs(s) < 1e-6:
        return np.random.uniform(0, dx, N)
    s_min = -2 * T / dx**2
    s_max =  2 * T / dx**2
    if s < s_min or s > s_max:
        if adjust_slope:
            s = max(s_min, min(s, s_max))
        else:
            raise ValueError("Invalid slope for sample_linear_density")
    xis = np.random.uniform(0, 1, N)
    assert (s / (2 * T) - 1 / dx)**2 + (2 * s) / (T * dx) >= 0, "Negative sqrt in sample_linear_density"
    return (-2*T + s*dx + 2*T*dx*np.sqrt((s/(2.*T) - 1/dx)**2 + (2*s*xis)/(T*dx))) / (2.*s)


@jit(nopython=True, cache=True)
def equilibrium_sample(N, T, mesh):
    """Sample N equilibrium particles in slab geometry (uniform within each cell)."""
    I = int(mesh.shape[0])
    dx = mesh[:, 1] - mesh[:, 0]
    total_emitted = np.sum(__a * T**4 * dx)
    energy_per_zone = __a * T**4 * dx
    emitted_per_zone = np.ceil(energy_per_zone / total_emitted * N).astype("int")
    N = np.sum(emitted_per_zone)
    weights      = np.empty(N)
    positions    = np.empty(N)
    cell_indices = np.empty(N, dtype=np.int64)
    offset = 0
    for i in range(I):
        n = emitted_per_zone[i]
        weights[offset:offset+n]      = energy_per_zone[i] / n
        positions[offset:offset+n]    = np.random.uniform(mesh[i, 0], mesh[i, 1], n)
        cell_indices[offset:offset+n] = i
        offset += n
    mus   = np.random.uniform(-1, 1, N)
    times = np.zeros(N)
    assert len(weights) == len(mus) == len(times) == len(positions) == len(cell_indices)
    return weights, mus, times, positions, cell_indices


@jit(nopython=True, cache=True)
def emitted_particles(Ntarget, Temperatures, dt, mesh, sigma_a):
    """Sample emission particles for slab geometry (linear-T tilt in x)."""
    dx = mesh[:, 1] - mesh[:, 0]
    slopes = np.zeros(Temperatures.shape)
    slopes[1:-1] = (Temperatures[2:] - Temperatures[:-2]) / (dx[2:] + dx[:-2]) * 2
    slopes[0]    = (Temperatures[1]  - Temperatures[0])   / (dx[1]  + dx[0])   * 2
    slopes[-1]   = (Temperatures[-1] - Temperatures[-2])  / (dx[-1] + dx[-2])  * 2
    left_vals  = Temperatures - slopes * dx / 2
    right_vals = Temperatures + slopes * dx / 2
    slopes[left_vals  < 0] = 0.0
    slopes[right_vals < 0] = 0.0

    assert np.all(Temperatures > 0), "Negative temperature in emitted_particles"
    emitted_energies = __a * __c * Temperatures**4 * sigma_a * dt * dx
    total_emission   = np.sum(emitted_energies)
    Ns = np.empty(len(emitted_energies), dtype=np.int64)
    for i in range(len(emitted_energies)):
        Ns[i] = int(math.ceil(Ntarget * emitted_energies[i] / total_emission))
    N_total      = np.sum(Ns)
    weights      = np.empty(N_total)
    mus          = np.empty(N_total)
    times        = np.empty(N_total)
    positions    = np.empty(N_total)
    cell_indices = np.empty(N_total, dtype=np.int64)
    offset = 0
    for i in range(len(emitted_energies)):
        n = Ns[i]
        weights[offset:offset+n]      = emitted_energies[i] / n
        mus[offset:offset+n]          = np.random.uniform(-1, 1, n)
        times[offset:offset+n]        = np.random.uniform(0, dt, n)
        positions[offset:offset+n]    = sample_linear_density(dx[i], slopes[i], Temperatures[i], n) + mesh[i, 0]
        cell_indices[offset:offset+n] = i
        offset += n
    assert np.allclose(total_emission, np.sum(weights)), "Energy mismatch in emitted_particles"
    return weights, mus, times, positions, cell_indices, emitted_energies


@jit(nopython=True, cache=True)
def sample_source(N, source, mesh, dt):
    """Sample N fixed-source particles in slab geometry."""
    dxs = mesh[:, 1] - mesh[:, 0]
    N_per_zone = np.ceil(N * source * dt * dxs / np.sum(source * dt * dxs)).astype(np.int64)
    N_total = np.sum(N_per_zone)
    weights      = np.empty(N_total)
    mus          = np.empty(N_total)
    times        = np.empty(N_total)
    positions    = np.empty(N_total)
    cell_indices = np.empty(N_total, dtype=np.int64)
    offset = 0
    for i in range(mesh.shape[0]):
        if N_per_zone[i] > 0:
            n = N_per_zone[i]
            weights[offset:offset+n]      = source[i] * dt * dxs[i] / n
            mus[offset:offset+n]          = np.random.uniform(-1, 1, n)
            times[offset:offset+n]        = np.random.uniform(0, 1, n)
            positions[offset:offset+n]    = np.random.uniform(mesh[i, 0], mesh[i, 1], n)
            cell_indices[offset:offset+n] = i
            offset += n
    assert len(weights) == len(mus) == len(times) == len(positions) == len(cell_indices)
    return weights, mus, times, positions, cell_indices


@jit(nopython=True, cache=True)
def create_boundary(N, T, dt):
    """Half-Lambertian boundary source for slab geometry (left boundary, mu > 0)."""
    assert N > 0
    total_emission = __a * __c * T**4 / 4 * dt
    weights      = np.zeros(N) + total_emission / N
    mus          = np.sqrt(np.random.uniform(0, 1, N))
    times        = np.random.uniform(0, dt, N)
    positions    = np.zeros(N) + 1e-8
    cell_indices = np.zeros(N, dtype="int")
    return weights, mus, times, positions, cell_indices


# =============================================================================
# SPHERICAL-GEOMETRY — new functions
# =============================================================================

@jit(nopython=True, cache=True)
def move_particle_spherical(weight, mu, r, r_inner, r_outer, sigma_a, sigma_s, distance_to_census):
    """Move one particle in 1-D spherical geometry (Box 9.2).

    Parameters
    ----------
    r       : current radius (position)
    r_inner : inner radius of current cell
    r_outer : outer radius of current cell
    mu      : direction cosine µ = r̂·Ω̂

    Returns
    -------
    (weight, mu_new, r_new, new_location,
     deposited_weight, deposited_intensity, distance_to_census)

    ``new_location`` is -1 (crossed inner shell), 0 (scatter/census),
    or +1 (crossed outer shell).
    """
    b2 = r * r * (1.0 - mu * mu)       # impact parameter squared

    # --- Distance to outer shell (always real, always positive) ---
    disc_outer = r_outer * r_outer - b2
    dist_outer = -r * mu + math.sqrt(disc_outer)   # s_+ in Box 9.2

    # --- Distance to inner shell (only if mu < 0 and b < r_inner) ---
    disc_inner = r_inner * r_inner - b2
    if mu < 0.0 and disc_inner > 0.0:
        # s_- = -r*mu - sqrt(disc_inner); since mu<0, -r*mu = r|mu| > 0
        dist_inner = -r * mu - math.sqrt(disc_inner)
    else:
        dist_inner = 1e30

    distance_to_boundary = min(dist_outer, dist_inner)
    hit_outer = (dist_outer <= dist_inner)

    # --- Distance to scatter ---
    dr = r_outer - r_inner          # radial cell width (used only for the cap)
    if sigma_s > 1e-10:
        distance_to_scatter = -math.log(1.0 - random.random()) / sigma_s
    else:
        distance_to_scatter = 1e30
    if sigma_s * dr > 10000:
        distance_to_scatter = -math.log(1.0 - random.random()) / (10000.0 / dr)

    distance_to_next_event = min(distance_to_boundary, distance_to_scatter, distance_to_census)
    assert distance_to_next_event > 0.0, "Non-positive distance in spherical move_particle"

    s = distance_to_next_event

    # --- Update position and direction (Box 9.2 streaming) ---
    r_new2 = r * r + 2.0 * r * s * mu + s * s
    r_new = math.sqrt(r_new2) if r_new2 > 0.0 else 0.0
    if r_new > 1e-15:
        mu_new = (r * mu + s) / r_new
    else:
        mu_new = 1.0  # degenerate case at the origin

    # Clamp mu to [-1, 1] for floating-point safety
    mu_new = max(-1.0, min(1.0, mu_new))

    new_location = 0
    if math.fabs(distance_to_boundary - distance_to_next_event) < 1e-10:
        if hit_outer:
            r_new   = r_outer
            mu_new  = (r * mu + s) / r_outer if r_outer > 1e-15 else 1.0
            mu_new  = max(-1.0, min(1.0, mu_new))
            new_location = 1
        else:
            r_new   = r_inner
            mu_new  = (r * mu + s) / r_inner if r_inner > 1e-15 else -1.0
            mu_new  = max(-1.0, min(1.0, mu_new))
            new_location = -1
    elif distance_to_next_event == distance_to_scatter:
        mu_new = random.uniform(-1.0, 1.0)     # isotropic scatter

    if r_new < r_inner or r_new > r_outer:
        print("Particle left cell in spherical move_particle")
        print("r:", r_new, "cell:", r_inner, r_outer)
    assert r_new >= r_inner and r_new <= r_outer, "Particle not in spherical cell"

    # --- Energy deposition (normalised by cell volume) ---
    # V = (4/3) π (r_outer³ − r_inner³)
    cell_vol = (4.0 / 3.0) * math.pi * (r_outer**3 - r_inner**3)
    distance_to_census -= s
    weight_factor = math.exp(-sigma_a * s)
    if sigma_a > 1e-10:
        deposited_intensity = weight * (1.0 - weight_factor) / sigma_a / cell_vol
    else:
        deposited_intensity = weight * s / cell_vol
    deposited_weight = deposited_intensity * sigma_a
    weight *= weight_factor

    if math.fabs(distance_to_boundary - distance_to_next_event) < 1e-10:
        event_code = 1  # boundary crossing
    elif distance_to_next_event == distance_to_scatter:
        event_code = 2  # scatter
    else:
        event_code = 3  # census
    return (weight, mu_new, r_new, new_location, deposited_weight,
            deposited_intensity, distance_to_census, event_code)


@jit(nopython=True, parallel=True)
def move_particles_spherical(weights, mus, times, positions, cell_indices,
                              mesh, sigma_a, sigma_s, dt, refl, stats_accum,
                              weight_floor=0.0):
    """Transport all spherical particles for one time step (numba prange)."""
    N = len(weights)
    n_cells = len(sigma_a)
    n_threads = get_num_threads()
    dep_threads = np.zeros((n_threads, n_cells))
    si_threads  = np.zeros((n_threads, n_cells))
    stats_threads = np.zeros((n_threads, 6), dtype=np.int64)

    for i in prange(N):
        tid = get_thread_id()
        distance_to_census = (dt - times[i]) * __c
        while distance_to_census > 0:
            loc = int(cell_indices[i])
            stats_threads[tid, 0] += 1
            output = move_particle_spherical(
                weights[i], mus[i], positions[i],
                mesh[loc][0], mesh[loc][1],
                sigma_a[loc], sigma_s[loc], distance_to_census)
            weights[i]   = output[0]
            mus[i]       = output[1]
            positions[i] = output[2]
            if output[3] == 1:
                cell_indices[i] += 1
            elif output[3] == -1:
                cell_indices[i] -= 1
            dep_threads[tid, loc] += output[4]
            si_threads[tid, loc]  += output[5] / dt
            distance_to_census = output[6]
            if output[7] == 1:
                stats_threads[tid, 1] += 1
            elif output[7] == 2:
                stats_threads[tid, 2] += 1
            else:
                stats_threads[tid, 3] += 1
            if cell_indices[i] == len(mesh):
                if refl[1]:
                    mus[i] = -mus[i]
                    cell_indices[i] -= 1
                    stats_threads[tid, 5] += 1
                else:
                    distance_to_census = 0.0
            elif cell_indices[i] == -1:
                if refl[0]:
                    mus[i] = -mus[i]
                    cell_indices[i] += 1
                    stats_threads[tid, 5] += 1
                else:
                    distance_to_census = 0.0
            if weights[i] < weight_floor:
                stats_threads[tid, 4] += 1
                loc = int(cell_indices[i])
                if 0 < loc < len(sigma_a) - 1:
                    r0 = mesh[loc][0]; r1 = mesh[loc][1]
                    cell_vol = (4.0 / 3.0) * math.pi * (r1**3 - r0**3)
                    dep_threads[tid, loc] += weights[i] / cell_vol
                    si_threads[tid, loc]  += (weights[i] / dt
                                              / sigma_a[loc] / cell_vol)
                weights[i] = 0.0
                distance_to_census = 0.0

    stats_sum = stats_threads.sum(axis=0)
    for j in range(len(stats_sum)):
        stats_accum[j] += stats_sum[j]
    return dep_threads.sum(axis=0), si_threads.sum(axis=0)


@jit(nopython=True, cache=True)
def equilibrium_sample_spherical(N, T, mesh):
    """Sample N equilibrium particles in spherical geometry.

    Positions drawn uniformly in volume: r = (r₀³ + (r₁³−r₀³)ξ)^(1/3).
    Energy per particle weighted by the cell radiation energy a T⁴ V_i.
    """
    I = int(mesh.shape[0])
    # Cell volumes (proportional to (4/3)π factor cancels in ratios below)
    r0s = mesh[:, 0]
    r1s = mesh[:, 1]
    volumes = (4.0 / 3.0) * math.pi * (r1s**3 - r0s**3)
    energy_per_zone = __a * T**4 * volumes
    total_emitted   = np.sum(energy_per_zone)
    emitted_per_zone = np.ceil(energy_per_zone / total_emitted * N).astype("int")
    N = np.sum(emitted_per_zone)
    weights      = np.empty(N)
    positions    = np.empty(N)
    cell_indices = np.empty(N, dtype=np.int64)
    offset = 0
    for i in range(I):
        n   = emitted_per_zone[i]
        r0  = mesh[i, 0]
        r1  = mesh[i, 1]
        # Uniform-in-volume sampling
        xis = np.random.uniform(0.0, 1.0, n)
        r_samples = (r0**3 + (r1**3 - r0**3) * xis) ** (1.0 / 3.0)
        weights[offset:offset+n]      = energy_per_zone[i] / n
        positions[offset:offset+n]    = r_samples
        cell_indices[offset:offset+n] = i
        offset += n
    mus   = np.random.uniform(-1.0, 1.0, N)
    times = np.zeros(N)
    assert len(weights) == len(mus) == len(times) == len(positions) == len(cell_indices)
    return weights, mus, times, positions, cell_indices


@jit(nopython=True, cache=True)
def _solve_spherical_cdf(r0, r1, a_coeff, M_coeff, xi):
    """Bracketed Newton-bisection to invert the spherical-emission CDF.

    Solves for r in [r0, r1]:
        g(r) = a/3*r³ + M/4*r⁴ - rhs = 0
    where  rhs = xi*C + a/3*r0³ + M/4*r0⁴
    and    C   = a/3*(r1³−r0³) + M/4*(r1⁴−r0⁴).

    Because g(r0) = -xi*C ≤ 0 and g(r1) = (1-xi)*C ≥ 0 the root is always
    bracketed in [r0, r1].  Newton steps are used when they stay inside the
    bracket; otherwise a bisection step is taken so convergence is guaranteed.
    """
    C = (a_coeff / 3.0 * (r1**3 - r0**3)
         + M_coeff / 4.0 * (r1**4 - r0**4))
    if C <= 0.0:
        return r0 + xi * (r1 - r0)          # degenerate cell — uniform fallback

    rhs = xi * C + (a_coeff / 3.0 * r0**3 + M_coeff / 4.0 * r0**4)

    # g(r) = a/3*r³ + M/4*r⁴ - rhs,  g'(r) = r²*(a + M*r)
    lo = r0
    hi = r1
    r  = r0 + xi * (r1 - r0)               # linear initial guess
    for _ in range(60):
        r3 = r * r * r
        r4 = r3 * r
        g  = a_coeff / 3.0 * r3 + M_coeff / 4.0 * r4 - rhs
        if math.fabs(g) < 1e-14 * (math.fabs(rhs) + 1e-30):
            break
        # Update bracket
        if g < 0.0:
            lo = r
        else:
            hi = r
        # Newton step
        dg = r * r * (a_coeff + M_coeff * r)
        if math.fabs(dg) > 1e-30:
            r_new = r - g / dg
        else:
            r_new = 0.5 * (lo + hi)
        # Fall back to bisection if Newton leaves the bracket
        if r_new <= lo or r_new >= hi:
            r_new = 0.5 * (lo + hi)
        if math.fabs(r_new - r) < 1e-14 * (r1 - r0 + 1e-30):
            r = r_new
            break
        r = r_new
    return r


@jit(nopython=True, cache=True)
def emitted_particles_spherical(Ntarget, Temperatures, dt, mesh, sigma_a):
    """Sample emission particles for spherical geometry (linear-T⁴ tilt in r).

    Uses the CDF-inversion formula from Box 9.2 ("Linear in r Sourcing").
    The T⁴ slope is estimated with a centred finite difference across adjacent
    cells (same approach as the slab linear-T tilt, but applied to T⁴ and r).
    """
    r0s = mesh[:, 0]
    r1s = mesh[:, 1]
    r_mid = 0.5 * (r0s + r1s)              # cell centroid radii
    dr    = r1s - r0s
    volumes = (4.0 / 3.0) * math.pi * (r1s**3 - r0s**3)

    # Linear reconstruction of T⁴ in r using centred differences on cell centres
    T4 = Temperatures**4
    I  = len(Temperatures)
    slopes_T4 = np.zeros(I)
    if I > 1:
        slopes_T4[1:-1] = (T4[2:] - T4[:-2]) / (r_mid[2:] - r_mid[:-2])
        slopes_T4[0]    = (T4[1]  - T4[0])   / (r_mid[1]  - r_mid[0])
        slopes_T4[-1]   = (T4[-1] - T4[-2])  / (r_mid[-1] - r_mid[-2])

    # Clamp slopes so that T⁴(r) stays non-negative inside every cell
    # T⁴(r) = a_i + M_i*(r − r0_i) where a_i = T4[i] − M_i*(r_mid[i]−r0_i)
    for i in range(I):
        M = slopes_T4[i]
        a = T4[i] - M * (r_mid[i] - r0s[i])   # T⁴ at inner edge
        b = T4[i] + M * (r1s[i]  - r_mid[i])   # T⁴ at outer edge
        if a < 0.0 or b < 0.0:
            slopes_T4[i] = 0.0

    assert np.all(Temperatures > 0.0), "Negative temperature in emitted_particles_spherical"
    emitted_energies = __a * __c * T4 * sigma_a * dt * volumes
    total_emission   = np.sum(emitted_energies)

    Ns = np.empty(I, dtype=np.int64)
    for i in range(I):
        Ns[i] = int(math.ceil(Ntarget * emitted_energies[i] / total_emission))
    N_total      = np.sum(Ns)
    weights      = np.empty(N_total)
    mus          = np.empty(N_total)
    times        = np.empty(N_total)
    positions    = np.empty(N_total)
    cell_indices = np.empty(N_total, dtype=np.int64)
    offset = 0
    for i in range(I):
        n   = Ns[i]
        r0  = r0s[i]
        r1  = r1s[i]
        M   = slopes_T4[i]
        # a_coeff = T⁴ value at the inner edge of the cell
        a_coeff = T4[i] - M * (r_mid[i] - r0)
        if n > 0:
            if math.fabs(M) < 1e-30 or math.fabs(a_coeff) < 1e-30:
                # Uniform-in-volume fallback
                xis = np.random.uniform(0.0, 1.0, n)
                r_samples = (r0**3 + (r1**3 - r0**3) * xis) ** (1.0 / 3.0)
            else:
                r_samples = np.empty(n)
                for k in range(n):
                    xi = random.random()
                    r_samples[k] = _solve_spherical_cdf(r0, r1, a_coeff, M, xi)
            weights[offset:offset+n]      = emitted_energies[i] / n
            mus[offset:offset+n]          = np.random.uniform(-1.0, 1.0, n)
            times[offset:offset+n]        = np.random.uniform(0.0, dt, n)
            positions[offset:offset+n]    = r_samples
            cell_indices[offset:offset+n] = i
            offset += n

    assert np.allclose(total_emission, np.sum(weights[:offset])), \
        "Energy mismatch in emitted_particles_spherical"
    return (weights[:offset], mus[:offset], times[:offset],
            positions[:offset], cell_indices[:offset], emitted_energies)


@jit(nopython=True, cache=True)
def sample_source_spherical(N, source, mesh, dt):
    """Sample N fixed-source particles in spherical geometry (uniform in volume).

    The source array has units of power per unit volume (same as slab).
    """
    r0s = mesh[:, 0]
    r1s = mesh[:, 1]
    volumes = (4.0 / 3.0) * math.pi * (r1s**3 - r0s**3)
    total_source = np.sum(source * dt * volumes)
    N_per_zone = np.ceil(N * source * dt * volumes / total_source).astype(np.int64)
    N_total = np.sum(N_per_zone)
    weights      = np.empty(N_total)
    mus          = np.empty(N_total)
    times        = np.empty(N_total)
    positions    = np.empty(N_total)
    cell_indices = np.empty(N_total, dtype=np.int64)
    offset = 0
    for i in range(mesh.shape[0]):
        if N_per_zone[i] > 0:
            n  = N_per_zone[i]
            r0 = r0s[i]; r1 = r1s[i]
            xis = np.random.uniform(0.0, 1.0, n)
            r_samples = (r0**3 + (r1**3 - r0**3) * xis) ** (1.0 / 3.0)
            weights[offset:offset+n]      = source[i] * dt * volumes[i] / n
            mus[offset:offset+n]          = np.random.uniform(-1.0, 1.0, n)
            times[offset:offset+n]        = np.random.uniform(0.0, dt, n)
            positions[offset:offset+n]    = r_samples
            cell_indices[offset:offset+n] = i
            offset += n
    assert len(weights) == len(mus) == len(times) == len(positions) == len(cell_indices)
    return weights, mus, times, positions, cell_indices


@jit(nopython=True, cache=True)
def create_boundary_spherical(N, T, dt, R, outward=True):
    """Half-Lambertian boundary source on a sphere of radius R.

    Total emission = (a c T⁴ / 4) × 4π R² × dt  (half-flux × surface area).

    Parameters
    ----------
    outward : True  → µ > 0 (inner boundary, particles emitted outward)
              False → µ < 0 (outer boundary, particles emitted inward)
    """
    assert N > 0
    total_emission = __a * __c * T**4 / 4.0 * dt * 4.0 * math.pi * R * R
    weights = np.zeros(N) + total_emission / N
    raw_mus = np.sqrt(np.random.uniform(0.0, 1.0, N))
    mus     = raw_mus if outward else -raw_mus
    times   = np.random.uniform(0.0, dt, N)
    positions    = np.zeros(N) + R
    cell_indices = np.zeros(N, dtype="int")
    return weights, mus, times, positions, cell_indices


# =============================================================================
# GEOMETRY-INDEPENDENT UTILITIES
# =============================================================================

def comb(weights, cell_indices, mus, times, positions, Ntarget, n_cells):
    """Per-cell stochastic comb (geometry-independent).

    For each particle: numcomb = floor(w / ew_target(cell) + ξ) where
    ew_target(cell) = ecen(cell) / ncen_desired(cell) and ncen_desired is
    proportional to that cell's share of the global Ntarget.
    """
    # Remove zero-weight dead particles before combing (they accumulate when
    # total_ecen == 0 would otherwise return the array unchanged).
    alive = weights > 0.0
    weights      = weights[alive]
    cell_indices = cell_indices[alive]
    mus          = mus[alive]
    times        = times[alive]
    positions    = positions[alive]

    ecen = np.bincount(cell_indices, weights=weights, minlength=n_cells)
    total_ecen = np.sum(ecen)
    if total_ecen == 0:
        return weights, cell_indices, mus, times, positions, np.zeros(n_cells)

    ncen_desired = np.zeros(n_cells, dtype=int)
    active = ecen > 0
    ncen_desired[active] = np.maximum(
        1,
        np.round(Ntarget * ecen[active] / total_ecen).astype(int)
    )
    ew_cen = np.zeros(n_cells)
    positive_counts = ncen_desired > 0
    ew_cen[positive_counts] = ecen[positive_counts] / ncen_desired[positive_counts]

    new_weights       = []
    new_mus           = []
    new_times         = []
    new_positions     = []
    new_cell_indices  = []

    for i in range(len(weights)):
        cell      = int(cell_indices[i])
        ew_target = ew_cen[cell]
        if ew_target <= 0:
            continue
        numcomb = int(weights[i] / ew_target + random.random())
        for _ in range(numcomb):
            new_weights.append(ew_target)
            new_mus.append(mus[i])
            new_times.append(times[i])
            new_positions.append(positions[i])
            new_cell_indices.append(cell)

    new_weights_arr      = np.array(new_weights)
    new_cell_indices_arr = np.array(new_cell_indices, dtype=int)
    if len(new_weights_arr) > 0:
        ecen_after = np.bincount(new_cell_indices_arr, weights=new_weights_arr, minlength=n_cells)
    else:
        ecen_after = np.zeros(n_cells)
    return (new_weights_arr, new_cell_indices_arr,
            np.array(new_mus), np.array(new_times), np.array(new_positions),
            ecen - ecen_after)


# =============================================================================
# SIMULATION STATE AND HIGH-LEVEL API
# =============================================================================

@dataclass
class SimulationState:
    """Mutable state passed between time steps."""
    weights:               np.ndarray
    mus:                   np.ndarray
    times:                 np.ndarray
    positions:             np.ndarray
    cell_indices:          np.ndarray
    internal_energy:       np.ndarray
    temperature:           np.ndarray
    radiation_temperature: np.ndarray
    time:                  float
    previous_total_energy: float
    count:                 int = 0


def _cell_volumes(mesh, geometry):
    """Return per-cell 'volumes': dx for slab, (4/3)π(r₁³−r₀³) for spherical."""
    if geometry == 'slab':
        return mesh[:, 1] - mesh[:, 0]
    else:  # spherical
        return (4.0 / 3.0) * np.pi * (mesh[:, 1]**3 - mesh[:, 0]**3)


def init_simulation(Ntarget, Tinit, Tr_init, mesh, eos, inv_eos,
                    Ntarget_ic=None, geometry='slab'):
    """Initialise particle arrays and material state; return a SimulationState at t=0.

    Parameters
    ----------
    geometry : 'slab' (default) or 'spherical'
    """
    I       = mesh.shape[0]
    volumes = _cell_volumes(mesh, geometry)

    internal_energy = eos(Tinit)
    temperature     = Tinit.copy()
    assert np.allclose(inv_eos(internal_energy), Tinit), "Inverse EOS failed"

    N_ic = Ntarget_ic if Ntarget_ic is not None else Ntarget
    if geometry == 'slab':
        p = equilibrium_sample(N_ic, Tr_init, mesh)
    else:
        p = equilibrium_sample_spherical(N_ic, Tr_init, mesh)

    weights      = p[0]; mus = p[1]; times = p[2]
    positions    = p[3]; cell_indices = p[4]

    radiation_temperature = (
        np.bincount(cell_indices, weights=weights, minlength=I) / volumes / __a
    ) ** 0.25

    total_internal_energy  = np.sum(internal_energy * volumes)
    total_radiation_energy = np.sum(weights)
    previous_total_energy  = total_internal_energy + total_radiation_energy

    print("Time", "N", "Total Energy", "Total Internal Energy",
          "Total Radiation Energy", "Boundary Emission", "Lost Energy", sep='\t')
    print("=" * 111)
    print("{:.6f}".format(0.0), len(weights),
          "{:.6f}".format(previous_total_energy),
          "{:.6f}".format(total_internal_energy),
          "{:.6f}".format(total_radiation_energy),
          "{:.6f}".format(0.0), "{:.6f}".format(0.0), sep='\t')

    return SimulationState(
        weights=weights, mus=mus, times=times, positions=positions,
        cell_indices=cell_indices, internal_energy=internal_energy,
        temperature=temperature, radiation_temperature=radiation_temperature,
        time=0.0, previous_total_energy=previous_total_energy, count=0,
    )


def step(state, Ntarget, Nboundary, Nsource, NMax, T_boundary, dt, mesh,
         sigma_a_func, inv_eos, cv, source, reflect=(False, False), theta=1.0,
         use_scalar_intensity_Tr=True, conserve_comb_energy=False,
         geometry='slab'):
    """Advance the simulation by one time step dt.

    Parameters
    ----------
    geometry : 'slab' (default) or 'spherical'

    All other parameters identical to IMCSlab.py ``step()``.
    """
    t_step_start = time.perf_counter()
    I       = mesh.shape[0]
    volumes = _cell_volumes(mesh, geometry)

    weights         = state.weights
    mus             = state.mus
    times           = state.times
    positions       = state.positions
    cell_indices    = state.cell_indices
    internal_energy = state.internal_energy
    temperature     = state.temperature

    # --- Fleck-factor cross sections ---
    sigma_a_true = sigma_a_func(temperature)
    beta = 4.0 * __a * temperature**3 / cv(temperature)
    f    = 1.0 / (1.0 + theta * beta * sigma_a_true * __c * dt)
    assert np.all(f >= 0) and np.all(f <= 1), "Fleck factor out of bounds"
    sigma_s = sigma_a_true * (1.0 - f)
    sigma_a = sigma_a_true * f

    t_sampling_start = time.perf_counter()

    # --- Boundary sources ---
    T_left  = T_boundary[0](state.time) if callable(T_boundary[0]) else T_boundary[0]
    T_right = T_boundary[1](state.time) if callable(T_boundary[1]) else T_boundary[1]
    boundary_emission = 0.0

    if T_left > 0 and Nboundary > 0:
        if geometry == 'slab':
            bs = create_boundary(Nboundary, T_left, dt)
        else:
            bs = create_boundary_spherical(Nboundary, T_left, dt,
                                           mesh[0, 0], outward=True)
        weights      = np.concatenate((weights,      bs[0]))
        mus          = np.concatenate((mus,           bs[1]))
        times        = np.concatenate((times,         bs[2]))
        positions    = np.concatenate((positions,     bs[3]))
        cell_indices = np.concatenate((cell_indices,  bs[4]))
        boundary_emission += np.sum(bs[0])

    if T_right > 0 and Nboundary > 0:
        if geometry == 'slab':
            bs = create_boundary(Nboundary, T_right, dt)
            bs_pos = np.full(len(bs[0]), mesh[-1, 1])
            bs_idx = np.full(len(bs[0]), I - 1, dtype=int)
        else:
            bs = create_boundary_spherical(Nboundary, T_right, dt,
                                           mesh[-1, 1], outward=False)
            bs_pos = np.full(len(bs[0]), mesh[-1, 1])
            bs_idx = np.full(len(bs[0]), I - 1, dtype=int)
        weights      = np.concatenate((weights,      bs[0]))
        mus          = np.concatenate((mus,          -bs[1] if geometry == 'slab' else bs[1]))
        times        = np.concatenate((times,         bs[2]))
        positions    = np.concatenate((positions,     bs_pos))
        cell_indices = np.concatenate((cell_indices,  bs_idx))
        boundary_emission += np.sum(bs[0])

    # --- Fixed source ---
    source_emission = 0.0
    if np.max(source) > 0 and Nsource > 0:
        if geometry == 'slab':
            sp = sample_source(Nsource, source, mesh, dt)
        else:
            sp = sample_source_spherical(Nsource, source, mesh, dt)
        weights      = np.concatenate((weights,      sp[0]))
        mus          = np.concatenate((mus,           sp[1]))
        times        = np.concatenate((times,         sp[2]))
        positions    = np.concatenate((positions,     sp[3]))
        cell_indices = np.concatenate((cell_indices,  sp[4]))
        source_emission = np.sum(sp[0])

    # --- Internal emission ---
    if geometry == 'slab':
        internal_source = emitted_particles(Ntarget, temperature, dt, mesh, sigma_a)
    else:
        internal_source = emitted_particles_spherical(Ntarget, temperature, dt, mesh, sigma_a)
    weights      = np.concatenate((weights,      internal_source[0]))
    mus          = np.concatenate((mus,           internal_source[1]))
    times        = np.concatenate((times,         internal_source[2]))
    positions    = np.concatenate((positions,     internal_source[3]))
    cell_indices = np.concatenate((cell_indices,  internal_source[4]))

    t_transport_start = time.perf_counter()

    # --- Transport ---
    weight_floor = 1e-10 * np.sum(weights) / max(len(weights), 1)
    transport_stats = np.zeros(6, dtype=np.int64)
    n_particles_transported = len(weights)
    if geometry == 'slab':
        deposited, scalar_intensity = move_particles(
            weights, mus, times, positions, cell_indices,
            mesh, sigma_a, sigma_s, dt, reflect, transport_stats, weight_floor)
    else:
        deposited, scalar_intensity = move_particles_spherical(
            weights, mus, times, positions, cell_indices,
            mesh, sigma_a, sigma_s, dt, reflect, transport_stats, weight_floor)

    t_post_start = time.perf_counter()

    # --- Update material state ---
    emitted_energies = internal_source[5]
    internal_energy  = internal_energy + deposited - emitted_energies / volumes
    temperature      = inv_eos(internal_energy)

    # --- Radiation temperature ---
    new_time = state.time + dt
    if use_scalar_intensity_Tr:
        radiation_temperature = (scalar_intensity / __a / __c) ** 0.25
    else:
        valid = (cell_indices >= 0) & (cell_indices < I)
        radiation_temperature = (
            np.bincount(cell_indices[valid], weights=weights[valid], minlength=I)
            / volumes / __a
        ) ** 0.25

    # --- Remove escaped particles ---
    boundary_loss = 0.0
    mask = cell_indices < 0
    if reflect[0]:
        mus[mask] = -mus[mask]
        cell_indices[mask] = 0
    else:
        boundary_loss = np.sum(weights[mask])
        keep = ~mask
        weights = weights[keep]; mus = mus[keep]; times = times[keep]
        positions = positions[keep]; cell_indices = cell_indices[keep]

    mask = cell_indices >= I
    if reflect[1]:
        mus[mask] = -mus[mask]
        cell_indices[mask] = I - 1
    else:
        boundary_loss += np.sum(weights[mask])
        keep = ~mask
        weights = weights[keep]; mus = mus[keep]; times = times[keep]
        positions = positions[keep]; cell_indices = cell_indices[keep]

    # --- Particle combing ---
    weights, cell_indices, mus, times, positions, comb_discrepancy = comb(
        weights, cell_indices, mus, times, positions, NMax, I)

    if conserve_comb_energy:
        internal_energy = internal_energy + comb_discrepancy / volumes
        temperature     = inv_eos(internal_energy)

    times = np.zeros(times.shape)

    # --- Energy diagnostics ---
    total_internal_energy  = np.sum(internal_energy * volumes)
    total_radiation_energy = np.sum(weights)
    total_energy = total_internal_energy + total_radiation_energy
    energy_loss  = (total_energy - state.previous_total_energy
                    - boundary_emission + boundary_loss - source_emission)

    # --- Update state in-place ---
    state.weights               = weights
    state.mus                   = mus
    state.times                 = times
    state.positions             = positions
    state.cell_indices          = cell_indices
    state.internal_energy       = internal_energy
    state.temperature           = temperature
    state.radiation_temperature = radiation_temperature
    state.time                  = new_time
    state.previous_total_energy = total_energy
    state.count                += 1

    t_step_end = time.perf_counter()
    events_total = int(transport_stats[0])
    avg_events_per_particle = events_total / max(n_particles_transported, 1)

    info = {
        'time':                   new_time,
        'radiation_temperature':  radiation_temperature,
        'temperature':            temperature,
        'N_particles':            len(weights),
        'total_energy':           total_energy,
        'total_internal_energy':  total_internal_energy,
        'total_radiation_energy': total_radiation_energy,
        'boundary_emission':      boundary_emission,
        'boundary_loss':          boundary_loss,
        'source_emission':        source_emission,
        'energy_loss':            energy_loss,
        'profiling': {
            'phase_times_s': {
                'sampling': t_transport_start - t_sampling_start,
                'transport': t_post_start - t_transport_start,
                'postprocess': t_step_end - t_post_start,
                'total': t_step_end - t_step_start,
            },
            'transport_events': {
                'total': events_total,
                'boundary_crossings': int(transport_stats[1]),
                'scatter_events': int(transport_stats[2]),
                'census_events': int(transport_stats[3]),
                'weight_floor_kills': int(transport_stats[4]),
                'reflections': int(transport_stats[5]),
                'avg_events_per_particle': avg_events_per_particle,
                'n_particles_transported': int(n_particles_transported),
            },
        },
    }
    return state, info


def run_simulation(Ntarget, Nboundary, Nsource, NMax, Tinit, Tr_init,
                   T_boundary, dt, mesh, sigma_a_func, eos, inv_eos, cv,
                   source, final_time, reflect=(False, False), output_freq=1,
                   theta=1.0, use_scalar_intensity_Tr=True, Ntarget_ic=None,
                   conserve_comb_energy=False, geometry='slab'):
    """Run the full simulation from t=0 to ``final_time``.

    Parameters
    ----------
    geometry : 'slab' (default) or 'spherical'

    Returns
    -------
    (time_values, radiation_temperatures, temperatures)  — numpy arrays
    """
    state = init_simulation(Ntarget, Tinit, Tr_init, mesh, eos, inv_eos,
                            Ntarget_ic=Ntarget_ic, geometry=geometry)

    radiation_temperatures = [state.radiation_temperature.copy()]
    temperatures           = [state.temperature.copy()]
    time_values            = [0.0]

    while state.time < final_time:
        step_dt = min(dt, final_time - state.time)
        state, info = step(state, Ntarget, Nboundary, Nsource, NMax,
                           T_boundary, step_dt, mesh, sigma_a_func, inv_eos,
                           cv, source, reflect, theta=theta,
                           use_scalar_intensity_Tr=use_scalar_intensity_Tr,
                           conserve_comb_energy=conserve_comb_energy,
                           geometry=geometry)

        if (state.count - 1) % output_freq == 0 or (info['time'] - final_time) < step_dt:
            radiation_temperatures.append(state.radiation_temperature.copy())
            temperatures.append(state.temperature.copy())
            time_values.append(info['time'])
            print("{:.6f}".format(info['time']), info['N_particles'],
                  "{:.6f}".format(info['total_energy']),
                  "{:.6f}".format(info['total_internal_energy']),
                  "{:.6f}".format(info['total_radiation_energy']),
                  "{:.6f}".format(info['boundary_emission']),
                  "{:.6e}".format(info['energy_loss']), sep='\t')

    return np.array(time_values), np.array(radiation_temperatures), np.array(temperatures)


# =============================================================================
# QUICK SMOKE TESTS (run as __main__)
# =============================================================================

if __name__ == "__main__":
    # ---- Slab: equilibrium relaxation (same as IMCSlab.py __main__) ----
    Ntarget = 20000; Nboundary = 0; Nsource = 0; NMax = 100000
    dt = 0.01; L = 0.1; I = 2
    mesh_slab = np.array([[i * L / I, (i + 1) * L / I] for i in range(I)])
    Tinit   = np.zeros(I) + 0.5
    Trinit  = np.zeros(I) + 0.5
    T_boundary = (0.0, 0.0)
    sigma_a_f = lambda T: 1e2 + 0 * T
    source   = np.zeros(I)
    cv_val   = 0.01
    eos      = lambda T: cv_val * T
    inv_eos  = lambda u: u / cv_val
    cv_f     = lambda T: cv_val
    final_time = dt * 5

    print("\n=== Slab geometry ===")
    times_s, Tr_s, T_s = run_simulation(
        Ntarget, Nboundary, Nsource, NMax, Tinit, Trinit, T_boundary,
        dt, mesh_slab, sigma_a_f, eos, inv_eos, cv_f, source,
        final_time, reflect=(True, True), geometry='slab')

    # ---- Spherical: uniform sphere, equilibrium relaxation ----
    R = 0.1; I_sph = 4
    mesh_sph = np.array([[i * R / I_sph, (i + 1) * R / I_sph]
                          for i in range(I_sph)])
    Tinit_sph  = np.zeros(I_sph) + 0.5
    Trinit_sph = np.zeros(I_sph) + 0.5
    source_sph = np.zeros(I_sph)

    print("\n=== Spherical geometry ===")
    times_sp, Tr_sp, T_sp = run_simulation(
        Ntarget, Nboundary, Nsource, NMax, Tinit_sph, Trinit_sph, T_boundary,
        dt, mesh_sph, sigma_a_f, eos, inv_eos, cv_f, source_sph,
        final_time, reflect=(True, True), geometry='spherical')

    print("\nSlab final T[0]:", T_s[-1, 0])
    print("Spherical final T[0]:", T_sp[-1, 0])
    print("Both should be near 0.5 (equilibrium).")
