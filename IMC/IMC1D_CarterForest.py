"""IMC1D_CarterForest.py — Carter-Forest (time-delayed re-emission) IMC for 1-D geometry.

Unlike Fleck-Cummings IMC (IMC1D.py), which uses effective scattering to handle
the absorption-re-emission process implicitly, Carter-Forest IMC models physical
absorption followed by time-delayed re-emission.

Key differences from Fleck-Cummings:
-------------------------------------
1. No Fleck factor (f=1): Use true absorption cross section σ_a, no effective scattering
2. True absorption: Particles are actually absorbed (not pseudo-scattered)
3. Time-delayed re-emission: Sample re-emission time from exponential distribution:
   
   t_emit = t_absorbed - (1/(c·σ_a·β)) · ln(ξ),   ξ ~ U([0,1])
   
   where β = 4aT³/C_v is held fixed during the time step
   
4. Conditional re-emission:
   - If t_emit < t_n+1 (within time step): particle is re-emitted at t_emit
   - If t_emit >= t_n+1 (beyond time step): energy deposited, no re-emission
   
5. Track-length estimators: Still used for radiation temperature estimation

This method is exact in time for the linearized equations (no time discretization
error), but retains the linearization error from freezing material properties.

Reference: Box 9.3 "Carter-Forest Monte Carlo (time-delayed re-emission)"
"""

import math
import random
import numpy as np
import time
from dataclasses import dataclass

try:
    from numba import jit, prange, get_thread_id, get_num_threads
except Exception:
    def jit(*jit_args, **jit_kwargs):
        def decorator(func):
            return func
        if len(jit_args) == 1 and callable(jit_args[0]):
            return jit_args[0]
        return decorator

    def prange(*args):
        return range(*args)

    def get_thread_id():
        return 0

    def get_num_threads():
        return 1


__c = 29.98   # cm/ns
__a = 0.01372  # radiation constant


# =============================================================================
# SLAB-GEOMETRY CARTER-FOREST TRANSPORT
# =============================================================================

@jit(nopython=True, cache=True)
def move_particle_cf(weight, mu, position, time, cell_l, cell_r, sigma_a, 
                     beta, distance_to_census, fastpath_threshold=0.0):
    """
    Move one particle in Carter-Forest mode (true absorption + time-delayed re-emission).
    
    Returns:
    --------
    weight : float
        Particle weight (0 if absorbed with no re-emission within dt)
    mu : float
        New direction cosine
    position : float
        New position
    time : float
        Particle time (updated if re-emission occurs)
    new_location : int
        -1 (left boundary), 0 (in cell), 1 (right boundary)
    deposited_weight : float
        Energy deposited in this cell
    deposited_intensity : float
        Track-length contribution for radiation temperature
    distance_to_census : float
        Remaining distance to census
    re_emission_flag : int
        0 if absorbed (no re-emission), 1 if continuing (scatter or re-emission)
    """
    dx = cell_r - cell_l
    
    # Check if already at census
    if distance_to_census < 1e-14:
        return (weight, mu, position, time, 0, 0.0, 0.0, 0.0, 3, 1)
    
    # Distance to boundaries
    if mu > 0:
        distance_to_boundary = max(0.0, (cell_r - position) / mu)
    else:
        distance_to_boundary = max(0.0, (cell_l - position) / mu)
    
    # Distance to absorption (no scattering in Carter-Forest)
    if sigma_a > 1e-10:
        distance_to_absorption = -math.log(1 - random.random()) / sigma_a
    else:
        distance_to_absorption = 1e30
    
    # Prevent overflow in optically thick cells
    if sigma_a * dx > 10000:
        distance_to_absorption = -math.log(1 - random.random()) / (10000 / dx)
    
    # Find next event
    distance_to_next_event = min(distance_to_boundary, distance_to_absorption, distance_to_census)
    
    # Handle very small distances
    if distance_to_next_event <= 0:
        if distance_to_next_event < -1e-10:
            print(f"Warning: Negative distance {distance_to_next_event}")
        return (weight, mu, position, time, 0, 0.0, 0.0, 0.0, 3, 1)
    
    # Move particle
    position = position + mu * distance_to_next_event
    
    # Track-length estimator (always computed, even if particle is absorbed)
    deposited_intensity = weight * distance_to_next_event / dx
    
    new_location = 0
    deposited_weight = 0.0
    re_emission_flag = 1  # Assume continuing
    event_code = 3  # default census
    
    # Handle event type
    if math.fabs(distance_to_boundary - distance_to_next_event) < 1e-10:
        event_code = 1
        # Boundary crossing
        if mu > 0:
            position = cell_r
            new_location = 1
        else:
            position = cell_l
            new_location = -1
        
        # Update distance to census
        distance_to_census = distance_to_census - distance_to_next_event
    elif math.fabs(distance_to_absorption - distance_to_next_event) < 1e-10:
        event_code = 2
        # ABSORPTION EVENT (Carter-Forest specific)
        # Sample re-emission time: t_emit = t_absorbed + delay
        if beta > 1e-15 and sigma_a > 1e-15:
            rate = __c * sigma_a * beta
            time_remaining = (distance_to_census - distance_to_next_event) / __c

            # Optional hybrid shortcut for very stiff absorption/re-emission:
            # sample net same-step survival vs capture without micro-cycling.
            if fastpath_threshold > 0.0 and time_remaining > 1e-14 and rate * time_remaining > fastpath_threshold:
                p_capture = math.exp(-rate * time_remaining)
                if random.random() < p_capture:
                    deposited_weight = weight
                    weight = 0.0
                    event_code = 4
                    re_emission_flag = 0
                else:
                    mu = random.uniform(-1, 1)
                    distance_to_census = distance_to_census - distance_to_next_event
            else:
                xi = random.random()
                if xi < 1e-15:
                    xi = 1e-15  # Avoid log(0)

                # Time when absorption occurs
                t_absorbed = time + distance_to_next_event / __c

                # Sample re-emission delay from exponential distribution
                re_emission_delay = -math.log(xi) / rate
                t_emit = t_absorbed + re_emission_delay

                # Time remaining in this time step AFTER absorption
                # Check if re-emission occurs within remaining time
                if time_remaining > 1e-14 and re_emission_delay < time_remaining:
                    # RE-EMISSION within time step
                    # Particle continues with isotropic direction at t_emit
                    mu = random.uniform(-1, 1)
                    time = t_emit
                    distance_to_census = (time_remaining - re_emission_delay) * __c
                    # No energy deposited (absorbed then re-emitted)
                    deposited_weight = 0.0
                else:
                    # NO RE-EMISSION within time step - energy stays in material
                    deposited_weight = weight
                    weight = 0.0
                    event_code = 4
                    re_emission_flag = 0
        else:
            # No absorption if beta or sigma_a negligible
            mu = random.uniform(-1, 1)
            # Still consume traveled distance in this step
            distance_to_census = distance_to_census - distance_to_next_event
    else:
        # Census
        distance_to_census = 0.0
    
    # Ensure particle is in cell
    assert position >= cell_l - 1e-10 and position <= cell_r + 1e-10, "Particle not in cell"
    
    
    return (weight, mu, position, time, new_location, deposited_weight, 
            deposited_intensity, distance_to_census, event_code, re_emission_flag)


@jit(nopython=True, parallel=True)
def move_particles_cf(weights, mus, times, positions, cell_indices, mesh, 
                      sigma_a, beta, dt, refl, stats_accum,
                      weight_floor=0.0, max_events_per_particle=0,
                      fastpath_threshold=0.0):
    """
    Transport all particles for one time step using Carter-Forest method.
    
    Parameters:
    -----------
    sigma_a : ndarray
        True absorption cross section (no Fleck factor)
    beta : ndarray
        Material coupling parameter β = 4aT³/C_v (held fixed during step)
    """
    N = len(weights)
    n_cells = len(sigma_a)
    n_threads = get_num_threads()
    dep_threads = np.zeros((n_threads, n_cells))  # Deposited energy
    si_threads  = np.zeros((n_threads, n_cells))  # Scalar intensity (track-length)
    # stats columns: [0]=events, [1]=boundary, [2]=absorb+reemit/continue,
    # [3]=census, [4]=absorb+capture, [5]=weight_floor_kill,
    # [6]=reflections, [7]=event_cap_hits
    stats_threads = np.zeros((n_threads, 8), dtype=np.int64)
    
    for i in prange(N):
        tid = get_thread_id()
        distance_to_census = (dt - times[i]) * __c
        local_events = 0
        
        while distance_to_census > 0 and weights[i] > weight_floor:
            loc = int(cell_indices[i])
            stats_threads[tid, 0] += 1
            local_events += 1

            if max_events_per_particle > 0 and local_events >= max_events_per_particle:
                stats_threads[tid, 7] += 1
                distance_to_census = 0.0
                break
            
            # Boundary reflections
            if loc < 0:
                if refl[0]:
                    cell_indices[i] = 0
                    loc = 0
                    mus[i] = -mus[i]  # Flip direction on reflection
                else:
                    break
            elif loc >= n_cells:
                if refl[1]:
                    cell_indices[i] = n_cells - 1
                    loc = n_cells - 1
                    mus[i] = -mus[i]  # Flip direction on reflection
                else:
                    break
            
            # Move particle
            output = move_particle_cf(
                weights[i], mus[i], positions[i], times[i],
                mesh[loc][0], mesh[loc][1],
                sigma_a[loc], beta[loc], distance_to_census,
                fastpath_threshold
            )
            
            weights[i]   = output[0]
            mus[i]       = output[1]
            positions[i] = output[2]
            times[i]     = output[3]
            
            # Update cell index if boundary crossed
            if output[4] == 1:
                cell_indices[i] += 1
            elif output[4] == -1:
                cell_indices[i] -= 1
            
            # Accumulate deposited energy and scalar intensity
            dep_threads[tid, loc] += output[5]  # deposited_weight
            si_threads[tid, loc]  += output[6] / dt  # deposited_intensity normalized by dt
            
            distance_to_census = output[7]
            if output[8] == 1:
                stats_threads[tid, 1] += 1
            elif output[8] == 2:
                stats_threads[tid, 2] += 1
            elif output[8] == 4:
                stats_threads[tid, 4] += 1
            else:
                stats_threads[tid, 3] += 1
            
            # If particle was absorbed (no re-emission), exit loop
            if output[9] == 0:
                break

            if cell_indices[i] == n_cells:
                if refl[1]:
                    mus[i] = -mus[i]
                    cell_indices[i] -= 1
                    stats_threads[tid, 6] += 1
                else:
                    break
            elif cell_indices[i] == -1:
                if refl[0]:
                    mus[i] = -mus[i]
                    cell_indices[i] += 1
                    stats_threads[tid, 6] += 1
                else:
                    break
        if weights[i] > 0.0 and weights[i] <= weight_floor:
            stats_threads[tid, 5] += 1
            weights[i] = 0.0
    
    # Sum thread contributions
    deposited = np.sum(dep_threads, axis=0)
    scalar_intensity = np.sum(si_threads, axis=0)
    stats_sum = np.sum(stats_threads, axis=0)
    for j in range(len(stats_sum)):
        stats_accum[j] += stats_sum[j]
    
    return deposited, scalar_intensity


# =============================================================================
# EMISSION SAMPLING (unchanged from IMC1D.py)
# =============================================================================

@jit(nopython=True, cache=True)
def sample_linear_density(dx, s, T, N, adjust_slope=True):
    """
    Samples N points from a probability density function P(x) ~ s*x + (a - s*dx/2),
    ensuring that density is maximum at f(dx/2) = a and follows slope s.

    Parameters:
        dx (float): Size of the cell (range [0, dx]).
        s (float): Desired slope of the density function.
        T (float): Temperature at cell center.
        N (int): Number of samples to generate.
        adjust_slope (bool): Whether to automatically adjust invalid slopes.

    Returns:
        np.array: Array of sampled positions (relative to cell start).
    """
    # Compute valid slope limits
    if math.fabs(s) < 1e-6:
        return np.random.uniform(0, dx, N)
    s_min = -2 * T / dx**2
    s_max = 2 * T / dx**2

    # Validate or adjust the slope
    if s < s_min or s > s_max:
        if adjust_slope:
            s = max(s_min, min(s, s_max))
        else:
            raise ValueError(f"Slope {s} is outside valid range [{s_min}, {s_max}]")

    # Compute the points
    xis = np.random.uniform(0, 1, N)
    # assert sqrt is positive
    sqrt_term = (s/(2.*T) - 1/dx)**2 + (2*s*xis)/(T*dx)
    assert np.all(sqrt_term >= 0), f"{T} {dx} {s} gives negative square root"
    x_samples = (-2*T + s*dx + 2*T*dx*np.sqrt(sqrt_term))/(2.*s)
    
    return x_samples


@jit(nopython=True, cache=True)
def sample_linear_density_T4(dx, s, T, N, adjust_slope=True):
    """Sample positions from linear temperature^4 profile (Box 9.2).
    
    This version samples from T^4 profile between two temperatures.
    Used in equilibrium_sample.
    """
    a = T[0]**4
    b = T[1]**4
    M = (b - a) / dx if adjust_slope else (b - a)
    
    positions = np.empty(N)
    for i in range(N):
        xi = random.random()
        if abs(M) < 1e-10:
            positions[i] = s + xi * dx
        else:
            positions[i] = s + (math.sqrt(a**2 + 2*M*dx*xi*a + M**2*dx**2*xi**2) - a) / M
    
    return positions


@jit(nopython=True, cache=True)
def equilibrium_sample(N, T, mesh):
    """Sample N particles from temperature distribution (slab geometry)."""
    I = len(T)
    T4 = T**4
    emissions = __a * T4  # Radiation energy density (no factor of c for initial condition)
    total_emission = np.sum(emissions * (mesh[:, 1] - mesh[:, 0]))
    
    weights = np.full(N, total_emission / N)
    mus = np.empty(N)
    times = np.zeros(N)
    positions = np.empty(N)
    cell_indices = np.empty(N, dtype=np.int64)
    
    for n in range(N):
        xi = random.random()
        cumsum = 0.0
        for i in range(I):
            dx = mesh[i, 1] - mesh[i, 0]
            cumsum += emissions[i] * dx / total_emission
            if xi <= cumsum:
                cell_indices[n] = i
                break
        
        i = int(cell_indices[n])
        if i >= I - 1:
            i = I - 1
            cell_indices[n] = i
        
        dx = mesh[i, 1] - mesh[i, 0]
        s = mesh[i, 0]
        
        if i == 0:
            T_pair = np.array([T[0], T[1]])
        elif i == I - 1:
            T_pair = np.array([T[I-2], T[I-1]])
        else:
            T_pair = np.array([T[i], T[i+1]])
        
        pos_arr = sample_linear_density_T4(dx, s, T_pair, 1, adjust_slope=True)
        positions[n] = pos_arr[0]
        mus[n] = random.uniform(-1, 1)
    
    return weights, mus, times, positions, cell_indices


@jit(nopython=True, cache=True)
def emitted_particles(Ntarget, Temperatures, dt, mesh, sigma_a, beta):
    """
    Sample particles from thermal emission with exponential time distribution.
    
    In Carter-Forest IMC, emission is time-delayed with exponential weighting:
    Source(t) = (σ_a * a * c * T^4 / 4π) * exp(-c * σ_a * β * t)
    
    Total emission over dt: a * T^4 * [1 - exp(-c * σ_a * β * dt)] / β
    
    Emission times are sampled from the exponential distribution to properly
    account for the time-delayed nature of re-emission.
    
    Parameters:
    - beta: Material coupling parameter 4*a*T^3/C_v (array per cell)
    """
    I = len(Temperatures)
    T4 = Temperatures**4
    volumes = mesh[:, 1] - mesh[:, 0]
    
    # Compute slopes for spatial temperature reconstruction
    slopes = np.zeros(Temperatures.shape)
    dx = mesh[:, 1] - mesh[:, 0]
    
    slopes[1:-1] = (Temperatures[2:] - Temperatures[:-2]) / (dx[2:] + dx[:-2]) * 2
    slopes[0] = (Temperatures[1] - Temperatures[0]) / (dx[1] + dx[0]) * 2
    slopes[-1] = (Temperatures[-1] - Temperatures[-2]) / (dx[-1] + dx[-2]) * 2
    
    # Check that interpolants are non-negative at cell boundaries
    # T(x) = s*x + (T - s*dx/2)
    left_vals = slopes * 0 + (Temperatures - slopes * dx / 2)
    right_vals = slopes * dx + (Temperatures - slopes * dx / 2)
    mask = left_vals < 0
    slopes[mask] = 0
    mask = right_vals < 0
    slopes[mask] = 0
    
    # Assert that all temperatures are positive
    assert np.all(Temperatures > 0), "Negative temperature"
    
    # Calculate emission with exponential time weighting
    emissions = np.zeros(I)
    for i in range(I):
        rate = __c * sigma_a[i] * beta[i]
        if rate > 1e-10:
            # Emission = a * T^4 * [1 - exp(-rate * dt)] / beta
            emissions[i] = __a * T4[i] * (1.0 - math.exp(-rate * dt)) / beta[i]
        else:
            # For small rate, use Taylor expansion to avoid numerical issues
            # [1 - exp(-x)] / x ≈ 1 - x/2 + x^2/6 for small x
            # But [1 - exp(-rate*dt)]/beta = c*sigma_a*dt*[1 - exp(-rate*dt)]/(rate*dt)
            # ≈ c*sigma_a*dt for small rate*dt
            emissions[i] = __a * T4[i] * __c * sigma_a[i] * dt
    
    total_emission = np.sum(emissions * volumes)
    
    if total_emission < 1e-30:
        empty_float = np.empty(0, dtype=np.float64)
        empty_int = np.empty(0, dtype=np.int64)
        return (empty_float, empty_float, empty_float, empty_float, 
                empty_int, np.zeros(I))
    
    cell_emissions = emissions * volumes / total_emission
    N_target_per_cell = np.maximum(1, np.round(Ntarget * cell_emissions).astype(np.int64))
    N = np.sum(N_target_per_cell)
    
    weights = np.empty(N)
    mus = np.empty(N)
    times = np.empty(N)
    positions = np.empty(N)
    cell_indices = np.empty(N, dtype=np.int64)
    emitted_energies = np.zeros(I)
    
    idx = 0
    for i in range(I):
        n_cell = N_target_per_cell[i]
        if n_cell == 0:
            continue
        
        e_cell = emissions[i] * volumes[i]
        w_cell = e_cell / n_cell
        
        # Calculate rate for exponential time sampling
        rate = __c * sigma_a[i] * beta[i]
        
        for j in range(n_cell):
            weights[idx] = w_cell
            mus[idx] = random.uniform(-1, 1)
            
            # Sample emission time from exponential distribution
            # CDF: F(t) = [1 - exp(-rate*t)] / [1 - exp(-rate*dt)]
            # Inverse: t = -ln(1 - xi * [1 - exp(-rate*dt)]) / rate
            if rate > 1e-10:
                xi = random.random()
                times[idx] = -math.log(1.0 - xi * (1.0 - math.exp(-rate * dt))) / rate
            else:
                # For small rate, fall back to uniform (limit case)
                times[idx] = random.random() * dt
            
            # Sample position using slope reconstruction (like standard IMC)
            pos_arr = sample_linear_density(dx[i], slopes[i], Temperatures[i], 1, adjust_slope=True)
            positions[idx] = pos_arr[0] + mesh[i, 0]
            cell_indices[idx] = i
            idx += 1
        
        emitted_energies[i] = e_cell
    assert (np.isclose(np.sum(emitted_energies), total_emission) or np.isclose(total_emission, 0.0)), "Emitted energy does not match total emission"
    return weights, mus, times, positions, cell_indices, emitted_energies


@jit(nopython=True, cache=True)
def sample_source(N, source, mesh, dt):
    """Sample particles from fixed source."""
    I = len(source)
    volumes = mesh[:, 1] - mesh[:, 0]
    total_source = np.sum(source * volumes * dt)
    
    if total_source < 1e-30:
        empty_float = np.empty(0, dtype=np.float64)
        empty_int = np.empty(0, dtype=np.int64)
        return empty_float, empty_float, empty_float, empty_float, empty_int
    
    weights = np.full(N, total_source / N)
    mus = np.empty(N)
    times = np.empty(N)
    positions = np.empty(N)
    cell_indices = np.empty(N, dtype=np.int64)
    
    cell_probs = source * volumes / np.sum(source * volumes)
    
    for n in range(N):
        xi = random.random()
        cumsum = 0.0
        for i in range(I):
            cumsum += cell_probs[i]
            if xi <= cumsum:
                cell_indices[n] = i
                break
        
        i = int(cell_indices[n])
        times[n] = random.random() * dt
        positions[n] = mesh[i, 0] + random.random() * volumes[i]
        mus[n] = random.uniform(-1, 1)
    
    return weights, mus, times, positions, cell_indices


@jit(nopython=True, cache=True)
def create_boundary(N, T, dt):
    """Create boundary source particles (Lambertian)."""
    total_emission = 0.25 * __a * __c * T**4 * dt  # Factor of 1/4 for Lambert boundary
    weights = np.full(N, total_emission / N)
    
    # Lambertian distribution: sample from sqrt(uniform) for |mu|
    mus = np.empty(N)
    for i in range(N):
        mus[i] = math.sqrt(random.random())
    
    times = np.empty(N)
    for i in range(N):
        times[i] = random.random() * dt
    
    positions = np.zeros(N)
    cell_indices = np.zeros(N, dtype=np.int64)
    
    return weights, mus, times, positions, cell_indices


# =============================================================================
# SPHERICAL-GEOMETRY CARTER-FOREST TRANSPORT
# =============================================================================

@jit(nopython=True, cache=True)
def move_particle_spherical_cf(weight, mu, r, time, r_inner, r_outer, sigma_a,
                               beta, distance_to_census, fastpath_threshold=0.0):
    """Move one particle in 1-D spherical geometry for Carter-Forest IMC."""
    if distance_to_census < 1e-14:
        return (weight, mu, r, time, 0, 0.0, 0.0, 0.0, 3, 1)

    b2 = r * r * (1.0 - mu * mu)
    disc_outer = r_outer * r_outer - b2
    dist_outer = -r * mu + math.sqrt(disc_outer)

    disc_inner = r_inner * r_inner - b2
    if mu < 0.0 and disc_inner > 0.0:
        dist_inner = -r * mu - math.sqrt(disc_inner)
    else:
        dist_inner = 1e30

    distance_to_boundary = min(dist_outer, dist_inner)
    hit_outer = dist_outer <= dist_inner

    dr = r_outer - r_inner
    if sigma_a > 1e-10:
        distance_to_absorption = -math.log(1.0 - random.random()) / sigma_a
    else:
        distance_to_absorption = 1e30
    if sigma_a * dr > 10000:
        distance_to_absorption = -math.log(1.0 - random.random()) / (10000.0 / dr)

    distance_to_next_event = min(distance_to_boundary, distance_to_absorption, distance_to_census)
    if distance_to_next_event <= 0.0:
        return (weight, mu, r, time, 0, 0.0, 0.0, 0.0, 3, 1)

    s = distance_to_next_event
    r_new2 = r * r + 2.0 * r * s * mu + s * s
    r_new = math.sqrt(r_new2) if r_new2 > 0.0 else 0.0
    if r_new > 1e-15:
        mu_new = (r * mu + s) / r_new
    else:
        mu_new = 1.0
    mu_new = max(-1.0, min(1.0, mu_new))

    cell_vol = (4.0 / 3.0) * math.pi * (r_outer**3 - r_inner**3)
    deposited_intensity = weight * s / cell_vol

    new_location = 0
    deposited_weight = 0.0
    re_emission_flag = 1
    event_code = 3

    if math.fabs(distance_to_boundary - distance_to_next_event) < 1e-10:
        event_code = 1
        if hit_outer:
            r_new = r_outer
            mu_new = (r * mu + s) / r_outer if r_outer > 1e-15 else 1.0
            mu_new = max(-1.0, min(1.0, mu_new))
            new_location = 1
        else:
            r_new = r_inner
            mu_new = (r * mu + s) / r_inner if r_inner > 1e-15 else -1.0
            mu_new = max(-1.0, min(1.0, mu_new))
            new_location = -1
        distance_to_census -= s
    elif math.fabs(distance_to_absorption - distance_to_next_event) < 1e-10:
        event_code = 2
        if beta > 1e-15 and sigma_a > 1e-15:
            rate = __c * sigma_a * beta
            time_remaining = (distance_to_census - s) / __c

            if fastpath_threshold > 0.0 and time_remaining > 1e-14 and rate * time_remaining > fastpath_threshold:
                p_capture = math.exp(-rate * time_remaining)
                if random.random() < p_capture:
                    deposited_weight = weight
                    weight = 0.0
                    event_code = 4
                    re_emission_flag = 0
                else:
                    mu_new = random.uniform(-1.0, 1.0)
                    distance_to_census = distance_to_census - s
            else:
                xi = random.random()
                if xi < 1e-15:
                    xi = 1e-15
                t_absorbed = time + s / __c
                re_emission_delay = -math.log(xi) / rate
                if time_remaining > 1e-14 and re_emission_delay < time_remaining:
                    mu_new = random.uniform(-1.0, 1.0)
                    time = t_absorbed + re_emission_delay
                    distance_to_census = (time_remaining - re_emission_delay) * __c
                else:
                    deposited_weight = weight
                    weight = 0.0
                    event_code = 4
                    re_emission_flag = 0
        else:
            mu_new = random.uniform(-1.0, 1.0)
            # Still consume traveled distance in this step
            distance_to_census = distance_to_census - s
    else:
        distance_to_census = 0.0

    if r_new < r_inner or r_new > r_outer:
        print("Particle left cell in spherical CF move")
        print("r:", r_new, "cell:", r_inner, r_outer)
    assert r_new >= r_inner - 1e-10 and r_new <= r_outer + 1e-10, "Particle not in spherical cell"

    return (weight, mu_new, r_new, time, new_location, deposited_weight,
            deposited_intensity, distance_to_census, event_code, re_emission_flag)


@jit(nopython=True, parallel=True)
def move_particles_spherical_cf(weights, mus, times, positions, cell_indices, mesh,
                                sigma_a, beta, dt, refl, stats_accum,
                                weight_floor=0.0, max_events_per_particle=0,
                                fastpath_threshold=0.0):
    """Transport all spherical particles for one Carter-Forest time step."""
    N = len(weights)
    n_cells = len(sigma_a)
    n_threads = get_num_threads()
    dep_threads = np.zeros((n_threads, n_cells))
    si_threads = np.zeros((n_threads, n_cells))
    stats_threads = np.zeros((n_threads, 8), dtype=np.int64)

    for i in prange(N):
        tid = get_thread_id()
        distance_to_census = (dt - times[i]) * __c
        local_events = 0

        while distance_to_census > 0 and weights[i] > weight_floor:
            loc = int(cell_indices[i])
            stats_threads[tid, 0] += 1
            local_events += 1

            if max_events_per_particle > 0 and local_events >= max_events_per_particle:
                stats_threads[tid, 7] += 1
                distance_to_census = 0.0
                break

            if loc < 0:
                if refl[0]:
                    cell_indices[i] = 0
                    loc = 0
                    mus[i] = -mus[i]
                else:
                    break
            elif loc >= n_cells:
                if refl[1]:
                    cell_indices[i] = n_cells - 1
                    loc = n_cells - 1
                    mus[i] = -mus[i]
                else:
                    break

            output = move_particle_spherical_cf(
                weights[i], mus[i], positions[i], times[i],
                mesh[loc][0], mesh[loc][1], sigma_a[loc], beta[loc], distance_to_census,
                fastpath_threshold
            )

            weights[i] = output[0]
            mus[i] = output[1]
            positions[i] = output[2]
            times[i] = output[3]

            if output[4] == 1:
                cell_indices[i] += 1
            elif output[4] == -1:
                cell_indices[i] -= 1

            dep_threads[tid, loc] += output[5]
            si_threads[tid, loc] += output[6] / dt
            distance_to_census = output[7]

            if output[8] == 1:
                stats_threads[tid, 1] += 1
            elif output[8] == 2:
                stats_threads[tid, 2] += 1
            elif output[8] == 4:
                stats_threads[tid, 4] += 1
            else:
                stats_threads[tid, 3] += 1

            if output[9] == 0:
                break

            if cell_indices[i] == n_cells:
                if refl[1]:
                    mus[i] = -mus[i]
                    cell_indices[i] -= 1
                    stats_threads[tid, 6] += 1
                else:
                    break
            elif cell_indices[i] == -1:
                if refl[0]:
                    mus[i] = -mus[i]
                    cell_indices[i] += 1
                    stats_threads[tid, 6] += 1
                else:
                    break
        if weights[i] > 0.0 and weights[i] <= weight_floor:
            stats_threads[tid, 5] += 1
            weights[i] = 0.0

    stats_sum = np.sum(stats_threads, axis=0)
    for j in range(len(stats_sum)):
        stats_accum[j] += stats_sum[j]
    return np.sum(dep_threads, axis=0), np.sum(si_threads, axis=0)


@jit(nopython=True, cache=True)
def equilibrium_sample_spherical(N, T, mesh):
    """Sample N equilibrium particles in spherical geometry."""
    I = int(mesh.shape[0])
    r0s = mesh[:, 0]
    r1s = mesh[:, 1]
    volumes = (4.0 / 3.0) * math.pi * (r1s**3 - r0s**3)
    energy_per_zone = __a * T**4 * volumes
    total_emitted = np.sum(energy_per_zone)
    emitted_per_zone = np.ceil(energy_per_zone / total_emitted * N).astype(np.int64)
    N = np.sum(emitted_per_zone)
    weights = np.empty(N)
    positions = np.empty(N)
    cell_indices = np.empty(N, dtype=np.int64)
    offset = 0
    for i in range(I):
        n = emitted_per_zone[i]
        r0 = mesh[i, 0]
        r1 = mesh[i, 1]
        xis = np.random.uniform(0.0, 1.0, n)
        r_samples = (r0**3 + (r1**3 - r0**3) * xis) ** (1.0 / 3.0)
        weights[offset:offset+n] = energy_per_zone[i] / n
        positions[offset:offset+n] = r_samples
        cell_indices[offset:offset+n] = i
        offset += n
    mus = np.random.uniform(-1.0, 1.0, N)
    times = np.zeros(N)
    assert len(weights) == len(mus) == len(times) == len(positions) == len(cell_indices)
    return weights, mus, times, positions, cell_indices


@jit(nopython=True, cache=True)
def _solve_spherical_cdf(r0, r1, a_coeff, M_coeff, xi):
    """Invert the quartic spherical emission CDF with bracketed Newton-bisection."""
    C = (a_coeff / 3.0 * (r1**3 - r0**3)
         + M_coeff / 4.0 * (r1**4 - r0**4))
    if C <= 0.0:
        return r0 + xi * (r1 - r0)

    rhs = xi * C + (a_coeff / 3.0 * r0**3 + M_coeff / 4.0 * r0**4)
    lo = r0
    hi = r1
    r = r0 + xi * (r1 - r0)
    for _ in range(60):
        r3 = r * r * r
        r4 = r3 * r
        g = a_coeff / 3.0 * r3 + M_coeff / 4.0 * r4 - rhs
        if math.fabs(g) < 1e-14 * (math.fabs(rhs) + 1e-30):
            break
        if g < 0.0:
            lo = r
        else:
            hi = r
        dg = r * r * (a_coeff + M_coeff * r)
        if math.fabs(dg) > 1e-30:
            r_new = r - g / dg
        else:
            r_new = 0.5 * (lo + hi)
        if r_new <= lo or r_new >= hi:
            r_new = 0.5 * (lo + hi)
        if math.fabs(r_new - r) < 1e-14 * (r1 - r0 + 1e-30):
            r = r_new
            break
        r = r_new
    return r


@jit(nopython=True, cache=True)
def emitted_particles_spherical(Ntarget, Temperatures, dt, mesh, sigma_a, beta):
    """Sample Carter-Forest emission particles in spherical geometry."""
    r0s = mesh[:, 0]
    r1s = mesh[:, 1]
    r_mid = 0.5 * (r0s + r1s)
    volumes = (4.0 / 3.0) * math.pi * (r1s**3 - r0s**3)

    T4 = Temperatures**4
    I = len(Temperatures)
    slopes_T4 = np.zeros(I)
    if I > 1:
        slopes_T4[1:-1] = (T4[2:] - T4[:-2]) / (r_mid[2:] - r_mid[:-2])
        slopes_T4[0] = (T4[1] - T4[0]) / (r_mid[1] - r_mid[0])
        slopes_T4[-1] = (T4[-1] - T4[-2]) / (r_mid[-1] - r_mid[-2])

    for i in range(I):
        M = slopes_T4[i]
        a_edge = T4[i] - M * (r_mid[i] - r0s[i])
        b_edge = T4[i] + M * (r1s[i] - r_mid[i])
        if a_edge < 0.0 or b_edge < 0.0:
            slopes_T4[i] = 0.0

    emissions = np.zeros(I)
    for i in range(I):
        rate = __c * sigma_a[i] * beta[i]
        if rate > 1e-10:
            emissions[i] = __a * T4[i] * (1.0 - math.exp(-rate * dt)) / beta[i]
        else:
            emissions[i] = __a * T4[i] * __c * sigma_a[i] * dt

    emitted_energies = emissions * volumes
    total_emission = np.sum(emitted_energies)
    if total_emission < 1e-30:
        empty_float = np.empty(0, dtype=np.float64)
        empty_int = np.empty(0, dtype=np.int64)
        return empty_float, empty_float, empty_float, empty_float, empty_int, np.zeros(I)

    Ns = np.maximum(1, np.round(Ntarget * emitted_energies / total_emission).astype(np.int64))
    N_total = np.sum(Ns)
    weights = np.empty(N_total)
    mus = np.empty(N_total)
    times = np.empty(N_total)
    positions = np.empty(N_total)
    cell_indices = np.empty(N_total, dtype=np.int64)

    offset = 0
    for i in range(I):
        n = Ns[i]
        r0 = r0s[i]
        r1 = r1s[i]
        M = slopes_T4[i]
        a_coeff = T4[i] - M * (r_mid[i] - r0)
        rate = __c * sigma_a[i] * beta[i]
        if n > 0:
            if math.fabs(M) < 1e-30 or math.fabs(a_coeff) < 1e-30:
                xis = np.random.uniform(0.0, 1.0, n)
                r_samples = (r0**3 + (r1**3 - r0**3) * xis) ** (1.0 / 3.0)
            else:
                r_samples = np.empty(n)
                for k in range(n):
                    xi = random.random()
                    r_samples[k] = _solve_spherical_cdf(r0, r1, a_coeff, M, xi)

            weights[offset:offset+n] = emitted_energies[i] / n
            mus[offset:offset+n] = np.random.uniform(-1.0, 1.0, n)
            if rate > 1e-10:
                xis_t = np.random.uniform(0.0, 1.0, n)
                times[offset:offset+n] = -np.log(1.0 - xis_t * (1.0 - math.exp(-rate * dt))) / rate
            else:
                times[offset:offset+n] = np.random.uniform(0.0, dt, n)
            positions[offset:offset+n] = r_samples
            cell_indices[offset:offset+n] = i
            offset += n

    return (weights[:offset], mus[:offset], times[:offset],
            positions[:offset], cell_indices[:offset], emitted_energies)


@jit(nopython=True, cache=True)
def sample_source_spherical(N, source, mesh, dt):
    """Sample fixed-source particles in spherical geometry."""
    r0s = mesh[:, 0]
    r1s = mesh[:, 1]
    volumes = (4.0 / 3.0) * math.pi * (r1s**3 - r0s**3)
    total_source = np.sum(source * dt * volumes)
    if total_source < 1e-30:
        empty_float = np.empty(0, dtype=np.float64)
        empty_int = np.empty(0, dtype=np.int64)
        return empty_float, empty_float, empty_float, empty_float, empty_int

    N_per_zone = np.ceil(N * source * dt * volumes / total_source).astype(np.int64)
    N_total = np.sum(N_per_zone)
    weights = np.empty(N_total)
    mus = np.empty(N_total)
    times = np.empty(N_total)
    positions = np.empty(N_total)
    cell_indices = np.empty(N_total, dtype=np.int64)
    offset = 0
    for i in range(mesh.shape[0]):
        if N_per_zone[i] > 0:
            n = N_per_zone[i]
            r0 = r0s[i]
            r1 = r1s[i]
            xis = np.random.uniform(0.0, 1.0, n)
            r_samples = (r0**3 + (r1**3 - r0**3) * xis) ** (1.0 / 3.0)
            weights[offset:offset+n] = source[i] * dt * volumes[i] / n
            mus[offset:offset+n] = np.random.uniform(-1.0, 1.0, n)
            times[offset:offset+n] = np.random.uniform(0.0, dt, n)
            positions[offset:offset+n] = r_samples
            cell_indices[offset:offset+n] = i
            offset += n
    return weights, mus, times, positions, cell_indices


@jit(nopython=True, cache=True)
def create_boundary_spherical(N, T, dt, R, outward=True):
    """Create half-Lambertian boundary particles on a spherical surface."""
    total_emission = __a * __c * T**4 / 4.0 * dt * 4.0 * math.pi * R * R
    weights = np.full(N, total_emission / N)
    raw_mus = np.sqrt(np.random.uniform(0.0, 1.0, N))
    mus = raw_mus if outward else -raw_mus
    times = np.random.uniform(0.0, dt, N)
    positions = np.zeros(N) + R
    cell_indices = np.zeros(N, dtype=np.int64)
    return weights, mus, times, positions, cell_indices


def _cell_volumes(mesh, geometry):
    """Return per-cell dx for slab or shell volume for spherical geometry."""
    if geometry == 'slab':
        return mesh[:, 1] - mesh[:, 0]
    return (4.0 / 3.0) * np.pi * (mesh[:, 1]**3 - mesh[:, 0]**3)


# =============================================================================
# UTILITIES
# =============================================================================

def comb(weights, cell_indices, mus, times, positions, Ntarget, n_cells):
    """Per-cell stochastic comb (geometry-independent)."""
    alive = weights > 0.0
    weights      = weights[alive]
    cell_indices = cell_indices[alive]
    mus          = mus[alive]
    times        = times[alive]
    positions    = positions[alive]
    
    ecen = np.bincount(cell_indices, weights=weights, minlength=n_cells)
    total_ecen = np.sum(ecen)
    if total_ecen == 0:
        return (weights, cell_indices, mus, times, positions, 
                np.zeros(n_cells))
    
    ncen_desired = np.zeros(n_cells, dtype=int)
    active = ecen > 0
    ncen_desired[active] = np.maximum(
        1,
        np.round(Ntarget * ecen[active] / total_ecen).astype(int)
    )
    ew_cen = np.zeros(n_cells)
    positive_counts = ncen_desired > 0
    ew_cen[positive_counts] = ecen[positive_counts] / ncen_desired[positive_counts]
    
    new_weights = []
    new_mus = []
    new_times = []
    new_positions = []
    new_cell_indices = []
    
    for i in range(len(weights)):
        cell = cell_indices[i]
        if ew_cen[cell] > 0:
            numcomb = int(weights[i] / ew_cen[cell] + random.random())
            for _ in range(numcomb):
                new_weights.append(ew_cen[cell])
                new_mus.append(mus[i])
                new_times.append(times[i])
                new_positions.append(positions[i])
                new_cell_indices.append(cell)
    
    new_weights_arr = np.array(new_weights)
    new_cell_indices_arr = np.array(new_cell_indices, dtype=int)
    
    if len(new_weights_arr) > 0:
        ecen_after = np.bincount(new_cell_indices_arr, weights=new_weights_arr, 
                                 minlength=n_cells)
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
    weights: np.ndarray
    mus: np.ndarray
    times: np.ndarray
    positions: np.ndarray
    cell_indices: np.ndarray
    internal_energy: np.ndarray
    temperature: np.ndarray
    radiation_temperature: np.ndarray
    time: float
    previous_total_energy: float
    count: int


def init_simulation(Ntarget, Tinit, Tr_init, mesh, eos, inv_eos,
                    Ntarget_ic=None, geometry='slab'):
    """Initialize Carter-Forest simulation at t=0."""
    I = mesh.shape[0]
    volumes = _cell_volumes(mesh, geometry)
    
    internal_energy = eos(Tinit)
    temperature = Tinit.copy()
    
    N_ic = Ntarget_ic if Ntarget_ic is not None else Ntarget
    if geometry == 'slab':
        p = equilibrium_sample(N_ic, Tr_init, mesh)
    else:
        p = equilibrium_sample_spherical(N_ic, Tr_init, mesh)
    
    weights = p[0]
    mus = p[1]
    times = p[2]
    positions = p[3]
    cell_indices = p[4]
    
    radiation_temperature = (
        np.bincount(cell_indices, weights=weights, minlength=I) / volumes / __a
    ) ** 0.25
    
    total_internal_energy = np.sum(internal_energy * volumes)
    total_radiation_energy = np.sum(weights)
    previous_total_energy = total_internal_energy + total_radiation_energy
    
    print("Carter-Forest IMC Initialization")
    print("=" * 111)
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
         sigma_a_func, inv_eos, cv, source, reflect=(False, False),
         use_scalar_intensity_Tr=True, conserve_comb_energy=False,
         geometry='slab', event_cap_per_particle=0,
         fastpath_threshold=0.0):
    """
    Advance Carter-Forest simulation by one time step.
    
    Key differences from Fleck-Cummings:
    - No Fleck factor: sigma_a is true absorption cross section
    - No effective scattering
    - Time-delayed re-emission based on exponential sampling
    """
    t_step_start = time.perf_counter()
    I = mesh.shape[0]
    volumes = _cell_volumes(mesh, geometry)
    
    weights = state.weights
    mus = state.mus
    times = state.times
    positions = state.positions
    cell_indices = state.cell_indices
    internal_energy = state.internal_energy
    temperature = state.temperature
    
    # --- Carter-Forest: Use TRUE absorption cross section (no Fleck factor) ---
    sigma_a = sigma_a_func(temperature)
    
    # --- Material coupling parameter β = 4aT³/C_v (held fixed during step) ---
    beta = 4.0 * __a * temperature**3 / cv(temperature)

    t_sampling_start = time.perf_counter()
    
    # --- Boundary sources ---
    T_left = T_boundary[0](state.time) if callable(T_boundary[0]) else T_boundary[0]
    T_right = T_boundary[1](state.time) if callable(T_boundary[1]) else T_boundary[1]
    boundary_emission = 0.0
    
    if T_left > 0 and Nboundary > 0:
        if geometry == 'slab':
            bs = create_boundary(Nboundary, T_left, dt)
        else:
            bs = create_boundary_spherical(Nboundary, T_left, dt, mesh[0, 0], outward=True)
        weights = np.concatenate((weights, bs[0]))
        mus = np.concatenate((mus, bs[1]))
        times = np.concatenate((times, bs[2]))
        positions = np.concatenate((positions, bs[3]))
        cell_indices = np.concatenate((cell_indices, bs[4]))
        boundary_emission += np.sum(bs[0])
    
    if T_right > 0 and Nboundary > 0:
        if geometry == 'slab':
            bs = create_boundary(Nboundary, T_right, dt)
        else:
            bs = create_boundary_spherical(Nboundary, T_right, dt, mesh[-1, 1], outward=False)
        bs_pos = np.full(len(bs[0]), mesh[-1, 1])
        bs_idx = np.full(len(bs[0]), I - 1, dtype=int)
        weights = np.concatenate((weights, bs[0]))
        mus = np.concatenate((mus, -bs[1] if geometry == 'slab' else bs[1]))
        times = np.concatenate((times, bs[2]))
        positions = np.concatenate((positions, bs_pos))
        cell_indices = np.concatenate((cell_indices, bs_idx))
        boundary_emission += np.sum(bs[0])
    
    # --- Fixed source ---
    source_emission = 0.0
    if np.max(source) > 0 and Nsource > 0:
        if geometry == 'slab':
            sp = sample_source(Nsource, source, mesh, dt)
        else:
            sp = sample_source_spherical(Nsource, source, mesh, dt)
        weights = np.concatenate((weights, sp[0]))
        mus = np.concatenate((mus, sp[1]))
        times = np.concatenate((times, sp[2]))
        positions = np.concatenate((positions, sp[3]))
        cell_indices = np.concatenate((cell_indices, sp[4]))
        source_emission = np.sum(sp[0])
    
    # --- Internal emission ---
    if geometry == 'slab':
        internal_source = emitted_particles(Ntarget, temperature, dt, mesh, sigma_a, beta)
    else:
        internal_source = emitted_particles_spherical(Ntarget, temperature, dt, mesh, sigma_a, beta)
    weights = np.concatenate((weights, internal_source[0]))
    mus = np.concatenate((mus, internal_source[1]))
    times = np.concatenate((times, internal_source[2]))
    positions = np.concatenate((positions, internal_source[3]))
    cell_indices = np.concatenate((cell_indices, internal_source[4]))

    t_transport_start = time.perf_counter()
    
    # --- Carter-Forest Transport ---
    weight_floor = 1e-10 * np.sum(weights) / max(len(weights), 1)
    transport_stats = np.zeros(8, dtype=np.int64)
    n_particles_transported = len(weights)
    if geometry == 'slab':
        deposited, scalar_intensity = move_particles_cf(
            weights, mus, times, positions, cell_indices,
            mesh, sigma_a, beta, dt, reflect, transport_stats, weight_floor,
            event_cap_per_particle, fastpath_threshold
        )
    else:
        deposited, scalar_intensity = move_particles_spherical_cf(
            weights, mus, times, positions, cell_indices,
            mesh, sigma_a, beta, dt, reflect, transport_stats, weight_floor,
            event_cap_per_particle, fastpath_threshold
        )

    t_post_start = time.perf_counter()
    
    # --- Update material state ---
    emitted_energies = internal_source[5]
    internal_energy = internal_energy + deposited / volumes - emitted_energies / volumes
    # Safety: prevent negative internal energy (unphysical)
    internal_energy = np.maximum(internal_energy, 1e-10)
    
    temperature = inv_eos(internal_energy)
    
    # --- Remove escaped particles BEFORE computing radiation temperature ---
    boundary_loss = 0.0
    mask = cell_indices < 0
    if reflect[0]:
        cell_indices[mask] = 0
    else:
        boundary_loss += np.sum(weights[mask])
        weights = weights[~mask]
        mus = mus[~mask]
        times = times[~mask]
        positions = positions[~mask]
        cell_indices = cell_indices[~mask]
    
    mask = cell_indices >= I
    if reflect[1]:
        cell_indices[mask] = I - 1
    else:
        boundary_loss += np.sum(weights[mask])
        weights = weights[~mask]
        mus = mus[~mask]
        times = times[~mask]
        positions = positions[~mask]
        cell_indices = cell_indices[~mask]
    
    # --- Radiation temperature (track-length estimator) ---
    new_time = state.time + dt
    if use_scalar_intensity_Tr:
        radiation_temperature = (scalar_intensity / __a / __c) ** 0.25
    else:
        # Scalar energy density method (now with valid cell_indices only)
        ecen = np.bincount(cell_indices, weights=weights, minlength=I)
        Er = ecen / volumes
        radiation_temperature = (Er / __a) ** 0.25
    
    # --- Particle combing ---
    weights, cell_indices, mus, times, positions, comb_discrepancy = comb(
        weights, cell_indices, mus, times, positions, NMax, I)
    
    if conserve_comb_energy:
        internal_energy += comb_discrepancy / volumes
        temperature = inv_eos(internal_energy)
    
    times = np.zeros(times.shape)  # Reset particle times for next step
    
    # --- Energy diagnostics ---
    total_internal_energy = np.sum(internal_energy * volumes)
    total_radiation_energy = np.sum(weights)
    total_energy = total_internal_energy + total_radiation_energy
    energy_loss = (total_energy - state.previous_total_energy
                   - boundary_emission + boundary_loss - source_emission)
    
    # --- Update state ---
    state.weights = weights
    state.mus = mus
    state.times = times
    state.positions = positions
    state.cell_indices = cell_indices
    state.internal_energy = internal_energy
    state.temperature = temperature
    state.radiation_temperature = radiation_temperature
    state.time = new_time
    state.previous_total_energy = total_energy
    state.count += 1

    t_step_end = time.perf_counter()
    events_total = int(transport_stats[0])
    avg_events_per_particle = events_total / max(n_particles_transported, 1)
    
    info = {
        'time': new_time,
        'radiation_temperature': radiation_temperature,
        'temperature': temperature,
        'N_particles': len(weights),
        'total_energy': total_energy,
        'total_internal_energy': total_internal_energy,
        'total_radiation_energy': total_radiation_energy,
        'boundary_emission': boundary_emission,
        'boundary_loss': boundary_loss,
        'source_emission': source_emission,
        'energy_loss': energy_loss,
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
                'absorption_continue_events': int(transport_stats[2]),
                'census_events': int(transport_stats[3]),
                'absorption_capture_events': int(transport_stats[4]),
                'weight_floor_kills': int(transport_stats[5]),
                'reflections': int(transport_stats[6]),
                'event_cap_hits': int(transport_stats[7]),
                'avg_events_per_particle': avg_events_per_particle,
                'n_particles_transported': int(n_particles_transported),
            },
        },
    }
    
    print("{:.6f}".format(new_time), len(weights),
          "{:.6f}".format(total_energy),
          "{:.6f}".format(total_internal_energy),
          "{:.6f}".format(total_radiation_energy),
          "{:.6f}".format(boundary_emission),
          "{:.6f}".format(energy_loss), sep='\t')
    
    return state, info


def run_simulation(Ntarget, Nboundary, Nsource, NMax, Tinit, Tr_init,
                   T_boundary, dt, mesh, sigma_a_func, eos, inv_eos, cv,
                   source, final_time, reflect=(False, False), output_freq=1,
                   use_scalar_intensity_Tr=True, Ntarget_ic=None,
                   conserve_comb_energy=False, geometry='slab',
                   event_cap_per_particle=0, fastpath_threshold=0.0):
    """Run the full Carter-Forest simulation from t=0 to final_time."""
    state = init_simulation(Ntarget, Tinit, Tr_init, mesh, eos, inv_eos,
                            Ntarget_ic=Ntarget_ic, geometry=geometry)
    
    radiation_temperatures = [state.radiation_temperature.copy()]
    temperatures = [state.temperature.copy()]
    time_values = [0.0]
    
    while state.time < final_time:
        step_dt = min(dt, final_time - state.time)
        state, info = step(
            state, Ntarget, Nboundary, Nsource, NMax, T_boundary, step_dt, mesh,
            sigma_a_func, inv_eos, cv, source, reflect=reflect,
            use_scalar_intensity_Tr=use_scalar_intensity_Tr,
            conserve_comb_energy=conserve_comb_energy,
            geometry=geometry,
            event_cap_per_particle=event_cap_per_particle,
            fastpath_threshold=fastpath_threshold,
        )
        
        if state.count % output_freq == 0:
            radiation_temperatures.append(state.radiation_temperature.copy())
            temperatures.append(state.temperature.copy())
            time_values.append(state.time)
    
    return (np.array(time_values),
            np.array(radiation_temperatures),
            np.array(temperatures))


# =============================================================================
# SMOKE TEST
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("Carter-Forest IMC Test: Equilibrium relaxation")
    print("="*70)
    
    # Setup
    Ntarget = 20000
    Nboundary = 0
    Nsource = 0
    NMax = 100000
    dt = 0.01
    L = 0.1
    I = 2
    
    mesh = np.array([[i * L / I, (i + 1) * L / I] for i in range(I)])
    Tinit = np.zeros(I) + 0.5
    Trinit = np.zeros(I) + 0.5
    T_boundary = (0.0, 0.0)
    
    sigma_a_f = lambda T: 1e2 + 0 * T
    source = np.zeros(I)
    cv_val = 0.01
    eos = lambda T: cv_val * T
    inv_eos = lambda u: u / cv_val
    cv_f = lambda T: cv_val + 0 * T
    
    final_time = dt * 5
    
    time_values, rad_temps, mat_temps = run_simulation(
        Ntarget, Nboundary, Nsource, NMax, Tinit, Trinit, T_boundary,
        dt, mesh, sigma_a_f, eos, inv_eos, cv_f, source, final_time,
        reflect=(False, False), output_freq=1, use_scalar_intensity_Tr=True
    )
    
    print("\n" + "="*70)
    print("Final slab state:")
    print(f"  Material temperature: {mat_temps[-1]}")
    print(f"  Radiation temperature: {rad_temps[-1]}")
    print("="*70)

    print("\n" + "="*70)
    print("Carter-Forest IMC Test: Spherical equilibrium relaxation")
    print("="*70)

    R = 0.1
    I_sph = 4
    mesh_sph = np.array([[i * R / I_sph, (i + 1) * R / I_sph] for i in range(I_sph)])
    Tinit_sph = np.zeros(I_sph) + 0.5
    Trinit_sph = np.zeros(I_sph) + 0.5
    source_sph = np.zeros(I_sph)

    times_sp, rad_temps_sp, mat_temps_sp = run_simulation(
        Ntarget, Nboundary, Nsource, NMax, Tinit_sph, Trinit_sph, T_boundary,
        dt, mesh_sph, sigma_a_f, eos, inv_eos, cv_f, source_sph, final_time,
        reflect=(True, True), output_freq=1, use_scalar_intensity_Tr=True,
        geometry='spherical'
    )

    print("\n" + "="*70)
    print("Final spherical state:")
    print(f"  Material temperature: {mat_temps_sp[-1]}")
    print(f"  Radiation temperature: {rad_temps_sp[-1]}")
    print("="*70)
