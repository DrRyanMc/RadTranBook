"""IMC2D_CarterForest.py - 2D Carter-Forest IMC with xy/rz support.

This module mirrors IMC2D.py API but uses Carter-Forest transport:
- true absorption (no Fleck effective scattering)
- delayed re-emission using exponential sampling
- optional stiff fast-path and event cap

Implemented to be feature-compatible with IMC2D for:
- geometry='xy' and geometry='rz'
- rz_linear_source source tilting
- boundary_source_func for position-dependent boundary emission
"""

import math
import random
import time as _time

import numpy as np

import IMC2D as fc2d


__c = fc2d.__c
__a = fc2d.__a
SimulationState2D = fc2d.SimulationState2D
init_simulation = fc2d.init_simulation
jit = fc2d.jit
prange = fc2d.prange
get_thread_id = fc2d.get_thread_id
get_num_threads = fc2d.get_num_threads

_GEOM_XY = 0
_GEOM_RZ = 1

_CROSS_NONE = 0
_CROSS_I_PLUS = 1
_CROSS_I_MINUS = 2
_CROSS_J_PLUS = 3
_CROSS_J_MINUS = 4

_EVT_CENSUS = 0
_EVT_BOUNDARY = 1
_EVT_ABSORB_CONTINUE = 2
_EVT_CAPTURE = 4

# Relative tolerance for declaring "no progress" in distance-to-census updates.
_NO_PROGRESS_REL_TOL = 1e-12
_NO_PROGRESS_MAX_STREAK = 8


@jit(nopython=True, cache=True)
def _sample_truncated_exp_time_jit(rate, dt):
    if rate <= 1e-12:
        return random.random() * dt
    xi = random.random()
    return -math.log(1.0 - xi * (1.0 - math.exp(-rate * dt))) / rate


@jit(nopython=True, cache=True)
def _sample_isotropic_xy_single_jit():
    uz = 2.0 * random.random() - 1.0
    phi = 2.0 * math.pi * random.random()
    r_xy = math.sqrt(max(0.0, 1.0 - uz * uz))
    return r_xy * math.cos(phi), r_xy * math.sin(phi)


@jit(nopython=True, cache=True)
def _sample_isotropic_rz_single_jit():
    eta = 2.0 * random.random() - 1.0
    mu_perp = math.cos(2.0 * math.pi * random.random())
    return mu_perp, eta


@jit(nopython=True, cache=True)
def _distance_to_radial_boundary_rz_jit(r, mu_perp, eta, R):
    one_minus_eta2 = 1.0 - eta * eta
    if one_minus_eta2 < 1e-15:
        return 1e30

    b2 = r * r * (1.0 - mu_perp * mu_perp)
    disc = R * R - b2
    if disc <= 0.0:
        return 1e30

    root = math.sqrt(disc)
    l_plus = -r * mu_perp + root
    l_minus = -r * mu_perp - root

    denom = math.sqrt(one_minus_eta2)
    smin = 1e30
    if l_plus > 1e-15:
        s = l_plus / denom
        if s < smin:
            smin = s
    if l_minus > 1e-15:
        s = l_minus / denom
        if s < smin:
            smin = s
    return smin


@jit(nopython=True, cache=True)
def _move_particle_xy_cf_jit(
    weight,
    ux,
    uy,
    x,
    y,
    i,
    j,
    x_edges,
    y_edges,
    sigma_a,
    beta,
    distance_to_census,
    time_local,
    fastpath_threshold,
):
    x_l = x_edges[i]
    x_r = x_edges[i + 1]
    y_l = y_edges[j]
    y_r = y_edges[j + 1]

    sx = 1e30
    sy = 1e30
    if ux > 0.0:
        sx = (x_r - x) / ux
    elif ux < 0.0:
        sx = (x_l - x) / ux

    if uy > 0.0:
        sy = (y_r - y) / uy
    elif uy < 0.0:
        sy = (y_l - y) / uy

    s_boundary = sx if sx < sy else sy

    if sigma_a > 1e-12:
        s_absorb = -math.log(max(1e-16, 1.0 - random.random())) / sigma_a
    else:
        s_absorb = 1e30

    cell_len = min(x_r - x_l, y_r - y_l)
    eps_s = max(1e-14, 1e-12 * cell_len)
    eps_x = max(1e-14, 1e-12 * (x_r - x_l))
    eps_y = max(1e-14, 1e-12 * (y_r - y_l))
    if sigma_a * cell_len > 10000.0:
        s_absorb = -math.log(max(1e-16, 1.0 - random.random())) / (10000.0 / cell_len)

    s_event = s_boundary
    if s_absorb < s_event:
        s_event = s_absorb
    if distance_to_census < s_event:
        s_event = distance_to_census

    if abs(s_event - s_boundary) < 1e-11 and s_event < eps_s and distance_to_census > eps_s:
        s_event = eps_s

    if s_event <= 0.0:
        return weight, ux, uy, x, y, time_local, _CROSS_NONE, 0.0, 0.0, 0.0, _EVT_CENSUS

    x_new = x + ux * s_event
    y_new = y + uy * s_event
    deposited_intensity = weight * s_event
    distance_to_census -= s_event

    if abs(s_event - s_boundary) < 1e-11:
        if sx <= sy:
            if ux > 0.0:
                x_new = x_r + eps_x
                crossed = _CROSS_I_PLUS
            else:
                x_new = x_l - eps_x
                crossed = _CROSS_I_MINUS
        else:
            if uy > 0.0:
                y_new = y_r + eps_y
                crossed = _CROSS_J_PLUS
            else:
                y_new = y_l - eps_y
                crossed = _CROSS_J_MINUS
        return weight, ux, uy, x_new, y_new, time_local, crossed, 0.0, deposited_intensity, distance_to_census, _EVT_BOUNDARY

    if abs(s_event - s_absorb) < 1e-11:
        if beta > 1e-15 and sigma_a > 1e-15:
            rate = __c * sigma_a * beta
            time_remaining = distance_to_census / __c

            if fastpath_threshold > 0.0 and time_remaining > 1e-14 and rate * time_remaining > fastpath_threshold:
                p_capture = math.exp(-rate * time_remaining)
                if random.random() < p_capture:
                    return 0.0, ux, uy, x_new, y_new, time_local, _CROSS_NONE, weight, deposited_intensity, 0.0, _EVT_CAPTURE
                ux_new, uy_new = _sample_isotropic_xy_single_jit()
                return weight, ux_new, uy_new, x_new, y_new, time_local, _CROSS_NONE, 0.0, deposited_intensity, distance_to_census, _EVT_ABSORB_CONTINUE

            # Sample re-emission delay from UNTRUNCATED exponential (like 1D)
            # This allows delays > time_remaining, leading to capture
            xi = random.random()
            if xi < 1e-15:
                xi = 1e-15
            delay = -math.log(xi) / rate
            
            if delay < time_remaining - 1e-15:
                ux_new, uy_new = _sample_isotropic_xy_single_jit()
                time_local = time_local + s_event / __c + delay
                distance_to_census = max(0.0, (time_remaining - delay) * __c)
                if distance_to_census <= eps_s:
                    return weight, ux_new, uy_new, x_new, y_new, time_local, _CROSS_NONE, 0.0, deposited_intensity, 0.0, _EVT_CENSUS
                return weight, ux_new, uy_new, x_new, y_new, time_local, _CROSS_NONE, 0.0, deposited_intensity, distance_to_census, _EVT_ABSORB_CONTINUE

        return 0.0, ux, uy, x_new, y_new, time_local, _CROSS_NONE, weight, deposited_intensity, 0.0, _EVT_CAPTURE

    return weight, ux, uy, x_new, y_new, time_local, _CROSS_NONE, 0.0, deposited_intensity, 0.0, _EVT_CENSUS


@jit(nopython=True, cache=True)
def _move_particle_rz_cf_jit(
    weight,
    mu_perp,
    eta,
    r,
    z,
    i,
    j,
    r_edges,
    z_edges,
    sigma_a,
    beta,
    distance_to_census,
    time_local,
    fastpath_threshold,
):
    r_in = r_edges[i]
    r_out = r_edges[i + 1]
    z_lo = z_edges[j]
    z_hi = z_edges[j + 1]

    s_r_in = _distance_to_radial_boundary_rz_jit(r, mu_perp, eta, r_in) if r_in > 0.0 else 1e30
    s_r_out = _distance_to_radial_boundary_rz_jit(r, mu_perp, eta, r_out)

    if eta > 1e-15:
        s_z_lo = 1e30
        s_z_hi = (z_hi - z) / eta
    elif eta < -1e-15:
        s_z_lo = (z_lo - z) / eta
        s_z_hi = 1e30
    else:
        s_z_lo = 1e30
        s_z_hi = 1e30

    s_boundary = s_r_in
    if s_r_out < s_boundary:
        s_boundary = s_r_out
    if s_z_lo < s_boundary:
        s_boundary = s_z_lo
    if s_z_hi < s_boundary:
        s_boundary = s_z_hi

    if sigma_a > 1e-12:
        s_absorb = -math.log(max(1e-16, 1.0 - random.random())) / sigma_a
    else:
        s_absorb = 1e30

    cell_len = min(r_out - r_in, z_hi - z_lo)
    eps_s = max(1e-14, 1e-12 * cell_len)
    eps_r = max(1e-14, 1e-12 * (r_out - r_in))
    eps_z = max(1e-14, 1e-12 * (z_hi - z_lo))
    if sigma_a * cell_len > 10000.0:
        s_absorb = -math.log(max(1e-16, 1.0 - random.random())) / (10000.0 / cell_len)

    s_event = s_boundary
    if s_absorb < s_event:
        s_event = s_absorb
    if distance_to_census < s_event:
        s_event = distance_to_census

    if abs(s_event - s_boundary) < 1e-11 and s_event < eps_s and distance_to_census > eps_s:
        s_event = eps_s
    if s_event <= 0.0:
        return weight, mu_perp, eta, r, z, time_local, _CROSS_NONE, 0.0, 0.0, 0.0, _EVT_CENSUS

    one_minus_eta2 = max(0.0, 1.0 - eta * eta)
    l = s_event * math.sqrt(one_minus_eta2)

    r_new2 = r * r + 2.0 * r * mu_perp * l + l * l
    r_new = math.sqrt(max(r_new2, 0.0))
    z_new = z + s_event * eta

    if r_new > 1e-15 and l > 0.0:
        mu_new = (r * mu_perp + l) / r_new
        mu_new = max(-1.0, min(1.0, mu_new))
    else:
        mu_new = mu_perp

    eta_new = eta
    deposited_intensity = weight * s_event
    distance_to_census -= s_event

    if abs(s_event - s_boundary) < 1e-11:
        if s_boundary == s_r_in:
            r_new = max(1e-12, (r_in - eps_r)) if r_in > 0.0 else 1e-12
            crossed = _CROSS_I_MINUS
        elif s_boundary == s_r_out:
            r_new = r_out + eps_r
            crossed = _CROSS_I_PLUS
        elif s_boundary == s_z_lo:
            z_new = z_lo - eps_z
            crossed = _CROSS_J_MINUS
        else:
            z_new = z_hi + eps_z
            crossed = _CROSS_J_PLUS
        return weight, mu_new, eta_new, r_new, z_new, time_local, crossed, 0.0, deposited_intensity, distance_to_census, _EVT_BOUNDARY

    if abs(s_event - s_absorb) < 1e-11:
        if beta > 1e-15 and sigma_a > 1e-15:
            rate = __c * sigma_a * beta
            time_remaining = distance_to_census / __c

            if fastpath_threshold > 0.0 and time_remaining > 1e-14 and rate * time_remaining > fastpath_threshold:
                p_capture = math.exp(-rate * time_remaining)
                if random.random() < p_capture:
                    return 0.0, mu_new, eta_new, r_new, z_new, time_local, _CROSS_NONE, weight, deposited_intensity, 0.0, _EVT_CAPTURE
                mu_s, eta_s = _sample_isotropic_rz_single_jit()
                return weight, mu_s, eta_s, r_new, z_new, time_local, _CROSS_NONE, 0.0, deposited_intensity, distance_to_census, _EVT_ABSORB_CONTINUE

            # Sample re-emission delay from UNTRUNCATED exponential (like 1D)
            # This allows delays > time_remaining, leading to capture
            xi = random.random()
            if xi < 1e-15:
                xi = 1e-15
            delay = -math.log(xi) / rate
            
            if delay < time_remaining - 1e-15:
                mu_s, eta_s = _sample_isotropic_rz_single_jit()
                time_local = time_local + s_event / __c + delay
                distance_to_census = max(0.0, (time_remaining - delay) * __c)
                if distance_to_census <= eps_s:
                    return weight, mu_s, eta_s, r_new, z_new, time_local, _CROSS_NONE, 0.0, deposited_intensity, 0.0, _EVT_CENSUS
                return weight, mu_s, eta_s, r_new, z_new, time_local, _CROSS_NONE, 0.0, deposited_intensity, distance_to_census, _EVT_ABSORB_CONTINUE

        return 0.0, mu_new, eta_new, r_new, z_new, time_local, _CROSS_NONE, weight, deposited_intensity, 0.0, _EVT_CAPTURE

    return weight, mu_new, eta_new, r_new, z_new, time_local, _CROSS_NONE, 0.0, deposited_intensity, 0.0, _EVT_CENSUS


@jit(nopython=True, parallel=True)
def _transport_particles_2d_cf(
    weights,
    dir1,
    dir2,
    times,
    pos1,
    pos2,
    cell_i,
    cell_j,
    edges1,
    edges2,
    sigma_a,
    beta,
    volumes,
    dt,
    reflect,
    max_events_per_particle,
    geometry_code,
    weight_floor,
    fastpath_threshold,
    no_progress_rel_tol,
    no_progress_max_streak,
):
    n = len(weights)
    nx, ny = sigma_a.shape
    n_threads = get_num_threads()
    dep_threads = np.zeros((n_threads, nx, ny))
    si_threads = np.zeros((n_threads, nx, ny))
    bl_threads = np.zeros(n_threads)
    stats_threads = np.zeros((n_threads, 8), dtype=np.int64)

    for k in prange(n):
        tid = get_thread_id()
        dcen = (dt - times[k]) * __c
        local_events = 0
        no_progress_count = 0

        while dcen > 0.0 and weights[k] > weight_floor:
            dcen_before = dcen
            i = int(cell_i[k])
            j = int(cell_j[k])
            if i < 0 or i >= nx or j < 0 or j >= ny:
                break

            local_events += 1
            stats_threads[tid, 0] += 1
            if max_events_per_particle > 0 and local_events > max_events_per_particle:
                stats_threads[tid, 7] += 1
                dcen = 0.0
                break

            if geometry_code == _GEOM_XY:
                out = _move_particle_xy_cf_jit(
                    weights[k], dir1[k], dir2[k], pos1[k], pos2[k], i, j,
                    edges1, edges2, sigma_a[i, j], beta[i, j], dcen, times[k],
                    fastpath_threshold,
                )
            else:
                out = _move_particle_rz_cf_jit(
                    weights[k], dir1[k], dir2[k], pos1[k], pos2[k], i, j,
                    edges1, edges2, sigma_a[i, j], beta[i, j], dcen, times[k],
                    fastpath_threshold,
                )

            (
                w_new,
                d1_new,
                d2_new,
                p1_new,
                p2_new,
                t_new,
                crossed,
                dep_w,
                dep_i,
                dcen,
                evt,
            ) = out

            vol = volumes[i, j]
            dep_threads[tid, i, j] += dep_w
            si_threads[tid, i, j] += dep_i / (dt * vol)

            weights[k] = w_new
            dir1[k] = d1_new
            dir2[k] = d2_new
            pos1[k] = p1_new
            pos2[k] = p2_new
            times[k] = t_new

            if evt == _EVT_BOUNDARY:
                stats_threads[tid, 1] += 1
            elif evt == _EVT_ABSORB_CONTINUE:
                stats_threads[tid, 2] += 1
            elif evt == _EVT_CAPTURE:
                stats_threads[tid, 4] += 1
            else:
                stats_threads[tid, 3] += 1

            if dcen >= dcen_before * (1.0 - no_progress_rel_tol):
                no_progress_count += 1
                if no_progress_count >= no_progress_max_streak:
                    dcen = 0.0
                    break
            else:
                no_progress_count = 0

            if crossed != _CROSS_NONE:
                if crossed == _CROSS_I_PLUS:
                    cell_i[k] += 1
                elif crossed == _CROSS_I_MINUS:
                    cell_i[k] -= 1
                elif crossed == _CROSS_J_PLUS:
                    cell_j[k] += 1
                elif crossed == _CROSS_J_MINUS:
                    cell_j[k] -= 1

                i2 = int(cell_i[k])
                j2 = int(cell_j[k])

                if i2 < 0:
                    if reflect[0]:
                        dir1[k] = -dir1[k]
                        cell_i[k] = 0
                        stats_threads[tid, 6] += 1
                    else:
                        bl_threads[tid] += weights[k]
                        weights[k] = 0.0
                        dcen = 0.0
                elif i2 >= nx:
                    if reflect[1]:
                        dir1[k] = -dir1[k]
                        cell_i[k] = nx - 1
                        stats_threads[tid, 6] += 1
                    else:
                        bl_threads[tid] += weights[k]
                        weights[k] = 0.0
                        dcen = 0.0
                elif j2 < 0:
                    if reflect[2]:
                        dir2[k] = -dir2[k]
                        cell_j[k] = 0
                        stats_threads[tid, 6] += 1
                    else:
                        bl_threads[tid] += weights[k]
                        weights[k] = 0.0
                        dcen = 0.0
                elif j2 >= ny:
                    if reflect[3]:
                        dir2[k] = -dir2[k]
                        cell_j[k] = ny - 1
                        stats_threads[tid, 6] += 1
                    else:
                        bl_threads[tid] += weights[k]
                        weights[k] = 0.0
                        dcen = 0.0

            if weights[k] <= weight_floor and weights[k] > 0.0:
                stats_threads[tid, 5] += 1
                weights[k] = 0.0
                dcen = 0.0

    return dep_threads.sum(axis=0), si_threads.sum(axis=0), bl_threads.sum(), stats_threads.sum(axis=0)


def _sample_truncated_exp_time(rate, dt):
    if rate <= 1e-12:
        return random.random() * dt
    xi = random.random()
    return -math.log(1.0 - xi * (1.0 - math.exp(-rate * dt))) / rate


def _move_particle_xy_cf(weight, ux, uy, x, y, i, j, x_edges, y_edges, sigma_a, beta, distance_to_census, time_local, fastpath_threshold=0.0):
    x_l = x_edges[i]
    x_r = x_edges[i + 1]
    y_l = y_edges[j]
    y_r = y_edges[j + 1]

    sx = 1e30
    sy = 1e30
    if ux > 0.0:
        sx = (x_r - x) / ux
    elif ux < 0.0:
        sx = (x_l - x) / ux

    if uy > 0.0:
        sy = (y_r - y) / uy
    elif uy < 0.0:
        sy = (y_l - y) / uy

    s_boundary = sx if sx < sy else sy

    if sigma_a > 1e-12:
        s_absorb = -math.log(max(1e-16, 1.0 - random.random())) / sigma_a
    else:
        s_absorb = 1e30

    cell_len = min(x_r - x_l, y_r - y_l)
    eps_s = max(1e-14, 1e-12 * cell_len)
    eps_x = max(1e-14, 1e-12 * (x_r - x_l))
    eps_y = max(1e-14, 1e-12 * (y_r - y_l))
    if sigma_a * cell_len > 10000.0:
        s_absorb = -math.log(max(1e-16, 1.0 - random.random())) / (10000.0 / cell_len)

    s_event = s_boundary
    if s_absorb < s_event:
        s_event = s_absorb
    if distance_to_census < s_event:
        s_event = distance_to_census

    # Prevent tiny near-boundary steps from causing excessive event churn.
    if abs(s_event - s_boundary) < 1e-11 and s_event < eps_s and distance_to_census > eps_s:
        s_event = eps_s

    if s_event <= 0.0:
        return weight, ux, uy, x, y, time_local, "none", 0.0, 0.0, 0.0, "census"

    x_new = x + ux * s_event
    y_new = y + uy * s_event
    deposited_intensity = weight * s_event
    distance_to_census -= s_event

    # Boundary event
    if abs(s_event - s_boundary) < 1e-11:
        if sx <= sy:
            if ux > 0.0:
                x_new = x_r + eps_x
                crossed = "x+"
            else:
                x_new = x_l - eps_x
                crossed = "x-"
        else:
            if uy > 0.0:
                y_new = y_r + eps_y
                crossed = "y+"
            else:
                y_new = y_l - eps_y
                crossed = "y-"
        return weight, ux, uy, x_new, y_new, time_local, crossed, 0.0, deposited_intensity, distance_to_census, "boundary"

    # Absorption event
    if abs(s_event - s_absorb) < 1e-11:
        if beta > 1e-15 and sigma_a > 1e-15:
            rate = __c * sigma_a * beta
            time_remaining = distance_to_census / __c

            if fastpath_threshold > 0.0 and time_remaining > 1e-14 and rate * time_remaining > fastpath_threshold:
                p_capture = math.exp(-rate * time_remaining)
                if random.random() < p_capture:
                    return 0.0, ux, uy, x_new, y_new, time_local, "none", weight, deposited_intensity, 0.0, "capture"
                # net continue
                ux_new, uy_new = fc2d._sample_isotropic_xy_single()
                return weight, ux_new, uy_new, x_new, y_new, time_local, "none", 0.0, deposited_intensity, distance_to_census, "absorb_continue"

            delay = _sample_truncated_exp_time(rate, max(time_remaining, 1e-30))
            if delay < time_remaining - 1e-15:
                ux_new, uy_new = fc2d._sample_isotropic_xy_single()
                time_local = time_local + s_event / __c + delay
                distance_to_census = max(0.0, (time_remaining - delay) * __c)
                if distance_to_census <= eps_s:
                    # Remaining path is numerically negligible; census now.
                    return weight, ux_new, uy_new, x_new, y_new, time_local, "none", 0.0, deposited_intensity, 0.0, "census"
                return weight, ux_new, uy_new, x_new, y_new, time_local, "none", 0.0, deposited_intensity, distance_to_census, "absorb_continue"

        # capture in material this step
        return 0.0, ux, uy, x_new, y_new, time_local, "none", weight, deposited_intensity, 0.0, "capture"

    # Census event
    return weight, ux, uy, x_new, y_new, time_local, "none", 0.0, deposited_intensity, 0.0, "census"


def _move_particle_rz_cf(weight, mu_perp, eta, r, z, i, j, r_edges, z_edges, sigma_a, beta, distance_to_census, time_local, fastpath_threshold=0.0):
    r_in = r_edges[i]
    r_out = r_edges[i + 1]
    z_lo = z_edges[j]
    z_hi = z_edges[j + 1]

    s_r_in = fc2d._distance_to_radial_boundary_rz(r, mu_perp, eta, r_in) if r_in > 0.0 else 1e30
    s_r_out = fc2d._distance_to_radial_boundary_rz(r, mu_perp, eta, r_out)

    if eta > 1e-15:
        s_z_lo = 1e30
        s_z_hi = (z_hi - z) / eta
    elif eta < -1e-15:
        s_z_lo = (z_lo - z) / eta
        s_z_hi = 1e30
    else:
        s_z_lo = 1e30
        s_z_hi = 1e30

    s_boundary = min(s_r_in, s_r_out, s_z_lo, s_z_hi)

    if sigma_a > 1e-12:
        s_absorb = -math.log(max(1e-16, 1.0 - random.random())) / sigma_a
    else:
        s_absorb = 1e30

    cell_len = min(r_out - r_in, z_hi - z_lo)
    eps_s = max(1e-14, 1e-12 * cell_len)
    eps_r = max(1e-14, 1e-12 * (r_out - r_in))
    eps_z = max(1e-14, 1e-12 * (z_hi - z_lo))
    if sigma_a * cell_len > 10000.0:
        s_absorb = -math.log(max(1e-16, 1.0 - random.random())) / (10000.0 / cell_len)

    s_event = min(s_boundary, s_absorb, distance_to_census)
    if abs(s_event - s_boundary) < 1e-11 and s_event < eps_s and distance_to_census > eps_s:
        s_event = eps_s
    if s_event <= 0.0:
        return weight, mu_perp, eta, r, z, time_local, "none", 0.0, 0.0, 0.0, "census"

    one_minus_eta2 = max(0.0, 1.0 - eta * eta)
    l = s_event * math.sqrt(one_minus_eta2)

    r_new2 = r * r + 2.0 * r * mu_perp * l + l * l
    r_new = math.sqrt(max(r_new2, 0.0))
    z_new = z + s_event * eta

    if r_new > 1e-15 and l > 0.0:
        mu_new = (r * mu_perp + l) / r_new
        mu_new = max(-1.0, min(1.0, mu_new))
    else:
        mu_new = mu_perp

    eta_new = eta
    deposited_intensity = weight * s_event
    distance_to_census -= s_event

    if abs(s_event - s_boundary) < 1e-11:
        if s_boundary == s_r_in:
            r_new = max(1e-12, (r_in - eps_r)) if r_in > 0.0 else 1e-12
            crossed = "r-"
        elif s_boundary == s_r_out:
            r_new = r_out + eps_r
            crossed = "r+"
        elif s_boundary == s_z_lo:
            z_new = z_lo - eps_z
            crossed = "z-"
        else:
            z_new = z_hi + eps_z
            crossed = "z+"
        return weight, mu_new, eta_new, r_new, z_new, time_local, crossed, 0.0, deposited_intensity, distance_to_census, "boundary"

    if abs(s_event - s_absorb) < 1e-11:
        if beta > 1e-15 and sigma_a > 1e-15:
            rate = __c * sigma_a * beta
            time_remaining = distance_to_census / __c

            if fastpath_threshold > 0.0 and time_remaining > 1e-14 and rate * time_remaining > fastpath_threshold:
                p_capture = math.exp(-rate * time_remaining)
                if random.random() < p_capture:
                    return 0.0, mu_new, eta_new, r_new, z_new, time_local, "none", weight, deposited_intensity, 0.0, "capture"
                mu_s, eta_s = fc2d._sample_isotropic_rz_single()
                return weight, mu_s, eta_s, r_new, z_new, time_local, "none", 0.0, deposited_intensity, distance_to_census, "absorb_continue"

            delay = _sample_truncated_exp_time(rate, max(time_remaining, 1e-30))
            if delay < time_remaining - 1e-15:
                mu_s, eta_s = fc2d._sample_isotropic_rz_single()
                time_local = time_local + s_event / __c + delay
                distance_to_census = max(0.0, (time_remaining - delay) * __c)
                if distance_to_census <= eps_s:
                    return weight, mu_s, eta_s, r_new, z_new, time_local, "none", 0.0, deposited_intensity, 0.0, "census"
                return weight, mu_s, eta_s, r_new, z_new, time_local, "none", 0.0, deposited_intensity, distance_to_census, "absorb_continue"

        return 0.0, mu_new, eta_new, r_new, z_new, time_local, "none", weight, deposited_intensity, 0.0, "capture"

    return weight, mu_new, eta_new, r_new, z_new, time_local, "none", 0.0, deposited_intensity, 0.0, "census"


def _transport_particles_2d_cf_py(
    weights,
    dir1,
    dir2,
    times,
    pos1,
    pos2,
    cell_i,
    cell_j,
    edges1,
    edges2,
    sigma_a,
    beta,
    volumes,
    dt,
    reflect,
    max_events_per_particle,
    geometry,
    weight_floor,
    fastpath_threshold,
    no_progress_rel_tol,
    no_progress_max_streak,
):
    nx, ny = sigma_a.shape
    dep_cell = np.zeros_like(sigma_a)
    si_cell = np.zeros_like(sigma_a)
    boundary_loss = 0.0

    stats = np.zeros(8, dtype=np.int64)

    for k in range(len(weights)):
        dcen = (dt - times[k]) * __c
        local_events = 0
        no_progress_count = 0
        while dcen > 0.0 and weights[k] > weight_floor:
            dcen_before = dcen
            i = int(cell_i[k])
            j = int(cell_j[k])
            if i < 0 or i >= nx or j < 0 or j >= ny:
                break

            local_events += 1
            stats[0] += 1
            if max_events_per_particle > 0 and local_events > max_events_per_particle:
                stats[7] += 1
                dcen = 0.0
                break

            if geometry == "xy":
                out = _move_particle_xy_cf(
                    weights[k], dir1[k], dir2[k], pos1[k], pos2[k], i, j,
                    edges1, edges2, sigma_a[i, j], beta[i, j], dcen, times[k],
                    fastpath_threshold=fastpath_threshold,
                )
            else:
                out = _move_particle_rz_cf(
                    weights[k], dir1[k], dir2[k], pos1[k], pos2[k], i, j,
                    edges1, edges2, sigma_a[i, j], beta[i, j], dcen, times[k],
                    fastpath_threshold=fastpath_threshold,
                )

            (
                w_new,
                d1_new,
                d2_new,
                p1_new,
                p2_new,
                t_new,
                crossed,
                dep_w,
                dep_i,
                dcen,
                evt,
            ) = out

            vol = volumes[i, j]
            dep_cell[i, j] += dep_w
            si_cell[i, j] += dep_i / (dt * vol)

            weights[k] = w_new
            dir1[k] = d1_new
            dir2[k] = d2_new
            pos1[k] = p1_new
            pos2[k] = p2_new
            times[k] = t_new

            if evt == "boundary":
                stats[1] += 1
            elif evt == "absorb_continue":
                stats[2] += 1
            elif evt == "capture":
                stats[4] += 1
            else:
                stats[3] += 1

            # Stop rare pathological loops where dcen does not decrease meaningfully.
            if dcen >= dcen_before * (1.0 - no_progress_rel_tol):
                no_progress_count += 1
                if no_progress_count >= no_progress_max_streak:
                    dcen = 0.0
                    break
            else:
                no_progress_count = 0

            if crossed != "none":
                if crossed in ("x+", "r+"):
                    cell_i[k] += 1
                elif crossed in ("x-", "r-"):
                    cell_i[k] -= 1
                elif crossed == "y+" or crossed == "z+":
                    cell_j[k] += 1
                elif crossed == "y-" or crossed == "z-":
                    cell_j[k] -= 1

                i2 = int(cell_i[k])
                j2 = int(cell_j[k])

                if i2 < 0:
                    if reflect[0]:
                        dir1[k] = -dir1[k]
                        cell_i[k] = 0
                        stats[6] += 1
                    else:
                        boundary_loss += weights[k]
                        weights[k] = 0.0
                        dcen = 0.0
                elif i2 >= nx:
                    if reflect[1]:
                        dir1[k] = -dir1[k]
                        cell_i[k] = nx - 1
                        stats[6] += 1
                    else:
                        boundary_loss += weights[k]
                        weights[k] = 0.0
                        dcen = 0.0
                elif j2 < 0:
                    if reflect[2]:
                        dir2[k] = -dir2[k]
                        cell_j[k] = 0
                        stats[6] += 1
                    else:
                        boundary_loss += weights[k]
                        weights[k] = 0.0
                        dcen = 0.0
                elif j2 >= ny:
                    if reflect[3]:
                        dir2[k] = -dir2[k]
                        cell_j[k] = ny - 1
                        stats[6] += 1
                    else:
                        boundary_loss += weights[k]
                        weights[k] = 0.0
                        dcen = 0.0

            if weights[k] <= weight_floor and weights[k] > 0.0:
                stats[5] += 1
                weights[k] = 0.0
                dcen = 0.0

    return dep_cell, si_cell, boundary_loss, stats


def _sample_internal_emission_cf(Ntarget, temperature, dt, edges1, edges2, volumes, sigma_a, beta, geometry, rz_linear_source):
    rate = __c * sigma_a * beta
    emissions = np.zeros_like(temperature)

    mask = (rate > 1e-12) & (beta > 1e-15)
    emissions[mask] = __a * np.maximum(temperature[mask], 0.0) ** 4 * (1.0 - np.exp(-rate[mask] * dt)) / beta[mask]
    emissions[~mask] = __a * np.maximum(temperature[~mask], 0.0) ** 4 * __c * sigma_a[~mask] * dt

    emitted_energies = emissions * volumes
    E_emit = float(np.sum(emitted_energies))
    if E_emit <= 0.0 or Ntarget <= 0:
        return None, emitted_energies

    source_density = emitted_energies / (dt * volumes + 1e-300)
    if geometry == "xy":
        p = fc2d._sample_source_xy(Ntarget, source_density, dt, edges1, edges2)
    else:
        if rz_linear_source:
            p = fc2d._sample_source_rz_linear(Ntarget, source_density, temperature, dt, edges1, edges2)
        else:
            p = fc2d._sample_source_rz_uniform(Ntarget, source_density, dt, edges1, edges2)

    w, d1, d2, t, p1, p2, ci, cj = p
    if len(w) == 0:
        return None, emitted_energies

    for k in range(len(t)):
        rloc = rate[ci[k], cj[k]]
        t[k] = _sample_truncated_exp_time(float(rloc), dt)

    return (w, d1, d2, t, p1, p2, ci, cj), emitted_energies


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
    sigma_a_func,
    inv_eos,
    cv,
    source,
    reflect=(False, False, False, False),
    theta=1.0,
    use_scalar_intensity_Tr=True,
    conserve_comb_energy=False,
    geometry="xy",
    rz_linear_source=True,
    max_events_per_particle=1_000_000,
    boundary_source_func=None,
    fastpath_threshold=0.0,
    no_progress_rel_tol=_NO_PROGRESS_REL_TOL,
    no_progress_max_streak=_NO_PROGRESS_MAX_STREAK,
):
    del theta  # Carter-Forest does not use Fleck theta

    nx, ny = fc2d._shape_from_edges(edges1, edges2)
    volumes = fc2d._cell_volumes(edges1, edges2, geometry)

    t_step_start = _time.perf_counter()

    weights = state.weights
    dir1 = state.dir1
    dir2 = state.dir2
    times = state.times
    pos1 = state.pos1
    pos2 = state.pos2
    cell_i = state.cell_i
    cell_j = state.cell_j
    internal_energy = state.internal_energy
    temperature = state.temperature

    sigma_a = sigma_a_func(temperature)
    beta = 4.0 * __a * np.maximum(temperature, 1e-12) ** 3 / cv(temperature)

    t_sampling_start = _time.perf_counter()

    b_left = fc2d._boundary_temperature_value(T_boundary[0], state.time)
    b_right = fc2d._boundary_temperature_value(T_boundary[1], state.time)
    b_bottom = fc2d._boundary_temperature_value(T_boundary[2], state.time)
    b_top = fc2d._boundary_temperature_value(T_boundary[3], state.time)

    boundary_emission = 0.0
    if Nboundary > 0:
        if geometry == "xy":
            for side, Tb in (("left", b_left), ("right", b_right), ("bottom", b_bottom), ("top", b_top)):
                s = fc2d._sample_boundary_xy(Nboundary, side, Tb, dt, edges1, edges2, boundary_source_func)
                if s is None:
                    continue
                w, d1, d2, t, p1, p2 = s
                ci, cj = fc2d._locate_indices(p1, p2, edges1, edges2)
                weights = np.concatenate((weights, w))
                dir1 = np.concatenate((dir1, d1))
                dir2 = np.concatenate((dir2, d2))
                times = np.concatenate((times, t))
                pos1 = np.concatenate((pos1, p1))
                pos2 = np.concatenate((pos2, p2))
                cell_i = np.concatenate((cell_i, ci))
                cell_j = np.concatenate((cell_j, cj))
                boundary_emission += float(np.sum(w))
        else:
            for side, Tb in (("rmin", b_left), ("rmax", b_right), ("zmin", b_bottom), ("zmax", b_top)):
                s = fc2d._sample_boundary_rz(Nboundary, side, Tb, dt, edges1, edges2, boundary_source_func)
                if s is None:
                    continue
                w, d1, d2, t, p1, p2 = s
                ci, cj = fc2d._locate_indices(p1, p2, edges1, edges2)
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
                    boundary_emission += float(np.sum(w[valid]))

    source_emission = 0.0
    if Nsource > 0 and np.max(source) > 0.0:
        if geometry == "xy":
            s = fc2d._sample_source_xy(Nsource, source, dt, edges1, edges2)
        else:
            if rz_linear_source:
                s = fc2d._sample_source_rz_linear(Nsource, source, temperature, dt, edges1, edges2)
            else:
                s = fc2d._sample_source_rz_uniform(Nsource, source, dt, edges1, edges2)
        w, d1, d2, t, p1, p2, ci, cj = s
        if len(w) > 0:
            weights = np.concatenate((weights, w))
            dir1 = np.concatenate((dir1, d1))
            dir2 = np.concatenate((dir2, d2))
            times = np.concatenate((times, t))
            pos1 = np.concatenate((pos1, p1))
            pos2 = np.concatenate((pos2, p2))
            cell_i = np.concatenate((cell_i, ci))
            cell_j = np.concatenate((cell_j, cj))
            source_emission = float(np.sum(w))

    internal_sample, emitted_energies = _sample_internal_emission_cf(
        Ntarget, temperature, dt, edges1, edges2, volumes, sigma_a, beta, geometry, rz_linear_source
    )
    if internal_sample is not None:
        w, d1, d2, t, p1, p2, ci, cj = internal_sample
        weights = np.concatenate((weights, w))
        dir1 = np.concatenate((dir1, d1))
        dir2 = np.concatenate((dir2, d2))
        times = np.concatenate((times, t))
        pos1 = np.concatenate((pos1, p1))
        pos2 = np.concatenate((pos2, p2))
        cell_i = np.concatenate((cell_i, ci))
        cell_j = np.concatenate((cell_j, cj))

    t_transport_start = _time.perf_counter()

    weight_floor = 1e-10 * float(np.sum(weights)) / max(len(weights), 1)
    geometry_code = _GEOM_XY if geometry == "xy" else _GEOM_RZ
    
    # Report thread count on first step
    if state.count == 0:
        print(f"[IMC2D CF] Using {get_num_threads()} threads for transport")
    
    dep_cell, si_cell, boundary_loss, stats = _transport_particles_2d_cf(
        weights,
        dir1,
        dir2,
        times,
        pos1,
        pos2,
        cell_i,
        cell_j,
        edges1,
        edges2,
        sigma_a,
        beta,
        volumes,
        dt,
        reflect,
        max_events_per_particle,
        geometry_code,
        weight_floor,
        fastpath_threshold,
        no_progress_rel_tol,
        no_progress_max_streak,
    )

    t_post_start = _time.perf_counter()

    internal_energy = internal_energy + dep_cell / volumes - emitted_energies / volumes
    internal_energy = np.maximum(internal_energy, 1e-12)
    temperature = inv_eos(internal_energy)

    if use_scalar_intensity_Tr:
        radiation_temperature = (si_cell / __a / __c) ** 0.25
    else:
        valid = weights > 0.0
        flat = fc2d._flatten_index(cell_i[valid], cell_j[valid], nx)
        rad_cell = np.bincount(flat, weights=weights[valid], minlength=nx * ny).reshape(nx, ny)
        radiation_temperature = (rad_cell / volumes / __a) ** 0.25

    (
        weights,
        cell_i,
        cell_j,
        dir1,
        dir2,
        times,
        pos1,
        pos2,
        comb_disc,
    ) = fc2d._comb(weights, cell_i, cell_j, dir1, dir2, times, pos1, pos2, Nmax, nx, ny)

    if conserve_comb_energy:
        internal_energy = internal_energy + comb_disc.reshape(ny, nx).T / volumes
        internal_energy = np.maximum(internal_energy, 1e-12)
        temperature = inv_eos(internal_energy)

    times = np.zeros_like(times)

    total_internal = float(np.sum(internal_energy * volumes))
    total_rad = float(np.sum(weights))
    total_energy = total_internal + total_rad
    energy_loss = total_energy - state.previous_total_energy - boundary_emission + boundary_loss - source_emission

    state.weights = weights
    state.dir1 = dir1
    state.dir2 = dir2
    state.times = times
    state.pos1 = pos1
    state.pos2 = pos2
    state.cell_i = cell_i
    state.cell_j = cell_j
    state.internal_energy = internal_energy
    state.temperature = temperature
    state.radiation_temperature = radiation_temperature
    state.time += dt
    state.previous_total_energy = total_energy
    state.count += 1

    t_end = _time.perf_counter()
    events_total = int(stats[0])
    n_transported = max(len(weights), 1)

    info = {
        "time": state.time,
        "temperature": temperature,
        "radiation_temperature": radiation_temperature,
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
                "sampling": t_transport_start - t_sampling_start,
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
    sigma_a_func,
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
    rz_linear_source=True,
    max_events_per_particle=1_000_000,
    fastpath_threshold=0.0,
    no_progress_rel_tol=_NO_PROGRESS_REL_TOL,
    no_progress_max_streak=_NO_PROGRESS_MAX_STREAK,
):
    state = init_simulation(
        Ntarget,
        Tinit,
        Tr_init,
        edges1,
        edges2,
        eos,
        inv_eos,
        Ntarget_ic=Ntarget_ic,
        geometry=geometry,
    )

    times = [0.0]
    Tr_hist = [state.radiation_temperature.copy()]
    T_hist = [state.temperature.copy()]

    while state.time < final_time - 1e-15:
        step_dt = min(dt, final_time - state.time)
        state, info = step(
            state,
            Ntarget,
            Nboundary,
            Nsource,
            Nmax,
            T_boundary,
            step_dt,
            edges1,
            edges2,
            sigma_a_func,
            inv_eos,
            cv,
            source,
            reflect=reflect,
            theta=theta,
            use_scalar_intensity_Tr=use_scalar_intensity_Tr,
            conserve_comb_energy=conserve_comb_energy,
            geometry=geometry,
            rz_linear_source=rz_linear_source,
            max_events_per_particle=max_events_per_particle,
            fastpath_threshold=fastpath_threshold,
            no_progress_rel_tol=no_progress_rel_tol,
            no_progress_max_streak=no_progress_max_streak,
        )

        if (state.count - 1) % output_freq == 0 or (final_time - state.time) < 1e-12:
            times.append(info["time"])
            Tr_hist.append(state.radiation_temperature.copy())
            T_hist.append(state.temperature.copy())
            print(
                "{:.6f}".format(info["time"]),
                info["N_particles"],
                "{:.6f}".format(info["total_energy"]),
                "{:.6f}".format(info["total_internal_energy"]),
                "{:.6f}".format(info["total_radiation_energy"]),
                "{:.6f}".format(info["boundary_emission"]),
                "{:.6e}".format(info["energy_loss"]),
                sep="\t",
            )

    return np.array(times), np.array(Tr_hist), np.array(T_hist)
