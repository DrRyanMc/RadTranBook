"""IMC2D.py - Implicit Monte Carlo in 2D Cartesian (xy) and cylindrical (r-z).

This module mirrors the IMC1D high-level workflow (init_simulation, step,
run_simulation) while supporting two 2D geometries:

- geometry='xy': Cartesian x-y slab extension
- geometry='rz': Axisymmetric cylindrical r-z tracking/sourcing

For cylindrical r-z tracking/sourcing, formulas follow Box 9.3 and Box 9.4:
- Streaming and boundary intersection with (r, z, mu_perp, eta)
- Uniform and linear-in-r / linear-in-z source sampling

Units are consistent with IMC1D:
- distance: cm
- time: ns
- temperature: keV
- energy: GJ
"""

from dataclasses import dataclass
import math
import random
import time as _time

import numpy as np
try:
    from numba import jit, prange, get_thread_id, get_num_threads
except Exception:
    # Fallback path when numba/llvmlite are unavailable.
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

__c = 29.98
__a = 0.01372

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
class SimulationState2D:
    """Mutable 2D IMC state passed between timesteps."""

    weights: np.ndarray
    dir1: np.ndarray
    dir2: np.ndarray
    times: np.ndarray
    pos1: np.ndarray
    pos2: np.ndarray
    cell_i: np.ndarray
    cell_j: np.ndarray

    internal_energy: np.ndarray
    temperature: np.ndarray
    radiation_temperature: np.ndarray

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


def _sample_isotropic_xy(n):
    """Sample isotropic directions for 2D-in-space, 3D-in-angle transport.

    A particle has a 3D unit direction (ux, uy, uz) with ux^2+uy^2+uz^2=1.
    Since the problem is z-symmetric, uz is irrelevant for x-y transport and
    is discarded.  We sample the full 3D isotropic sphere and return (ux, uy),
    giving <ux^2> = <uy^2> = 1/3, consistent with D = c/(3 sigma_a).
    """
    uz  = np.random.uniform(-1.0, 1.0, n)          # z-cosine, uniform on [-1,1]
    phi = np.random.uniform(0.0, 2.0 * np.pi, n)   # azimuthal angle
    r_xy = np.sqrt(np.maximum(0.0, 1.0 - uz * uz))
    return r_xy * np.cos(phi), r_xy * np.sin(phi)


def _sample_isotropic_rz(n):
    """Sample axisymmetric direction pair (mu_perp, eta).

    eta = zhat . Omega in [-1,1]
    mu_perp = cos(azimuth in transverse plane), see Box 9.3/9.4.
    """
    eta = np.random.uniform(-1.0, 1.0, n)
    mu_perp = np.cos(2.0 * np.pi * np.random.uniform(0.0, 1.0, n))
    return mu_perp, eta


def _boundary_temperature_value(Tb, t):
    return Tb(t) if callable(Tb) else Tb


def _sample_boundary_xy(n, side, T, dt, x_edges, y_edges, boundary_source_func=None):
    """Half-Lambertian boundary source for Cartesian geometry.

    side in {'left', 'right', 'bottom', 'top'}
    
    boundary_source_func: optional callable(x, y, side) -> float
        Returns temperature (keV) at position (x, y) on given side.
        If None, uses uniform temperature T.
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

    # If boundary_source_func is provided, sample from cells with position-dependent temperature
    if boundary_source_func is not None:
        # Identify which boundary cells should emit and at what temperature
        if side in ("left", "right"):
            # For left/right boundaries, cells are indexed by y
            X = x0 if side == "left" else x1
            y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
            
            # Get temperature for each y-cell
            cell_temps = np.array([boundary_source_func(X, y_c, side) for y_c in y_centers])
            emit_mask = cell_temps > 0.0
            
            if not np.any(emit_mask):
                return None  # No cells should emit
            
            # Calculate emission from each cell (proportional to length * T^4)
            dy = np.diff(y_edges)
            cell_lengths = dy[emit_mask]
            cell_temps_emit = cell_temps[emit_mask]
            cell_emissions = __a * __c * cell_temps_emit**4 / 4.0 * cell_lengths * dt
            total_emission = np.sum(cell_emissions)
            
            # Distribute particles proportional to cell emission
            cell_fractions = cell_emissions / total_emission
            n_per_cell = np.random.multinomial(n, cell_fractions)
            
            # Sample positions within each emitting cell
            x_all = []
            y_all = []
            for idx, (emit_idx, n_cell) in enumerate(zip(np.where(emit_mask)[0], n_per_cell)):
                if n_cell > 0:
                    x_all.append(np.full(n_cell, X + (1e-12 if side == "left" else -1e-12)))
                    y_all.append(np.random.uniform(y_edges[emit_idx], y_edges[emit_idx + 1], n_cell))
            
            if len(x_all) == 0:
                return None
            
            x = np.concatenate(x_all)
            y = np.concatenate(y_all)
            
        else:  # bottom or top
            # For bottom/top boundaries, cells are indexed by x
            Y = y0 if side == "bottom" else y1
            x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
            
            # Get temperature for each x-cell
            cell_temps = np.array([boundary_source_func(x_c, Y, side) for x_c in x_centers])
            emit_mask = cell_temps > 0.0
            
            if not np.any(emit_mask):
                return None  # No cells should emit
            
            # Calculate emission from each cell (proportional to length * T^4)
            dx = np.diff(x_edges)
            cell_lengths = dx[emit_mask]
            cell_temps_emit = cell_temps[emit_mask]
            cell_emissions = __a * __c * cell_temps_emit**4 / 4.0 * cell_lengths * dt
            total_emission = np.sum(cell_emissions)
            
            # Distribute particles proportional to cell emission
            cell_fractions = cell_emissions / total_emission
            n_per_cell = np.random.multinomial(n, cell_fractions)
            
            # Sample positions within each emitting cell
            x_all = []
            y_all = []
            for idx, (emit_idx, n_cell) in enumerate(zip(np.where(emit_mask)[0], n_per_cell)):
                if n_cell > 0:
                    x_all.append(np.random.uniform(x_edges[emit_idx], x_edges[emit_idx + 1], n_cell))
                    y_all.append(np.full(n_cell, Y + (1e-12 if side == "bottom" else -1e-12)))
            
            if len(x_all) == 0:
                return None
            
            x = np.concatenate(x_all)
            y = np.concatenate(y_all)
        
        # Sample directions
        n_actual = len(x)
        weights = np.full(n_actual, total_emission / n_actual)
        times = np.random.uniform(0.0, dt, n_actual)
        
        mu_n = np.sqrt(np.random.uniform(0.0, 1.0, n_actual))
        mu_t = np.sqrt(np.maximum(0.0, 1.0 - mu_n * mu_n))
        phi  = np.random.uniform(0.0, 2.0 * np.pi, n_actual)

        if side == "left":
            ux = mu_n
            uy = mu_t * np.cos(phi)
        elif side == "right":
            ux = -mu_n
            uy = mu_t * np.cos(phi)
        elif side == "bottom":
            uy = mu_n
            ux = mu_t * np.cos(phi)
        elif side == "top":
            uy = -mu_n
            ux = mu_t * np.cos(phi)
        
        return weights, ux, uy, times, x, y
    
    # Original uniform sampling if no boundary_source_func
    total_emission = __a * __c * T**4 / 4.0 * area * dt
    weights = np.full(n, total_emission / n)
    times = np.random.uniform(0.0, dt, n)

    # 3D Lambertian: normal component mu_n = sqrt(xi) (same as 1D).
    # Tangential plane has two components (y and z); sample azimuthal angle
    # phi uniformly so that uy = mu_t*cos(phi) and uz = mu_t*sin(phi).
    # uz is discarded (z-symmetry), leaving (ux, uy) with ux^2+uy^2 <= 1.
    mu_n = np.sqrt(np.random.uniform(0.0, 1.0, n))
    mu_t = np.sqrt(np.maximum(0.0, 1.0 - mu_n * mu_n))
    phi  = np.random.uniform(0.0, 2.0 * np.pi, n)

    if side == "left":
        ux = mu_n
        uy = mu_t * np.cos(phi)
        x = np.full(n, x0 + 1e-12)
        y = np.random.uniform(y0, y1, n)
    elif side == "right":
        ux = -mu_n
        uy = mu_t * np.cos(phi)
        x = np.full(n, x1 - 1e-12)
        y = np.random.uniform(y0, y1, n)
    elif side == "bottom":
        uy = mu_n
        ux = mu_t * np.cos(phi)
        y = np.full(n, y0 + 1e-12)
        x = np.random.uniform(x0, x1, n)
    elif side == "top":
        uy = -mu_n
        ux = mu_t * np.cos(phi)
        y = np.full(n, y1 - 1e-12)
        x = np.random.uniform(x0, x1, n)
    else:
        raise ValueError(f"Invalid side: {side}")

    return weights, ux, uy, times, x, y


def _sample_boundary_rz(n, side, T, dt, r_edges, z_edges, boundary_source_func=None):
    """Boundary source for cylindrical r-z geometry.

    side in {'rmin','rmax','zmin','zmax'}
    boundary_source_func: optional callable(r, z, side) -> float
        If provided, returns temperature (keV) for emission at position (r, z).
        Return 0.0 or negative for no emission. Overrides T parameter.
        Allows position-dependent boundary temperatures.
        Particles are distributed among emitting cells proportional to area * T^4.
        For zmin/zmax: cells are annular rings at different radii.
        For rmin/rmax: cells are axial segments at different z.
    """
    if n <= 0:
        return None
    if T <= 0.0 and boundary_source_func is None:
        return None

    r0 = r_edges[0]
    r1 = r_edges[-1]
    z0 = z_edges[0]
    z1 = z_edges[-1]

    if side in ("rmin", "rmax"):
        R = r0 if side == "rmin" else r1
        area = 2.0 * np.pi * R * (z1 - z0)
    else:
        area = np.pi * (r1 * r1 - r0 * r0)

    # If boundary_source_func is provided, sample from cells with position-dependent temperature
    if boundary_source_func is not None:
        # Identify which boundary cells should emit and at what temperature
        if side in ("rmin", "rmax"):
            # For radial boundaries, cells are indexed by z
            R = r0 if side == "rmin" else r1
            z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
            
            # Get temperature for each z-cell
            cell_temps = np.array([boundary_source_func(R, z_c, side) for z_c in z_centers])
            emit_mask = cell_temps > 0.0
            
            if not np.any(emit_mask):
                return None  # No cells should emit
            
            # Calculate emission from each cell (proportional to area * T^4)
            dz = np.diff(z_edges)
            cell_areas = 2.0 * np.pi * R * dz[emit_mask]
            cell_temps_emit = cell_temps[emit_mask]
            cell_emissions = __a * __c * cell_temps_emit**4 / 4.0 * cell_areas * dt
            total_emission = np.sum(cell_emissions)
            
            # Distribute particles proportional to cell emission
            cell_fractions = cell_emissions / total_emission
            n_per_cell = np.random.multinomial(n, cell_fractions)
            
            # Sample positions within each emitting cell
            r_all = []
            z_all = []
            for idx, (emit_idx, n_cell) in enumerate(zip(np.where(emit_mask)[0], n_per_cell)):
                if n_cell > 0:
                    r_all.append(np.full(n_cell, R + (1e-12 if side == "rmin" else -1e-12)))
                    z_all.append(np.random.uniform(z_edges[emit_idx], z_edges[emit_idx + 1], n_cell))
            
            if len(r_all) == 0:
                return None
            
            r = np.concatenate(r_all)
            z = np.concatenate(z_all)
            
        else:  # zmin or zmax
            # For z boundaries, cells are indexed by r (annular rings)
            Z = z0 if side == "zmin" else z1
            r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
            
            # Get temperature for each r-cell
            cell_temps = np.array([boundary_source_func(r_c, Z, side) for r_c in r_centers])
            emit_mask = cell_temps > 0.0
            
            if not np.any(emit_mask):
                return None  # No cells should emit
            
            # Calculate emission from each cell (proportional to area * T^4)
            r_inner = r_edges[:-1][emit_mask]
            r_outer = r_edges[1:][emit_mask]
            cell_areas = np.pi * (r_outer**2 - r_inner**2)
            cell_temps_emit = cell_temps[emit_mask]
            cell_emissions = __a * __c * cell_temps_emit**4 / 4.0 * cell_areas * dt
            total_emission = np.sum(cell_emissions)
            
            # Distribute particles proportional to cell emission
            cell_fractions = cell_emissions / total_emission
            n_per_cell = np.random.multinomial(n, cell_fractions)
            
            # Sample positions within each emitting cell (uniform in annular ring)
            r_all = []
            z_all = []
            for idx, (emit_idx, n_cell) in enumerate(zip(np.where(emit_mask)[0], n_per_cell)):
                if n_cell > 0:
                    r_in = r_edges[emit_idx]
                    r_out = r_edges[emit_idx + 1]
                    # Sample radius uniformly in annular area: r = sqrt(r_in^2 + xi*(r_out^2 - r_in^2))
                    xi = np.random.uniform(0.0, 1.0, n_cell)
                    r_all.append(np.sqrt(r_in**2 + xi * (r_out**2 - r_in**2)))
                    z_all.append(np.full(n_cell, Z + (1e-12 if side == "zmin" else -1e-12)))
            
            if len(r_all) == 0:
                return None
            
            r = np.concatenate(r_all)
            z = np.concatenate(z_all)
        
        # Total emission already calculated above
        n_actual = len(r)
        weights = np.full(n_actual, total_emission / n_actual)
        times = np.random.uniform(0.0, dt, n_actual)
        
        # Sample directions
        eta = np.random.uniform(-1.0, 1.0, n_actual)
        mu_abs = np.sqrt(np.random.uniform(0.0, 1.0, n_actual))
        
        if side == "rmin":
            mu = mu_abs
        elif side == "rmax":
            mu = -mu_abs
        elif side == "zmin":
            eta = np.sqrt(np.random.uniform(0.0, 1.0, n_actual))
            mu = np.cos(2.0 * np.pi * np.random.uniform(0.0, 1.0, n_actual))
        elif side == "zmax":
            eta = -np.sqrt(np.random.uniform(0.0, 1.0, n_actual))
            mu = np.cos(2.0 * np.pi * np.random.uniform(0.0, 1.0, n_actual))
        
        return weights, mu, eta, times, r, z
    
    # Original uniform sampling if no boundary_source_func
    total_emission = __a * __c * T**4 / 4.0 * area * dt
    weights = np.full(n, total_emission / n)
    times = np.random.uniform(0.0, dt, n)

    eta = np.random.uniform(-1.0, 1.0, n)
    mu_abs = np.sqrt(np.random.uniform(0.0, 1.0, n))

    if side == "rmin":
        # Emit outward from inner cylinder.
        mu = mu_abs
        r = np.full(n, r0 + 1e-12)
        z = np.random.uniform(z0, z1, n)
    elif side == "rmax":
        # Emit inward from outer cylinder.
        mu = -mu_abs
        r = np.full(n, r1 - 1e-12)
        z = np.random.uniform(z0, z1, n)
    elif side == "zmin":
        # Emit toward +z: Lambertian normal component eta = sqrt(xi).
        eta = np.sqrt(np.random.uniform(0.0, 1.0, n))
        mu = np.cos(2.0 * np.pi * np.random.uniform(0.0, 1.0, n))
        z = np.full(n, z0 + 1e-12)
        xi = np.random.uniform(0.0, 1.0, n)
        r = np.sqrt(r0 * r0 + (r1 * r1 - r0 * r0) * xi)
    elif side == "zmax":
        # Emit toward -z: Lambertian normal component |eta| = sqrt(xi).
        eta = -np.sqrt(np.random.uniform(0.0, 1.0, n))
        mu = np.cos(2.0 * np.pi * np.random.uniform(0.0, 1.0, n))
        z = np.full(n, z1 - 1e-12)
        xi = np.random.uniform(0.0, 1.0, n)
        r = np.sqrt(r0 * r0 + (r1 * r1 - r0 * r0) * xi)
    else:
        raise ValueError(f"Invalid side: {side}")

    return weights, mu, eta, times, r, z


def _move_particle_xy(weight, ux, uy, x, y, i, j, x_edges, y_edges, sigma_a, sigma_s, distance_to_census):
    """Move one Cartesian particle to next event (boundary/scatter/census)."""
    x_l = x_edges[i]
    x_r = x_edges[i + 1]
    y_l = y_edges[j]
    y_r = y_edges[j + 1]

    # Distances to each face in current cell.
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

    if sigma_s > 1e-12:
        s_scatter = -math.log(1.0 - random.random()) / sigma_s
    else:
        s_scatter = 1e30

    # Match IMC1D stabilization: cap excessively small scatter distances in
    # highly optically thick cells to prevent pathological event counts.
    cell_len = min(x_r - x_l, y_r - y_l)
    if sigma_s * cell_len > 10000.0:
        s_scatter = -math.log(1.0 - random.random()) / (10000.0 / cell_len)

    s_boundary = min(sx, sy)
    s_event = min(s_boundary, s_scatter, distance_to_census)
    assert s_event > 0.0, "Non-positive distance in xy transport"

    x_new = x + ux * s_event
    y_new = y + uy * s_event

    crossed = None
    if abs(s_event - s_boundary) < 1e-11:
        if sx <= sy:
            if ux > 0.0:
                x_new = x_r
                crossed = "x+"
            else:
                x_new = x_l
                crossed = "x-"
        else:
            if uy > 0.0:
                y_new = y_r
                crossed = "y+"
            else:
                y_new = y_l
                crossed = "y-"
    elif s_event == s_scatter:
        ux, uy = _sample_isotropic_xy(1)
        ux = float(ux[0])
        uy = float(uy[0])

    distance_to_census -= s_event

    weight_factor = math.exp(-sigma_a * s_event)
    if sigma_a > 1e-12:
        deposited_intensity = weight * (1.0 - weight_factor) / sigma_a
    else:
        deposited_intensity = weight * s_event
    deposited_weight = deposited_intensity * sigma_a
    weight *= weight_factor

    return (
        weight,
        ux,
        uy,
        x_new,
        y_new,
        crossed,
        deposited_weight,
        deposited_intensity,
        distance_to_census,
    )


def _distance_to_radial_boundary_rz(r, mu_perp, eta, R):
    """Physical distance to cylindrical surface of radius R (Box 9.3)."""
    one_minus_eta2 = max(0.0, 1.0 - eta * eta)
    if one_minus_eta2 < 1e-15:
        return 1e30

    b2 = r * r * (1.0 - mu_perp * mu_perp)
    disc = R * R - b2
    if disc <= 0.0:
        return 1e30

    l_plus = -r * mu_perp + math.sqrt(disc)
    l_minus = -r * mu_perp - math.sqrt(disc)

    denom = math.sqrt(one_minus_eta2)
    s_candidates = []
    if l_plus > 1e-15:
        s_candidates.append(l_plus / denom)
    if l_minus > 1e-15:
        s_candidates.append(l_minus / denom)

    if not s_candidates:
        return 1e30
    return min(s_candidates)


def _move_particle_rz(weight, mu_perp, eta, r, z, i, j, r_edges, z_edges, sigma_a, sigma_s, distance_to_census):
    """Move one cylindrical r-z particle to next event (Box 9.3)."""
    r_in = r_edges[i]
    r_out = r_edges[i + 1]
    z_lo = z_edges[j]
    z_hi = z_edges[j + 1]

    # Distances to radial cylinders.
    s_r_in = _distance_to_radial_boundary_rz(r, mu_perp, eta, r_in) if r_in > 0.0 else 1e30
    s_r_out = _distance_to_radial_boundary_rz(r, mu_perp, eta, r_out)

    # Distances to z-planes.
    if eta > 1e-15:
        s_z_lo = 1e30
        s_z_hi = (z_hi - z) / eta
    elif eta < -1e-15:
        s_z_lo = (z_lo - z) / eta
        s_z_hi = 1e30
    else:
        s_z_lo = 1e30
        s_z_hi = 1e30

    if sigma_s > 1e-12:
        s_scatter = -math.log(1.0 - random.random()) / sigma_s
    else:
        s_scatter = 1e30

    # Same high-opacity cap used in 1D transport kernels.
    cell_len = min(r_out - r_in, z_hi - z_lo)
    if sigma_s * cell_len > 10000.0:
        s_scatter = -math.log(1.0 - random.random()) / (10000.0 / cell_len)

    s_boundary = min(s_r_in, s_r_out, s_z_lo, s_z_hi)
    s_event = min(s_boundary, s_scatter, distance_to_census)
    assert s_event > 0.0, "Non-positive distance in rz transport"

    # Streaming update (Box 9.3).
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

    crossed = None
    if abs(s_event - s_boundary) < 1e-11:
        if s_boundary == s_r_in:
            r_new = r_in * (1.0 - 1e-12) if r_in > 0.0 else 1e-12
            crossed = "r-"
        elif s_boundary == s_r_out:
            r_new = r_out * (1.0 + 1e-12)
            crossed = "r+"
        elif s_boundary == s_z_lo:
            z_new = z_lo
            crossed = "z-"
        else:
            z_new = z_hi
            crossed = "z+"
    elif s_event == s_scatter:
        mu_s, eta_s = _sample_isotropic_rz(1)
        mu_new = float(mu_s[0])
        eta_new = float(eta_s[0])

    distance_to_census -= s_event

    weight_factor = math.exp(-sigma_a * s_event)
    if sigma_a > 1e-12:
        deposited_intensity = weight * (1.0 - weight_factor) / sigma_a
    else:
        deposited_intensity = weight * s_event
    deposited_weight = deposited_intensity * sigma_a
    weight *= weight_factor

    return (
        weight,
        mu_new,
        eta_new,
        r_new,
        z_new,
        crossed,
        deposited_weight,
        deposited_intensity,
        distance_to_census,
    )


@jit(nopython=True, cache=True)
def _sample_isotropic_xy_single():
    """3D isotropic direction projected to (ux, uy); uz discarded (z-symmetry)."""
    uz  = 2.0 * random.random() - 1.0
    phi = 2.0 * math.pi * random.random()
    r_xy = math.sqrt(max(0.0, 1.0 - uz * uz))
    return r_xy * math.cos(phi), r_xy * math.sin(phi)


@jit(nopython=True, cache=True)
def _sample_isotropic_rz_single():
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
def _move_particle_xy_jit(weight, ux, uy, x, y, i, j, x_edges, y_edges, sigma_a, sigma_s, distance_to_census):
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

    if sigma_s > 1e-12:
        s_scatter = -math.log(1.0 - random.random()) / sigma_s
    else:
        s_scatter = 1e30

    cell_len = min(x_r - x_l, y_r - y_l)
    if sigma_s * cell_len > 10000.0:
        s_scatter = -math.log(1.0 - random.random()) / (10000.0 / cell_len)

    s_boundary = sx if sx < sy else sy
    s_event = s_boundary
    if s_scatter < s_event:
        s_event = s_scatter
    if distance_to_census < s_event:
        s_event = distance_to_census

    if s_event <= 0.0:
        return weight, ux, uy, x, y, _CROSS_NONE, 0.0, 0.0, 0.0, _EVT_CENSUS

    x_new = x + ux * s_event
    y_new = y + uy * s_event

    crossed = _CROSS_NONE
    evt = _EVT_CENSUS
    if abs(s_event - s_boundary) < 1e-11:
        evt = _EVT_BOUNDARY
        if sx <= sy:
            if ux > 0.0:
                x_new = x_r
                crossed = _CROSS_I_PLUS
            else:
                x_new = x_l
                crossed = _CROSS_I_MINUS
        else:
            if uy > 0.0:
                y_new = y_r
                crossed = _CROSS_J_PLUS
            else:
                y_new = y_l
                crossed = _CROSS_J_MINUS
    elif s_event == s_scatter:
        evt = _EVT_SCATTER
        ux, uy = _sample_isotropic_xy_single()

    distance_to_census -= s_event

    weight_factor = math.exp(-sigma_a * s_event)
    if sigma_a > 1e-12:
        deposited_intensity = weight * (1.0 - weight_factor) / sigma_a
    else:
        deposited_intensity = weight * s_event
    deposited_weight = deposited_intensity * sigma_a
    weight *= weight_factor

    return (
        weight,
        ux,
        uy,
        x_new,
        y_new,
        crossed,
        deposited_weight,
        deposited_intensity,
        distance_to_census,
        evt,
    )


@jit(nopython=True, cache=True)
def _move_particle_rz_jit(weight, mu_perp, eta, r, z, i, j, r_edges, z_edges, sigma_a, sigma_s, distance_to_census):
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

    if sigma_s > 1e-12:
        s_scatter = -math.log(1.0 - random.random()) / sigma_s
    else:
        s_scatter = 1e30

    cell_len = min(r_out - r_in, z_hi - z_lo)
    if sigma_s * cell_len > 10000.0:
        s_scatter = -math.log(1.0 - random.random()) / (10000.0 / cell_len)

    s_boundary = s_r_in
    if s_r_out < s_boundary:
        s_boundary = s_r_out
    if s_z_lo < s_boundary:
        s_boundary = s_z_lo
    if s_z_hi < s_boundary:
        s_boundary = s_z_hi

    s_event = s_boundary
    if s_scatter < s_event:
        s_event = s_scatter
    if distance_to_census < s_event:
        s_event = distance_to_census

    if s_event <= 0.0:
        return weight, mu_perp, eta, r, z, _CROSS_NONE, 0.0, 0.0, 0.0, _EVT_CENSUS

    one_minus_eta2 = 1.0 - eta * eta
    if one_minus_eta2 < 0.0:
        one_minus_eta2 = 0.0
    l = s_event * math.sqrt(one_minus_eta2)

    r_new2 = r * r + 2.0 * r * mu_perp * l + l * l
    if r_new2 < 0.0:
        r_new2 = 0.0
    r_new = math.sqrt(r_new2)
    z_new = z + s_event * eta

    if r_new > 1e-15 and l > 0.0:
        mu_new = (r * mu_perp + l) / r_new
        if mu_new > 1.0:
            mu_new = 1.0
        elif mu_new < -1.0:
            mu_new = -1.0
    else:
        mu_new = mu_perp
    eta_new = eta

    crossed = _CROSS_NONE
    evt = _EVT_CENSUS
    if abs(s_event - s_boundary) < 1e-11:
        evt = _EVT_BOUNDARY
        if s_boundary == s_r_in:
            # Nudge just inside new cell to prevent near-tangential ping-pong.
            r_new = r_in * (1.0 - 1e-12) if r_in > 0.0 else 1e-12
            crossed = _CROSS_I_MINUS
        elif s_boundary == s_r_out:
            r_new = r_out * (1.0 + 1e-12)
            crossed = _CROSS_I_PLUS
        elif s_boundary == s_z_lo:
            z_new = z_lo
            crossed = _CROSS_J_MINUS
        else:
            z_new = z_hi
            crossed = _CROSS_J_PLUS
    elif s_event == s_scatter:
        evt = _EVT_SCATTER
        mu_new, eta_new = _sample_isotropic_rz_single()

    distance_to_census -= s_event

    weight_factor = math.exp(-sigma_a * s_event)
    if sigma_a > 1e-12:
        deposited_intensity = weight * (1.0 - weight_factor) / sigma_a
    else:
        deposited_intensity = weight * s_event
    deposited_weight = deposited_intensity * sigma_a
    weight *= weight_factor

    return (
        weight,
        mu_new,
        eta_new,
        r_new,
        z_new,
        crossed,
        deposited_weight,
        deposited_intensity,
        distance_to_census,
        evt,
    )


@jit(nopython=True, parallel=True)
def _transport_particles_2d(
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
    sigma_s,
    volumes,
    dt,
    reflect,
    max_events_per_particle,
    geometry_code,
    weight_floor,
):
    n = len(weights)
    nx, ny = sigma_a.shape
    n_threads = get_num_threads()
    dep_threads = np.zeros((n_threads, nx, ny))
    si_threads  = np.zeros((n_threads, nx, ny))
    bl_threads  = np.zeros(n_threads)
    stats_threads = np.zeros((n_threads, 8), dtype=np.int64)

    for k in prange(n):
        tid = get_thread_id()
        dcen = (dt - times[k]) * __c
        events = 0
        while dcen > 0.0:
            events += 1
            stats_threads[tid, 0] += 1
            if events > max_events_per_particle:
                stats_threads[tid, 7] += 1
                dcen = 0.0
                break

            i = int(cell_i[k])
            j = int(cell_j[k])
            if i < 0 or i >= nx or j < 0 or j >= ny:
                break

            sa = sigma_a[i, j]
            ss = sigma_s[i, j]

            if geometry_code == _GEOM_XY:
                out = _move_particle_xy_jit(
                    weights[k],
                    dir1[k],
                    dir2[k],
                    pos1[k],
                    pos2[k],
                    i,
                    j,
                    edges1,
                    edges2,
                    sa,
                    ss,
                    dcen,
                )
            else:
                out = _move_particle_rz_jit(
                    weights[k],
                    dir1[k],
                    dir2[k],
                    pos1[k],
                    pos2[k],
                    i,
                    j,
                    edges1,
                    edges2,
                    sa,
                    ss,
                    dcen,
                )

            (
                w_new,
                d1_new,
                d2_new,
                p1_new,
                p2_new,
                crossed,
                dep_w,
                dep_i,
                dcen,
                evt,
            ) = out

            if evt == _EVT_BOUNDARY:
                stats_threads[tid, 1] += 1
            elif evt == _EVT_SCATTER:
                stats_threads[tid, 2] += 1
            else:
                stats_threads[tid, 3] += 1

            vol = volumes[i, j]
            dep_threads[tid, i, j] += dep_w / vol
            si_threads[tid, i, j]  += dep_i / (dt * vol)

            weights[k] = w_new
            if w_new <= weight_floor:
                # Weight dropped to/below floor (fully absorbed or negligible);
                # deposit remainder and stop.
                stats_threads[tid, 5] += 1
                break
            dir1[k] = d1_new
            dir2[k] = d2_new
            pos1[k] = p1_new
            pos2[k] = p2_new

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

            if weights[k] <= 1e-300:
                weights[k] = 0.0
                dcen = 0.0

    return dep_threads.sum(axis=0), si_threads.sum(axis=0), bl_threads.sum(), stats_threads.sum(axis=0)


def _equilibrium_sample_xy(N, Tr, x_edges, y_edges):
    nx, ny = _shape_from_edges(x_edges, y_edges)
    volumes = _cell_volumes_xy(x_edges, y_edges)
    E_cell = __a * Tr**4 * volumes
    E_tot = np.sum(E_cell)
    counts = np.maximum(1, np.ceil(N * E_cell / max(E_tot, 1e-300)).astype(int))

    n_total = int(np.sum(counts))
    weights = np.empty(n_total)
    ux = np.empty(n_total)
    uy = np.empty(n_total)
    times = np.zeros(n_total)
    x = np.empty(n_total)
    y = np.empty(n_total)
    ci = np.empty(n_total, dtype=int)
    cj = np.empty(n_total, dtype=int)

    k = 0
    for i in range(nx):
        for j in range(ny):
            n = int(counts[i, j])
            if n <= 0:
                continue
            w = E_cell[i, j] / n
            x0, x1 = x_edges[i], x_edges[i + 1]
            y0, y1 = y_edges[j], y_edges[j + 1]
            weights[k : k + n] = w
            x[k : k + n] = np.random.uniform(x0, x1, n)
            y[k : k + n] = np.random.uniform(y0, y1, n)
            ci[k : k + n] = i
            cj[k : k + n] = j
            u1, u2 = _sample_isotropic_xy(n)
            ux[k : k + n] = u1
            uy[k : k + n] = u2
            k += n

    return weights, ux, uy, times, x, y, ci, cj


def _equilibrium_sample_rz(N, Tr, r_edges, z_edges):
    nx, ny = _shape_from_edges(r_edges, z_edges)
    volumes = _cell_volumes_rz(r_edges, z_edges)
    E_cell = __a * Tr**4 * volumes
    E_tot = np.sum(E_cell)
    counts = np.maximum(1, np.ceil(N * E_cell / max(E_tot, 1e-300)).astype(int))

    n_total = int(np.sum(counts))
    weights = np.empty(n_total)
    mu = np.empty(n_total)
    eta = np.empty(n_total)
    times = np.zeros(n_total)
    r = np.empty(n_total)
    z = np.empty(n_total)
    ci = np.empty(n_total, dtype=int)
    cj = np.empty(n_total, dtype=int)

    k = 0
    for i in range(nx):
        for j in range(ny):
            n = int(counts[i, j])
            if n <= 0:
                continue
            w = E_cell[i, j] / n
            r0, r1 = r_edges[i], r_edges[i + 1]
            z0, z1 = z_edges[j], z_edges[j + 1]
            xi_r = np.random.uniform(0.0, 1.0, n)
            r_s = np.sqrt(r0 * r0 + (r1 * r1 - r0 * r0) * xi_r)
            z_s = np.random.uniform(z0, z1, n)
            weights[k : k + n] = w
            r[k : k + n] = r_s
            z[k : k + n] = z_s
            ci[k : k + n] = i
            cj[k : k + n] = j
            m, e = _sample_isotropic_rz(n)
            mu[k : k + n] = m
            eta[k : k + n] = e
            k += n

    return weights, mu, eta, times, r, z, ci, cj


def _sample_source_xy(N, source, dt, x_edges, y_edges):
    nx, ny = _shape_from_edges(x_edges, y_edges)
    volumes = _cell_volumes_xy(x_edges, y_edges)
    QV = source * dt * volumes
    Qtot = float(np.sum(QV))
    if Qtot <= 0.0 or N <= 0:
        return (
            np.zeros(0),
            np.zeros(0),
            np.zeros(0),
            np.zeros(0),
            np.zeros(0),
            np.zeros(0, dtype=int),
            np.zeros(0, dtype=int),
        )

    counts = np.ceil(N * QV / Qtot).astype(int)
    n_total = int(np.sum(counts))

    weights = np.empty(n_total)
    ux = np.empty(n_total)
    uy = np.empty(n_total)
    times = np.empty(n_total)
    x = np.empty(n_total)
    y = np.empty(n_total)
    ci = np.empty(n_total, dtype=int)
    cj = np.empty(n_total, dtype=int)

    k = 0
    for i in range(nx):
        for j in range(ny):
            n = int(counts[i, j])
            if n <= 0:
                continue
            x0, x1 = x_edges[i], x_edges[i + 1]
            y0, y1 = y_edges[j], y_edges[j + 1]
            weights[k : k + n] = QV[i, j] / n
            times[k : k + n] = np.random.uniform(0.0, dt, n)
            x[k : k + n] = np.random.uniform(x0, x1, n)
            y[k : k + n] = np.random.uniform(y0, y1, n)
            ci[k : k + n] = i
            cj[k : k + n] = j
            u1, u2 = _sample_isotropic_xy(n)
            ux[k : k + n] = u1
            uy[k : k + n] = u2
            k += n

    return weights, ux, uy, times, x, y, ci, cj


def _sample_source_rz_uniform(N, source, dt, r_edges, z_edges):
    nx, ny = _shape_from_edges(r_edges, z_edges)
    volumes = _cell_volumes_rz(r_edges, z_edges)
    QV = source * dt * volumes
    Qtot = float(np.sum(QV))
    if Qtot <= 0.0 or N <= 0:
        return (
            np.zeros(0),
            np.zeros(0),
            np.zeros(0),
            np.zeros(0),
            np.zeros(0),
            np.zeros(0),
            np.zeros(0, dtype=int),
            np.zeros(0, dtype=int),
        )

    counts = np.ceil(N * QV / Qtot).astype(int)
    n_total = int(np.sum(counts))

    weights = np.empty(n_total)
    mu = np.empty(n_total)
    eta = np.empty(n_total)
    times = np.empty(n_total)
    r = np.empty(n_total)
    z = np.empty(n_total)
    ci = np.empty(n_total, dtype=int)
    cj = np.empty(n_total, dtype=int)

    k = 0
    for i in range(nx):
        for j in range(ny):
            n = int(counts[i, j])
            if n <= 0:
                continue
            r0, r1 = r_edges[i], r_edges[i + 1]
            z0, z1 = z_edges[j], z_edges[j + 1]
            xi_r = np.random.uniform(0.0, 1.0, n)
            weights[k : k + n] = QV[i, j] / n
            times[k : k + n] = np.random.uniform(0.0, dt, n)
            r[k : k + n] = np.sqrt(r0 * r0 + (r1 * r1 - r0 * r0) * xi_r)
            z[k : k + n] = np.random.uniform(z0, z1, n)
            ci[k : k + n] = i
            cj[k : k + n] = j
            m, e = _sample_isotropic_rz(n)
            mu[k : k + n] = m
            eta[k : k + n] = e
            k += n

    return weights, mu, eta, times, r, z, ci, cj


def _solve_r_linear_cdf(r0, r1, a, M, xi):
    """Invert Box 9.4 linear-in-r CDF for cylindrical shell sourcing."""
    C = (a * 0.5 * (r1 * r1 - r0 * r0) + M / 3.0 * (r1**3 - r0**3))
    if C <= 0.0:
        return math.sqrt(r0 * r0 + (r1 * r1 - r0 * r0) * xi)

    rhs = xi * C + a * 0.5 * (r0 * r0) + M / 3.0 * (r0**3)

    lo = r0
    hi = r1
    r = math.sqrt(r0 * r0 + (r1 * r1 - r0 * r0) * xi)
    for _ in range(50):
        g = a * 0.5 * r * r + M / 3.0 * (r**3) - rhs
        if abs(g) < 1e-14 * (abs(rhs) + 1e-30):
            break
        if g < 0.0:
            lo = r
        else:
            hi = r
        dg = a * r + M * r * r
        if abs(dg) > 1e-30:
            r_new = r - g / dg
        else:
            r_new = 0.5 * (lo + hi)
        if r_new <= lo or r_new >= hi:
            r_new = 0.5 * (lo + hi)
        if abs(r_new - r) < 1e-14 * (r1 - r0 + 1e-30):
            r = r_new
            break
        r = r_new
    return r


def _sample_source_rz_linear(N, source, T, dt, r_edges, z_edges):
    """Cylindrical source sampling with Box 9.4 linear-in-r and linear-in-z tilt.

    source is used for total cell power; T provides local tilt reconstruction.
    """
    nx, ny = _shape_from_edges(r_edges, z_edges)
    volumes = _cell_volumes_rz(r_edges, z_edges)
    QV = source * dt * volumes
    Qtot = float(np.sum(QV))
    if Qtot <= 0.0 or N <= 0:
        return (
            np.zeros(0),
            np.zeros(0),
            np.zeros(0),
            np.zeros(0),
            np.zeros(0),
            np.zeros(0),
            np.zeros(0, dtype=int),
            np.zeros(0, dtype=int),
        )

    counts = np.ceil(N * QV / Qtot).astype(int)
    n_total = int(np.sum(counts))

    # Reconstruct T^4 tilt from neighboring cell centers.
    T4 = np.maximum(T, 0.0) ** 4
    r_cent = 0.5 * (r_edges[:-1] + r_edges[1:])
    z_cent = 0.5 * (z_edges[:-1] + z_edges[1:])

    M_r = np.zeros_like(T4)
    M_z = np.zeros_like(T4)

    for i in range(nx):
        for j in range(ny):
            if nx == 1:
                M_r[i, j] = 0.0
            elif i == 0:
                M_r[i, j] = (T4[i + 1, j] - T4[i, j]) / (r_cent[i + 1] - r_cent[i])
            elif i == nx - 1:
                M_r[i, j] = (T4[i, j] - T4[i - 1, j]) / (r_cent[i] - r_cent[i - 1])
            else:
                M_r[i, j] = (T4[i + 1, j] - T4[i - 1, j]) / (r_cent[i + 1] - r_cent[i - 1])

            if ny == 1:
                M_z[i, j] = 0.0
            elif j == 0:
                M_z[i, j] = (T4[i, j + 1] - T4[i, j]) / (z_cent[j + 1] - z_cent[j])
            elif j == ny - 1:
                M_z[i, j] = (T4[i, j] - T4[i, j - 1]) / (z_cent[j] - z_cent[j - 1])
            else:
                M_z[i, j] = (T4[i, j + 1] - T4[i, j - 1]) / (z_cent[j + 1] - z_cent[j - 1])

    weights = np.empty(n_total)
    mu = np.empty(n_total)
    eta = np.empty(n_total)
    times = np.empty(n_total)
    r = np.empty(n_total)
    z = np.empty(n_total)
    ci = np.empty(n_total, dtype=int)
    cj = np.empty(n_total, dtype=int)

    k = 0
    for i in range(nx):
        for j in range(ny):
            n = int(counts[i, j])
            if n <= 0:
                continue

            r0, r1 = r_edges[i], r_edges[i + 1]
            z0, z1 = z_edges[j], z_edges[j + 1]
            dr = r1 - r0
            dz = z1 - z0

            # Linear-in-r reconstruction: T4(r) = a_r + M_r*(r-r0)
            a_r = T4[i, j] - M_r[i, j] * (r_cent[i] - r0)
            t4_r0 = a_r
            t4_r1 = a_r + M_r[i, j] * dr
            if t4_r0 < 0.0 or t4_r1 < 0.0:
                a_r = max(T4[i, j], 0.0)
                Mloc_r = 0.0
            else:
                Mloc_r = M_r[i, j]

            # Linear-in-z branch probability (Box 9.4 Eq. 9.83 style).
            a_z = T4[i, j] - M_z[i, j] * (z_cent[j] - z0)
            t4_z0 = max(a_z, 0.0)
            t4_z1 = max(a_z + M_z[i, j] * dz, 0.0)
            denom = t4_z0 + t4_z1
            if denom > 0.0:
                Pn = t4_z0 / denom
            else:
                Pn = 0.5

            xi_branch = np.random.uniform(0.0, 1.0, n)
            xi_shape = np.random.uniform(0.0, 1.0, n)

            r_s = np.empty(n)
            for q in range(n):
                xi_r = random.random()
                r_s[q] = _solve_r_linear_cdf(r0, r1, a_r, Mloc_r, xi_r)

            z_s = np.empty(n)
            for q in range(n):
                if xi_branch[q] <= Pn:
                    z_s[q] = z1 - dz * math.sqrt(xi_shape[q])
                else:
                    z_s[q] = z0 + dz * math.sqrt(xi_shape[q])

            weights[k : k + n] = QV[i, j] / n
            times[k : k + n] = np.random.uniform(0.0, dt, n)
            r[k : k + n] = r_s
            z[k : k + n] = z_s
            ci[k : k + n] = i
            cj[k : k + n] = j
            m, e = _sample_isotropic_rz(n)
            mu[k : k + n] = m
            eta[k : k + n] = e
            k += n

    return weights, mu, eta, times, r, z, ci, cj


def _comb(weights, cell_i, cell_j, dir1, dir2, times, pos1, pos2, Nmax, nx, ny):
    """Per-cell stochastic comb to cap total particle count."""
    alive = weights > 0.0
    weights = weights[alive]
    cell_i = cell_i[alive]
    cell_j = cell_j[alive]
    dir1 = dir1[alive]
    dir2 = dir2[alive]
    times = times[alive]
    pos1 = pos1[alive]
    pos2 = pos2[alive]

    n_cells = nx * ny
    flat = _flatten_index(cell_i, cell_j, nx)
    ecen = np.bincount(flat, weights=weights, minlength=n_cells)
    E = np.sum(ecen)
    if E <= 0.0:
        return weights, cell_i, cell_j, dir1, dir2, times, pos1, pos2, np.zeros(n_cells)

    desired = np.where(ecen > 0.0, np.maximum(1, np.round(Nmax * ecen / E).astype(int)), 0)
    ew = np.where(desired > 0, ecen / desired, 0.0)

    nw = []
    ni = []
    nj = []
    nd1 = []
    nd2 = []
    nt = []
    np1 = []
    np2 = []

    for k in range(len(weights)):
        c = int(flat[k])
        w_target = ew[c]
        if w_target <= 0.0:
            continue
        n = int(weights[k] / w_target + random.random())
        i, j = _unflatten_index(c, nx)
        for _ in range(n):
            nw.append(w_target)
            ni.append(i)
            nj.append(j)
            nd1.append(dir1[k])
            nd2.append(dir2[k])
            nt.append(times[k])
            np1.append(pos1[k])
            np2.append(pos2[k])

    nw = np.array(nw)
    ni = np.array(ni, dtype=int)
    nj = np.array(nj, dtype=int)
    nd1 = np.array(nd1)
    nd2 = np.array(nd2)
    nt = np.array(nt)
    np1 = np.array(np1)
    np2 = np.array(np2)

    if len(nw) > 0:
        flat_new = _flatten_index(ni, nj, nx)
        ecen_after = np.bincount(flat_new, weights=nw, minlength=n_cells)
    else:
        ecen_after = np.zeros(n_cells)

    return nw, ni, nj, nd1, nd2, nt, np1, np2, ecen - ecen_after


def init_simulation(
    Ntarget,
    Tinit,
    Tr_init,
    edges1,
    edges2,
    eos,
    inv_eos,
    Ntarget_ic=None,
    geometry="xy",
):
    """Initialize particle arrays and material state for 2D IMC."""
    nx, ny = _shape_from_edges(edges1, edges2)
    volumes = _cell_volumes(edges1, edges2, geometry)

    internal_energy = eos(Tinit)
    temperature = Tinit.copy()
    assert np.allclose(inv_eos(internal_energy), Tinit), "Inverse EOS failed"

    N_ic = Ntarget if Ntarget_ic is None else Ntarget_ic
    if geometry == "xy":
        p = _equilibrium_sample_xy(N_ic, Tr_init, edges1, edges2)
    elif geometry == "rz":
        p = _equilibrium_sample_rz(N_ic, Tr_init, edges1, edges2)
    else:
        raise ValueError(f"Unknown geometry: {geometry}")

    weights, dir1, dir2, times, pos1, pos2, cell_i, cell_j = p

    flat = _flatten_index(cell_i, cell_j, nx)
    rad_cell = np.bincount(flat, weights=weights, minlength=nx * ny).reshape(nx, ny)
    radiation_temperature = (rad_cell / volumes / __a) ** 0.25

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

    return SimulationState2D(
        weights=weights,
        dir1=dir1,
        dir2=dir2,
        times=times,
        pos1=pos1,
        pos2=pos2,
        cell_i=cell_i,
        cell_j=cell_j,
        internal_energy=internal_energy,
        temperature=temperature,
        radiation_temperature=radiation_temperature,
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
    _timing=False,
):
    """Advance one 2D IMC step for geometry='xy' or 'rz'.

    T_boundary ordering is always:
      (left/min-1, right/max-1, bottom/min-2, top/max-2)
    For rz this maps to:
      (rmin, rmax, zmin, zmax)
    
    boundary_source_func: optional callable(r_or_x, z_or_y, side) -> float
        For 'rz' geometry only. Returns boundary temperature (keV) at position.
        Called with (r, z, side) where side is 'rmin', 'rmax', 'zmin', or 'zmax'.
        Return value > 0.0 to emit at that temperature, 0.0 for no emission.
        Enables position-dependent boundary temperatures (e.g., spatially-varying sources).
        When provided, overrides T_boundary for that boundary side.
    """
    nx, ny = _shape_from_edges(edges1, edges2)
    volumes = _cell_volumes(edges1, edges2, geometry)

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

    t_step_start = _time.perf_counter()

    _t0 = _time.perf_counter() if _timing else 0.0
    sigma_a_true = sigma_a_func(temperature)
    beta = 4.0 * __a * temperature**3 / cv(temperature)
    f = 1.0 / (1.0 + theta * beta * sigma_a_true * __c * dt)
    f = np.clip(f, 0.0, 1.0)
    sigma_s = sigma_a_true * (1.0 - f)
    sigma_a = sigma_a_true * f

    # Boundary injection.
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
                s = _sample_boundary_xy(Nboundary, side, Tb, dt, edges1, edges2, boundary_source_func)
                if s is None:
                    continue
                w, d1, d2, t, p1, p2 = s
                ci, cj = _locate_indices(p1, p2, edges1, edges2)
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
            for side, Tb in (
                ("rmin", b_left),
                ("rmax", b_right),
                ("zmin", b_bottom),
                ("zmax", b_top),
            ):
                s = _sample_boundary_rz(Nboundary, side, Tb, dt, edges1, edges2, boundary_source_func)
                if s is None:
                    continue
                w, d1, d2, t, p1, p2 = s
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
                    boundary_emission += float(np.sum(w[valid]))

    if _timing: print(f"  [step/{geometry}] boundary: {_time.perf_counter()-_t0:.3f}s  N={len(weights)}", flush=True); _t0 = _time.perf_counter()
    # Fixed source.
    source_emission = 0.0
    if Nsource > 0 and np.max(source) > 0.0:
        if geometry == "xy":
            s = _sample_source_xy(Nsource, source, dt, edges1, edges2)
        else:
            if rz_linear_source:
                s = _sample_source_rz_linear(Nsource, source, temperature, dt, edges1, edges2)
            else:
                s = _sample_source_rz_uniform(Nsource, source, dt, edges1, edges2)
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

    if _timing: print(f"  [step/{geometry}] fixed_source: {_time.perf_counter()-_t0:.3f}s  N={len(weights)}", flush=True); _t0 = _time.perf_counter()
    # Internal emission proportional to sigma_a * a c T^4 * dt * V.
    emitted_energies = __a * __c * np.maximum(temperature, 0.0) ** 4 * sigma_a * dt * volumes
    E_emit = float(np.sum(emitted_energies))
    if E_emit > 0.0 and Ntarget > 0:
        if geometry == "xy":
            p = _sample_source_xy(Ntarget, emitted_energies / (dt * volumes + 1e-300), dt, edges1, edges2)
        else:
            if rz_linear_source:
                p = _sample_source_rz_linear(
                    Ntarget,
                    emitted_energies / (dt * volumes + 1e-300),
                    temperature,
                    dt,
                    edges1,
                    edges2,
                )
            else:
                p = _sample_source_rz_uniform(
                    Ntarget,
                    emitted_energies / (dt * volumes + 1e-300),
                    dt,
                    edges1,
                    edges2,
                )
        w, d1, d2, t, p1, p2, ci, cj = p
        if len(w) > 0:
            weights = np.concatenate((weights, w))
            dir1 = np.concatenate((dir1, d1))
            dir2 = np.concatenate((dir2, d2))
            times = np.concatenate((times, t))
            pos1 = np.concatenate((pos1, p1))
            pos2 = np.concatenate((pos2, p2))
            cell_i = np.concatenate((cell_i, ci))
            cell_j = np.concatenate((cell_j, cj))

    if _timing: print(f"  [step/{geometry}] emission_sample: {_time.perf_counter()-_t0:.3f}s  N={len(weights)}", flush=True); _t0 = _time.perf_counter()
    # Transport particles with JIT-compiled kernel.
    t_transport_start = _time.perf_counter()
    n_particles_transported = len(weights)
    geometry_code = _GEOM_XY if geometry == "xy" else _GEOM_RZ
    weight_floor = 1e-10 * float(np.sum(weights)) / max(len(weights), 1)
    
    # Report thread count on first step
    if state.count == 0:
        print(f"[IMC2D FC] Using {get_num_threads()} threads for transport")
    
    dep_cell, si_cell, boundary_loss, stats = _transport_particles_2d(
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
        sigma_s,
        volumes,
        dt,
        reflect,
        max_events_per_particle,
        geometry_code,
        weight_floor,
    )
    t_post_start = _time.perf_counter()

    if _timing: print(f"  [step/{geometry}] transport: {_time.perf_counter()-_t0:.3f}s", flush=True); _t0 = _time.perf_counter()
    # Material/radiation update.
    internal_energy = internal_energy + dep_cell - emitted_energies / volumes
    temperature = inv_eos(internal_energy)

    if use_scalar_intensity_Tr:
        radiation_temperature = (si_cell / __a / __c) ** 0.25
    else:
        valid = (weights > 0.0)
        flat = _flatten_index(cell_i[valid], cell_j[valid], nx)
        rad_cell = np.bincount(flat, weights=weights[valid], minlength=nx * ny).reshape(nx, ny)
        radiation_temperature = (rad_cell / volumes / __a) ** 0.25

    if _timing: print(f"  [step/{geometry}] material_update: {_time.perf_counter()-_t0:.3f}s", flush=True); _t0 = _time.perf_counter()
    # Combing.
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
    ) = _comb(weights, cell_i, cell_j, dir1, dir2, times, pos1, pos2, Nmax, nx, ny)

    if conserve_comb_energy:
        internal_energy = internal_energy + comb_disc.reshape(ny, nx).T / volumes
        temperature = inv_eos(internal_energy)

    if _timing: print(f"  [step/{geometry}] comb: {_time.perf_counter()-_t0:.3f}s  N_out={len(weights)}", flush=True); _t0 = _time.perf_counter()
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
    state.internal_energy = internal_energy
    state.temperature = temperature
    state.radiation_temperature = radiation_temperature
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
):
    """Run full 2D IMC simulation and return time/radiation/material histories."""
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
