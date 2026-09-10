"""Local multigroup Kompaneets frequency redistribution.

The update evolves group photon numbers with an exponential-fit finite-volume
flux on an arbitrary energy grid.  Zero endpoint fluxes conserve photon
number, and the material receives the equal-and-opposite radiation-energy
change.  The default linear operator omits induced scattering and preserves
Wien equilibrium; an optional nonlinear occupation correction retains the
stimulated term.
"""

import numpy as np
from scipy.optimize import brentq


ELECTRON_REST_KEV = 510.99895


def _bernoulli(value):
    """Return ``value / expm1(value)`` without cancellation or overflow."""
    value = np.asarray(value, dtype=float)
    result = np.empty_like(value)
    small = np.abs(value) < 1.0e-5
    x = value[small]
    result[small] = 1.0 - 0.5 * x + x * x / 12.0 - x**4 / 720.0
    positive = (~small) & (value > 50.0)
    result[positive] = value[positive] * np.exp(-value[positive])
    remaining = ~(small | positive)
    result[remaining] = value[remaining] / np.expm1(value[remaining])
    return result


def _solve_tridiagonal_batched(lower, diagonal, upper, rhs):
    """Solve independent tridiagonal group systems for every spatial point."""
    lower = np.asarray(lower, dtype=float).copy()
    diagonal = np.asarray(diagonal, dtype=float).copy()
    upper = np.asarray(upper, dtype=float).copy()
    solution = np.asarray(rhs, dtype=float).copy()
    groups = diagonal.shape[0]
    for group in range(1, groups):
        multiplier = lower[group] / diagonal[group - 1]
        diagonal[group] -= multiplier * upper[group - 1]
        solution[group] -= multiplier * solution[group - 1]
    solution[-1] /= diagonal[-1]
    for group in range(groups - 2, -1, -1):
        solution[group] = (
            solution[group] - upper[group] * solution[group + 1]
        ) / diagonal[group]
    return solution


def _opacity_field(thomson_opacity, temperature, spatial_shape):
    values = thomson_opacity(temperature) if callable(thomson_opacity) else thomson_opacity
    values = np.asarray(values, dtype=float)
    if values.ndim == 0:
        values = np.full(spatial_shape, float(values))
    else:
        values = np.broadcast_to(values, spatial_shape)
    if np.any(values < 0.0) or not np.all(np.isfinite(values)):
        raise ValueError("Thomson opacity must be finite and nonnegative")
    return values


def _densmore_linear_number_update(
    number_star, temperature, dt, c_light, energy_edges_kev,
    group_mean_energy_kev, thomson_opacity, electron_rest_energy_kev,
    occupation=None,
):
    """Apply the Densmore flux, optionally with an induced-scattering drift."""
    groups = number_star.shape[0]
    spatial_shape = number_star.shape[1:]
    cells = int(np.prod(spatial_shape))
    temperature_flat = np.maximum(
        np.asarray(temperature, dtype=float), 1.0e-12,
    ).reshape(cells)
    opacity_flat = _opacity_field(
        thomson_opacity, temperature, spatial_shape,
    ).reshape(cells)
    widths = np.diff(energy_edges_kev)
    centers = np.asarray(group_mean_energy_kev, dtype=float)
    faces = np.asarray(energy_edges_kev[1:-1], dtype=float)
    delta = (centers[1:, None] - centers[:-1, None]) / temperature_flat[None, :]
    potential_jump = delta
    if occupation is not None:
        occupation_flat = np.asarray(occupation, dtype=float).reshape(groups, cells)
        potential_jump = potential_jump + (
            np.log1p(occupation_flat[:-1])
            - np.log1p(occupation_flat[1:])
        )
    face_scale = faces[:, None]**4 / electron_rest_energy_kev
    coefficient_left = (
        -face_scale * _bernoulli(potential_jump)
        / (centers[:-1, None]**3 * delta)
    )
    coefficient_right = (
        face_scale * _bernoulli(-potential_jump)
        / (centers[1:, None]**3 * delta)
    )

    rate = c_light * opacity_flat[None, :] * centers[:, None] / widths[:, None]
    operator_lower = np.zeros((groups, cells))
    operator_diagonal = np.zeros((groups, cells))
    operator_upper = np.zeros((groups, cells))
    operator_diagonal[:-1] += rate[:-1] * coefficient_left
    operator_upper[:-1] += rate[:-1] * coefficient_right
    operator_lower[1:] -= rate[1:] * coefficient_left
    operator_diagonal[1:] -= rate[1:] * coefficient_right

    lower = -dt * operator_lower
    diagonal = 1.0 - dt * operator_diagonal
    upper = -dt * operator_upper
    energy_density_star = (
        number_star.reshape(groups, cells)
        * centers[:, None] / widths[:, None]
    )
    energy_density_new = _solve_tridiagonal_batched(
        lower, diagonal, upper, energy_density_star,
    )
    number_new = (
        energy_density_new * widths[:, None] / centers[:, None]
    )
    minimum = float(np.min(number_new))
    if minimum < -1.0e-12 or not np.all(np.isfinite(number_new)):
        raise RuntimeError(
            "Densmore Kompaneets update produced a non-finite or negative "
            "group photon number"
        )
    return np.maximum(number_new, 0.0).reshape(number_star.shape)


def _linear_number_update(
    number_star, temperature, dt, c_light, energy_edges_kev,
    group_mean_energy_kev, thomson_opacity, electron_rest_energy_kev,
    occupation=None,
):
    """Apply one fixed-temperature backward-Euler Kompaneets solve."""
    return _densmore_linear_number_update(
        number_star, temperature, dt, c_light, energy_edges_kev,
        group_mean_energy_kev, thomson_opacity, electron_rest_energy_kev,
        occupation=occupation,
    )


def densmore_implicit_kompaneets_update(
    phi_star,
    e_star,
    T_star,
    dt,
    c_light,
    invEOS,
    energy_edges_kev,
    group_mean_energy_kev,
    thomson_opacity,
    electron_rest_energy_kev=ELECTRON_REST_KEV,
    tolerance=1.0e-10,
    max_iterations=100,
):
    """Solve Densmore et al.'s linear implicit Kompaneets split.

    For each spatial point, Eq. (19) is a tridiagonal solve at a specified
    material temperature.  Substitution into the energy balance, Eq. (21),
    leaves one scalar equation for the end-of-step temperature.  This routine
    brackets and solves that scalar equation, rather than applying a Picard
    iteration.  It is restricted to the linear (no induced scattering) form
    used in the 2007 IMC method.
    """
    if dt <= 0.0 or tolerance <= 0.0 or max_iterations < 1:
        raise ValueError("dt, tolerance, and iteration count must be positive")
    groups = len(phi_star)
    edges = np.asarray(energy_edges_kev, dtype=float)
    mean_energy = np.asarray(group_mean_energy_kev, dtype=float)
    if edges.shape != (groups + 1,) or mean_energy.shape != (groups,):
        raise ValueError("energy grid does not match the scalar-intensity groups")
    phi_array = np.stack([np.asarray(value, dtype=float) for value in phi_star])
    spatial_shape = np.asarray(T_star).shape
    if phi_array.shape != (groups,) + spatial_shape:
        raise ValueError("all scalar intensities must match T_star.shape")
    if np.any(mean_energy <= 0.0) or np.any(np.diff(edges) <= 0.0):
        raise ValueError("energy grid and group energies must be positive")
    e_array = np.asarray(e_star, dtype=float)
    if e_array.shape != spatial_shape:
        raise ValueError("e_star and T_star must have the same shape")
    cells = int(np.prod(spatial_shape))
    phi_flat = phi_array.reshape(groups, cells)
    energy_flat = e_array.reshape(cells)
    temperature_flat = np.asarray(T_star, dtype=float).reshape(cells)
    opacity_flat = _opacity_field(
        thomson_opacity, np.asarray(T_star, dtype=float), spatial_shape,
    ).reshape(cells)
    number_star = phi_flat / (c_light * mean_energy[:, None])
    number_new = np.empty_like(number_star)
    energy_new = np.empty(cells)
    temperature_new = np.empty(cells)
    total_iterations = 0

    for cell in range(cells):
        number_column = number_star[:, cell:cell + 1]
        radiation_star = float(np.sum(phi_flat[:, cell]) / c_light)
        material_star = float(energy_flat[cell])
        # Pure Compton scattering cannot create photons.  In a 2-D IMC
        # calculation, most cells can therefore be skipped before the source
        # reaches them; this also avoids thousands of unnecessary scalar
        # bracketing solves in cold, unilluminated wall cells.
        if not np.any(number_column > 0.0):
            number_new[:, cell] = number_column[:, 0]
            energy_new[cell] = material_star
            temperature_new[cell] = temperature_flat[cell]
            continue
        evaluations = 0
        cell_opacity = opacity_flat[cell]

        def residual(candidate_temperature):
            nonlocal evaluations
            evaluations += 1
            candidate_number = _linear_number_update(
                number_column, np.array([candidate_temperature]), dt, c_light,
                edges, mean_energy, cell_opacity, electron_rest_energy_kev,
            )
            radiation_new = float(np.sum(candidate_number[:, 0] * mean_energy))
            candidate_energy = material_star + radiation_star - radiation_new
            recovered_temperature = float(np.asarray(
                invEOS(np.array(candidate_energy, ndmin=1))
            ).flat[0])
            if not np.isfinite(recovered_temperature):
                raise RuntimeError("inverse EOS returned a non-finite temperature")
            return candidate_temperature - recovered_temperature

        lower = 1.0e-12
        lower_value = residual(lower)
        if lower_value > 0.0:
            raise RuntimeError("could not bracket a positive Kompaneets temperature")
        upper = max(1.0, 2.0 * float(temperature_flat[cell]))
        upper_value = residual(upper)
        for _ in range(100):
            if upper_value >= 0.0:
                break
            upper *= 2.0
            upper_value = residual(upper)
        else:
            raise RuntimeError("could not bracket the implicit Kompaneets temperature")
        root = brentq(
            residual, lower, upper, xtol=tolerance,
            rtol=4.0 * np.finfo(float).eps, maxiter=max_iterations,
        )
        solved_number = _linear_number_update(
            number_column, np.array([root]), dt, c_light,
            edges, mean_energy, cell_opacity, electron_rest_energy_kev,
        )[:, 0]
        radiation_new = float(np.sum(solved_number * mean_energy))
        number_new[:, cell] = solved_number
        energy_new[cell] = material_star + radiation_star - radiation_new
        temperature_new[cell] = root
        total_iterations += evaluations

    phi_new_array = number_new * c_light * mean_energy[:, None]
    photon_number_error = float(np.max(np.abs(np.sum(number_new - number_star, axis=0))))
    total_before = energy_flat + np.sum(phi_flat, axis=0) / c_light
    total_after = energy_new + np.sum(phi_new_array, axis=0) / c_light
    energy_error = float(np.max(np.abs(total_after - total_before)))
    return (
        [phi_new_array[group].reshape(spatial_shape).copy() for group in range(groups)],
        energy_new.reshape(spatial_shape),
        temperature_new.reshape(spatial_shape),
        {"converged": True, "residual": 0.0,
         "iterations": total_iterations, "photon_number_error": photon_number_error,
         "energy_error": energy_error},
    )


def operator_split_kompaneets_update(
    phi_star,
    psi_star,
    e_star,
    T_star,
    dt,
    c_light,
    invEOS,
    energy_edges_kev,
    group_mean_energy_kev,
    thomson_opacity,
    induced_scattering=False,
    group_scalar_capacity=None,
    electron_rest_energy_kev=ELECTRON_REST_KEV,
    tolerance=1.0e-10,
    max_iterations=50,
    relaxation=1.0,
    on_nonconvergence="raise",
    return_diagnostics=False,
):
    """Apply a local Kompaneets update.

    The frequency solve is tridiagonal for a fixed material temperature.  A
    fixed-point iteration closes the implicit material-temperature dependence
    through exact local radiation-plus-material energy conservation.  When
    ``induced_scattering`` is true, it also iterates the occupation-dependent
    drift using ``group_scalar_capacity`` to form ``n_g=phi_g/Q_g``.
    """
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if not 0.0 < relaxation <= 1.0:
        raise ValueError("relaxation must lie in (0, 1]")
    if max_iterations < 1 or tolerance <= 0.0:
        raise ValueError("iteration count and tolerance must be positive")
    if on_nonconvergence not in {"raise", "accept"}:
        raise ValueError("on_nonconvergence must be 'raise' or 'accept'")

    groups = len(phi_star)
    edges = np.asarray(energy_edges_kev, dtype=float)
    mean_energy = np.asarray(group_mean_energy_kev, dtype=float)
    if edges.shape != (groups + 1,) or mean_energy.shape != (groups,):
        raise ValueError("energy edges and group mean energies do not match phi_star")
    if np.any(np.diff(edges) <= 0.0) or np.any(mean_energy <= 0.0):
        raise ValueError("energy grid and representative energies must be positive")
    if induced_scattering:
        scalar_capacity = np.asarray(group_scalar_capacity, dtype=float)
        if scalar_capacity.shape != (groups,) or np.any(scalar_capacity <= 0.0):
            raise ValueError(
                "stimulated Kompaneets requires positive group_scalar_capacity"
            )
        number_capacity = scalar_capacity / (c_light * mean_energy)
    else:
        number_capacity = None

    phi_star_array = np.stack([np.asarray(value, dtype=float) for value in phi_star])
    spatial_shape = np.asarray(T_star).shape
    if phi_star_array.shape != (groups,) + spatial_shape:
        raise ValueError("all scalar intensities must match T_star.shape")
    e_star = np.asarray(e_star, dtype=float)
    temperature_iter = np.asarray(T_star, dtype=float).copy()
    number_star = phi_star_array / (
        c_light * mean_energy.reshape((groups,) + (1,) * len(spatial_shape))
    )
    number_iter = number_star.copy()
    photon_number_before = np.sum(number_star, axis=0)
    converged = False
    change = np.inf

    for iteration in range(1, max_iterations + 1):
        occupation = None
        if induced_scattering:
            occupation = number_iter / number_capacity.reshape(
                (groups,) + (1,) * len(spatial_shape)
            )
        number_new = _linear_number_update(
            number_star, temperature_iter, dt, c_light, edges, mean_energy,
            thomson_opacity, electron_rest_energy_kev,
            occupation=occupation,
        )
        phi_new_array = number_new * (
            c_light * mean_energy.reshape((groups,) + (1,) * len(spatial_shape))
        )
        energy_new = e_star + np.sum(phi_star_array - phi_new_array, axis=0) / c_light
        temperature_candidate = np.asarray(invEOS(energy_new), dtype=float)
        if not np.all(np.isfinite(temperature_candidate)) or np.any(temperature_candidate <= 0.0):
            raise RuntimeError("Kompaneets update produced a nonpositive material temperature")
        denominator = np.maximum.reduce((
            np.abs(temperature_candidate), np.abs(temperature_iter),
            np.full(spatial_shape, 1.0e-12),
        ))
        temperature_change = float(np.max(
            np.abs(temperature_candidate - temperature_iter) / denominator
        ))
        if induced_scattering:
            number_scale = np.max(
                np.abs(number_star), axis=0, keepdims=True,
            )
            number_denominator = np.maximum.reduce((
                np.abs(number_new), np.abs(number_iter),
                np.broadcast_to(1.0e-12 * number_scale, number_new.shape),
                np.full(number_new.shape, 1.0e-300),
            ))
            number_change = float(np.max(
                np.abs(number_new - number_iter) / number_denominator
            ))
        else:
            number_change = 0.0
        change = max(temperature_change, number_change)
        if change < tolerance:
            converged = True
            break
        temperature_iter = (
            relaxation * temperature_candidate
            + (1.0 - relaxation) * temperature_iter
        )
        if induced_scattering:
            number_iter = (
                relaxation * number_new
                + (1.0 - relaxation) * number_iter
            )
    else:
        if on_nonconvergence == "raise":
            raise RuntimeError(
                "operator-split Kompaneets update did not converge after "
                f"{max_iterations} iterations: residual={change:.3e}, "
                f"tolerance={tolerance:.3e}"
            )

    psi_new = []
    for group in range(groups):
        old_phi = phi_star_array[group]
        new_phi = phi_new_array[group]
        old_psi = np.asarray(psi_star[group], dtype=float)
        loss = (new_phi <= old_phi) & (old_phi > 0.0)
        ratio = np.where(loss, new_phi / np.maximum(old_phi, 1.0e-300), 1.0)
        additive = np.where(loss, 0.0, new_phi - old_phi)
        psi_new.append(old_psi * ratio[..., None, :] + additive[..., None, :])

    photon_number_after = np.sum(number_new, axis=0)
    photon_number_error = float(np.max(np.abs(
        photon_number_after - photon_number_before
    )))
    total_before = e_star + np.sum(phi_star_array, axis=0) / c_light
    total_after = energy_new + np.sum(phi_new_array, axis=0) / c_light
    energy_error = float(np.max(np.abs(total_after - total_before)))
    result = (
        [phi_new_array[group].copy() for group in range(groups)],
        psi_new, energy_new, temperature_candidate, iteration,
    )
    if return_diagnostics:
        return (*result, {
            "converged": converged,
            "residual": change,
            "photon_number_error": photon_number_error,
            "energy_error": energy_error,
        })
    return result
