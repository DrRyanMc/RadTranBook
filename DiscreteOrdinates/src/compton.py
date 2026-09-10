"""Local operator-split multigroup Compton update for :mod:`mg_sn_solver`.

The update follows the OS-FBC formulation of McGraw, Till, and Warsa
(``JCP 478 (2023) 111980``).  The transport solver supplies the state after
streaming, absorption/emission, and physical angle scattering.  This module
then performs the local frequency redistribution and applies the equal and
opposite material-energy change.

The full-Boltzmann group matrices are deliberately inputs.  In the paper they
are produced by CSK by averaging the Klein--Nishina kernel over a
Maxwell--Juttner electron distribution.  Treating those data as an input keeps
the transport algorithm independent of a particular Compton-data generator
and avoids presenting a cold-electron approximation as thermal Compton data.

All scalar intensities in this module use the convention already used by the
multigroup S_N solver: ``phi_g = 4*pi*B_g`` at equilibrium.  Thus the group
Planck functions supplied to ``operator_split_compton_update`` must have that
same convention.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import warnings


@dataclass
class ComptonMatrices:
    """Macroscopic full-Boltzmann Compton matrices.

    Every matrix uses ``[source_group, destination_group, ...]`` ordering.
    The trailing dimensions are either absent (spatially uniform data) or
    match the shape of the material-temperature array.  Units are cm^-1
    for the linear matrices; induced matrices have the additional reciprocal
    scalar-intensity unit required by their multiplication by ``phi``.

    Parameters correspond to Eqs. (13)--(15) of McGraw et al.:

    ``in_scatter``
        :math:`\\sigma^{IL}_{s,g'\\to g}`.
    ``out_scatter``
        :math:`\\sigma^{OL}_{s,g\\to g'}`.
    ``out_induced``
        Optional :math:`\\sigma^{ON}_{s,g\\to g'}`.
    ``in_induced``
        Optional :math:`\\sigma^{IN}_{s,g'\\to g}`.
    """

    in_scatter: Optional[np.ndarray]
    out_scatter: np.ndarray
    out_induced: Optional[np.ndarray] = None
    in_induced: Optional[np.ndarray] = None
    planck_net_out: Optional[np.ndarray] = None
    group_mean_energy: Optional[np.ndarray] = None
    group_scalar_capacity: Optional[np.ndarray] = None


def _as_matrices(value):
    """Accept a ``ComptonMatrices``, mapping, or four-/two-tuple."""
    if isinstance(value, ComptonMatrices):
        return value
    if isinstance(value, dict):
        return ComptonMatrices(
            in_scatter=value["in_scatter"],
            out_scatter=value["out_scatter"],
            out_induced=value.get("out_induced"),
            in_induced=value.get("in_induced"),
            planck_net_out=value.get("planck_net_out"),
            group_mean_energy=value.get("group_mean_energy"),
            group_scalar_capacity=value.get("group_scalar_capacity"),
        )
    if isinstance(value, (tuple, list)):
        if len(value) == 2:
            return ComptonMatrices(value[0], value[1])
        if len(value) == 4:
            return ComptonMatrices(value[0], value[1], value[2], value[3])
    raise TypeError(
        "compton_matrix_func(T) must return ComptonMatrices, a mapping, "
        "or a two-/four-element tuple"
    )


def _matrix_value(matrix, source, destination, index, spatial_shape):
    """Return one matrix entry, broadcasting spatially uniform tables."""
    array = np.asarray(matrix, dtype=float)
    if array.ndim == 2:
        return array[source, destination]
    if array.shape[2:] != spatial_shape:
        raise ValueError(
            "Compton matrix trailing dimensions must match the temperature "
            f"shape {spatial_shape}; got {array.shape[2:]}"
        )
    return array[(source, destination) + index]


def _validate_matrices(matrices, groups, spatial_shape):
    for name in ("in_scatter", "out_scatter", "out_induced", "in_induced"):
        matrix = getattr(matrices, name)
        if matrix is None:
            continue
        matrix = np.asarray(matrix)
        if matrix.ndim < 2 or matrix.shape[:2] != (groups, groups):
            raise ValueError(
                f"{name} must start with shape ({groups}, {groups}); "
                f"got {matrix.shape}"
            )
        if matrix.ndim > 2 and matrix.shape[2:] != spatial_shape:
            raise ValueError(
                f"{name} trailing dimensions must be {spatial_shape}; "
                f"got {matrix.shape[2:]}"
            )
    if matrices.planck_net_out is not None:
        values = np.asarray(matrices.planck_net_out)
        allowed = ((groups,), (groups,) + spatial_shape)
        if values.shape not in allowed:
            raise ValueError(
                "planck_net_out must have shape "
                f"({groups},) or {(groups,) + spatial_shape}; got {values.shape}"
            )
    compact_values = (matrices.group_mean_energy, matrices.group_scalar_capacity)
    if matrices.in_scatter is None:
        if any(value is None for value in compact_values):
            raise ValueError(
                "in_scatter may be omitted only for compact conservative matrices"
            )
        for name, values in zip(
            ("group_mean_energy", "group_scalar_capacity"), compact_values,
        ):
            array = np.asarray(values)
            if array.shape != (groups,) or np.any(array <= 0.0):
                raise ValueError(f"{name} must be positive with shape ({groups},)")


def _planck_net_out_value(values, source, index, spatial_shape):
    """Return a Planck-balanced net out-scattering opacity."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        return values[source]
    return values[(source,) + index]


def _matrix_field(matrix, groups, spatial_shape):
    """Return a group matrix broadcast over the local spatial field."""
    array = np.asarray(matrix, dtype=float)
    if array.ndim == 2:
        return np.broadcast_to(
            array.reshape((groups, groups) + (1,) * len(spatial_shape)),
            (groups, groups) + spatial_shape,
        )
    return array


def _group_field(values, groups, spatial_shape):
    """Return a group vector broadcast over the local spatial field."""
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        return np.broadcast_to(
            array.reshape((groups,) + (1,) * len(spatial_shape)),
            (groups,) + spatial_shape,
        )
    return array


def operator_split_compton_update(
    phi_star,
    psi_star,
    e_star,
    T_star,
    dt,
    c_light,
    invEOS,
    Bg_funcs,
    compton_matrix_func,
    induced_mode="planck",
    tolerance=1.0e-10,
    max_iterations=50,
    relaxation=1.0,
    positivity_limiter="damp",
    on_nonconvergence="raise",
    return_diagnostics=False,
):
    r"""Apply McGraw et al.'s local backward-Euler Compton update.

    Parameters
    ----------
    phi_star, psi_star
        Radiation state after the ordinary S_N transport update.  ``phi_star``
        is a list with one array per group, all with shape ``T_star.shape``;
        angular intensities have shape ``(I, N, nop1)``.
    e_star, T_star
        Material state after the ordinary transport update.
    compton_matrix_func
        Callable ``T -> ComptonMatrices``.  Its matrices must already be
        macroscopic, e.g. microscopic CSK data multiplied by the local free
        electron density.
    induced_mode : {"nonlinear", "planck", "planck-lagged", "lagged", "wien"}
        McGraw et al.'s treatments of induced scattering.  ``planck`` is the
        default and uses the current local iterate of ``4*pi*B_g(T)``;
        ``planck-lagged`` freezes that Planck spectrum and all matrices at
        ``T_star`` and completes one linear solve;
        ``lagged`` uses ``phi_star``; ``wien`` omits induced terms.
        ``nonlinear`` uses the current end-of-step scalar-intensity iterate in
        the induced terms and Picard-iterates the coupled radiation/material
        system to convergence.

    Returns
    -------
    phi_new, psi_new, e_new, T_new, iterations
        The updated radiation and material state.  Angular intensities are
        reconstructed multiplicatively when a group loses energy and with an
        isotropic additive term when it gains energy, which preserves
        non-negativity for non-negative input intensities.

    on_nonconvergence : {"raise", "warn", "accept"}
        Action when the Picard iteration reaches ``max_iterations``.  The
        ``accept`` option returns the last conservative iterate; it is useful
        for deliberately lagged, fixed-iteration Compton updates.
    positivity_limiter : {"raise", "damp"}
        Response when a linearized group solve produces a negative scalar
        intensity.  ``damp`` limits each affected group update between the
        nonnegative Picard iterate and the raw linear solution.  The material
        energy is then updated from this accepted, limited radiation change.
    return_diagnostics : bool
        If true, append a dictionary containing convergence and residual
        information to the returned tuple.
    """
    if induced_mode not in {"nonlinear", "planck", "planck-lagged", "lagged", "wien"}:
        raise ValueError(
            "induced_mode must be 'nonlinear', 'planck', 'planck-lagged', "
            "'lagged', or 'wien'"
        )
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if not (0.0 < relaxation <= 1.0):
        raise ValueError("relaxation must lie in (0, 1]")
    if on_nonconvergence not in {"raise", "warn", "accept"}:
        raise ValueError("on_nonconvergence must be 'raise', 'warn', or 'accept'")
    if positivity_limiter not in {"raise", "damp"}:
        raise ValueError("positivity_limiter must be 'raise' or 'damp'")

    groups = len(phi_star)
    if groups == 0:
        raise ValueError("at least one energy group is required")
    spatial_shape = np.asarray(T_star).shape
    if len(Bg_funcs) != groups:
        raise ValueError("Bg_funcs must contain one function per group")
    if any(np.asarray(phi).shape != spatial_shape for phi in phi_star):
        raise ValueError("all scalar intensities must have shape T_star.shape")

    phi_star_array = np.stack([np.asarray(phi, dtype=float) for phi in phi_star])
    e_star = np.asarray(e_star, dtype=float)
    T_iter = np.asarray(T_star, dtype=float).copy()
    phi_iter = phi_star_array.copy()
    cdt = c_light * dt

    # The nonlinear option Picard-iterates both the radiation field in the
    # induced coefficients and the material temperature.  Other modes retain
    # the material iteration because their tabulated data depend on T.
    converged = False
    temperature_change = np.inf
    intensity_change = np.inf if induced_mode == "nonlinear" else 0.0
    change = np.inf
    positivity_limited_cells = 0
    minimum_positivity_damping = 1.0
    minimum_picard_relaxation = relaxation
    previous_change = np.inf
    lagged_temperature_change = 0.0
    iteration_limit = 1 if induced_mode == "planck-lagged" else max_iterations
    for iteration in range(1, iteration_limit + 1):
        matrices = _as_matrices(compton_matrix_func(T_iter))
        _validate_matrices(matrices, groups, spatial_shape)

        if induced_mode in {"planck", "planck-lagged"}:
            phi_induced = np.stack([
                np.asarray(Bg_funcs[g](T_iter), dtype=float)
                for g in range(groups)
            ])
        elif induced_mode == "lagged":
            phi_induced = phi_star_array
        elif induced_mode == "nonlinear":
            phi_induced = phi_iter
        else:
            phi_induced = np.zeros_like(phi_star_array)

        # Arrange the local systems as batches of small dense matrices.  The
        # former implementation performed one Python-level ``np.linalg.solve``
        # call per cell corner; batching shifts that loop into LAPACK and is
        # markedly faster for the 32-group channel calculation.
        cells = int(np.prod(spatial_shape))
        phi_star_flat = phi_star_array.reshape(groups, cells)
        phi_induced_flat = phi_induced.reshape(groups, cells)
        phi_iter_flat = phi_iter.reshape(groups, cells)
        phi_new_flat = np.zeros_like(phi_star_flat)
        out_scatter = _matrix_field(matrices.out_scatter, groups, spatial_shape).reshape(
            groups, groups, cells
        )
        compact_conservative = matrices.in_scatter is None
        if compact_conservative:
            group_energy = np.asarray(matrices.group_mean_energy, dtype=float)
            group_capacity = np.asarray(matrices.group_scalar_capacity, dtype=float)
            energy_ratio = group_energy[None, :] / group_energy[:, None]
            in_scatter = None
            out_induced = None
            in_induced = None
        else:
            in_scatter = _matrix_field(
                matrices.in_scatter, groups, spatial_shape
            ).reshape(groups, groups, cells)
            out_induced = (None if matrices.out_induced is None else
                           _matrix_field(matrices.out_induced, groups, spatial_shape).reshape(
                               groups, groups, cells
                           ))
            in_induced = (None if matrices.in_induced is None else
                          _matrix_field(matrices.in_induced, groups, spatial_shape).reshape(
                              groups, groups, cells
                          ))
        planck_net_out = (None if matrices.planck_net_out is None else
                          _group_field(matrices.planck_net_out, groups, spatial_shape).reshape(
                              groups, cells
                          ))

        # A cell with exactly zero scalar intensity has a zero right-hand side
        # and therefore no Compton update.  Avoiding its dense group solve is
        # particularly valuable while the boundary source is still crossing
        # the channel.
        active_cells = np.flatnonzero(np.any(phi_star_flat != 0.0, axis=0))
        solve_block_size = 256
        diagonal = np.arange(groups)
        for block_start in range(0, active_cells.size, solve_block_size):
            block = active_cells[block_start:block_start + solve_block_size]
            rhs = phi_star_flat[:, block]
            out_block = out_scatter[:, :, block]
            if compact_conservative:
                in_block = out_block * energy_ratio[:, :, None]
            else:
                in_block = in_scatter[:, :, block]
            if induced_mode in {"planck", "planck-lagged"} and planck_net_out is not None:
                sigma_out = planck_net_out[:, block].copy()
            else:
                sigma_out = np.sum(out_block, axis=1)
                if induced_mode != "wien":
                    occupation = phi_induced_flat[:, block]
                    if compact_conservative:
                        sigma_out += np.einsum(
                            "sdn,dn->sn",
                            out_block / group_capacity[None, :, None],
                            occupation,
                            optimize=True,
                        )
                        sigma_out -= np.einsum(
                            "dsn,dn->sn",
                            in_block / group_capacity[None, :, None],
                            occupation,
                            optimize=True,
                        )
                    elif out_induced is not None:
                        sigma_out += np.einsum(
                            "sdn,dn->sn", out_induced[:, :, block], occupation,
                            optimize=True,
                        )
                    if in_induced is not None:
                        sigma_out -= np.einsum(
                            "dsn,dn->sn", in_induced[:, :, block], occupation,
                            optimize=True,
                        )

            # (I - c dt K) phi^{n+1} = phi^*.  The batched system has axes
            # (cell, destination group, source group).
            system = -cdt * np.transpose(in_block, (2, 1, 0)).copy()
            system[:, diagonal, diagonal] += 1.0 + cdt * sigma_out.T
            raw_solution = np.linalg.solve(system, rhs.T[..., None])[..., 0].T
            if not np.all(np.isfinite(raw_solution)):
                raise RuntimeError("Compton update produced a non-finite scalar intensity")

            negative = raw_solution < -1.0e-12
            if np.any(negative):
                if positivity_limiter == "raise":
                    raise RuntimeError(
                        "Compton update produced a negative scalar intensity; "
                        "check the group matrices and time step"
                    )
                # Limit every affected component along the segment from the
                # nonnegative Picard state to the raw candidate.  The material
                # update below uses this accepted radiation state and therefore
                # retains local total-energy conservation.
                reference = (phi_iter_flat[:, block] if induced_mode == "nonlinear"
                             else rhs)
                reference = np.maximum(reference, 0.0)
                direction = raw_solution - reference
                damping = np.ones_like(raw_solution)
                damping[negative] = np.clip(
                    reference[negative] / np.maximum(-direction[negative], 1.0e-300),
                    0.0, 1.0,
                )
                solution = reference + damping * direction
                limited = np.any(negative, axis=0)
                positivity_limited_cells += int(np.count_nonzero(limited))
                minimum_positivity_damping = min(
                    minimum_positivity_damping, float(np.min(damping))
                )
            else:
                solution = raw_solution
            phi_new_flat[:, block] = np.maximum(solution, 0.0)

        phi_new_array = phi_new_flat.reshape((groups,) + spatial_shape)

        # The material update is the equal and opposite of the discrete
        # radiation-energy change.  This form is algebraically equivalent to
        # Eq. (15b) for energy-consistent matrices, but avoids cancellation
        # between large induced terms when the photon occupation is high.
        e_candidate = e_star + np.sum(phi_star_array - phi_new_array, axis=0) / c_light
        T_candidate = invEOS(e_candidate)

        temperature_denominator = np.maximum.reduce((
            np.abs(T_candidate), np.abs(T_iter), np.full(spatial_shape, 1.0e-12),
        ))
        temperature_change = np.max(
            np.abs(T_candidate - T_iter) / temperature_denominator
        )
        if induced_mode == "nonlinear":
            intensity_change = (
                np.max(np.abs(phi_new_array - phi_iter))
                / max(
                    np.max(np.abs(phi_new_array)),
                    np.max(np.abs(phi_iter)),
                    1.0e-30,
                )
            )
        else:
            intensity_change = 0.0
        change = max(temperature_change, intensity_change)
        if induced_mode == "planck-lagged":
            lagged_temperature_change = float(temperature_change)
            temperature_change = 0.0
            change = 0.0
            converged = True
            break
        if induced_mode == "nonlinear":
            picard_relaxation = relaxation
            if np.isfinite(previous_change) and change > previous_change:
                picard_relaxation *= min(1.0, 0.8 * previous_change / change)

            while True:
                phi_trial = (
                    picard_relaxation * phi_new_array
                    + (1.0 - picard_relaxation) * phi_iter
                )
                e_trial = e_star + np.sum(
                    phi_star_array - phi_trial, axis=0
                ) / c_light
                T_trial = np.asarray(invEOS(e_trial), dtype=float)
                if np.all(np.isfinite(T_trial)) and np.all(T_trial > 0.0):
                    break
                picard_relaxation *= 0.5
                if picard_relaxation < 1.0e-12:
                    raise RuntimeError(
                        "Compton Picard iteration could not preserve a finite, "
                        "positive material temperature; reduce the time step"
                    )
            minimum_picard_relaxation = min(
                minimum_picard_relaxation, picard_relaxation
            )
            phi_iter = phi_trial
            T_iter = T_trial
            previous_change = change
        else:
            T_iter = relaxation * T_candidate + (1.0 - relaxation) * T_iter
        if change < tolerance:
            converged = True
            break
    else:
        message = (
            "operator-split Compton update did not converge after "
            f"{max_iterations} iterations: residual={change:.3e}, "
            f"temperature residual={temperature_change:.3e}, "
            f"intensity residual={intensity_change:.3e}, tolerance={tolerance:.3e}."
        )
        if on_nonconvergence == "raise":
            raise RuntimeError(
                message + " Reduce dt, use relaxation < 1, or increase max_iterations."
            )
        if on_nonconvergence == "warn":
            warnings.warn(message + " Accepting the final Picard iterate.", RuntimeWarning,
                          stacklevel=2)

    if induced_mode == "nonlinear" and not converged:
        phi_new_array = phi_iter
        e_candidate = e_star + np.sum(phi_star_array - phi_new_array, axis=0) / c_light
        T_candidate = invEOS(e_candidate)

    # The state and matrices at the converged temperature have been used to
    # form the final candidate.  Reconstruct directions following Eqs. 16--18
    # of McGraw et al.  The S_N weights in this code sum to one, so the
    # isotropic additive amount is phi_new - phi_star (not / 4*pi).
    psi_new = []
    for group in range(groups):
        old_phi = phi_star_array[group]
        new_phi = phi_new_array[group]
        old_psi = np.asarray(psi_star[group], dtype=float)
        loss = (new_phi <= old_phi) & (old_phi > 0.0)
        ratio = np.where(loss, new_phi / np.maximum(old_phi, 1.0e-300), 1.0)
        additive = np.where(loss, 0.0, new_phi - old_phi)
        # ``phi`` has shape ``spatial_shape`` and the angular ordinate axis
        # is immediately before its final corner/finite-element axis.  The
        # ellipsis makes this reconstruction work for both 1-D and 2-D S_N.
        psi_new.append(old_psi * ratio[..., None, :] + additive[..., None, :])

    result = ([phi_new_array[g].copy() for g in range(groups)], psi_new,
              e_candidate, T_candidate, iteration)
    if return_diagnostics:
        return (*result, {
            "converged": converged,
            "residual": float(change),
            "temperature_residual": float(temperature_change),
            "intensity_residual": float(intensity_change),
            "tolerance": float(tolerance),
            "positivity_limited_cells": int(positivity_limited_cells),
            "minimum_positivity_damping": float(minimum_positivity_damping),
            "minimum_picard_relaxation": float(minimum_picard_relaxation),
            "lagged_temperature_change": float(lagged_temperature_change),
        })
    return result
