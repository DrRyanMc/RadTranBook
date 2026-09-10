"""Implicit Kompaneets frequency updates for IMC census particles.

This module implements the last stage of the operator split described by
Densmore, Warsa, and Morel: ordinary IMC first transports the particles with
Compton scattering omitted; a cell-local, backward-Euler Kompaneets solve then
updates the census spectrum and the material energy.  The census-particle
directions and positions are retained.  Their weights are scaled by the local
radiation-energy ratio and their frequencies are resampled from the updated
piecewise-constant spectral-energy histogram.
"""

from __future__ import annotations

import numpy as np

try:
    from DiscreteOrdinates.src.kompaneets import densmore_implicit_kompaneets_update
except ImportError:
    # The stand-alone IMC drivers put ``DiscreteOrdinates`` itself on
    # ``sys.path`` rather than importing it as a namespace package.
    from kompaneets import densmore_implicit_kompaneets_update


def implicit_kompaneets_census_update(
    weights,
    cell_indices,
    groups,
    photon_energies,
    volumes,
    internal_energy,
    temperature,
    dt,
    energy_edges_kev,
    thomson_opacity,
    c_light,
    inv_eos,
    group_mean_energy_kev=None,
    tolerance=1.0e-10,
    max_iterations=50,
    relaxation=1.0,
    induced_scattering=False,
    group_scalar_capacity=None,
):
    """Apply an implicit Kompaneets update to the current census particles.

    ``weights`` are particle energy weights in GJ.  The returned weights
    retain the exact cell radiation energy calculated by the deterministic
    Kompaneets solve; the group spectrum is represented statistically by the
    resampled census-particle frequencies, as in Densmore et al.'s Eq. (22).
    """
    weights = np.asarray(weights, dtype=float).copy()
    cell_indices = np.asarray(cell_indices, dtype=np.int64)
    groups = np.asarray(groups, dtype=np.int32).copy()
    photon_energies = np.asarray(photon_energies, dtype=float).copy()
    volumes = np.asarray(volumes, dtype=float)
    internal_energy = np.asarray(internal_energy, dtype=float)
    temperature = np.asarray(temperature, dtype=float)
    edges = np.asarray(energy_edges_kev, dtype=float)
    cells = volumes.size
    group_count = edges.size - 1
    if dt <= 0.0 or group_count < 1 or np.any(np.diff(edges) <= 0.0):
        raise ValueError("dt and a strictly increasing energy grid are required")
    if internal_energy.shape != (cells,) or temperature.shape != (cells,):
        raise ValueError("material arrays must have one entry per spatial cell")
    if np.any(volumes <= 0.0):
        raise ValueError("cell volumes must be positive")
    if group_mean_energy_kev is None:
        # Densmore et al. evaluate their flux at arithmetic group centers.
        mean_energy = 0.5 * (edges[:-1] + edges[1:])
    else:
        mean_energy = np.asarray(group_mean_energy_kev, dtype=float)
    if mean_energy.shape != (group_count,) or np.any(mean_energy <= 0.0):
        raise ValueError("group mean energies must be positive and match the grid")

    live = ((weights > 0.0) & (cell_indices >= 0) & (cell_indices < cells)
            & (groups >= 0) & (groups < group_count))
    bin_id = groups[live].astype(np.int64) * cells + cell_indices[live]
    group_energy_star = np.bincount(
        bin_id, weights=weights[live], minlength=group_count * cells,
    ).reshape(group_count, cells)
    phi_star = c_light * group_energy_star / volumes[None, :]
    # The deterministic update only needs scalar intensity.  A dummy angular
    # dimension allows reuse of the multigroup scalar representation.
    phi_argument = [phi_star[group, :, None] for group in range(group_count)]
    energy_argument = internal_energy[:, None]
    temperature_argument = temperature[:, None]

    if induced_scattering:
        raise NotImplementedError(
            "the Densmore IMC split currently implements the linear, "
            "non-induced Kompaneets equation"
        )
    phi_new, energy_new, temperature_new, diagnostics = densmore_implicit_kompaneets_update(
        phi_argument,
        energy_argument,
        temperature_argument,
        dt,
        c_light,
        lambda value: np.asarray(inv_eos(value), dtype=float),
        edges,
        mean_energy,
        thomson_opacity,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    group_energy_new = np.asarray(
        [value[:, 0] for value in phi_new], dtype=float,
    ) * volumes[None, :] / c_light

    # Equation (23): every census particle in a cell gets the same energy
    # weight multiplier.  Equation (22): sample its new frequency from the
    # updated, piecewise-constant spectral-energy histogram.  Directions and
    # positions do not appear here and are intentionally unchanged.
    for cell in range(cells):
        selected = np.flatnonzero(live & (cell_indices == cell))
        old_energy = float(np.sum(weights[selected]))
        new_energy = float(np.sum(group_energy_new[:, cell]))
        if old_energy <= 0.0:
            if new_energy > 1.0e-20:
                raise RuntimeError(
                    "Kompaneets update produced radiation energy in an empty census cell"
                )
            continue
        weights[selected] *= new_energy / old_energy
        if new_energy <= 0.0:
            weights[selected] = 0.0
            continue
        probabilities = group_energy_new[:, cell] / new_energy
        probabilities = np.maximum(probabilities, 0.0)
        probabilities /= np.sum(probabilities)
        sampled_groups = np.random.choice(group_count, size=selected.size, p=probabilities)
        groups[selected] = sampled_groups.astype(np.int32)
        photon_energies[selected] = (
            edges[sampled_groups]
            + np.random.random(selected.size) * np.diff(edges)[sampled_groups]
        )

    diagnostics = dict(diagnostics)
    diagnostics.update({
        "group_energy_star": group_energy_star,
        "group_energy_new": group_energy_new,
    })
    return (
        weights,
        groups,
        photon_energies,
        np.asarray(energy_new[:, 0]),
        np.asarray(temperature_new[:, 0]),
        diagnostics,
    )
