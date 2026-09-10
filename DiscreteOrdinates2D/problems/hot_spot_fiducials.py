import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

"""Shared output helpers for the 2-D hot-spot cooling problems."""

import os

import numpy as np


def save_fiducial_history(results, filename, method):
    """Save every time step of the fiducial-point histories to an NPZ file.

    The temperature arrays have shape ``(number of times, number of points)``.
    Only ordinary NumPy arrays are stored, so loading does not require
    ``allow_pickle=True``.
    """
    times = np.asarray(results['ts'], dtype=float)
    if times.ndim != 1:
        raise ValueError("results['ts'] must be a one-dimensional array")

    fid_data = results['fid_data']
    labels = list(fid_data)
    if not labels:
        raise ValueError("results['fid_data'] must contain at least one point")

    T_mat = np.column_stack([
        np.asarray(fid_data[label]['T_mat'], dtype=float) for label in labels
    ])
    T_rad = np.column_stack([
        np.asarray(fid_data[label]['T_rad'], dtype=float) for label in labels
    ])
    has_E_rad = all('E_rad' in fid_data[label] for label in labels)
    if has_E_rad:
        E_rad = np.column_stack([
            np.asarray(fid_data[label]['E_rad'], dtype=float) for label in labels
        ])
    expected_shape = (times.size, len(labels))
    if T_mat.shape != expected_shape or T_rad.shape != expected_shape:
        raise ValueError(
            'Each fiducial temperature history must have one value per time step'
        )
    if has_E_rad and E_rad.shape != expected_shape:
        raise ValueError(
            'Each fiducial radiation-energy history must have one value per time step'
        )

    fid_indices = results['fid_indices']
    indices = np.asarray([fid_indices[label] for label in labels], dtype=np.int64)
    if indices.shape != (len(labels), 2):
        raise ValueError('Each fiducial point must have a two-dimensional index')

    x_centers = np.asarray(results['x_centers'], dtype=float)
    y_centers = np.asarray(results['y_centers'], dtype=float)
    coordinates = np.asarray([
        (x_centers[i], y_centers[j]) for i, j in indices
    ])

    output_path = os.fspath(filename)
    if not output_path.endswith('.npz'):
        output_path += '.npz'

    np.savez_compressed(
        output_path,
        method=np.asarray(method),
        times=times,
        fiducial_labels=np.asarray(labels),
        fiducial_indices=indices,
        fiducial_coordinates=coordinates,
        T_mat=T_mat,
        T_rad=T_rad,
        time_units=np.asarray('ns'),
        coordinate_units=np.asarray('cm'),
        temperature_units=np.asarray('keV'),
        **({'E_rad': E_rad, 'energy_density_units': np.asarray('GJ/cm^3')} if has_E_rad else {}),
    )
    print(f'  Fiducial histories saved: {output_path}')
    return output_path
