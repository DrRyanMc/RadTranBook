"""Plot fiducial temperature histories from multiple hot-spot solution NPZ files.

This script expects NPZ files produced by hot_spot_fiducials.save_fiducial_history,
which stores arrays:
  - times: shape (Nt,)
  - fiducial_labels: shape (Nf,)
  - T_mat: shape (Nt, Nf)
  - T_rad: shape (Nt, Nf)
  - method: scalar string

Plot convention:
  - color encodes fiducial point
  - line style encodes method/solver

Run from DiscreteOrdinates2D/problems:
  python plot_hot_spot_fiducial_compare.py \
      hot_spot_sn_fiducials.npz \
      hot_spot_imc_fiducials.npz \
      hot_spot_noneq_fiducials.npz \
      hot_spot_eqdiff_fiducials.npz
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


@dataclass
class FiducialDataset:
    path: str
    method: str
    times: np.ndarray
    labels: list[str]
    T_mat: np.ndarray
    T_rad: np.ndarray


def _to_text_list(arr: np.ndarray) -> list[str]:
    out = []
    for item in arr.tolist():
        if isinstance(item, bytes):
            out.append(item.decode('utf-8'))
        else:
            out.append(str(item))
    return out


def _load_dataset(path: str) -> FiducialDataset:
    with np.load(path) as data:
        required = ['times', 'fiducial_labels', 'T_mat', 'T_rad', 'method']
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(f"{path}: missing keys {missing}")

        times = np.asarray(data['times'], dtype=float)
        labels = _to_text_list(np.asarray(data['fiducial_labels']))
        T_mat = np.asarray(data['T_mat'], dtype=float)
        T_rad = np.asarray(data['T_rad'], dtype=float)

        method_raw = np.asarray(data['method'])
        if method_raw.ndim == 0:
            method_val = method_raw.item()
        else:
            method_val = method_raw.flat[0]
        method = method_val.decode('utf-8') if isinstance(method_val, bytes) else str(method_val)

    if times.ndim != 1:
        raise ValueError(f"{path}: times must be 1-D")
    if T_mat.shape != (times.size, len(labels)):
        raise ValueError(f"{path}: T_mat has shape {T_mat.shape}, expected {(times.size, len(labels))}")
    if T_rad.shape != (times.size, len(labels)):
        raise ValueError(f"{path}: T_rad has shape {T_rad.shape}, expected {(times.size, len(labels))}")

    return FiducialDataset(
        path=path,
        method=method,
        times=times,
        labels=labels,
        T_mat=T_mat,
        T_rad=T_rad,
    )


def _collect_all_labels(datasets: list[FiducialDataset]) -> list[str]:
    labels: list[str] = []
    for ds in datasets:
        for label in ds.labels:
            if label not in labels:
                labels.append(label)
    return labels


def _build_style_map(methods: list[str]) -> dict[str, str]:
    styles = ['-', '--', '-.', ':']
    style_map = {}
    for i, method in enumerate(methods):
        style_map[method] = styles[i % len(styles)]
    return style_map


def _plot_field(
    datasets: list[FiducialDataset],
    field_name: str,
    ylabel: str,
    out_path: str,
    xmax: float | None,
    ymin: float | None,
    ymax: float | None,
) -> None:
    all_labels = _collect_all_labels(datasets)
    methods = [ds.method for ds in datasets]

    cmap = plt.get_cmap('tab10')
    color_map = {
        label: cmap(i % 10)
        for i, label in enumerate(all_labels)
    }
    style_map = _build_style_map(methods)

    fig, ax = plt.subplots(figsize=(9, 6))

    for ds in datasets:
        field = ds.T_mat if field_name == 'T_mat' else ds.T_rad
        label_to_col = {lbl: k for k, lbl in enumerate(ds.labels)}

        for fid_label in all_labels:
            if fid_label not in label_to_col:
                continue
            col = label_to_col[fid_label]
            ax.semilogx(
                ds.times,
                field[:, col],
                color=color_map[fid_label],
                linestyle=style_map[ds.method],
                linewidth=2.0,
                alpha=0.95,
            )

    ax.set_xlabel('Time (ns)')
    ax.set_ylabel(ylabel)
    ax.grid(True, which='both', alpha=0.3, linestyle='--')

    if xmax is None:
        xmax = max(ds.times[-1] for ds in datasets)
    ax.set_xlim(0.0, xmax)

    if ymin is not None or ymax is not None:
        ax.set_ylim(bottom=ymin, top=ymax)

    fid_handles = [
        Line2D([0], [0], color=color_map[label], linestyle='-', linewidth=2.0, label=label)
        for label in all_labels
    ]
    method_handles = [
        Line2D([0], [0], color='black', linestyle=style_map[method], linewidth=2.0, label=method)
        for method in methods
    ]

    legend_fid = ax.legend(handles=fid_handles, title='Fiducial point', loc='upper right', fontsize=9)
    ax.add_artist(legend_fid)
    ax.legend(handles=method_handles, title='Method', loc='lower left', fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def _parse_inputs(input_specs: list[str]) -> list[tuple[str, str | None]]:
    pairs: list[tuple[str, str | None]] = []
    for spec in input_specs:
        if ':' in spec:
            path, alias = spec.split(':', 1)
            pairs.append((path, alias.strip() or None))
        else:
            pairs.append((spec, None))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Compare hot-spot fiducial histories across methods.'
    )
    parser.add_argument(
        'inputs',
        nargs='+',
        help='NPZ inputs. Use path or path:MethodName to override the method label.',
    )
    parser.add_argument(
        '--out-prefix',
        default='hot_spot_fiducial_compare',
        help='Prefix for output figure filenames.',
    )
    parser.add_argument(
        '--xmax',
        type=float,
        default=None,
        help='Optional x-axis max in ns. Defaults to max time across inputs.',
    )
    parser.add_argument(
        '--ymin',
        type=float,
        default=None,
        help='Optional y-axis lower bound for semilogy.',
    )
    parser.add_argument(
        '--ymax',
        type=float,
        default=None,
        help='Optional y-axis upper bound for semilogy.',
    )
    args = parser.parse_args()

    datasets: list[FiducialDataset] = []
    seen_methods: set[str] = set()

    for path, alias in _parse_inputs(args.inputs):
        if not os.path.exists(path):
            raise FileNotFoundError(f'Input file not found: {path}')

        ds = _load_dataset(path)
        if alias is not None:
            ds.method = alias

        base_method = ds.method
        suffix = 2
        while ds.method in seen_methods:
            ds.method = f'{base_method} ({suffix})'
            suffix += 1
        seen_methods.add(ds.method)

        datasets.append(ds)
        print(f"Loaded: {path}  method={ds.method}  Nt={ds.times.size}  Nf={len(ds.labels)}")

    _plot_field(
        datasets,
        field_name='T_mat',
        ylabel='Material Temperature (keV)',
        out_path=f'{args.out_prefix}_T_mat.png',
        xmax=args.xmax,
        ymin=args.ymin,
        ymax=args.ymax,
    )

    _plot_field(
        datasets,
        field_name='T_rad',
        ylabel='Radiation Temperature (keV)',
        out_path=f'{args.out_prefix}_T_rad.png',
        xmax=args.xmax,
        ymin=args.ymin,
        ymax=args.ymax,
    )


if __name__ == '__main__':
    main()
