#!/usr/bin/env python3
import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

"""
Compare fiducial-point temperature histories between crooked-pipe M1 closures
(Kershaw, Levermore, P1) and a reference method (IMC, S16, or P9).

Usage
-----
# Auto-detect M1 + IMC:
python plot_crooked_pipe_m1_imc_compare.py --compare-to imc

# Auto-detect M1 + S16:
python plot_crooked_pipe_m1_imc_compare.py --compare-to s16

# Auto-detect M1 + P9:
python plot_crooked_pipe_m1_imc_compare.py --compare-to p9

# Explicit files (any order):
python plot_crooked_pipe_m1_imc_compare.py --compare-to imc \
    M1/results/crooked_pipe/crooked_pipe_m1_kershaw_296x532.npz \
    M1/results/crooked_pipe/crooked_pipe_m1_levermore_296x532.npz \
    M1/results/crooked_pipe/crooked_pipe_m1_p1_296x532.npz \
    DiscreteOrdinates2D/results/crooked_pipe/IMC/crooked_pipe_imc_solution_fc_refined_Nb400000_dtmax10.0_168x354.npz

python plot_crooked_pipe_m1_imc_compare.py --compare-to s16 \
    M1/results/crooked_pipe/crooked_pipe_m1_kershaw_296x532.npz \
    M1/results/crooked_pipe/crooked_pipe_m1_levermore_296x532.npz \
    M1/results/crooked_pipe/crooked_pipe_m1_p1_296x532.npz \
    DiscreteOrdinates2D/results/crooked_pipe/S16_fine/crooked_pipe_sn_168x354.npz

python plot_crooked_pipe_m1_imc_compare.py --compare-to p9 \
    M1/results/crooked_pipe/crooked_pipe_m1_kershaw_296x532.npz \
    M1/results/crooked_pipe/crooked_pipe_m1_levermore_296x532.npz \
    M1/results/crooked_pipe/crooked_pipe_m1_p1_296x532.npz \
    results/crooked_pipe_gray/P9_no_filter/crooked_pipe_pn_148x266.npz
"""

import sys
import os
import glob
import re
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.plotfuncs import show, font

# ---------------------------------------------------------------------------
# Matplotlib style (match PN comparison style)
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Univers LT Std", "TeX Gyre Heros", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 12,
    "axes.labelsize": 12,
    "font.variant": "small-caps",
    "axes.titlesize": 18,
    "mathtext.fontset": "dejavusans",
    "mathtext.default": "it",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "axes.edgecolor": "black",
    "axes.linewidth": 1.15,
    "axes.grid": False,
    "xtick.color": "black",
    "ytick.color": "black",
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "lines.linewidth": 1.8,
    "lines.solid_capstyle": "round",
    "lines.dash_capstyle": "round",
    "legend.frameon": False,
})


def load_m1_npz(path):
    """Load M1 NPZ produced by M1/problems/crooked_pipe_m1.py."""
    d = np.load(path, allow_pickle=True)
    required = {"fid_times", "fid_labels", "fid_Tmat", "fid_Trad"}
    if not required.issubset(set(d.keys())):
        raise ValueError(f"File does not match M1 fiducial format: {path}")

    times = np.asarray(d["fid_times"])
    labels = [str(x) for x in np.asarray(d["fid_labels"]).tolist()]
    fid_tmat_arr = np.asarray(d["fid_Tmat"])
    fid_trad_arr = np.asarray(d["fid_Trad"])

    if fid_tmat_arr.ndim != 2 or fid_trad_arr.ndim != 2:
        raise ValueError(f"Unexpected M1 fiducial array shape in {path}")
    if fid_tmat_arr.shape[0] != len(times) or fid_trad_arr.shape[0] != len(times):
        raise ValueError(f"M1 fiducial history length mismatch in {path}")
    if fid_tmat_arr.shape[1] != len(labels) or fid_trad_arr.shape[1] != len(labels):
        raise ValueError(f"M1 fiducial label count mismatch in {path}")

    fiducial_data = {}
    for j, label in enumerate(labels):
        fiducial_data[label] = {
            "T_mat": fid_tmat_arr[:, j],
            "T_rad": fid_trad_arr[:, j],
        }

    closure_name = "unknown"
    if "closure_name" in d:
        closure_name = str(d["closure_name"])

    return times, fiducial_data, closure_name


def load_fiducial_npz(path):
    """Load fiducial NPZ with keys fid_times/fid_labels/fid_Tmat/fid_Trad."""
    d = np.load(path, allow_pickle=True)
    required = {"fid_times", "fid_labels", "fid_Tmat", "fid_Trad"}
    if not required.issubset(set(d.keys())):
        raise ValueError(f"File does not match fiducial format: {path}")

    times = np.asarray(d["fid_times"])
    labels = [str(x) for x in np.asarray(d["fid_labels"]).tolist()]
    fid_tmat_arr = np.asarray(d["fid_Tmat"])
    fid_trad_arr = np.asarray(d["fid_Trad"])

    if fid_tmat_arr.ndim != 2 or fid_trad_arr.ndim != 2:
        raise ValueError(f"Unexpected fiducial array shape in {path}")
    if fid_tmat_arr.shape[0] != len(times) or fid_trad_arr.shape[0] != len(times):
        raise ValueError(f"Fiducial history length mismatch in {path}")
    if fid_tmat_arr.shape[1] != len(labels) or fid_trad_arr.shape[1] != len(labels):
        raise ValueError(f"Fiducial label count mismatch in {path}")

    fiducial_data = {}
    for j, label in enumerate(labels):
        fiducial_data[label] = {
            "T_mat": fid_tmat_arr[:, j],
            "T_rad": fid_trad_arr[:, j],
        }
    return times, fiducial_data


def load_imc_npz(path):
    """Load IMC solution NPZ file."""
    d = np.load(path, allow_pickle=True)
    required = {"times", "fiducial_data", "fiducial_data_rad"}
    if not required.issubset(set(d.keys())):
        raise ValueError(f"File does not match IMC fiducial format: {path}")

    times = np.asarray(d["times"])
    fiducial_mat = d["fiducial_data"].item()
    fiducial_rad = d["fiducial_data_rad"].item()

    fiducial_data = {}
    for label in fiducial_mat.keys():
        fiducial_data[str(label)] = {
            "T_mat": np.asarray(fiducial_mat[label]),
            "T_rad": np.asarray(fiducial_rad[label]),
        }

    return times, fiducial_data


def _canonical_label(label):
    """Normalize label text so fiducial points can be matched across methods."""
    s = str(label).lower().strip()
    if ":" in s:
        s = s.split(":", 1)[1].strip()

    m = re.search(r"r\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*z\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", s)
    if m:
        rv = float(m.group(1))
        zv = float(m.group(2))
        return f"r={rv:.6f},z={zv:.6f}"

    return " ".join(s.split())


def _find_common_label_maps(datasets):
    """Return method -> {canonical_label: original_label} for shared fiducials only."""
    maps = {}
    for method, _, fid in datasets:
        cur_map = {}
        for original_label in fid.keys():
            canon = _canonical_label(original_label)
            if canon not in cur_map:
                cur_map[canon] = original_label
        maps[method] = cur_map

    common = None
    for method in maps:
        keyset = set(maps[method].keys())
        common = keyset if common is None else (common & keyset)

    if not common:
        raise RuntimeError("No shared fiducial labels found across selected files.")

    common = sorted(common)
    out = {method: {canon: maps[method][canon] for canon in common} for method in maps}
    return common, out


def detect_file_type(path, compare_to=None):
    """Classify a path as M1 Kershaw/Levermore/P1, IMC, S16, or P9."""
    lower = path.replace("\\", "/").lower()
    base = os.path.basename(lower)

    if "/p9" in lower or "p9" in base:
        return "P9"
    if "/imc/" in lower or "imc" in base:
        return "IMC"
    if "/s16" in lower or "s16" in base or "_sn_" in base or base.startswith("sn_"):
        return "S16"
    if "m1" in base or "/m1/" in lower:
        if "kershaw" in base:
            return "M1 Kershaw"
        if "levermore" in base:
            return "M1 Levermore"
        if "minerbo" in base:
            return "M1 Minerbo"
        if "_p1_" in base or base.endswith("_p1.npz"):
            return "M1 P1"

        d = np.load(path, allow_pickle=True)
        if "closure_name" in d:
            cname = str(d["closure_name"]).lower()
            if "kershaw" in cname:
                return "M1 Kershaw"
            if "levermore" in cname:
                return "M1 Levermore"
            if cname == "p1":
                return "M1 P1"

    # Fallback for generic PN/SN-like fiducial files when comparator is specified.
    try:
        d = np.load(path, allow_pickle=True)
        keys = set(d.keys())
        if {"fid_times", "fid_labels", "fid_Tmat", "fid_Trad"}.issubset(keys):
            if compare_to == "p9":
                return "P9"
            if compare_to == "s16":
                return "S16"
    except Exception:
        pass

    raise ValueError(f"Could not determine file type for {path}")


_POINT_COLORS = ["tab:blue", "tab:red", "tab:green", "tab:purple", "tab:orange", "tab:brown"]
_MARKERS = ["o", "s", "^", "d", "v", "P"]
_LINESTYLE_BY_METHOD = {
    "M1 Kershaw": "-.",
    "M1 Levermore": "--",
    "M1 P1": ":",
    "IMC": "-",
    "S16": "-",
    "P9": "-",
}


def plot_comparison(datasets, outbase):
    """Create material and radiation plots for M1 closures vs IMC."""
    common_canon_labels, label_maps = _find_common_label_maps(datasets)
    method_order = [method for method, _, _ in datasets]

    for quantity, ylabel, suffix in [
        ("T_mat", "temperature (keV)", "material"),
        ("T_rad", r"radiation temperature (keV)", "radiation"),
    ]:
        fig, ax = plt.subplots(figsize=(6, 4.5))

        for method, times, fiducial_data in datasets:
            linestyle = _LINESTYLE_BY_METHOD.get(method, ":")

            for pt_idx, canon_label in enumerate(common_canon_labels):
                pt_label = label_maps[method][canon_label]
                if pt_label not in fiducial_data:
                    print(f'Warning: fiducial label "{pt_label}" not found in {method} data; skipping.')
                    continue
                if quantity not in fiducial_data[pt_label]:
                    print(f'Warning: quantity "{quantity}" not found for fiducial label "{pt_label}" in {method} data; skipping.')
                    continue

                vals = np.asarray(fiducial_data[pt_label][quantity])
                color = _POINT_COLORS[pt_idx % len(_POINT_COLORS)]
                marker = _MARKERS[pt_idx % len(_MARKERS)]

                ax.loglog(
                    times,
                    vals,
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.8,
                    marker=marker,
                    markersize=4,
                    markevery=max(1, len(times) // 20),
                    alpha=0.85,
                )

        for method_name in method_order:
            linestyle = _LINESTYLE_BY_METHOD.get(method_name, ":")
            if method_name == "M1 P1":
                label = r"P$_1$"
            elif method_name.startswith("M1 "):
                label = method_name.split(" ", 1)[1]
            elif method_name == "P9":
                label = r"P$_9$"
            else:
                label = method_name
            ax.plot([], [], color="black", linestyle=linestyle, linewidth=2.5, label=label)
        if suffix == "radiation": #change limits
            ax.set_ylim(top=0.5)
            print("Radiation plot: setting y-axis limit to 0 - 0.5 keV")
            ax.legend(fontsize=12, loc="best", ncol=1, handlelength=2, handletextpad=0.5)
        else:
            ax.legend(fontsize=12, loc="best", ncol=1, handlelength=3.5, handletextpad=1.0)
        ax.set_xlabel("time (ns)")
        ax.set_ylabel(ylabel)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_linewidth(1.5)
        ax.spines["left"].set_linewidth(1.5)

        plt.tight_layout()
        outname = f"{outbase}_{suffix}.pdf"
        plt.savefig(outname, dpi=600)
        print(f"Saved: {outname}")
        plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare crooked-pipe M1 closures (Kershaw, Levermore, P1) against IMC, S16, or P9.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Four NPZ files to compare. If omitted, auto-detects M1 Kershaw + Levermore + P1 + comparator.",
    )
    parser.add_argument(
        "--compare-to",
        type=str,
        default="imc",
        choices=["imc", "s16", "p9"],
        help="Reference method to compare against.",
    )
    parser.add_argument(
        "--output-base",
        type=str,
        default=None,
        help="Prefix for output PDF filenames. If omitted, uses crooked_pipe_m1_<compare-to>_compare.",
    )
    return parser.parse_args()


def _latest_match(patterns):
    candidates = []
    for pat in patterns:
        candidates.extend(glob.glob(pat))
    if not candidates:
        return None
    candidates = sorted(set(candidates), key=lambda p: os.path.getmtime(p))
    return candidates[-1]


def auto_detect_paths(compare_to):
    """Auto-detect one M1 Kershaw, one M1 Levermore, one M1 Minerbo Poly, and one comparator file."""
    m1_dir = os.path.join(project_root, "M1", "results", "crooked_pipe")
    imc_dirs = [
        os.path.join(project_root, "DiscreteOrdinates2D", "results", "crooked_pipe", "IMC"),
        os.path.join(project_root, "IMC"),
        project_root,
    ]

    kershaw = _latest_match([os.path.join(m1_dir, "crooked_pipe_m1_kershaw_*.npz")])
    levermore = _latest_match([os.path.join(m1_dir, "crooked_pipe_m1_levermore_*.npz")])
    #p1 = _latest_match([os.path.join(m1_dir, "crooked_pipe_m1_p1_*.npz")])
    minerbo = _latest_match([os.path.join(m1_dir, "crooked_pipe_m1_minerbo_poly*.npz")])
    print("Auto-detected M1 files:")
    print(f"  Kershaw: {kershaw}")
    print(f"  Levermore: {levermore}")
    print(f"  Minerbo: {minerbo}")

    if compare_to == "imc":
        ref_patterns = []
        for d in imc_dirs:
            ref_patterns.extend([
                os.path.join(d, "crooked_pipe_imc_solution_*.npz"),
                os.path.join(d, "**", "crooked_pipe_imc_solution_*.npz"),
            ])
        ref = _latest_match(ref_patterns)
        ref_desc = "IMC (crooked_pipe_imc_solution_*.npz)"
    elif compare_to == "s16":
        s16_dir = os.path.join(project_root, "DiscreteOrdinates2D", "results", "crooked_pipe")
        ref_patterns = [
            os.path.join(s16_dir, "S16*", "crooked_pipe_sn_*.npz"),
            os.path.join(s16_dir, "*S16*", "crooked_pipe_sn_*.npz"),
            os.path.join(project_root, "**", "crooked_pipe_sn_*.npz"),
        ]
        ref = _latest_match(ref_patterns)
        ref_desc = "S16 (crooked_pipe_sn_*.npz)"
    else:
        p9_dir = os.path.join(project_root, "results", "crooked_pipe_gray")
        ref_patterns = [
            os.path.join(p9_dir, "P9*", "crooked_pipe_pn_*.npz"),
            os.path.join(p9_dir, "*P9*", "crooked_pipe_pn_*.npz"),
            os.path.join(project_root, "**", "crooked_pipe_pn_*.npz"),
        ]
        ref = _latest_match(ref_patterns)
        ref_desc = "P9 (crooked_pipe_pn_*.npz)"

    missing = []
    if kershaw is None:
        missing.append("M1 Kershaw (crooked_pipe_m1_kershaw_*.npz)")
    if levermore is None:
        missing.append("M1 Levermore (crooked_pipe_m1_levermore_*.npz)")
    if minerbo is None:
        missing.append("M1 Minerbo (crooked_pipe_m1_minerbo_*.npz)")
    if ref is None:
        missing.append(ref_desc)

    if missing:
        print("Error: Missing expected files:")
        for msg in missing:
            print(f"  {msg}")
        sys.exit(1)

    return [kershaw, levermore, minerbo, ref]


def method_sort_key(path):
    method = detect_file_type(path)
    order = {
        "M1 Kershaw": 0,
        "M1 Levermore": 1,
        "M1 Minerbo": 2,
        "IMC": 3,
        "S16": 3,
        "P9": 3,
    }
    return order.get(method, 99)


def main():
    args = parse_args()
    compare_to = args.compare_to.lower()

    if args.output_base is None:
        outbase = f"crooked_pipe_m1_{compare_to}_compare"
    else:
        outbase = args.output_base

    if args.files:
        if len(args.files) != 4:
            print(f"Error: Need exactly 4 files, got {len(args.files)}")
            sys.exit(1)
        npz_paths = sorted(args.files, key=lambda p: method_sort_key(p))
    else:
        npz_paths = auto_detect_paths(compare_to)

    print("Comparing:")
    for idx, path in enumerate(npz_paths, start=1):
        print(f"  File {idx}: {path}")

    datasets = []
    for path in npz_paths:
        file_type = detect_file_type(path, compare_to=compare_to)
        print(f"Detected file type for {path}: {file_type}")
        if file_type == "IMC":
            times, fiducial_data = load_imc_npz(path)
            label = "IMC"
        elif file_type in ("S16", "P9"):
            times, fiducial_data = load_fiducial_npz(path)
            label = file_type
        else:
            times, fiducial_data, _ = load_m1_npz(path)
            label = file_type

        print(f"  [{label}] {len(times)} time points")
        datasets.append((label, times, fiducial_data))

    methods = [m for m, _, _ in datasets]
    ref_name = "IMC" if compare_to == "imc" else ("S16" if compare_to == "s16" else "P9")
    required = ["M1 Kershaw", "M1 Levermore", "M1 Minerbo", ref_name]
    missing = [m for m in required if m not in methods]
    if missing:
        print("Error: Missing required methods:")
        for m in missing:
            print(f"  {m}")
        sys.exit(1)

    print("\nGenerating plots...")
    plot_comparison(datasets, outbase=outbase)
    print("\nDone.")


if __name__ == "__main__":
    main()
