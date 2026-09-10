"""Validation workflow for cylindrical r-z Zel'dovich wave (M1 vs S_N).

Runs three checks:
1) Multi-closure M1 comparison.
2) Radial convergence sweep in Ir for M1 (Levermore).
3) Direct M1-vs-S_N radial overlays at matched times.

Outputs figures, CSV metrics, and an auto-written markdown report directory.
"""

import argparse
import csv
import os
import sys
import importlib.util
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_M1_DIR = os.path.dirname(_THIS_DIR)
_REPO_ROOT = os.path.dirname(_M1_DIR)
sys.path.insert(0, _REPO_ROOT)


def _import_module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Could not load module {name} from {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_M1_ZEL_PATH = os.path.join(_THIS_DIR, 'zeldovich_wave_rz.py')
_SN_ZEL_PATH = os.path.join(_REPO_ROOT, 'DiscreteOrdinates2D', 'problems', 'zeldovich_wave_rz.py')

_m1_z = _import_module_from_path('m1_zeldovich_wave_rz', _M1_ZEL_PATH)
_sn_z = _import_module_from_path('sn_zeldovich_wave_rz', _SN_ZEL_PATH)

run_zeldovich_rz_m1 = _m1_z.run_zeldovich_rz_m1
extract_m1_radial_profile = _m1_z.extract_radial_profile
zeldovich_self_similar = _m1_z.zeldovich_self_similar
save_npz = _m1_z.save_npz

run_sn_rz = _sn_z.run_zeldovich_rz
extract_sn_radial_profile = _sn_z.extract_radial_profile


def rel_l2(a, b):
    num = float(np.linalg.norm(a - b))
    den = float(np.linalg.norm(b))
    return num / den if den > 0.0 else np.nan


def m1_z_var(result, t_target):
    ts = result['ts']
    idx = int(np.argmin(np.abs(ts - t_target)))
    Er = result['Ers'][idx]
    Er_z_mean = np.mean(Er, axis=1, keepdims=True)
    return float(np.max(np.abs(Er - Er_z_mean)) / (Er_z_mean.max() + 1e-30))


def sn_z_var(result, t_target):
    ts = result['ts']
    idx = int(np.argmin(np.abs(ts - t_target)))
    phi_snap = result['phis'][idx]
    phi_cell = np.mean(phi_snap, axis=2)
    phi_z_mean = np.mean(phi_cell, axis=1, keepdims=True)
    return float(np.max(np.abs(phi_cell - phi_z_mean)) / (phi_z_mean.max() + 1e-30))


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def run_closure_suite(args, output_dir):
    closures = [c.strip() for c in args.closures.split(',') if c.strip()]
    m1_results = {}

    for cname in closures:
        result = run_zeldovich_rz_m1(
            Ir=args.Ir,
            Iz=args.Iz,
            closure_name=cname,
            Lr=args.Lr,
            Lz=args.Lz,
            t_start=args.t_start,
            tfinal=args.tfinal,
            dt_min=args.dt_min,
            dt_max=args.dt_max,
            dt_increase_factor=args.dt_increase_factor,
            stretch=args.stretch,
            output_times=args.output_times,
            LOUD=args.loud,
            print_stride=args.print_stride,
        )
        m1_results[cname] = result
        npz_name = os.path.join(output_dir, f'm1_{cname}_{args.Ir}x{args.Iz}.npz')
        save_npz(result, npz_name)

    return m1_results


def run_sn_reference(args):
    return run_sn_rz(
        Ir=args.Ir,
        Iz=args.Iz,
        N_quad=args.sn_order,
        quad_type=args.sn_quad,
        Lr=args.Lr,
        Lz=args.Lz,
        t_start=args.t_start,
        tfinal=args.tfinal,
        dt_min=args.sn_dt_min,
        dt_max=args.sn_dt_max,
        stretch=args.stretch,
        use_dmd=(not args.sn_no_dmd),
        LOUD=args.loud,
        print_stride=args.print_stride,
        output_times=np.array(args.output_times, dtype=float),
    )


def write_step1_plots(m1_results, args, output_dir):
    r_ref = np.linspace(0.0, args.Lr, 400)
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
    closure_list = list(m1_results.keys())

    for t_tgt in args.output_times:
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, cname in enumerate(closure_list):
            r, T_rad, t_act = extract_m1_radial_profile(m1_results[cname], t_tgt)
            ax.plot(r, T_rad, lw=2, color=colors[i % len(colors)],
                    label=f'M1 {cname} t={t_act:.2f} ns')

        T_ref, r_front = zeldovich_self_similar(r_ref, t_tgt, N=2)
        ax.plot(r_ref, T_ref, 'k--', lw=2, label=f'Self-similar t={t_tgt:.2f} ns')
        ax.axvline(r_front, color='k', ls=':', lw=1, alpha=0.5)

        ax.set_xlabel('r (cm)')
        ax.set_ylabel('Radiation temperature T_r (keV)')
        ax.set_title(f'M1 closure comparison at t={t_tgt:.2f} ns')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        plt.tight_layout()
        out = os.path.join(output_dir, f'step1_closure_compare_t{t_tgt:.2f}ns.png')
        plt.savefig(out, dpi=160, bbox_inches='tight')
        plt.close(fig)


def run_step2_convergence(args, output_dir):
    Ir_vals = [int(x.strip()) for x in args.Ir_convergence.split(',') if x.strip()]
    rows = []

    r_ref = np.linspace(0.0, max(args.Lr, 2.0), 500)
    T_ref, _ = zeldovich_self_similar(r_ref, args.compare_time, N=2)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(r_ref, T_ref, 'k--', lw=2, label='Self-similar')

    for i, Ir in enumerate(Ir_vals):
        Iz = max(4, Ir // 10)
        result = run_zeldovich_rz_m1(
            Ir=Ir,
            Iz=Iz,
            closure_name=args.convergence_closure,
            Lr=args.Lr,
            Lz=args.Lz,
            t_start=args.t_start,
            tfinal=max(args.tfinal, args.compare_time),
            dt_min=args.dt_min,
            dt_max=args.dt_max,
            dt_increase_factor=args.dt_increase_factor,
            stretch=args.stretch,
            output_times=[args.compare_time],
            LOUD=False,
        )
        r_num, T_num, t_act = extract_m1_radial_profile(result, args.compare_time)
        T_interp = np.interp(r_num, r_ref, T_ref)
        l2 = float(np.sqrt(np.mean((T_num - T_interp) ** 2)))
        rl2 = rel_l2(T_num, T_interp)

        rows.append({
            'Ir': Ir,
            'Iz': Iz,
            'closure': args.convergence_closure,
            't_actual_ns': t_act,
            'L2': l2,
            'relL2': rl2,
        })

        ax.plot(r_num, T_num, lw=1.8, label=f'Ir={Ir}, Iz={Iz}')

    ax.set_xlabel('r (cm)')
    ax.set_ylabel('Radiation temperature T_r (keV)')
    ax.set_title(f'Step 2 convergence at t={args.compare_time:.2f} ns')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'step2_convergence_profiles.png')
    plt.savefig(fig_path, dpi=160, bbox_inches='tight')
    plt.close(fig)

    csv_path = os.path.join(output_dir, 'step2_convergence_metrics.csv')
    with open(csv_path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['Ir', 'Iz', 'closure', 't_actual_ns', 'L2', 'relL2'])
        w.writeheader()
        w.writerows(rows)

    return rows


def run_step3_overlay(m1_results, sn_result, args, output_dir):
    rows = []
    colors = {
        'SN': 'k',
        'levermore': '#d62728',
        'kershaw': '#2ca02c',
        'p1': '#1f77b4',
        'minerbo_poly': '#9467bd',
    }

    for t_tgt in args.output_times:
        r_sn, T_sn, t_sn = extract_sn_radial_profile(sn_result, t_tgt)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(r_sn, T_sn, color=colors['SN'], lw=2.2, label=f'S_N t={t_sn:.2f} ns')

        for cname, m1_result in m1_results.items():
            r_m1, T_m1, t_m1 = extract_m1_radial_profile(m1_result, t_tgt)
            T_sn_interp = np.interp(r_m1, r_sn, T_sn)
            rows.append({
                'time_target_ns': t_tgt,
                'method': cname,
                't_actual_ns': t_m1,
                'relL2_vs_SN': rel_l2(T_m1, T_sn_interp),
                'max_rel_pointwise_vs_SN': float(np.max(np.abs(T_m1 - T_sn_interp)) / (np.max(np.abs(T_sn_interp)) + 1e-30)),
                'z_var': m1_z_var(m1_result, t_tgt),
            })

            ax.plot(
                r_m1,
                T_m1,
                lw=2,
                ls='--',
                color=colors.get(cname, None),
                label=f'M1 {cname} t={t_m1:.2f} ns',
            )

        rows.append({
            'time_target_ns': t_tgt,
            'method': 'SN',
            't_actual_ns': t_sn,
            'relL2_vs_SN': 0.0,
            'max_rel_pointwise_vs_SN': 0.0,
            'z_var': sn_z_var(sn_result, t_tgt),
        })

        ax.set_xlabel('r (cm)')
        ax.set_ylabel('Radiation temperature T_r (keV)')
        ax.set_title(f'Step 3 M1 vs S_N at t={t_tgt:.2f} ns')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        plt.tight_layout()
        out = os.path.join(output_dir, f'step3_m1_vs_sn_t{t_tgt:.2f}ns.png')
        plt.savefig(out, dpi=160, bbox_inches='tight')
        plt.close(fig)

    csv_path = os.path.join(output_dir, 'step3_m1_vs_sn_metrics.csv')
    with open(csv_path, 'w', newline='') as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=['time_target_ns', 'method', 't_actual_ns', 'relL2_vs_SN',
                        'max_rel_pointwise_vs_SN', 'z_var'],
        )
        w.writeheader()
        w.writerows(rows)
    return rows


def write_step1_metrics(m1_results, args, output_dir):
    rows = []
    for t_tgt in args.output_times:
        for cname, result in m1_results.items():
            r, T_num, t_act = extract_m1_radial_profile(result, t_tgt)
            T_ref, _ = zeldovich_self_similar(r, t_tgt, N=2)
            rows.append({
                'time_target_ns': t_tgt,
                'method': cname,
                't_actual_ns': t_act,
                'relL2_vs_selfsimilar': rel_l2(T_num, T_ref),
                'max_rel_pointwise_vs_selfsimilar': float(np.max(np.abs(T_num - T_ref)) / (np.max(np.abs(T_ref)) + 1e-30)),
                'z_var': m1_z_var(result, t_tgt),
            })

    csv_path = os.path.join(output_dir, 'step1_m1_closure_metrics.csv')
    with open(csv_path, 'w', newline='') as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=['time_target_ns', 'method', 't_actual_ns',
                        'relL2_vs_selfsimilar', 'max_rel_pointwise_vs_selfsimilar', 'z_var'],
        )
        w.writeheader()
        w.writerows(rows)
    return rows


def write_report_md(args, output_dir, step1_rows, step2_rows, step3_rows):
    def best_by_time(rows, metric_key):
        out = {}
        for row in rows:
            t = float(row['time_target_ns'])
            if row['method'] == 'SN':
                continue
            val = float(row[metric_key])
            if t not in out or val < out[t][1]:
                out[t] = (row['method'], val)
        return out

    step1_best = best_by_time(step1_rows, 'relL2_vs_selfsimilar')
    step3_best = best_by_time(step3_rows, 'relL2_vs_SN')

    lines = []
    lines.append('# M1 r-z Zel\'dovich Validation Report')
    lines.append('')
    lines.append('## Setup')
    lines.append(f'- Grid for closure/S_N comparison: Ir={args.Ir}, Iz={args.Iz}')
    lines.append(f'- Time window: t_start={args.t_start} ns to tfinal={args.tfinal} ns')
    lines.append(f'- Output times: {args.output_times}')
    lines.append(f'- M1 closures: {args.closures}')
    lines.append(f'- S_N quadrature: {args.sn_quad}, order N={args.sn_order}')
    lines.append('')

    lines.append('## Step 1: M1 closure comparison vs self-similar')
    for t in sorted(step1_best):
        m, v = step1_best[t]
        lines.append(f'- t={t:.2f} ns: best closure = {m}, relL2={v:.4e}')
    lines.append('- Figures: step1_closure_compare_t*.png')
    lines.append('- Metrics CSV: step1_m1_closure_metrics.csv')
    lines.append('')

    lines.append('## Step 2: Ir convergence sweep (M1)')
    for row in step2_rows:
        lines.append(
            f"- Ir={row['Ir']}, Iz={row['Iz']}: relL2={float(row['relL2']):.4e} at t={float(row['t_actual_ns']):.3f} ns"
        )
    lines.append('- Figure: step2_convergence_profiles.png')
    lines.append('- Metrics CSV: step2_convergence_metrics.csv')
    lines.append('')

    lines.append('## Step 3: M1 vs S_N overlays')
    for t in sorted(step3_best):
        m, v = step3_best[t]
        lines.append(f'- t={t:.2f} ns: closest M1 closure to S_N = {m}, relL2={v:.4e}')
    lines.append('- Figures: step3_m1_vs_sn_t*.png')
    lines.append('- Metrics CSV: step3_m1_vs_sn_metrics.csv')
    lines.append('')

    lines.append('## Symmetry sanity check')
    sn_rows = [r for r in step3_rows if r['method'] == 'SN']
    if sn_rows:
        lines.append('- S_N z-variation values:')
        for r in sn_rows:
            lines.append(f"  - t={float(r['time_target_ns']):.2f} ns: {float(r['z_var']):.3e}")
    lines.append('- M1 z-variation values are included in both step1 and step3 CSV files.')

    out = os.path.join(output_dir, 'report.md')
    with open(out, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')


def parse_args():
    parser = argparse.ArgumentParser(description='M1 r-z Zel\'dovich validation workflow')
    parser.add_argument('--Ir', type=int, default=60)
    parser.add_argument('--Iz', type=int, default=8)
    parser.add_argument('--Lr', type=float, default=1.5)
    parser.add_argument('--Lz', type=float, default=0.2)
    parser.add_argument('--t-start', type=float, default=0.1)
    parser.add_argument('--tfinal', type=float, default=1.0)
    parser.add_argument('--dt-min', type=float, default=5e-5)
    parser.add_argument('--dt-max', type=float, default=2e-3)
    parser.add_argument('--dt-increase-factor', type=float, default=1.1)
    parser.add_argument('--stretch', type=float, default=1.5)
    parser.add_argument('--output-times', type=float, nargs='+', default=[0.3, 0.5, 1.0])
    parser.add_argument('--closures', type=str, default='levermore,kershaw,p1,minerbo_poly')
    parser.add_argument('--compare-time', type=float, default=0.5)
    parser.add_argument('--convergence-closure', type=str, default='levermore')
    parser.add_argument('--Ir-convergence', type=str, default='30,60,120')

    parser.add_argument('--sn-order', type=int, default=4)
    parser.add_argument('--sn-quad', type=str, default='product_square')
    parser.add_argument('--sn-dt-min', type=float, default=1e-5)
    parser.add_argument('--sn-dt-max', type=float, default=1e-3)
    parser.add_argument('--sn-no-dmd', action='store_true')

    parser.add_argument('--loud', action='store_true')
    parser.add_argument('--print-stride', type=int, default=50)
    parser.add_argument(
        '--output-dir',
        type=str,
        default=os.path.join(_REPO_ROOT, 'M1', 'doc', 'zeldovich_rz_m1_validation'),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    print('\n=== Step 1: run M1 closures ===')
    m1_results = run_closure_suite(args, args.output_dir)
    step1_rows = write_step1_metrics(m1_results, args, args.output_dir)
    write_step1_plots(m1_results, args, args.output_dir)

    print('\n=== Step 2: convergence sweep ===')
    step2_rows = run_step2_convergence(args, args.output_dir)

    print('\n=== Step 3: run S_N and overlay ===')
    sn_result = run_sn_reference(args)
    step3_rows = run_step3_overlay(m1_results, sn_result, args, args.output_dir)

    print('\n=== Write report ===')
    write_report_md(args, args.output_dir, step1_rows, step2_rows, step3_rows)

    print(f'\nDone. Report folder: {args.output_dir}')


if __name__ == '__main__':
    main()
