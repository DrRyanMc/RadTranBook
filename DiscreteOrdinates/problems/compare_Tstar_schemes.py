"""
Compare temperature-update schemes for the simple Marshak wave.

Four runs: {Richardson, DMD} × {W=0 (fixed T*), W=5 (iterated T*)}.
Fixed time step 0.025 ns by default.

Produces two figures:
  1. Source iterations per time step vs time for all four methods.
  2. Material temperature profiles at 1, 5, and tfinal ns for all four
     methods plus the self-similar reference.

Run from the DiscreteOrdinates directory:
    python problems/compare_Tstar_schemes.py [--save-fig PREFIX]
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import simple_marshak_wave_ld as smw


# ── helpers ───────────────────────────────────────────────────────────────────

def _centres(vals, x):
    """Cell-centre averages of LD left/right-edge values."""
    return 0.5 * (vals[:, 0] + vals[:, 1])


def _find_step(ts, tval):
    return int(np.argmin(np.abs(np.asarray(ts) - tval)))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='T* scheme comparison for simple Marshak wave')
    parser.add_argument('--zones',    type=int,   default=50)
    parser.add_argument('--N',        type=int,   default=8)
    parser.add_argument('--L',        type=float, default=0.20)
    parser.add_argument('--tfinal',   type=float, default=10.0)
    parser.add_argument('--dt-min',   type=float, default=0.025,
                        help='Fixed time step (default: 0.025 ns)')
    parser.add_argument('--dt-max',   type=float, default=0.025,
                        help='Fixed time step (default: 0.025 ns)')
    parser.add_argument('--W',        type=int,   default=5,
                        help='Max T* outer iterations for the W>0 runs (default: 5)')
    parser.add_argument('--K',        type=int,   default=10,
                        help='DMD snapshot count (default: 10)')
    parser.add_argument('--R',        type=int,   default=3,
                        help='Richardson steps between DMD updates (default: 3)')
    parser.add_argument('--tolerance', type=float, default=1e-8,
                        help='Inner convergence tolerance (default: 1e-8)')
    parser.add_argument('--maxits',   type=int,   default=10000,
                        help='Max inner iterations per step (default: 10000)')
    parser.add_argument('--tau-phi-max', type=float, default=0.1,
                        help='Loose inner tolerance for first outer iter (default: 0.1)')
    parser.add_argument('--tau-T',    type=float, default=1e-6,
                        help='T* outer convergence tolerance (default: 1e-6)')
    parser.add_argument('--omega-T',  type=float, default=0.5,
                        help='T* damping factor: 1=undamped, <1=damped (default: 0.5)')
    parser.add_argument('--save-fig', type=str,   default='',
                        help='Prefix for saved PDFs; empty = interactive display')
    args = parser.parse_args()

    dt   = args.dt_min
    W    = args.W
    common = dict(
        I=args.zones, N=args.N, L=args.L, tfinal=args.tfinal,
        dt_min=dt, dt_max=dt,
        K=args.K, R=args.R, maxits=args.maxits, tolerance=args.tolerance,
        time_outputs=None,
        tau_phi_max=args.tau_phi_max, tau_T=args.tau_T,
        omega_T=args.omega_T,
    )

    runs = [
        ('Richardson  W=0',  dict(use_dmd=False, W=0)),
        (f'Richardson  W={W}', dict(use_dmd=False, W=W)),
        ('DMD  W=0',         dict(use_dmd=True,  W=0)),
        (f'DMD  W={W}',      dict(use_dmd=True,  W=W)),
    ]

    results = {}
    for label, kw in runs:
        print()
        print('=' * 60)
        print(f'Run: {label}')
        print('=' * 60)
        results[label] = smw.setup_and_run(**common, **kw)

    print()
    print('Total sweep counts:')
    ref = results['Richardson  W=0']['iterations']
    for label, _ in runs:
        tot = results[label]['iterations']
        print(f'  {label:25s}  {tot:8,}   ({ref/tot:.2f}× vs Richardson W=0)')

    x = results['Richardson  W=0']['x']

    # ── line styles per run ───────────────────────────────────────────────────
    styles = {
        'Richardson  W=0':       dict(color='C1', ls='-',  lw=1.2),
        f'Richardson  W={W}':    dict(color='C1', ls='--', lw=1.2),
        'DMD  W=0':              dict(color='C0', ls='-',  lw=1.2),
        f'DMD  W={W}':           dict(color='C0', ls='--', lw=1.2),
    }

    # ── Figure 1: iterations per step ────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    for label, _ in runs:
        r   = results[label]
        ts  = np.asarray(r['ts'])
        its = np.asarray(r['its_per_step'])
        st  = styles[label]
        ax1.semilogy(ts[1:], its,
                     color=st['color'], ls=st['ls'], lw=st['lw'],
                     label=f'{label}  (total {r["iterations"]:,})')
    ax1.set_xlabel('Time (ns)')
    ax1.set_ylabel('Source iterations per time step')
    ax1.set_title('Iterations per time step — T* scheme comparison\n'
                  f'Simple Marshak wave  (I={args.zones}, N={args.N}, '
                  f'dt={dt} ns)')
    ax1.legend(fontsize=8)
    plt.tight_layout()
    if args.save_fig:
        name = f'{args.save_fig}_iterations.pdf'
        fig1.savefig(name, dpi=300, bbox_inches='tight')
        print(f'Saved {name}')
    else:
        plt.show()

    # ── Figure 2: temperature profiles ───────────────────────────────────────
    plot_times = [1.0, 5.0, args.tfinal]
    time_colors = ['C2', 'C3', 'C4']

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for i, tval in enumerate(plot_times):
        tc = time_colors[i]
        # self-similar reference (use Richardson W=0 result for metadata)
        r_ss, T_ss = smw.self_similar_solution(tval, results['Richardson  W=0'])
        ax2.plot(r_ss, T_ss, ':', color=tc, lw=2.0,
                 label=f'Self-similar  $t={tval:.0f}$ ns')

        for label, _ in runs:
            r  = results[label]
            ts = np.asarray(r['ts'])
            step = _find_step(ts, tval)
            T    = _centres(r['Ts'][step], x)
            st   = styles[label]
            ax2.plot(x, T,
                     color=tc, ls=st['ls'], lw=1.4, alpha=0.85,
                     label=f'{label}  $t={float(ts[step]):.1f}$ ns')

    ax2.set_xlim(0, args.L)
    ax2.set_xlabel('Position (cm)')
    ax2.set_ylabel('Material temperature (keV)')
    ax2.set_title('Temperature profiles — T* scheme comparison\n'
                  f'Simple Marshak wave  (I={args.zones}, N={args.N}, '
                  f'dt={dt} ns)')
    ax2.legend(fontsize=6, ncol=3)
    plt.tight_layout()
    if args.save_fig:
        name = f'{args.save_fig}_profiles.pdf'
        fig2.savefig(name, dpi=300, bbox_inches='tight')
        print(f'Saved {name}')
    else:
        plt.show()


if __name__ == '__main__':
    main()
