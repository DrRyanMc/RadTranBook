"""
Compare DMD-accelerated vs pure Richardson (no-DMD) source iteration
for the simple Marshak wave with a fixed linearisation temperature
(W=0, single Fleck-Cummings linearisation per time step).

Produces two figures:
  1. Source iterations per time step vs time — shows the iteration-count
     savings from DMD.
  2. Material temperature profiles at 1, 5, and 10 ns for DMD, no-DMD,
     and the self-similar reference.

Run from the DiscreteOrdinates directory:
    python problems/compare_dmd_vs_richardson.py
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
        description='DMD vs Richardson comparison for simple Marshak wave')
    parser.add_argument('--zones',   type=int,   default=50)
    parser.add_argument('--N',       type=int,   default=8)
    parser.add_argument('--L',       type=float, default=0.20)
    parser.add_argument('--tfinal',  type=float, default=10.0)
    parser.add_argument('--dt-min',  type=float, default=0.01)
    parser.add_argument('--dt-max',  type=float, default=0.01)
    parser.add_argument('--K',        type=int,   default=10,
                        help='DMD snapshot count (default: 10)')
    parser.add_argument('--R',        type=int,   default=3,
                        help='Richardson steps between DMD updates (default: 3)')
    parser.add_argument('--tolerance', type=float, default=1e-8,
                        help='Inner iteration convergence tolerance (default: 1e-8)')
    parser.add_argument('--maxits',  type=int,   default=10000,
                        help='Max inner iterations per step (default: 10000)')
    parser.add_argument('--save-fig', type=str,  default='',
                        help='Save figures with this prefix (e.g. "dmd_compare"); '
                             'leave empty to display interactively')
    args = parser.parse_args()

    plot_times = [1.0, 5.0, args.tfinal]

    # ── run no-DMD (Richardson only) ─────────────────────────────────────────
    print("=" * 60)
    print("Run 1: Richardson (no DMD), W=0")
    print("=" * 60)
    r_rich = smw.setup_and_run(
        I=args.zones, N=args.N, L=args.L, tfinal=args.tfinal,
        dt_min=args.dt_min, dt_max=args.dt_max,
        K=args.K, R=args.R, maxits=args.maxits, tolerance=args.tolerance,
        use_dmd=False, W=0,
        time_outputs=None,
    )

    # ── run DMD ──────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("Run 2: DMD-accelerated, W=0")
    print("=" * 60)
    r_dmd = smw.setup_and_run(
        I=args.zones, N=args.N, L=args.L, tfinal=args.tfinal,
        dt_min=args.dt_min, dt_max=args.dt_max,
        K=args.K, R=args.R, maxits=args.maxits, tolerance=args.tolerance,
        use_dmd=True, W=0,
        time_outputs=None,
    )

    print()
    print(f"Total sweeps — Richardson: {r_rich['iterations']:,}   "
          f"DMD: {r_dmd['iterations']:,}   "
          f"speedup: {r_rich['iterations'] / r_dmd['iterations']:.2f}×")

    x        = r_rich['x']
    ts_rich  = np.asarray(r_rich['ts'])
    ts_dmd   = np.asarray(r_dmd['ts'])
    its_rich = np.asarray(r_rich['its_per_step'])
    its_dmd  = np.asarray(r_dmd['its_per_step'])

    # ts[0] = t=0, no solve was done there; its_per_step[k] corresponds to ts[k+1]
    step_times_rich = ts_rich[1:]
    step_times_dmd  = ts_dmd[1:]

    # ── Figure 1: iterations per step vs time ────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.semilogy(step_times_rich, its_rich, color='C1', lw=1.2,
                 label=f'Richardson  (total {r_rich["iterations"]:,})')
    ax1.semilogy(step_times_dmd,  its_dmd,  color='C0', lw=1.2,
                 label=f'DMD  (total {r_dmd["iterations"]:,})')
    ax1.set_xlabel('Time (ns)')
    ax1.set_ylabel('Source iterations per time step')
    ax1.set_title('Iterations per time step — DMD vs Richardson\n'
                  f'Simple Marshak wave  (I={args.zones}, N={args.N}, '
                  f'dt={args.dt_min} ns, maxits={args.maxits})')
    ax1.legend()
    plt.tight_layout()
    if args.save_fig:
        name = f'{args.save_fig}_iterations.pdf'
        fig1.savefig(name, dpi=300, bbox_inches='tight')
        print(f'Saved {name}')
    else:
        plt.show()

    # ── Figure 2: temperature profiles at plot_times ─────────────────────────
    fig2, ax2 = plt.subplots(figsize=(8, 5))

    for i, tval in enumerate(plot_times):
        color = f'C{i}'

        # Richardson
        step_r = _find_step(ts_rich, tval)
        tact_r = float(ts_rich[step_r])
        T_rich = _centres(r_rich['Ts'][step_r], x)
        ax2.plot(x, T_rich, '-', color=color, alpha=0.7, lw=1.5,
                 label=f'Richardson  $t={tact_r:.1f}$ ns')

        # DMD
        step_d = _find_step(ts_dmd, tval)
        tact_d = float(ts_dmd[step_d])
        T_dmd  = _centres(r_dmd['Ts'][step_d], x)
        ax2.plot(x, T_dmd, '--', color=color, lw=1.5,
                 label=f'DMD  $t={tact_d:.1f}$ ns')

        # Self-similar
        r_ss, T_ss = smw.self_similar_solution(tval, r_rich)
        ax2.plot(r_ss, T_ss, ':', color=color, lw=1.8,
                 label=f'Self-similar  $t={tval:.1f}$ ns')

    ax2.set_xlim(0, args.L)
    ax2.set_xlabel('Position (cm)')
    ax2.set_ylabel('Material temperature (keV)')
    ax2.set_title('Temperature profiles — DMD vs Richardson vs self-similar\n'
                  f'Simple Marshak wave  (I={args.zones}, N={args.N}, '
                  f'dt={args.dt_min} ns)')
    ax2.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    if args.save_fig:
        name = f'{args.save_fig}_profiles.pdf'
        fig2.savefig(name, dpi=300, bbox_inches='tight')
        print(f'Saved {name}')
    else:
        plt.show()


if __name__ == '__main__':
    main()
