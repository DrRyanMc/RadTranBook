#!/usr/bin/env python3
"""Converging Marshak Wave MPI driver (spherical geometry).

Run with:
  mpiexec -n 4 python IMC/ConvergingMarshakWaveMPI.py --run-full --numba-threads 4

This uses IMC1DMPI (MPI across ranks + numba threads within rank).
Only rank 0 writes figures/NPZ and prints progress.
"""

import argparse
import os
import sys

import numpy as np
from mpi4py import MPI

# Keep local IMC directory importable.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import IMC1DMPI as imc
import ConvergingMarshakWave as base


def _configure_numba_threads(num_threads, rank):
    """Set numba worker threads per rank when requested."""
    if num_threads is None:
        return

    from numba import set_num_threads, get_num_threads

    if num_threads < 1:
        raise ValueError("--numba-threads must be >= 1")

    set_num_threads(num_threads)
    if rank == 0:
        print(f"Numba threads per rank: {get_num_threads()}")


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Converging Marshak Wave IMC MPI driver (spherical).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n-cells", type=int, default=500)
    parser.add_argument("--Ntarget", type=int, default=2 * 10**5)
    parser.add_argument("--Nboundary", type=int, default=10**5)
    parser.add_argument("--NMax", type=int, default=10**6)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--output-freq", type=int, default=10)
    parser.add_argument("--save-prefix", type=str, default="converging_marshak_wave_imc_mpi")

    parser.add_argument(
        "--final-output-time",
        type=float,
        default=base.OUTPUT_TIMES_NS[-1],
        help="Final physical time [ns] to stop the run.",
    )
    parser.add_argument(
        "--run-full",
        action="store_true",
        help="Ignore --final-output-time and run to T_FINAL_NS.",
    )

    parser.add_argument("--numba-threads", type=int, default=None)
    return parser


def run_mpi(
    n_cells,
    Ntarget,
    Nboundary,
    NMax,
    dt,
    output_freq,
    save_prefix,
    final_output_time,
):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    duration = final_output_time - base.T_INIT_NS
    output_elapsed = sorted(
        t - base.T_INIT_NS for t in base.OUTPUT_TIMES_NS if t <= final_output_time + 1e-9
    )

    if rank == 0:
        print("=" * 70)
        print("Converging Marshak Wave  (IMC / spherical geometry / MPI)")
        print("=" * 70)
        print(f"  MPI ranks={size}")
        print(
            f"  n_cells={n_cells},  Ntarget={Ntarget},  "
            f"Nboundary={Nboundary},  NMax={NMax}"
        )
        print(f"  dt={dt} ns,  total elapsed={duration:.4f} ns")
        print(
            f"  T_INIT_NS={base.T_INIT_NS:.6f} ns,  T_FINAL_NS={final_output_time} ns"
        )
        print("=" * 70)

    r_edges = np.linspace(0.0, base.R, n_cells + 1)
    mesh = np.column_stack([r_edges[:-1], r_edges[1:]])
    r_mid = 0.5 * (mesh[:, 0] + mesh[:, 1])

    T_init_keV = 1e-4
    T_init = np.full(n_cells, T_init_keV)
    Tr_init = np.full(n_cells, T_init_keV)

    T_boundary = (0.0, base.outer_T_keV)
    source = np.zeros(n_cells)

    state = imc.init_simulation(
        Ntarget,
        T_init,
        Tr_init,
        mesh,
        base.eos,
        base.inv_eos,
        geometry="spherical",
        comm=comm,
    )

    snapshots = []
    output_saved = set()
    output_t_ns = sorted(t for t in base.OUTPUT_TIMES_NS if t <= final_output_time + 1e-9)
    step_count = 0

    while state.time < duration - 1e-12:
        step_dt = min(dt, duration - state.time)
        for tau_out in output_elapsed:
            if tau_out > state.time and state.time + step_dt > tau_out + 1e-12:
                step_dt = tau_out - state.time
                break

        state, info = imc.step(
            state,
            Ntarget,
            Nboundary,
            0,
            NMax,
            T_boundary,
            step_dt,
            mesh,
            base.opacity,
            base.inv_eos,
            base.cv,
            source,
            reflect=(True, False),
            geometry="spherical",
        )

        step_count += 1
        t_phys = base.T_INIT_NS + state.time

        if rank == 0 and (step_count % output_freq == 0 or step_count <= 2):
            T_bath_HeV = base.T_bath_keV(t_phys) * base.T_HEV_PER_KEV
            T_surf_HeV = base.T_analytic_keV(base.R, t_phys) * base.T_HEV_PER_KEV
            print(
                f"  step {step_count:5d}  t_phys={t_phys:9.4f} ns"
                f"  T_surf={T_surf_HeV:.4f} HeV"
                f"  T_bath={T_bath_HeV:.4f} HeV"
                f"  T_center={state.temperature[0]*base.T_HEV_PER_KEV:.4f} HeV"
                f"  N={info['N_particles']:6d}"
                f"  dE={info['energy_loss']:.3e} GJ"
            )

        for t_out in output_t_ns:
            if t_out not in output_saved and abs(t_phys - t_out) < 1e-9:
                if rank == 0:
                    snapshots.append((t_out, state.temperature.copy(), r_mid.copy()))
                    T_max_HeV = state.temperature.max() * base.T_HEV_PER_KEV
                    print(
                        f"  >> Snapshot at t_phys={t_out:.4f} ns"
                        f"  T_max={T_max_HeV:.4f} HeV"
                    )
                output_saved.add(t_out)

    if rank == 0:
        print(
            f"\nDone. Total steps: {step_count}, "
            f"final t_phys={base.T_INIT_NS + state.time:.4f} ns"
        )

        if snapshots:
            base.plot_results(snapshots, save_prefix=save_prefix)

            snap_times = np.array([s[0] for s in snapshots])
            snap_T = np.array([s[1] for s in snapshots])
            snap_r = np.array([s[2] for s in snapshots])
            npz_path = f"{save_prefix}.npz"
            np.savez(
                npz_path,
                snap_times=snap_times,
                snap_T_keV=snap_T,
                snap_r_mid=snap_r,
                n_cells=n_cells,
                Ntarget=Ntarget,
                Nboundary=Nboundary,
                NMax=NMax,
                dt=dt,
                T_INIT_NS=base.T_INIT_NS,
                R=base.R,
                mpi_ranks=size,
            )
            print(f"Saved: {npz_path}")
        else:
            print("No snapshots captured - check output time clipping.")

    comm.Barrier()
    return state, snapshots


def main(argv=None):
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    _configure_numba_threads(args.numba_threads, rank)

    final_output_time = base.T_FINAL_NS if args.run_full else args.final_output_time

    run_mpi(
        n_cells=args.n_cells,
        Ntarget=args.Ntarget,
        Nboundary=args.Nboundary,
        NMax=args.NMax,
        dt=args.dt,
        output_freq=args.output_freq,
        save_prefix=args.save_prefix,
        final_output_time=final_output_time,
    )


if __name__ == "__main__":
    main()
