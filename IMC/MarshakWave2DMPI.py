#!/usr/bin/env python3
"""MPI driver for 2D Marshak-wave cases using IMC2DMPI.

Examples:
  mpiexec -n 4 python IMC/MarshakWave2DMPI.py --geometry xy --direction x --run-full
  mpiexec -n 8 python IMC/MarshakWave2DMPI.py --geometry rz --direction z --r0 10 --dr 0.2
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import IMC2DMPI as imc2dmpi


CV_VAL = 0.3


def sigma_a_f(T):
    T_safe = np.maximum(T, 1e-8)
    return 300.0 * T_safe ** -3


def eos(T):
    return CV_VAL * T


def inv_eos(u):
    return u / CV_VAL


def cv(T):
    return 0.0 * T + CV_VAL


def _configure_numba_threads(num_threads, rank):
    if num_threads is None:
        return
    from numba import set_num_threads, get_num_threads

    if num_threads < 1:
        raise ValueError("--numba-threads must be >= 1")

    set_num_threads(num_threads)
    if rank == 0:
        print(f"Numba threads per rank: {get_num_threads()}")


def _average_profile(T, geometry, edges1, edges2, direction):
    if geometry == "xy":
        if direction == "x":
            x_centers = 0.5 * (edges1[:-1] + edges1[1:])
            return x_centers, T.mean(axis=1)
        y_centers = 0.5 * (edges2[:-1] + edges2[1:])
        return y_centers, T.mean(axis=0)

    # rz: return z profile, radially volume-weighted.
    z_centers = 0.5 * (edges2[:-1] + edges2[1:])
    vols = np.pi * (edges1[1:] ** 2 - edges1[:-1] ** 2)[:, None] * np.diff(edges2)[None, :]
    w = vols / np.sum(vols, axis=0, keepdims=True)
    return z_centers, np.sum(w * T, axis=0)


def main():
    parser = argparse.ArgumentParser(description="MPI 2D Marshak-wave driver.")
    parser.add_argument("--geometry", choices=["xy", "rz"], default="xy")
    parser.add_argument("--direction", choices=["x", "y", "z"], default="x")

    parser.add_argument("--L", type=float, default=0.2)
    parser.add_argument("--n1", type=int, default=60, help="cells in primary axis (x or r)")
    parser.add_argument("--n2", type=int, default=60, help="cells in secondary axis (y or z)")
    parser.add_argument("--r0", type=float, default=10.0, help="rz inner radius")
    parser.add_argument("--dr", type=float, default=0.2, help="rz radial thickness")

    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--times", type=float, nargs="+", default=[1.0, 5.0, 10.0])

    parser.add_argument("--Ntarget", type=int, default=20000)
    parser.add_argument("--Nboundary", type=int, default=12000)
    parser.add_argument("--Nmax", type=int, default=120000)

    parser.add_argument("--output-freq", type=int, default=10)
    parser.add_argument("--save-prefix", type=str, default="marshak_wave_2d_mpi")
    parser.add_argument("--numba-threads", type=int, default=None)
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    _configure_numba_threads(args.numba_threads, rank)

    output_times = sorted(args.times)

    if args.geometry == "xy":
        edges1 = np.linspace(0.0, args.L, args.n1 + 1)
        edges2 = np.linspace(0.0, args.L, args.n2 + 1)

        if args.direction == "x":
            T_boundary = (1.0, 0.0, 0.0, 0.0)
            reflect = (False, True, True, True)
        elif args.direction == "y":
            T_boundary = (0.0, 0.0, 1.0, 0.0)
            reflect = (True, True, False, True)
        else:
            raise ValueError("For geometry=xy, direction must be x or y")
    else:
        if args.direction != "z":
            raise ValueError("For geometry=rz, direction must be z")
        edges1 = np.linspace(args.r0, args.r0 + args.dr, args.n1 + 1)
        edges2 = np.linspace(0.0, args.L, args.n2 + 1)
        T_boundary = (0.0, 0.0, 1.0, 0.0)
        reflect = (True, True, False, True)

    if rank == 0:
        print("=" * 70)
        print("2D Marshak MPI run")
        print("=" * 70)
        print(f"  ranks={size}, geometry={args.geometry}, direction={args.direction}")
        print(f"  Ntarget={args.Ntarget}, Nboundary={args.Nboundary}, Nmax={args.Nmax}")
        print(f"  dt={args.dt}, times={output_times}")
        print("=" * 70)

    nx = len(edges1) - 1
    ny = len(edges2) - 1
    Tinit = np.full((nx, ny), 1e-4)
    Trinit = np.full((nx, ny), 1e-4)
    source = np.zeros((nx, ny))

    state = imc2dmpi.init_simulation(
        args.Ntarget,
        Tinit,
        Trinit,
        edges1,
        edges2,
        eos,
        inv_eos,
        geometry=args.geometry,
        comm=comm,
    )

    snapshots = []
    for tout in output_times:
        while state.time < tout - 1e-12:
            step_dt = min(args.dt, tout - state.time)
            state, info = imc2dmpi.step(
                state,
                args.Ntarget,
                args.Nboundary,
                0,
                args.Nmax,
                T_boundary,
                step_dt,
                edges1,
                edges2,
                sigma_a_f,
                inv_eos,
                cv,
                source,
                reflect=reflect,
                geometry=args.geometry,
                rz_linear_source=True,
            )
            if rank == 0 and ((state.count - 1) % args.output_freq == 0):
                print(
                    f" step={state.count:6d} t={info['time']:.4f} ns "
                    f"N={info['N_particles']:8d} dE={info['energy_loss']:.3e}"
                )

        snapshots.append((tout, state.temperature.copy()))

    if rank == 0:
        # Save z/x averaged profiles by snapshot.
        x_like = None
        profs = []
        for t, T in snapshots:
            x_like, p = _average_profile(T, args.geometry, edges1, edges2, args.direction)
            profs.append(p)

        npz_path = f"{args.save_prefix}.npz"
        np.savez(
            npz_path,
            times=np.array([t for t, _ in snapshots]),
            coord=x_like,
            profiles=np.array(profs),
            geometry=args.geometry,
            direction=args.direction,
            ranks=size,
            n1=args.n1,
            n2=args.n2,
            dt=args.dt,
        )

        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for (t, _), p in zip(snapshots, profs):
            ax.plot(x_like, p, lw=2, label=f"t={t:g} ns")
        ax.set_xlabel("distance (cm)")
        ax.set_ylabel("T (keV)")
        ax.set_title(f"2D Marshak MPI ({args.geometry}, dir={args.direction})")
        ax.grid(alpha=0.3)
        ax.legend()
        plt.tight_layout()
        fig_path = f"{args.save_prefix}.png"
        plt.savefig(fig_path, dpi=160)
        plt.close(fig)

        print(f"Saved: {npz_path}")
        print(f"Saved: {fig_path}")

    comm.Barrier()


if __name__ == "__main__":
    main()
