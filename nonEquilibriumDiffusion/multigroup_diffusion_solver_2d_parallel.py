#!/usr/bin/env python3
"""
Thread-Parallel 2D Multigroup Diffusion Solver
===============================================
Subclass of MultigroupDiffusionSolver2D that executes the G independent
per-group diffusion solves concurrently using a persistent ThreadPoolExecutor.

Why threads work here
---------------------
The inner solve per group is scipy.sparse.linalg.spsolve, which calls the
UMFPACK or SuperLU C/Fortran libraries.  These release the GIL during
the factorization and back-substitution, so Python threads genuinely run
on multiple CPU cores for those sections.

Each group owns its own DiffusionOperatorSolver2D instance (self.solvers[g]),
so the matrix cache (self._cached_T, self._cached_A) is never shared between
threads.

Thread-safety summary
---------------------
- self.sigma_a, self.chi, self.fleck_factor : read-only inside _solve (safe)
- self.solvers[g] : each accessed by exactly one thread (safe)
- result / rhs accumulation : done serially in the main thread after all
  futures complete (safe)

Performance characteristics
---------------------------
On the 60×210 production grid with G=10 groups you can expect roughly 1.2–1.3×
end-to-end speedup with 4–10 threads.

The modest speedup is due to Amdahl's law: the diffusion-coefficient evaluation
inside assemble_matrix (Python-level loop over ~25 k face positions per group)
holds the GIL, so only the spsolve portion (~60–70 % of each group's work) is
truly parallel.  If D(T,x,y) functions are later replaced with vectorised NumPy
or Numba implementations, a much larger fraction of the group solve will become
GIL-free and the speedup will increase proportionally.

Why a separate file
-------------------
The parallel version adds concurrency complexity.  Keeping the serial
MultigroupDiffusionSolver2D clean and stable makes it easy to compare,
debug, and fall back to the simpler code when needed.

Usage
-----
    from multigroup_diffusion_solver_2d_parallel import (
        MultigroupDiffusionSolver2DParallel
    )

    # Use as a context manager to ensure the thread pool is shut down cleanly
    with MultigroupDiffusionSolver2DParallel(..., n_threads=4) as solver:
        solver.step()

    # Or manage lifetime manually
    solver = MultigroupDiffusionSolver2DParallel(..., n_threads=4)
    solver.step()
    solver.close()

MPI note
--------
For distributed-memory (multi-node) parallelism, each MPI rank would own a
subset of groups and reduce the accumulated results with MPI_Allreduce.
That path requires mpi4py and a different launch mechanism; threads are the
simpler first step for a shared-memory workstation.
"""

import os
import sys
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from diffusion_operator_solver import C_LIGHT
from multigroup_diffusion_solver_2d import (
    MultigroupDiffusionSolver2D,
    flatten_2d,
    unflatten_2d,
)


class MultigroupDiffusionSolver2DParallel(MultigroupDiffusionSolver2D):
    """
    Thread-parallel version of MultigroupDiffusionSolver2D.

    Overrides the three methods that contain independent per-group sparse
    solves, running them concurrently with a persistent ThreadPoolExecutor.
    Everything else (Newton loop, GMRES, preconditioner, …) is inherited
    unchanged from the base class.
    """

    def __init__(self, *args, n_threads: Optional[int] = None, **kwargs):
        """
        Parameters
        ----------
        n_threads : int or None
            Worker-thread count.  Defaults to min(n_groups, cpu_count).
            Pass n_threads=1 to disable parallelism for debugging.
        All other arguments are forwarded unchanged to
        MultigroupDiffusionSolver2D.__init__.
        """
        super().__init__(*args, **kwargs)
        cpu_count = os.cpu_count() or 1
        self.n_threads = n_threads if n_threads is not None else min(self.n_groups, cpu_count)
        self._executor: Optional[ThreadPoolExecutor] = ThreadPoolExecutor(
            max_workers=self.n_threads
        )
        print(f"  Parallel solver: {self.n_threads} threads for {self.n_groups} groups")

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def close(self):
        """Shut down the thread pool gracefully."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _parallel_map(self, fn, n_groups: int):
        """Submit fn(g) for g in range(n_groups), return list of results in order."""
        return list(self._executor.map(fn, range(n_groups)))

    # ------------------------------------------------------------------
    # Parallel overrides
    # ------------------------------------------------------------------

    def apply_operator_B(self, kappa: np.ndarray, T_star: np.ndarray,
                         xi_g_list: List[np.ndarray],
                         bc_time: float = 0.0) -> np.ndarray:
        """
        B·κ = κ - Σ_g σ*_{a,g} · A_g^{-1}[χ_g(1-f)κ]

        The G solves are executed concurrently (one task per group).
        """
        f = self.fleck_factor
        T_2d = unflatten_2d(T_star, self.nx_cells, self.ny_cells)

        # Pre-compute RHS vectors in the main thread.
        nu_kappa = (1.0 - f) * kappa
        rhs_vecs = [self.chi[g] * nu_kappa for g in range(self.n_groups)]

        solvers = self.solvers
        sigma_a = self.sigma_a
        nx, ny  = self.nx_cells, self.ny_cells

        # Homogeneous-BC solve: C_bc = 0 everywhere.
        # skip_bc_rhs=True means the per-cell BC function calls are skipped
        # entirely — the cached LU back-sub is the only work done here.
        # This also removes all Python-GIL-holding loops from the threaded
        # path, so threads truly run in parallel on the C-level LU solves.
        def _solve(g):
            rhs_g_2d = unflatten_2d(rhs_vecs[g], nx, ny)
            phi_g_2d = solvers[g].solve(
                rhs_g_2d, T_2d,
                bc_time=bc_time,
                skip_bc_rhs=True,
            )
            phi_g = flatten_2d(phi_g_2d, nx, ny)
            return sigma_a[g, :] * phi_g

        result = kappa.copy().astype(np.float64)
        for contrib in self._parallel_map(_solve, self.n_groups):
            result -= contrib
        return result

    def compute_rhs_for_kappa(self, T_star: np.ndarray,
                               xi_g_list: List[np.ndarray],
                               bc_time: float = 0.0) -> np.ndarray:
        """
        RHS = Σ_g σ*_{a,g} · A_g^{-1} · ξ_g

        The G solves are executed concurrently.
        """
        T_2d    = unflatten_2d(T_star, self.nx_cells, self.ny_cells)
        solvers = self.solvers
        sigma_a = self.sigma_a
        nx, ny  = self.nx_cells, self.ny_cells

        def _solve(g):
            xi_g_2d  = unflatten_2d(xi_g_list[g], nx, ny)
            phi_g_2d = solvers[g].solve(xi_g_2d, T_2d, bc_time=bc_time)
            phi_g    = flatten_2d(phi_g_2d, nx, ny)
            return sigma_a[g, :] * phi_g

        rhs = np.zeros(self.n_total)
        for contrib in self._parallel_map(_solve, self.n_groups):
            rhs += contrib
        return rhs

    def compute_radiation_energy_from_kappa(
            self, kappa: np.ndarray, T_star: np.ndarray,
            xi_g_list: List[np.ndarray],
            bc_time: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        φ_g^{n+1} = A_g^{-1}(χ_g(1-f)κ + ξ_g)   for each g (parallel)
        E_r = (1/c) Σ_g φ_g^{n+1}
        """
        f       = self.fleck_factor
        T_2d    = unflatten_2d(T_star, self.nx_cells, self.ny_cells)
        solvers = self.solvers
        nx, ny  = self.nx_cells, self.ny_cells

        nu_kappa = (1.0 - f) * kappa
        rhs_vecs = [self.chi[g] * nu_kappa + xi_g_list[g] for g in range(self.n_groups)]

        def _solve(g):
            rhs_g_2d = unflatten_2d(rhs_vecs[g], nx, ny)
            phi_g_2d = solvers[g].solve(rhs_g_2d, T_2d, bc_time=bc_time)
            return flatten_2d(phi_g_2d, nx, ny)

        phi_g_array = np.array(self._parallel_map(_solve, self.n_groups))  # (G, N)

        phi_total = phi_g_array.sum(axis=0)
        E_r = phi_total / C_LIGHT

        phi_g_fraction = phi_g_array / (phi_total[np.newaxis, :] + 1e-30)
        self.phi_g_stored[:] = phi_g_array

        return E_r, phi_g_fraction
