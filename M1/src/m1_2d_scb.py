"""2-D M1 solver using a Simple-Corner-Balance (SCB) spatial discretisation.

Motivation
----------
The plain first-order finite-volume M1 solver (``M1Solver2D`` in
``m1_2d.py``) applies full Rusanov numerical dissipation at every cell
face.  In optically thick regions this can smear the solution relative
to the diffusion limit unless the mesh resolves the mean free path —
exactly the same issue that motivates the Simple Corner Balance (SCB)
discretisation used for the P_N solver (``SphericalHarmonics/pn_solver_2d.py``,
``pn_solver_2d_rz.py``).

SCB idea (mirrors the P_N corner-balance scheme, Sec. 12.3.1)
--------------------------------------------------------------
Each original ("coarse") cell is split into 4 corner sub-cells (NE, NW,
SW, SE) of half the width in each direction.  Faces between corners of
the *same* coarse cell ("internal" faces) use a mostly-centred coupling
with reduced Rusanov dissipation; faces coincident with the *original*
coarse-cell boundaries ("external" faces) always use the full upwinded
Rusanov coupling, exactly as in the plain FV scheme.  This gives the
solution sub-cell (linear-discontinuous-like) resolution without adding
full numerical dissipation at the sub-cell scale, while the
numerical-diffusion length scale stays tied to the *original* (coarse)
mesh spacing — the same mechanism that makes SCB accurate for the P_N
solver in optically thick regions.

Opacity-dependent internal dissipation
----------------------------------------
Removing *all* dissipation at internal faces (fraction 0.0) supports an
undamped high-frequency "checkerboard" null mode between the two corner
sub-cells sharing an internal face — this mode can be excited at sharp
fronts/material interfaces and grow without bound.  A *fixed* nonzero
dissipation fraction damps it but sacrifices most of SCB's accuracy
benefit everywhere, including deep in the diffusive regime where it is
not needed.

Physical argument (per user insight): Rusanov dissipation is a
wave-speed (hyperbolic, ~c/Δx) numerical viscosity.  In the optically
THICK limit the true physical diffusion coefficient (~1/(3σ)) is small,
so *any* fixed c/Δx-scale dissipation swamps it — SCB's benefit is
precisely in reducing this at internal faces.  In the optically THIN
limit, the equations are genuinely hyperbolic/advective and want a
properly upwinded (fully dissipative) scheme for a stable, accurate
free-streaming solution.  So the internal-face dissipation fraction
should *decrease* with increasing local optical thickness:

    tau           = sigma_face * dx_coarse      (local cell optical depth)
    frac_internal = 1 / (1 + tau)                (1 as tau->0, 0 as tau->inf)

evaluated per internal face from the (average) opacity of the two
corner sub-cells sharing it and the width of the *coarse* cell they
belong to.  External (coarse-cell-boundary) faces always keep the full
Rusanov dissipation (fraction 1.0), independent of opacity.

For P_N the streaming operator is linear (fixed matrices A_x, A_z), so
the corner-balance system reduces to a fixed 4S×4S block matrix.  M1's
flux is nonlinear (through the closure), so instead of assembling and
solving that block system directly, this implementation reuses
``M1Solver2D``'s existing cell-implicit symmetric Gauss-Seidel (SGS)
machinery: the *corner* sub-cells are treated as ordinary FV cells on a
doubled-resolution mesh, and each of the 4 faces surrounding a corner
independently carries a (possibly opacity-dependent) Rusanov-dissipation
weight via the ``_get_ext_weights`` hook (see ``m1_2d.M1Solver2D``).  The
nonlinear pressure-tensor terms are still evaluated (lagged, once per
sweep pass) exactly as in the base class.

Usage
-----
``M1SolverSCB2D`` has the *same* constructor signature as
``M1Solver2D`` — ``n1``, ``n2`` are the number of **original (coarse)**
cells.  Internally the state arrays (``Er``, ``F1``, ``F2``, ``T``, and
any ``sigma_func``/``EOS``/``invEOS`` outputs) live on the doubled
*corner* grid of shape ``(2*n1, 2*n2)``.  Material-property callables
should therefore be evaluated at the corner cell-centres
(``solver.x1_c``, ``solver.x2_c``, each of length ``2*n1``/``2*n2``) —
this is the direct analogue of how the P_N SCB crooked-pipe scripts
build per-corner opacity/heat-capacity arrays.
"""

import numpy as np

from .m1_2d import M1Solver2D
from .m1_1d import (
    closure_p1,
    closure_kershaw,
    closure_levermore,
    closure_minerbo_poly,
    closure_minerbo_rational,
    C_LIGHT,
    A_RAD,
)

__all__ = ["M1SolverSCB2D"]


def _corner_faces(orig_faces):
    """Insert a midpoint between each pair of coarse faces.

    Parameters
    ----------
    orig_faces : ndarray, shape (n+1,)
        Original (coarse) cell-face positions.

    Returns
    -------
    corner_faces : ndarray, shape (2n+1,)
        Doubled-resolution face positions: even indices coincide with
        the original coarse faces, odd indices are the inserted
        midpoints.
    is_external : ndarray of bool, shape (2n+1,)
        True at coarse-cell boundaries (even indices, including the two
        domain boundaries); False at the inserted sub-cell midpoints.
    """
    orig_faces = np.asarray(orig_faces, dtype=float)
    n = len(orig_faces) - 1
    corner_faces = np.empty(2 * n + 1)
    is_external = np.zeros(2 * n + 1, dtype=bool)
    corner_faces[0::2] = orig_faces
    corner_faces[1::2] = 0.5 * (orig_faces[:-1] + orig_faces[1:])
    is_external[0::2] = True
    return corner_faces, is_external


class M1SolverSCB2D(M1Solver2D):
    """2-D M1 solver with Simple-Corner-Balance (SCB) spatial discretisation.

    Identical constructor and public interface to ``M1Solver2D`` (see that
    class for the full parameter list) — ``n1``/``n2`` (or ``x1_faces``/
    ``x2_faces``) describe the *original, coarse* mesh.  Internally this
    class builds a doubled-resolution *corner* mesh (``2*n1`` × ``2*n2``
    cells).  Faces coincident with the original coarse-cell boundaries
    are always "external" (full Rusanov dissipation); the inserted
    sub-cell midpoint faces are "internal", with a dissipation fraction
    that depends on the *local optical thickness* of the coarse cell
    (see module docstring): ``1/(1+sigma*dx_coarse)`` — near 1 (fully
    upwinded) in optically thin regions, near 0 (purely centred, full
    SCB sub-cell accuracy) in optically thick regions.  All state arrays
    (``Er``, ``F1``, ``F2``, ``T``) and material callables
    (``sigma_func``, ``scat_func``, ``EOS``, ``invEOS``) operate on this
    doubled corner grid.

    Parameters
    ----------
    internal_dissipation_floor : float
        Minimum internal-face dissipation fraction (default 0.0),
        applied as a floor to ``1/(1+tau)`` for extra robustness margin
        if needed.

    Extra attributes (beyond ``M1Solver2D``)
    -----------------------------------------
    n1_cells, n2_cells : int
        Number of *original* (coarse) cells in each direction.
    x1_faces_coarse, x2_faces_coarse : ndarray
        The original (coarse) face arrays passed in (or generated from
        x1_min/x1_max/n1 etc.) before doubling.
    dx1_coarse, dx2_coarse : ndarray, shape (n1_cells,)/(n2_cells,)
        Widths of the original coarse cells.
    """

    def __init__(
        self,
        x1_min, x1_max, n1,
        x2_min, x2_max, n2,
        x1_faces=None,
        x2_faces=None,
        internal_dissipation_floor=0.0,
        **kwargs,
    ):
        # ── Build the ORIGINAL (coarse) face arrays ──────────────────────
        if x1_faces is not None:
            orig_x1 = np.asarray(x1_faces, dtype=float)
            if len(orig_x1) != n1 + 1:
                raise ValueError(
                    f"x1_faces length {len(orig_x1)} != n1+1={n1 + 1}")
        else:
            orig_x1 = np.linspace(x1_min, x1_max, n1 + 1)

        if x2_faces is not None:
            orig_x2 = np.asarray(x2_faces, dtype=float)
            if len(orig_x2) != n2 + 1:
                raise ValueError(
                    f"x2_faces length {len(orig_x2)} != n2+1={n2 + 1}")
        else:
            orig_x2 = np.linspace(x2_min, x2_max, n2 + 1)

        corner_x1, ext_x1 = _corner_faces(orig_x1)
        corner_x2, ext_x2 = _corner_faces(orig_x2)

        # ── Build the parent M1Solver2D directly on the doubled corner
        #    grid — all physics (closure, Fleck-Cummings, BCs, SGS sweep)
        #    is inherited unchanged. ────────────────────────────────────
        super().__init__(
            x1_min=orig_x1[0], x1_max=orig_x1[-1], n1=2 * n1,
            x2_min=orig_x2[0], x2_max=orig_x2[-1], n2=2 * n2,
            x1_faces=corner_x1, x2_faces=corner_x2,
            **kwargs,
        )

        self.n1_cells = int(n1)
        self.n2_cells = int(n2)
        self.x1_faces_coarse = orig_x1
        self.x2_faces_coarse = orig_x2
        self.dx1_coarse = np.diff(orig_x1)
        self.dx2_coarse = np.diff(orig_x2)
        self.internal_dissipation_floor = float(internal_dissipation_floor)

        # Structural external/internal marker (bool): True at coarse-cell
        # boundaries.  _get_ext_weights (below) uses this to know *where*
        # to apply the opacity-dependent formula.
        self.is_ext_x1 = ext_x1
        self.is_ext_x2 = ext_x2

    def _get_ext_weights(self, sigma_tot):
        """Opacity-dependent face-dissipation weights (overrides base).

        External (coarse-cell-boundary) faces always keep weight 1.0.
        Internal (sub-cell midpoint) faces get ``1/(1+tau)`` where
        ``tau = sigma_face * dx_coarse`` is the local coarse-cell optical
        thickness (``sigma_face`` = average opacity of the two corner
        sub-cells sharing that internal face) — near 1 (upwind) when
        optically thin, near 0 (centred) when optically thick.
        """
        n1, n2 = self.n1, self.n2
        ext_x1_full = np.ones((n1 + 1, n2))
        ext_x2_full = np.ones((n1, n2 + 1))
        floor = self.internal_dissipation_floor

        # x1-direction internal faces: index 2k+1 for coarse cell k,
        # between corner columns 2k and 2k+1.
        for k in range(self.n1_cells):
            i_face = 2 * k + 1
            sigma_face = 0.5 * (sigma_tot[i_face - 1, :] + sigma_tot[i_face, :])
            tau = sigma_face * self.dx1_coarse[k]
            ext_x1_full[i_face, :] = np.maximum(1.0 / (1.0 + tau), floor)

        # x2-direction internal faces: index 2k+1 for coarse cell k,
        # between corner rows 2k and 2k+1.
        for k in range(self.n2_cells):
            j_face = 2 * k + 1
            sigma_face = 0.5 * (sigma_tot[:, j_face - 1] + sigma_tot[:, j_face])
            tau = sigma_face * self.dx2_coarse[k]
            ext_x2_full[:, j_face] = np.maximum(1.0 / (1.0 + tau), floor)

        return ext_x1_full, ext_x2_full

