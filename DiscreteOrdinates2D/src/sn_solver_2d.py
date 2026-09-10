"""
2-D Discrete Ordinates (S_N) solver using Simple Corner Balance on
rectangular meshes.

Spatial discretisation follows Chapter 11.2 of the textbook:
  - Each cell is divided into 4 corners (subcells).
  - Step closure on external (inter-cell) faces.
  - Internal (intra-cell) face coupling gives a 4×4 system per cell.
  - Sweep order determined by sign of direction cosines.

Physics:
  - Gray thermal radiative transfer with Fleck-Cummings linearization.
  - DMD-accelerated source iteration.
  - Nonlinear T_star outer loop (optional).
  - Reflecting and vacuum boundary conditions.
  - Variable zone widths in x and y.

Array conventions:
  - phi, T, e, sigma: (Ix, Iy, 4) — corner-averaged values
  - Corner numbering: 0=NE, 1=NW, 2=SW, 3=SE (counterclockwise from NE)

Physical constants are in CGS units with time in nanoseconds.
"""

import numpy as np
from numba import jit, prange
import math
import sys
import os

# Add parent for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'DiscreteOrdinates'))

from DiscreteOrdinates2D.src.quadratures import get_2d_quadrature
from DiscreteOrdinates.src.sn_solver import solver_with_dmd_inc as _solver_with_dmd_inc_1d

# Physical constants (CGS, time in ns)
c = 29.98       # speed of light (cm/ns)
a = 0.01372     # radiation constant (GJ/(cm^3 keV^4))
ac = a * c


# ===========================================================================
# Corner-Balance Sweep Kernel (allocation-free, scalar Gaussian elimination)
# ===========================================================================

@jit(nopython=True, cache=True)
def _solve_cell_corners(Ox, Oy, dx, dy, sig0, sig1, sig2, sig3,
                        src0, src1, src2, src3,
                        inc_x0, inc_x1, inc_y0, inc_y1):
    """Solve the 4×4 corner-balance system for one cell using only scalars.

    Uses explicit Gaussian elimination on the known sparse structure
    (3 non-zeros per row). No array allocations — purely scalar arithmetic.

    The system is strictly diagonally dominant (sigma_hat > 0), so no
    pivoting is needed.

    Parameters (all scalars):
        Ox, Oy    – direction cosines
        dx, dy    – cell widths
        sig0..3   – sigma_hat per corner (NE, NW, SW, SE)
        src0..3   – total source per corner
        inc_x0/1  – incoming on x-face [bottom, top]
        inc_y0/1  – incoming on y-face [left, right]

    Returns (I0, I1, I2, I3) – corner-averaged intensities.

    Corner numbering: 0=NE, 1=NW, 2=SW, 3=SE
    """
    # Geometry constants
    Ax = dy * 0.5       # half-cell height (vertical face length)
    Ay = dx * 0.5       # half-cell width (horizontal face length)
    V = dx * dy * 0.25  # corner volume

    abs_Ox = abs(Ox)
    abs_Oy = abs(Oy)
    half_Ax = Ax * 0.5   # Ax/2
    half_Ay = Ay * 0.5   # Ay/2

    # Diagonal entries: sigma_hat*V + (Ax/2)|Ox| + (Ay/2)|Oy|
    diag_geom = half_Ax * abs_Ox + half_Ay * abs_Oy
    d0 = sig0 * V + diag_geom
    d1 = sig1 * V + diag_geom
    d2 = sig2 * V + diag_geom
    d3 = sig3 * V + diag_geom

    # Off-diagonal entries from internal face coupling:
    #   A[c, cx[c]] = -(Ax/2) * sx[c] * Ox
    #   A[c, cy[c]] = -(Ay/2) * sy[c] * Oy
    # with sx=[+1,-1,-1,+1], sy=[+1,+1,-1,-1], cx=[1,0,3,2], cy=[3,2,1,0]
    half_Ax_Ox = half_Ax * Ox
    half_Ay_Oy = half_Ay * Oy

    # Matrix (only non-zero off-diagonals shown):
    # Row 0: A[0,1]=-half_Ax_Ox, A[0,3]=-half_Ay_Oy
    # Row 1: A[1,0]=+half_Ax_Ox, A[1,2]=-half_Ay_Oy
    # Row 2: A[2,1]=+half_Ay_Oy, A[2,3]=+half_Ax_Ox
    # Row 3: A[3,0]=+half_Ay_Oy, A[3,2]=-half_Ax_Ox
    a01 = -half_Ax_Ox
    a03 = -half_Ay_Oy
    a10 =  half_Ax_Ox
    a12 = -half_Ay_Oy
    a21 =  half_Ay_Oy
    a23 =  half_Ax_Ox
    a30 =  half_Ay_Oy
    a32 = -half_Ax_Ox

    # RHS: V*source + external boundary incoming contributions
    b0 = V * src0
    b1 = V * src1
    b2 = V * src2
    b3 = V * src3

    # External x-face: Ax * max(-sx[c]*Ox, 0) * incoming
    # c=0 (sx=+1): incoming when Ox<0, uses inc_x1 (top of right face)
    # c=1 (sx=-1): incoming when Ox>0, uses inc_x1 (top of left face)
    # c=2 (sx=-1): incoming when Ox>0, uses inc_x0 (bottom of left face)
    # c=3 (sx=+1): incoming when Ox<0, uses inc_x0 (bottom of right face)
    if Ox > 0.0:
        Ax_Ox = Ax * Ox
        b1 += Ax_Ox * inc_x1
        b2 += Ax_Ox * inc_x0
    elif Ox < 0.0:
        Ax_nOx = Ax * (-Ox)
        b0 += Ax_nOx * inc_x1
        b3 += Ax_nOx * inc_x0

    # External y-face: Ay * max(-sy[c]*Oy, 0) * incoming
    # c=0 (sy=+1): incoming when Oy<0, uses inc_y1 (right of top face)
    # c=1 (sy=+1): incoming when Oy<0, uses inc_y0 (left of top face)
    # c=2 (sy=-1): incoming when Oy>0, uses inc_y0 (left of bottom face)
    # c=3 (sy=-1): incoming when Oy>0, uses inc_y1 (right of bottom face)
    if Oy > 0.0:
        Ay_Oy = Ay * Oy
        b2 += Ay_Oy * inc_y0
        b3 += Ay_Oy * inc_y1
    elif Oy < 0.0:
        Ay_nOy = Ay * (-Oy)
        b0 += Ay_nOy * inc_y1
        b1 += Ay_nOy * inc_y0

    # --- Gaussian elimination (no pivoting; system is diag. dominant) ---
    # Sparse structure: A[0,2]=0, A[1,3]=0, A[2,0]=0, A[3,1]=0.
    #
    # Step 1: eliminate column 0 from rows 1 and 3 (row 2 has A[2,0]=0)
    inv_d0 = 1.0 / d0
    f1 = a10 * inv_d0
    f3 = a30 * inv_d0

    d1p  = d1  - f1 * a01    # updated diagonal row 1
    # a12 unchanged (A[0,2]=0 so no fill)
    g13  = -f1 * a03          # fill-in at [1,3]
    b1p  = b1  - f1 * b0

    g31  = -f3 * a01          # fill-in at [3,1]
    # a32 unchanged
    d3p  = d3  - f3 * a03    # updated diagonal row 3
    b3p  = b3  - f3 * b0

    # Step 2: eliminate column 1 from rows 2 and 3
    inv_d1p = 1.0 / d1p
    f2  = a21 * inv_d1p
    f3b = g31 * inv_d1p

    d2p  = d2  - f2  * a12    # updated diagonal row 2
    g23  = a23 - f2  * g13    # updated [2,3]
    b2p  = b2  - f2  * b1p

    g32  = a32 - f3b * a12    # updated [3,2]
    d3pp = d3p - f3b * g13    # updated diagonal row 3
    b3pp = b3p - f3b * b1p

    # Step 3: eliminate column 2 from row 3
    inv_d2p = 1.0 / d2p
    f3c = g32 * inv_d2p

    d3ppp = d3pp - f3c * g23
    b3ppp = b3pp - f3c * b2p

    # Back substitution
    I3 = b3ppp / d3ppp
    I2 = (b2p - g23 * I3) * inv_d2p
    I1 = (b1p - a12 * I2 - g13 * I3) * inv_d1p
    I0 = (b0  - a01 * I1 - a03 * I3) * inv_d0

    return I0, I1, I2, I3


# ===========================================================================
# Full Mesh Sweep
# ===========================================================================

@jit(nopython=True, parallel=True, cache=True)
def sweep_2d(Ix, Iy, dx_arr, dy_arr, sigma_hat, source, Omega_x, Omega_y,
             weights, BCs_xlo, BCs_xhi, BCs_ylo, BCs_yhi,
             compute_phi=True, compute_boundaries=True,
             phi_angle_workspace=None):
    """Perform a full 2-D transport sweep for all angles.

    Parameters
    ----------
    Ix, Iy : int
        Number of cells in x and y.
    dx_arr : (Ix,) float64
        Cell widths in x.
    dy_arr : (Iy,) float64
        Cell widths in y.
    sigma_hat : (Ix, Iy, 4) float64
        Effective total opacity per corner.
    source : (Ix, Iy, M, 4) float64
        Total source per corner per angle.
    Omega_x, Omega_y : (M,) float64
        Direction cosines.
    weights : (M,) float64
        Angular weights.
    BCs_xlo : (Iy, M, 2) float64
        Incoming intensity on x=x_min face. Shape: [j, angle, (bottom, top) corner].
    BCs_xhi : (Iy, M, 2) float64
        Incoming intensity on x=x_max face.
    BCs_ylo : (Ix, M, 2) float64
        Incoming intensity on y=y_min face. Shape: [i, angle, (left, right) corner].
    BCs_yhi : (Ix, M, 2) float64
        Incoming intensity on y=y_max face.

    Returns
    -------
    phi : (Ix, Iy, 4) float64
        Scalar flux (angular integral of intensity) per corner.
    psi_xlo_out : (Iy, M, 2) float64
        Outgoing angular flux on x=x_min face.
    psi_xhi_out : (Iy, M, 2) float64
        Outgoing angular flux on x=x_max face.
    psi_ylo_out : (Ix, M, 2) float64
        Outgoing angular flux on y=y_min face.
    psi_yhi_out : (Ix, M, 2) float64
        Outgoing angular flux on y=y_max face.
    """
    M = Omega_x.size
    phi = np.zeros((Ix, Iy, 4))
    if compute_phi:
        # Each prange iteration owns one complete angle plane.  Reducing these
        # planes after the parallel sweep avoids races on phi while retaining
        # the serial n=0,...,M-1 accumulation order.
        if phi_angle_workspace is None:
            phi_angle = np.empty((M, Ix, Iy, 4))
        else:
            phi_angle = phi_angle_workspace
    else:
        phi_angle = np.empty((1, 1, 1, 1))

    if compute_boundaries:
        psi_xlo_out = np.zeros((Iy, M, 2))
        psi_xhi_out = np.zeros((Iy, M, 2))
        psi_ylo_out = np.zeros((Ix, M, 2))
        psi_yhi_out = np.zeros((Ix, M, 2))
    else:
        # Preserve a uniform return type without allocating unused boundary
        # fields in phi-only sweeps.
        psi_xlo_out = np.empty((1, 1, 1))
        psi_xhi_out = np.empty((1, 1, 1))
        psi_ylo_out = np.empty((1, 1, 1))
        psi_yhi_out = np.empty((1, 1, 1))

    for n in prange(M):
        Ox = Omega_x[n]
        Oy = Omega_y[n]
        w_n = weights[n]

        # Determine sweep order based on direction
        if Ox > 0:
            x_range_start = 0
            x_range_end = Ix
            x_step = 1
        else:
            x_range_start = Ix - 1
            x_range_end = -1
            x_step = -1

        if Oy > 0:
            y_range_start = 0
            y_range_end = Iy
            y_step = 1
        else:
            y_range_start = Iy - 1
            y_range_end = -1
            y_step = -1

        # Boundary fluxes for this angle stored as face-averaged values
        # psi_x[j] = incoming from x-face for cell column j
        # psi_y[i] = incoming from y-face for cell row i
        # We need face values for each cell edge
        # x-faces: (Ix+1) faces, y-faces: (Iy+1) faces
        # But we only need the "current" face values during the sweep

        # For x-direction sweep: track outgoing x-face intensity per y-row
        # Shape: (Iy, 2) — bottom and top corner values on the face
        psi_x_face = np.zeros((Iy, 2))
        # For y-direction: track outgoing y-face intensity per x-column
        psi_y_face = np.zeros((Ix, 2))

        # Initialize boundary face values
        if Ox > 0:
            # Sweeping left to right: incoming from x_min boundary
            for j in range(Iy):
                psi_x_face[j, 0] = BCs_xlo[j, n, 0]
                psi_x_face[j, 1] = BCs_xlo[j, n, 1]
        else:
            # Sweeping right to left: incoming from x_max boundary
            for j in range(Iy):
                psi_x_face[j, 0] = BCs_xhi[j, n, 0]
                psi_x_face[j, 1] = BCs_xhi[j, n, 1]

        if Oy > 0:
            # Sweeping bottom to top: incoming from y_min boundary
            for i in range(Ix):
                psi_y_face[i, 0] = BCs_ylo[i, n, 0]
                psi_y_face[i, 1] = BCs_ylo[i, n, 1]
        else:
            # Sweeping top to bottom: incoming from y_max boundary
            for i in range(Ix):
                psi_y_face[i, 0] = BCs_yhi[i, n, 0]
                psi_y_face[i, 1] = BCs_yhi[i, n, 1]

        # Sweep cells in order
        i = x_range_start
        while (x_step > 0 and i < x_range_end) or (x_step < 0 and i > x_range_end):
            j = y_range_start
            while (y_step > 0 and j < y_range_end) or (y_step < 0 and j > y_range_end):
                # Solve corner-balance with all-scalar arguments (no allocations)
                I0, I1, I2, I3 = _solve_cell_corners(
                    Ox, Oy, dx_arr[i], dy_arr[j],
                    sigma_hat[i, j, 0], sigma_hat[i, j, 1],
                    sigma_hat[i, j, 2], sigma_hat[i, j, 3],
                    source[i, j, n, 0], source[i, j, n, 1],
                    source[i, j, n, 2], source[i, j, n, 3],
                    psi_x_face[j, 0], psi_x_face[j, 1],
                    psi_y_face[i, 0], psi_y_face[i, 1])

                # Positivity fix-up
                if I0 < 0.0:
                    I0 = 0.0
                if I1 < 0.0:
                    I1 = 0.0
                if I2 < 0.0:
                    I2 = 0.0
                if I3 < 0.0:
                    I3 = 0.0

                if compute_phi:
                    phi_angle[n, i, j, 0] = w_n * I0
                    phi_angle[n, i, j, 1] = w_n * I1
                    phi_angle[n, i, j, 2] = w_n * I2
                    phi_angle[n, i, j, 3] = w_n * I3

                # Set outgoing face values for downstream cells
                if Ox > 0:
                    psi_x_face[j, 0] = I3  # SE = bottom
                    psi_x_face[j, 1] = I0  # NE = top
                else:
                    psi_x_face[j, 0] = I2  # SW = bottom
                    psi_x_face[j, 1] = I1  # NW = top

                if Oy > 0:
                    psi_y_face[i, 0] = I1  # NW = left
                    psi_y_face[i, 1] = I0  # NE = right
                else:
                    psi_y_face[i, 0] = I2  # SW = left
                    psi_y_face[i, 1] = I3  # SE = right

                j += y_step

            # Record outgoing x-boundary if this is the last i
            if compute_boundaries and Ox > 0 and i == Ix - 1:
                for jj in range(Iy):
                    psi_xhi_out[jj, n, 0] = psi_x_face[jj, 0]
                    psi_xhi_out[jj, n, 1] = psi_x_face[jj, 1]
            elif compute_boundaries and Ox < 0 and i == 0:
                for jj in range(Iy):
                    psi_xlo_out[jj, n, 0] = psi_x_face[jj, 0]
                    psi_xlo_out[jj, n, 1] = psi_x_face[jj, 1]

            i += x_step

        # Record outgoing y-boundaries
        if compute_boundaries and Oy > 0:
            for ii in range(Ix):
                psi_yhi_out[ii, n, 0] = psi_y_face[ii, 0]
                psi_yhi_out[ii, n, 1] = psi_y_face[ii, 1]
        elif compute_boundaries:
            for ii in range(Ix):
                psi_ylo_out[ii, n, 0] = psi_y_face[ii, 0]
                psi_ylo_out[ii, n, 1] = psi_y_face[ii, 1]

    if compute_phi:
        for ii in prange(Ix):
            for jj in range(Iy):
                for cc in range(4):
                    total = 0.0
                    for n in range(M):
                        total += phi_angle[n, ii, jj, cc]
                    phi[ii, jj, cc] = total

    return phi, psi_xlo_out, psi_xhi_out, psi_ylo_out, psi_yhi_out


@jit(nopython=True, parallel=True, cache=True)
def sweep_2d_psi(Ix, Iy, dx_arr, dy_arr, sigma_hat, source, Omega_x, Omega_y,
                 weights, BCs_xlo, BCs_xhi, BCs_ylo, BCs_yhi,
                 compute_phi=True, compute_boundaries=True):
    """Sweep returning both scalar flux and per-angle intensity.

    Same as sweep_2d but additionally returns psi (Ix, Iy, M, 4).
    """
    M = Omega_x.size
    phi = np.zeros((Ix, Iy, 4))
    psi = np.zeros((Ix, Iy, M, 4))
    if compute_boundaries:
        psi_xlo_out = np.zeros((Iy, M, 2))
        psi_xhi_out = np.zeros((Iy, M, 2))
        psi_ylo_out = np.zeros((Ix, M, 2))
        psi_yhi_out = np.zeros((Ix, M, 2))
    else:
        psi_xlo_out = np.empty((1, 1, 1))
        psi_xhi_out = np.empty((1, 1, 1))
        psi_ylo_out = np.empty((1, 1, 1))
        psi_yhi_out = np.empty((1, 1, 1))

    for n in prange(M):
        Ox = Omega_x[n]
        Oy = Omega_y[n]
        w_n = weights[n]

        if Ox > 0:
            x_range_start = 0
            x_range_end = Ix
            x_step = 1
        else:
            x_range_start = Ix - 1
            x_range_end = -1
            x_step = -1

        if Oy > 0:
            y_range_start = 0
            y_range_end = Iy
            y_step = 1
        else:
            y_range_start = Iy - 1
            y_range_end = -1
            y_step = -1

        psi_x_face = np.zeros((Iy, 2))
        psi_y_face = np.zeros((Ix, 2))

        if Ox > 0:
            for j in range(Iy):
                psi_x_face[j, 0] = BCs_xlo[j, n, 0]
                psi_x_face[j, 1] = BCs_xlo[j, n, 1]
        else:
            for j in range(Iy):
                psi_x_face[j, 0] = BCs_xhi[j, n, 0]
                psi_x_face[j, 1] = BCs_xhi[j, n, 1]

        if Oy > 0:
            for i in range(Ix):
                psi_y_face[i, 0] = BCs_ylo[i, n, 0]
                psi_y_face[i, 1] = BCs_ylo[i, n, 1]
        else:
            for i in range(Ix):
                psi_y_face[i, 0] = BCs_yhi[i, n, 0]
                psi_y_face[i, 1] = BCs_yhi[i, n, 1]

        i = x_range_start
        while (x_step > 0 and i < x_range_end) or (x_step < 0 and i > x_range_end):
            j = y_range_start
            while (y_step > 0 and j < y_range_end) or (y_step < 0 and j > y_range_end):
                I0, I1, I2, I3 = _solve_cell_corners(
                    Ox, Oy, dx_arr[i], dy_arr[j],
                    sigma_hat[i, j, 0], sigma_hat[i, j, 1],
                    sigma_hat[i, j, 2], sigma_hat[i, j, 3],
                    source[i, j, n, 0], source[i, j, n, 1],
                    source[i, j, n, 2], source[i, j, n, 3],
                    psi_x_face[j, 0], psi_x_face[j, 1],
                    psi_y_face[i, 0], psi_y_face[i, 1])

                if I0 < 0.0:
                    I0 = 0.0
                if I1 < 0.0:
                    I1 = 0.0
                if I2 < 0.0:
                    I2 = 0.0
                if I3 < 0.0:
                    I3 = 0.0

                psi[i, j, n, 0] = I0
                psi[i, j, n, 1] = I1
                psi[i, j, n, 2] = I2
                psi[i, j, n, 3] = I3

                if Ox > 0:
                    psi_x_face[j, 0] = I3
                    psi_x_face[j, 1] = I0
                else:
                    psi_x_face[j, 0] = I2
                    psi_x_face[j, 1] = I1

                if Oy > 0:
                    psi_y_face[i, 0] = I1
                    psi_y_face[i, 1] = I0
                else:
                    psi_y_face[i, 0] = I2
                    psi_y_face[i, 1] = I3

                j += y_step

            if compute_boundaries and Ox > 0 and i == Ix - 1:
                for jj in range(Iy):
                    psi_xhi_out[jj, n, 0] = psi_x_face[jj, 0]
                    psi_xhi_out[jj, n, 1] = psi_x_face[jj, 1]
            elif compute_boundaries and Ox < 0 and i == 0:
                for jj in range(Iy):
                    psi_xlo_out[jj, n, 0] = psi_x_face[jj, 0]
                    psi_xlo_out[jj, n, 1] = psi_x_face[jj, 1]

            i += x_step

        if compute_boundaries and Oy > 0:
            for ii in range(Ix):
                psi_yhi_out[ii, n, 0] = psi_y_face[ii, 0]
                psi_yhi_out[ii, n, 1] = psi_y_face[ii, 1]
        elif compute_boundaries:
            for ii in range(Ix):
                psi_ylo_out[ii, n, 0] = psi_y_face[ii, 0]
                psi_ylo_out[ii, n, 1] = psi_y_face[ii, 1]

    if compute_phi:
        for ii in prange(Ix):
            for jj in range(Iy):
                for cc in range(4):
                    total = 0.0
                    for n in range(M):
                        total += weights[n] * psi[ii, jj, n, cc]
                    phi[ii, jj, cc] = total

    return psi, phi, psi_xlo_out, psi_xhi_out, psi_ylo_out, psi_yhi_out


@jit(nopython=True, parallel=True, cache=True)
def sweep_2d_iso(Ix, Iy, dx_arr, dy_arr, sigma_hat, source_iso, Omega_x, Omega_y,
                 weights, BCs_xlo, BCs_xhi, BCs_ylo, BCs_yhi,
                 compute_phi=True, compute_boundaries=True,
                 phi_angle_workspace=None):
    """Sweep with angle-independent (isotropic) source — avoids (Ix,Iy,M,4) allocation.

    Parameters
    ----------
    source_iso : (Ix, Iy, 4) float64
        Isotropic source (same for all angles).

    Returns same as sweep_2d.
    """
    M = Omega_x.size
    phi = np.zeros((Ix, Iy, 4))
    if compute_phi:
        if phi_angle_workspace is None:
            phi_angle = np.empty((M, Ix, Iy, 4))
        else:
            phi_angle = phi_angle_workspace
    else:
        phi_angle = np.empty((1, 1, 1, 1))

    if compute_boundaries:
        psi_xlo_out = np.zeros((Iy, M, 2))
        psi_xhi_out = np.zeros((Iy, M, 2))
        psi_ylo_out = np.zeros((Ix, M, 2))
        psi_yhi_out = np.zeros((Ix, M, 2))
    else:
        psi_xlo_out = np.empty((1, 1, 1))
        psi_xhi_out = np.empty((1, 1, 1))
        psi_ylo_out = np.empty((1, 1, 1))
        psi_yhi_out = np.empty((1, 1, 1))

    for n in prange(M):
        Ox = Omega_x[n]
        Oy = Omega_y[n]
        w_n = weights[n]

        if Ox > 0:
            x_range_start = 0
            x_range_end = Ix
            x_step = 1
        else:
            x_range_start = Ix - 1
            x_range_end = -1
            x_step = -1

        if Oy > 0:
            y_range_start = 0
            y_range_end = Iy
            y_step = 1
        else:
            y_range_start = Iy - 1
            y_range_end = -1
            y_step = -1

        psi_x_face = np.zeros((Iy, 2))
        psi_y_face = np.zeros((Ix, 2))

        if Ox > 0:
            for j in range(Iy):
                psi_x_face[j, 0] = BCs_xlo[j, n, 0]
                psi_x_face[j, 1] = BCs_xlo[j, n, 1]
        else:
            for j in range(Iy):
                psi_x_face[j, 0] = BCs_xhi[j, n, 0]
                psi_x_face[j, 1] = BCs_xhi[j, n, 1]

        if Oy > 0:
            for i in range(Ix):
                psi_y_face[i, 0] = BCs_ylo[i, n, 0]
                psi_y_face[i, 1] = BCs_ylo[i, n, 1]
        else:
            for i in range(Ix):
                psi_y_face[i, 0] = BCs_yhi[i, n, 0]
                psi_y_face[i, 1] = BCs_yhi[i, n, 1]

        i = x_range_start
        while (x_step > 0 and i < x_range_end) or (x_step < 0 and i > x_range_end):
            j = y_range_start
            while (y_step > 0 and j < y_range_end) or (y_step < 0 and j > y_range_end):
                I0, I1, I2, I3 = _solve_cell_corners(
                    Ox, Oy, dx_arr[i], dy_arr[j],
                    sigma_hat[i, j, 0], sigma_hat[i, j, 1],
                    sigma_hat[i, j, 2], sigma_hat[i, j, 3],
                    source_iso[i, j, 0], source_iso[i, j, 1],
                    source_iso[i, j, 2], source_iso[i, j, 3],
                    psi_x_face[j, 0], psi_x_face[j, 1],
                    psi_y_face[i, 0], psi_y_face[i, 1])

                if I0 < 0.0:
                    I0 = 0.0
                if I1 < 0.0:
                    I1 = 0.0
                if I2 < 0.0:
                    I2 = 0.0
                if I3 < 0.0:
                    I3 = 0.0

                if compute_phi:
                    phi_angle[n, i, j, 0] = w_n * I0
                    phi_angle[n, i, j, 1] = w_n * I1
                    phi_angle[n, i, j, 2] = w_n * I2
                    phi_angle[n, i, j, 3] = w_n * I3

                if Ox > 0:
                    psi_x_face[j, 0] = I3
                    psi_x_face[j, 1] = I0
                else:
                    psi_x_face[j, 0] = I2
                    psi_x_face[j, 1] = I1

                if Oy > 0:
                    psi_y_face[i, 0] = I1
                    psi_y_face[i, 1] = I0
                else:
                    psi_y_face[i, 0] = I2
                    psi_y_face[i, 1] = I3

                j += y_step

            if compute_boundaries and Ox > 0 and i == Ix - 1:
                for jj in range(Iy):
                    psi_xhi_out[jj, n, 0] = psi_x_face[jj, 0]
                    psi_xhi_out[jj, n, 1] = psi_x_face[jj, 1]
            elif compute_boundaries and Ox < 0 and i == 0:
                for jj in range(Iy):
                    psi_xlo_out[jj, n, 0] = psi_x_face[jj, 0]
                    psi_xlo_out[jj, n, 1] = psi_x_face[jj, 1]

            i += x_step

        if compute_boundaries and Oy > 0:
            for ii in range(Ix):
                psi_yhi_out[ii, n, 0] = psi_y_face[ii, 0]
                psi_yhi_out[ii, n, 1] = psi_y_face[ii, 1]
        elif compute_boundaries:
            for ii in range(Ix):
                psi_ylo_out[ii, n, 0] = psi_y_face[ii, 0]
                psi_ylo_out[ii, n, 1] = psi_y_face[ii, 1]

    if compute_phi:
        for ii in prange(Ix):
            for jj in range(Iy):
                for cc in range(4):
                    total = 0.0
                    for n in range(M):
                        total += phi_angle[n, ii, jj, cc]
                    phi[ii, jj, cc] = total

    return phi, psi_xlo_out, psi_xhi_out, psi_ylo_out, psi_yhi_out


@jit(nopython=True, parallel=True, cache=True)
def sweep_2d_transient(Ix, Iy, dx_arr, dy_arr, sigma_hat,
                       source_iso, psi_time, icdt,
                       Omega_x, Omega_y, weights,
                       BCs_xlo, BCs_xhi, BCs_ylo, BCs_yhi,
                       compute_phi=True, store_psi=False,
                       compute_boundaries=True,
                       phi_angle_workspace=None):
    """Sweep a source of the form ``source_iso + icdt * psi_time``.

    Keeping the isotropic and time-derivative terms separate avoids building
    an ``(Ix, Iy, M, 4)`` source temporary for every nonlinear iteration.
    Output flags let internal callers skip fields that they do not consume.
    """
    M = Omega_x.size
    phi = np.zeros((Ix, Iy, 4))

    if compute_phi:
        if phi_angle_workspace is None:
            phi_angle = np.empty((M, Ix, Iy, 4))
        else:
            phi_angle = phi_angle_workspace
    else:
        phi_angle = np.empty((1, 1, 1, 1))

    if store_psi:
        psi = np.empty((Ix, Iy, M, 4))
    else:
        psi = np.empty((1, 1, 1, 1))

    if compute_boundaries:
        psi_xlo_out = np.zeros((Iy, M, 2))
        psi_xhi_out = np.zeros((Iy, M, 2))
        psi_ylo_out = np.zeros((Ix, M, 2))
        psi_yhi_out = np.zeros((Ix, M, 2))
    else:
        psi_xlo_out = np.empty((1, 1, 1))
        psi_xhi_out = np.empty((1, 1, 1))
        psi_ylo_out = np.empty((1, 1, 1))
        psi_yhi_out = np.empty((1, 1, 1))

    for n in prange(M):
        Ox = Omega_x[n]
        Oy = Omega_y[n]
        w_n = weights[n]

        if Ox > 0:
            x_range_start = 0
            x_range_end = Ix
            x_step = 1
        else:
            x_range_start = Ix - 1
            x_range_end = -1
            x_step = -1

        if Oy > 0:
            y_range_start = 0
            y_range_end = Iy
            y_step = 1
        else:
            y_range_start = Iy - 1
            y_range_end = -1
            y_step = -1

        psi_x_face = np.zeros((Iy, 2))
        psi_y_face = np.zeros((Ix, 2))

        if Ox > 0:
            for j in range(Iy):
                psi_x_face[j, 0] = BCs_xlo[j, n, 0]
                psi_x_face[j, 1] = BCs_xlo[j, n, 1]
        else:
            for j in range(Iy):
                psi_x_face[j, 0] = BCs_xhi[j, n, 0]
                psi_x_face[j, 1] = BCs_xhi[j, n, 1]

        if Oy > 0:
            for i in range(Ix):
                psi_y_face[i, 0] = BCs_ylo[i, n, 0]
                psi_y_face[i, 1] = BCs_ylo[i, n, 1]
        else:
            for i in range(Ix):
                psi_y_face[i, 0] = BCs_yhi[i, n, 0]
                psi_y_face[i, 1] = BCs_yhi[i, n, 1]

        i = x_range_start
        while (x_step > 0 and i < x_range_end) or (x_step < 0 and i > x_range_end):
            j = y_range_start
            while (y_step > 0 and j < y_range_end) or (y_step < 0 and j > y_range_end):
                I0, I1, I2, I3 = _solve_cell_corners(
                    Ox, Oy, dx_arr[i], dy_arr[j],
                    sigma_hat[i, j, 0], sigma_hat[i, j, 1],
                    sigma_hat[i, j, 2], sigma_hat[i, j, 3],
                    source_iso[i, j, 0] + icdt * psi_time[i, j, n, 0],
                    source_iso[i, j, 1] + icdt * psi_time[i, j, n, 1],
                    source_iso[i, j, 2] + icdt * psi_time[i, j, n, 2],
                    source_iso[i, j, 3] + icdt * psi_time[i, j, n, 3],
                    psi_x_face[j, 0], psi_x_face[j, 1],
                    psi_y_face[i, 0], psi_y_face[i, 1])

                if I0 < 0.0:
                    I0 = 0.0
                if I1 < 0.0:
                    I1 = 0.0
                if I2 < 0.0:
                    I2 = 0.0
                if I3 < 0.0:
                    I3 = 0.0

                if compute_phi:
                    phi_angle[n, i, j, 0] = w_n * I0
                    phi_angle[n, i, j, 1] = w_n * I1
                    phi_angle[n, i, j, 2] = w_n * I2
                    phi_angle[n, i, j, 3] = w_n * I3

                if store_psi:
                    psi[i, j, n, 0] = I0
                    psi[i, j, n, 1] = I1
                    psi[i, j, n, 2] = I2
                    psi[i, j, n, 3] = I3

                if Ox > 0:
                    psi_x_face[j, 0] = I3
                    psi_x_face[j, 1] = I0
                else:
                    psi_x_face[j, 0] = I2
                    psi_x_face[j, 1] = I1

                if Oy > 0:
                    psi_y_face[i, 0] = I1
                    psi_y_face[i, 1] = I0
                else:
                    psi_y_face[i, 0] = I2
                    psi_y_face[i, 1] = I3

                j += y_step

            if compute_boundaries and Ox > 0 and i == Ix - 1:
                for jj in range(Iy):
                    psi_xhi_out[jj, n, 0] = psi_x_face[jj, 0]
                    psi_xhi_out[jj, n, 1] = psi_x_face[jj, 1]
            elif compute_boundaries and Ox < 0 and i == 0:
                for jj in range(Iy):
                    psi_xlo_out[jj, n, 0] = psi_x_face[jj, 0]
                    psi_xlo_out[jj, n, 1] = psi_x_face[jj, 1]

            i += x_step

        if compute_boundaries and Oy > 0:
            for ii in range(Ix):
                psi_yhi_out[ii, n, 0] = psi_y_face[ii, 0]
                psi_yhi_out[ii, n, 1] = psi_y_face[ii, 1]
        elif compute_boundaries:
            for ii in range(Ix):
                psi_ylo_out[ii, n, 0] = psi_y_face[ii, 0]
                psi_ylo_out[ii, n, 1] = psi_y_face[ii, 1]

    if compute_phi:
        for ii in prange(Ix):
            for jj in range(Iy):
                for cc in range(4):
                    total = 0.0
                    for n in range(M):
                        total += phi_angle[n, ii, jj, cc]
                    phi[ii, jj, cc] = total

    return (psi, phi, psi_xlo_out, psi_xhi_out,
            psi_ylo_out, psi_yhi_out)


# ===========================================================================
# DMD Acceleration (adapted from 1-D)
# ===========================================================================

def _one_incSVD(y, U, S, V, r, k, eps=1e-14, eps_sv=1e-14):
    """Rank-one incremental SVD update."""
    if k == 0:
        U_new = y.reshape(-1, 1) / np.linalg.norm(y)
        S_new = np.array([np.linalg.norm(y)])
        V_new = np.ones((1, 1))
        return U_new, S_new, V_new, 1, 1

    # Project y onto existing basis
    m = U[:, :r].T @ y
    p = y - U[:, :r] @ m
    p_norm = np.linalg.norm(p)

    if p_norm > eps:
        # Extend basis
        p_hat = p / p_norm
        # Form the (r+1) x (k+1) matrix to SVD
        K_mat = np.zeros((r+1, k+1))
        K_mat[:r, :k] = np.diag(S[:r]) @ V[:r, :k]
        K_mat[:r, k] = m
        K_mat[r, k] = p_norm

        U_k, S_k, Vt_k = np.linalg.svd(K_mat, full_matrices=False)

        # Truncate small singular values
        r_new = np.sum(S_k > eps_sv * S_k[0])
        r_new = max(r_new, 1)

        U_new = np.hstack([U[:, :r], p_hat.reshape(-1, 1)]) @ U_k[:, :r_new]
        S_new = S_k[:r_new]
        V_new = Vt_k[:r_new, :].T
    else:
        # y is in the span of U
        K_mat = np.zeros((r, k+1))
        K_mat[:r, :k] = np.diag(S[:r]) @ V[:r, :k]
        K_mat[:r, k] = m

        U_k, S_k, Vt_k = np.linalg.svd(K_mat, full_matrices=False)

        r_new = np.sum(S_k > eps_sv * S_k[0])
        r_new = max(r_new, 1)

        U_new = U[:, :r] @ U_k[:, :r_new]
        S_new = S_k[:r_new]
        V_new = Vt_k[:r_new, :].T

    return U_new, S_new, V_new, r_new, k + 1


def DMD_prec_inc_2d(matvec, b, K=10, x=None, res=1.0):
    """DMD preconditioner with incremental SVD for 2-D problems."""
    res = min(1.0e-6, res)
    res = max(res, 1e-11)
    N = b.size
    if x is None:
        x = b.copy()

    K_max = 2 * K
    Yplus = np.zeros((N, K_max - 1))
    Yminus = np.zeros((N, K_max - 1))
    r = 0
    k_val = 0

    U = np.zeros((N, 1))
    S = np.array([0.0])
    V = np.zeros((1, 1))

    change = np.empty(K_max)
    change_linf = np.empty(K_max)
    n_filled = 0
    eigs_ok = False
    steady_update = x.copy()

    for k in range(K_max):
        x_new = matvec(x) + b
        L2err = np.sqrt(np.mean(((x - x_new) / (np.abs(x_new) + 1e-14))**2))
        Linferr = np.max(np.abs(x - x_new) / (np.max(np.abs(x_new)) + 1e-14))
        change[k] = L2err
        change_linf[k] = Linferr
        n_filled = k + 1

        if k < K_max - 1:
            Yminus[:, k] = x_new - x

        if k > 0:
            Yplus[:, k-1] = x_new - x
            U, S, V, r, k_val = _one_incSVD(
                Yminus[:, k-1], U, S, V, r=r, k=k_val,
                eps=res*1e-14, eps_sv=res*1e-14)

        x = x_new.copy()

        if k > 1 and r > 0:
            try:
                # Build reduced DMD operator
                spos = S[S > 0]
                mat_size = min(K_max, len(spos))
                if mat_size == 0:
                    continue

                S_inv = np.zeros((mat_size, mat_size))
                S_inv[np.diag_indices(mat_size)] = 1.0 / spos[:mat_size]

                U_r = U[:, :mat_size]
                V_r = V[:k, :mat_size]

                part1 = U_r.T @ Yplus[:, :k]
                part2 = part1 @ V_r
                Atilde = part2 @ S_inv

                eigs, vecs = np.linalg.eig(Atilde)
                eigs_ok = bool(np.max(np.abs(eigs)) <= 1.0)

                if not eigs_ok:
                    eigs[np.abs(eigs) > 1.0] = 0.0

                eigs = np.real(eigs)
                Atilde = np.real(vecs @ np.diag(eigs) @ np.linalg.inv(vecs))

                # Steady-state update
                rhs = U_r.T @ Yplus[:, k-1]
                delta_y = np.linalg.solve(
                    np.eye(Atilde.shape[0]) - Atilde, rhs)
                x_old = -(Yplus[:, k-1] - x)
                steady_update = x_old + U_r @ delta_y

                if k > r + 1 and eigs_ok:
                    return steady_update, change[:n_filled], change_linf[:n_filled]
            except (np.linalg.LinAlgError, ValueError):
                pass

    return steady_update, change[:n_filled], change_linf[:n_filled]


def solver_with_dmd_inc_2d(matvec, b, K=10, Rits=3, x=None,
                           L2_tol=1e-8, Linf_tol=1e-3, max_its=10, LOUD=0):
    """Richardson + DMD-accelerated iterative solver for 2-D problems.

    Uses the proven 1-D DMD implementation (solver_with_dmd_inc from sn_solver)
    which operates on arbitrary-length vectors.
    """
    N = b.size
    if x is None or x.size != N:
        x = b.copy()

    try:
        x_sol, total_its, _chg, _chgL, _At, _Yp, _Ym = _solver_with_dmd_inc_1d(
            matvec=matvec, b=b, K=K, max_its=max_its, steady=1,
            x=x, Rits=Rits, LOUD=LOUD,
            L2_tol=L2_tol, Linf_tol=Linf_tol)
        if np.any(~np.isfinite(x_sol)):
            raise np.linalg.LinAlgError("DMD produced non-finite solution")
        return x_sol, total_its
    except np.linalg.LinAlgError:
        # Fallback to Richardson
        x_sol, total_its = _richardson_solve_2d(
            matvec, b, x, max_its * (Rits + K), L2_tol, Linf_tol, LOUD)
        return x_sol, total_its


def _richardson_solve_2d(matvec, b, x, max_its, L2_tol, Linf_tol, LOUD):
    """Fallback pure Richardson solver."""
    total_its = 0
    for _ in range(max_its):
        x_new = matvec(x) + b
        L2err = np.sqrt(np.mean(((x - x_new) / (np.abs(x_new) + 1e-14))**2))
        Linferr = np.max(np.abs(x - x_new) / (np.max(np.abs(x_new)) + 1e-14))
        x = x_new
        total_its += 1
        if LOUD:
            print(f"  Richardson {total_its}: L2={L2err:.3e}  Linf={Linferr:.3e}")
        if L2err < L2_tol and Linferr < Linf_tol:
            break
    return x, total_its


# ===========================================================================
# Reflecting Boundary Conditions
# ===========================================================================

def _build_reflection_maps(Omega_x, Omega_y):
    """Return angle-partner maps for specular x- and y-reflection."""
    M = len(Omega_x)
    tol = 1e-10
    reflect_map_x = np.zeros(M, dtype=np.int64)
    reflect_map_y = np.zeros(M, dtype=np.int64)

    for n in range(M):
        found_x = False
        found_y = False
        for m in range(M):
            if (abs(Omega_x[m] + Omega_x[n]) < tol and
                    abs(Omega_y[m] - Omega_y[n]) < tol):
                reflect_map_x[n] = m
                found_x = True
            if (abs(Omega_x[m] - Omega_x[n]) < tol and
                    abs(Omega_y[m] + Omega_y[n]) < tol):
                reflect_map_y[n] = m
                found_y = True
            if found_x and found_y:
                break

    return reflect_map_x, reflect_map_y


def build_reflecting_BCs_2d(Omega_x, Omega_y, psi_xlo_out, psi_xhi_out,
                            psi_ylo_out, psi_yhi_out,
                            reflect_xlo=False, reflect_xhi=False,
                            reflect_ylo=False, reflect_yhi=False,
                            reflect_map_x=None, reflect_map_y=None):
    """Build reflecting boundary conditions from outgoing fluxes.

    For reflection: the reflected angle has the normal component reversed.
    - x-face reflection: Omega_x → -Omega_x (find partner angle)
    - y-face reflection: Omega_y → -Omega_y (find partner angle)

    Parameters
    ----------
    Omega_x, Omega_y : (M,) float64
    psi_*_out : outgoing angular fluxes on each boundary face
    reflect_* : bool flags for each boundary

    Returns
    -------
    BCs_xlo, BCs_xhi, BCs_ylo, BCs_yhi : updated incoming BCs
    """
    M = len(Omega_x)
    Iy = psi_xlo_out.shape[0]
    Ix = psi_ylo_out.shape[0]

    BCs_xlo = np.zeros((Iy, M, 2))
    BCs_xhi = np.zeros((Iy, M, 2))
    BCs_ylo = np.zeros((Ix, M, 2))
    BCs_yhi = np.zeros((Ix, M, 2))

    if reflect_map_x is None or reflect_map_y is None:
        reflect_map_x, reflect_map_y = _build_reflection_maps(
            Omega_x, Omega_y)

    # Apply reflections
    for n in range(M):
        if reflect_xlo and Omega_x[n] > 0:
            # Incoming at x_min for angle n (moving right) =
            # outgoing at x_min for reflected angle (moving left)
            n_ref = reflect_map_x[n]
            for j in range(Iy):
                BCs_xlo[j, n, 0] = psi_xlo_out[j, n_ref, 0]
                BCs_xlo[j, n, 1] = psi_xlo_out[j, n_ref, 1]

        if reflect_xhi and Omega_x[n] < 0:
            # Incoming at x_max for angle n (moving left) =
            # outgoing at x_max for reflected angle (moving right)
            n_ref = reflect_map_x[n]
            for j in range(Iy):
                BCs_xhi[j, n, 0] = psi_xhi_out[j, n_ref, 0]
                BCs_xhi[j, n, 1] = psi_xhi_out[j, n_ref, 1]

        if reflect_ylo and Omega_y[n] > 0:
            n_ref = reflect_map_y[n]
            for i in range(Ix):
                BCs_ylo[i, n, 0] = psi_ylo_out[i, n_ref, 0]
                BCs_ylo[i, n, 1] = psi_ylo_out[i, n_ref, 1]

        if reflect_yhi and Omega_y[n] < 0:
            n_ref = reflect_map_y[n]
            for i in range(Ix):
                BCs_yhi[i, n, 0] = psi_yhi_out[i, n_ref, 0]
                BCs_yhi[i, n, 1] = psi_yhi_out[i, n_ref, 1]

    return BCs_xlo, BCs_xhi, BCs_ylo, BCs_yhi


# ===========================================================================
# Main Time-Dependent Gray TRT Solver
# ===========================================================================

def temp_solve_2d(
    Ix, Iy, dx_arr, dy_arr,
    q_ext, sigma_func, scat_func,
    quad_type, N_quad,
    BCs_func, EOS, invEOS,
    phi_init, T_init,
    dt_min=1e-5, dt_max=0.001, tfinal=1.0,
    Linf_tol=1e-5, tolerance=1e-8, maxits=100,
    LOUD=False, K=50, R=3,
    time_outputs=None,
    reflect_xlo=False, reflect_xhi=False,
    reflect_ylo=False, reflect_yhi=False,
    print_stride=0,
    use_dmd=True,
    W=0,
    tau_T=1e-6,
    C_T=1.0,
    omega_T=1.0,
    T_floor=1e-10,
    store_full_history=True,
    step_callback=None,
):
    r"""Time-dependent gray TRT with 2-D S_N corner-balance and DMD acceleration.

    Parameters
    ----------
    Ix, Iy : int
        Number of cells in x and y.
    dx_arr : (Ix,) float64
        Cell widths in x direction.
    dy_arr : (Iy,) float64
        Cell widths in y direction.
    q_ext : (Ix, Iy, 4) or callable(t) → (Ix, Iy, 4)
        External volumetric source (isotropic, per corner).
    sigma_func : callable(T) → (Ix, Iy, 4)
        Absorption opacity as function of temperature array.
    scat_func : callable(T) → (Ix, Iy, 4)
        Scattering opacity as function of temperature.
    quad_type : str
        Angular quadrature type.
    N_quad : int
        Quadrature order parameter.
    BCs_func : callable(t) → dict with keys 'xlo', 'xhi', 'ylo', 'yhi'
        Each value is (face_cells, M, 2) or scalar for uniform incoming.
        Can return None for vacuum (zero) on a face.
    EOS : callable(T) → (Ix, Iy, 4)
        Internal energy from temperature.
    invEOS : callable(e) → (Ix, Iy, 4)
        Temperature from internal energy.
    phi_init : (Ix, Iy, 4) float64
        Initial scalar flux.
    T_init : (Ix, Iy, 4) float64
        Initial temperature.
    dt_min, dt_max : float
        Time step bounds.
    tfinal : float
        Final time.
    tolerance, Linf_tol : float
        Convergence tolerances.
    maxits : int
        Max DMD iterations per step.
    K, R : int
        DMD snapshot count, Richardson iterations between DMD.
    reflect_* : bool
        Reflecting boundary flags.
    W : int
        T_star outer iterations (0 = single linearization).
    store_full_history : bool
        Store full ``phi`` and ``T`` fields at every accepted step. If false,
        store only the initial state, requested output times, and final state.
    step_callback : callable(t, phi, T) or None
        Optional read-only callback invoked for the initial state and every
        accepted step. This supports lightweight diagnostics such as
        fiducial-point histories without retaining all full fields.

    Returns
    -------
    phis : list of (Ix, Iy, 4)
    Ts : list of (Ix, Iy, 4)
    iterations : int
    ts : ndarray
    its_per_step : list
    """
    # Get quadrature
    Omega_x, Omega_y, weights = get_2d_quadrature(quad_type, N_quad)
    M = len(Omega_x)

    # Convert to contiguous arrays for numba
    Omega_x = np.ascontiguousarray(Omega_x, dtype=np.float64)
    Omega_y = np.ascontiguousarray(Omega_y, dtype=np.float64)
    weights = np.ascontiguousarray(weights, dtype=np.float64)
    dx_arr = np.ascontiguousarray(dx_arr, dtype=np.float64)
    dy_arr = np.ascontiguousarray(dy_arr, dtype=np.float64)

    # Initialize state
    phi = phi_init.copy()
    T = T_init.copy()
    T_old = T.copy()
    T_old2 = T.copy()
    e_old = EOS(T)

    # Per-angle flux from previous step (needed for time-derivative source)
    # Initialize isotropically from phi_init
    psi_old = np.broadcast_to(phi_init[:, :, np.newaxis, :],
                              (Ix, Iy, M, 4)).copy()

    # Outgoing boundary fluxes (for reflecting BCs)
    psi_xlo_out = np.zeros((Iy, M, 2))
    psi_xhi_out = np.zeros((Iy, M, 2))
    psi_ylo_out = np.zeros((Ix, M, 2))
    psi_yhi_out = np.zeros((Ix, M, 2))

    # Invariant workspaces used by every scattering matvec.
    zero_xlo = np.zeros((Iy, M, 2))
    zero_xhi = np.zeros((Iy, M, 2))
    zero_ylo = np.zeros((Ix, M, 2))
    zero_yhi = np.zeros((Ix, M, 2))
    phi_angle_workspace = np.empty((M, Ix, Iy, 4))

    reflection_enabled = (reflect_xlo or reflect_xhi or
                          reflect_ylo or reflect_yhi)
    if reflection_enabled:
        reflect_map_x, reflect_map_y = _build_reflection_maps(
            Omega_x, Omega_y)
    else:
        reflect_map_x = None
        reflect_map_y = None

    # Storage
    phis = [phi.copy()]
    Ts = [T.copy()]
    ts = [0.0]
    its_per_step = []
    iterations = 0

    t_current = 0.0
    step_num = 0
    dt = dt_min
    dt_old = dt_min
    deriv_val = 0.0
    delta_step = 1e-3
    curr_step = 0
    t_output_index = 0
    _T_max = 1e50

    if time_outputs is not None:
        time_outputs = np.asarray(time_outputs, dtype=float)

    if step_callback is not None:
        step_callback(0.0, phi, T)

    print(f"2D-S_N Corner Balance: Ix={Ix}, Iy={Iy}, M={M} angles, quad={quad_type} N={N_quad}")
    print(f"  dt range: [{dt_min:.2e}, {dt_max:.2e}], tfinal={tfinal}")
    print(f"  tolerances: L2={tolerance:.1e}, Linf={Linf_tol:.1e}, maxits={maxits}")
    print(f"  reflecting BCs: xlo={reflect_xlo} xhi={reflect_xhi} ylo={reflect_ylo} yhi={reflect_yhi}")
    print("|", end="", flush=True)

    while t_current < tfinal:
        dt_old2 = dt_old
        dt_old = dt
        step_num += 1

        # Adaptive time step
        if step_num > 2:
            dt_prop = np.sqrt(delta_step * deriv_val) if deriv_val > 0 else dt_max
            dt_prop = np.clip(dt_prop, dt_min, dt_max)
            if dt_prop > 2.0 * dt:
                dt_prop = dt * 1.5
            dt = dt_prop
        else:
            dt = dt_min

        # Snap to final time or output times
        if (tfinal - t_current) < dt:
            snap = tfinal - t_current
            if snap > 1e-10 * dt_min:
                dt = snap
            else:
                break

        if time_outputs is not None:
            output_tol = max(1e-12, 1e-10 * max(abs(t_current), dt_min))
            while (t_output_index < len(time_outputs) and
                   time_outputs[t_output_index] <= t_current + output_tol):
                t_output_index += 1

        if time_outputs is not None and t_output_index < len(time_outputs):
            if t_current + dt > time_outputs[t_output_index]:
                snap_dt = time_outputs[t_output_index] - t_current
                if snap_dt > 1e-10 * dt_min:
                    dt = snap_dt
                    t_output_index += 1

        if math.isnan(dt):
            dt = dt_min

        if LOUD:
            print(f"t = {t_current:.4e}, dt = {dt:.4e}")
        t_current += dt

        if int(10 * t_current / tfinal) > curr_step:
            curr_step += 1
            print(curr_step, end="", flush=True)

        icdt = 1.0 / (c * dt)
        iterations_step = 0

        # External sources and prescribed BCs depend on time, but not on the
        # nonlinear T_star iteration, so prepare them once per accepted step.
        if callable(q_ext):
            q_now = q_ext(t_current)
        else:
            q_now = q_ext

        bc_data = BCs_func(t_current - dt/2.0)
        BCs_xlo_use = np.zeros((Iy, M, 2))
        BCs_xhi_use = np.zeros((Iy, M, 2))
        BCs_ylo_use = np.zeros((Ix, M, 2))
        BCs_yhi_use = np.zeros((Ix, M, 2))

        if bc_data is not None:
            if 'xlo' in bc_data and bc_data['xlo'] is not None:
                bc_xlo = bc_data['xlo']
                if np.isscalar(bc_xlo):
                    BCs_xlo_use[:, :, :] = bc_xlo
                else:
                    BCs_xlo_use = np.ascontiguousarray(
                        bc_xlo, dtype=np.float64)
            if 'xhi' in bc_data and bc_data['xhi'] is not None:
                bc_xhi = bc_data['xhi']
                if np.isscalar(bc_xhi):
                    BCs_xhi_use[:, :, :] = bc_xhi
                else:
                    BCs_xhi_use = np.ascontiguousarray(
                        bc_xhi, dtype=np.float64)
            if 'ylo' in bc_data and bc_data['ylo'] is not None:
                bc_ylo = bc_data['ylo']
                if np.isscalar(bc_ylo):
                    BCs_ylo_use[:, :, :] = bc_ylo
                else:
                    BCs_ylo_use = np.ascontiguousarray(
                        bc_ylo, dtype=np.float64)
            if 'yhi' in bc_data and bc_data['yhi'] is not None:
                bc_yhi = bc_data['yhi']
                if np.isscalar(bc_yhi):
                    BCs_yhi_use[:, :, :] = bc_yhi
                else:
                    BCs_yhi_use = np.ascontiguousarray(
                        bc_yhi, dtype=np.float64)

        # === T_star nonlinear outer loop ===
        T_star = np.minimum(T_old, _T_max)
        do_final = False
        k_outer = 0
        x_sol = phi.ravel()
        # Preserve start-of-step psi for icdt*psi^m source (must not change during T_star iterations)
        psi_step_start = psi_old

        while True:
            # Linearize at T_star
            h_cv = np.maximum(np.abs(T_star) * 1e-4, 1e-12)
            Cv = np.maximum(
                (EOS(T_star + h_cv) - EOS(np.maximum(T_star - h_cv, T_floor))) / (2.0 * h_cv),
                1e-30)
            beta_val = 4.0 * a * T_star**3 / Cv
            sigma = sigma_func(T_star)
            scat = scat_func(T_star)
            f = 1.0 / (1.0 + beta_val * c * dt * sigma)
            sigma_a = f * sigma
            sigma_s = (1.0 - f) * sigma + scat
            sigma_t = sigma + scat
            sigma_hat = sigma_t + icdt  # (Ix, Iy, 4)
            emission = sigma_a * ac * T_star**4

            # Delta-e source term
            delta_e_src = -(1.0 - f) * (EOS(T_star) - e_old) / dt

            # Isotropic part of the fixed source. The angle-dependent
            # time-derivative term is fused into sweep_2d_transient.
            source_isotropic = emission + delta_e_src + q_now  # (Ix, Iy, 4)

            sigma_hat_arr = np.ascontiguousarray(sigma_hat)

            # Apply reflecting BCs
            if reflect_xlo or reflect_xhi or reflect_ylo or reflect_yhi:
                ref_xlo, ref_xhi, ref_ylo, ref_yhi = build_reflecting_BCs_2d(
                    Omega_x, Omega_y,
                    psi_xlo_out, psi_xhi_out, psi_ylo_out, psi_yhi_out,
                    reflect_xlo, reflect_xhi, reflect_ylo, reflect_yhi,
                    reflect_map_x, reflect_map_y)
                if reflect_xlo:
                    BCs_xlo_use = ref_xlo
                if reflect_xhi:
                    BCs_xhi_use = ref_xhi
                if reflect_ylo:
                    BCs_ylo_use = ref_ylo
                if reflect_yhi:
                    BCs_yhi_use = ref_yhi

            # === Source iteration with DMD, wrapped in reflecting BC loop ===
            def matvec(phi_vec):
                """One scattering iteration: sweep(sigma_s * phi).
                Uses sweep_2d_iso to avoid allocating (Ix,Iy,M,4) array."""
                phi_3d = phi_vec.reshape((Ix, Iy, 4))
                scatter_iso = np.ascontiguousarray(sigma_s * phi_3d)
                phi_new, _, _, _, _ = sweep_2d_iso(
                    Ix, Iy, dx_arr, dy_arr, sigma_hat_arr,
                    scatter_iso,
                    Omega_x, Omega_y, weights,
                    zero_xlo, zero_xhi, zero_ylo, zero_yhi,
                    True, False, phi_angle_workspace)
                return phi_new.ravel()

            tau_phi = tolerance if (W == 0 or do_final) else tolerance * (1e-3 / tolerance) ** (1.0 - k_outer / max(W, 1))
            max_reflect_its = 10 if reflection_enabled else 1
            reflect_tol = 1e-10

            for _ref_it in range(max_reflect_its):
                # Update reflecting BCs from current outgoing fluxes
                if reflection_enabled and _ref_it > 0:
                    ref_xlo, ref_xhi, ref_ylo, ref_yhi = build_reflecting_BCs_2d(
                        Omega_x, Omega_y,
                        psi_xlo_out, psi_xhi_out, psi_ylo_out, psi_yhi_out,
                        reflect_xlo, reflect_xhi, reflect_ylo, reflect_yhi,
                        reflect_map_x, reflect_map_y)
                    if reflect_xlo:
                        BCs_xlo_use = ref_xlo
                    if reflect_xhi:
                        BCs_xhi_use = ref_xhi
                    if reflect_ylo:
                        BCs_ylo_use = ref_ylo
                    if reflect_yhi:
                        BCs_yhi_use = ref_yhi

                # Compute b = sweep of fixed source with real BCs
                _, b_phi, _, _, _, _ = sweep_2d_transient(
                    Ix, Iy, dx_arr, dy_arr, sigma_hat_arr,
                    np.ascontiguousarray(source_isotropic),
                    psi_step_start, icdt,
                    Omega_x, Omega_y, weights,
                    np.ascontiguousarray(BCs_xlo_use),
                    np.ascontiguousarray(BCs_xhi_use),
                    np.ascontiguousarray(BCs_ylo_use),
                    np.ascontiguousarray(BCs_yhi_use),
                    True, False, False, phi_angle_workspace)
                b_vec = b_phi.ravel()

                # Iterative solve (full DMD)
                if use_dmd:
                    try:
                        x_sol, total_its = solver_with_dmd_inc_2d(
                            matvec, b_vec, K=K, Rits=R, x=x_sol,
                            L2_tol=tau_phi, Linf_tol=Linf_tol,
                            max_its=maxits, LOUD=LOUD)
                    except Exception:
                        x_sol, total_its = _richardson_solve_2d(
                            matvec, b_vec, x_sol, maxits, tau_phi, Linf_tol, LOUD)
                else:
                    x_sol, total_its = _richardson_solve_2d(
                        matvec, b_vec, x_sol, maxits, tau_phi, Linf_tol, LOUD)

                iterations += total_its
                iterations_step += total_its
                phi = x_sol.reshape((Ix, Iy, 4))

                if LOUD and _ref_it > 0:
                    print(f"    reflect iter {_ref_it}: sweeps={total_its}")

                if not reflection_enabled:
                    break

                # Reconstruct only the outgoing boundary fluxes.
                phi_3d_ref = x_sol.reshape((Ix, Iy, 4))
                scatter_ref = sigma_s * phi_3d_ref
                boundary_source_iso = scatter_ref + source_isotropic
                _, _, psi_xlo_new, psi_xhi_new, psi_ylo_new, psi_yhi_new = \
                    sweep_2d_transient(
                        Ix, Iy, dx_arr, dy_arr, sigma_hat_arr,
                        np.ascontiguousarray(boundary_source_iso),
                        psi_step_start, icdt,
                        Omega_x, Omega_y, weights,
                        np.ascontiguousarray(BCs_xlo_use),
                        np.ascontiguousarray(BCs_xhi_use),
                        np.ascontiguousarray(BCs_ylo_use),
                        np.ascontiguousarray(BCs_yhi_use),
                        False, False, True)

                # Check reflecting BC convergence
                bc_change = (np.max(np.abs(psi_xlo_new - psi_xlo_out)) +
                             np.max(np.abs(psi_xhi_new - psi_xhi_out)) +
                             np.max(np.abs(psi_ylo_new - psi_ylo_out)) +
                             np.max(np.abs(psi_yhi_new - psi_yhi_out)))
                psi_xlo_out = psi_xlo_new
                psi_xhi_out = psi_xhi_new
                psi_ylo_out = psi_ylo_new
                psi_yhi_out = psi_yhi_new

                if bc_change < reflect_tol:
                    if LOUD:
                        print(f"    reflecting BCs converged: {_ref_it+1} iters, change={bc_change:.2e}")
                    break

            # T_star exit/update
            if W == 0 or do_final:
                break

            # Update T_star from material energy equation
            delta_e_star = EOS(T_star) - e_old
            e_cand = (e_old + sigma_a * dt * (phi - ac * T_star**4)
                      + (1.0 - f) * delta_e_star)
            T_cand = invEOS(e_cand)
            T_star_prev = T_star.copy()
            T_star = np.clip(
                (1.0 - omega_T) * T_star + omega_T * T_cand,
                T_floor, _T_max)

            eta_T = float(np.sqrt(np.mean(
                ((T_star - T_star_prev) / (np.abs(T_star_prev) + T_floor))**2)))
            k_outer += 1
            if eta_T < tau_T or k_outer >= W:
                do_final = True

        # === Material energy update ===
        delta_e_star = EOS(T_star) - e_old
        e = (e_old + sigma_a * dt * (phi - ac * T_star**4)
             + (1.0 - f) * delta_e_star)
        T = invEOS(e)

        # === Update per-angle flux for next step's time-derivative source ===
        # Do a final sweep with converged phi to get the per-angle intensities
        scatter_final = sigma_s * phi
        final_source_iso = scatter_final + source_isotropic
        psi_new, _, psi_xlo_out, psi_xhi_out, psi_ylo_out, psi_yhi_out = \
            sweep_2d_transient(
                Ix, Iy, dx_arr, dy_arr, sigma_hat_arr,
                np.ascontiguousarray(final_source_iso), psi_old, icdt,
                Omega_x, Omega_y, weights,
                np.ascontiguousarray(BCs_xlo_use),
                np.ascontiguousarray(BCs_xhi_use),
                np.ascontiguousarray(BCs_ylo_use),
                np.ascontiguousarray(BCs_yhi_use),
                False, True, True)
        psi_old = psi_new

        # Max temperature change this step
        dT_max = float(np.max(np.abs(T - T_old)))

        # Diagnostics — always print periodically so user can track progress
        if print_stride > 0 and step_num % print_stride == 0:
            print(f"  step {step_num:5d}  t={t_current:.4e} ns  dt={dt:.3e}  "
                  f"T_max={np.max(T):.6f} keV  dT_max={dT_max:.3e}  "
                  f"sweeps={iterations_step}  T_iters={k_outer}")
        elif step_num <= 3 or (step_num <= 20 and step_num % 5 == 0):
            # Always print the first few steps so user sees progress immediately
            print(f"  step {step_num:5d}  t={t_current:.4e} ns  dt={dt:.3e}  "
                  f"T_max={np.max(T):.6f} keV  dT_max={dT_max:.3e}  "
                  f"sweeps={iterations_step}  T_iters={k_outer}")

        # Adaptive dt: second time derivative of T
        if step_num >= 2:
            T_flat = T.ravel()
            T_old_flat = T_old.ravel()
            T_old2_flat = T_old2.ravel()
            denom = np.mean(np.abs(2.0 * (
                T_flat / (dt * (dt + dt_old))
                - T_old_flat / (dt * dt_old)
                + T_old2_flat / (dt_old * (dt + dt_old)))))
            mean_T = np.mean(T_flat)
            if denom > 0 and np.isfinite(mean_T) and np.isfinite(denom):
                deriv_val = mean_T / denom
            else:
                deriv_val = dt_max**2 / delta_step

        e_old = EOS(T).copy()
        T_old2 = T_old.copy()
        T_old = T.copy()
        its_per_step.append(iterations_step)

        if step_callback is not None:
            step_callback(t_current, phi, T)

        store_step = store_full_history
        if not store_step:
            store_tol = max(1e-12, 1e-10 * max(abs(t_current), dt_min))
            at_final = abs(t_current - tfinal) <= store_tol
            at_output = (time_outputs is not None and
                         np.any(np.abs(time_outputs - t_current) <= store_tol))
            store_step = at_final or at_output

        if store_step:
            ts.append(t_current)
            phis.append(phi.copy())
            Ts.append(T.copy())

    print(f"\n  Finished: {step_num} steps, {iterations} total sweeps, "
          f"avg {iterations/max(step_num,1):.1f} sweeps/step")
    return phis, Ts, iterations, np.array(ts), its_per_step
