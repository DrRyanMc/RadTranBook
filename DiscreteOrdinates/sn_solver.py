"""
1D Discrete Ordinates (S_N) solver for time-dependent radiative transfer.

Uses Bernstein polynomial basis functions for the spatial discretization
and DMD (Dynamic Mode Decomposition) to accelerate convergence of
transport sweeps.
"""

import numpy as np
import math
from numba import jit, int32, float64, prange

# Physical constants (CGS units, time in nanoseconds)
c = 29.98       # speed of light (cm/ns)
a = 0.01372     # radiation constant (GJ/(cm^3 keV^4))
ac = a * c


# ---------------------------------------------------------------------------
# Caches for quadrature and precomputed matrices
# ---------------------------------------------------------------------------
_quad_cache = {}
_mk_cache = {}


def _get_quadrature(N):
    """Return cached (MU, W) Gauss-Legendre quadrature, W normalised."""
    if N not in _quad_cache:
        MU, W = np.polynomial.legendre.leggauss(N)
        W = W / np.sum(W)
        _quad_cache[N] = (np.ascontiguousarray(MU, dtype=np.float64),
                          np.ascontiguousarray(W, dtype=np.float64))
    return _quad_cache[N]


def _build_MK(order, hx):
    """Compute lumped mass diagonal (1-D) and stiffness matrix K for *order*."""
    if order == 1:
        M_raw = np.array([[0.333333, 0.166667],
                          [0.166667, 0.333333]])
        K = np.array([[-0.5, -0.5],
                       [0.5,  0.5]])
    elif order == 2:
        M_raw = np.array([[0.2, 0.1, 0.0333333],
                          [0.1, 0.133333, 0.1],
                          [0.0333333, 0.1, 0.2]])
        K = np.array([[-0.5, -0.333333, -0.166667],
                       [0.333333, 0., -0.333333],
                       [0.166667, 0.333333, 0.5]])
    elif order == 3:
        M_raw = np.array([[0.142857, 0.0714286, 0.0285714, 0.00714286],
                          [0.0714286, 0.0857143, 0.0642857, 0.0285714],
                          [0.0285714, 0.0642857, 0.0857143, 0.0714286],
                          [0.00714286, 0.0285714, 0.0714286, 0.142857]])
        K = np.array([[-0.5, -0.3, -0.15, -0.05],
                       [0.3, 0., -0.15, -0.15],
                       [0.15, 0.15, 0., -0.3],
                       [0.05, 0.15, 0.3, 0.5]])
    elif order == 4:
        M_raw = np.array([[0.111111, 0.0555556, 0.0238095, 0.00793651, 0.0015873],
                          [0.0555556, 0.0634921, 0.047619, 0.0253968, 0.00793651],
                          [0.0238095, 0.047619, 0.0571429, 0.047619, 0.0238095],
                          [0.00793651, 0.0253968, 0.047619, 0.0634921, 0.0555556],
                          [0.0015873, 0.00793651, 0.0238095, 0.0555556, 0.111111]])
        K = np.array([[-0.5, -0.285714, -0.142857, -0.0571429, -0.0142857],
                       [0.285714, 0., -0.114286, -0.114286, -0.0571429],
                       [0.142857, 0.114286, 0., -0.114286, -0.142857],
                       [0.0571429, 0.114286, 0.114286, 0., -0.285714],
                       [0.0142857, 0.0571429, 0.142857, 0.285714, 0.5]])
    elif order == 5:
        M_raw = np.array([
            [0.0909091, 0.0454545, 0.020202, 0.00757576, 0.0021645, 0.00036075],
            [0.0454545, 0.0505051, 0.0378788, 0.021645, 0.00901876, 0.0021645],
            [0.020202, 0.0378788, 0.04329, 0.036075, 0.021645, 0.00757576],
            [0.00757576, 0.021645, 0.036075, 0.04329, 0.0378788, 0.020202],
            [0.0021645, 0.00901876, 0.021645, 0.0378788, 0.0505051, 0.0454545],
            [0.00036075, 0.0021645, 0.00757576, 0.020202, 0.0454545, 0.0909091]])
        K = np.array([
            [-0.5, -0.277778, -0.138889, -0.0595238, -0.0198413, -0.00396825],
            [0.277778, 0., -0.0992063, -0.0992063, -0.0595238, -0.0198413],
            [0.138889, 0.0992063, 0., -0.0793651, -0.0992063, -0.0595238],
            [0.0595238, 0.0992063, 0.0793651, 0., -0.0992063, -0.138889],
            [0.0198413, 0.0595238, 0.0992063, 0.0992063, 0., -0.277778],
            [0.00396825, 0.0198413, 0.0595238, 0.138889, 0.277778, 0.5]])
    elif order == 6:
        M_raw = np.array([
            [0.0769231, 0.0384615, 0.0174825, 0.00699301, 0.002331, 0.000582751, 0.0000832501],
            [0.0384615, 0.041958, 0.0314685, 0.018648, 0.00874126, 0.002997, 0.000582751],
            [0.0174825, 0.0314685, 0.034965, 0.0291375, 0.0187313, 0.00874126, 0.002331],
            [0.00699301, 0.018648, 0.0291375, 0.0333, 0.0291375, 0.018648, 0.00699301],
            [0.002331, 0.00874126, 0.0187313, 0.0291375, 0.034965, 0.0314685, 0.0174825],
            [0.000582751, 0.002997, 0.00874126, 0.018648, 0.0314685, 0.041958, 0.0384615],
            [0.0000832501, 0.000582751, 0.002331, 0.00699301, 0.0174825, 0.0384615, 0.0769231]])
        K = np.array([
            [-0.5, -0.272727, -0.136364, -0.0606061, -0.0227273, -0.00649351, -0.00108225],
            [0.272727, 0., -0.0909091, -0.0909091, -0.0584416, -0.025974, -0.00649351],
            [0.136364, 0.0909091, 0., -0.0649351, -0.0811688, -0.0584416, -0.0227273],
            [0.0606061, 0.0909091, 0.0649351, 0., -0.0649351, -0.0909091, -0.0606061],
            [0.0227273, 0.0584416, 0.0811688, 0.0649351, 0., -0.0909091, -0.136364],
            [0.00649351, 0.025974, 0.0584416, 0.0909091, 0.0909091, 0., -0.272727],
            [0.00108225, 0.00649351, 0.0227273, 0.0606061, 0.136364, 0.272727, 0.5]])
    else:
        raise ValueError(f"Unsupported order {order}")
    nop1 = order + 1
    M_diag = np.empty(nop1)
    for i in range(nop1):
        M_diag[i] = np.sum(M_raw[i, :]) * hx
    return np.ascontiguousarray(M_diag), np.ascontiguousarray(K, dtype=np.float64)


def _get_matrices(order, hx):
    """Return cached (M_diag, K) for given order and zone width."""
    key = (order, hx)
    if key not in _mk_cache:
        _mk_cache[key] = _build_MK(order, hx)
    return _mk_cache[key]


# ---------------------------------------------------------------------------
# Optimised core sweep (precomputed M_diag / K, base-LHS hoisted)
# ---------------------------------------------------------------------------
@jit(nopython=True, cache=True)
def _sweep_core(I, M_diag, K, q, sigma_t, mu, boundary, order, fix):
    """Single-direction sweep with precomputed lumped M diagonal and K."""
    nop1 = order + 1
    psi = np.zeros((I, nop1))
    if mu > 0:
        psi_left = boundary.copy()
        # base LHS: -mu*K + diag(iplushalf)*mu  (constant across cells)
        base_lhs = -mu * K.copy()
        base_lhs[nop1 - 1, nop1 - 1] += mu
        for i in range(I):
            rhs = M_diag * q[i, :]
            rhs[0] += mu * psi_left[nop1 - 1]
            lhs = base_lhs.copy()
            for j in range(nop1):
                lhs[j, j] += M_diag[j] * sigma_t[i, j]
            tmp = np.linalg.solve(lhs, rhs)
            if (fix > 0) and (np.min(tmp) < 0):
                tmpZ = (tmp > 0) * tmp
                tmp = tmpZ * (np.sum(rhs) / (np.sum(np.dot(lhs, tmpZ)) + 1e-14))
            psi[i, :] = tmp
            psi_left = tmp
    else:
        psi_right = boundary.copy()
        base_lhs = -mu * K.copy()
        base_lhs[0, 0] -= mu
        for i in range(I - 1, -1, -1):
            rhs = M_diag * q[i, :]
            rhs[nop1 - 1] += -mu * psi_right[0]
            lhs = base_lhs.copy()
            for j in range(nop1):
                lhs[j, j] += M_diag[j] * sigma_t[i, j]
            tmp = np.linalg.solve(lhs, rhs)
            if (fix > 0) and (np.min(tmp) < 0):
                tmpZ = (tmp > 0) * tmp
                tmp = tmpZ * (np.sum(rhs) / (np.sum(np.dot(lhs, tmpZ)) + 1e-14))
            psi[i, :] = tmp
            psi_right = tmp
    return psi


# ---------------------------------------------------------------------------
# Thread-parallel all-angles sweeps
# ---------------------------------------------------------------------------
@jit(nopython=True, parallel=True, cache=True)
def _sweep_all_angles_phi(I, M_diag, K, source_3d, sigma_t, MU, W, BCs,
                          order, fix):
    """Sweep all angles with prange, return scalar flux phi (I, nop1)."""
    N = MU.size
    nop1 = order + 1
    psi_tmp = np.zeros((N, I, nop1))
    for n in prange(N):
        q_n = source_3d[:, n, :].copy()
        bc = BCs[n, :].copy()
        psi_tmp[n] = _sweep_core(I, M_diag, K, q_n, sigma_t,
                                 MU[n], bc, order, fix)
    phi = np.zeros((I, nop1))
    for n in range(N):
        phi += psi_tmp[n] * W[n]
    return phi


@jit(nopython=True, parallel=True, cache=True)
def _sweep_all_angles_psi(I, M_diag, K, source_3d, sigma_t, MU, W, BCs,
                          order, fix):
    """Sweep all angles with prange, return psi (I, N, nop1)."""
    N = MU.size
    nop1 = order + 1
    psi_tmp = np.zeros((N, I, nop1))
    for n in prange(N):
        q_n = source_3d[:, n, :].copy()
        bc = BCs[n, :].copy()
        psi_tmp[n] = _sweep_core(I, M_diag, K, q_n, sigma_t,
                                 MU[n], bc, order, fix)
    # reorder to (I, N, nop1)
    psi = np.empty((I, N, nop1))
    for n in range(N):
        psi[:, n, :] = psi_tmp[n]
    return psi


# ---------------------------------------------------------------------------
# Transport sweep (backward-compatible public API)
# ---------------------------------------------------------------------------
@jit(float64[:,:](int32, float64, float64[:,:], float64[:,:], float64,
                  float64[:], int32, int32), nopython=True, cache=True)
def sweep1D_bern(I, hx, q, sigma_t, mu, boundary, order=1, fix=0):
    """Compute a transport sweep in a single direction.

    Parameters
    ----------
    I : int
        Number of spatial zones.
    hx : float
        Zone width.
    q : ndarray (I, order+1)
        Source in each zone / node.
    sigma_t : ndarray (I, order+1)
        Total cross-section in each zone / node.
    mu : float
        Direction cosine.
    boundary : ndarray (order+1,)
        Incoming angular flux on the upstream boundary.
    order : int
        Bernstein polynomial order (1-6).
    fix : int
        If >0 apply a positivity fix-up.

    Returns
    -------
    psi : ndarray (I, order+1)
    """
    lumped = 1
    psi = np.zeros((I, order + 1))
    h = hx

    # ---- mass and stiffness matrices for each supported order ----
    if order == 1:
        M = np.array([[0.333333, 0.166667],
                       [0.166667, 0.333333]]) * h
        K = np.array([[-0.5, -0.5],
                       [0.5,  0.5]])
    elif order == 2:
        M = np.array([[0.2, 0.1, 0.0333333],
                       [0.1, 0.133333, 0.1],
                       [0.0333333, 0.1, 0.2]]) * h
        K = np.array([[-0.5, -0.333333, -0.166667],
                       [0.333333, 0., -0.333333],
                       [0.166667, 0.333333, 0.5]])
    elif order == 3:
        M = np.array([[0.142857, 0.0714286, 0.0285714, 0.00714286],
                       [0.0714286, 0.0857143, 0.0642857, 0.0285714],
                       [0.0285714, 0.0642857, 0.0857143, 0.0714286],
                       [0.00714286, 0.0285714, 0.0714286, 0.142857]]) * h
        K = np.array([[-0.5, -0.3, -0.15, -0.05],
                       [0.3, 0., -0.15, -0.15],
                       [0.15, 0.15, 0., -0.3],
                       [0.05, 0.15, 0.3, 0.5]])
    elif order == 4:
        M = np.array([[0.111111, 0.0555556, 0.0238095, 0.00793651, 0.0015873],
                       [0.0555556, 0.0634921, 0.047619, 0.0253968, 0.00793651],
                       [0.0238095, 0.047619, 0.0571429, 0.047619, 0.0238095],
                       [0.00793651, 0.0253968, 0.047619, 0.0634921, 0.0555556],
                       [0.0015873, 0.00793651, 0.0238095, 0.0555556, 0.111111]]) * h
        K = np.array([[-0.5, -0.285714, -0.142857, -0.0571429, -0.0142857],
                       [0.285714, 0., -0.114286, -0.114286, -0.0571429],
                       [0.142857, 0.114286, 0., -0.114286, -0.142857],
                       [0.0571429, 0.114286, 0.114286, 0., -0.285714],
                       [0.0142857, 0.0571429, 0.142857, 0.285714, 0.5]])
    elif order == 5:
        M = np.array([
            [0.0909091, 0.0454545, 0.020202, 0.00757576, 0.0021645, 0.00036075],
            [0.0454545, 0.0505051, 0.0378788, 0.021645, 0.00901876, 0.0021645],
            [0.020202, 0.0378788, 0.04329, 0.036075, 0.021645, 0.00757576],
            [0.00757576, 0.021645, 0.036075, 0.04329, 0.0378788, 0.020202],
            [0.0021645, 0.00901876, 0.021645, 0.0378788, 0.0505051, 0.0454545],
            [0.00036075, 0.0021645, 0.00757576, 0.020202, 0.0454545, 0.0909091]
        ]) * h
        K = np.array([
            [-0.5, -0.277778, -0.138889, -0.0595238, -0.0198413, -0.00396825],
            [0.277778, 0., -0.0992063, -0.0992063, -0.0595238, -0.0198413],
            [0.138889, 0.0992063, 0., -0.0793651, -0.0992063, -0.0595238],
            [0.0595238, 0.0992063, 0.0793651, 0., -0.0992063, -0.138889],
            [0.0198413, 0.0595238, 0.0992063, 0.0992063, 0., -0.277778],
            [0.00396825, 0.0198413, 0.0595238, 0.138889, 0.277778, 0.5]
        ])
    elif order == 6:
        M = np.array([
            [0.0769231, 0.0384615, 0.0174825, 0.00699301, 0.002331, 0.000582751, 0.0000832501],
            [0.0384615, 0.041958, 0.0314685, 0.018648, 0.00874126, 0.002997, 0.000582751],
            [0.0174825, 0.0314685, 0.034965, 0.0291375, 0.0187313, 0.00874126, 0.002331],
            [0.00699301, 0.018648, 0.0291375, 0.0333, 0.0291375, 0.018648, 0.00699301],
            [0.002331, 0.00874126, 0.0187313, 0.0291375, 0.034965, 0.0314685, 0.0174825],
            [0.000582751, 0.002997, 0.00874126, 0.018648, 0.0314685, 0.041958, 0.0384615],
            [0.0000832501, 0.000582751, 0.002331, 0.00699301, 0.0174825, 0.0384615, 0.0769231]
        ]) * h
        K = np.array([
            [-0.5, -0.272727, -0.136364, -0.0606061, -0.0227273, -0.00649351, -0.00108225],
            [0.272727, 0., -0.0909091, -0.0909091, -0.0584416, -0.025974, -0.00649351],
            [0.136364, 0.0909091, 0., -0.0649351, -0.0811688, -0.0584416, -0.0227273],
            [0.0606061, 0.0909091, 0.0649351, 0., -0.0649351, -0.0909091, -0.0606061],
            [0.0227273, 0.0584416, 0.0811688, 0.0649351, 0., -0.0909091, -0.136364],
            [0.00649351, 0.025974, 0.0584416, 0.0909091, 0.0909091, 0., -0.272727],
            [0.00108225, 0.00649351, 0.0227273, 0.0606061, 0.136364, 0.272727, 0.5]
        ])

    # lump the mass matrix
    if lumped > 0:
        M_new = M * 0
        for i in range(order + 1):
            M_new[i, i] = np.sum(M[i, :])
        M = M_new

    if mu > 0:
        psi_left = boundary
        for i in range(I):
            iplushalf = np.zeros(order + 1)
            iplushalf[-1] = 1

            rhs = np.dot(M, q[i, :])
            rhs[0] += mu * psi_left[-1]
            lhs = M * np.diag(sigma_t[i]) - mu * K + np.diag(iplushalf) * mu
            tmp = np.linalg.solve(lhs, rhs)
            if (fix > 0) and (np.min(tmp) < 0):
                tmpZ = (tmp > 0) * tmp
                tmp = tmpZ * (np.sum(rhs) / (np.sum(np.dot(lhs, tmpZ)) + 1e-14))
            psi[i, :] = tmp.reshape(order + 1)
            psi_left = tmp
    else:
        psi_right = boundary
        iminhalf = np.zeros(order + 1)
        iminhalf[0] = 1
        for i in range(I - 1, -1, -1):
            rhs = np.dot(M, q[i, :])
            rhs[-1] += -mu * psi_right[0]
            lhs = M * np.diag(sigma_t[i]) - mu * K - np.diag(iminhalf) * mu
            tmp = np.linalg.solve(lhs, rhs)
            if (fix > 0) and (np.min(tmp) < 0):
                tmpZ = (tmp > 0) * tmp
                tmp = tmpZ * (np.sum(rhs) / (np.sum(np.dot(lhs, tmpZ)) + 1e-14))
            psi[i, :] = tmp.reshape(order + 1)
            psi_right = tmp
    return psi


# ---------------------------------------------------------------------------
# Source iteration helpers
# ---------------------------------------------------------------------------
def source_iteration(I, hx, q, sigma_t, sigma_s, N, BCs, phi=np.zeros(1),
                     Linf_tol=1.0e-5, tolerance=1.0e-8, maxits=100,
                     LOUD=False, plot=False, DSA=False, order=4, fix=0):
    """Full source iteration for a single-group steady-state problem."""
    if phi.size != I * (order + 1):
        phi = np.zeros((I, order + 1))
    phi_old = phi.copy()
    converged = False
    MU, W = _get_quadrature(N)
    psi = np.zeros((I, N, order + 1))
    psi[0, :, :] = phi[0, :] / np.sum(W)
    psi[-1, :, :] = phi[-1, :] / np.sum(W)
    iteration = 1
    if plot:
        plotvar = np.zeros(1)
        plotvar[0] = np.mean(phi_old)
    while not converged:
        phi = np.zeros((I, order + 1))
        for n in range(N):
            if (BCs[n, 0] < 0) and (n >= N // 2):
                tmpBC = psi[-1, n - N // 2, :]
            elif (BCs[n, 0] < 0) and (n < N // 2):
                tmpBC = psi[0, n + N // 2, :]
            else:
                tmpBC = BCs[n, :]
            # vectorised scattering source (replaces per-node loop)
            q_n = q[:, n, :] + phi_old * sigma_s
            tmp_psi = sweep1D_bern(I, hx, np.ascontiguousarray(q_n),
                                   np.array(sigma_t), MU[n], tmpBC,
                                   order=order, fix=fix)
            psi[:, n, :] = tmp_psi.copy()
            phi += tmp_psi * W[n]
        if plot:
            plotvar = np.append(plotvar, np.mean(phi))
        L2err = np.sum((phi_old.reshape(I * (order + 1)) /
                        phi.reshape(I * (order + 1)) - 1)**2 / math.sqrt(I))
        if iteration != 1:
            change = np.append(change, L2err)
        else:
            change = np.zeros(1) + L2err
        Linferr = np.max(np.abs(phi_old / phi - 1))
        converged = ((L2err < tolerance) and (Linferr < Linf_tol)) or (iteration > maxits)
        if (LOUD > 0) or (converged and LOUD < 0):
            print("Iteration", iteration, ": Relative Change =", L2err, Linferr)
        if (iteration > maxits) and (not DSA):
            print("Warning: Source Iteration did not converge")
        iteration += 1
        phi_old = phi.copy()
    x = np.linspace(hx / 2, I * hx - hx / 2, I)
    if DSA:
        return x, phi, psi, iteration - 1
    if plot:
        return x, phi, iteration - 1, plotvar
    return x, phi, iteration - 1, change


def single_source_iteration(I, hx, source, sigma_t, N, BCs,
                            order=4, fix=0):
    """One sweep over all directions, returning the scalar flux phi.

    Uses thread-parallel angle sweeps via numba prange.
    """
    MU, W = _get_quadrature(N)
    M_diag, K = _get_matrices(order, hx)
    source_3d = np.ascontiguousarray(source.reshape((I, N, order + 1)))
    return _sweep_all_angles_phi(I, M_diag, K, source_3d, sigma_t,
                                 MU, W, BCs, order, fix)


def single_source_iteration_psi(I, hx, source, sigma_t, N, BCs,
                                order=4, fix=0):
    """One sweep over all directions, returning the angular flux psi.

    Uses thread-parallel angle sweeps via numba prange.
    """
    MU, W = _get_quadrature(N)
    M_diag, K = _get_matrices(order, hx)
    source_3d = np.ascontiguousarray(source.reshape((I, N, order + 1)))
    return _sweep_all_angles_psi(I, M_diag, K, source_3d, sigma_t,
                                 MU, W, BCs, order, fix)


# ---------------------------------------------------------------------------
# DMD acceleration
# ---------------------------------------------------------------------------
def DMD_prec(matvec, b, K=10, steady=0, x=np.zeros(1),
             step_size=10, GM=0, res=1):
    """DMD preconditioner (batch SVD version)."""
    res = np.min([1.0e-6, res])
    res = np.max([res, 1e-11])
    N = b.size
    if x.size != b.size:
        x = b.copy()
    assert len(b.shape) == 1
    x_new = x * 0
    x_0 = x.copy()
    Yplus = np.zeros((N, K - 1))
    Yminus = np.zeros((N, K - 1))

    change = np.empty(K)
    change_linf = np.empty(K)
    for k in range(K):
        x_new = matvec(x) + b
        L2err = np.sum((x / x_new - 1)**2 / math.sqrt(N))
        Linferr = np.max(np.abs(x / x_new - 1))
        change[k] = L2err
        change_linf[k] = Linferr
        if k < K - 1:
            Yminus[:, k] = x_new - x
            x_0 = x_new.copy()
        if k > 0:
            Yplus[:, k - 1] = x_new - x
        x = x_new.copy()

    # compute SVD
    [u, s, v] = np.linalg.svd(Yminus, full_matrices=False)
    if (x.size > 1) and (s[(1 - np.cumsum(s) / np.sum(s)) > (1.e-3) * res].size >= 1):
        spos = s[(1 - np.cumsum(s) / np.sum(s)) > (1.e-3) * res].copy()
    else:
        spos = s[s > 0].copy()
    mat_size = np.min([K, len(spos)])
    S = np.zeros((mat_size, mat_size))
    unew = 1.0 * u[:, 0:mat_size]
    vnew = 1.0 * v[0:mat_size, :]
    S[np.diag_indices(mat_size)] = 1 / spos

    part1 = unew.conj().T @ Yplus
    part2 = part1 @ vnew.conj().T
    Atilde = part2 @ S.conj().T
    if Atilde.shape[0] > 0:
        try:
            [eigsN, vsN] = np.linalg.eig(Atilde)
            if np.max(np.abs(eigsN)) > 1:
                print("*****Warning*****  The number of steps may be too small")
                eigsN[np.abs(eigsN) > 1] = 0
            eigsN = np.real(eigsN)
            Atilde = np.real(np.dot(np.dot(vsN, np.diag(eigsN)), np.linalg.inv(vsN)))
            if steady:
                Z = np.dot(unew, vsN)
                ZH = np.asarray(Z).conj().T
                Zdagger = np.linalg.solve(ZH @ Z, ZH)
                rhs = unew.conj().T @ Yplus[:, -1]
                delta_y = np.linalg.solve(
                    np.identity(Atilde.shape[0]) - Atilde, rhs)
                x_old = -(Yplus[:, K - 1 - 1] - x)
                steady_update = x_old + np.asarray(np.dot(unew, delta_y)).ravel()
                return steady_update, change, change_linf, Atilde, Yplus, Yminus
            else:
                Z = np.dot(unew, vsN)
                ZH = np.asarray(Z).conj().T
                Zdagger = np.linalg.solve(ZH @ Z, ZH)
                step_1 = (Zdagger @ Yplus[:, -1]).conj()
                step_2 = np.linalg.solve(
                    np.identity(Atilde.shape[0]) - np.diag(eigsN), step_1)
                step_3 = np.dot(
                    np.identity(Atilde.shape[0]) - np.diag(eigsN**step_size), step_2)
                step_4 = np.dot(Z, step_3)
                x_old = -(Yplus[:, K - 1 - 1] - x)
                nonsteady = np.zeros(N)
                nonsteady[0:N] = x_old + np.asarray(step_4).ravel()
                return nonsteady, change
        except Exception as e:
            print("There is an unexpected problem", e)
            return x, change, 0
    else:
        print(spos)


def one_incSVD(u, W, sigma, V, r, k, eps=1e-18, eps_sv=1e-12):
    """Incremental SVD update (rank-one addition)."""
    found_basis = r
    rows = W.shape[0]
    if len(sigma) == 0:
        sigma = np.zeros(1)
    if r == 0:
        while not found_basis:
            sigma_tmp = np.array(np.linalg.norm(u))
            if sigma_tmp > eps:
                found_basis = 1
                sigma[0] = sigma_tmp
                W[:, 0] = u / sigma_tmp
                V = np.ones(1)
            k += 1
        r = r + 1
    else:
        ell = np.dot(W[:, 0:r].transpose(), u)
        p = 0
        if (np.dot(u, u) - np.dot(ell, ell)) > eps:
            p = np.sqrt(np.dot(u, u) - np.dot(ell, ell))
        j = (u - np.dot(W[:, 0:r], ell))
        if p > eps:
            j /= p
        Q = np.zeros((r + 1, r + 1))
        Q[0:r, 0:r] = np.diag(sigma[0:r])
        Q[0:r, r] = ell
        if p > eps:
            Q[r, r] = p
        [Wbar, Sbar, Vbar] = np.linalg.svd(Q, full_matrices=True)
        Vbar = Vbar.transpose()
        if p < eps:
            W = np.dot(W, Wbar[0:r, 0:r])
            sigma = Sbar[0:r]
            Vnew = np.zeros((k + 1, r + 1))
            Vnew[0:k, 0:r] = V.copy()
            Vnew[-1, -1] = 1
            V = np.dot(Vnew, Vbar[:, 0:r])
            k += 1
        else:
            Wnew = np.zeros((W.shape[0], r + 1))
            Wnew[:, 0:r] = W.copy()
            Wnew[:, r] = j
            W = np.dot(Wnew, Wbar)
            sigma = Sbar.copy()
            Vnew = np.zeros((k + 1, r + 1))
            Vnew[0:k, 0:r] = V.copy()
            Vnew[-1, -1] = 1
            V = np.dot(Vnew, Vbar)
            r += 1
            k += 1

        # truncate if small
        if sigma[-1] < eps_sv:
            print("Truncating")
            r -= 1
            V = V[:, 0:r]
            W = W[:, 0:r]
            sigma = sigma[0:r]

    return W, sigma, V, r, k


def DMD_prec_inc(matvec, b, K=10, steady=0, x=np.zeros(1),
                 step_size=10, GM=0, res=1):
    """DMD preconditioner with incremental SVD."""
    res = np.min([1.0e-6, res])
    res = np.max([res, 1e-11])
    N = b.size
    if x.size != b.size:
        x = b.copy()
    assert len(b.shape) == 1
    x_new = x * 0
    x_0 = x.copy()
    Yplus = np.zeros((N, K - 1))
    Yminus = np.zeros((N, K - 1))
    r = 0
    k_val = 0

    update_old = x.copy()
    steady_update = x.copy()
    u = np.zeros((N, 1))
    s = []
    v = [0]
    linf = 0
    change = np.empty(K)
    change_linf = np.empty(K)
    n_filled = 0
    for k in range(K):
        x_new = matvec(x) + b
        L2err = np.sum((x / x_new - 1)**2 / math.sqrt(N))
        Linferr = np.max(np.abs(x / x_new - 1))
        change[k] = L2err
        change_linf[k] = Linferr
        n_filled = k + 1
        if k < K - 1:
            Yminus[:, k] = x_new - x
            x_0 = x_new.copy()
        if k > 0:
            Yplus[:, k - 1] = x_new - x
            [u, s, v, r, k_val] = one_incSVD(
                Yminus[:, k - 1], u, s, v, r=r, k=k_val,
                eps=res * 1e-14, eps_sv=res * 1e-14)
        x = x_new.copy()

        if k > 1:
            vT = (v.T).copy()
            if (x.size > 1) and (s[(1 - np.cumsum(s) / np.sum(s)) >= (1.e-6) * res].size >= 1):
                spos = s[(1 - np.cumsum(s) / np.sum(s)) >= (1.e-6) * res].copy()
            else:
                spos = s[s > 0].copy()
            mat_size = np.min([K, len(spos)])
            S = np.zeros((mat_size, mat_size))
            unew = 1.0 * u[:, 0:mat_size]
            vnew = 1.0 * vT[0:mat_size, 0:k]
            S[np.diag_indices(mat_size)] = 1 / spos

            part1 = unew.conj().T @ Yplus[:, 0:k]
            part2 = part1 @ vnew.conj().T
            Atilde = part2 @ S.conj().T
            if Atilde.shape[0] > 0:
                try:
                    [eigsN, vsN] = np.linalg.eig(Atilde)
                    if np.max(np.abs(eigsN)) > 1:
                        print("*****Warning*****  The number of steps may be too small")
                        eigsN[np.abs(eigsN) > 1] = 0
                    eigsN = np.real(eigsN)
                    Atilde = np.real(np.dot(np.dot(vsN, np.diag(eigsN)),
                                            np.linalg.inv(vsN)))
                    Z = np.dot(unew, vsN)
                    ZH = np.asarray(Z).conj().T
                    Zdagger = np.linalg.solve(ZH @ Z, ZH)
                    rhs = unew.conj().T @ Yplus[:, k - 1]
                    delta_y = np.linalg.solve(
                        np.identity(Atilde.shape[0]) - Atilde, rhs)
                    x_old = -(Yplus[:, k - 1] - x)
                    steady_update = x_old + np.asarray(np.dot(unew, delta_y)).ravel()
                    update_old = steady_update.copy()
                except Exception as e:
                    print("There is an unexpected problem", e)
                    print(spos)
                    return x, change[:n_filled], change_linf[:n_filled], Atilde, Yplus, Yminus
            if k > r + 1:
                return steady_update, change[:n_filled], change_linf[:n_filled], Atilde, Yplus, Yminus

    return steady_update, change[:n_filled], change_linf[:n_filled], Atilde, Yplus, Yminus


# ---------------------------------------------------------------------------
# DMD-accelerated iterative solvers
# ---------------------------------------------------------------------------
def solver_with_dmd(matvec, b, K=10, Rits=2, steady=1, x=np.zeros(1),
                    step_size=10, L2_tol=1e-8, Linf_tol=1e-3,
                    max_its=10, LOUD=0, order=4):
    """Solve via Richardson iteration + DMD acceleration (batch SVD)."""
    N = b.size
    if x.size != b.size:
        x = b.copy()
    assert len(b.shape) == 1
    iteration = 0
    converged = 0
    total_its = 0
    Atil = []
    Yplus = []
    Yminus = []
    change_list = []
    change_linf_list = []
    while (not converged) and (iteration < max_its):
        for r in range(Rits):
            x_new = matvec(x) + b
            L2err = np.sum(((x - x_new) / (x_new + 1e-14))**2 / math.sqrt(N))
            Linferr = np.max(np.abs(x - x_new) / np.max(np.abs(x_new) + 1e-14))
            if (L2err < L2_tol) and (Linferr < Linf_tol):
                converged = 1
            change_list.append(L2err)
            change_linf_list.append(Linferr)
            x = x_new.copy()
            if LOUD:
                print("Iteration:", iteration + 1, " Rich:", r, "Resid=", L2err, Linferr)
            total_its += 1
            if converged:
                break
        if not converged:
            x[0:N], change_dmd, change_dmd_linf, Atilde, Yplus_tmp, Yminus_tmp = \
                DMD_prec(matvec, b, K, steady, x=x, step_size=step_size, res=L2err)
            Atil.append(Atilde)
            Yplus.append(Yplus_tmp)
            Yminus.append(Yminus_tmp)
            if LOUD:
                print("Iteration:", iteration + 1, "DMD completed.")
            change_list.extend(change_dmd.tolist())
            change_linf_list.extend(change_dmd_linf.tolist())
            total_its += K
        iteration += 1
    if LOUD:
        print("Total iterations is", total_its)
    change = np.array(change_list)
    change_linf = np.array(change_linf_list)
    return x, total_its, change, change_linf, Atil, Yplus, Yminus


def solver_with_dmd_inc(matvec, b, K=10, Rits=2, steady=1, x=np.zeros(1),
                        step_size=10, L2_tol=1e-8, Linf_tol=1e-3,
                        max_its=10, LOUD=0, order=4):
    """Solve via Richardson iteration + DMD acceleration (incremental SVD)."""
    N = b.size
    if x.size != b.size:
        x = b.copy()
    assert len(b.shape) == 1
    iteration = 0
    converged = 0
    total_its = 0
    Atil = []
    Yplus = []
    Yminus = []
    change_list = []
    change_linf_list = []
    while (not converged) and (iteration < max_its):
        for r in range(Rits):
            x_new = matvec(x) + b
            L2err = np.sum(((x - x_new) / (x_new + 1e-14))**2 / math.sqrt(N))
            Linferr = np.max(np.abs(x - x_new) / np.max(np.abs(x_new) + 1e-14))
            if (L2err < L2_tol) and (Linferr < Linf_tol):
                converged = 1
            change_list.append(L2err)
            change_linf_list.append(Linferr)
            x = x_new.copy()
            if LOUD:
                print("Iteration:", iteration + 1, " Rich:", r, "Resid=", L2err, Linferr)
            total_its += 1
            if converged:
                break
        if not converged:
            x[0:N], change_dmd, change_dmd_linf, Atilde, Yplus_tmp, Yminus_tmp = \
                DMD_prec_inc(matvec, b, K, steady, x=x,
                             step_size=step_size, res=L2err)
            its_out = change_dmd.size
            if LOUD != 0:
                print("DMD Iterations:", its_out)
            Atil.append(Atilde)
            Yplus.append(Yplus_tmp)
            Yminus.append(Yminus_tmp)
            if LOUD:
                print("Iteration:", iteration + 1, "DMD completed.")
            change_list.extend(change_dmd.tolist())
            change_linf_list.extend(change_dmd_linf.tolist())
            total_its += its_out
        iteration += 1
    if LOUD:
        print("Total iterations is", total_its)
    change = np.array(change_list)
    change_linf = np.array(change_linf_list)
    return x, total_its, change, change_linf, Atil, Yplus, Yminus


# ---------------------------------------------------------------------------
# Time-dependent radiative transfer solvers
# ---------------------------------------------------------------------------
def temp_solve_dmd_inc(I, hx, q, sigma_func, scat_func, N, BCs, EOS, invEOS,
                       phi, psi, T,
                       dt_min=1e-5, dt_max=0.001, tfinal=1.0,
                       Linf_tol=1.0e-5, tolerance=1.0e-8, maxits=100,
                       LOUD=False, order=4, fix=0, K=100, R=3,
                       time_outputs=None):
    """Time-dependent thermal radiative transfer with DMD-accelerated S_N.

    Uses an implicit linearized scheme with adaptive time stepping based
    on the second time-derivative of T, with DMD (incremental SVD version)
    to accelerate the within-step source iterations.

    Parameters
    ----------
    I : int
        Number of spatial zones.
    hx : float
        Zone width.
    q : ndarray (I, N, order+1)
        Fixed external source (usually zero).
    sigma_func : callable(T) -> (I, order+1)
        Absorption opacity as a function of temperature.
    scat_func : callable(T) -> (I, order+1)
        Scattering cross-section as a function of temperature.
    N : int
        Number of discrete ordinates (S_N order).
    BCs : callable(t) -> (N, order+1)
        Boundary conditions as a function of time.
    EOS : callable(T) -> (I, order+1)
        Equation of state: internal energy from temperature.
    invEOS : callable(e) -> (I, order+1)
        Inverse EOS: temperature from internal energy.
    phi : ndarray (I, order+1)
        Initial scalar flux.
    psi : ndarray (I, N, order+1)
        Initial angular flux.
    T : ndarray (I, order+1)
        Initial temperature.
    dt_min, dt_max : float
        Adaptive time step bounds.
    tfinal : float
        Final simulation time.
    Linf_tol, tolerance : float
        Convergence tolerances for source iteration.
    maxits : int
        Maximum number of DMD solver iterations per time step.
    LOUD : bool/int
        Verbosity level.
    order : int
        Bernstein polynomial order.
    fix : int
        Positivity fix-up flag.
    K : int
        Number of DMD inner iterations.
    R : int
        Number of Richardson iterations between DMD steps.
    time_outputs : ndarray or None
        Specific times to include in the output.

    Returns
    -------
    phis : list of ndarray
        Scalar flux at each saved time step.
    Ts : list of ndarray
        Temperature at each saved time step.
    iterations : int
        Total number of transport sweeps.
    ts : ndarray
        Array of time values at each step.
    """
    t_current = 0.0
    phis = []
    Ts = []
    phi_old = phi.copy()
    print(psi.shape)
    psi_old = psi.copy()
    T_old = T.copy()
    T_old2 = T.copy()
    e = EOS(T)
    e_old = e.copy()
    phis.append(phi_old)
    Ts.append(T_old)
    print("|", end='')
    curr_step = 0
    iterations = 0
    hreal = 1e-7
    ts = [t_current]
    delta_step = 1e-3
    step_num = 0
    dt_old = dt_min
    dt_old2 = dt_min
    deriv_val = 0
    dt = dt_min
    t_output_index = 0

    while t_current < tfinal:
        dt_old2 = dt_old
        dt_old = dt
        step_num += 1
        if step_num > 2:
            dt_prop = np.sqrt(delta_step * deriv_val)
            print(dt_prop)
            if dt_prop > dt_max:
                dt_prop = dt_max
            if dt_prop < dt_min:
                dt_prop = dt_min
            if dt_prop > 2 * dt:
                dt_prop = dt * 1.5
            dt = dt_prop
        else:
            dt = dt_min
        # don't step past the endpoint
        if (tfinal - t_current) < dt:
            dt = tfinal - t_current
        # don't step past the next output time
        try:
            if (time_outputs is not None and
                    t_current + dt > time_outputs[t_output_index] and
                    t_output_index < time_outputs.size):
                dt = time_outputs[t_output_index] - t_current
                t_output_index += 1
        except Exception:
            print("Finished the last time")
        if math.isnan(dt):
            dt = dt_min
        print("t = %0.4e, Current dt = %0.4e, old dt = %0.4e" % (t_current, dt, dt_old))
        t_current += dt
        ts.append(t_current)
        if int(10 * t_current / tfinal) > curr_step:
            curr_step += 1
            print(curr_step, end='')
        icdt = 1.0 / (c * dt)
        Cv = (EOS(T + hreal) - EOS(T - hreal)) / (2 * hreal)
        beta_val = 4 * a * T_old**3 / Cv
        sigma = sigma_func(T_old)
        f = 1.0 / (1 + beta_val * c * dt * sigma)
        sigma_a = f * sigma
        sigma_s = (1 - f) * sigma + scat_func(T_old)
        source = q + (sigma_a * ac * T_old**4)[:, None, :] + icdt * psi_old
        sigma_t = sigma + icdt + scat_func(T_old)

        # Pre-compute zero BCs and zero source template for the mv lambda
        zero_BCs = np.zeros((N, order + 1))
        zero_source = np.zeros_like(source)
        _sigma_s = sigma_s   # capture for lambda
        _sigma_t = sigma_t

        mv = lambda phi: single_source_iteration(
            I, hx,
            ((_sigma_s * phi.reshape((I, order + 1)))[:, None, :] + zero_source),
            _sigma_t, N, zero_BCs,
            order=order, fix=1).reshape((order + 1) * I)
        b = single_source_iteration(
            I, hx, source, sigma_t, N, BCs(t_current - dt / 2),
            order=order, fix=1).reshape((order + 1) * I)

        phi, total_its, change, change_linf, Atil, Yp, Ym = \
            solver_with_dmd_inc(
                matvec=mv, b=b, K=K, max_its=maxits, steady=1,
                x=phi.flatten(), Rits=R, LOUD=LOUD, order=order,
                L2_tol=tolerance, Linf_tol=Linf_tol)
        psi = single_source_iteration_psi(
            I, hx,
            ((sigma_s * phi.reshape((I, order + 1)))[:, None, :] + source),
            sigma_t, N, BCs(t_current),
            order=order, fix=1)
        iterations += total_its
        phi = phi.reshape((I, order + 1))
        e = e_old + sigma_a * dt * (phi - ac * T_old**4)
        T = invEOS(e)
        # adaptive time step: second derivative of T in time
        if step_num >= 2:
            deriv_val = np.mean(T) / np.mean(np.abs(
                T / (dt**2) - (dt + dt_old) / (dt**2 * dt_old) * T_old
                + T_old2 / (dt_old * dt)))

        e_old = e.copy()
        T_old2 = T_old.copy()
        T_old = T.copy()
        phi_old = phi.copy()
        psi_old = psi.copy()
        phis.append(phi_old)
        Ts.append(T_old)

    return phis, Ts, iterations, np.array(ts)
