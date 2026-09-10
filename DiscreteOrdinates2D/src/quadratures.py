"""
Angular quadrature sets for multidimensional discrete ordinates (S_N).

Implements three families described in the textbook (Chapter 11):

1. Product quadratures (square and triangular)
   - Gauss-Legendre in polar × doubled Chebyshev-Gauss in azimuth

2. Level-symmetric (fully symmetric) quadratures (S_2 through S_20)

3. Equal-weight (EQ_N) quadratures

All quadratures return arrays of shape (M, 3) for directions (Omega_x,
Omega_y, Omega_z) and (M,) for weights, normalized so that sum(w) = 1.

For 2-D x-y problems, only the upper hemisphere (Omega_z > 0) is needed
(or equivalently, the z-symmetry is exploited), and the weights are doubled.
A helper function `get_2d_quadrature` returns the reduced set.
"""

import numpy as np
from functools import lru_cache


# ===========================================================================
# Product Quadratures (Section 11.1.1)
# ===========================================================================

def product_quadrature_square(Np):
    """Square product quadrature with Np polar × Np azimuthal points.

    Total directions: 2 * Np^2 (full sphere).

    Parameters
    ----------
    Np : int
        Number of Gauss-Legendre points in polar direction and number of
        Chebyshev-Gauss points per half-azimuth.

    Returns
    -------
    omegas : (N_total, 3) float64
        Direction cosines (Omega_x, Omega_y, Omega_z).
    weights : (N_total,) float64
        Normalized weights (sum = 1).
    """
    # Polar: Gauss-Legendre on [-1, 1]
    mu_nodes, w_polar = np.polynomial.legendre.leggauss(Np)
    w_polar = w_polar / np.sum(w_polar)  # normalize to 1

    # Azimuthal: doubled Chebyshev-Gauss
    Na = Np
    theta_j = np.array([(2*j - 1) * np.pi / (2*Na) for j in range(1, Na+1)])
    # phi_plus = theta_j, phi_minus = 2*pi - theta_j
    w_az = 1.0 / (2.0 * Na)  # each azimuthal weight

    directions = []
    weights = []

    for i in range(Np):
        mu_i = mu_nodes[i]
        sin_theta = np.sqrt(1.0 - mu_i**2)
        for j in range(Na):
            for sigma in [+1, -1]:  # +: phi_j^+, -: phi_j^-
                if sigma == +1:
                    phi = theta_j[j]
                else:
                    phi = 2*np.pi - theta_j[j]
                Ox = sin_theta * np.cos(phi)
                Oy = sin_theta * np.sin(phi)
                Oz = mu_i
                directions.append([Ox, Oy, Oz])
                weights.append(w_polar[i] * w_az)

    return np.array(directions), np.array(weights)


def product_quadrature_triangular(Np):
    """Triangular product quadrature with Np polar levels.

    The number of azimuthal points varies by polar level:
    Na_i = Np at the equator, decreasing by 2 each level toward the pole.
    Total directions: Np * (Np + 2) (full sphere).

    Parameters
    ----------
    Np : int
        Number of Gauss-Legendre points in polar direction.

    Returns
    -------
    omegas : (N_total, 3) float64
    weights : (N_total,) float64
    """
    # Polar: Gauss-Legendre on [-1, 1]
    mu_nodes, w_polar_raw = np.polynomial.legendre.leggauss(Np)
    w_polar_raw = w_polar_raw / np.sum(w_polar_raw)

    # Na_i varies by *unique* |mu| level, not by sequential position.
    # Levels that share the same |mu| (i.e. +mu_i and -mu_i) get the same Na.
    # Rank 0 = smallest |mu| → Na = Np; rank increases toward the pole → Na decreases by 2.
    abs_mu = np.abs(mu_nodes)
    unique_abs_mu = np.unique(abs_mu)          # sorted ascending by numpy

    directions = []
    weights = []

    for i in range(Np):
        mu_i = mu_nodes[i]
        sin_theta = np.sqrt(max(0.0, 1.0 - mu_i**2))

        # Rank of this level's |mu| among unique values (0 = closest to equator)
        rank = int(np.searchsorted(unique_abs_mu, abs_mu[i]))
        Na_i = max(2, Np - 2 * rank)

        theta_j = np.array([(2*j - 1) * np.pi / (2*Na_i)
                            for j in range(1, Na_i+1)])
        w_az = 1.0 / (2.0 * Na_i)

        for j in range(Na_i):
            for sigma in [+1, -1]:
                if sigma == +1:
                    phi = theta_j[j]
                else:
                    phi = 2*np.pi - theta_j[j]
                Ox = sin_theta * np.cos(phi)
                Oy = sin_theta * np.sin(phi)
                Oz = mu_i
                directions.append([Ox, Oy, Oz])
                weights.append(w_polar_raw[i] * w_az)

    omegas = np.array(directions)
    weights = np.array(weights)
    weights /= np.sum(weights)  # renormalize
    return omegas, weights


# ===========================================================================
# Level-Symmetric Quadratures (Section 11.1.2)
# ===========================================================================

# Tabulated values from Table 11.3
_LS_DATA = {
    2: {
        'mu1': 0.577350269,
        'mu_vals': [0.577350269],
        'octant_points': [(1, 1)],  # (i, j) indices into mu_vals (1-based)
        'weights': [1.0],
    },
    4: {
        'mu1': 0.350021174,
        'mu_vals': [0.350021174, 0.868890300],
        'octant_points': [(1, 1), (1, 2), (2, 1)],
        'weights': [1.0/3, 1.0/3, 1.0/3],
    },
    6: {
        'mu1': 0.266635401,
        'mu_vals': [0.266635401, 0.681507726, 0.926180936],
        'octant_points': [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (3, 1)],
        'weights': [0.176126, 0.157207, 0.176126, 0.157207, 0.157207, 0.176126],
    },
    8: {
        'mu1': 0.218217890,
        'mu_vals': [0.218217890, 0.577350269, 0.786795792, 0.951189731],
        'octant_points': [
            (1, 1), (1, 2), (1, 3), (1, 4),
            (2, 1), (2, 2), (2, 3),
            (3, 1), (3, 2),
            (4, 1),
        ],
        'weights': [
            0.120987654, 0.090740741, 0.090740741, 0.120987654,
            0.090740741, 0.092592593, 0.090740741,
            0.090740741, 0.090740741,
            0.120987654,
        ],
    },
}


def level_symmetric_quadrature(N):
    """Level-symmetric (fully symmetric) S_N quadrature.

    Parameters
    ----------
    N : int (even, 2 to 20)
        Quadrature order. N(N+2)/8 points per octant, N(N+2) total.

    Returns
    -------
    omegas : (N_total, 3) float64
    weights : (N_total,) float64
    """
    if N in _LS_DATA:
        data = _LS_DATA[N]
        mu_vals = np.array(data['mu_vals'])
        octant_pts = data['octant_points']
        octant_wts = np.array(data['weights'])
    else:
        #IDon't trust this right now, so we'll just raise an error for N > 8. The code below is a placeholder for future implementation.
        raise ValueError(f"Level-symmetric quadrature for N={N} is not implemented. Please use N=2,4,6,8.")
        # Compute from formula (11.20)
        m = N // 2
        # For N > 8, use the standard formula with mu1 chosen to satisfy
        # the even-moment conditions. We use a simple optimization.
        mu1_sq = _solve_mu1_ls(N)
        mu_vals = np.zeros(m)
        Delta = (1.0 - 3.0 * mu1_sq) / (m - 1)
        for i in range(m):
            mu_vals[i] = np.sqrt(mu1_sq + i * Delta)

        # Generate admissible triples
        octant_pts = []
        for i in range(1, m+1):
            for j in range(1, m+1):
                k = m + 2 - i - j
                if 1 <= k <= m and k <= j:  # avoid duplicates
                    octant_pts.append((i, j, k))
                    if j != k:
                        octant_pts.append((i, k, j))

        # Remove duplicates via sorting
        unique_pts = []
        seen = set()
        for pt in octant_pts:
            key = tuple(sorted(pt, reverse=True))
            if key not in seen:
                seen.add(key)
                unique_pts.append(pt)
        octant_pts = unique_pts

        # For higher orders, assign equal weights (approximation)
        n_oct = len(octant_pts)
        octant_wts = np.ones(n_oct) / n_oct

    # Expand octant points to full sphere
    directions = []
    weights_full = []

    for idx, pt in enumerate(octant_pts):
        if len(pt) == 2:
            i, j = pt
            mu_x = mu_vals[i-1]
            mu_y = mu_vals[j-1]
            mu_z_sq = 1.0 - mu_x**2 - mu_y**2
            if mu_z_sq < 0:
                mu_z_sq = 0.0
            mu_z = np.sqrt(mu_z_sq)
        else:
            i, j, k = pt
            mu_x = mu_vals[i-1]
            mu_y = mu_vals[j-1]
            mu_z = mu_vals[k-1]

        w = octant_wts[idx]

        # Reflect into all 8 octants
        for sx in [+1, -1]:
            for sy in [+1, -1]:
                for sz in [+1, -1]:
                    directions.append([sx*mu_x, sy*mu_y, sz*mu_z])
                    weights_full.append(w)

    omegas = np.array(directions)
    weights_full = np.array(weights_full)
    # Normalize: octant weights sum to 1, so 8 octants sum to 8
    weights_full = weights_full / np.sum(weights_full)
    return omegas, weights_full


def _solve_mu1_ls(N):
    """Find mu1^2 for level-symmetric quadrature by solving moment conditions."""
    from scipy.optimize import minimize_scalar
    m = N // 2

    def objective(mu1_sq):
        if mu1_sq <= 0 or mu1_sq >= 1.0/3.0:
            return 1e10
        Delta = (1.0 - 3.0 * mu1_sq) / (m - 1)
        mu_sq = np.array([mu1_sq + i * Delta for i in range(m)])
        if np.any(mu_sq <= 0):
            return 1e10
        mu_vals = np.sqrt(mu_sq)

        # Generate octant points and equal weights
        octant_pts = []
        for i in range(1, m+1):
            for j in range(1, m+1):
                k = m + 2 - i - j
                if 1 <= k <= m:
                    octant_pts.append((i, j, k))
        n_oct = len(octant_pts)
        w = 1.0 / n_oct

        # Check moment condition: sum w * mu_x^4 = 1/5
        moment4 = 0.0
        for pt in octant_pts:
            i, j, k = pt
            moment4 += w * mu_vals[i-1]**4
        err = (moment4 - 1.0/5.0)**2
        return err

    result = minimize_scalar(objective, bounds=(0.01, 0.33), method='bounded')
    return result.x


# ===========================================================================
# Equal-Weight (EQ_N) Quadratures (Section 11.1.3)
# ===========================================================================

# Tabulated from Tables 11.4 and 11.5
_EQ_DATA = {
    4: np.array([
        [0.3500212, 0.3500212, 0.8688903],
        [0.3500212, 0.8688903, 0.3500212],
        [0.8688903, 0.3500212, 0.3500212],
    ]),
    6: np.array([
        [0.2561428, 0.2561428, 0.9320846],
        [0.2561428, 0.9320846, 0.2561428],
        [0.9320846, 0.2561428, 0.2561428],
        [0.2663445, 0.6815646, 0.6815646],
        [0.6815646, 0.2663445, 0.6815646],
        [0.6815646, 0.6815646, 0.2663445],
    ]),
    8: np.array([
        [0.1971380, 0.1971380, 0.9603506],
        [0.1971380, 0.9603506, 0.1971380],
        [0.9603506, 0.1971380, 0.1971380],
        [0.2133981, 0.5512958, 0.8065570],
        [0.2133981, 0.8065570, 0.5512958],
        [0.5512958, 0.2133981, 0.8065570],
        [0.5512958, 0.8065570, 0.2133981],
        [0.8065570, 0.2133981, 0.5512958],
        [0.8065570, 0.5512958, 0.2133981],
        [0.5773503, 0.5773503, 0.5773503],
    ]),
    10: np.array([
        [0.1631408, 0.1631408, 0.9730212],
        [0.1631408, 0.9730212, 0.1631408],
        [0.9730212, 0.1631408, 0.1631408],
        [0.1755273, 0.4567576, 0.8721024],
        [0.1755273, 0.8721024, 0.4567576],
        [0.4567576, 0.1755273, 0.8721024],
        [0.4567576, 0.8721024, 0.1755273],
        [0.8721024, 0.1755273, 0.4567576],
        [0.8721024, 0.4567576, 0.1755273],
        [0.1755273, 0.6961286, 0.6961286],
        [0.6961286, 0.1755273, 0.6961286],
        [0.6961286, 0.6961286, 0.1755273],
        [0.4897749, 0.4897749, 0.7212773],
        [0.4897749, 0.7212773, 0.4897749],
        [0.7212773, 0.4897749, 0.4897749],
    ]),
    12: np.array([
        [0.1370611, 0.1370611, 0.9810344],
        [0.1370611, 0.9810344, 0.1370611],
        [0.9810344, 0.1370611, 0.1370611],
        [0.1497456, 0.3911744, 0.9080522],
        [0.1497456, 0.9080522, 0.3911744],
        [0.3911744, 0.1497456, 0.9080522],
        [0.3911744, 0.9080522, 0.1497456],
        [0.9080522, 0.1497456, 0.3911744],
        [0.9080522, 0.3911744, 0.1497456],
        [0.1497456, 0.6040252, 0.7827706],
        [0.1497456, 0.7827706, 0.6040252],
        [0.6040252, 0.1497456, 0.7827706],
        [0.6040252, 0.7827706, 0.1497456],
        [0.7827706, 0.1497456, 0.6040252],
        [0.7827706, 0.6040252, 0.1497456],
        [0.4213515, 0.4213515, 0.8030727],
        [0.4213515, 0.8030727, 0.4213515],
        [0.8030727, 0.4213515, 0.4213515],
        [0.4249785, 0.6400755, 0.6400755],
        [0.6400755, 0.4249785, 0.6400755],
        [0.6400755, 0.6400755, 0.4249785],
    ]),
    14: np.array([
        [0.1196230, 0.1196230, 0.9855865],
        [0.1196230, 0.9855865, 0.1196230],
        [0.9855865, 0.1196230, 0.1196230],
        [0.1301514, 0.3399241, 0.9314034],
        [0.1301514, 0.9314034, 0.3399241],
        [0.3399241, 0.1301514, 0.9314034],
        [0.3399241, 0.9314034, 0.1301514],
        [0.9314034, 0.1301514, 0.3399241],
        [0.9314034, 0.3399241, 0.1301514],
        [0.1301514, 0.5326235, 0.8362851],
        [0.1301514, 0.8362851, 0.5326235],
        [0.5326235, 0.1301514, 0.8362851],
        [0.5326235, 0.8362851, 0.1301514],
        [0.8362851, 0.1301514, 0.5326235],
        [0.8362851, 0.5326235, 0.1301514],
        [0.1301514, 0.7010922, 0.7010922],
        [0.7010922, 0.1301514, 0.7010922],
        [0.7010922, 0.7010922, 0.1301514],
        [0.3700438, 0.3700438, 0.8521356],
        [0.3700438, 0.8521356, 0.3700438],
        [0.8521356, 0.3700438, 0.3700438],
        [0.3736206, 0.5691722, 0.7324279],
        [0.3736206, 0.7324279, 0.5691722],
        [0.5691722, 0.3736206, 0.7324279],
        [0.5691722, 0.7324279, 0.3736206],
        [0.7324279, 0.3736206, 0.5691722],
        [0.7324279, 0.5691722, 0.3736206],
        [0.5773503, 0.5773503, 0.5773503],
    ]),
}


def equal_weight_quadrature(N):
    """Equal-weight (EQ_N) quadrature.

    Parameters
    ----------
    N : int (even, 4 to 14)
        Quadrature order.

    Returns
    -------
    omegas : (N_total, 3) float64
    weights : (N_total,) float64
    """
    if N not in _EQ_DATA:
        raise ValueError(f"EQ_{N} not tabulated. Available: {sorted(_EQ_DATA.keys())}")

    octant_pts = _EQ_DATA[N]
    Q_N = octant_pts.shape[0]  # points per octant

    # Equal weight per octant point
    w_oct = 1.0 / Q_N

    directions = []
    weights_full = []

    for m in range(Q_N):
        mu, eta, xi = octant_pts[m]
        # Reflect into all 8 octants
        for sx in [+1, -1]:
            for sy in [+1, -1]:
                for sz in [+1, -1]:
                    directions.append([sx*mu, sy*eta, sz*xi])
                    weights_full.append(w_oct)

    omegas = np.array(directions)
    weights_full = np.array(weights_full)
    weights_full /= np.sum(weights_full)
    return omegas, weights_full


# ===========================================================================
# 2-D Quadrature Helper
# ===========================================================================

def get_2d_quadrature(quad_type='level_symmetric', N=8):
    """Get angular quadrature for 2-D x-y problems.

    For 2-D problems independent of z, we exploit symmetry under
    Omega_z -> -Omega_z and only keep Omega_y > 0 directions, with
    weights doubled.

    Parameters
    ----------
    quad_type : str
        One of 'product_square', 'product_triangular',
        'level_symmetric', 'equal_weight'.
    N : int
        Quadrature order parameter.

    Returns
    -------
    Omega_x : (M,) float64
    Omega_y : (M,) float64
    weights : (M,) float64
        Normalized so sum(weights) = 1.
    """
    if quad_type == 'product_square':
        omegas, weights = product_quadrature_square(N)
    elif quad_type == 'product_triangular':
        omegas, weights = product_quadrature_triangular(N)
    elif quad_type == 'level_symmetric':
        omegas, weights = level_symmetric_quadrature(N)
    elif quad_type == 'equal_weight':
        omegas, weights = equal_weight_quadrature(N)
    else:
        raise ValueError(f"Unknown quadrature type: {quad_type}")

    # For 2-D x-y: exploit Omega_z symmetry
    # Keep only Omega_y > 0 (using y as the "extra" dimension that
    # the solution doesn't depend on) — actually for 2-D x-y problems,
    # we use Omega_x and Omega_y directly (solution varies in x and y).
    # The z-direction has the symmetry. Keep Omega_z > 0, double weights.
    mask = omegas[:, 2] > 0  # Omega_z > 0
    Omega_x = omegas[mask, 0]
    Omega_y = omegas[mask, 1]
    w = weights[mask] * 2.0  # double weights for z-symmetry

    # Renormalize
    w = w / np.sum(w)

    return Omega_x, Omega_y, w


def get_full_quadrature(quad_type='level_symmetric', N=8):
    """Get full-sphere angular quadrature (all directions).

    Parameters
    ----------
    quad_type : str
    N : int

    Returns
    -------
    Omega_x : (M,) float64
    Omega_y : (M,) float64
    Omega_z : (M,) float64
    weights : (M,) float64
    """
    if quad_type == 'product_square':
        omegas, weights = product_quadrature_square(N)
    elif quad_type == 'product_triangular':
        omegas, weights = product_quadrature_triangular(N)
    elif quad_type == 'level_symmetric':
        omegas, weights = level_symmetric_quadrature(N)
    elif quad_type == 'equal_weight':
        omegas, weights = equal_weight_quadrature(N)
    else:
        raise ValueError(f"Unknown quadrature type: {quad_type}")

    return omegas[:, 0], omegas[:, 1], omegas[:, 2], weights
