#!/usr/bin/env python3
"""
2-cell (minimal spatial) LMFGK eigenvalue diagnostic for the kappa (absorption-rate) system.

Uses your Planck integral helpers correctly:
  Bg_multigroup(energy_edges, T)
  dBgdT_multigroup(energy_edges, T)

Run:
  python lmfgk_2cell_kappa_eigs.py
"""

import numpy as np
import matplotlib.pyplot as plt

from planck_integrals import Bg_multigroup, dBgdT_multigroup


def two_cell_periodic_K(dx: float) -> np.ndarray:
    """2-cell periodic diffusion stiffness for -div(D grad)."""
    k = 2.0 / (dx * dx)
    return k * np.array([[1.0, -1.0],
                         [-1.0, 1.0]], dtype=float)


def build_lambda_tilde(sigma: np.ndarray, chi: np.ndarray) -> np.ndarray:
    """
    LMFG gray weights commonly used in MFGA/LMFG implementations:
      lambda_tilde_g ∝ chi_g / sigma_g
    normalized so that sum_g sigma_g * lambda_tilde_g = 1.
    """
    raw = chi / sigma
    denom = np.sum(sigma * raw)  # = sum chi if chi sums to 1
    return raw / denom


def main():
    # ----------------------------
    # Parameters
    # ----------------------------
    G = 60
    T0 = 1.0           # keV
    rho = 1.0
    cv = 0.01
    c_light = 29.9792458
    dt = 10

    # minimal spatial setup: 2 cells, periodic coupling
    L = 5.0e-3
    n_cells = 2
    dx = L / n_cells

    energy_edges = np.logspace(-4, 1, G + 1)

    # geometric opacities (very stiff)
    sigma_min = 1e-6
    sigma_max = 1e6
    sigma = sigma_min * (sigma_max / sigma_min) ** (np.linspace(0.0, 1.0, G))

    # simple diffusion: D = 1/(3 sigma)
    D = 1.0 / (3.0 * sigma)

    # uniform emission fractions
    chi = np.ones(G) / G

    # ----------------------------
    # Fleck factor (scalar here)
    # ----------------------------
    dBgdT = dBgdT_multigroup(energy_edges, T0)
    beta = (c_light * dt / (rho * cv)) * float(np.sum(sigma * dBgdT))
    f = 1.0 / (1.0 + beta)
    nu = 1.0 - f

    print("2-cell LMFGK kappa-spectrum diagnostic")
    print(f"G={G}, dx={dx:.6g}, dt={dt}, T0={T0}")
    print(f"beta={beta:.6e}, f={f:.6e}, nu=1-f={nu:.6e}")

    # diffusion stiffness
    K = two_cell_periodic_K(dx)
    I2 = np.eye(2)
    inv_c_dt = 1.0 / (c_light * dt)

    # ----------------------------
    # Build B = I - nu * sum_g chi_g * sigma_g * A_g^{-1}
    # with A_g = (nu*sigma_g + 1/(c dt)) I + D_g K
    # ----------------------------
    S = np.zeros((2, 2), dtype=float)
    for g in range(G):
        Ag = (nu * sigma[g] + inv_c_dt) * I2 + D[g] * K
        Ag_inv = np.linalg.inv(Ag)
        S += chi[g] * sigma[g] * Ag_inv

    B = I2 - nu * S

    # ----------------------------
    # LMFG gray operator H and left multiplier C
    # ----------------------------
    lam_tilde = build_lambda_tilde(sigma, chi)
    ssum = float(np.sum(sigma * lam_tilde))
    print(f"sum_g sigma_g * lambda_tilde_g = {ssum:.16e} (should be 1)")

    sigma_gray = float(np.sum(sigma * lam_tilde))
    D_gray = float(np.sum(D * lam_tilde))
    print(f"sigma_gray = {sigma_gray:.6e}, D_gray = {D_gray:.6e}")

    H = (nu * sigma_gray + inv_c_dt) * I2 + D_gray * K
    H_inv = np.linalg.inv(H)

    C = I2 + sigma_gray * (nu * H_inv)
    CB = C @ B
    v_const = np.array([1.0, 1.0]);  v_const /= np.linalg.norm(v_const)
    v_alt   = np.array([1.0,-1.0]);  v_alt   /= np.linalg.norm(v_alt)

    def rayleigh(M, v):
        return float(v @ (M @ v))

    print("\nMode Rayleigh quotients (should match eigenvalues):")
    print("B  const:", rayleigh(B, v_const), " alt:", rayleigh(B, v_alt))
    print("CB const:", rayleigh(CB, v_const), " alt:", rayleigh(CB, v_alt))
    # eigen / cond
    eigB = np.linalg.eigvals(B)
    eigCB = np.linalg.eigvals(CB)
    condB = np.linalg.cond(B)
    condCB = np.linalg.cond(CB)

    print("\nB =\n", B)
    print("\nC =\n", C)
    print("\nCB =\n", CB)

    print("\nEigenvalues:")
    print("eig(B): ", eigB)
    print("eig(CB):", eigCB)

    print("\nCondition numbers (2-norm):")
    print(f"cond(B)  = {condB:.6e}")
    print(f"cond(CB) = {condCB:.6e}")

    # plot
    plt.figure(figsize=(7, 5))
    plt.scatter(eigB.real, eigB.imag, label="B (unpreconditioned)", marker="o")
    plt.scatter(eigCB.real, eigCB.imag, label="C B (LMFGK)", marker="x")
    plt.axvline(0.0, linewidth=0.5)
    plt.xlabel("Real")
    plt.ylabel("Imag")
    plt.title("2-cell kappa operator eigenvalues: B vs C B")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
