"""
Example: 2-Group Marshak Wave with Multigroup IMC — x vs y symmetry check

Runs the same Marshak wave problem twice:
  Run A — wave propagates in x: 50 cells in x, 5 transverse cells in y.
  Run B — wave propagates in y: 5 transverse cells in x, 50 cells in y.

Both runs use reflecting boundaries on the transverse faces, so the physics is
identical.  The temperature profiles (averaged over the transverse direction)
must agree within Monte Carlo noise.  This is a geometric-symmetry verification
test for the 2-D multigroup IMC transport.
"""

import numpy as np
import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, parent_dir)

from MG_IMC import (
    run_simulation,
    C_LIGHT as _C_LIGHT,
    A_RAD as _A_RAD,
)

# ── Problem parameters ────────────────────────────────────────────────────────
T_hot   = 1.0    # keV  (driving boundary temperature)
T_init  = 0.01   # keV  (uniform initial temperature)
final_time = 1.0  # ns
dt         = 0.01  # ns

# Spatial domain: 1 cm along the wave direction, 1 cm transverse
L_wave      = 0.2   # cm  (wave propagation extent)
L_transverse = 1.0  # cm  (transverse width)
n_wave       = 50   # cells along wave direction
n_trans      = 5    # cells in transverse direction

# Energy groups
energy_edges = np.array([0.1, 1.0, 10.0])   # keV
n_groups = len(energy_edges) - 1

# Material / EOS
rho = 1.0   # g/cm³
cv0 = 0.3   # GJ/(g·keV)

def cv_func(T):
    return cv0 * np.ones_like(T)

def eos(T):
    return rho * cv0 * T

def inv_eos(e):
    return e / (rho * cv0)

# Opacities: power-law σ_g(T) = σ₀ T^{-3}
sigma_0_vals = [300.0, 300.0]

def sigma_a_group_0(T):
    return sigma_0_vals[0] * T**(-3)

def sigma_a_group_1(T):
    return sigma_0_vals[1] * T**(-3)

sigma_a_funcs = [sigma_a_group_0, sigma_a_group_1]

# ── Self-similar solution (gray equilibrium diffusion, n=3 opacity) ─────────
# Both groups share the same σ₀ T^{-3} opacity, so the total gray opacity is
# identical and the equilibrium gray self-similar solution applies.
def self_similar_T(r, t):
    """Material temperature for the Marshak wave self-similar solution.

    Valid for σ = σ₀ T^{-n} with n=3, equilibrium radiation diffusion.
    xi_max and omega are the n=3 eigenvalues (Pomraning / Larsen tabulation).
    """
    _sigma0  = sigma_0_vals[0]          # same for both groups
    _cv_vol  = rho * cv0                # volumetric heat capacity
    n        = 3
    xi_max   = 1.11305
    omega    = 0.05989
    K        = 8 * _A_RAD * _C_LIGHT / ((n + 4) * 3 * _sigma0 * _cv_vol)
    xi       = np.asarray(r, dtype=float) / np.sqrt(K * t)
    T        = np.zeros_like(xi)
    m        = xi < xi_max
    T[m]     = np.power((1 - xi[m] / xi_max) * (1 + omega * xi[m] / xi_max),
                        1.0 / 6)
    return T


# Particle counts
Ntarget    = 10000
Nboundary  = 5000
Nsource    = 0
Nmax       = 50000
Ntarget_ic = 10000

# Emission floor: skip cells that are still cold (20% above initial T)
# This avoids wasting particles on negligible emission in the unheated region.
T_emit_floor = 1.2 * T_init   # keV  (= 0.012 keV for T_init=0.01)

# IMC parameters
theta                  = 1.0
use_scalar_intensity_Tr = True
conserve_comb_energy    = True

# ── Common run helper ─────────────────────────────────────────────────────────
def _run(label, edges1, edges2, T_boundary_vec, reflect_vec):
    nx_loc = len(edges1) - 1
    ny_loc = len(edges2) - 1
    Tinit  = np.full((nx_loc, ny_loc), T_init)
    source = np.zeros((nx_loc, ny_loc))
    print(f"\n{'='*70}")
    print(f"Run {label}: {nx_loc}×{ny_loc} grid")
    print(f"  T_boundary = {T_boundary_vec}  reflect = {reflect_vec}")
    print('='*70)
    history, state = run_simulation(
        Ntarget=Ntarget,
        Nboundary=Nboundary,
        Nsource=Nsource,
        Nmax=Nmax,
        Tinit=Tinit,
        Tr_init=Tinit.copy(),
        T_boundary=T_boundary_vec,
        dt=dt,
        edges1=edges1,
        edges2=edges2,
        energy_edges=energy_edges,
        sigma_a_funcs=sigma_a_funcs,
        eos=eos,
        inv_eos=inv_eos,
        cv=cv_func,
        source=source,
        final_time=final_time,
        reflect=reflect_vec,
        output_freq=10,
        theta=theta,
        use_scalar_intensity_Tr=use_scalar_intensity_Tr,
        Ntarget_ic=Ntarget_ic,
        conserve_comb_energy=conserve_comb_energy,
        geometry="xy",
        T_emit_floor=T_emit_floor,
    )
    print(f"  Done: t={state.time:.4f} ns, {len(state.weights)} census particles")
    return state

# ── Run A: wave in x ──────────────────────────────────────────────────────────
# 50 cells in x (wave direction), 5 cells in y (transverse)
# Hot left boundary; reflect top and bottom.
x_edges_A = np.linspace(0.0, L_wave,       n_wave  + 1)
y_edges_A = np.linspace(0.0, L_transverse, n_trans + 1)
state_A = _run(
    "A (wave in x)",
    edges1=x_edges_A,
    edges2=y_edges_A,
    T_boundary_vec=(T_hot, 0.0, 0.0, 0.0),   # left=hot, right/bottom/top=vacuum
    reflect_vec=(False, False, True, True),    # reflect top and bottom
)
# Profile: average over the 5 transverse y-cells
x_centers = 0.5 * (x_edges_A[:-1] + x_edges_A[1:])
T_A   = state_A.temperature.mean(axis=1)      # shape (n_wave,)
Tr_A  = state_A.radiation_temperature.mean(axis=1)
Er_A  = [state_A.radiation_energy_by_group[g].mean(axis=1) for g in range(n_groups)]

# ── Run B: wave in y ──────────────────────────────────────────────────────────
# 5 cells in x (transverse), 50 cells in y (wave direction)
# Hot bottom boundary; reflect left and right.
x_edges_B = np.linspace(0.0, L_transverse, n_trans + 1)
y_edges_B = np.linspace(0.0, L_wave,       n_wave  + 1)
state_B = _run(
    "B (wave in y)",
    edges1=x_edges_B,
    edges2=y_edges_B,
    T_boundary_vec=(0.0, 0.0, T_hot, 0.0),   # bottom=hot, left/right/top=vacuum
    reflect_vec=(True, True, False, False),    # reflect left and right
)
# Profile: average over the 5 transverse x-cells
y_centers = 0.5 * (y_edges_B[:-1] + y_edges_B[1:])
T_B   = state_B.temperature.mean(axis=0)      # shape (n_wave,)
Tr_B  = state_B.radiation_temperature.mean(axis=0)
Er_B  = [state_B.radiation_energy_by_group[g].mean(axis=0) for g in range(n_groups)]

# ── Statistical comparison ────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("Symmetry check: run A (x-wave) vs run B (y-wave)")
print(f"{'='*70}")
print(f"{'Position':>10}  {'T_mat A':>10}  {'T_mat B':>10}  {'|diff|/A':>10}")
print(f"{'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
for i in range(0, n_wave, 5):
    rel = abs(T_A[i] - T_B[i]) / max(T_A[i], 1e-12)
    print(f"{x_centers[i]:>10.3f}  {T_A[i]:>10.4f}  {T_B[i]:>10.4f}  {rel:>10.4f}")

# Self-similar reference at final time
T_ss = self_similar_T(x_centers, final_time)

rms_rel = np.sqrt(np.mean(((T_A - T_B) / np.maximum(T_A, 1e-12))**2))
print(f"\nRMS relative difference in T_mat: {rms_rel:.4f}")
# A rough MC noise estimate: ~1/sqrt(Ntarget) per cell
mc_noise = 1.0 / np.sqrt(Ntarget)
print(f"Expected MC noise level (~1/√N):  {mc_noise:.4f}")
if rms_rel < 5 * mc_noise:
    print("PASS: profiles agree within 5× MC noise level.")
else:
    print("WARN: profiles differ by more than 5× MC noise level.")

# ── Plot ──────────────────────────────────────────────────────────────────────
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        f"2-Group Marshak Wave: x vs y symmetry check  (t = {final_time} ns)",
        fontsize=13,
    )

    # Top-left: material temperature
    ax = axes[0, 0]
    ax.plot(x_centers, T_ss, 'k-',  lw=2, label='Self-similar (eq. diffusion)')
    ax.plot(x_centers, T_A,  'b-',  lw=1.5, alpha=0.8, label='Run A (x-wave)')
    ax.plot(y_centers, T_B,  'r--', lw=1.5, alpha=0.8, label='Run B (y-wave)')
    ax.set_xlabel('Position along wave direction (cm)')
    ax.set_ylabel('Material temperature (keV)')
    ax.set_title('Material Temperature')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Top-right: radiation temperature
    ax = axes[0, 1]
    ax.plot(x_centers, Tr_A, 'b-',  lw=2, label='Run A (x-wave)')
    ax.plot(y_centers, Tr_B, 'r--', lw=2, label='Run B (y-wave)')
    ax.set_xlabel('Position along wave direction (cm)')
    ax.set_ylabel('Radiation temperature (keV)')
    ax.set_title('Radiation Temperature')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom panels: radiation energy per group
    for g in range(n_groups):
        ax = axes[1, g]
        ax.semilogy(x_centers, Er_A[g], 'b-',  lw=2, label='Run A (x-wave)')
        ax.semilogy(y_centers, Er_B[g], 'r--', lw=2, label='Run B (y-wave)')
        ax.set_xlabel('Position along wave direction (cm)')
        ax.set_ylabel('Radiation energy density (GJ/cm³)')
        ax.set_title(
            f'Group {g}  [{energy_edges[g]:.1f}–{energy_edges[g+1]:.1f} keV]'
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), 'test_2group_marshak.png')
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")

except ImportError:
    print("Matplotlib not available, skipping plot.")

print("\nTest complete!")
