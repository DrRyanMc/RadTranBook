"""test_mg_imc1d_spherical.py — Verify MG_IMC1D.py with a gray-equivalent test.

TEST DESCRIPTION
----------------
Run a simple spherical IMC problem with a constant opacity σ_a = σ_0 for ALL
groups.  Because the opacity is group-independent, the multigroup simulation
must produce results that are statistically equivalent to the gray simulation
from IMC1D.py using the same σ_a.

Geometry:   Sphere of radius R = 10 cm, I = 20 radial zones
Initial conditions:
    T_mat = 1.0 keV,  T_rad = 0.1 keV  (out of equilibrium)
Opacity:    σ_a = σ_0 = 10 cm⁻¹  (constant, independent of T and group)
Heat cap:   cv = 1.0  GJ / cm³ / keV
Boundary:   Vacuum (no reflection, no incoming radiation)

For the multigroup case we use  N_groups = 4  groups with equal width edges in
[0, 40] keV.  Since all groups see the same σ_a, the multigroup result must
match the gray result (within MC noise).

The test prints the cell-centre T(r) and Tr(r) at the final time for both
solvers and also generates a comparison plot saved to `test_mg_vs_gray.pdf`.

Usage:
    python test_mg_imc1d_spherical.py
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for CI/server use
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Path setup – allow running from either the MG_IMC or the IMC1D directory
# ---------------------------------------------------------------------------
_this_dir = os.path.dirname(os.path.abspath(__file__))
_mg_dir   = os.path.dirname(_this_dir)   # MG_IMC/ — contains MG_IMC1D.py
_parent   = os.path.dirname(_mg_dir)     # RadTranBook/

# MG_IMC1D is in the MG_IMC/ directory
sys.path.insert(0, _mg_dir)

# IMC1D.py may be in parent/IMC/ or parent/
for candidate in [os.path.join(_parent, "IMC"), _parent]:
    if os.path.isfile(os.path.join(candidate, "IMC1D.py")):
        sys.path.insert(0, candidate)
        break

# ---------------------------------------------------------------------------
# Problem parameters
# ---------------------------------------------------------------------------
R       = 10.0       # cm,  outer radius
I_CELLS = 20         # number of radial zones
T_MAT0  = 1.0        # keV, initial material temperature
T_RAD0  = 0.1        # keV, initial radiation temperature
SIGMA_A = 10.0       # cm⁻¹, constant group-independent opacity
CV      = 1.0        # GJ / cm³ / keV
DT      = 0.001      # ns,  time step
N_STEPS = 5          # number of steps (a few ns)
NTARGET = 8000       # target number of emission particles per step
NBOUNDARY = 0        # no incoming boundary radiation
NMAX    = 40000      # maximum particle count after combing

N_GROUPS = 4
E_MAX    = 40.0      # keV (covers the relevant Planck range at T ~ 1 keV)
ENERGY_EDGES = np.linspace(0.0, E_MAX, N_GROUPS + 1)

SEED = 42            # reproducibility

# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------
r_edges = np.linspace(0.0, R, I_CELLS + 1)
mesh    = np.column_stack([r_edges[:-1], r_edges[1:]])   # (I, 2)
r_centers = 0.5 * (mesh[:, 0] + mesh[:, 1])

# ---------------------------------------------------------------------------
# EOS (linear internal energy)
# ---------------------------------------------------------------------------
eos     = lambda T: CV * T
inv_eos = lambda u: u / CV
cv_func = lambda T: np.full_like(T, CV)

# ---------------------------------------------------------------------------
# Opacity functions
# ---------------------------------------------------------------------------
sigma_a_gray = lambda T: np.full(len(T) if hasattr(T, '__len__') else 1, SIGMA_A)
sigma_a_mg_funcs = [
    (lambda T: np.full(len(T) if hasattr(T, '__len__') else 1, SIGMA_A))
    for _ in range(N_GROUPS)
]

source = np.zeros(I_CELLS)

# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------
T_init  = np.full(I_CELLS, T_MAT0)
Tr_init = np.full(I_CELLS, T_RAD0)


# ===========================================================================
# 1. Gray IMC run  (IMC1D.py)
# ===========================================================================
def run_gray():
    """Run the gray 1-D spherical IMC simulation."""
    import IMC1D

    print("\n" + "=" * 60)
    print("GRAY IMC RUN  (IMC1D.py, geometry='spherical')")
    print("=" * 60)
    np.random.seed(SEED)

    # IMC1D.run_simulation returns (time_values, radiation_temperatures, temperatures)
    time_vals, Tr_hist, T_hist = IMC1D.run_simulation(
        NTARGET,
        NBOUNDARY,
        0,
        NMAX,
        T_init.copy(),
        Tr_init.copy(),
        (0.0, 0.0),
        DT,
        mesh,
        sigma_a_gray,
        eos,
        inv_eos,
        cv_func,
        source,
        DT * N_STEPS,
        reflect=(True, False),
        output_freq=1,
        theta=1.0,
        geometry="spherical",
    )

    return time_vals, Tr_hist, T_hist


# ===========================================================================
# 2. Multigroup IMC run  (MG_IMC1D.py)
# ===========================================================================
def run_multigroup():
    """Run the multigroup 1-D spherical IMC simulation."""
    import MG_IMC1D

    print("\n" + "=" * 60)
    print(f"MULTIGROUP IMC RUN  (MG_IMC1D.py, N_groups={N_GROUPS})")
    print("=" * 60)
    np.random.seed(SEED)

    # MG_IMC1D.run_simulation returns (time_values, Tr_history, T_history, state, history)
    time_vals, Tr_hist, T_hist, state, hist = MG_IMC1D.run_simulation(
        Ntarget=NTARGET,
        Nboundary=NBOUNDARY,
        Nsource=0,
        Nmax=NMAX,
        Tinit=T_init.copy(),
        Tr_init=Tr_init.copy(),
        T_boundary=(0.0, 0.0),
        dt=DT,
        mesh=mesh,
        energy_edges=ENERGY_EDGES,
        sigma_a_funcs=sigma_a_mg_funcs,
        eos=eos,
        inv_eos=inv_eos,
        cv=cv_func,
        source=source,
        final_time=DT * N_STEPS,
        reflect=(True, False),
        output_freq=1,
        theta=1.0,
        use_scalar_intensity_Tr=True,
    )

    return time_vals, Tr_hist, T_hist, state, hist


# ===========================================================================
# 3. Comparison and plot
# ===========================================================================
def compare_and_plot(gray_results, mg_results):
    """Compare final-step T and Tr profiles; produce comparison plot."""
    # gray returns (time, Tr_hist, T_hist); mg returns (time, Tr_hist, T_hist, state, hist)
    time_g, Tr_hist_g, T_hist_g = gray_results[0], gray_results[1], gray_results[2]
    time_m, Tr_hist_m, T_hist_m = mg_results[0], mg_results[1], mg_results[2]
    hist_m = mg_results[4] if len(mg_results) > 4 else []

    # Use the last available step for both
    T_final_g  = T_hist_g[-1]
    Tr_final_g = Tr_hist_g[-1]
    T_final_m  = T_hist_m[-1]
    Tr_final_m = Tr_hist_m[-1]
    t_final = min(time_g[-1], time_m[-1])

    print("\n" + "=" * 60)
    print(f"COMPARISON at t = {t_final:.4f} ns")
    print("=" * 60)
    print(f"{'r (cm)':>10}  {'T_gray':>10}  {'T_mg':>10}  {'|T_rel|':>10}  "
          f"{'Tr_gray':>10}  {'Tr_mg':>10}  {'|Tr_rel|':>10}")
    print("-" * 80)
    for i, r in enumerate(r_centers):
        dT   = abs(T_final_g[i]  - T_final_m[i])  / (abs(T_final_g[i])  + 1e-30)
        dTr  = abs(Tr_final_g[i] - Tr_final_m[i]) / (abs(Tr_final_g[i]) + 1e-30)
        print(f"{r:10.4f}  {T_final_g[i]:10.5f}  {T_final_m[i]:10.5f}  {dT:10.4f}  "
              f"{Tr_final_g[i]:10.5f}  {Tr_final_m[i]:10.5f}  {dTr:10.4f}")

    # Compute RMS relative differences
    rms_T  = np.sqrt(np.mean(((T_final_g  - T_final_m)  / (T_final_g  + 1e-30))**2))
    rms_Tr = np.sqrt(np.mean(((Tr_final_g - Tr_final_m) / (Tr_final_g + 1e-30))**2))
    print(f"\nRMS relative difference  T : {rms_T:.4f}")
    print(f"RMS relative difference  Tr: {rms_Tr:.4f}")

    # The RMS difference should be < 0.3 (30%) given the modest particle count.
    # A very tight threshold is not appropriate for stochastic MC, but large
    # systematic disagreement would indicate a bug.
    THRESHOLD = 0.30
    T_ok  = rms_T  < THRESHOLD
    Tr_ok = rms_Tr < THRESHOLD
    print(f"\nT  within {THRESHOLD*100:.0f}% RMS tolerance: {'PASS' if T_ok else 'FAIL'}")
    print(f"Tr within {THRESHOLD*100:.0f}% RMS tolerance: {'PASS' if Tr_ok else 'FAIL'}")

    # Energy conservation check

    print("--- Energy residuals (multigroup) ---")
    for h in hist_m:
        print(f"  t={h['time']:.4f} ns  residual={h.get('energy_residual', float('nan')):.3e}")

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(r_centers, T_final_g,  "b-o", ms=4, label="Gray IMC")
    axes[0].plot(r_centers, T_final_m,  "r--s", ms=4, label=f"MG IMC ({N_GROUPS} groups)")
    axes[0].set_xlabel("r (cm)")
    axes[0].set_ylabel("T (keV)")
    axes[0].set_title(f"Material Temperature  at t={t_final:.4f} ns")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(r_centers, Tr_final_g, "b-o", ms=4, label="Gray IMC")
    axes[1].plot(r_centers, Tr_final_m, "r--s", ms=4, label=f"MG IMC ({N_GROUPS} groups)")
    axes[1].set_xlabel("r (cm)")
    axes[1].set_ylabel("Tr (keV)")
    axes[1].set_title(f"Radiation Temperature  at t={t_final:.4f} ns")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(
        f"Gray vs Multigroup 1-D Spherical IMC\n"
        f"σ_a = {SIGMA_A} cm⁻¹ (all groups equal), {N_GROUPS} groups, "
        f"{N_STEPS} steps × {DT} ns",
        fontsize=11
    )
    plt.tight_layout()
    out_path = os.path.join(_this_dir, "test_mg_vs_gray.pdf")
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")

    return T_ok and Tr_ok


# ===========================================================================
# MAIN
# ===========================================================================
if __name__ == "__main__":
    gray_results = run_gray()
    mg_results   = run_multigroup()
    all_ok       = compare_and_plot(gray_results, mg_results)

    print("\n" + ("ALL TESTS PASSED" if all_ok else "SOME TESTS FAILED"))
    sys.exit(0 if all_ok else 1)
