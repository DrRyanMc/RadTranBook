#!/usr/bin/env python3
"""
Dilute Spectrum Shell — comparison methods (diffusion + gray IMC).

Runs one of five methods on the dilute-spectrum-shell benchmark and writes
snapshots in the same .npz format as run_dilute_spectrum_shell.py so that
plot_dilute_spectrum_shell.py can overlay all results.

Methods
-------
  mg        — multigroup diffusion, G groups, no flux limiter
  mg_fl     — multigroup diffusion, G groups, Larsen flux limiter
  gray      — gray diffusion (1 group over [ν_min, ν_max]), no flux limiter
  gray_fl   — gray diffusion (1 group), Larsen flux limiter
  imc_gray  — gray 1-D spherical IMC (IMC/IMC1D.py)

Opacity conventions
-------------------
  σ_{a,g}  : absorption (energy-exchange) cross section.
              Uses the geometric mean of boundary values, following the
              existing Marshak wave convention:
                  σ_{a,g} = sqrt( σ_a(T, E_low) · σ_a(T, E_high) )

  D_g       : diffusion coefficient.
              Uses the Rosseland harmonic mean within each group:
                  1/σ_{R,g} = ∫_g (1/σ_a(T,E)) dB/dT dE  /  ∫_g dB/dT dE
                  D_g = 1 / (3 σ_{R,g})
              Evaluated numerically with 60-point quadrature.

  σ_P_gray  : Planck mean absorption for gray IMC.
              Used instead of geometric mean because in the gray (1-group)
              limit the IMC absorption should reflect the frequency-weighted
              average:
                  σ_P = ∫ σ_a(T,E) B_E(T) dE  /  ∫ B_E(T) dE

Usage
-----
  python run_dilute_spectrum_shell_comparison.py --method mg --mode quick
  python run_dilute_spectrum_shell_comparison.py --method mg_fl --G 32
  python run_dilute_spectrum_shell_comparison.py --method gray
  python run_dilute_spectrum_shell_comparison.py --method imc_gray --mode standard
"""

import argparse
import os
import sys
import time as _time

import numpy as np

# ---------------------------------------------------------------------------
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(os.path.dirname(_here))  # problems -> MG_IMC -> RadTranBook
if _root not in sys.path:
    sys.path.insert(0, _root)

# Add nonEquilibriumDiffusion to path for MultigroupDiffusionSolver1D
_noneq_dir = os.path.join(_root, "nonEquilibriumDiffusion")
if _noneq_dir not in sys.path:
    sys.path.insert(0, _noneq_dir)

# Add IMC directory for gray IMC
_imc_dir = os.path.join(_root, "IMC")
if _imc_dir not in sys.path:
    sys.path.insert(0, _imc_dir)

from MG_IMC.problems.dilute_spectrum_shell import (
    C_LIGHT, A_RAD,
    R_S, R_1, R_2, R_OUT,
    T_S, T_INIT, T_FLOOR,
    RHO_CAVITY, RHO_SHELL, CV_SPEC,
    C_OPA, A_OPA, B_OPA, SIGMA_MAX,
    N_GROUPS_DEFAULT, NU_MIN, NU_MAX,
    T_FINAL, DT_DEFAULT, DUMP_TIMES,
    make_energy_edges,
)

# ===========================================================================
# Rosseland and Planck mean opacities (numerical quadrature)
# ===========================================================================

_N_QUAD = 60  # quadrature points for group integrals

def _rosseland_D(T_scalar, r, E_low, E_high):
    """Diffusion coefficient  D = 1 / (3 σ_R)  via Rosseland mean.

    Parameters
    ----------
    T_scalar : float — temperature (keV), scalar
    r        : float — radial position (cm), used only for density lookup
    E_low, E_high : float — group energy bounds (keV)

    Returns
    -------
    D : float — diffusion coefficient (cm)
    """
    T_use = max(float(T_scalar), T_FLOOR)
    rho_r = RHO_SHELL if (R_1 <= r < R_2) else RHO_CAVITY

    if rho_r == 0.0:
        # Vacuum cell: sigma_R = 0, return large D (free-streaming floor);
        # the FLD flux limiter will saturate the flux to |F| ≤ c·E anyway.
        return 1.0 / (3.0 * 1e-20)

    E = np.linspace(E_low, E_high, _N_QUAD)
    x = E / T_use
    # Stable evaluation of exp(x) / (exp(x)-1)^2 for all x
    x_clip = np.minimum(x, 500.0)
    ex = np.exp(x_clip)
    # For large x: ex/(ex-1)^2 ≈ exp(-x), for small x: ≈ 1/x^2
    dBdT_kernel = np.where(
        x_clip < 100.0,
        ex / (ex - 1.0 + 1e-300)**2,
        np.exp(-x_clip),
    )
    dBdT = (2.0 * E**4 / (C_LIGHT**2 * T_use**2)) * dBdT_kernel

    # 1/σ_a(T, E) = T^{-A} / (ρ·C) · E^{-B}
    # rho_r > 0 is guaranteed by the early-return guard above.
    inv_sigma_a = (T_use ** (-A_OPA)) / (rho_r * C_OPA) * (E ** (-B_OPA))

    denom = np.trapz(dBdT, E)
    if denom < 1e-300:
        # Fallback: geometric-mean opacity
        nu_bar = np.sqrt(E_low * E_high)
        sigma_geom = rho_r * C_OPA * T_use**A_OPA * nu_bar**B_OPA
        return 1.0 / (3.0 * max(min(sigma_geom, SIGMA_MAX), 1e-20))

    # Rosseland harmonic mean: 1/σ_R = ⟨1/σ_a⟩_{dB/dT}
    inv_sigma_R = np.trapz(inv_sigma_a * dBdT, E) / denom
    # D = (1/σ_R) / 3  =  ⟨1/σ_a⟩ / 3
    return float(max(inv_sigma_R, 1e-20)) / 3.0


def _planck_sigma(T_scalar, r, E_low, E_high):
    """Planck-mean absorption opacity  σ_P = ⟨σ_a⟩_{B_E(T)}.

    Used only for the gray IMC absorption cross section.
    """
    T_use = max(float(T_scalar), T_FLOOR)
    rho_r = RHO_SHELL if (R_1 <= r < R_2) else RHO_CAVITY

    E = np.linspace(E_low, E_high, _N_QUAD)
    x_clip = np.minimum(E / T_use, 500.0)
    B_E = (2.0 * E**3 / C_LIGHT**2) / (np.exp(x_clip) - 1.0 + 1e-300)

    denom = np.trapz(B_E, E)
    if denom < 1e-300:
        # Fallback: geometric-mean value
        nu_bar = np.sqrt(E_low * E_high)
        return min(rho_r * C_OPA * T_use**A_OPA * nu_bar**B_OPA, SIGMA_MAX)

    sigma_E = rho_r * C_OPA * T_use**A_OPA * (E**B_OPA)
    sigma_P = np.trapz(sigma_E * B_E, E) / denom
    return float(min(sigma_P, SIGMA_MAX))


def _rosseland_sigma(T_scalar, r, E_low, E_high):
    """Rosseland-mean absorption opacity  σ_R = 1 / (3 D_Ross).

    Harmonic mean weighted by dB/dT — dominated by high-energy (low-opacity)
    channels, making it appropriate for transport in non-equilibrium regimes
    where T_rad >> T_mat.
    """
    D = _rosseland_D(T_scalar, r, E_low, E_high)
    return float(min(1.0 / (3.0 * max(D, 1e-300)), SIGMA_MAX))


def _geom_mean_sigma(T_scalar, r, E_low, E_high):
    """Geometric-mean group absorption  √(σ_a(T,E_low)·σ_a(T,E_high)).

    Standard for diffusion absorption terms (existing codebase convention).
    """
    T_use = max(float(T_scalar), T_FLOOR)
    rho_r = RHO_SHELL if (R_1 <= r < R_2) else RHO_CAVITY
    s_low  = rho_r * C_OPA * T_use**A_OPA * E_low**B_OPA
    s_high = rho_r * C_OPA * T_use**A_OPA * E_high**B_OPA
    return float(min(np.sqrt(s_low * s_high), SIGMA_MAX))


# ===========================================================================
# Build sigma_a / D function lists for the diffusion solver
# ===========================================================================

def _make_diff_funcs(energy_edges, T_opacity=None):
    """Return (sigma_a_funcs, D_funcs, sigma_R_funcs) for the diffusion solver.

    All are lists of callables  f(T_scalar, r_scalar) → float.

    sigma_a_funcs : absorption opacity (geometric mean), for the emission term.
                    Always evaluated at the local material temperature.
    D_funcs       : diffusion coefficient D = 1/(3*sigma_R), Rosseland mean.
    sigma_R_funcs : Rosseland opacity sigma_R = 1/(3*D), required for FL methods.

    Parameters
    ----------
    T_opacity : float or None
        If given, D and sigma_R are evaluated at this fixed temperature instead
        of the local material temperature.  Useful for source-driven problems
        where the transport opacity should reflect the incoming radiation field
        (e.g. T_opacity = T_S, the inner-boundary source temperature).
        sigma_a is unaffected and always uses the local T.
    """
    n_groups = len(energy_edges) - 1
    sigma_a_funcs = []
    D_funcs       = []
    sigma_R_funcs = []
    for g in range(n_groups):
        E_lo = float(energy_edges[g])
        E_hi = float(energy_edges[g + 1])

        def _make(elo, ehi, T_fix):
            def sigma(T, r):
                return _geom_mean_sigma(T, r, elo, ehi)
            def D(T, r):
                T_eval = T_fix if T_fix is not None else T
                return _rosseland_D(T_eval, r, elo, ehi)
            def sigmaR(T, r):
                T_eval = T_fix if T_fix is not None else T
                d = _rosseland_D(T_eval, r, elo, ehi)
                return 1.0 / (3.0 * max(d, 1e-300))
            return sigma, D, sigmaR

        s, d, sR = _make(E_lo, E_hi, T_opacity)
        sigma_a_funcs.append(s)
        D_funcs.append(d)
        sigma_R_funcs.append(sR)

    return sigma_a_funcs, D_funcs, sigma_R_funcs


# ===========================================================================
# Boundary condition helpers
# (These are created inside run_diffusion to avoid importing planck_integrals
#  at module level; Bg_multigroup comes from multigroup_diffusion_solver.)
# ===========================================================================


# ===========================================================================
# Radiation flux from diffusion solution
# ===========================================================================

def _compute_diff_flux_by_group(phi_g_stored, D_funcs, T_arr, r_centers,
                                sigma_R_funcs=None):
    """Compute cell-centred radial flux  F_g = -D_g^{eff}(T, r) · dφ_g/dr.

    For plain diffusion:  D_g^{eff} = D_g^{Ross} = 1/(3 σ_{R,g}).
    For flux-limited diffusion (sigma_R_funcs provided): applies the Larsen
    (n=2) limiter so that

        R_g     = |∇φ_g| / (σ_{R,g} φ_g)
        λ(R_g)  = (9 + R_g²)^{-1/2}             ← Larsen n=2
        D_g^FL  = λ(R_g) / σ_{R,g}

    recovering D_g^Ross in the diffusion limit (R→0) and saturating the
    flux at |F_g| → φ_g in the free-streaming limit (R→∞).

    Parameters
    ----------
    phi_g_stored  : (n_groups, n_cells)  scalar flux φ_g = c E_g  [GJ/cm²/ns]
    D_funcs       : list of n_groups callables  D_g(T, r) → float  [cm]
    T_arr         : (n_cells,)  material temperature [keV]
    r_centers     : (n_cells,)  cell-centre radii [cm]
    sigma_R_funcs : list of n_groups callables σ_{R,g}(T, r) → float  [1/cm],
                    or None for plain diffusion.

    Returns
    -------
    F_by_group : (n_groups, n_cells)  radial flux  [GJ/cm²/ns]
    """
    n_groups, n_cells = phi_g_stored.shape

    # Central differences of scalar flux (one-sided at boundaries)
    dphi_dr = np.zeros_like(phi_g_stored)
    dphi_dr[:, 1:-1] = ((phi_g_stored[:, 2:] - phi_g_stored[:, :-2])
                        / (r_centers[2:] - r_centers[:-2])[np.newaxis, :])
    dphi_dr[:,  0  ] = ((phi_g_stored[:,  1] - phi_g_stored[:,  0])
                        / (r_centers[1] - r_centers[0]))
    dphi_dr[:, -1  ] = ((phi_g_stored[:, -1] - phi_g_stored[:, -2])
                        / (r_centers[-1] - r_centers[-2]))

    if sigma_R_funcs is None:
        # Plain diffusion: D_eff = D_Ross
        D_vals = np.array([[D_funcs[g](float(T_arr[i]), float(r_centers[i]))
                            for i in range(n_cells)]
                           for g in range(n_groups)])
        return -D_vals * dphi_dr

    # Flux-limited diffusion: D_eff = λ(R_g) / σ_{R,g}  (Larsen n=2)
    sigma_R_vals = np.array(
        [[sigma_R_funcs[g](float(T_arr[i]), float(r_centers[i]))
          for i in range(n_cells)]
         for g in range(n_groups)])                       # (n_groups, n_cells)
    phi_floor = np.maximum(np.abs(phi_g_stored), 1e-300)
    R_g       = np.abs(dphi_dr) / (sigma_R_vals * phi_floor)  # dimensionless
    lam       = (9.0 + R_g**2) ** (-0.5)                 # Larsen n=2, λ(0)=1/3
    D_FL      = lam / np.maximum(sigma_R_vals, 1e-300)
    return -D_FL * dphi_dr


# ===========================================================================
# Snapshot saving (same format as MG IMC run script)
# ===========================================================================

def _save_snapshot(out_dir, t, r_centers, r_edges, T_mat, T_rad,
                   E_rad, E_rad_by_group, energy_edges, rho_per_cell,
                   F_rad=None, F_rad_by_group=None):
    fname = os.path.join(out_dir, f"snapshot_t_{t:.5f}ns.npz")
    arrays = dict(
        r_centers=r_centers,
        r_edges=r_edges,
        T_mat=T_mat,
        T_rad=T_rad,
        E_rad=E_rad,
        E_rad_by_group=E_rad_by_group,
        energy_edges=energy_edges,
        rho=rho_per_cell,
        time=np.float64(t),
    )
    if F_rad_by_group is not None:
        arrays["F_rad_by_group"] = F_rad_by_group
        arrays["F_rad"] = (np.sum(F_rad_by_group, axis=0)
                           if F_rad is None else F_rad)
    np.savez_compressed(fname, **arrays)
    print(f"  *** Snapshot → {fname}")


# ===========================================================================
# DIFFUSION METHODS (mg, mg_fl, gray, gray_fl)
# ===========================================================================

def run_diffusion(args):
    from multigroup_diffusion_solver import (
        MultigroupDiffusionSolver1D,
        flux_limiter_larsen, Bg_multigroup,
    )

    use_fl  = "_fl" in args.method
    is_gray = args.method.startswith("gray")
    is_src  = args.method.endswith("_src")
    n_groups = 1 if is_gray else args.G

    mode_cells = {"quick": 60, "standard": 90, "publication": 120}
    n_cells = mode_cells[args.mode]
    dt = args.dt

    # Uniform mesh on [R_S, R_OUT]
    r_edges   = np.linspace(R_S, R_OUT, n_cells + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    rho_per_cell = np.where((r_centers >= R_1) & (r_centers < R_2),
                            RHO_SHELL, RHO_CAVITY)
    #make cv_vol rho*CV_SPEC when rho >1 else make cv_vol = 1 
    cv_vol = np.where(rho_per_cell > 1, rho_per_cell * CV_SPEC, 1.0)  # GJ / (cm³ · keV)

    energy_edges = np.array([NU_MIN, NU_MAX]) if is_gray else make_energy_edges(n_groups)

    # Build per-group sigma_a, D (Rosseland), and sigma_R functions.
    # For *_src variants: D and sigma_R are fixed at the source temperature T_S
    # so that transport opacity reflects the incoming radiation field, not the
    # cold local material.  sigma_a (energy-exchange term) always uses local T.
    T_opacity = T_S if is_src else None
    sigma_a_funcs, D_funcs, sigma_R_funcs = _make_diff_funcs(energy_edges,
                                                              T_opacity=T_opacity)

    # --- Marshak BCs ---
    # Inner (r = R_S): blackbody at T_S
    #   Robin:  0.5·φ + D_g(T_S, R_S)·(dφ/dr)|_face = χ_g · (a·c·T_S⁴) / 2
    # Outer (r = R_OUT): vacuum
    #   Robin:  0.5·φ + D_g(T_init, R_OUT)·(dφ/dr)|_face = 0
    B_g_bc = Bg_multigroup(energy_edges, T_S)
    chi    = B_g_bc / (B_g_bc.sum() + 1e-300)  # group fractions at T_S
    F_total = (A_RAD * C_LIGHT * T_S**4) / 2.0  # total incoming half-flux

    def _make_left_bc(g, elo, ehi):
        D_bc = _rosseland_D(T_S, R_S, elo, ehi)
        C_g  = float(chi[g]) * F_total
        def bc(phi, r):
            return 0.5, D_bc, C_g
        return bc

    def _make_right_bc(g, elo, ehi):
        D_cold = _rosseland_D(T_INIT, R_OUT, elo, ehi)
        def bc(phi, r):
            return 0.5, D_cold, 0.0
        return bc

    left_bcs  = [_make_left_bc(g, float(energy_edges[g]), float(energy_edges[g+1]))
                 for g in range(n_groups)]
    right_bcs = [_make_right_bc(g, float(energy_edges[g]), float(energy_edges[g+1]))
                 for g in range(n_groups)]

    # For FL methods: pass sigma_R_funcs so that D_FL = λ(R) / σ_R
    # For non-FL:     flux_limiter_funcs=None → solver handles D directly
    fl_arg      = flux_limiter_larsen if use_fl else None
    sigma_R_arg = sigma_R_funcs       if use_fl else None

    print("=" * 72)
    src_note = ", D @ T_src" if is_src else ""
    print(f"Dilute Spectrum Shell — {'MG' if not is_gray else 'Gray'} Diffusion"
          f"{' + Larsen FL' if use_fl else ''}{src_note}")
    print(f"  Groups: {n_groups},  Cells: {n_cells},  dt: {dt} ns")
    print(f"  Method tag: {args.method}")
    if use_fl:
        print(f"  Streaming-limit R floor: {'on' if args.enforce_streaming_R_floor else 'off'}")
    print("=" * 72)

    cv_vol_ref = cv_vol  # capture before lambda
    # Vacuum cells (rho=0) have cv_vol=0; return inf so beta=4aT³/cv→0 and
    # the Fleck factor stays finite.  sigma_a=0 there so the value is moot.
    cv_safe    = np.where(rho_per_cell > 0, cv_vol, np.inf)
    solver = MultigroupDiffusionSolver1D(
        n_groups=n_groups,
        r_min=R_S,
        r_max=R_OUT,
        n_cells=n_cells,
        energy_edges=energy_edges,
        geometry="spherical",
        dt=dt,
        diffusion_coeff_funcs=D_funcs,
        flux_limiter_funcs=fl_arg,
        rosseland_opacity_funcs=sigma_R_arg,  # σ_R for FL; None for non-FL
        absorption_coeff_funcs=sigma_a_funcs,
        left_bc_funcs=left_bcs,
        right_bc_funcs=right_bcs,
        rho=1.0,                           # placeholder; EOS overridden below
        cv=lambda T: cv_safe,              # inf for vacuum → beta=0, no NaN
        material_energy_func=lambda T: cv_vol_ref * T,
        inverse_material_energy_func=lambda e: np.where(
            cv_vol_ref > 0, e / np.where(cv_vol_ref > 0, cv_vol_ref, 1.0), T_INIT
        ),
        enforce_streaming_R_floor=(use_fl and args.enforce_streaming_R_floor),
    )

    # Cold initial condition: T = T_INIT everywhere, equilibrium E_r
    T0 = np.full(n_cells, T_INIT)
    solver.T     = T0.copy()
    solver.T_old = T0.copy()
    solver.E_r     = A_RAD * T0**4
    solver.E_r_old = A_RAD * T0**4
    solver.kappa     = np.zeros(n_cells)
    solver.kappa_old = np.zeros(n_cells)
    # Initialize per-group phi guess (phi_g = a*c*T^4 / n_groups)
    solver.phi_g_stored[:, :] = A_RAD * C_LIGHT * T_INIT**4 / n_groups
    solver.t = 0.0

    # Output directory
    tag     = f"{args.method}_{n_groups}g_{args.mode}"
    out_dir = os.path.join("results", "dilute_spectrum_shell", tag)
    os.makedirs(out_dir, exist_ok=True)

    dump_times   = sorted(DUMP_TIMES)
    dump_idx     = 0
    saved_dumps  = set()
    step_count   = 0
    time_tol     = 1e-12 * max(T_FINAL, 1.0)
    MIN_DT_FLOOR = 1e-12

    print(f"\n{'Step':>5}  {'t (ns)':>9}  {'T_max':>9}  {'T_min':>9}  "
          f"{'E_r_max':>12}  {'Newton':>6}")
    print("-" * 65)

    while solver.t < T_FINAL - time_tol:
        step_dt = dt
        if dump_idx < len(dump_times):
            gap = dump_times[dump_idx] - solver.t
            if gap > MIN_DT_FLOOR:
                step_dt = min(step_dt, gap)

        step_dt = min(step_dt, T_FINAL - solver.t)
        if step_dt < MIN_DT_FLOOR:
            solver.advance_time()
            continue

        solver.dt = step_dt
        t0_wall = _time.perf_counter()
        info = solver.step(
            max_newton_iter=5,
            newton_tol=1e-6,
            gmres_tol=1e-10,
            gmres_maxiter=400,
            use_preconditioner=False,
            max_relative_change=2.0,
        )
        solver.advance_time()
        step_count += 1
        wall = _time.perf_counter() - t0_wall

        print(f"{step_count:>5}  {solver.t:>9.5f}  {solver.T.max():>9.5f}"
              f"  {solver.T.min():>9.5f}  {solver.E_r.max():>12.4e}"
              f"  {info['newton_iter']:>6}  [{wall:.1f}s]")

        # Per-group E_r from stored fractional distribution:
        #   solver.phi_g_fraction[g, :] = phi_g / sum_g(phi_g), updated each step
        #   E_r_g = phi_g_fraction[g] * E_r_total
        E_rad_by_group = solver.phi_g_fraction * solver.E_r[np.newaxis, :]

        T_rad = (np.maximum(solver.E_r, 0.0) / A_RAD) ** 0.25

        # Check dump times
        while (dump_idx < len(dump_times) and
               solver.t >= dump_times[dump_idx] - time_tol):
            if dump_idx not in saved_dumps:
                # Flux: F_g = -D_g^{eff} * d(phi_g)/dr  [GJ/cm²/ns]
                # For FL methods pass sigma_R_funcs so the Larsen limiter
                # is applied consistently with how the solver evolves the field.
                F_rad_by_group = _compute_diff_flux_by_group(
                    solver.phi_g_stored, D_funcs, solver.T, r_centers,
                    sigma_R_funcs=sigma_R_funcs if use_fl else None)
                _save_snapshot(
                    out_dir, solver.t, r_centers, r_edges,
                    solver.T, T_rad, solver.E_r, E_rad_by_group,
                    energy_edges, rho_per_cell,
                    F_rad_by_group=F_rad_by_group,
                )
                saved_dumps.add(dump_idx)
            dump_idx += 1

    print(f"\nDone. {step_count} steps.")


# ===========================================================================
# GRAY IMC METHOD (imc_gray)
# ===========================================================================

def run_imc_gray(args):
    import pickle, random
    from IMC1D import SimulationState, init_simulation, step

    mode_params = {
        "quick":       dict(Ntarget=2_000,  Nboundary=5_000,   NMax=20_000),
        "standard":    dict(Ntarget=10_000, Nboundary=50_000,  NMax=100_000),
        "publication": dict(Ntarget=50_000, Nboundary=200_000, NMax=5_000_000),
    }
    p = mode_params[args.mode]
    Ntarget   = p["Ntarget"]
    Nboundary = p["Nboundary"]
    NMax      = p["NMax"]
    dt        = args.dt

    # Non-uniform mesh (same as MG IMC quick/standard modes)
    mesh_mode = "quick" if args.mode == "quick" else "standard"
    from MG_IMC.problems.dilute_spectrum_shell import make_mesh
    mesh, r_centers, rho_per_cell = make_mesh(mode=mesh_mode)
    n_cells = mesh.shape[0]
    #make cv_vol same as MG diffusion cv_vol = rho*CV_SPEC when rho >1 else make cv_vol = 1 for consistency with diffusion solver
    cv_vol = np.where(rho_per_cell > 1.0, rho_per_cell * CV_SPEC, 1.0)  # GJ / (cm³·keV)

    # Gray absorption: Planck or Rosseland mean over [NU_MIN, NU_MAX].
    # Planck mean (default): weighted by B(T_mat) → dominated by low-E channels
    #   with high opacity; tends to over-absorb in non-equilibrium cavities.
    # Rosseland mean: harmonic mean weighted by dB/dT → dominated by high-E
    #   (low-opacity) channels; much smaller in cold transparent regions.
    _scalar_opacity = {
        "planck":    _planck_sigma,
        "rosseland": _rosseland_sigma,
    }[args.gray_opacity]
    _T_fixed = T_S if args.gray_T_eval == "src" else None

    def sigma_a_gray(T_arr):
        out = np.empty(len(T_arr))
        for i in range(len(T_arr)):
            T_use = _T_fixed if _T_fixed is not None else T_arr[i]
            out[i] = _scalar_opacity(T_use, r_centers[i], NU_MIN, NU_MAX)
        return out

    def eos(T):
        return cv_vol * T

    def inv_eos(u):
        return u / cv_vol

    def cv_func(T):
        return cv_vol * np.ones_like(T)

    T_boundary = (T_S, 0.0)
    reflect    = (False, False)
    source     = np.zeros(n_cells)

    opa_label  = {"planck": "Planck-mean  σ_P", "rosseland": "Rosseland-mean  σ_R"}[args.gray_opacity]
    T_tag      = "_src" if args.gray_T_eval == "src" else ""
    T_eval_lbl = f"@ T_S={T_S} keV (fixed)" if args.gray_T_eval == "src" else "@ local T_mat"
    tag     = f"imc_gray_{args.gray_opacity}{T_tag}_{args.mode}"
    out_dir = os.path.join("results", "dilute_spectrum_shell", tag)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 72)
    print(f"Dilute Spectrum Shell — Gray IMC  ({args.gray_opacity} opacity, T eval {T_eval_lbl})")
    print(f"  Mode: {args.mode},  Cells: {n_cells},  dt: {dt} ns")
    print(f"  Ntarget: {Ntarget},  Nboundary: {Nboundary},  NMax: {NMax}")
    print(f"  Gray opacity: {opa_label}  over [{NU_MIN}, {NU_MAX}] keV")
    print("=" * 72)

    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)

    Tinit   = np.full(n_cells, T_INIT)
    Tr_init = np.full(n_cells, T_INIT)
    state = init_simulation(Ntarget, Tinit, Tr_init, mesh, eos, inv_eos,
                             geometry="spherical", T_emit_floor=0.05)

    dump_times   = sorted(DUMP_TIMES)
    dump_idx     = 0
    saved_dumps  = set()
    step_count   = 0
    time_tol     = 1e-12 * max(T_FINAL, 1.0)
    r_edges = np.concatenate([mesh[:, 0], [mesh[-1, 1]]])
    energy_edges = np.array([NU_MIN, NU_MAX])  # single-group edges

    print(f"\n{'Step':>6}  {'t (ns)':>9}  {'N_part':>7}  {'Resid':>10}")
    print("-" * 40)

    while state.time < T_FINAL - time_tol:
        step_dt = dt
        if dump_idx < len(dump_times):
            gap = dump_times[dump_idx] - state.time
            if gap > 1e-12:
                step_dt = min(step_dt, gap)
        step_dt = min(step_dt, T_FINAL - state.time)

        state, info = step(
            state, Ntarget, Nboundary, 0, NMax,
            T_boundary, step_dt, mesh,
            sigma_a_gray, inv_eos, cv_func, source,
            reflect=reflect, theta=1.0,
            use_scalar_intensity_Tr=True,
            conserve_comb_energy=True,
            T_emit_floor=0.05,   # no emission from cells colder than 0.05 keV
            geometry="spherical",
        )
        step_count += 1

        print(f"{step_count:>6}  {info['time']:>9.5f}  "
              f"{info['N_particles']:>7d}  {info['energy_loss']:>10.3e}")

        # Gray IMC: use radiation_temperature already computed by step()
        T_rad = state.radiation_temperature
        E_rad = A_RAD * T_rad**4          # energy density (GJ/cm³)
        E_rad_by_group = E_rad[np.newaxis, :]  # shape (1, n_cells)

        while (dump_idx < len(dump_times) and
               state.time >= dump_times[dump_idx] - time_tol):
            if dump_idx not in saved_dumps:
                _save_snapshot(
                    out_dir, state.time, r_centers, r_edges,
                    state.temperature, T_rad, E_rad, E_rad_by_group,
                    energy_edges, rho_per_cell,
                )
                saved_dumps.add(dump_idx)
            dump_idx += 1

    print(f"\nDone. {step_count} steps.")


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Run comparison methods for the dilute-spectrum-shell benchmark."
    )
    p.add_argument(
        "--method",
        choices=["mg", "mg_fl", "gray", "gray_fl",
                 "gray_src", "gray_fl_src", "imc_gray"],
        required=True,
        help="Solver method to run.  The *_src variants evaluate D and σ_R"
             " at the source BC temperature T_S rather than local T.",
    )
    p.add_argument(
        "--mode", choices=["quick", "standard", "publication"],
        default="standard",
    )
    p.add_argument(
        "--G", type=int, default=N_GROUPS_DEFAULT, metavar="N_GROUPS",
        help=f"Number of energy groups for MG methods (default: {N_GROUPS_DEFAULT}).",
    )
    p.add_argument(
        "--dt", type=float, default=DT_DEFAULT, metavar="DT",
        help=f"Nominal timestep in ns (default: {DT_DEFAULT}).",
    )
    p.add_argument(
        "--enforce_streaming_R_floor",
        action="store_true",
        help="For FL methods, enforce R >= n_geom/(sigma_R*r) in limiter evaluation."
             " Disabled by default.",
    )
    p.add_argument(
        "--gray_opacity", choices=["planck", "rosseland"], default="planck",
        help="Gray-IMC opacity average: 'planck' (default) weights by B(T_mat),"
             " 'rosseland' weights by dB/dT (harmonic mean, transport-appropriate).",
    )
    p.add_argument(
        "--gray_T_eval", choices=["mat", "src"], default="mat",
        help="Temperature at which the gray opacity is evaluated: 'mat' (default)"
             " uses the local material temperature; 'src' fixes T = T_S so the"
             " opacity reflects the source photon spectrum rather than cold matter.",
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help="Random seed (gray IMC only).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.method in ("mg", "mg_fl", "gray", "gray_fl", "gray_src", "gray_fl_src"):
        run_diffusion(args)
    else:
        run_imc_gray(args)
