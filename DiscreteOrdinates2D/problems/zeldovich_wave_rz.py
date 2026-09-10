import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

"""
2-D Cylindrical (r-z) Zel'dovich Wave — Discrete Ordinates S_N version.

Initializes from the N=2 (cylindrical) self-similar solution at t_start
and runs to t_final.  Since the problem is axially uniform, the S_N
solution should stay z-independent and match the 1-D cylindrical
self-similar profile.

Physics:
  σ = σ₀ · T^{-n},   σ₀ = 3000 cm⁻¹,  n = 3
  c_v = 100 GJ/(cm³·keV)
  E₀  = 10 GJ/cm   (energy per unit z-length, matched to Cartesian run)
  All boundaries: reflecting.

Quadrature:
  Product quadrature in (μ, λ) is natural for r-z because directions are
  grouped into levels of constant μ.  Level-symmetric and equal-weight
  sets also work.

Run from the DiscreteOrdinates2D/problems directory:
    python zeldovich_wave_rz.py
    python zeldovich_wave_rz.py --Ir 80 --Iz 10 --N 4 --tfinal 1.0
    python zeldovich_wave_rz.py --study
"""

import sys
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from DiscreteOrdinates2D.src.sn_solver_2d_rz import temp_solve_rz, build_rz_mesh, c, a, ac


# ===========================================================================
# Physics parameters (matched to Cartesian Zeldovich run)
# ===========================================================================

_n       = 3.0          # opacity exponent: σ = σ₀ · T^{-n}
_sigma0  = 3000.0       # reference opacity (cm⁻¹)
_cv_vol  = 100.0        # volumetric heat capacity (GJ/(cm³·keV))
_E0      = 10.0         # total injected energy per unit z-length (GJ/cm)

_K0      = 4.0 * ac / (3.0 * _sigma0)   # diffusion prefactor
_m_diff  = _n + 3                        # = 6


# ===========================================================================
# Self-similar solution (N=2 cylindrical)
# ===========================================================================

def zeldovich_self_similar(r, t, N=2):
    """Self-similar solution of c_v ∂T/∂t = ∇·(K₀ T^m ∇T) in N dimensions.

    Parameters
    ----------
    r : array-like   radial positions (cm)
    t : float        time (ns)
    N : int          spatial dimension (2 = cylindrical)

    Returns
    -------
    T : ndarray      temperature (keV)
    R_front : float  wave-front radius (cm)
    """
    from scipy.special import beta as Beta_func
    from math import pi, gamma

    r = np.asarray(r, dtype=float)
    m = _m_diff     # 6
    D = _K0 / _cv_vol

    beta  = 1.0 / (N * m + 2.0)
    alpha = N * beta
    B     = m * beta / (2.0 * D)

    beta_int = Beta_func(N / 2.0, 1.0 / m + 1.0)
    S_N      = 2.0 * pi**(N / 2.0) / gamma(N / 2.0)

    Q     = _E0 / _cv_vol
    power = 1.0 / m + N / 2.0
    A     = (Q * 2.0 * B**(N / 2.0) / (S_N * beta_int)) ** (1.0 / power)

    eta_f   = np.sqrt(A / B)
    R_front = eta_f * t**beta

    T = np.zeros_like(r, dtype=float)
    mask = r < R_front
    if np.any(mask):
        xi        = r[mask] / R_front
        T[mask]   = t**(-alpha) * A**(1.0 / m) * (1.0 - xi**2)**(1.0 / m)

    return T, R_front


# ===========================================================================
# Material closures
# ===========================================================================

def _make_materials():
    def EOS(T):
        return _cv_vol * T

    def invEOS(e):
        return e / _cv_vol

    def sigma_func(T):
        return _sigma0 * np.maximum(T, 1e-6) ** (-_n)

    def scat_func(T):
        return np.zeros_like(T)

    return EOS, invEOS, sigma_func, scat_func


# ===========================================================================
# Grid generation with optional refinement near r = 0
# ===========================================================================

def _make_radial_grid(Ir, Lr, stretch=1.0):
    """Build a radial grid: uniform (stretch=1) or refined near r=0 (stretch>1)."""
    if abs(stretch - 1.0) < 1e-10:
        return np.full(Ir, Lr / Ir)
    # Power-law stretch: first cell = dr_min, ratio r ~ stretch^(1/(Ir-1))
    r = stretch ** (1.0 / max(Ir - 1, 1))
    dr0 = Lr * (r - 1.0) / (r**Ir - 1.0) if abs(r - 1.0) > 1e-12 else Lr / Ir
    return np.array([dr0 * r**k for k in range(Ir)])


def _make_axial_grid(Iz, Lz):
    """Build a uniform axial grid."""
    return np.full(Iz, Lz / Iz)


# ===========================================================================
# Main problem runner
# ===========================================================================

def run_zeldovich_rz(
    Ir=60, Iz=8,
    N_quad=4, quad_type='product_square',
    Lr=1.5, Lz=0.2,
    t_start=0.1, tfinal=1.0,
    dt_min=1e-5, dt_max=1e-3,
    K=10, maxits=100, W=3, R=3,
    stretch=1.5,
    use_dmd=True, LOUD=False,
    print_stride=50,
    output_times=None,
):
    """Run the 2-D cylindrical r-z Zel'dovich wave problem.

    Parameters
    ----------
    Ir, Iz : int
        Number of cells in r and z.
    N_quad : int
        S_N quadrature order (product quadrature).
    quad_type : str
        Quadrature type.  'product_square' recommended for r-z geometry.
    Lr, Lz : float
        Domain extents (cm).
    t_start : float
        Initialization time — self-similar IC evaluated here.
    tfinal : float
        Final time (ns).
    dt_min, dt_max : float
        Time step bounds (ns).
    K, maxits, W, R : int
        DMD / source-iteration parameters.
    stretch : float
        Radial grid stretching (>1 refines near r=0).
    output_times : array-like or None
        Snapshot times.  Defaults to a few evenly spaced times.

    Returns
    -------
    result : dict with keys 'phis', 'Ts', 'ts', 'r_centers', 'z_centers',
             'r_faces', 'z_faces', 't_start', 'tfinal', 'Ir', 'Iz'.
    """
    print(f"\n{'='*60}")
    print(f"  2-D r-z Zel'dovich Wave — S_N version")
    print(f"  Grid: Ir={Ir}, Iz={Iz},  {quad_type} N={N_quad}")
    print(f"  Domain: r∈[0,{Lr}] cm, z∈[0,{Lz}] cm")
    print(f"  Time: {t_start:.3f} → {tfinal:.3f} ns")
    print(f"  σ₀={_sigma0}, n={_n}, c_v={_cv_vol}, E₀={_E0}")
    print(f"{'='*60}\n")

    # --- Grid ---
    dr_arr  = _make_radial_grid(Ir, Lr, stretch)
    dz_arr  = _make_axial_grid(Iz, Lz)
    r_faces = np.concatenate([[0.0], np.cumsum(dr_arr)])
    z_faces = np.concatenate([[0.0], np.cumsum(dz_arr)])
    r_centers, z_centers, _, _ = build_rz_mesh(r_faces, z_faces)

    print(f"  dr range: [{dr_arr.min():.4e}, {dr_arr.max():.4e}] cm")
    print(f"  dz range: [{dz_arr.min():.4e}, {dz_arr.max():.4e}] cm")

    # --- Initial condition from self-similar solution at t_start ---
    T_floor = 1e-4   # keV — cold background
    T_init  = np.full((Ir, Iz, 4), T_floor)

    # Corner offsets: 0=NE (+r,+z), 1=NW (-r,+z), 2=SW (-r,-z), 3=SE (+r,-z)
    dr_off = np.array([+0.25, -0.25, -0.25, +0.25])
    dz_off = np.array([+0.25, +0.25, -0.25, -0.25])

    for i in range(Ir):
        for j in range(Iz):
            for cc in range(4):
                rc = r_centers[i] + dr_off[cc] * dr_arr[i]
                rc = max(rc, 0.0)
                T_val, _ = zeldovich_self_similar(np.array([rc]), t_start, N=2)
                T_init[i, j, cc] = max(T_val[0], T_floor)

    phi_init = ac * T_init**4

    _, R_front_init = zeldovich_self_similar(np.array([0.0]), t_start, N=2)
    _, R_front_final = zeldovich_self_similar(np.array([0.0]), tfinal, N=2)

    print(f"  Self-similar front at t_start = {R_front_init:.4f} cm")
    print(f"  Self-similar front at tfinal  = {R_front_final:.4f} cm")
    print(f"  T_max(t_start) = {T_init.max():.4f} keV")

    if R_front_final > 0.9 * Lr:
        print(f"  WARNING: front at tfinal ({R_front_final:.3f} cm) near "
              f"r_max ({Lr:.2f} cm) — consider a larger domain.")

    # --- Material closures ---
    EOS, invEOS, sigma_func, scat_func = _make_materials()

    # --- Boundary conditions ---
    # r=0: axis condition (handled automatically, no explicit BC needed).
    # r=r_max: reflecting (radiation stays inside domain).
    # z=0 and z=z_max: reflecting (domain is periodic in z, so no flux).
    # All external BCs return None (vacuum = zero incoming) since
    # reflecting is handled via the reflect_* flags.
    def BCs_func(t):
        return {'rlo': None, 'rhi': None, 'zlo': None, 'zhi': None}

    # --- Output times ---
    if output_times is None:
        output_times = np.array([0.3, 0.5, 1.0, 2.0, 3.0, 5.0])
        output_times = output_times[
            (output_times > t_start) & (output_times <= tfinal)]
    output_times = np.asarray(output_times, dtype=float)

    q_ext = np.zeros((Ir, Iz, 4))

    # --- Run solver ---
    phis, Ts, iterations, ts, its_per_step = temp_solve_rz(
        Ir, Iz, dr_arr, dz_arr, r_faces, z_faces,
        q_ext, sigma_func, scat_func,
        quad_type, N_quad,
        BCs_func, EOS, invEOS,
        phi_init, T_init,
        dt_min=dt_min, dt_max=dt_max, tfinal=tfinal,
        tolerance=1e-5, Linf_tol=1e-3, maxits=maxits,
        K=K, R=R, W=W,
        reflect_rlo=False,   # axis (handled by on_axis condition)
        reflect_rhi=True,    # outer radial wall
        reflect_zlo=True,    # bottom axial wall
        reflect_zhi=True,    # top axial wall
        use_dmd=use_dmd,
        LOUD=LOUD,
        print_stride=print_stride,
        time_outputs=output_times,
        store_full_history=False,
    )

    print(f"\n  Finished: {iterations} sweeps, {len(its_per_step)} steps")

    return {
        'phis': phis, 'Ts': Ts, 'ts': ts,
        'r_centers': r_centers, 'z_centers': z_centers,
        'r_faces': r_faces, 'z_faces': z_faces,
        't_start': t_start, 'tfinal': tfinal,
        'Ir': Ir, 'Iz': Iz,
        'its_per_step': its_per_step,
    }


# ===========================================================================
# Post-processing helpers
# ===========================================================================

def extract_radial_profile(result, t_target):
    """Extract z-averaged radial radiation temperature profile at t ≈ t_target.

    Returns (r_centers, T_rad, t_actual).
    """
    ts   = result['ts']
    phis = result['phis']
    r_centers = result['r_centers']

    idx     = int(np.argmin(np.abs(ts - t_target)))
    t_actual = float(ts[idx])
    phi_snap = phis[idx]                      # (Ir, Iz, 4)
    phi_cell = np.mean(phi_snap, axis=2)      # (Ir, Iz)  corner-average
    phi_r    = np.mean(phi_cell, axis=1)      # (Ir,)     z-average
    T_rad    = (np.maximum(phi_r, 0.0) / ac) ** 0.25

    return r_centers, T_rad, t_actual


# ===========================================================================
# Plotting
# ===========================================================================

def plot_radial_profiles(result, output_times, savefile='zeldovich_rz_sn.png'):
    """Plot z-averaged radial T_rad profiles vs self-similar solution."""
    _, R_max = zeldovich_self_similar(np.array([0.0]), max(output_times), N=2)
    r_ref = np.linspace(0.0, max(result['r_centers'][-1], R_max * 1.05), 300)

    colors = ['C0', 'C1', 'C2', 'C3', 'C4']
    fig, ax = plt.subplots(figsize=(8, 5))

    for k, t_tgt in enumerate(output_times):
        r_num, T_rad, t_act = extract_radial_profile(result, t_tgt)
        col = colors[k % len(colors)]

        ax.plot(r_num, T_rad, color=col, lw=2,
                label=f'S_N  t = {t_act:.2f} ns')

        T_ref, R_front = zeldovich_self_similar(r_ref, t_tgt, N=2)
        ax.plot(r_ref, T_ref, color=col, lw=1.5, ls='--',
                label=f'Self-similar  t = {t_tgt:.2f} ns')
        ax.axvline(R_front, color=col, ls=':', lw=1, alpha=0.5)

    ax.set_xlabel('r (cm)')
    ax.set_ylabel(r'Radiation temperature $T_r$ (keV)')
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xlim(0.0, result['r_centers'][-1])
    plt.tight_layout()
    plt.savefig(savefile, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {savefile}')


def plot_rz_heatmap(result, t_target, savefile=None):
    """Plot the r-z 2-D radiation temperature field at t ≈ t_target."""
    r_centers = result['r_centers']
    z_centers = result['z_centers']
    r_faces   = result['r_faces']
    z_faces   = result['z_faces']

    ts   = result['ts']
    phis = result['phis']
    idx  = int(np.argmin(np.abs(ts - t_target)))
    phi_snap = phis[idx]
    t_actual = float(ts[idx])

    phi_cell = np.mean(phi_snap, axis=2)              # (Ir, Iz)
    T_rad    = (np.maximum(phi_cell, 0.0) / ac) ** 0.25

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.pcolormesh(r_faces, z_faces, T_rad.T,
                       shading='flat', cmap='plasma')
    ax.set_xlabel('r (cm)')
    ax.set_ylabel('z (cm)')
    ax.set_title(f'Radiation temperature  t = {t_actual:.2f} ns')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label=r'$T_r$ (keV)')
    plt.tight_layout()

    if savefile is None:
        savefile = f'zeldovich_rz_sn_heatmap_t{t_target:.2f}ns.png'
    plt.savefig(savefile, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {savefile}')


# ===========================================================================
# Symmetry diagnostic: check z-independence
# ===========================================================================

def print_z_symmetry(result, t_target):
    """Print max relative variation in z (should be ~0 for this problem)."""
    ts   = result['ts']
    phis = result['phis']
    idx  = int(np.argmin(np.abs(ts - t_target)))
    phi_snap = phis[idx]                          # (Ir, Iz, 4)
    phi_cell = np.mean(phi_snap, axis=2)          # (Ir, Iz)

    # z-variation at each r
    phi_z_mean = np.mean(phi_cell, axis=1, keepdims=True)  # (Ir,1)
    z_var = np.max(np.abs(phi_cell - phi_z_mean)) / (phi_z_mean.max() + 1e-30)
    print(f"  t={ts[idx]:.3f} ns: max relative z-variation = {z_var:.3e}")


# ===========================================================================
# Convergence study
# ===========================================================================

def convergence_study(
    Ir_vals=(30, 60, 120),
    N_quad=4, quad_type='product_square',
    t_start=0.1, tfinal=1.0,
    compare_time=0.5,
):
    """Compare front position accuracy across radial mesh resolutions."""
    print(f"\n{'='*60}")
    print(f"  Convergence study: {quad_type} N={N_quad}, t={compare_time} ns")
    print(f"{'='*60}\n")

    r_ref = np.linspace(0.0, 2.0, 400)
    T_ref, R_ref = zeldovich_self_similar(r_ref, compare_time, N=2)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(r_ref, T_ref, 'k--', lw=2, label='Self-similar')

    colors = ['C0', 'C1', 'C2', 'C3']
    for k, Ir in enumerate(Ir_vals):
        Lz = 0.2; Iz = max(4, Ir // 10)
        _, R_front = zeldovich_self_similar(np.array([0.0]), tfinal, N=2)
        Lr = max(R_front * 1.1, 1.5)
        result = run_zeldovich_rz(
            Ir=Ir, Iz=Iz, N_quad=N_quad, quad_type=quad_type,
            Lr=Lr, Lz=Lz, t_start=t_start, tfinal=tfinal,
            dt_max=5e-4, output_times=[compare_time], use_dmd=True,
        )

        r_num, T_rad, t_act = extract_radial_profile(result, compare_time)
        ax.plot(r_num, T_rad, color=colors[k % len(colors)], lw=1.5,
                label=f'Ir={Ir}')

        # L2 error vs self-similar
        T_interp = np.interp(r_num, r_ref, T_ref)
        L2 = np.sqrt(np.mean((T_rad - T_interp)**2))
        print(f"  Ir={Ir:4d}: L2 error at t={t_act:.3f} = {L2:.4e}")

    ax.set_xlabel('r (cm)')
    ax.set_ylabel(r'Radiation temperature $T_r$ (keV)')
    ax.set_title(f'Cylindrical r-z S_N convergence at t = {compare_time} ns')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.0, 2.0)
    plt.tight_layout()
    plt.savefig('zeldovich_rz_sn_convergence.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: zeldovich_rz_sn_convergence.png")


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description='2-D r-z Zeldovich wave S_N')
    parser.add_argument('--Ir',      type=int,   default=60)
    parser.add_argument('--Iz',      type=int,   default=8)
    parser.add_argument('--N',       type=int,   default=4,   help='S_N order')
    parser.add_argument('--quad',    type=str,   default='product_square')
    parser.add_argument('--Lr',      type=float, default=1.5, help='r extent (cm)')
    parser.add_argument('--Lz',      type=float, default=0.2, help='z extent (cm)')
    parser.add_argument('--t-start', type=float, default=0.1)
    parser.add_argument('--tfinal',  type=float, default=1.0)
    parser.add_argument('--dt-min',  type=float, default=1e-5)
    parser.add_argument('--dt-max',  type=float, default=1e-3)
    parser.add_argument('--stretch', type=float, default=1.5)
    parser.add_argument('--no-dmd',  action='store_true')
    parser.add_argument('--loud',    action='store_true')
    parser.add_argument('--study',   action='store_true',
                        help='Run convergence study')
    parser.add_argument('--output-times', type=float, nargs='+',
                        default=None)
    parser.add_argument('--prefix',  type=str,   default='zeldovich_rz_sn')
    args = parser.parse_args()

    if args.study:
        convergence_study()
        return

    # Determine output times
    output_times = args.output_times
    if output_times is None:
        output_times = [t for t in [0.3, 0.5, 1.0, 2.0]
                        if t > args.t_start and t <= args.tfinal]

    result = run_zeldovich_rz(
        Ir=args.Ir, Iz=args.Iz,
        N_quad=args.N, quad_type=args.quad,
        Lr=args.Lr, Lz=args.Lz,
        t_start=args.t_start, tfinal=args.tfinal,
        dt_min=args.dt_min, dt_max=args.dt_max,
        stretch=args.stretch,
        use_dmd=not args.no_dmd,
        LOUD=args.loud,
        output_times=output_times,
    )

    print("\nPost-processing...")
    for t in output_times:
        print_z_symmetry(result, t)

    plot_radial_profiles(result, output_times,
                         savefile=f'{args.prefix}_profiles.png')
    for t in output_times:
        plot_rz_heatmap(result, t,
                        savefile=f'{args.prefix}_heatmap_t{t:.2f}ns.png')


if __name__ == '__main__':
    main()
