"""
2-D Cylindrical (r-z) Zel'dovich Wave — M1 version.

Mirrors DiscreteOrdinates2D/problems/zeldovich_wave_rz.py so the M1 and S_N
setups are directly comparable.

Physics:
  sigma_a = sigma0 * T^{-n},   sigma0 = 3000 cm^-1,  n = 3
  c_v = 100 GJ/(cm^3*keV)
  E0  = 10 GJ/cm   (energy per unit z-length)
  All boundaries reflecting.

Initialization:
  Uses the N=2 cylindrical self-similar profile at t_start.

Run from M1/problems:
  python zeldovich_wave_rz.py
  python zeldovich_wave_rz.py --Ir 80 --Iz 10 --tfinal 1.0
  python zeldovich_wave_rz.py --closure kershaw --prefix zeldovich_rz_m1_kershaw
"""

import sys
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_this_dir = os.path.dirname(os.path.abspath(__file__))
_m1_dir = os.path.dirname(_this_dir)
_repo_root = os.path.dirname(_m1_dir)
sys.path.insert(0, _m1_dir)
sys.path.insert(0, _repo_root)

from M1.src.m1_2d import (
    M1Solver2D,
    closure_levermore,
    closure_kershaw,
    closure_p1,
    closure_minerbo_poly,
    A_RAD,
    C_LIGHT,
)


# ===========================================================================
# Physics parameters (matched to S_N zeldovich_wave_rz.py)
# ===========================================================================

_n = 3.0
_sigma0 = 3000.0
_cv_vol = 100.0
_E0 = 10.0

_ac = A_RAD * C_LIGHT
_K0 = 4.0 * _ac / (3.0 * _sigma0)
_m_diff = _n + 3.0


# ===========================================================================
# Self-similar solution (N=2 cylindrical)
# ===========================================================================

def zeldovich_self_similar(r, t, N=2):
    """Self-similar solution of c_v dT/dt = div(K0 T^m grad T)."""
    from scipy.special import beta as beta_func
    from math import pi, gamma

    r = np.asarray(r, dtype=float)
    m = _m_diff
    D = _K0 / _cv_vol

    beta = 1.0 / (N * m + 2.0)
    alpha = N * beta
    B = m * beta / (2.0 * D)

    beta_int = beta_func(N / 2.0, 1.0 / m + 1.0)
    s_n = 2.0 * pi ** (N / 2.0) / gamma(N / 2.0)

    Q = _E0 / _cv_vol
    power = 1.0 / m + N / 2.0
    A = (Q * 2.0 * B ** (N / 2.0) / (s_n * beta_int)) ** (1.0 / power)

    eta_f = np.sqrt(A / B)
    r_front = eta_f * t ** beta

    T = np.zeros_like(r, dtype=float)
    mask = r < r_front
    if np.any(mask):
        xi = r[mask] / r_front
        T[mask] = t ** (-alpha) * A ** (1.0 / m) * (1.0 - xi ** 2) ** (1.0 / m)

    return T, r_front


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
# Grid generation
# ===========================================================================

def _make_radial_faces(Ir, Lr, stretch=1.0):
    if abs(stretch - 1.0) < 1e-12:
        dr = np.full(Ir, Lr / Ir)
    else:
        ratio = stretch ** (1.0 / max(Ir - 1, 1))
        if abs(ratio - 1.0) < 1e-12:
            dr = np.full(Ir, Lr / Ir)
        else:
            dr0 = Lr * (ratio - 1.0) / (ratio ** Ir - 1.0)
            dr = np.array([dr0 * ratio ** k for k in range(Ir)])
    return np.concatenate([[0.0], np.cumsum(dr)])


def _make_axial_faces(Iz, Lz):
    return np.linspace(0.0, Lz, Iz + 1)


# ===========================================================================
# Runner
# ===========================================================================

def run_zeldovich_rz_m1(
    Ir=60,
    Iz=8,
    closure_name='levermore',
    Lr=1.5,
    Lz=0.2,
    t_start=0.1,
    tfinal=1.0,
    dt_min=1e-5,
    dt_max=1e-3,
    dt_increase_factor=1.1,
    stretch=1.5,
    max_nonlinear_iter=20,
    nonlinear_tol=1e-6,
    relaxation=1.0,
    output_times=None,
    LOUD=False,
    print_stride=50,
):
    """Run the 2-D cylindrical r-z Zel'dovich wave with M1."""
    closure_map = {
        'levermore': closure_levermore,
        'kershaw': closure_kershaw,
        'p1': closure_p1,
        'minerbo_poly': closure_minerbo_poly,
    }
    if closure_name not in closure_map:
        raise ValueError(f"Unknown closure '{closure_name}'")

    print(f"\n{'='*60}")
    print('  2-D r-z Zel\'dovich Wave - M1 version')
    print(f"  Grid: Ir={Ir}, Iz={Iz}, closure={closure_name}")
    print(f"  Domain: r in [0,{Lr}] cm, z in [0,{Lz}] cm")
    print(f"  Time: {t_start:.3f} -> {tfinal:.3f} ns")
    print(f"  sigma0={_sigma0}, n={_n}, c_v={_cv_vol}, E0={_E0}")
    print(f"{'='*60}\n")

    r_faces = _make_radial_faces(Ir, Lr, stretch)
    z_faces = _make_axial_faces(Iz, Lz)
    dr_arr = np.diff(r_faces)
    dz_arr = np.diff(z_faces)
    r_centers = 0.5 * (r_faces[:-1] + r_faces[1:])
    z_centers = 0.5 * (z_faces[:-1] + z_faces[1:])

    print(f"  dr range: [{dr_arr.min():.4e}, {dr_arr.max():.4e}] cm")
    print(f"  dz range: [{dz_arr.min():.4e}, {dz_arr.max():.4e}] cm")

    T_floor = 1e-4
    T_init = np.full((Ir, Iz), T_floor)

    for i in range(Ir):
        T_r, _ = zeldovich_self_similar(np.array([max(r_centers[i], 0.0)]), t_start, N=2)
        T_init[i, :] = max(T_r[0], T_floor)

    _, r_front_init = zeldovich_self_similar(np.array([0.0]), t_start, N=2)
    _, r_front_final = zeldovich_self_similar(np.array([0.0]), tfinal, N=2)

    print(f"  Self-similar front at t_start = {r_front_init:.4f} cm")
    print(f"  Self-similar front at tfinal  = {r_front_final:.4f} cm")
    print(f"  T_max(t_start) = {T_init.max():.4f} keV")

    if r_front_final > 0.9 * Lr:
        print(
            f"  WARNING: front at tfinal ({r_front_final:.3f} cm) near "
            f"r_max ({Lr:.2f} cm); consider larger domain."
        )

    EOS, invEOS, sigma_func, scat_func = _make_materials()

    solver = M1Solver2D(
        x1_min=0.0,
        x1_max=Lr,
        n1=Ir,
        x2_min=0.0,
        x2_max=Lz,
        n2=Iz,
        geometry='cylindrical',
        sigma_func=sigma_func,
        scat_func=scat_func,
        EOS=EOS,
        invEOS=invEOS,
        dt=dt_min,
        closure_func=closure_map[closure_name],
        bc_x1_lo='reflect',
        bc_x1_hi='reflect',
        bc_x2_lo='reflect',
        bc_x2_hi='reflect',
        max_nonlinear_iter=max_nonlinear_iter,
        nonlinear_tol=nonlinear_tol,
        relaxation=relaxation,
        x1_faces=r_faces,
        x2_faces=z_faces,
    )

    solver.initialize(T_init=T_floor)
    solver.T[:, :] = T_init
    solver.Er[:, :] = A_RAD * T_init ** 4
    solver.F1[:, :] = 0.0
    solver.F2[:, :] = 0.0

    if output_times is None:
        output_times = np.array([0.3, 0.5, 1.0, 2.0, 3.0, 5.0])
        output_times = output_times[(output_times > t_start) & (output_times <= tfinal)]
    output_times = np.asarray(output_times, dtype=float)

    snapshots = {}
    ts = [float(t_start)]
    ers = [solver.Er.copy()]
    Ts = [solver.T.copy()]

    t = float(t_start)
    dt = float(dt_min)
    out_idx = 0
    step = 0

    while t < tfinal - 1e-14:
        dt_step = min(dt, tfinal - t)
        if out_idx < len(output_times):
            dt_step = min(dt_step, output_times[out_idx] - t)

        solver.dt = max(dt_step, 1e-16)
        solver.step(source=np.zeros((Ir, Iz)), verbose=(LOUD and step % print_stride == 0))

        t += solver.dt
        step += 1

        ts.append(float(t))
        ers.append(solver.Er.copy())
        Ts.append(solver.T.copy())

        if LOUD and step % print_stride == 0:
            print(
                f"  step {step:6d}: t={t:.6e} ns, dt={solver.dt:.3e} ns, "
                f"Tmax={solver.T.max():.4e}, Ermax={solver.Er.max():.4e}"
            )

        while out_idx < len(output_times) and t >= output_times[out_idx] - 1e-12:
            snapshots[float(output_times[out_idx])] = {
                'Er': solver.Er.copy(),
                'F1': solver.F1.copy(),
                'F2': solver.F2.copy(),
                'T': solver.T.copy(),
                'T_rad': solver.T_rad.copy(),
                't_actual': float(t),
            }
            out_idx += 1

        dt = min(dt_max, max(dt_min, dt * dt_increase_factor))

    print(f"\n  Finished: {step} M1 steps")

    return {
        'Ers': ers,
        'Ts': Ts,
        'ts': np.array(ts),
        'snapshots': snapshots,
        'r_centers': r_centers,
        'z_centers': z_centers,
        'r_faces': r_faces,
        'z_faces': z_faces,
        't_start': t_start,
        'tfinal': tfinal,
        'Ir': Ir,
        'Iz': Iz,
        'closure_name': closure_name,
    }


# ===========================================================================
# Diagnostics and plotting
# ===========================================================================

def extract_radial_profile(result, t_target):
    """Extract z-averaged radial T_rad profile at time nearest t_target."""
    ts = result['ts']
    idx = int(np.argmin(np.abs(ts - t_target)))
    t_actual = float(ts[idx])

    Er_snap = result['Ers'][idx]
    Er_r = np.mean(Er_snap, axis=1)
    T_rad = (np.maximum(Er_r, 0.0) / A_RAD) ** 0.25
    return result['r_centers'], T_rad, t_actual


def plot_radial_profiles(result, output_times, savefile='zeldovich_rz_m1_profiles.png'):
    """Plot M1 z-averaged radial T_rad profiles vs self-similar solution."""
    _, r_max = zeldovich_self_similar(np.array([0.0]), max(output_times), N=2)
    r_ref = np.linspace(0.0, max(result['r_centers'][-1], 1.05 * r_max), 300)

    colors = ['C0', 'C1', 'C2', 'C3', 'C4']
    fig, ax = plt.subplots(figsize=(8, 5))

    for k, t_tgt in enumerate(output_times):
        r_num, T_rad, t_act = extract_radial_profile(result, t_tgt)
        col = colors[k % len(colors)]

        ax.plot(r_num, T_rad, color=col, lw=2, label=f"M1 t = {t_act:.2f} ns")

        T_ref, r_front = zeldovich_self_similar(r_ref, t_tgt, N=2)
        ax.plot(r_ref, T_ref, color=col, lw=1.5, ls='--',
                label=f"Self-similar t = {t_tgt:.2f} ns")
        ax.axvline(r_front, color=col, ls=':', lw=1, alpha=0.5)

    ax.set_xlabel('r (cm)')
    ax.set_ylabel(r'Radiation temperature $T_r$ (keV)')
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xlim(0.0, result['r_centers'][-1])
    plt.tight_layout()
    plt.savefig(savefile, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {savefile}")


def plot_rz_heatmap(result, t_target, savefile=None):
    """Plot r-z radiation temperature heatmap at t ~= t_target."""
    ts = result['ts']
    idx = int(np.argmin(np.abs(ts - t_target)))
    t_actual = float(ts[idx])

    Er_snap = result['Ers'][idx]
    T_rad = (np.maximum(Er_snap, 0.0) / A_RAD) ** 0.25

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.pcolormesh(result['r_faces'], result['z_faces'], T_rad.T,
                       shading='flat', cmap='plasma')
    ax.set_xlabel('r (cm)')
    ax.set_ylabel('z (cm)')
    ax.set_title(f'Radiation temperature t = {t_actual:.2f} ns')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label=r'$T_r$ (keV)')
    plt.tight_layout()

    if savefile is None:
        savefile = f"zeldovich_rz_m1_heatmap_t{t_target:.2f}ns.png"
    plt.savefig(savefile, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {savefile}")


def print_z_symmetry(result, t_target):
    """Print max relative z-variation; should remain small for this setup."""
    ts = result['ts']
    idx = int(np.argmin(np.abs(ts - t_target)))
    Er = result['Ers'][idx]

    Er_z_mean = np.mean(Er, axis=1, keepdims=True)
    z_var = np.max(np.abs(Er - Er_z_mean)) / (Er_z_mean.max() + 1e-30)
    print(f"  t={ts[idx]:.3f} ns: max relative z-variation = {z_var:.3e}")


def save_npz(result, savefile='zeldovich_rz_m1.npz'):
    """Save core arrays and snapshots to NPZ."""
    snap_times = np.array(sorted(result['snapshots'].keys()), dtype=float)
    snap_Er = np.array([result['snapshots'][t]['Er'] for t in snap_times])
    snap_T = np.array([result['snapshots'][t]['T'] for t in snap_times])
    snap_Trad = np.array([result['snapshots'][t]['T_rad'] for t in snap_times])

    np.savez(
        savefile,
        ts=result['ts'],
        Er_hist=np.array(result['Ers']),
        T_hist=np.array(result['Ts']),
        snap_times=snap_times,
        snap_Er=snap_Er,
        snap_T=snap_T,
        snap_Trad=snap_Trad,
        r_centers=result['r_centers'],
        z_centers=result['z_centers'],
        r_faces=result['r_faces'],
        z_faces=result['z_faces'],
        t_start=np.array(result['t_start']),
        tfinal=np.array(result['tfinal']),
        Ir=np.array(result['Ir']),
        Iz=np.array(result['Iz']),
        closure_name=np.array(result['closure_name']),
    )
    print(f"  Saved: {savefile}")


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description='2-D r-z Zel\'dovich wave M1')
    parser.add_argument('--Ir', type=int, default=60)
    parser.add_argument('--Iz', type=int, default=8)
    parser.add_argument('--closure', type=str, default='levermore',
                        choices=['levermore', 'kershaw', 'p1', 'minerbo_poly'])
    parser.add_argument('--Lr', type=float, default=1.5, help='r extent (cm)')
    parser.add_argument('--Lz', type=float, default=0.2, help='z extent (cm)')
    parser.add_argument('--t-start', type=float, default=0.1)
    parser.add_argument('--tfinal', type=float, default=1.0)
    parser.add_argument('--dt-min', type=float, default=1e-5)
    parser.add_argument('--dt-max', type=float, default=1e-3)
    parser.add_argument('--dt-increase-factor', type=float, default=1.1)
    parser.add_argument('--stretch', type=float, default=1.5)
    parser.add_argument('--max-nonlinear-iter', type=int, default=20)
    parser.add_argument('--nonlinear-tol', type=float, default=1e-6)
    parser.add_argument('--relaxation', type=float, default=1.0)
    parser.add_argument('--output-times', type=float, nargs='+', default=None)
    parser.add_argument('--prefix', type=str, default='zeldovich_rz_m1')
    parser.add_argument('--loud', action='store_true')
    parser.add_argument('--print-stride', type=int, default=50)
    args = parser.parse_args()

    output_times = args.output_times
    if output_times is None:
        output_times = [t for t in [0.3, 0.5, 1.0, 2.0] if t > args.t_start and t <= args.tfinal]

    result = run_zeldovich_rz_m1(
        Ir=args.Ir,
        Iz=args.Iz,
        closure_name=args.closure,
        Lr=args.Lr,
        Lz=args.Lz,
        t_start=args.t_start,
        tfinal=args.tfinal,
        dt_min=args.dt_min,
        dt_max=args.dt_max,
        dt_increase_factor=args.dt_increase_factor,
        stretch=args.stretch,
        max_nonlinear_iter=args.max_nonlinear_iter,
        nonlinear_tol=args.nonlinear_tol,
        relaxation=args.relaxation,
        output_times=output_times,
        LOUD=args.loud,
        print_stride=args.print_stride,
    )

    print('\nPost-processing...')
    for t in output_times:
        print_z_symmetry(result, t)

    plot_radial_profiles(result, output_times, savefile=f'{args.prefix}_profiles.png')
    for t in output_times:
        plot_rz_heatmap(result, t, savefile=f'{args.prefix}_heatmap_t{t:.2f}ns.png')

    save_npz(result, savefile=f'{args.prefix}_solution_{args.closure}_{args.Ir}x{args.Iz}.npz')


if __name__ == '__main__':
    main()
