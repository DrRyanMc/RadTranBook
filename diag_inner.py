import numpy as np, glob

d = 'results/dilute_spectrum_shell/imc_32g_standard'
files = sorted(glob.glob(f'{d}/snapshot_t_0.9*.npz'))

a, c_light = 1.459e-2, 30.0
dt = 0.01
E_bc_inner = a * c_light * 1.0**4 / 4.0 * 4.0 * np.pi * 1.0**2 * dt

print(f"E_bc_inner per step = {E_bc_inner:.4e} GJ")
print()
hdr = f"{'t(ns)':>7}  {'T_r[1.25]':>10}  {'T_r[1.75]':>10}  {'mono?':>6}  {'T_mat_shellmax':>15}  {'E_emit_approx':>14}  {'N_inner_est':>12}"
print(hdr)
print("-" * len(hdr))

Ntarget = 300_000
fmin = 0.05
N_floor = max(1, int(round(fmin * Ntarget)))

for f in files:
    data = np.load(f)
    t = float(data['time'])
    r = data['r_centers']
    Tr = data['T_rad']
    Tm = data['T_mat']
    edges = data['r_edges']
    vols = (4.0/3.0)*np.pi*(edges[1:]**3 - edges[:-1]**3)

    mono = 'Y' if Tr[0] >= Tr[1] else 'N'

    shell_mask = r >= 25.0
    T_shell_max = np.max(Tm[shell_mask]) if np.any(shell_mask) else np.nan

    # Rough proxy for E_emit (ignores Fleck factor; uses raw a*c*T^4 opacity weighting)
    # Real code uses sigma_a_fleck*b_star, but let's use T^4 * volume as proxy
    # More importantly: zero where T < T_emit_floor = 0.025 keV
    hot = Tm > 0.025
    E_emit_approx = float(np.sum(hot * a * c_light * Tm**4 * vols * dt))

    E_all = E_bc_inner + E_emit_approx
    if E_all > 0.0:
        N_inner_est = max(int(round(Ntarget * E_bc_inner / E_all)), N_floor)
    else:
        N_inner_est = Ntarget

    print(f"{t:7.3f}  {Tr[0]:10.4f}  {Tr[1]:10.4f}  {mono:>6}  {T_shell_max:15.5f}  {E_emit_approx:14.4e}  {N_inner_est:12d}")
