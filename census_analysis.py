"""Analyse census diagnostics from t=0.90–1.00 ns snapshots."""
import numpy as np, glob, os

__c = 29.98
__a = 0.01372
T_S = 1.0
R_S = 1.0

base = "results/dilute_spectrum_shell/imc_32g_standard"
snaps = sorted(glob.glob(f"{base}/snapshot_t_0.9*.npz")) + \
        [f"{base}/snapshot_t_1.00000ns.npz"]

# ── 1. Quick summary table: per-cell T_r vs free-streaming, first 15 cells ────
print("\n" + "="*110)
print("TABLE 1 – per-cell T_r  (st=state, cen=census-derived, fs=free-streaming)")
print("="*110)

for fpath in snaps:
    tag = os.path.basename(fpath)
    d   = np.load(fpath)
    if 'census_N_per_cell' not in d:
        print(f"\n{tag}: NO CENSUS DATA"); continue

    r_c    = d['r_centers']
    T_rad  = d['T_rad']
    N_c    = d['census_N_per_cell']
    W_c    = d['census_W_per_cell']
    re     = d['r_edges']
    vols   = (4/3)*np.pi*(re[1:]**3 - re[:-1]**3)
    n_tot  = len(d['census_weights'])

    E_den_cen = W_c * __c / vols
    T_r_cen   = np.where(E_den_cen > 0, (E_den_cen / __a)**0.25, 0.0)
    T_r_fs    = T_S * np.sqrt(R_S / (2*r_c))

    print(f"\n--- {tag}  N_total={n_tot:,} ---")
    hdr = f"{'ci':>3} {'r_c':>6} {'N':>6} {'T_r_st':>8} {'T_r_cen':>8} {'T_r_fs':>8} {'st/fs':>7} {'cen/fs':>7}"
    print(hdr)
    for ci in range(min(20, len(r_c))):
        rs = T_rad[ci]/T_r_fs[ci] if T_r_fs[ci]>0 else 0
        rc = T_r_cen[ci]/T_r_fs[ci] if T_r_fs[ci]>0 else 0
        print(f"{ci:>3} {r_c[ci]:>6.3f} {N_c[ci]:>6} "
              f"{T_rad[ci]:>8.4f} {T_r_cen[ci]:>8.4f} {T_r_fs[ci]:>8.4f} "
              f"{rs:>7.3f} {rc:>7.3f}")

# ── 2. mu distribution in innermost 4 cells at t=1.00 ns ─────────────────────
print("\n" + "="*80)
print("TABLE 2 – mu (cos θ) distribution in cells 0–3 at t=1.00000ns")
print("="*80)

d  = np.load(f"{base}/snapshot_t_1.00000ns.npz")
re = d['r_edges']
vols = (4/3)*np.pi*(re[1:]**3 - re[:-1]**3)

if 'census_cell_indices' in d:
    ci_all = d['census_cell_indices'].astype(int)
    mu_all = d['census_mus']
    w_all  = d['census_weights']
    g_all  = d['census_groups'].astype(int)
    n_groups = len(d['energy_edges']) - 1

    print(f"{'cell':>4} {'r_c':>6} {'N':>6} {'mu_min':>8} {'mu_mean':>8} "
          f"{'mu_max':>8} {'frac_mu>0':>10} {'W_fwd/W_tot':>11}")
    for ci in range(min(6, len(d['r_centers']))):
        mask = ci_all == ci
        if mask.sum() == 0:
            print(f"{ci:>4} {d['r_centers'][ci]:>6.3f}  (empty)")
            continue
        mu_c = mu_all[mask]
        w_c  = w_all[mask]
        frac_fwd = w_c[mu_c > 0].sum() / w_c.sum()
        print(f"{ci:>4} {d['r_centers'][ci]:>6.3f} {mask.sum():>6} "
              f"{mu_c.min():>8.3f} {np.average(mu_c,weights=w_c):>8.3f} "
              f"{mu_c.max():>8.3f} {(mu_c>0).mean():>10.3f} {frac_fwd:>11.3f}")

    # ── 3. Energy consistency check ────────────────────────────────────────────
    print("\n" + "="*80)
    print("TABLE 3 – E_rad consistency: state vs census-derived, cells 0–9 at t=1ns")
    print("="*80)
    W_c   = d['census_W_per_cell']
    E_st  = d['E_rad']                       # already energy density (GJ/cm³)
    E_cen = W_c * __c / vols
    T_cen = np.where(E_cen>0, (E_cen/__a)**0.25, 0.0)
    T_st  = d['T_rad']
    r_c   = d['r_centers']
    print(f"{'ci':>3} {'r_c':>6} {'E_st':>12} {'E_cen':>12} {'ratio':>7} {'T_st':>8} {'T_cen':>8}")
    for ci in range(min(15, len(r_c))):
        rat = E_cen[ci]/E_st[ci] if E_st[ci]>0 else float('nan')
        print(f"{ci:>3} {r_c[ci]:>6.3f} {E_st[ci]:>12.4e} {E_cen[ci]:>12.4e} "
              f"{rat:>7.3f} {T_st[ci]:>8.4f} {T_cen[ci]:>8.4f}")
else:
    print("snapshot_t_1.00000ns.npz has no census arrays")
