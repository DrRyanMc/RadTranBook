"""
Diagnostic: per-step energy balance for cells 0-10 in imc_32g_standard.
Loads snapshots from t=0.90 to 1.01 ns and reconstructs:
  - E_cen[cell]  = radiation energy in census (post-comb) per cell
  - T_r_SI[cell] = SI-based radiation temperature
  - T_r_cen[cell]= census-based radiation temperature
  - E_bc          = boundary energy entering per step
  - net outflow from cell 0 to cell 1 per step

Then checks: does the total across cells 0-10 change proportionally with E_total?
"""
import numpy as np
import os

base = "results/dilute_spectrum_shell/imc_32g_standard"
snap_dir = base

# Load all snapshots 0.90 to 1.01 ns
times_list = [0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.00, 1.01]
snaps = {}
for t in times_list:
    fname = os.path.join(snap_dir, f"snapshot_t_{t:.5f}ns.npz")
    if os.path.exists(fname):
        d = np.load(fname)
        snaps[t] = dict(d)

if not snaps:
    print("No snapshots found!")
    raise SystemExit

# From first snapshot, get mesh and volumes
first = snaps[times_list[0]]
r_centers = first['r_centers']
energy_edges = first['energy_edges']
r_edges = first['r_edges']  # (n_cells+1,) cell boundary positions
r_inner_arr = r_edges[:-1]
r_outer_arr = r_edges[1:]
volumes = (4./3.) * np.pi * (r_outer_arr**3 - r_inner_arr**3)

print(f"Cell 0: r=[{r_inner_arr[0]:.4f}, {r_outer_arr[0]:.4f}] cm, V={volumes[0]:.4f} cm^3")
print(f"Cell 1: r=[{r_inner_arr[1]:.4f}, {r_outer_arr[1]:.4f}] cm, V={volumes[1]:.4f} cm^3")
print(f"Cell 2: r=[{r_inner_arr[2]:.4f}, {r_outer_arr[2]:.4f}] cm, V={volumes[2]:.4f} cm^3")
print()

# Physical constants
a_rad = 7.5657e-15 * 1e-40  # radiation constant in GJ/cm^3/keV^4 ... let me use correct units
# In units where E is in GJ and T is in keV:
# E_rad = a * T^4 * V => a = E / (T^4 * V)
# We need to infer a from the data.
# From snapshot: radiation_temperature is T_r_SI, radiation_energy_by_group is census

# Let me just compute E_cen and T_cen from the census data
# radiation_energy_by_group has shape (n_groups, n_cells) and units of energy density GJ/cm^3
# OR it could be total energy per (group, cell)
# Let me check the shapes
for t, d in snaps.items():
    reb = d.get('E_rad_by_group', None)
    if reb is not None:
        print(f"t={t}: E_rad_by_group shape = {reb.shape}")
        break

print()

# Compute per-cell census energy (all groups combined)
# Check if radiation_energy_by_group is energy density or total energy
# From code: rad_flat = bincount(bin_id, weights=weights) -> total energy per (group, cell) bin
# So radiation_energy_by_group[g, cell] = sum of particle weights in that bin = TOTAL ENERGY (GJ)
def get_E_cen(d):
    """Total radiation energy per cell from census (all groups summed), in GJ.
    E_rad in snapshot is GJ/cm^3 (energy density); multiply by volume to get GJ.
    """
    erad = d.get('E_rad', None)
    if erad is None:
        return None
    return erad * volumes  # (n_cells,) in GJ

# Verify units: T_r_cen should match T_r_SI approximately
# T_r_cen = (E_cen / (a * V))^{0.25}
# Try to infer a from the first snapshot at steady state
t_ref = 0.90
d_ref = snaps[t_ref]
T_si = d_ref['T_rad']  # (n_cells,) in keV
E_cen_ref = get_E_cen(d_ref)

# At cell 0: T_r_SI = 0.664 keV (from earlier analysis)
print(f"T_r_SI at t={t_ref} ns: cell 0={T_si[0]:.4f}, cell 1={T_si[1]:.4f}, cell 2={T_si[2]:.4f}")

# Check if radiation_energy_by_group is total energy or energy density
if E_cen_ref is not None:
    print(f"E_cen (all groups) at t={t_ref} ns: cell 0={E_cen_ref[0]:.4e}, cell 1={E_cen_ref[1]:.4e}")
    print(f"(if these are GJ, they should be ~0.028 and ~0.024 GJ)")
    print(f"(if these are GJ/cm^3, they'd be much smaller)")

print()

# -----------------------------------------------------------------------
# Main analysis: per-step energy table
# -----------------------------------------------------------------------
__a = 7.5657e-15 * 6.24151e-12  # erg/cm^3/K^4 → GJ/cm^3/keV^4
# Actually let me use: a in GJ/cm^3/keV^4
# 1 erg = 1e-40 GJ ??? no
# erg = 1e-7 J = 1e-16 GJ
# keV = 1.1605e7 K
# a = 7.5657e-15 erg/cm^3/K^4 = 7.5657e-15 * 1e-7 J / cm^3 / K^4 = 7.5657e-22 J/cm^3/K^4
# In GJ/cm^3/keV^4: a = 7.5657e-22 * 1e-9 J/GJ * (1.1605e7 K/keV)^4
# = 7.5657e-31 GJ/cm^3 / K^4 * 1.1605^4 * 10^28 K^4/keV^4
# = 7.5657e-31 * 1.811e28 = 7.5657 * 1.811e-3 * 1e-31 * 1e28 = ...
# Let me just do this numerically:
keV_to_K = 1.1605e7  # K/keV
a_SI = 7.5657e-15  # erg/cm^3/K^4
a_in_GJ_keV = a_SI * 1e-7 / 1e9 * keV_to_K**4  # J/cm^3/K^4 → GJ/cm^3/keV^4 * (K/keV)^4
print(f"a_in_GJ_keV = {a_in_GJ_keV:.6e} GJ/cm^3/keV^4")

# From T_r_SI and E_cen, infer units of radiation_energy_by_group:
# If units are GJ: E_cen[0] / (a * V[0]) = T_r^4 => T_r = (E_cen[0] / (a * V[0]))^0.25
if E_cen_ref is not None:
    T_r_cen_from_GJ = (E_cen_ref[0] / (a_in_GJ_keV * volumes[0])) ** 0.25
    print(f"T_r_cen at cell 0 (assuming E in GJ): {T_r_cen_from_GJ:.4f} keV")
    print(f"T_r_SI at cell 0: {T_si[0]:.4f} keV (for comparison)")
    # If T_r_cen_from_GJ matches T_r_SI, E is in GJ. If not, try GJ/cm^3.
    T_r_cen_from_GJcm3 = (E_cen_ref[0] / a_in_GJ_keV) ** 0.25
    print(f"T_r_cen at cell 0 (assuming E in GJ/cm^3): {T_r_cen_from_GJcm3:.4f} keV")

print()

# -----------------------------------------------------------------------
# Compute energy balance table
# -----------------------------------------------------------------------
# Physical constants
c_light = 29.979  # cm/ns
R_S = 1.0  # cm
T_S = 1.0  # keV (inner boundary temperature)
dt = 0.01  # ns (step size)
E_bc_inner = a_in_GJ_keV * c_light * T_S**4 / 4.0 * 4.0 * np.pi * R_S**2 * dt
print(f"E_bc_inner per step = {E_bc_inner:.6e} GJ")
print()

print("Step-by-step energy table for inner cells (t=0.90 to 1.01 ns)")
print("E_cen = total radiation energy (census, post-comb) for cells 0-5")
print("Outflow_c0 = E_bc + E_cen_c0_prev - E_cen_c0 (net outflow from cell 0 per step)")
print()
print(f"{'t(ns)':>7}  {'E_tot':>12}  {'E_c0':>12}  {'E_c1':>12}  {'E_c2':>12}  {'Outflow_c0':>12}  {'T_r_SI_c0':>10}  {'T_r_cen_c0':>10}")
print("-" * 100)

E_cen_prev = None
t_prev = None
for t in times_list:
    if t not in snaps:
        continue
    d = snaps[t]
    E_cen = get_E_cen(d)
    T_si = d['T_rad']
    T_mat = d['T_mat']
    
    if E_cen is None:
        continue
    
    # Determine if E is in GJ (total) or GJ/cm^3 (density)
    # From earlier: T_r_cen_from_GJ should match T_r_SI if E is in GJ
    # E_cen is total (not density) since it's bincount of weights
    E_tot = np.sum(E_cen)
    
    # Census-based T_r at each cell
    # T_r_cen = (E_cen / (a * V))^0.25
    T_r_cen = (np.maximum(E_cen, 0) / np.maximum(a_in_GJ_keV * volumes, 1e-300)) ** 0.25
    
    if E_cen_prev is not None:
        outflow_c0 = E_bc_inner + E_cen_prev[0] - E_cen[0]
    else:
        outflow_c0 = float('nan')
    
    print(f"{t:>7.2f}  {E_tot:>12.5e}  {E_cen[0]:>12.5e}  {E_cen[1]:>12.5e}  "
          f"{E_cen[2]:>12.5e}  {outflow_c0:>12.5e}  {T_si[0]:>10.4f}  {T_r_cen[0]:>10.4f}")
    
    E_cen_prev = E_cen.copy()
    t_prev = t

print()

# -----------------------------------------------------------------------
# Detailed analysis of the anomaly: where does energy go?
# -----------------------------------------------------------------------
print("Detailed cell-by-cell energy flow (cells 0-10):")
print(f"{'t(ns)':>7}", end="")
for c in range(8):
    print(f"  {'E_c'+str(c):>12}", end="")
print()
print("-" * 110)

for t in times_list:
    if t not in snaps:
        continue
    d = snaps[t]
    E_cen = get_E_cen(d)
    if E_cen is None:
        continue
    print(f"{t:>7.2f}", end="")
    for c in range(8):
        if c < len(E_cen):
            print(f"  {E_cen[c]:>12.4e}", end="")
    print()

print()

# -----------------------------------------------------------------------
# Check if cell 0 + cell 1 + ... is conserved (modulo E_bc_inner inflow)
# -----------------------------------------------------------------------
print("Check energy conservation for cells 0-10 combined:")
print(f"{'t(ns)':>7}  {'E_0to10':>14}  {'dE_from_BC_if_zero_cross':>25}")
print("-" * 50)

E_inner_prev = None
for t in times_list:
    if t not in snaps:
        continue
    d = snaps[t]
    E_cen = get_E_cen(d)
    if E_cen is None:
        continue
    E_inner = np.sum(E_cen[:11])
    if E_inner_prev is not None:
        delta = E_inner - E_inner_prev
        # This delta should be E_bc_inner (entering) minus outflow to cell 11
        print(f"{t:>7.2f}  {E_inner:>14.6e}  {delta:>14.6e} (expected ~{E_bc_inner:.4e} if no outflow)")
    else:
        print(f"{t:>7.2f}  {E_inner:>14.6e}")
    E_inner_prev = E_inner

print()

# -----------------------------------------------------------------------
# SI estimator vs census comparison at key cells
# -----------------------------------------------------------------------
print("SI vs Census comparison for cells 0-2:")
print(f"{'t(ns)':>7}  {'T_SI_c0':>10}  {'T_cen_c0':>10}  {'T_SI_c1':>10}  {'T_cen_c1':>10}  "
      f"{'T_SI_c2':>10}  {'T_cen_c2':>10}  {'ratio_SI/cen_c0':>15}")
print("-" * 100)
for t in times_list:
    if t not in snaps:
        continue
    d = snaps[t]
    E_cen = get_E_cen(d)
    T_si = d['T_rad']
    if E_cen is None:
        continue
    T_r_cen = (np.maximum(E_cen, 0) / np.maximum(a_in_GJ_keV * volumes, 1e-300)) ** 0.25
    # SI^4 / cen^4 ratio
    ratio = (T_si[0] / T_r_cen[0]) ** 4 if T_r_cen[0] > 0 else float('nan')
    print(f"{t:>7.2f}  {T_si[0]:>10.4f}  {T_r_cen[0]:>10.4f}  {T_si[1]:>10.4f}  "
          f"{T_r_cen[1]:>10.4f}  {T_si[2]:>10.4f}  {T_r_cen[2]:>10.4f}  {ratio:>15.4f}")
