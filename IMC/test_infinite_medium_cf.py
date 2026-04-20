"""
Infinite medium equilibration test for Carter-Forest IMC.
Matches Densmore & Larsen 2004 paper setup.
"""
import IMC1D_CarterForest as imc_cf  
import numpy as np
import matplotlib.pyplot as plt

# Paper parameters
sigma_a = 100.0  # cm^-1
c_v = 0.01  # GJ/(cm³·keV)
T_mat_init = 0.4  # keV  
T_rad_init = 1.0  # keV
dt = 0.01  # ns
final_time = 0.15  # ns

# Use multiple cells to resolve optical depth
# Target optical depth per cell ~ 1-2
I = 10  # number of cells
L = 0.2 / I  # total length 0.2 cm, so each cell is 0.02 cm
# Optical depth per cell = sigma_a * dx = 100 * 0.02 = 2 (reasonable)

mesh = np.zeros((I, 2))
for i in range(I):
    mesh[i] = [i*L, (i+1)*L]

Tinit = np.zeros(I) + T_mat_init
Trinit = np.zeros(I) + T_rad_init
T_boundary = (0.0, 0.0)
source = np.zeros(I)
sigma_a_f = lambda T: sigma_a + 0*T
eos = lambda T: c_v*T
inv_eos = lambda u: u/c_v
cv = lambda T: c_v + 0*T

# Particle counts
Ntarget = 10000
NMax = 50000

print("="*70)
print(f"Infinite Medium Test (Carter-Forest IMC)")
print("="*70)
print(f"Parameters:")
print(f"  σ_a = {sigma_a} cm⁻¹")
print(f"  C_v = {c_v} GJ/(cm³·keV)")
print(f"  T_mat(0) = {T_mat_init} keV")  
print(f"  T_rad(0) = {T_rad_init} keV")
print(f"  dt = {dt} ns")
print(f"  Cells: {I}, dx = {L} cm, optical depth/cell = {sigma_a*L}")
print(f"  Expected equilibrium: ~0.88 keV")
print("="*70)

times, radiation_temperatures, temperatures = imc_cf.run_simulation(
    Ntarget, 0, 0, NMax, Tinit, Trinit,
    T_boundary, dt, mesh, sigma_a_f,
    eos, inv_eos, cv, source, final_time,
    reflect=(True, True),  # Reflecting boundaries for infinite medium
    use_scalar_intensity_Tr=False
)

# Plot results
plt.figure(figsize=(8, 6))
# Average over all cells for infinite medium
T_mat_avg = np.mean(temperatures, axis=1)
T_rad_avg = np.mean(radiation_temperatures, axis=1)

plt.plot(times, T_mat_avg, 'o-', label='Material (Carter-Forest)', markersize=4)
plt.plot(times, T_rad_avg, 's-', label='Radiation (Carter-Forest)', markersize=4)
plt.axhline(0.88, color='k', linestyle='--', alpha=0.3, label='Expected Equilibrium')
plt.xlabel('Time (ns)')
plt.ylabel('Temperature (keV)')
plt.title(f'Infinite Medium Equilibration\n(σ={sigma_a} cm⁻¹, dt={dt} ns, {I} cells)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0, 1.6)
plt.tight_layout()
plt.savefig('infinite_medium_carter_forest.pdf', dpi=300)
print(f"\nSaved: infinite_medium_carter_forest.pdf")
print(f"Final material T: {T_mat_avg[-1]:.3f} keV")
print(f"Final radiation T: {T_rad_avg[-1]:.3f} keV")
