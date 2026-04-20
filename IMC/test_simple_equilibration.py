"""Simple test to verify Carter-Forest energy balance for a few steps."""
import IMC1D_CarterForest as imc_cf
import numpy as np

# Setup 0-D problem
Ntarget = 500
L = 0.1
mesh = np.array([[0.0, L]])
Tinit = np.array([0.4])
Trinit = np.array([1.0])
sigma_a_f = lambda T: 100.0 + 0*T  # Match paper: σ = 100 cm⁻¹
cv_val = 0.01
eos = lambda T: cv_val*T
inv_eos = lambda u: u/cv_val
cv = lambda T: cv_val + 0*T
dt = 0.01  # Small time step

# Initialize
state = imc_cf.init_simulation(Ntarget, Tinit, Trinit, mesh, eos, inv_eos)

print(f"\nInitial: T_mat={state.temperature[0]:.3f} keV, T_rad={state.radiation_temperature[0]:.3f} keV")
print(f"Total energy: {state.previous_total_energy:.6f} GJ\n")

# Run 2 steps
for step_num in range(5):
    state, info = imc_cf.step(
        state, Ntarget=Ntarget, Nboundary=0, Nsource=0, NMax=Ntarget,
        T_boundary=(0.0, 0.0), dt=dt, mesh=mesh, sigma_a_func=sigma_a_f,
        inv_eos=inv_eos, cv=cv, source=np.zeros(1), reflect=(True, True),
        use_scalar_intensity_Tr=False,conserve_comb_energy=True
    )
    
    print(f"Step {step_num+1}: T_mat={state.temperature[0]:.3f} keV, T_rad={state.radiation_temperature[0]:.3f} keV")
    print(f"  Particles: {len(state.weights)}, Energy loss: {info['energy_loss']:.6f} GJ")

print(f"\nFinal: T_mat={state.temperature[0]:.3f} keV, T_rad={state.radiation_temperature[0]:.3f} keV")
print(f"Expected equilibrium: ~0.88 keV")
