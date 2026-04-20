import IMC1D_CarterForest as imc_cf
import numpy as np

# Setup minimal 0-D problem
Ntarget = 20 #10000
L = 0.1
I = 1
mesh = np.array([[0.0, L]])
Tinit = np.array([0.4])
Trinit = np.array([1.0])
sigma_a_f = lambda T: 1.0 + 0*T
cv_val = 0.01
eos = lambda T: cv_val*T
inv_eos = lambda u: u/cv_val
cv = lambda T: cv_val + 0*T
dt = 0.025

# Initialize
state = imc_cf.init_simulation(Ntarget, Tinit, Trinit, mesh, eos, inv_eos)

print("Initial conditions:")
print(f"  T_mat = {state.temperature[0]:.4f} keV")
print(f"  T_rad = {state.radiation_temperature[0]:.4f} keV")
print(f"  Internal energy = {state.internal_energy[0]:.6f} GJ/cm³")
print(f" Radiation energy = {np.sum(state.weights):.6f} GJ")
print(f"  Total energy = {state.previous_total_energy:.6f} GJ")
print(f"  Number of particles = {len(state.weights)}")

# Compute expected emission
sigma_a = sigma_a_f(state.temperature[0])
beta = 4.0 * imc_cf.__a * state.temperature[0]**3 / cv_val
expected_emission = imc_cf.__a * imc_cf.__c * sigma_a * state.temperature[0]**4 * dt * L
print(f"\nExpected emission:")
print(f"  beta = {beta:.4f}")
print(f"  Mean re-emission time = {1/(imc_cf.__c * sigma_a * beta):.4f} ns")
print(f"  Expected emitted energy = {expected_emission:.6f} GJ")

# Take one step
print(f"\nBefore transport:")
print(f"  N particles to transport = {len(state.weights)}")
print(f"  Total weight = {np.sum(state.weights):.6f} GJ")

state, info = imc_cf.step(
    state,
    Ntarget=Ntarget,
    Nboundary=0,
    Nsource=0,
    NMax=Ntarget,
    T_boundary=(0.0, 0.0),
    dt=dt,
    mesh=mesh,
    sigma_a_func=sigma_a_f,
    inv_eos=inv_eos,
    cv=cv,
    source=np.zeros(I),
    reflect=(True, True),
    use_scalar_intensity_Tr=False
)

print(f"\nAfter one step:")
print(f"  T_mat = {state.temperature[0]:.4f} keV")
print(f"  T_rad = {state.radiation_temperature[0]:.4f} keV")
print(f"  Internal energy = {state.internal_energy[0]:.6f} GJ/cm³")
print(f"  Radiation energy = {np.sum(state.weights):.6f} GJ")
print(f"  Total energy = {state.previous_total_energy:.6f} GJ")
print(f"  Number of particles = {len(state.weights)}")
print(f"  Energy loss = {info['energy_loss']:.6f} GJ")

# Compute expected deterministic change
Tr = Trinit[0]
T_mat = Tinit[0]
rate = imc_cf.__a * imc_cf.__c * sigma_a * (Tr**4 - T_mat**4)
expected_du = rate * dt
print(f"\nExpected deterministic change:")
print(f"  Rate = {rate:.6f} GJ/(cm³·ns)")
print(f"  du = {expected_du:.6f} GJ/cm³")
print(f"  Expected final u = {eos(Tinit[0]) + expected_du:.6f} GJ/cm³")
print(f"  Expected final T_mat = {inv_eos(eos(Tinit[0]) + expected_du):.4f} keV")

# Actual change
actual_du = state.internal_energy[0] - eos(Tinit[0])
print(f"\nActual change:")
print(f"  du = {actual_du:.6f} GJ/cm³")
print(f"  Error = {(actual_du - expected_du)/expected_du * 100:.2f}%")
