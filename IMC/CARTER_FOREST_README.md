# Carter-Forest IMC Implementation

## Overview

This implementation provides the Carter-Forest (time-delayed re-emission) method for Implicit Monte Carlo, as described in Box 9.3 of the textbook. Unlike the standard Fleck-Cummings IMC which uses effective scattering, Carter-Forest IMC models physical absorption followed by time-delayed re-emission.

## Files

- **`IMC1D_CarterForest.py`**: Complete implementation of Carter-Forest IMC for 1D slab geometry
- **`compare_imc_methods.py`**: Comparison script that runs both Fleck-Cummings and Carter-Forest methods side-by-side

## Key Differences from Fleck-Cummings IMC

### Fleck-Cummings (IMC1D.py)
- Uses Fleck factor: `f = 1/(1 + θβσ_a c dt)`
- Effective scattering: `σ_s = σ_a(1-f)`, `σ_a_eff = σ_a f`
- Pseudo-scattering replaces absorption-re-emission
- Temporal discretization error: O(dt)

### Carter-Forest (IMC1D_CarterForest.py)
- **No Fleck factor**: Uses true absorption cross section σ_a
- **True absorption**: Particles are actually absorbed (not scattered)
- **Time-delayed re-emission**: Sample re-emission time from exponential distribution:
  ```
  t_emit = t_absorbed - (1/(c·σ_a·β)) · ln(ξ)
  ```
  where β = 4aT³/C_v (material coupling parameter)
  
- **Conditional re-emission**:
  - If `t_emit < t_end`: particle re-emitted at t_emit with isotropic direction
  - If `t_emit >= t_end`: energy deposited in material, no re-emission in this step
  
- **Exact in time** for linearized equations (no temporal discretization error)
- Still has linearization error from freezing material properties

## Usage

### Basic simulation with Carter-Forest:

```python
import IMC1D_CarterForest as imc_cf

# Setup problem
mesh = np.array([[0.0, 0.1], [0.1, 0.2], ...])
Tinit = np.array([0.01, 0.01, ...])
Trinit = np.array([0.01, 0.01, ...])

# Material properties
sigma_a_func = lambda T: 200.0 + 0*T  # cm^-1
cv = lambda T: 0.1 + 0*T  # GJ/(cm³·keV)
eos = lambda T: cv(T) * T
inv_eos = lambda u: u / cv(0.0)

# Run simulation
time, rad_temp, mat_temp = imc_cf.run_simulation(
    Ntarget=20000,
    Nboundary=10000,
    Nsource=0,
    NMax=100000,
    Tinit=Tinit,
    Tr_init=Trinit,
    T_boundary=(1.0, 0.0),  # Left/right boundaries
    dt=0.1,
    mesh=mesh,
    sigma_a_func=sigma_a_func,
    eos=eos,
    inv_eos=inv_eos,
    cv=cv,
    source=np.zeros(len(mesh)),
    final_time=2.0,
    reflect=(False, False)
)
```

### Running the comparison:

```bash
python compare_imc_methods.py
```

This will:
1. Run the same Marshak wave problem with both methods
2. Generate comparison plots showing:
   - Material temperature profiles
   - Radiation temperature profiles
   - Time history at a fiducial point
3. Print summary statistics showing differences between methods

## Implementation Details

### Key Changes from Fleck-Cummings

#### 1. Transport (move_particle_cf)
- No scattering events (σ_s = 0)
- When absorption occurs:
  - Sample re-emission time using exponential distribution with rate c·σ_a·β
  - Check if re-emission occurs within remaining time step
  - If yes: continue particle with new direction at t_emit
  - If no: deposit energy, terminate particle

#### 2. Cross Sections
- No Fleck factor calculation
- Use true absorption: `sigma_a = sigma_a_func(T)` (not modified)
- Material coupling: `beta = 4*a*T³/C_v` (held fixed during time step)

#### 3. Track-Length Estimator
- Still used for radiation temperature estimation
- Accumulated even for particles that get absorbed

## Physics

### Material Coupling Parameter β

The parameter β = 4aT³/C_v characterizes the strength of radiation-material coupling:
- **High β** (strong coupling): Short re-emission times, rapid equilibration
- **Low β** (weak coupling): Long re-emission times, slow equilibration

### Re-emission Time Distribution

The exponential sampling gives the probability density:
```
p(t|t') = c·σ_a·β·exp(-c·σ_a·β·(t-t'))
```
where t' is the absorption time and t is the re-emission time.

This is exact for the linearized system where T and σ_a are held fixed during the time step.

## Advantages and Disadvantages

### Advantages
- **Exact in time** for linearized equations (no Δt discretization error)
- **Physical interpretation**: Models actual absorption and delayed re-emission
- **Better for large time steps**: No temporal discretization error accumulation

### Disadvantages
- **Still has linearization error**: Material properties frozen during time step
- **More complex**: Need to track particle times and re-emission events
- **Potentially more expensive**: May need more events per particle

## Testing

The smoke test in `IMC1D_CarterForest.py` runs an equilibrium relaxation test:
```bash
python IMC1D_CarterForest.py
```

Expected behavior:
- Material equilibrates to near-zero temperature (no sources)
- Radiation cools as it transfers energy to material
- Energy approximately conserved
- No errors or crashes

## References

See textbook Box 9.3: "Carter–Forest Monte Carlo (time-delayed re-emission)"

Equations (9.86)-(9.92) derive the method and show:
- The probability density p(t|t') for re-emission
- How the method removes temporal discretization error
- The physical interpretation as delayed re-emission
