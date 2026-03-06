# Marshak Wave Steady-State Diagnostic Report

## Problem Description

Single-zone Marshak wave problem with:
- Domain: [0, 0.001] cm (1 mm slab)
- Single cell (n_cells = 1)
- Boundary temperature: T_bc = 0.05 keV (constant)
- Initial temperature: T_init = 0.01 keV
- Expected behavior: T → T_bc at steady state

## Observed Behavior

**Material temperature EXCEEDS boundary temperature:**
- Initial: T = 0.01 keV
- After 20 steps (0.19 ns): T = 0.0687 keV
- **T/T_bc = 1.375 (37% higher than boundary!)**

## Key Findings

### 1. Radiation vs. Material Temperature

At step 20:
- Material temperature: T = 0.0687 keV
- Radiation temperature: T_rad = 0.0498 keV ≈ T_bc
- **T_rad/T = 0.724** (radiation and material are NOT in equilibrium)

The radiation field is nearly equilibrated with the boundary (T_rad ≈ T_bc), but the material is much hotter.

### 2. Massive Absorption Rate

Initial state (t=0):
```
Total absorption: Σ c*σ_a*φ = 29.9 GJ/(cm³·ns)
Total emission: 4π*a*c*T^4 = 5.17e-8 GJ/(cm³·ns)
Net heating rate: dT/dt = 598 keV/ns
```

**Absorption is ~10^9 times larger than emission!**

### 3. Energy Balance Violation

Expected power input from boundary:
```
F_in = 1.29e-6 GJ/(cm²·ns)
Δx = 1e-3 cm
P_in ≈ F_in / Δx ≈ 1.3e-3 GJ/(cm³·ns)
```

But measured absorption is 29.9 GJ/(cm³·ns), which is **23,000 times larger** than what's entering from the boundary!

### 4. Boundary Condition Issue

The Marshak BC is: `A*φ + B*dφ/dr = C`
where `A = 0.5`, `B = D_g`, `C = F_in,g`

For a single zone with Δx = 0.001 cm:
```
Group 0: D = 1.5e-13 cm << Δx  ✓ (well-resolved)
Group 1: D = 1.0e-10 cm << Δx  ✓ (well-resolved)
Group 2: D = 6.8e-8 cm << Δx   ✓ (well-resolved)
Group 3: D = 4.5e-5 cm < Δx    ⚠️ (marginally resolved)
Group 4: D = 0.028 cm >> Δx    ❌ (UNDER-RESOLVED!)
```

**Problem:** Group 4 has D/Δx = 28, meaning the diffusion length is 28 times larger than the cell size!

### 5. Radiation Energy Density Mismatch

At step 20:
```
E_r (actual)           = 8.44e-8 GJ/cm³
E_r (expected from T)  = a*T^4 = 3.06e-7 GJ/cm³
E_r (expected from BC) = a*T_bc^4 = 8.58e-8 GJ/cm³
```

The radiation energy is consistent with T_bc (as expected), but the material temperature is much higher.

## Root Cause Analysis

The problem appears to be a **combination of issues**:

1. **Spatial Resolution:** The single-zone setup cannot properly resolve the diffusion length for low-opacity (high-energy) groups.

2. **BC Formulation:** The Marshak BC assumes `dφ/dr = (φ_boundary - φ_cell)/Δx`, but for D >> Δx, this creates unphysical gradients.

3. **Radiation-Material Coupling:** The solver is correctly maintaining the radiation field at T_rad ≈ T_bc, but the material is absorbing energy faster than it can radiate, causing T > T_bc.

4. **Implicit Time Discretization:** The backward Euler method may be creating spurious energy sources when D >> Δx.

## Recommendations

1. **Increase spatial resolution:** Use n_cells > 1 to properly resolve diffusion lengths
   - Criterion: Δx < D/10 for all groups
   - For D_max ≈ 0.03 cm, need Δx < 0.003 cm → n_cells > 0.33

2. **Check Marshak BC implementation:** For single-zone diffusion-thick cells, may need asymptotic limit of BC

3. **Verify energy conservation:** Add explicit energy conservation check (radiation + material)

4. **Test with gray (single-group) opacity:** Simpler case to isolate spatial discretization from multigroup coupling

5. **Compare with analytical solution:** For gray diffusion in planar geometry, exact steady-state solution exists

## Next Steps

1. Run with n_cells = 10-100 to see if T → T_bc properly
2. Check total energy conservation explicitly
3. Verify Marshak BC implementation for diffusion-thick cells
4. Add energy flux diagnostics at boundaries
