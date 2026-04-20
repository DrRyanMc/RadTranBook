# Marshak Wave Test - Multigroup IMC

## Overview

This test implements the classic Marshak wave problem using multigroup IMC with **all group opacities set to identical values**. This is a crucial validation test because when all groups have the same opacity, the multigroup solution should exactly reproduce the gray (single-group) solution.

## Test File

`test_marshak_wave_multigroup.py` - Multigroup version with 5 energy groups, all having σ(T) = 300*T^(-3)

## Physical Setup

- **Problem**: Semi-infinite medium with temperature-dependent opacity
- **Boundary**: T = 1.0 keV at x=0 (left boundary)
- **Domain**: [0, 0.2] cm with 50 cells
- **Material**: 
  - Energy: e(T) = c_v*T with c_v = 0.3 GJ/(g·keV)
  - Density: ρ = 1.0 g/cm³
- **Opacity**: σ(T) = 300*T^(-3) for **all 5 groups** (identical)
- **Initial conditions**: Cold material at T = 10^(-4) keV

## Energy Groups

5 groups with boundaries: [0.1, 0.5, 1.0, 2.0, 5.0, 10.0] keV

**All groups have the same opacity function**, so this should behave exactly like the gray problem.

## Self-Similar Solution

The Marshak wave has an analytical self-similar solution:

```
T(x,t) / T_bc = [(1 - ξ/ξ_max)(1 + ωξ/ξ_max)]^(1/6)
```

where:
- ξ = x / sqrt(K*t)
- K = 8acT³/(3ρc_vσ(T))
- ξ_max = 1.11305
- ω = 0.05989

The wave front propagates as x ~ sqrt(t).

## Expected Results

1. **Temperature profiles** at t = 1, 5, and 10 ns
2. **Comparison with self-similar solution** - should show excellent agreement
3. **Group-by-group energy distribution** - should be similar across all groups
4. **Wave front propagation** - should match sqrt(t) scaling

## Running the Test

```bash
cd /path/to/MG_IMC
python3 test_marshak_wave_multigroup.py
```

**Note**: If you encounter Numba cache errors (ModuleNotFoundError: No module named 'MG_IMC'), clear the cache:

```bash
find __pycache__ -name "*.nbc" -delete
find __pycache__ -name "*.nbi" -delete
```

Or just delete the `__pycache__` directory.

## Outputs

- `test_marshak_wave_multigroup.png` - Temperature profiles vs self-similar solution
- `test_marshak_wave_multigroup_comparison.png` - Detailed comparison and relative error
- `marshak_wave_multigroup_output_10000ps_100000.npz` - Numerical data for post-processing

## Comparison with Gray Version

This test should be directly compared with `IMC/MarshakWave.py` from the gray IMC implementation. The results should be nearly identical, validating that:

1. Multigroup sampling algorithms work correctly
2. Energy coupling between groups is handled properly  
3. The code reduces to gray when all opacities are equal
4. Group-dependent transport produces correct behavior

## Validation Criteria

✓ Material and radiation temperatures match self-similar solution within ~5%  
✓ Wave front position agrees with analytical prediction  
✓ Energy conservation maintained throughout simulation  
✓ Results comparable to gray IMC implementation  

## Particle Counts

- Material emission: 100,000 particles/step
- Boundary source: 100,000 particles/step  
- Maximum census: 400,000 particles
- Time step: 0.01 ns

Total simulation time: ~2-3 minutes depending on hardware.
