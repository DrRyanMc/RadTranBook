# Su-Olson Picket Fence IMC Fixes

## Issues Found

### 1. **Different Reference Data Tables**
- **IMC version** (`test_su_olson_picket_fence.py`): Uses Table 2 data
- **Diffusion version** (`su_olson_picket_fence_flux_limiters.py`): Uses Table 4 data
- These tables have **different values** and represent different solution types
- Example: U1 at x=0, τ=0.1:
  - Table 2: 0.03873
  - Table 4: 0.04956

### 2. **Initial Conditions Mismatch**
- **IMC version**: `T_init = 0.01 keV`
- **Diffusion version**: `T_init = 0.001 keV`
- ✅ **FIXED**: Changed IMC to use `T_init = 0.001 keV`

### 3. **Emission Fractions - CRITICAL ISSUE**

The Su-Olson picket fence problem requires **equal Planck functions**, meaning:
- `B_g = 0.5 * (a*c*T⁴)/(4π)` for BOTH groups
- This translates to `b_g = 0.5` emission fraction for each group (constant!)

**Problem**: MG_IMC was computing `b_star` using actual Planck integrals `Bg()` over energy groups [0.1, 1.0, 10.0] keV, then normalizing. This gives temperature-dependent emission fractions, NOT the constant 0.5/0.5 required.

**Solution**: Added `emission_fractions` parameter to allow overriding Planck integrals:

```python
# In MG_IMC2D.py
def run_simulation(..., emission_fractions=None):
    # If emission_fractions provided, use them instead of Planck integrals
    
# In test file
emission_fractions = np.array([0.5, 0.5])  # Equal emission
```

### 4. **Radiation Energy Estimation**
- Changed from `use_scalar_intensity_Tr=True` to `use_scalar_intensity_Tr=False`
- Particle binning gives better accuracy for this benchmark

## Changes Made

### MG_IMC2D.py
1. Added `emission_fractions` parameter to `step()` function
2. Added `emission_fractions` parameter to `run_simulation()` function
3. Modified emission logic to use custom fractions when provided:
   ```python
   if emission_fractions is not None:
       # Use custom emission fractions (e.g., for picket fence)
       b_star = emission_fractions
   else:
       # Use Planck integrals (default behavior)
       b_star = Bg(...) normalized
   ```

### test_su_olson_picket_fence.py
1. Changed `T_init` from 0.01 to 0.001 keV
2. Added `emission_fractions = np.array([0.5, 0.5])`
3. Changed `use_scalar_intensity_Tr` from True to False
4. Updated docstring to clarify emission fractions

## Expected Result

With these fixes, the IMC version should now:
- Use the same initial conditions as diffusion
- Use equal emission fractions (b_g = 0.5) as required
- Compare against Table 2 reference data
- Give results on the same scale as the reference

## Notes

- Table 2 vs Table 4: Make sure you're comparing against the correct reference table
- The `emission_fractions` override is specifically for problems like picket fence where Planck functions are defined to be equal across groups
- For most problems, leave `emission_fractions=None` to use physical Planck integrals
