#!/usr/bin/env python3
"""
Verify the dt usage in BDF2 coefficients and matrix assembly
"""

import numpy as np

gamma = 2.0 - np.sqrt(2.0)
original_dt = 0.001

print("="*70)
print("TR-BDF2 Time Step Structure")
print("="*70)
print()
print(f"gamma = {gamma:.6f}")
print(f"Original full time step: dt = {original_dt}")
print()
print("Time levels:")
print(f"  t^n = 0")
print(f"  t^{{n+gamma}} = {gamma * original_dt:.6f} (after TR stage)")
print(f"  t^{{n+1}} = {original_dt:.6f} (after BDF2 stage)")
print()
print("Sub-steps:")
print(f"  TR stage: dt_TR = gamma * dt = {gamma * original_dt:.6f}")
print(f"  BDF2 stage: from t^n to t^{{n+1}}, full dt = {original_dt:.6f}")
print()

# BDF2 coefficients
c_0 = (2-gamma)/(1-gamma)
c_1 = (1-gamma)/gamma
c_2 = -1/(gamma*(1-gamma))

print("="*70)
print("BDF2 Formula Analysis")
print("="*70)
print()
print("The BDF2 formula for non-uniform steps is:")
print("  (1/Δt)[c_0*u^{n+1} + c_1*u^n + c_2*u^{n+γ}] = L(Er^{n+1})")
print()
print(f"Coefficients:")
print(f"  c_0 = {c_0:.6f}")
print(f"  c_1 = {c_1:.6f}")
print(f"  c_2 = {c_2:.6f}")
print(f"  Sum: {c_0 + c_1 + c_2:.10f} (should be 0 for consistency)")
print()
print("Question: What is Δt in this formula?")
print()
print("Option 1: Δt = full step from t^n to t^{n+1}")
print(f"  Δt = {original_dt}")
print(f"  c_0/Δt = {c_0/original_dt:.3f}")
print()
print("Option 2: Δt = sub-step from t^{n+γ} to t^{n+1}")
print(f"  Δt = (1-γ)*dt = {(1-gamma)*original_dt:.6f}")
print(f"  c_0/Δt = {c_0/((1-gamma)*original_dt):.3f}")
print()

# Check which makes sense dimensionally
print("="*70)
print("Dimensional Analysis")
print("="*70)
print()
print("For a constant solution u = u_0:")
print(f"  LHS: (1/Δt)[c_0 + c_1 + c_2]*u_0 = (1/Δt)*0*u_0 = 0 ✓")
print(f"  RHS: L(Er_0) = 0 for constant ✓")
print()
print("For linear in time: u(t) = u_0 + u_1*t")
print("  At t^n: u^n = u_0")
print(f"  At t^{{n+γ}}: u^{{n+γ}} = u_0 + u_1*{gamma*original_dt:.6f}")
print(f"  At t^{{n+1}}: u^{{n+1}} = u_0 + u_1*{original_dt:.6f}")
print()
print("  If Δt = full step:")
time_deriv_full = (c_0 * (original_dt) + c_1 * 0 + c_2 * (gamma * original_dt)) / original_dt
print(f"    (1/Δt)[c_0*u^{{n+1}} + c_1*u^n + c_2*u^{{n+γ}}]")
print(f"      = (1/{original_dt})[{c_0:.3f}*{original_dt:.6f} + {c_1:.3f}*0 + {c_2:.3f}*{gamma*original_dt:.6f}]")
print(f"      = {time_deriv_full:.6f}")
print(f"    True derivative: u_1 = 1.0 (for u_1*t with u_1=1)")
print(f"    Ratio: {time_deriv_full:.6f} {'✓ Correct!' if abs(time_deriv_full - 1.0) < 0.01 else '✗ Wrong!'}")
print()
print("  If Δt = (1-γ)*dt:")
time_deriv_sub = (c_0 * (original_dt) + c_1 * 0 + c_2 * (gamma * original_dt)) / ((1-gamma)*original_dt)
print(f"    (1/Δt)[c_0*u^{{n+1}} + c_1*u^n + c_2*u^{{n+γ}}]")
print(f"      = (1/{(1-gamma)*original_dt:.6f})[{c_0:.3f}*{original_dt:.6f} + {c_1:.3f}*0 + {c_2:.3f}*{gamma*original_dt:.6f}]")
print(f"      = {time_deriv_sub:.6f}")
print(f"    Ratio: {time_deriv_sub:.6f} {'✓ Correct!' if abs(time_deriv_sub - 1.0) < 0.01 else '✗ Wrong!'}")

print()
print("="*70)
print("CONCLUSION")
print("="*70)
print()
if abs(time_deriv_full - 1.0) < 0.01:
    print("The formula uses Δt = FULL time step from t^n to t^{n+1}")
    print("This is what we're currently implementing.")
elif abs(time_deriv_sub - 1.0) < 0.01:
    print("The formula uses Δt = SUB-STEP (1-γ)*dt from t^{n+γ} to t^{n+1}")
    print("We should divide by (1-γ)*dt, not the full dt!")
else:
    print("Neither interpretation gives the correct time derivative!")
    print("There might be an error in the coefficient derivation.")
