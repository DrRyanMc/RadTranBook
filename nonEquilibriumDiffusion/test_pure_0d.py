"""
Test pure 0-D equations WITHOUT boundary condition complications
We'll manually compute the system and compare
"""
import numpy as np

# Physical constants
C_LIGHT = 2.998e1  # cm/ns
A_RAD = 0.01372  # GJ/cm³/keV⁴
RHO = 1.0  # g/cm³

def test_pure_0d(theta, dt, T_n, phi_n, sigma_P, C_v):
    """
    Pure 0-D equations (no spatial terms):
    
    φ equation: (φ^{n+1} - φ^n)/(c·Δt) = σ_P·f·(acT★⁴ - φ̃) - (1-f)·Δe/Δt
    T equation: (e(T_{n+1}) - e(T_n))/Δt = f·σ_P(φ̃ - acT★⁴) + (1-f)·Δe/Δt
    
    where φ̃ = θφ^{n+1} + (1-θ)φ^n
    """
    print("="*70)
    print(f"Pure 0-D Test: θ = {theta}, Δt = {dt} ns")
    print("="*70)
    
    # For 1 Newton iteration: T★ = T^n
    T_star = T_n
    e_n = C_v * T_n
    e_star = C_v * T_star
    Delta_e = e_star - e_n  # = 0 for 1 iteration
    
    # Compute f and other parameters
    beta = (4.0 * A_RAD * T_star**3) / C_v
    f = 1.0 / (1.0 + beta * sigma_P * C_LIGHT * theta * dt)
    acT4_star = A_RAD * C_LIGHT * T_star**4
    
    print(f"\\nInput:")
    print(f"  T^n = {T_n:.6f} keV")
    print(f"  φ^n = {phi_n:.10e} GJ/cm²")
    print(f"  σ_P = {sigma_P} cm⁻¹, C_v = {C_v} GJ/(cm³·keV)")
    print(f"  β = {beta:.10f}, f = {f:.10f}")
    
    # Solve φ equation
    # φ^{n+1}·[1/(c·Δt) + σ_P·f·θ] = φ^n/(c·Δt) + σ_P·f·acT★⁴ - σ_P·f·(1-θ)·φ^n - (1-f)·Δe/Δt
    diag = 1.0 / (C_LIGHT * dt) + sigma_P * f * theta
    rhs = phi_n / (C_LIGHT * dt) + sigma_P * f * acT4_star - sigma_P * f * (1.0 - theta) * phi_n
    if abs(Delta_e) > 1e-15:
        rhs -= (1.0 - f) * Delta_e / dt
    
    phi_np1 = rhs / diag
    
    print(f"\\nφ equation:")
    print(f"  Diagonal coeff = {diag:.10e}")
    print(f"  RHS = {rhs:.10e}")
    print(f"  φ^{{n+1}} = {phi_np1:.10e}")
    
    # Solve T equation
    phi_tilde = theta * phi_np1 + (1.0 - theta) * phi_n
    e_np1 = e_n + dt * f * sigma_P * (phi_tilde - acT4_star)
    if abs(Delta_e) > 1e-15:
        e_np1 += (1.0 - f) * Delta_e
    T_np1 = e_np1 / C_v
    
    print(f"\\nT equation:")
    print(f"  φ̃ = {phi_tilde:.10e}")
    print(f"  e^{{n+1}} = {e_np1:.10e}")
    print(f"  T^{{n+1}} = {T_np1:.10f}")
    
    # Energy conservation check
    E_n = C_v * T_n + phi_n / C_LIGHT
    E_np1 = C_v * T_np1 + phi_np1 / C_LIGHT
    
    print(f"\\nEnergy conservation:")
    print(f"  E^n = {E_n:.10e} GJ/cm³")
    print(f"  E^{{n+1}} = {E_np1:.10e} GJ/cm³")
    print(f"  |ΔE|/E = {abs(E_np1 - E_n)/E_n:.10e}")
    
    if abs(E_np1 - E_n)/E_n < 1e-14:
        print(f"  ✓ Energy conserved to machine precision")
    else:
        print(f"  ✗ Energy NOT conserved!")
    
    return phi_np1, T_np1

def main():
    print("\\n" + "="*70)
    print("PURE 0-D SOLUTION (No Boundary Complications)")
    print("="*70)
    
    # Test parameters
    C_v = 0.01  # GJ/cm³/keV
    sigma_P = 10.0  # cm⁻¹
    T_n = 0.4  # keV
    T_rad_n = 1.0  # keV
    phi_n = A_RAD * C_LIGHT * T_rad_n**4
    
    print(f"\\nTest cases:")
    test_cases = [
        ("Backward Euler", 1.0, 0.001),
        ("Backward Euler", 1.0, 0.01),
        ("Crank-Nicolson", 0.5, 0.001),
        ("Crank-Nicolson", 0.5, 0.01),
    ]
    
    results = {}
    for name, theta, dt in test_cases:
        print(f"\\n")
        phi_np1, T_np1 = test_pure_0d(theta, dt, T_n, phi_n, sigma_P, C_v)
        results[(name, theta, dt)] = (phi_np1, T_np1)
    
    # Now compare with reference equilibrationTest.py implementations
    print(f"\\n\\n" + "="*70)
    print("COMPARISON WITH REFERENCE equilibrationTest.py")
    print("="*70)
    
    # For Backward Euler
    print(f"\\n\\nBackward Euler (θ=1.0):")
    for dt in [0.001, 0.01]:
        phi_np1, T_np1 = results[("Backward Euler", 1.0, dt)]
        
        # Reference computation (from equilibrationTest.py BE_update)
        T_star = T_n
        Delta_t = dt
        beta = 4*A_RAD*T_star**3/C_v
        f_ref = 1/(1 + beta*Delta_t*C_LIGHT*sigma_P)
        Ern = A_RAD*T_rad_n**4
        Er_new_ref = (Ern + f_ref*sigma_P*Delta_t*C_LIGHT*(A_RAD*T_star**4) - (1-f_ref)*(C_v*T_star-C_v*T_n))/(1+f_ref*Delta_t*C_LIGHT*sigma_P)
        T_new_ref = (C_v*T_n+f_ref*C_LIGHT*sigma_P*Delta_t*(Er_new_ref - A_RAD*T_star**4) + (1-f_ref)*(C_v*T_star-C_v*T_n))/(C_v)
        phi_new_ref = C_LIGHT * Er_new_ref
        
        print(f"  Δt = {dt} ns:")
        print(f"    This solver: φ = {phi_np1:.10e}, T = {T_np1:.10f}")
        print(f"    Reference:   φ = {phi_new_ref:.10e}, T = {T_new_ref:.10f}")
        print(f"    Δφ = {abs(phi_np1 - phi_new_ref):.6e}, ΔT = {abs(T_np1 - T_new_ref):.6e}")
        
    # For Crank-Nicolson
    print(f"\\n\\nCrank-Nicolson (θ=0.5):")
    for dt in [0.001, 0.01]:
        phi_np1, T_np1 = results[("Crank-Nicolson", 0.5, dt)]
        
        # Reference computation (from equilibrationTest.py CN_update)
        T_star = T_n
        Delta_t = dt
        beta = 4*A_RAD*T_star**3/C_v
        f_ref = 1/(1 + 0.5*beta*dt*C_LIGHT*sigma_P)
        Ern = A_RAD*T_rad_n**4
        Er_new_ref = (Ern + f_ref*sigma_P*dt*C_LIGHT*(A_RAD*T_star**4 - 0.5*Ern) - (1-f_ref)*(C_v*T_star-C_v*T_n))/(1+f_ref*dt*C_LIGHT*sigma_P*0.5)
        T_new_ref = (C_v*T_n+f_ref*C_LIGHT*sigma_P*dt*(0.5*(Er_new_ref + Ern) - A_RAD*T_star**4) + (1-f_ref)*(C_v*T_star-C_v*T_n))/(C_v)
        phi_new_ref = C_LIGHT * Er_new_ref
        
        print(f"  Δt = {dt} ns:")
        print(f"    This solver: φ = {phi_np1:.10e}, T = {T_np1:.10f}")
        print(f"    Reference:   φ = {phi_new_ref:.10e}, T = {T_new_ref:.10f}")
        print(f"    Δφ = {abs(phi_np1 - phi_new_ref):.6e}, ΔT = {abs(T_np1 - T_new_ref):.6e}")

if __name__ == "__main__":
    main()
