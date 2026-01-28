"""
Analysis of Nonlinear Correction Stability
==========================================

This script analyzes why the nonlinear correction N^(k)[φ] = -∇·(D_E^(k) φ ∇E_r^(k))
becomes unstable at large time steps, even though it comes directly from Taylor expansion.

The key insight: The linearization is only valid when higher-order terms are small.
"""

import numpy as np
import matplotlib.pyplot as plt

# Physical constants
C = 29979245800.0  # cm/s
A_RAD = 0.01372    # GJ/(cm^3 keV^4)
RHO = 1.0          # g/cm^3

def marshak_opacity(Er, T_min=0.01):
    """Marshak wave opacity: σ_R = 300 * T^-3"""
    T = (Er / A_RAD)**0.25
    T = np.maximum(T, T_min)
    return 300.0 * T**(-3)

def marshak_diffusion(Er):
    """Diffusion coefficient D = c/(3*σ_R)"""
    sigma = marshak_opacity(Er)
    return C / (3.0 * sigma)

def analytical_D_derivative(Er, T_min=0.01):
    """
    Analytical derivative ∂D/∂E_r for Marshak opacity.
    
    D = c/(3*σ) where σ = 300*T^-3 and T = (E_r/a)^(1/4)
    
    Therefore:
    ∂D/∂E_r = ∂/∂E_r [c/(900*T^-3)]
            = (c/900) * ∂/∂E_r [T^3]
            = (c/900) * 3*T^2 * ∂T/∂E_r
            = (c/900) * 3*T^2 * (1/4) * (1/a) * T^-3
            = (c/900) * (3/4) * (1/a) * T^-1
            = (c/3600) * (1/a) * T^-1
    """
    T = np.maximum((Er / A_RAD)**0.25, T_min)
    return (C / 3600.0) * (1.0 / A_RAD) * T**(-1)

def analytical_D_second_derivative(Er, T_min=0.01):
    """
    Second derivative ∂²D/∂E_r² for Marshak opacity.
    
    From above: ∂D/∂E_r = (c/3600) * (1/a) * T^-1
    
    Therefore:
    ∂²D/∂E_r² = (c/3600) * (1/a) * ∂/∂E_r [T^-1]
              = (c/3600) * (1/a) * (-1) * T^-2 * ∂T/∂E_r
              = (c/3600) * (1/a) * (-1) * T^-2 * (1/4) * (1/a) * T^-3
              = -(c/3600) * (1/4) * (1/a)^2 * T^-5
              = -(c/14400) * (1/a)^2 * T^-5
    """
    T = np.maximum((Er / A_RAD)**0.25, T_min)
    return -(C / 14400.0) * (1.0 / A_RAD)**2 * T**(-5)

def taylor_series_analysis(Er_base, delta_Er):
    """
    Analyze the Taylor series expansion of D(E_r + δE_r) around E_r.
    
    D(E_r + δE_r) = D(E_r) + D'(E_r)*δE_r + (1/2)*D''(E_r)*δE_r^2 + ...
    
    The nonlinear correction uses only the linear term D'(E_r)*δE_r.
    This analysis shows when higher-order terms become important.
    """
    # Compute derivatives
    D_0 = marshak_diffusion(Er_base)
    D_1 = analytical_D_derivative(Er_base)
    D_2 = analytical_D_second_derivative(Er_base)
    
    # True value
    D_true = marshak_diffusion(Er_base + delta_Er)
    
    # Taylor approximations
    D_linear = D_0 + D_1 * delta_Er
    D_quadratic = D_0 + D_1 * delta_Er + 0.5 * D_2 * delta_Er**2
    
    return {
        'D_true': D_true,
        'D_linear': D_linear,
        'D_quadratic': D_quadratic,
        'linear_term': D_1 * delta_Er,
        'quadratic_term': 0.5 * D_2 * delta_Er**2,
        'linear_error': D_true - D_linear,
        'quadratic_error': D_true - D_quadratic,
        'relative_quadratic_to_linear': np.abs(0.5 * D_2 * delta_Er**2) / np.abs(D_1 * delta_Er)
    }

def estimate_delta_Er_from_timestep(dt_ns, T_bc=1.0, L=0.1):
    """
    Estimate the change in E_r over a time step based on diffusion.
    
    For diffusion, characteristic change is:
    δE_r ~ D * ∇²E_r * δt ~ D * (E_r/L^2) * δt
    
    At the boundary, E_r ~ a*T_bc^4 and D ~ c/(3*σ) ~ c*T^3/900
    """
    dt = dt_ns * 1e-9  # Convert to seconds
    Er_bc = A_RAD * T_bc**4
    D_bc = marshak_diffusion(Er_bc)
    
    # Estimate: δE_r ~ D * (E_r/L^2) * δt
    delta_Er = D_bc * (Er_bc / L**2) * dt
    return delta_Er

def plot_taylor_convergence():
    """
    Plot how Taylor series converges for different base E_r values.
    Shows when higher-order terms become important.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Different base temperatures
    T_values = [0.1, 0.3, 0.5, 1.0]  # keV
    
    for idx, T_base in enumerate(T_values):
        ax = axes[idx // 2, idx % 2]
        Er_base = A_RAD * T_base**4
        
        # Range of delta_Er values (as fraction of Er_base)
        delta_Er_frac = np.logspace(-3, 0, 100)  # 0.001 to 1.0 times Er_base
        delta_Er_vals = delta_Er_frac * Er_base
        
        linear_errors = []
        quad_errors = []
        quad_to_linear_ratios = []
        
        for delta_Er in delta_Er_vals:
            result = taylor_series_analysis(Er_base, delta_Er)
            linear_errors.append(np.abs(result['linear_error']) / result['D_true'])
            quad_errors.append(np.abs(result['quadratic_error']) / result['D_true'])
            quad_to_linear_ratios.append(result['relative_quadratic_to_linear'])
        
        # Plot relative errors
        ax.loglog(delta_Er_frac, linear_errors, 'b-', linewidth=2, label='Linear approximation error')
        ax.loglog(delta_Er_frac, quad_errors, 'r--', linewidth=2, label='Quadratic approximation error')
        ax.loglog(delta_Er_frac, quad_to_linear_ratios, 'g:', linewidth=2, label='|Quadratic term| / |Linear term|')
        
        # Add reference lines
        ax.axhline(0.1, color='k', linestyle='--', alpha=0.3, label='10% error')
        ax.axhline(0.5, color='k', linestyle=':', alpha=0.3, label='50% error')
        
        ax.set_xlabel('δE_r / E_r', fontsize=11)
        ax.set_ylabel('Relative Error', fontsize=11)
        ax.set_title(f'T = {T_base:.1f} keV (E_r = {Er_base:.3e} GJ/cm³)', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_ylim([1e-6, 10])
    
    plt.tight_layout()
    plt.savefig('taylor_series_convergence.png', dpi=150, bbox_inches='tight')
    print("Plot saved: taylor_series_convergence.png")
    print()

def analyze_timestep_validity():
    """
    Analyze what time steps are valid for the linear approximation.
    Shows the relationship between dt and approximation error.
    """
    print("=" * 70)
    print("Time Step Validity Analysis")
    print("=" * 70)
    print()
    print("For the Marshak wave, we need δE_r << E_r for linear approximation.")
    print("The change in E_r scales as: δE_r ~ D * (E_r/L²) * δt")
    print()
    
    L = 0.1  # cm - domain size
    T_bc = 1.0  # keV - boundary temperature
    Er_bc = A_RAD * T_bc**4
    D_bc = marshak_diffusion(Er_bc)
    
    print(f"Boundary conditions:")
    print(f"  T_bc = {T_bc} keV")
    print(f"  E_r,bc = {Er_bc:.4e} GJ/cm³")
    print(f"  D_bc = {D_bc:.4e} cm²/s")
    print(f"  Domain size L = {L} cm")
    print()
    
    # Time steps to analyze
    dt_values = [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]  # ns
    
    print(f"{'dt (ns)':>10} {'δE_r (GJ/cm³)':>15} {'δE_r/E_r':>12} {'Quad/Linear':>12} {'Valid?':>8}")
    print("-" * 70)
    
    for dt_ns in dt_values:
        delta_Er = estimate_delta_Er_from_timestep(dt_ns, T_bc, L)
        delta_Er_frac = delta_Er / Er_bc
        
        result = taylor_series_analysis(Er_bc, delta_Er)
        quad_to_linear = result['relative_quadratic_to_linear']
        
        # Linear approximation valid if quadratic term < 10% of linear term
        valid = "Yes" if quad_to_linear < 0.1 else "No"
        if quad_to_linear > 0.5:
            valid = "NO!"
        
        print(f"{dt_ns:>10.4f} {delta_Er:>15.4e} {delta_Er_frac:>12.4f} {quad_to_linear:>12.4f} {valid:>8}")
    
    print()
    print("Interpretation:")
    print("  - When Quad/Linear < 0.1: Linear approximation is good (< 10% error)")
    print("  - When Quad/Linear ~ 0.5: Quadratic term is half the linear term (large error)")
    print("  - When Quad/Linear > 1.0: Higher-order terms dominate (linearization fails)")
    print()
    print("This explains why large time steps cause instabilities:")
    print("  → Large δt → Large δE_r → Large quadratic term")
    print("  → Quadratic term can be negative even when linear term is positive")
    print("  → Nonlinear correction overshoots, driving E_r negative!")
    print()

def demonstrate_overshoot():
    """
    Demonstrate how the nonlinear correction can overshoot and drive E_r negative.
    """
    print("=" * 70)
    print("Nonlinear Correction Overshoot Demonstration")
    print("=" * 70)
    print()
    
    # Start at high E_r, take large step to low E_r (simulating wave front)
    Er_high = A_RAD * 1.0**4  # 1 keV
    Er_low = A_RAD * 0.1**4   # 0.1 keV
    delta_Er = Er_low - Er_high  # Negative, moving to lower E_r
    
    result = taylor_series_analysis(Er_high, delta_Er)
    
    print(f"Wave front scenario: Moving from hot to cold region")
    print(f"  E_r (hot) = {Er_high:.4e} GJ/cm³ (T = 1.0 keV)")
    print(f"  E_r (cold) = {Er_low:.4e} GJ/cm³ (T = 0.1 keV)")
    print(f"  δE_r = {delta_Er:.4e} GJ/cm³")
    print()
    print(f"Taylor series terms:")
    print(f"  D(E_r,hot) = {result['D_true']:.4e} cm²/s")
    print(f"  Linear term: D' * δE_r = {result['linear_term']:.4e}")
    print(f"  Quadratic term: (1/2)*D'' * δE_r² = {result['quadratic_term']:.4e}")
    print()
    print(f"Approximation quality:")
    print(f"  |Quadratic| / |Linear| = {result['relative_quadratic_to_linear']:.4f}")
    print(f"  Linear error = {result['linear_error']:.4e} ({np.abs(result['linear_error']/result['D_true'])*100:.1f}%)")
    print()
    
    # Now show what happens in the nonlinear correction
    print("What happens in the nonlinear correction:")
    print(f"  N^(k)[φ] = -∇·(D_E * φ * ∇E_r)")
    print(f"  D_E = {analytical_D_derivative(Er_high):.4e} cm²·s⁻¹·(GJ/cm³)⁻¹")
    print()
    print("If ∇E_r is large (steep gradient at wave front):")
    print("  → D_E * ∇E_r is VERY large")
    print("  → Nonlinear correction term is huge")
    print("  → Can easily exceed the linear diffusion term")
    print("  → Drives solution in wrong direction, potentially to E_r < 0")
    print()
    print("The limiter prevents this by capping:")
    print("  |N^(k)[φ]| ≤ limiter * |Linear diffusion term|")
    print()

def plot_coefficient_behavior():
    """
    Plot how D, D', and D'' behave for the Marshak opacity.
    Shows the highly nonlinear nature of the problem.
    """
    T_vals = np.logspace(-1, 0, 100)  # 0.1 to 1.0 keV
    Er_vals = A_RAD * T_vals**4
    
    D_vals = np.array([marshak_diffusion(Er) for Er in Er_vals])
    D1_vals = np.array([analytical_D_derivative(Er) for Er in Er_vals])
    D2_vals = np.array([analytical_D_second_derivative(Er) for Er in Er_vals])
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # D vs T
    axes[0].loglog(T_vals, D_vals, 'b-', linewidth=2)
    axes[0].set_xlabel('Temperature (keV)', fontsize=11)
    axes[0].set_ylabel('D (cm²/s)', fontsize=11)
    axes[0].set_title('Diffusion Coefficient\nD = c/(3σ), σ = 300T⁻³', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # D' vs T
    axes[1].loglog(T_vals, D1_vals, 'r-', linewidth=2)
    axes[1].set_xlabel('Temperature (keV)', fontsize=11)
    axes[1].set_ylabel('∂D/∂E_r (cm²·s⁻¹·(GJ/cm³)⁻¹)', fontsize=11)
    axes[1].set_title('First Derivative\n∂D/∂E_r ∝ T⁻¹', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    # D'' vs T
    axes[2].loglog(T_vals, -D2_vals, 'g-', linewidth=2)  # Plot negative for log scale
    axes[2].set_xlabel('Temperature (keV)', fontsize=11)
    axes[2].set_ylabel('|∂²D/∂E_r²| (cm²·s⁻¹·(GJ/cm³)⁻²)', fontsize=11)
    axes[2].set_title('Second Derivative\n|∂²D/∂E_r²| ∝ T⁻⁵', fontsize=11)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('diffusion_coefficient_derivatives.png', dpi=150, bbox_inches='tight')
    print("Plot saved: diffusion_coefficient_derivatives.png")
    print()

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("NONLINEAR CORRECTION STABILITY ANALYSIS")
    print("=" * 70)
    print()
    print("This analysis explains why the nonlinear correction")
    print("N^(k)[φ] = -∇·(D_E^(k) φ ∇E_r^(k))")
    print("becomes unstable at large time steps, despite being mathematically correct.")
    print()
    
    # 1. Plot coefficient behavior
    print("1. Analyzing diffusion coefficient derivatives...")
    print()
    plot_coefficient_behavior()
    
    # 2. Taylor series convergence
    print("2. Analyzing Taylor series convergence...")
    print()
    plot_taylor_convergence()
    
    # 3. Time step validity
    print("3. Determining valid time step sizes...")
    print()
    analyze_timestep_validity()
    
    # 4. Demonstrate overshoot
    demonstrate_overshoot()
    
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("Key findings:")
    print()
    print("1. The nonlinear correction comes from Taylor series: D(E+δE) ≈ D(E) + D'δE")
    print("   This is only valid when higher-order terms D''(δE)²/2 + ... are small.")
    print()
    print("2. For the Marshak opacity σ = 300T⁻³:")
    print("   - D ∝ T³ (highly nonlinear)")
    print("   - D' ∝ T⁻¹ (derivative scales inversely with T)")
    print("   - D'' ∝ T⁻⁵ (second derivative very large at low T)")
    print()
    print("3. At large time steps:")
    print("   - δE_r ~ D*(E_r/L²)*δt becomes large")
    print("   - Quadratic term (D''/2)*δE_r² becomes comparable to D'*δE_r")
    print("   - Linear approximation breaks down")
    print()
    print("4. At steep gradients (wave front):")
    print("   - Large ∇E_r makes nonlinear term D_E*∇E_r enormous")
    print("   - Can easily exceed linear diffusion term")
    print("   - Drives E_r in wrong direction, potentially negative")
    print()
    print("5. The limiter fixes this by enforcing:")
    print("   |Nonlinear term| ≤ limiter × |Linear term|")
    print("   This ensures the linearization assumption holds.")
    print()
    print("Mathematical justification for the limiter:")
    print("  The limiter is not ad-hoc; it enforces the validity condition")
    print("  for Taylor series truncation: higher-order terms must be small.")
    print("  By capping the nonlinear correction, we ensure the linear")
    print("  approximation doesn't violate its own assumptions.")
    print()
    print("=" * 70)
