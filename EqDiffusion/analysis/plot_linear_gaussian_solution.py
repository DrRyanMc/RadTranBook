#!/usr/bin/env python3
"""
Plot linear Gaussian solution at initial and final times
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))
import numpy as np
import matplotlib.pyplot as plt
from plotfuncs import show

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

C_LIGHT = 299792458e-7  # cm/sh (speed of light in cm per nanosecond)
A_RAD = 0.01372  # GJ/(cm³·keV⁴) (radiation constant)

# =============================================================================
# LINEAR GAUSSIAN PROBLEM SETUP
# =============================================================================

# Physical parameters
SIGMA_R = 100.0  # cm^-1, constant opacity
D = C_LIGHT / (3.0 * SIGMA_R)  # Diffusion coefficient
K_COUPLING = 1.0e-5  # Weak material coupling for linearity
D_EFF = D / (1.0 + K_COUPLING)  # Effective diffusion with coupling

# Gaussian parameters
X0 = 2.0  # Center (cm)
SIGMA0 = 0.15  # Initial width (cm)
T_PEAK = 1.0  # Peak temperature (keV)
T_BACKGROUND = 0.1  # Background temperature (keV)

# Convert to Er
ER_PEAK = A_RAD * T_PEAK**4
ER_BACKGROUND = A_RAD * T_BACKGROUND**4
AMPLITUDE = ER_PEAK - ER_BACKGROUND


def temperature_from_Er(Er):
    """Convert radiation energy density to temperature"""
    return (Er / A_RAD)**(1.0/4.0)


def analytical_gaussian_1d(x, t):
    """
    Analytical solution for Gaussian diffusion
    
    Er(x,t) = Er_bg + A*σ0/σ(t) * exp(-(x-x0)²/(2σ(t)²))
    where σ(t)² = σ0² + 2*D_eff*t
    """
    sigma_t_sq = SIGMA0**2 + 2 * D_EFF * t
    sigma_t = np.sqrt(sigma_t_sq)
    
    # Amplitude decreases to conserve integral
    amplitude_t = AMPLITUDE * SIGMA0 / sigma_t
    
    return ER_BACKGROUND + amplitude_t * np.exp(-(x - X0)**2 / (2 * sigma_t_sq))


# =============================================================================
# GENERATE SOLUTIONS
# =============================================================================

# Domain
r_min = 0.0
r_max = 4.0
n_cells = 400
r = np.linspace(r_min, r_max, n_cells)

# Time points
t_initial = 0.0
t_final = 0.5  # sh

# Compute solutions
Er_initial = analytical_gaussian_1d(r, t_initial)
T_initial = temperature_from_Er(Er_initial)

Er_final = analytical_gaussian_1d(r, t_final)
T_final = temperature_from_Er(Er_final)

# Compute widths
sigma_initial = SIGMA0
sigma_final = np.sqrt(SIGMA0**2 + 2 * D_EFF * t_final)

print("="*70)
print("LINEAR GAUSSIAN SOLUTION")
print("="*70)
print(f"Initial time: t = {t_initial} ns")
print(f"  Width: σ = {sigma_initial:.3f} cm")
print(f"  Peak T = {T_initial.max():.4f} keV")
print(f"  Peak Er = {Er_initial.max():.4e} GJ/cm³")
print()
print(f"Final time: t = {t_final} ns")
print(f"  Width: σ = {sigma_final:.3f} cm")
print(f"  Peak T = {T_final.max():.4f} keV")
print(f"  Peak Er = {Er_final.max():.4e} GJ/cm³")
print(f"  Spreading: {(sigma_final/sigma_initial - 1)*100:.1f}%")
print("="*70)

# =============================================================================
# PLOT
# =============================================================================

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))

# Plot radiation energy
ax1.plot(r, Er_initial, 'b-', linewidth=2, label=f't = {t_initial} ns')
ax1.plot(r, Er_final, 'r--', linewidth=2, label=f't = {t_final} ns')
ax1.set_xlabel('Position x (cm)')
ax1.set_ylabel('Radiation Energy Density $E_r$ (GJ/cm$^3$)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot temperature
ax2.plot(r, T_initial, 'b-', linewidth=2, label=f't = {t_initial} ns')
ax2.plot(r, T_final, 'r--', linewidth=2, label=f't = {t_final} ns')
ax2.set_xlabel('Position x (cm)')
ax2.set_ylabel('Temperature T (keV)')
#ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
show('linear_gaussian_solution.pdf')
