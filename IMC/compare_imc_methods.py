#!/usr/bin/env python3
"""
Compare Fleck-Cummings IMC vs Carter-Forest IMC

This script runs the same problem with both methods to demonstrate
the differences in temporal discretization error.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import IMC1D as imc_fc  # Fleck-Cummings
import IMC1D_CarterForest as imc_cf  # Carter-Forest

# Add utils to path for plotting
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'utils'))
from plotfuncs import show


def run_comparison(problem_name="Marshak Wave"):
    """Run both IMC methods on the same problem and compare results."""
    
    print("="*70)
    print(f"Comparing Fleck-Cummings vs Carter-Forest IMC")
    print(f"Problem: {problem_name}")
    print("="*70)
    
    # ========== Problem Setup ==========
    # Marshak wave test problem
    Ntarget = 20000
    Nboundary = 10000
    Nsource = 0
    NMax = 100000
    
    # Fine time step to show temporal accuracy
    dt = 0.1  # ns
    final_time = 2.0  # ns
    
    # Spatial mesh
    L = 10.0  # cm
    I = 50
    mesh = np.array([[i * L / I, (i + 1) * L / I] for i in range(I)])
    
    # Initial conditions (cold material)
    Tinit = np.zeros(I) + 0.01  # keV
    Trinit = np.zeros(I) + 0.01
    
    # Boundary conditions (hot left boundary)
    T_left = 1.0  # keV
    T_right = 0.0
    T_boundary = (T_left, T_right)
    
    # Material properties
    sigma_a_const = 200.0  # cm^-1
    sigma_a_func = lambda T: sigma_a_const + 0 * T
    
    cv_val = 0.1  # GJ/(cm³·keV)
    eos = lambda T: cv_val * T
    inv_eos = lambda u: u / cv_val
    cv_func = lambda T: cv_val + 0 * T
    
    source = np.zeros(I)
    reflect = (False, False)
    
    print(f"\nMesh: {I} cells, L = {L} cm")
    print(f"Time step: dt = {dt} ns, final time = {final_time} ns")
    print(f"σ_a = {sigma_a_const} cm⁻¹, c_v = {cv_val} GJ/(cm³·keV)")
    print(f"Boundary: T_left = {T_left} keV, T_right = {T_right} keV")
    print(f"Ntarget = {Ntarget}, NMax = {NMax}")
    
    # ========== Run Fleck-Cummings IMC ==========
    print("\n" + "="*70)
    print("FLECK-CUMMINGS IMC (effective scattering)")
    print("="*70)
    
    time_fc, rad_temp_fc, mat_temp_fc = imc_fc.run_simulation(
        Ntarget, Nboundary, Nsource, NMax, Tinit, Trinit, T_boundary,
        dt, mesh, sigma_a_func, eos, inv_eos, cv_func, source, final_time,
        reflect=reflect, output_freq=1, theta=1.0,
        use_scalar_intensity_Tr=True, geometry='slab'
    )
    
    # ========== Run Carter-Forest IMC ==========
    print("\n" + "="*70)
    print("CARTER-FOREST IMC (time-delayed re-emission)")
    print("="*70)
    
    time_cf, rad_temp_cf, mat_temp_cf = imc_cf.run_simulation(
        Ntarget, Nboundary, Nsource, NMax, Tinit, Trinit, T_boundary,
        dt, mesh, sigma_a_func, eos, inv_eos, cv_func, source, final_time,
        reflect=reflect, output_freq=1,
        use_scalar_intensity_Tr=True
    )
    
    # ========== Plot Comparison ==========
    print("\n" + "="*70)
    print("Generating comparison plots...")
    print("="*70)
    
    x_centers = 0.5 * (mesh[:, 0] + mesh[:, 1])
    
    # Plot 1: Material temperature profiles at final time
    fig1, ax1 = plt.subplots(1, 1, figsize=(8, 5))
    
    ax1.plot(x_centers, mat_temp_fc[-1], 'b-o', linewidth=2, markersize=4,
             label='Fleck-Cummings', alpha=0.8, markevery=3)
    ax1.plot(x_centers, mat_temp_cf[-1], 'r--s', linewidth=2, markersize=4,
             label='Carter-Forest', alpha=0.8, markevery=3)
    
    ax1.set_xlabel('Position x (cm)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Material Temperature (keV)', fontsize=13, fontweight='bold')
    ax1.set_title(f'Material Temperature at t = {final_time:.2f} ns', 
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    show('imc_comparison_material_profile.pdf', close_after=True)
    print("Saved: imc_comparison_material_profile.pdf")
    
    # Plot 2: Radiation temperature profiles at final time
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 5))
    
    ax2.plot(x_centers, rad_temp_fc[-1], 'b-o', linewidth=2, markersize=4,
             label='Fleck-Cummings', alpha=0.8, markevery=3)
    ax2.plot(x_centers, rad_temp_cf[-1], 'r--s', linewidth=2, markersize=4,
             label='Carter-Forest', alpha=0.8, markevery=3)
    
    ax2.set_xlabel('Position x (cm)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Radiation Temperature (keV)', fontsize=13, fontweight='bold')
    ax2.set_title(f'Radiation Temperature at t = {final_time:.2f} ns', 
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11, loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    show('imc_comparison_radiation_profile.pdf', close_after=True)
    print("Saved: imc_comparison_radiation_profile.pdf")
    
    # Plot 3: Time history at a fiducial point
    fiducial_idx = I // 4  # Quarter way through domain
    
    fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(8, 8))
    
    # Material temperature history
    ax3a.plot(time_fc, mat_temp_fc[:, fiducial_idx], 'b-o', linewidth=2,
              markersize=5, label='Fleck-Cummings', alpha=0.8, markevery=2)
    ax3a.plot(time_cf, mat_temp_cf[:, fiducial_idx], 'r--s', linewidth=2,
              markersize=5, label='Carter-Forest', alpha=0.8, markevery=2)
    
    ax3a.set_xlabel('Time (ns)', fontsize=13, fontweight='bold')
    ax3a.set_ylabel('Material Temperature (keV)', fontsize=13, fontweight='bold')
    ax3a.set_title(f'Temperature History at x = {x_centers[fiducial_idx]:.2f} cm',
                   fontsize=14, fontweight='bold')
    ax3a.legend(fontsize=11, loc='best', framealpha=0.9)
    ax3a.grid(True, alpha=0.3, linestyle='--')
    
    # Radiation temperature history
    ax3b.plot(time_fc, rad_temp_fc[:, fiducial_idx], 'b-o', linewidth=2,
              markersize=5, label='Fleck-Cummings', alpha=0.8, markevery=2)
    ax3b.plot(time_cf, rad_temp_cf[:, fiducial_idx], 'r--s', linewidth=2,
              markersize=5, label='Carter-Forest', alpha=0.8, markevery=2)
    
    ax3b.set_xlabel('Time (ns)', fontsize=13, fontweight='bold')
    ax3b.set_ylabel('Radiation Temperature (keV)', fontsize=13, fontweight='bold')
    ax3b.legend(fontsize=11, loc='best', framealpha=0.9)
    ax3b.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    show('imc_comparison_time_history.pdf', close_after=True)
    print("Saved: imc_comparison_time_history.pdf")
    
    # ========== Summary Statistics ==========
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    # Difference metrics
    mat_diff = np.abs(mat_temp_fc[-1] - mat_temp_cf[-1])
    rad_diff = np.abs(rad_temp_fc[-1] - rad_temp_cf[-1])
    
    print(f"\nFinal time: {final_time:.2f} ns")
    print(f"\nMaterial temperature differences:")
    print(f"  Max difference: {np.max(mat_diff):.6e} keV")
    print(f"  RMS difference: {np.sqrt(np.mean(mat_diff**2)):.6e} keV")
    print(f"\nRadiation temperature differences:")
    print(f"  Max difference: {np.max(rad_diff):.6e} keV")
    print(f"  RMS difference: {np.sqrt(np.mean(rad_diff**2)):.6e} keV")
    
    print("\n" + "="*70)
    print("Key differences between methods:")
    print("  Fleck-Cummings: Uses Fleck factor f for effective scattering")
    print("                  Temporal discretization error ~ O(dt)")
    print("  Carter-Forest:  True absorption + exponential re-emission delay")
    print("                  Exact in time for linearized equations")
    print("="*70)
    
    return {
        'time_fc': time_fc,
        'mat_temp_fc': mat_temp_fc,
        'rad_temp_fc': rad_temp_fc,
        'time_cf': time_cf,
        'mat_temp_cf': mat_temp_cf,
        'rad_temp_cf': rad_temp_cf,
        'mesh': mesh,
    }


if __name__ == "__main__":
    results = run_comparison()
    print("\nComparison complete!")
