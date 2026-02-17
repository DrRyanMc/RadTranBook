#!/usr/bin/env python3
"""
Test and visualize the error filtering in compute_error
"""

import numpy as np
import matplotlib.pyplot as plt
from marshak_efficiency_study import (
    generate_self_similar_solution,
    compute_error,
    A_RAD
)

def test_error_filtering():
    """Demonstrate the error filtering on actual Marshak wave data"""
    
    print("="*70)
    print("Testing Error Filtering for Marshak Wave")
    print("="*70)
    
    # Generate self-similar reference solution
    t_final = 0.1  # ns
    ref = generate_self_similar_solution(t_final=t_final, r_max=0.2, n_cells=200)
    
    r = ref['r']
    Er_ref = ref['Er']
    T_ref = ref['T']
    
    # Initial energy threshold
    T_init = 0.1  # keV
    Er_init = A_RAD * T_init**4
    Er_threshold = 1.1 * Er_init
    
    # Mask for cells used in error computation
    mask = Er_ref > Er_threshold
    n_masked = np.sum(mask)
    n_total = len(r)
    
    print(f"\nInitial temperature: T_init = {T_init} keV")
    print(f"Initial energy: Er_init = {Er_init:.6e} GJ/cm^3")
    print(f"Threshold: 1.1 * Er_init = {Er_threshold:.6e} GJ/cm^3")
    print(f"\nCells used in error: {n_masked}/{n_total} ({100*n_masked/n_total:.1f}%)")
    print(f"Cells filtered out: {n_total - n_masked}/{n_total} ({100*(n_total-n_masked)/n_total:.1f}%)")
    
    # Create test solution with 5% error in wave region
    Er_test = Er_ref.copy()
    Er_test[mask] *= 1.05  # Add 5% error only in active region
    
    # Compute errors
    error_filtered = compute_error(r, Er_test, r, Er_ref)
    
    # Compute error without filtering (old method)
    error_unfiltered = np.linalg.norm(Er_test - Er_ref) / np.linalg.norm(Er_ref)
    
    print(f"\nWith 5% error in wave region:")
    print(f"  Filtered error (new): {error_filtered:.6e}")
    print(f"  Unfiltered error (old): {error_unfiltered:.6e}")
    print(f"  Ratio: {error_filtered/error_unfiltered:.2f}x")
    
    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot 1: Energy density with threshold
    ax = axes[0]
    ax.semilogy(r, Er_ref, 'b-', linewidth=2, label='Reference $E_r$')
    ax.axhline(Er_threshold, color='red', linestyle='--', linewidth=2, 
               label=f'Threshold = 1.1 × $E_r$(init)')
    ax.fill_between(r, 1e-10, Er_threshold, alpha=0.2, color='gray', 
                     label='Filtered out')
    ax.set_xlabel('Position r (cm)', fontsize=12)
    ax.set_ylabel('Radiation Energy $E_r$ (GJ/cm³)', fontsize=12)
    ax.set_title('Energy Filtering for Error Computation', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.2)
    ax.set_ylim(1e-7, 2e-2)
    
    # Plot 2: Mask showing which cells are included
    ax = axes[1]
    ax.plot(r, mask.astype(float), 'g-', linewidth=2)
    ax.fill_between(r, 0, mask.astype(float), alpha=0.3, color='green')
    ax.set_xlabel('Position r (cm)', fontsize=12)
    ax.set_ylabel('Included in Error', fontsize=12)
    ax.set_title(f'Cells Used in Error Computation ({n_masked}/{n_total} cells)', 
                 fontsize=14, fontweight='bold')
    ax.set_xlim(0, 0.2)
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Relative error distribution
    ax = axes[2]
    rel_diff = np.abs(Er_test - Er_ref) / Er_ref
    ax.semilogy(r, rel_diff, 'r-', linewidth=2, label='Relative difference')
    ax.axvline(r[mask][0] if np.any(mask) else 0, color='green', linestyle='--', 
               linewidth=1, alpha=0.7, label='Error region boundary')
    if np.any(mask):
        ax.axvline(r[mask][-1], color='green', linestyle='--', 
                   linewidth=1, alpha=0.7)
    ax.set_xlabel('Position r (cm)', fontsize=12)
    ax.set_ylabel('Relative Difference |ΔE_r|/E_r', fontsize=12)
    ax.set_title('Spatial Distribution of Error (5% perturbation in wave)', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.2)
    
    plt.tight_layout()
    plt.savefig('test_error_filtering.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as 'test_error_filtering.png'")
    # plt.show()  # Comment out to avoid display issues
    
    print("\n" + "="*70)
    print("Test complete!")
    print("="*70)


if __name__ == "__main__":
    test_error_filtering()
