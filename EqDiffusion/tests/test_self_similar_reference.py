#!/usr/bin/env python3
"""
Test that the self-similar analytical solution is working correctly
for the Marshak efficiency study
"""

import numpy as np
import matplotlib.pyplot as plt
from marshak_efficiency_study import (
    generate_self_similar_solution,
    load_or_generate_reference
)

def test_self_similar_solution():
    """Test the self-similar solution generation"""
    
    print("="*70)
    print("Testing Self-Similar Analytical Solution")
    print("="*70)
    
    # Test at different times
    times = [0.01, 0.1, 1.0]  # ns
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    colors = ['blue', 'green', 'red']
    
    for t_final, color in zip(times, colors):
        # Generate reference solution
        ref = generate_self_similar_solution(t_final=t_final, r_max=0.2, n_cells=200)
        
        r = ref['r']
        T = ref['T']
        Er = ref['Er']
        xi_max = ref['xi_max']
        K_const = ref['K_const']
        
        # Wave front location
        r_front = xi_max * np.sqrt(K_const * t_final)
        
        print(f"\nt = {t_final} ns:")
        print(f"  Wave front at r = {r_front:.4f} cm")
        print(f"  Max T = {T.max():.4f} keV")
        print(f"  Max Er = {Er.max():.4e} GJ/cm^3")
        
        # Plot temperature
        ax = axes[0]
        ax.plot(r, T, color=color, linewidth=2, label=f't = {t_final} ns')
        ax.axvline(r_front, color=color, linestyle='--', alpha=0.5, linewidth=1)
        
        # Plot radiation energy density
        ax = axes[1]
        ax.plot(r, Er, color=color, linewidth=2, label=f't = {t_final} ns')
        ax.axvline(r_front, color=color, linestyle='--', alpha=0.5, linewidth=1)
    
    # Format plots
    ax = axes[0]
    ax.set_xlabel('Position r (cm)', fontsize=12)
    ax.set_ylabel('Temperature T (keV)', fontsize=12)
    ax.set_title('Self-Similar Marshak Wave: Temperature', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_xlim(0, 0.2)
    
    ax = axes[1]
    ax.set_xlabel('Position r (cm)', fontsize=12)
    ax.set_ylabel('Radiation Energy Density $E_r$ (GJ/cm³)', fontsize=12)
    ax.set_title('Self-Similar Marshak Wave: Radiation Energy', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_xlim(0, 0.2)
    
    plt.tight_layout()
    plt.savefig('test_self_similar_reference.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as 'test_self_similar_reference.png'")
    # plt.show()  # Comment out to avoid display issues in non-interactive mode
    
    # Test that load_or_generate_reference works
    print("\n" + "="*70)
    print("Testing load_or_generate_reference()...")
    print("="*70)
    ref2 = load_or_generate_reference(t_final=0.1, r_max=0.2, n_cells=200)
    print(f"Type: {ref2.get('type', 'unknown')}")
    print(f"Successfully loaded reference with {len(ref2['r'])} points")
    
    print("\n" + "="*70)
    print("All tests passed!")
    print("="*70)


if __name__ == "__main__":
    test_self_similar_solution()
