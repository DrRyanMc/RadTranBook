#!/usr/bin/env python3
"""
Compare Non-Equilibrium Crooked Pipe Solutions from Different Mesh Refinements

Loads two npz files (refined and uniform mesh) and plots fiducial point
temperature histories on the same plot for comparison.
Creates separate plots for material temperature and radiation temperature.
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'utils'))
import numpy as np
import matplotlib.pyplot as plt
from plotfuncs import show

def compare_mesh_solutions(refined_npz, uniform_npz, 
                          output_material='crooked_pipe_noneq_mesh_comparison_material.pdf',
                          output_radiation='crooked_pipe_noneq_mesh_comparison_radiation.pdf'):
    """
    Compare fiducial point histories from two different mesh resolutions
    
    Parameters:
    -----------
    refined_npz : str
        Path to npz file with refined mesh solution
    uniform_npz : str
        Path to npz file with uniform mesh solution
    output_material : str
        Output filename for the material temperature comparison plot
    output_radiation : str
        Output filename for the radiation temperature comparison plot
    """
    print("="*70)
    print("NON-EQUILIBRIUM CROOKED PIPE MESH COMPARISON")
    print("="*70)
    
    # Load refined mesh solution
    print(f"\nLoading refined mesh solution: {refined_npz}")
    data_refined = np.load(refined_npz, allow_pickle=True)
    times_refined = data_refined['times']
    fiducial_data_refined = dict(data_refined['fiducial_data'].item())
    r_centers_refined = data_refined['r_centers']
    z_centers_refined = data_refined['z_centers']
    print(f"  Mesh size: {len(r_centers_refined)} × {len(z_centers_refined)} cells")
    print(f"  Time points: {len(times_refined)}")
    print(f"  Fiducial points: {len(fiducial_data_refined)}")
    
    # Load uniform mesh solution
    print(f"\nLoading uniform mesh solution: {uniform_npz}")
    data_uniform = np.load(uniform_npz, allow_pickle=True)
    times_uniform = data_uniform['times']
    fiducial_data_uniform = dict(data_uniform['fiducial_data'].item())
    r_centers_uniform = data_uniform['r_centers']
    z_centers_uniform = data_uniform['z_centers']
    print(f"  Mesh size: {len(r_centers_uniform)} × {len(z_centers_uniform)} cells")
    print(f"  Time points: {len(times_uniform)}")
    print(f"  Fiducial points: {len(fiducial_data_uniform)}")
    
    # Different markers and colors for each point
    markers = ['o', 's', '^', 'd', 'v']
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    
    # Get list of fiducial points (assuming same points in both files)
    fiducial_labels = list(fiducial_data_refined.keys())
    
    # =========================================================================
    # PLOT 1: Material Temperature Comparison
    # =========================================================================
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 7))
    
    # Plot each fiducial point - refined mesh (solid lines)
    for idx, label in enumerate(fiducial_labels):
        T_mat_refined = fiducial_data_refined[label]['T_mat']
        ax1.loglog(times_refined, T_mat_refined,
                  marker=markers[idx % len(markers)],
                  color=colors[idx % len(colors)],
                  linewidth=2,
                  linestyle='-',
                  markersize=6,
                  markevery=max(1, len(times_refined)//20),
                  label=f'{label}',
                  alpha=0.8)
    
    # Plot each fiducial point - uniform mesh (dashed lines)
    for idx, label in enumerate(fiducial_labels):
        T_mat_uniform = fiducial_data_uniform[label]['T_mat']
        ax1.loglog(times_uniform, T_mat_uniform,
                  marker=markers[idx % len(markers)],
                  color=colors[idx % len(colors)],
                  linewidth=2,
                  linestyle='--',
                  markersize=6,
                  markevery=max(1, len(times_uniform)//20),
                  #label=f'{label} (uniform)', no label for the uniform
                  alpha=0.8)
    
    ax1.set_xlabel('Time (ns)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Material Temperature T (keV)', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=8, loc='best', framealpha=0.9, ncol=1)
    ax1.grid(True, which='both', alpha=0.3, linestyle='--')
    ax1.grid(True, which='minor', alpha=0.15, linestyle=':')
    
    plt.tight_layout()
    show(output_material, close_after=True)
    print(f"\nSaved: {output_material}")
    
    # =========================================================================
    # PLOT 2: Radiation Temperature Comparison
    # =========================================================================
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 7))
    
    # Plot each fiducial point - refined mesh (solid lines)
    for idx, label in enumerate(fiducial_labels):
        T_rad_refined = fiducial_data_refined[label]['T_rad']
        ax2.loglog(times_refined, T_rad_refined,
                  marker=markers[idx % len(markers)],
                  color=colors[idx % len(colors)],
                  linewidth=2,
                  linestyle='-',
                  markersize=6,
                  markevery=max(1, len(times_refined)//20),
                  label=f'{label}',
                  alpha=0.8)
    
    # Plot each fiducial point - uniform mesh (dashed lines)
    for idx, label in enumerate(fiducial_labels):
        T_rad_uniform = fiducial_data_uniform[label]['T_rad']
        ax2.loglog(times_uniform, T_rad_uniform,
                  marker=markers[idx % len(markers)],
                  color=colors[idx % len(colors)],
                  linewidth=2,
                  linestyle='--',
                  markersize=6,
                  markevery=max(1, len(times_uniform)//20),
                  #label=f'{label} (uniform)',
                  alpha=0.8)
    
    ax2.set_xlabel('Time (ns)', fontsize=13, fontweight='bold')
    ax2.set_ylabel(r'Radiation Temperature $T_\mathrm{rad}$ (keV)', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=8, loc='best', framealpha=0.9, ncol=1)
    ax2.grid(True, which='both', alpha=0.3, linestyle='--')
    ax2.grid(True, which='minor', alpha=0.15, linestyle=':')
    
    plt.tight_layout()
    show(output_radiation, close_after=True)
    print(f"Saved: {output_radiation}")
    
    # Print some comparison statistics
    print("\n" + "="*70)
    print("COMPARISON STATISTICS (FINAL TIME)")
    print("="*70)
    
    for label in fiducial_labels:
        T_mat_refined_final = fiducial_data_refined[label]['T_mat'][-1]
        T_mat_uniform_final = fiducial_data_uniform[label]['T_mat'][-1]
        T_rad_refined_final = fiducial_data_refined[label]['T_rad'][-1]
        T_rad_uniform_final = fiducial_data_uniform[label]['T_rad'][-1]
        
        rel_diff_mat = abs(T_mat_refined_final - T_mat_uniform_final) / T_mat_refined_final * 100
        rel_diff_rad = abs(T_rad_refined_final - T_rad_uniform_final) / T_rad_refined_final * 100
        
        print(f"\n{label}:")
        print(f"  Material Temperature:")
        print(f"    Refined mesh:        {T_mat_refined_final:.6f} keV")
        print(f"    Uniform mesh:        {T_mat_uniform_final:.6f} keV")
        print(f"    Relative difference: {rel_diff_mat:.3f}%")
        print(f"  Radiation Temperature:")
        print(f"    Refined mesh:        {T_rad_refined_final:.6f} keV")
        print(f"    Uniform mesh:        {T_rad_uniform_final:.6f} keV")
        print(f"    Relative difference: {rel_diff_rad:.3f}%")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    # Default filenames - adjust these based on actual file names
    # The cell counts will depend on your actual mesh sizes
    
    # Example usage - you may need to adjust the filenames
    refined_file = "crooked_pipe_noneq_solution_refined_163x127.npz"
    uniform_file = "crooked_pipe_noneq_solution_uniform_60x210.npz"
    
    # Check if files exist, otherwise look for available files
    if not os.path.exists(refined_file):
        print(f"Warning: {refined_file} not found")
        print("Looking for available npz files...")
        npz_files = [f for f in os.listdir('.') if f.startswith('crooked_pipe_noneq_solution_') and f.endswith('.npz')]
        if npz_files:
            print("Available files:")
            for f in sorted(npz_files):
                print(f"  - {f}")
            refined_candidates = [f for f in npz_files if 'refined' in f]
            if refined_candidates:
                refined_file = refined_candidates[0]
                print(f"\nUsing refined file: {refined_file}")
        else:
            print("No crooked_pipe_noneq_solution_*.npz files found in current directory")
    
    if not os.path.exists(uniform_file):
        print(f"Warning: {uniform_file} not found")
        print("Looking for available npz files...")
        npz_files = [f for f in os.listdir('.') if f.startswith('crooked_pipe_noneq_solution_') and f.endswith('.npz')]
        if npz_files:
            print("Available files:")
            for f in sorted(npz_files):
                print(f"  - {f}")
            uniform_candidates = [f for f in npz_files if 'uniform' in f]
            if uniform_candidates:
                uniform_file = uniform_candidates[0]
                print(f"\nUsing uniform file: {uniform_file}")
        else:
            print("No crooked_pipe_noneq_solution_*.npz files found in current directory")
    
    # Run comparison
    if os.path.exists(refined_file) and os.path.exists(uniform_file):
        compare_mesh_solutions(refined_file, uniform_file)
    else:
        print("\nError: Could not find both refined and uniform mesh solution files.")
        print("Please ensure both npz files exist and adjust the filenames in the script if needed.")
        print("\nTo generate the solution files, run:")
        print("  python3 crooked_pipe_noneq.py")
        print("with use_refined_mesh=True and use_refined_mesh=False")
