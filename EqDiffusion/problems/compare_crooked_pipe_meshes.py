#!/usr/bin/env python3
"""
Compare Crooked Pipe Solutions from Different Mesh Refinements

Loads two npz files (refined and uniform mesh) and plots fiducial point
temperature histories on the same plot for comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
from plotfuncs import show

def compare_mesh_solutions(refined_npz, uniform_npz, output_filename='crooked_pipe_mesh_comparison.pdf'):
    """
    Compare fiducial point histories from two different mesh resolutions
    
    Parameters:
    -----------
    refined_npz : str
        Path to npz file with refined mesh solution
    uniform_npz : str
        Path to npz file with uniform mesh solution
    output_filename : str
        Output filename for the comparison plot
    """
    print("="*70)
    print("CROOKED PIPE MESH COMPARISON")
    print("="*70)
    
    # Load refined mesh solution
    print(f"\nLoading refined mesh solution: {refined_npz}")
    data_refined = np.load(refined_npz, allow_pickle=True)
    times_refined = data_refined['times']
    fiducial_temps_refined = dict(data_refined['fiducial_temps'].item())
    r_centers_refined = data_refined['r_centers']
    z_centers_refined = data_refined['z_centers']
    print(f"  Mesh size: {len(r_centers_refined)} × {len(z_centers_refined)} cells")
    print(f"  Time points: {len(times_refined)}")
    print(f"  Fiducial points: {len(fiducial_temps_refined)}")
    
    # Load uniform mesh solution
    print(f"\nLoading uniform mesh solution: {uniform_npz}")
    data_uniform = np.load(uniform_npz, allow_pickle=True)
    times_uniform = data_uniform['times']
    fiducial_temps_uniform = dict(data_uniform['fiducial_temps'].item())
    r_centers_uniform = data_uniform['r_centers']
    z_centers_uniform = data_uniform['z_centers']
    print(f"  Mesh size: {len(r_centers_uniform)} × {len(z_centers_uniform)} cells")
    print(f"  Time points: {len(times_uniform)}")
    print(f"  Fiducial points: {len(fiducial_temps_uniform)}")
    
    # Create comparison plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    
    # Different markers and colors for each point
    markers = ['o', 's', '^', 'd', 'v']
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    
    # Get list of fiducial points (assuming same points in both files)
    fiducial_labels = list(fiducial_temps_refined.keys())
    
    # Plot each fiducial point - refined mesh (solid lines)
    for idx, label in enumerate(fiducial_labels):
        temps_refined = fiducial_temps_refined[label]
        ax.loglog(times_refined, temps_refined,
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
        temps_uniform = fiducial_temps_uniform[label]
        ax.loglog(times_uniform, temps_uniform,
                 marker=markers[idx % len(markers)],
                 color=colors[idx % len(colors)],
                 linewidth=2,
                 linestyle='--',
                 markersize=6,
                 markevery=max(1, len(times_uniform)//20),
                 #label=f'{label} (uniform)',
                 alpha=0.8)
    
    ax.set_xlabel('Time (ns)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Temperature (keV)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='best', framealpha=0.9, ncol=1)
    ax.grid(True, which='both', alpha=0.3, linestyle='--')
    ax.grid(True, which='minor', alpha=0.15, linestyle=':')
    
    plt.tight_layout()
    show(output_filename, close_after=True)
    print(f"\nSaved: {output_filename}")
    plt.show()
    
    # Print some comparison statistics
    print("\n" + "="*70)
    print("COMPARISON STATISTICS")
    print("="*70)
    
    for label in fiducial_labels:
        temps_refined_final = fiducial_temps_refined[label][-1]
        temps_uniform_final = fiducial_temps_uniform[label][-1]
        rel_diff = abs(temps_refined_final - temps_uniform_final) / temps_refined_final * 100
        print(f"\n{label}:")
        print(f"  Refined mesh final T:  {temps_refined_final:.6f} keV")
        print(f"  Uniform mesh final T:  {temps_uniform_final:.6f} keV")
        print(f"  Relative difference:   {rel_diff:.3f}%")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    # Default filenames - adjust these based on actual file names
    # The cell counts will depend on your actual mesh sizes
    
    # Example usage - you may need to adjust the filenames
    refined_file = "crooked_pipe_solution.npz"
    uniform_file = "crooked_pipe_solution_uniform_60x210.npz"
    
    # Check if files exist, otherwise prompt user
    import os
    if not os.path.exists(refined_file):
        print(f"Warning: {refined_file} not found")
        print("Looking for available npz files...")
        npz_files = [f for f in os.listdir('.') if f.startswith('crooked_pipe_solution_') and f.endswith('.npz')]
        if npz_files:
            print("Available files:")
            for f in npz_files:
                print(f"  - {f}")
            refined_candidates = [f for f in npz_files if 'refined' in f]
            if refined_candidates:
                refined_file = refined_candidates[0]
                print(f"\nUsing: {refined_file}")
    
    if not os.path.exists(uniform_file):
        print(f"Warning: {uniform_file} not found")
        print("Looking for available npz files...")
        npz_files = [f for f in os.listdir('.') if f.startswith('crooked_pipe_solution_') and f.endswith('.npz')]
        if npz_files:
            print("Available files:")
            for f in npz_files:
                print(f"  - {f}")
            uniform_candidates = [f for f in npz_files if 'uniform' in f]
            if uniform_candidates:
                uniform_file = uniform_candidates[0]
                print(f"\nUsing: {uniform_file}")
    
    # Run comparison
    if os.path.exists(refined_file) and os.path.exists(uniform_file):
        compare_mesh_solutions(refined_file, uniform_file)
    else:
        print("\nError: Could not find both refined and uniform mesh solution files.")
        print("Please ensure both npz files exist and adjust the filenames in the script.")
