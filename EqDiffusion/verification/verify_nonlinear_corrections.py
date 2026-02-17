#!/usr/bin/env python3
"""
Nonlinear Correction Verification for Smooth Marshak Wave

This script compares solutions with and without nonlinear corrections
and provides diagnostics to verify the corrections are working properly.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))
import numpy as np
import matplotlib.pyplot as plt
from oneDFV import (
    RadiationDiffusionSolver, 
    temperature_from_Er, 
    A_RAD, 
    C_LIGHT,
    RHO
)


# =============================================================================
# MARSHAK WAVE MATERIAL PROPERTIES (same as main file)
# =============================================================================

def marshak_opacity(Er):
    """Temperature-dependent Rosseland opacity: σ_R = 3 * T^-3"""
    T = temperature_from_Er(Er)  # keV
    n = 3
    T_min = 0.05  # Minimum temperature to prevent overflow (keV)
    if T < T_min:
        T = T_min
    return 3.0 * T**(-n)


def marshak_specific_heat(T):
    """Specific heat: c_v = 0.3 GJ/(cm^3·keV)"""
    cv_volumetric = 0.3  # GJ/(cm^3·keV)
    return cv_volumetric / RHO  # GJ/(g·keV)


def marshak_material_energy(T):
    """Material energy density: e = c_v * T (volumetric)"""
    cv_volumetric = 0.3  # GJ/(cm^3·keV)
    return cv_volumetric * T


def marshak_left_bc(Er, x):
    """Left boundary: Dirichlet BC at T = 1 keV"""
    T_bc = 1.0  # keV
    Er_bc = A_RAD * T_bc**4
    return 1.0, 0.0, Er_bc


def marshak_right_bc(Er, x):
    """Right boundary: zero flux"""
    return 0.0, 1.0, 0.0


def smooth_initial_temperature(x):
    """Smooth initial temperature profile"""
    return 1.0 + (0.2 - 1.0) * (1.0 + np.tanh(50.0 * (x - 0.125))) / 2.0


# =============================================================================
# NONLINEAR CORRECTION DIAGNOSTICS
# =============================================================================

class DiagnosticSolver(RadiationDiffusionSolver):
    """Extended solver with nonlinear correction diagnostics"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nl_correction_history = []
        self.linear_term_history = []
        self.damping_history = []
        
    def time_step(self, n_steps=1, verbose=False):
        """Override to capture diagnostics"""
        for step in range(n_steps):
            # Store pre-step state
            Er_old = self.Er.copy()
            
            # Perform time step
            super().time_step(n_steps=1, verbose=verbose)
            
            # If nonlinear corrections are enabled, compute diagnostics
            if self.use_nonlinear_correction and hasattr(self, '_last_nl_correction'):
                self.nl_correction_history.append(self._last_nl_correction.copy())
                self.linear_term_history.append(self._last_linear_term.copy())
                if hasattr(self, '_last_damping'):
                    self.damping_history.append(self._last_damping)


def run_comparison_study(t_final=0.1, limiter_values=[0.0, 0.1, 0.3, 0.5]):
    """Run comparison between different nonlinear correction settings"""
    
    print("="*70)
    print("NONLINEAR CORRECTION VERIFICATION STUDY")
    print("="*70)
    
    # Problem setup
    r_min = 0.0
    r_max = 0.5  # Smaller domain for faster testing
    n_cells = 100
    dt = 0.0005  # Conservative time step
    
    print(f"Domain: [{r_min}, {r_max}] cm with {n_cells} cells")
    print(f"Time step: {dt} ns, Final time: {t_final} ns")
    print(f"Testing limiter values: {limiter_values}")
    
    results = {}
    
    for limiter in limiter_values:
        print(f"\n--- Running with limiter = {limiter} ---")
        
        # Create solver
        solver = DiagnosticSolver(
            r_min=r_min,
            r_max=r_max,
            n_cells=n_cells,
            d=0,
            dt=dt,
            max_newton_iter=20,
            newton_tol=1e-8,
            rosseland_opacity_func=marshak_opacity,
            specific_heat_func=marshak_specific_heat,
            material_energy_func=marshak_material_energy,
            left_bc_func=marshak_left_bc,
            right_bc_func=marshak_right_bc
        )
        
        # Configure nonlinear corrections
        if limiter == 0.0:
            solver.use_nonlinear_correction = False
            solver.nonlinear_limiter = 0.0
            name = "Linear only"
        else:
            solver.use_nonlinear_correction = True
            solver.use_secant_derivative = False
            solver.max_newton_iter_per_step = 20
            solver.nonlinear_skip_boundary_cells = 1
            solver.nonlinear_limiter = limiter
            name = f"NL limiter={limiter}"
        
        # Initial condition
        def initial_Er(r):
            T_init = smooth_initial_temperature(r)
            return A_RAD * T_init**4
        
        solver.set_initial_condition(initial_Er)
        
        # Time evolution
        n_steps = int(t_final / dt)
        print(f"  Running {n_steps} steps...", end='', flush=True)
        
        try:
            for step in range(n_steps):
                solver.time_step(n_steps=1, verbose=False)
                if (step + 1) % (n_steps // 4) == 0:
                    print(f" {100*(step+1)//n_steps}%", end='', flush=True)
            print(" Done!")
            
            # Get final solution
            r, Er = solver.get_solution()
            T = temperature_from_Er(Er)
            
            results[limiter] = {
                'name': name,
                'r': r.copy(),
                'Er': Er.copy(),
                'T': T.copy(),
                'success': True,
                'max_T': T.max(),
                'min_T': T.min()
            }
            
            # Diagnostics for nonlinear cases
            if limiter > 0.0 and hasattr(solver, 'nl_correction_history'):
                if solver.nl_correction_history:
                    nl_corrections = np.array(solver.nl_correction_history)
                    linear_terms = np.array(solver.linear_term_history)
                    
                    # Compute statistics
                    nl_magnitude = np.mean([np.linalg.norm(nl) for nl in nl_corrections])
                    linear_magnitude = np.mean([np.linalg.norm(lin) for lin in linear_terms])
                    
                    results[limiter]['nl_magnitude'] = nl_magnitude
                    results[limiter]['linear_magnitude'] = linear_magnitude
                    results[limiter]['nl_ratio'] = nl_magnitude / linear_magnitude if linear_magnitude > 0 else 0
                    
                    print(f"  NL correction magnitude: {nl_magnitude:.3e}")
                    print(f"  Linear term magnitude: {linear_magnitude:.3e}")
                    print(f"  Ratio (NL/Linear): {results[limiter]['nl_ratio']:.3f}")
            
            print(f"  Final T range: [{T.min():.3f}, {T.max():.3f}] keV")
            
        except Exception as e:
            print(f" FAILED: {str(e)}")
            results[limiter] = {
                'name': name,
                'success': False,
                'error': str(e)
            }
    
    return results


def plot_comparison(results, t_final):
    """Plot comparison of different nonlinear correction settings"""
    
    # Filter successful results
    successful = {k: v for k, v in results.items() if v.get('success', False)}
    
    if len(successful) < 2:
        print("Not enough successful runs to compare")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    colors = ['black', 'blue', 'green', 'red', 'purple', 'orange']
    
    # Temperature profiles
    ax = axes[0, 0]
    for i, (limiter, data) in enumerate(successful.items()):
        ax.plot(data['r'], data['T'], color=colors[i % len(colors)], 
                linewidth=2, label=data['name'])
    ax.set_xlabel('Position r (cm)')
    ax.set_ylabel('Temperature T (keV)')
    ax.set_title(f'Temperature Profiles (t = {t_final} ns)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Energy density profiles
    ax = axes[0, 1]
    for i, (limiter, data) in enumerate(successful.items()):
        ax.plot(data['r'], data['Er'], color=colors[i % len(colors)], 
                linewidth=2, label=data['name'])
    ax.set_xlabel('Position r (cm)')
    ax.set_ylabel('Energy Density $E_r$ (GJ/cm³)')
    ax.set_title(f'Energy Density Profiles (t = {t_final} ns)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Difference from linear solution
    ax = axes[1, 0]
    linear_data = None
    for limiter, data in successful.items():
        if limiter == 0.0:
            linear_data = data
            break
    
    if linear_data:
        for i, (limiter, data) in enumerate(successful.items()):
            if limiter > 0.0:
                # Interpolate to same grid if needed
                if len(data['r']) == len(linear_data['r']) and np.allclose(data['r'], linear_data['r']):
                    diff = data['T'] - linear_data['T']
                else:
                    T_interp = np.interp(linear_data['r'], data['r'], data['T'])
                    diff = T_interp - linear_data['T']
                
                ax.plot(linear_data['r'], diff, color=colors[i % len(colors)], 
                        linewidth=2, label=data['name'])
        
        ax.set_xlabel('Position r (cm)')
        ax.set_ylabel('$T_{NL} - T_{linear}$ (keV)')
        ax.set_title('Nonlinear Correction Effect on Temperature')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Nonlinear correction statistics
    ax = axes[1, 1]
    limiters = []
    nl_ratios = []
    for limiter, data in successful.items():
        if limiter > 0.0 and 'nl_ratio' in data:
            limiters.append(limiter)
            nl_ratios.append(data['nl_ratio'])
    
    if limiters:
        ax.bar(range(len(limiters)), nl_ratios, color='skyblue', alpha=0.7)
        ax.set_xticks(range(len(limiters)))
        ax.set_xticklabels([f'{l:.1f}' for l in limiters])
        ax.set_xlabel('Nonlinear Limiter Value')
        ax.set_ylabel('NL Correction / Linear Term Ratio')
        ax.set_title('Relative Magnitude of Nonlinear Corrections')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('nonlinear_correction_verification.png', dpi=150, bbox_inches='tight')
    print("\nComparison plot saved as 'nonlinear_correction_verification.png'")
    plt.show()


def main():
    """Main verification study"""
    
    # Test different limiter values
    limiter_values = [0.0, 0.1, 0.3, 0.5, 1.0]  # Include no limiter case
    t_final = 0.05  # Short time for quick testing
    
    print("Starting nonlinear correction verification study...")
    results = run_comparison_study(t_final=t_final, limiter_values=limiter_values)
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    for limiter, data in results.items():
        if data.get('success', False):
            print(f"{data['name']:20}: T ∈ [{data['min_T']:.3f}, {data['max_T']:.3f}] keV", end='')
            if 'nl_ratio' in data:
                print(f", NL/Linear ratio = {data['nl_ratio']:.3f}")
            else:
                print()
        else:
            print(f"{data.get('name', f'Limiter {limiter}'):20}: FAILED - {data.get('error', 'Unknown error')}")
    
    plot_comparison(results, t_final)


if __name__ == "__main__":
    main()