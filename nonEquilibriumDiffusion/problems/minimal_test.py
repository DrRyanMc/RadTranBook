import sys, os
sys.path.insert(0, os.path.dirname(os.getcwd()))
import numpy as np
from multigroup_diffusion_solver import MultigroupDiffusionSolver1D, flux_limiter_levermore_pomraning, flux_limiter_sum

print("Quick test with 1 group, 10 cells, 10 steps")

C_LIGHT, A_RAD = 2.998e1, 0.01372
n_groups, n_cells = 1, 10
energy_edges = [0.0, 100.0]
dt = 0.01 / C_LIGHT

results = {}
for name, lim in [('LP', flux_limiter_levermore_pomraning), ('Sum', flux_limiter_sum)]:
    print(f"\nRunning {name}...")
    s = MultigroupDiffusionSolver1D(
        n_groups=1, r_min=0, r_max=10, n_cells=n_cells,
        energy_edges=energy_edges, geometry='planar', dt=dt,
        diffusion_coeff_funcs=None,
        absorption_coeff_funcs=[lambda T,r: 1.0],
        left_bc_funcs=[lambda p,r: (0,1,0)],
        right_bc_funcs=[lambda p,r: (1,0,0)],
        source_funcs=[lambda r,t: 0.5 if r<0.5 else 0.0],
        emission_fractions=[1.0],
        material_energy_func=lambda T: A_RAD*T**4,
        inverse_material_energy_func=lambda e: (e/A_RAD)**0.25,
        cv=lambda T: 4*A_RAD*T**3,
        flux_limiter_funcs=[lim],
        rosseland_opacity_funcs=[lambda T,r: 1.0]
    )
    s.T = np.ones(n_cells)*0.001
    s.T[0:2] = 1.0  # Hot spot on left
    s.T_old = s.T.copy()
    s.E_r = A_RAD * s.T**4
    s.E_r_old = s.E_r.copy()
    s.kappa = np.zeros(n_cells)
    s.kappa_old = np.zeros(n_cells)
    
    for i in range(10):
        s.step(max_newton_iter=5, newton_tol=1e-6, gmres_tol=1e-6, gmres_maxiter=50, verbose=(i==0))
        s.advance_time()
        if (i+1) % 2 == 0:
            print(f"  Step {i+1}: E_r_max={s.E_r.max():.3e}, kappa_max={s.kappa.max():.3e}")
    
    results[name] = {'E_r': s.E_r.copy(), 'T': s.T.copy()}
    print(f"  Final: E_r_max={s.E_r.max():.3e}, T_max={s.T.max():.4f}")

print("\n" + "="*60)
diff_E = np.abs(results['LP']['E_r'] - results['Sum']['E_r']).max()
diff_T = np.abs(results['LP']['T'] - results['Sum']['T']).max()
print(f"Differences: ΔE_r={diff_E:.3e}, ΔT={diff_T:.3e}")
if diff_E > 1e-10:
    print("✓ SUCCESS! Flux limiters produce different results")
else:
    print("✗ FAILURE: Flux limiters still identical")
