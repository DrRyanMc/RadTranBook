import io
import sys
from contextlib import redirect_stdout
import numpy as np

root = '/Users/ryanmcclarren/Dropbox/Papers/RadTranBook'
for p in [root, root + '/DiscreteOrdinates']:
    if p not in sys.path:
        sys.path.insert(0, p)

from DiscreteOrdinates.problems import test_marshak_wave_multigroup_powerlaw as do_case
import sn_solver


def run_mode(mode):
    with redirect_stdout(io.StringIO()):
        res = do_case.setup_and_run(
            I=140,
            Lx=7.0,
            order=3,
            N=64,
            n_groups=10,
            tfinal=1.0e-2,
            dt_min=1e-4,
            dt_max=1e-4,
            K=100,
            maxits=200,
            LOUD=0,
            time_dependent_bc=True,
            output_times=None,
            fix=0,
            fleck_mode=mode,
        )
    cons = res['conservation']
    d = res['diagnostics_store']
    eout = float(sum(s['boundary_out'] for s in d))
    eout_l = float(sum(s['boundary_out_left'] for s in d))
    ein = float(sum(s['boundary_in'] for s in d))
    
    # Compute T_rad at final time
    phi_final = res['phi_g_hist'][-1]
    total_intensity = sum(phi_final[g] for g in range(res['n_groups']))
    trad_final = (total_intensity / sn_solver.ac) ** 0.25
    
    return {
        'mode': mode,
        'Ein_cum': ein,
        'Eout_cum': eout,
        'Eout_left_frac': eout_l / eout if eout > 0 else 0.0,
        'Enet_cum': float(cons['E_net_cum'][-1]),
        'drift': float(cons['drift'][-1]),
        'Tmax': float(res['T_hist'][-1].max()),
        'Trad_max': float(np.max(trad_final)),
    }


r_legacy = run_mode('legacy')
r_imc = run_mode('imc')

print('legacy', r_legacy)
print('imc', r_imc)
for k in ['Ein_cum', 'Eout_cum', 'Eout_left_frac', 'Enet_cum', 'Tmax', 'Trad_max']:
    a = r_legacy[k]
    b = r_imc[k]
    rel = abs(b - a) / max(abs(a), 1e-300)
    print(f'{k}: legacy={a:.6e} imc={b:.6e} rel_change={rel:.6e}')
print(f"drift: legacy={r_legacy['drift']:.3e} imc={r_imc['drift']:.3e}")
