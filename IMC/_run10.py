import sys, warnings
sys.path.insert(0, '/Users/ryanmcclarren/Dropbox/Papers/RadTranBook/IMC')
warnings.filterwarnings('ignore')
import ConvergingMarshakWave as cm
import numpy as np

n_cells = 50
r_edges = np.linspace(0.0, cm.R, n_cells + 1)
mesh    = np.column_stack([r_edges[:-1], r_edges[1:]])

T_init = np.full(n_cells, 1e-3)
state  = cm.imc.init_simulation(5000, T_init, T_init, mesh,
                                 cm.eos, cm.inv_eos, geometry='spherical')

T_boundary = (0.0, cm.outer_T_keV)
source     = np.zeros(n_cells)
dt = 0.1

print(f"{'step':>4}  {'t_phys (ns)':>12}  {'T_surf (HeV)':>13}  "
      f"{'T_centre (HeV)':>15}  {'N_particles':>11}  {'energy_loss (GJ)':>17}")
print('-' * 80)
for s in range(10):
    state, info = cm.imc.step(
        state, 5000, 2000, 0, 20000,
        T_boundary, dt, mesh, cm.opacity, cm.inv_eos, cm.cv, source,
        reflect=(True, False), geometry='spherical')
    t_phys = cm.T_INIT_NS + state.time
    T_surf_HeV = cm.outer_T_keV(state.time) * cm.T_HEV_PER_KEV
    T_ctr_HeV  = state.temperature[0] * cm.T_HEV_PER_KEV
    dE         = info['energy_loss']
    N          = info['N_particles']
    print(f"{s+1:4d}  {t_phys:12.4f}  {T_surf_HeV:13.4f}  "
          f"{T_ctr_HeV:15.6f}  {N:11d}  {dE:17.3e}")
