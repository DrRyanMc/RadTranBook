import sys; sys.path.insert(0, '.')
import numpy as np
import sn_solver

a=0.01372; c=29.98; ac=a*c
T0=0.5; sigma=10.0; f=0.3271
sigma_a=f*sigma; sigma_s=(1-f)*sigma; dt=0.01; icdt=1/(c*dt)
sigma_t_val = sigma + icdt
N=4; order=2; I=1; hx=0.1; nop1=order+1

st=np.full((I,nop1), sigma_t_val)
phi_old=np.full((I,nop1), ac*T0**4)
source_per_dir=sigma_a*ac*T0**4 + icdt*phi_old
source_3d = np.broadcast_to(source_per_dir[:,None,:], (I,N,nop1)).copy()
zero_bcs = np.zeros((N,nop1))

# zero BCs (vacuum): this is what the b vector uses (no reflecting here)
phi_b = sn_solver.single_source_iteration(I, hx, source_3d, st, N, zero_bcs, order=order, fix=0)
print("phi_b (zero BCs):", phi_b[0,0])

# reflecting BCs
from sn_solver import build_reflecting_BCs
psi_old = np.broadcast_to(phi_old[:,None,:], (I,N,nop1)).copy()
bcs_reflect = build_reflecting_BCs(zero_bcs, psi_old, True, True, N, order)
phi_b_reflect = sn_solver.single_source_iteration(I, hx, source_3d, st, N, bcs_reflect, order=order, fix=0)
print("phi_b (reflect BCs):", phi_b_reflect[0,0])

# mv function
phi_mv = sn_solver.single_source_iteration(I, hx,
    np.broadcast_to((sigma_s*phi_old)[:,None,:], (I,N,nop1)).copy(),
    st, N, zero_bcs, order=order, fix=0)
print("mv(phi):", phi_mv[0,0])
print("Ratio mv/phi:", phi_mv[0,0]/phi_old[0,0])
print("2*sigma_s/sigma_t:", 2*sigma_s/sigma_t_val)
print("phi_old:", phi_old[0,0])
print("Expected phi_eq from fixed point (if b=phi):", phi_b_reflect[0,0]/(1-phi_mv[0,0]/phi_old[0,0]))
