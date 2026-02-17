import IMCSlab as imc
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import IMCSlab as imc
import numpy as np
import matplotlib.pyplot as plt

a = imc.__a
c = imc.__c
Ntarget = 10000
Nboundary = 10000
NMax = 10**6
Nsource = 0
dt = 0.001
final_time = 1.
L = 0.325    #length of slab
I = 200 #number of cells
mesh = np.zeros((I,2))
dx = L/I
for i in range(I):
    mesh[i] = [i*dx,(i+1)*dx]
mesh_midpoints = 0.5*(mesh[:,0]+mesh[:,1])
Tinit = np.zeros(I) + 1e-4
#Tinit[0] = 1.0
Trinit = np.zeros(I) + 1e-4
T_boundary = (1.0,0)
source = np.zeros(I)
sigma_a_f = lambda T: 3*T**-3 #300*T**-1
cv_val = 0.3
eos = lambda T: cv_val*T
inv_eos = lambda u: (u/cv_val)
cv = lambda T: cv_val 
#run simulation
times, radiation_temperatures, temperatures = imc.run_simulation(Ntarget,Nboundary,Nsource, NMax,Tinit,Trinit,
                                                                T_boundary, dt, mesh, sigma_a_f,
                                                                eos,inv_eos,cv,source, final_time, reflect=(False,True),output_freq=50)
#analytic solutions
#t =1 
#plot temperatures as a function of space at final time
plt.plot(mesh_midpoints,(temperatures[-1]))
plt.plot(mesh_midpoints,radiation_temperatures[-1])
plt.xlim([0,L])
plt.xlabel("Position")
plt.ylabel("Temperature")
plt.legend(["Temperature","Radiation Temperature"])
plt.show()