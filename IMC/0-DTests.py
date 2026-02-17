import IMCSlab as imc
import numpy as np
import matplotlib.pyplot as plt

#set parameters
Ntarget = 100000
Nboundary = 0
NMax = 10**6
Nsource = 0
dt = 0.01
L = 0.1#0.5 #length of slab
I = 2 #number of cells
mesh = np.zeros((I,2))
dx = L/I
for i in range(I):
    mesh[i] = [i*dx,(i+1)*dx]
Tinit = np.zeros(I) + 0.4
#Tinit[0] = 1.0
Trinit = np.zeros(I) + .5
T_boundary = (0.0,0)
source = np.zeros(I)
sigma_a_f = lambda T: 1e2+0*T ##300*T**-3
cv_val = 0.01
eos = lambda T: cv_val*T
inv_eos = lambda u: (u/cv_val)
cv = lambda T: cv_val #4*cv_val*T**3
final_time = dt*10
#run simulation
times, radiation_temperatures, temperatures = imc.run_simulation(Ntarget,Nboundary,Nsource, NMax,Tinit,Trinit,
                                                                T_boundary, dt, mesh, sigma_a_f,
                                                                eos,inv_eos,cv,source, final_time, reflect=(True,True))

#solve problem with scipy integrate
from scipy.integrate import solve_ivp
def RHS(t,y):
    T = inv_eos(y[1])
    Tr = (y[0]/(imc.__a))**0.25
    emission = imc.__a *imc.__c*sigma_a_f(T)*( Tr**4 - T**4)
    return [-emission,emission]
sol = solve_ivp(RHS,[0,np.max(times)],[Trinit[0]**4*(imc.__a),eos(Tinit[0])],t_eval=times)
plt.plot(times,temperatures[:,0])
plt.plot(times,radiation_temperatures[:,0])
plt.plot(times,inv_eos(sol.y[1]))
plt.plot(times,((sol.y[0]/(imc.__a))**0.25))
plt.xlabel("Time")
plt.ylabel("Temperature")
plt.ylim([0.4,0.45])
plt.legend(["Temperature","Radiation Temperature"])
print("mean radiation temperature (not including initial) = ",np.mean(radiation_temperatures[1:,0]))
print("mean temperature (not including initial) = ",np.mean(temperatures[1:,0]))
print("mean sum of radiation and temperature (not including initial) = ",0.5*np.mean(radiation_temperatures[3:,0]+temperatures[3:,0]))
print("mean deterministic radiation temperature (not including initial) = ",np.mean(inv_eos(sol.y[1][1:])))
print("mean deterministic temperature (not including initial) = ",np.mean(((sol.y[0][1:]/(imc.__a))**0.25)))
plt.show()