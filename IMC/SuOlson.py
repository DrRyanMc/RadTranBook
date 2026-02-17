import IMCSlab as imc
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import IMCSlab as imc
import numpy as np
import matplotlib.pyplot as plt

#set parameters
def find_last_non_none_index(data, column):
    for i in range(len(data) - 1, -1, -1):
        if data[i][column] is not None:
            return i
    return None
a = imc.__a
c = imc.__c
Ntarget = 10000
Nboundary = 0
NMax = 10**6
Nsource = 20000
rad_data = [
    [0.01000, 0.09531, 0.27526, 0.64308, 1.20052, 2.23575, 0.69020, 0.35720],
    [0.10000, 0.09531, 0.27526, 0.63585, 1.18869, 2.21944, 0.68974, 0.35714],
    [0.17783, 0.09532, 0.27527, 0.61958, 1.16190, 2.18344, 0.68878, 0.35702],
    [0.31623, 0.09529, 0.26262, 0.56187, 1.07175, 2.06448, 0.68569, 0.35664],
    [0.45000, 0.08823, 0.20312, 0.44711, 0.90951, 1.86072, 0.68111, 0.35599],
    [0.50000, 0.04765, 0.13762, 0.35801, 0.79902, 1.73178, 0.67908, 0.35574],
    [0.56234, 0.00375, 0.06277, 0.25374, 0.66678, 1.57496, 0.67619, 0.35538],
    [0.75000, 0, 0.00280, 0.11430, 0.44675, 1.27398, 0.66548, 0.35393],
    [1.00000, 0, 0,0.03648, 0.27540, 0.98782, 0.64691, 0.35141],
    [1.33352, 0, 0, 0.00291, 0.14531, 0.70822, 0.61538, 0.34697],
    [1.77828, 0, 0,0, 0.05968, 0.45016, 0.56353, 0.33924],
    [3.16228, 0, 0,0, 0.00123, 0.09673, 0.36965, 0.30346],
    [5.62341, 0, 0, 0,0, 0.00375, 0.10830, 0.21382],
    [10.00000, 0, 0, 0,0, 0, 0.00390, 0.07220],
    [17.78279, 0, 0, 0, 0,0, 0, 0.00272]
]
#load in material temperature data from Su_Olson_Material.csv
import pandas as pd
mat_data = pd.read_csv("Su_Olson_Material.csv")
mat_data = mat_data.to_numpy()

rad_data = np.array(rad_data)
#column corresponds to times
times_data = [0.1,0.31623,1.0,3.16228,10.0,31.62278,100]
select_time = 2
dt = 0.0005
last_val = find_last_non_none_index(rad_data, select_time-1)
final_time = times_data[select_time-1]/imc.__c
print("final time = ",final_time*imc.__c)
L = 1.5*(0.5+final_time*imc.__c) #length of slab
I = 50 #number of cells
mesh = np.zeros((I,2))
dx = L/I
for i in range(I):
    mesh[i] = [i*dx,(i+1)*dx]
mesh_midpoints = 0.5*(mesh[:,0]+mesh[:,1])
Tinit = np.zeros(I) + 1e-8
#Tinit[0] = 1.0
Trinit = np.zeros(I) + 1e-8
T_boundary = (0.0,0)
source = np.zeros(I)
#set source for x<0.5 to be 1, x>0.5 to be 0
source[mesh_midpoints<=0.5] = imc.__a*imc.__c*2
sigma_a_f = lambda T: 1+0*T ##300*T**-3
cv_val = imc.__a
eos = lambda T: imc.__a*T**4
inv_eos = lambda u: (u/imc.__a)**0.25
cv = lambda T: 4*cv_val*T**3
#run simulation
times, radiation_temperatures, temperatures = imc.run_simulation(Ntarget,Nboundary,Nsource, NMax,Tinit,Trinit,
                                                                T_boundary, dt, mesh, sigma_a_f,
                                                                eos,inv_eos,cv,source, final_time, reflect=(True,False),output_freq=10)
#analytic solutions
#t =1 
#plot temperatures as a function of space at final time
plt.plot(mesh_midpoints,(temperatures[-1]))
plt.plot(mesh_midpoints,radiation_temperatures[-1])
print(times[-1]*imc.__c, times_data[select_time-1])
print(a*c*times[-1])
plt.plot(rad_data[0:last_val,0],(rad_data[0:last_val,select_time]/(imc.__a*imc.__c))**.25,"o")
plt.plot(mat_data[:,0],mat_data[:,select_time]**.25,"v")
plt.xlim([0,L])
plt.xlabel("Position")
plt.ylabel("Temperature")
plt.legend(["Temperature","Radiation Temperature"])
plt.show()