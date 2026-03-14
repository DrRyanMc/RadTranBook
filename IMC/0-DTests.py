import IMCSlab as imc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({
    # Typography
    "font.family": "sans-serif",
    "font.sans-serif": ["Univers LT Std","TeX Gyre Heros", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 12,
    "axes.labelsize": 12,
    "font.variant": "small-caps",
    "axes.titlesize": 18,
    "mathtext.fontset": "dejavusans",
    "mathtext.default": "it",

    # Figure
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",

    # Axes/spines
    "axes.edgecolor": "black",
    "axes.linewidth": 1.15,
    "axes.grid": False,

    # Ticks
    "xtick.color": "black",
    "ytick.color": "black",
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.direction": "out",
    "ytick.direction": "out",

    # Lines
    "lines.linewidth": 1.8,
    "lines.solid_capstyle": "round",
    "lines.dash_capstyle": "round",

    # Legend, if used
    "legend.frameon": False,
})
INK = "black"
MUTED = "#6f6f6f"       # close to black!55
PANEL_FILL = "#f2f2f2"  # close to black!5
PROJ = "#b0b0b0"        # light dashed guides
ACCENT = "#005cb9"      # your TikZ accent RGB(0,92,185)
#set parameters
Ntarget = 100
Nboundary = 0
NMax = 10**6
Nsource = 0
dt = 0.025
L = 0.1#0.5 #length of slab
I = 1 #number of cells
mesh = np.zeros((I,2))
dx = L/I
for i in range(I):
    mesh[i] = [i*dx,(i+1)*dx]
Tinit = np.zeros(I) + 0.4
#Tinit[0] = 1.0
Trinit = np.zeros(I) + 1.0
T_boundary = (0.0,0)
source = np.zeros(I)
sigma_a_f = lambda T: 1.0+0*T ##300*T**-3
cv_val = 0.01
eos = lambda T: cv_val*T
inv_eos = lambda u: (u/cv_val)
cv = lambda T: cv_val #4*cv_val*T**3
final_time = dt*15
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
time_eval = np.linspace(0, final_time, 1000)
sol = solve_ivp(RHS,[0,np.max(times)],[Trinit[0]**4*(imc.__a),eos(Tinit[0])],t_eval=time_eval)
#get first two matplolib colors
my_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
MAT = my_colors[0]
RAD = my_colors[0]
MAT_B = my_colors[1]
RAD_B = my_colors[1]
MAT_C = my_colors[2]
RAD_C = my_colors[3]
plt.figure(figsize=(6, 4.5))
plt.plot(time_eval,inv_eos(sol.y[1]), "-", color="black", alpha=0.5)
plt.plot(time_eval,((sol.y[0]/(imc.__a))**0.25), "--", color="black", alpha=0.5)

if I==1:#now re-run with dt = 0.01
    for i in range(I):
        plt.plot(times,temperatures[:,i],color=MAT, alpha=0.8, label=r"$\Delta $" + f"t={dt}" if i==0 else None, marker='^', markersize=8, markerfacecolor='white', markeredgewidth=0.5, zorder=5)
        plt.plot(times,radiation_temperatures[:,i],"--", color=RAD, alpha=0.8, marker='^', markersize=8, markerfacecolor='white', markeredgewidth=0.5, zorder=5)
    dt = 0.025
    times2, radiation_temperatures2, temperatures2 = imc.run_simulation(Ntarget,Nboundary,Nsource, NMax,Tinit,Trinit,
                                                                T_boundary, dt, mesh, sigma_a_f,
                                                                eos,inv_eos,cv,source, final_time, reflect=(True,True),theta=0.5)
    for i in range(I):  
        plt.plot(times2,temperatures2[:,i],color=MAT_C, alpha=0.8, label=r"$\Delta $" + f"t={dt}, " + r"$\theta$=0.5" if i==0 else None, marker='o', markersize=8, markerfacecolor='white', markeredgewidth=0.5)
        plt.plot(times2,radiation_temperatures2[:,i],"--", color=RAD_C, alpha=0.8, marker='o', markersize=8, markerfacecolor='white', markeredgewidth=0.5) 
    dt = 0.01
    times, radiation_temperatures, temperatures = imc.run_simulation(Ntarget,Nboundary,Nsource, NMax,Tinit,Trinit,
                                                                    T_boundary, dt, mesh, sigma_a_f,
                                                                    eos,inv_eos,cv,source, final_time, reflect=(True,True))
    for i in range(I):  
        plt.plot(times,temperatures[:,i],color=MAT_B, alpha=0.8, label=r"$\Delta $" + f"t={dt}" if i==0 else None)
        plt.plot(times,radiation_temperatures[:,i],"--", color=RAD_B, alpha=0.8)   
else:
    
    for i in range(I):
        plt.plot(times,temperatures[:,i],color=MAT, alpha=0.8, label=r"$\Delta $" + f"t={dt}" if i==-1 else None)
        plt.plot(times,radiation_temperatures[:,i],"--", color=RAD, alpha=0.8)
plt.xlabel("t (ns)")
plt.ylabel("T (keV)")
plt.ylim([0.4,1.075])
plt.xlim([0.,0.15])
plt.legend(loc="lower right", ncol=2)
print("mean radiation temperature (not including initial) = ",np.mean(radiation_temperatures[1:,0]))
print("mean temperature (not including initial) = ",np.mean(temperatures[1:,0]))
print("mean sum of radiation and temperature (not including initial) = ",0.5*np.mean(radiation_temperatures[3:,0]+temperatures[3:,0]))
print("mean deterministic radiation temperature (not including initial) = ",np.mean(inv_eos(sol.y[1][1:])))
print("mean deterministic temperature (not including initial) = ",np.mean(((sol.y[0][1:]/(imc.__a))**0.25)))
#have filename include I
fname = f"InfMedia_results_I={I}_dt={dt}.pdf"
#remove outer spines
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
plt.savefig(fname, dpi=600)
plt.show()