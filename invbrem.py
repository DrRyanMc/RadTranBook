import numpy as np
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
#now import solve_ivp from scipy.integrate
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
import scipy.optimize as opt
from scipy.optimize import curve_fit

# Updated function to include the threshold energy condition
font = fm.FontProperties(family = 'Gill Sans', fname = '/Library/Fonts/GillSans.ttc', size = 12)
def hide_spines(intx=False,inty=False):
    """Hides the top and rightmost axis spines from view for all active
    figures and their respective axes."""

    # Retrieve a list of all current figures.
    figures = [x for x in matplotlib._pylab_helpers.Gcf.get_all_fig_managers()]
    if (plt.gca().get_legend()):
        plt.setp(plt.gca().get_legend().get_texts(), fontproperties=font, size=18) 
    for figure in figures:
        # Get all Axis instances related to the figure.
        for ax in figure.canvas.figure.get_axes():
            # Disable spines.
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            # Disable ticks.
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
           # ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda v,_: ("10$^{%d}$" % math.log(v,10)) ))
            for label in ax.get_xticklabels() :
                label.set_fontproperties(font)
            for label in ax.get_yticklabels() :
                label.set_fontproperties(font)
            #ax.set_xticklabels(ax.get_xticks(), fontproperties = font)
            ax.set_xlabel(ax.get_xlabel(), fontproperties = font)
            ax.set_ylabel(ax.get_ylabel(), fontproperties = font)
            ax.set_title(ax.get_title(), fontproperties = font)
            if (inty):
                ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%d'))
            if (intx):
                ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%d'))
def show(nm,a=0,b=0):
    hide_spines(a,b)
    #ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda v,_: ("10$^{%d}$" % math.log(v,10)) ))
    #plt.yticks([1,1e-2,1e-4,1e-6,1e-8,1e-10,1e-12], labels)
    #ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v,_: ("10$^{%d}$" % math.log(v,10)) ))
    plt.savefig(nm);
    plt.show()
c = 2.99792458e1 # speed of light in cm/ns
a = 0.0137202 
sigmaa0 = 10
N = 200
Cv = 0.01
def inv_eos(e):
    return e/Cv
def eos(T):
    return Cv*T
h = 4.135667696e-9
piconst = 15/np.pi**4
Evals = np.logspace(np.log10(0.0001),np.log10(40),N) # energy values in MeV
 # Planck's constant in keV*ns
def B(u,T):
    ans = a*c*u**3/(np.exp(u/T)-1)*piconst
    return ans
def Bcolor(E,T,Tc):
    return a*c*(T**4/Tc**4)*(E**3)/(np.exp(E/Tc)-1)*piconst
def abs_term(u,T,phi):
    ans = sigmaa0*(1-np.exp(-u/T))/(u**3*np.sqrt(T))*phi
    return ans
def emis_term(u,T):
    ans =  sigmaa0*piconst*a*c/np.sqrt(T)*np.exp(-u/T)
    return ans
def rad(u,T,phi):
    ans = c*(emis_term(u,T)-abs_term(u,T,phi))
    return ans
def efunc(t,y):
    rad_vals = y[0:N]
    T = inv_eos(y[N])
    integrand = abs_term(Evals,T,rad_vals)
    #use scipy's integrate to over the samples 
    result = np.trapezoid(integrand, Evals)
    result -= np.trapezoid(emis_term(Evals,T), Evals)
    return result
def RHS(t,y):
    result = np.zeros_like(y)
    T = inv_eos(y[N])
    result[0:N] = rad(Evals,T,y[0:N])
    result[N] = efunc(t,y)
    #print(np.trapezoid(result[0:N], Evals) - result[N])
    return result
y0 = np.zeros(N+1)
Tr0 = 0.5
Tm0 = 0.4
y0[N] = eos(Tm0)
Tc0 = 1.0
y0[0:N] = B(Evals,Tr0) #Bcolor(Evals,Tr0,Tc0)
#plt.plot(Evals, y0[0:N])
plt.plot(Evals, B(Evals,Tr0))
plt.plot(Evals, Bcolor(Evals,Tr0,Tc0))
print(rad(Evals,0.5,y0[0:N]))
plt.show()
integrand = abs_term(Evals,0.5,y0[0:N])
#use scipy's integrate to over the samples 
result = np.trapezoid(integrand, Evals)
print(result)
print("a c T^4 =", np.trapezoid(B(Evals,0.5), Evals), a*c*0.5**4)
result_emis = np.trapezoid(emis_term(Evals,0.5), Evals)
print(result_emis)
print(RHS(0,y0))
sol = solve_ivp(RHS,[0,1], y0, method="BDF", max_step=1e-3, rtol=1e-6, atol=1e-6)
Tr = np.zeros(sol.t.shape[0])
for i in range(0,sol.t.shape[0]):
    Tr[i] = (np.trapezoid(sol.y[0:N,i], Evals)/(a*c))**.25
plt.plot(sol.t,inv_eos(sol.y[-1,:]), label="T")
plt.plot(sol.t,Tr,"--",label="$T_\\mathrm{r}$")
plt.legend(loc="best")
plt.xlabel("t (ns)")
plt.ylabel("Temperature (keV)")
show("invbrem_hist.pdf")

#make a plot grid for E = h nu


T0 = 0.25
T = inv_eos(sol.y[-1,-1])
Evals_plot = Evals

plt.plot(Evals_plot, sol.y[0:N,0],label=f"$t=0$ ns")

#find point closest to 0.1 ns
t_select = 0.01
t_index = np.argmin(np.abs(sol.t - t_select))
#plot the corresponding energy distribution
#plt.plot(Evals_plot, B(Evals,Tr[t_index]),"r--")
plt.plot(Evals_plot,sol.y[0:N,t_index],"--",label=f"t={sol.t[t_index]:.2f} ns")
t_select = 0.1
t_index = np.argmin(np.abs(sol.t - t_select))
#plot the corresponding energy distribution
#plt.plot(Evals_plot, B(Evals,Tr[t_index]),"k--")
plt.plot(Evals_plot,sol.y[0:N,t_index],"-.",label=f"t={sol.t[t_index]:.2f} ns")
t_select = 1.0
t_index = np.argmin(np.abs(sol.t - t_select))
#plot the corresponding energy distribution
#plt.plot(Evals_plot, B(Evals,Tr[t_index]),"k--")
plt.plot(Evals_plot,sol.y[0:N,t_index],":",label=f"t={sol.t[t_index]:.0f} ns")

plt.legend(loc="best")
plt.xlim(1e-2,6)
plt.ylim(1e-8,.0120)
plt.xlabel("Photon Energy, $E_\\nu$ (keV)")
plt.ylabel("$\\phi_\\nu(E_\\nu)$ (GJ /(keV ns cm$^2$)")
show("invbrem_spec.pdf")

#now plot sigma*phi
Tval = inv_eos(sol.y[-1,0])
u = Evals_plot
plt.plot(Evals_plot,sol.y[0:N,t_index]*sigmaa0*(1-np.exp(-u/T))/(u**3*np.sqrt(T)),"-.",label=f"t={sol.t[0]:.2f} ns")

#find point closest to 0.1 ns
t_select = 0.01
t_index = np.argmin(np.abs(sol.t - t_select))
Tval = inv_eos(sol.y[-1,t_index])
#plot the corresponding energy distribution
#plt.plot(Evals_plot, B(Evals,Tr[t_index]),"r--")
plt.plot(Evals_plot,sol.y[0:N,t_index]*sigmaa0*(1-np.exp(-u/T))/(u**3*np.sqrt(T)),"--",label=f"t={sol.t[t_index]:.2f} ns")
t_select = 0.1
t_index = np.argmin(np.abs(sol.t - t_select))
#plot the corresponding energy distribution
#plt.plot(Evals_plot, B(Evals,Tr[t_index]),"k--")
Tval = inv_eos(sol.y[-1,t_index])
plt.plot(Evals_plot,sol.y[0:N,t_index]*sigmaa0*(1-np.exp(-u/T))/(u**3*np.sqrt(T)),"-.",label=f"t={sol.t[t_index]:.2f} ns")
t_select = 1.0
t_index = np.argmin(np.abs(sol.t - t_select))
#plot the corresponding energy distribution
#plt.plot(Evals_plot, B(Evals,Tr[t_index]),"k--")
Tval = inv_eos(sol.y[-1,t_index])
plt.loglog(Evals_plot,sol.y[0:N,t_index]*sigmaa0*(1-np.exp(-u/T))/(u**3*np.sqrt(T)),":",label=f"t={sol.t[t_index]:.2f} ns")

plt.legend(loc="best")
plt.xlim(1e-2,6)
#plt.ylim(1e-8,.0120)
plt.xlabel("Photon Energy, $E_\\nu$ (keV)")
plt.ylabel("$\\phi_\\nu(E_\\nu)$ (GJ /(keV ns cm$^2$)")
show("invbrem_abs.pdf")