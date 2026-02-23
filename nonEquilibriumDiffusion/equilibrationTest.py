import numpy as np
from scipy.optimize import brentq, newton
from scipy.integrate import quad
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../Problems')
from plotfuncs import show

C_v = 0.01 #GJ/keV/cm^3
c = 2.99792458e1 #cm/ns
sigma = 5.0 #cm^-1, absorption cross section
a_rad = 0.01372 #GJ/cm^3/keV^4, radiation constant

#initial conditions
T_m_initial = 0.4 #keV
T_r_initial = 1.0 #keV

#initial total energy
U = C_v * T_m_initial + a_rad * T_r_initial**4

#helper constants
q = C_v / a_rad
r = -U / a_rad
P = -4*r
Q = -C_v**2/a_rad**2

y0 = np.cbrt(-Q/2+np.sqrt((Q/2)**2 + (P/3)**3)) + np.cbrt(-Q/2-np.sqrt((Q/2)**2 + (P/3)**3)) 
print(f"y0 = {y0}")
print(f"q = {q}")
print(f"r = {r}")
print(f"P = {P}")
print(f"Q = {Q}")

#print the terms in Teq
print(f"2*C_v/(a_rad*np.sqrt(y0)) = {2*C_v/(a_rad*np.sqrt(y0))}")
print(f"2*C_v/(a_rad*np.sqrt(y0)-y0) = {2*C_v/(a_rad*np.sqrt(y0))-y0}")
Teq = 0.5*(np.sqrt(2*C_v/(a_rad*np.sqrt(y0))-y0)-np.sqrt(y0))

R = a_rad*Teq**4/(C_v*Teq)

#print problem parameters and Teq
print(f"C_v = {C_v}, sigma = {sigma}, a_rad = {a_rad}")
print(f"T_m_initial = {T_m_initial}, T_r_initial = {T_r_initial}, U = {U}")
print(f"Teq = {Teq}, R = {R}")

#now set up a nonlinear solver to compute the temperature at time t

def integrand(T):
    return (1)/(R*(1-T**4) + (1-T))
import mpmath as mp

def I_eps(r, eps):
    try:
        D = lambda s: r*(4*s - 6*s**2 + 4*s**3 - s**4) + s

        return mp.quad(lambda s: 1/D(s), [eps, 1])
    except:
        print("epsilon =", eps)
        raise
#objective function is the integral from T_initial/T_eq to T/T_eq
#do the integration numerically with scipy
def objective(T, t):
    t_norm = t*c*sigma
    result = mp.quad(integrand, [T_m_initial/float(Teq), T/float(Teq)])
    return float(result) - t_norm

t_plot = np.logspace(-5, -1, 100)
T_plot = np.zeros_like(t_plot)

#do this loop manually to use the previous value as the initial guess
#use a closed rootfinding method such as Brent's method because T is between T_m_initial and Teq
for i, t in enumerate(t_plot):
    if i == 0:
        T_plot[i] = brentq(objective, T_m_initial, Teq-1e-8, args=(t,))
    else:
        if (np.abs(T_plot[i-1] - Teq) < 1e-4):
            T_plot[i] = Teq
        else:
            T_plot[i] = brentq(objective, T_plot[i-1], Teq, args=(t,))
    #print(f"t = {t}, T_m = {T_plot[i]}")


#now solve discrete equations to get solution

def BE_update(Tstar, Tn, Ern, max_iters=20):
    iteration_count = 0
    converged = False
    while (iteration_count < max_iters) and not(converged):
        beta = 4*a_rad*Tstar**3/C_v
        f = 1/(1 + beta*Delta_t*c*sigma)
        Er_new =(Ern + f*sigma*Delta_t*c*(a_rad*Tstar**4) - (1-f)*(C_v*Tstar-C_v*Tn))/(1+f*Delta_t*c*sigma)
        T_new = (C_v*Tn+f*c*sigma*Delta_t*(Er_new - a_rad*Tstar**4) + (1-f)*(C_v*Tstar-C_v*Tn))/(C_v)
        # Check for convergence
        if np.abs(T_new - Tstar) < 1e-6:
            converged = True
        Tstar = T_new
        iteration_count += 1
    return Er_new, T_new



Delta_t = 0.01 #ns
T_backward_euler = []
Er_backward_euler = []
t_be = []
t = 0.
t_be.append(np.min(t_plot))
T_backward_euler.append(T_m_initial)
Er_backward_euler.append(a_rad*T_r_initial**4)
while t < t_plot[-1]:
    Er_new, T_new = BE_update(T_backward_euler[-1], T_backward_euler[-1], Er_backward_euler[-1])
    t += Delta_t
    t_be.append(t)
    T_backward_euler.append(T_new)
    Er_backward_euler.append(Er_new)


#now get solutions with only a single iteration per time step
T_be_one_iter = []
Tr_be_one_iter = []
t_be_one_iter = []
t = 0.
t_be_one_iter.append(np.min(t_plot))
T_be_one_iter.append(T_m_initial)
Tr_be_one_iter.append(T_r_initial)
while t < t_plot[-1]:
    Er_new, T_new = BE_update(T_be_one_iter[-1], T_be_one_iter[-1], a_rad*Tr_be_one_iter[-1]**4, max_iters=1)
    t += Delta_t
    t_be_one_iter.append(t)
    T_be_one_iter.append(T_new)
    Tr_be_one_iter.append(np.power(Er_new/a_rad, 0.25))

Tr_plot = np.pow((U-C_v*T_plot)/a_rad, 0.25)
Er_backward_euler = np.array(Er_backward_euler)
T_backward_euler = np.array(T_backward_euler)
# plt.figure()
# plt.plot(t_plot, T_plot,"k-", label=r'$T_m(t)$')
# plt.plot(t_plot, Tr_plot,"k--")
# plt.plot(t_be, T_backward_euler, 'r-', label=r'$T_m$ BE')
# #only plot every 5th marker using an option to plt.plot
# plt.plot(t_be, np.power(Er_backward_euler/a_rad, 0.25), 'r--', markevery=5)
# plt.plot(t_be_one_iter, T_be_one_iter, 'b-', label=r'$T_m$ BE One Iter',markevery=1)
# plt.plot(t_be_one_iter, Tr_be_one_iter, 'b--', markevery=1)

# #plt.xscale('log')
# plt.xlabel('Time (ns)')
# plt.xlim([t_plot[0], 0.04])
# plt.ylabel('Temperature (keV)')
# plt.legend()
# plt.grid()
# plt.show()

#now do the same for Crank-Nicolson
def CN_update(Tstar, Tn, Ern, max_iters=20, dt=Delta_t):
    iteration_count = 0
    converged = False
    while (iteration_count < max_iters) and not(converged):
        beta = 4*a_rad*Tstar**3/C_v
        f = 1/(1 + 0.5*beta*dt*c*sigma)
        Er_new =(Ern + f*sigma*dt*c*(a_rad*Tstar**4 - 0.5*Ern) - (1-f)*(C_v*Tstar-C_v*Tn))/(1+f*dt*c*sigma*0.5)
        T_new = (C_v*Tn+f*c*sigma*dt*(0.5*(Er_new + Ern) - a_rad*Tstar**4) + (1-f)*(C_v*Tstar-C_v*Tn))/(C_v)
        # Check for convergence
        if np.abs(T_new - Tstar) < 1e-6:
            converged = True
        Tstar = T_new
        iteration_count += 1
    return Er_new, T_new

T_CN = []
Er_CN = []
t_cn = []
t = 0.
t_cn.append(np.min(t_plot))
T_CN.append(T_m_initial)
Er_CN.append(a_rad*T_r_initial**4)
while t < t_plot[-1]:
    Er_new, T_new = CN_update(T_CN[-1], T_CN[-1], Er_CN[-1])
    t += Delta_t
    t_cn.append(t)
    T_CN.append(T_new)
    Er_CN.append(Er_new)
Er_CN = np.array(Er_CN)
T_CN = np.array(T_CN)

#now one iteration CN
T_CN_one_iter = []
Er_CN_one_iter = []
t_cn_one_iter = []
t = 0.
t_cn_one_iter.append(np.min(t_plot))
T_CN_one_iter.append(T_m_initial)
Er_CN_one_iter.append(a_rad*T_r_initial**4)
while t < t_plot[-1]:
    Er_new, T_new = CN_update(T_CN_one_iter[-1], T_CN_one_iter[-1], Er_CN_one_iter[-1], max_iters=1)
    t += Delta_t
    t_cn_one_iter.append(t)
    T_CN_one_iter.append(T_new)
    Er_CN_one_iter.append(Er_new)
Er_CN_one_iter = np.array(Er_CN_one_iter)
T_CN_one_iter = np.array(T_CN_one_iter)

# plt.figure()
# plt.plot(t_cn, T_CN, 'g-', label=r'$T_m$ CN')
# plt.plot(t_cn, np.power(Er_CN/a_rad, 0.25), 'g--', label=r'$T_r$ CN', markevery=5)
# plt.plot(t_cn_one_iter, T_CN_one_iter, 'b-', label=r'$T_m$ CN One Iter', markevery=1)
# plt.plot(t_cn_one_iter, np.power(Er_CN_one_iter/a_rad, 0.25), 'b--', label=r'$T_r$ CN One Iter', markevery=1)
# plt.xlabel('Time (ns)')
# plt.ylabel('Temperature (keV)')
# plt.legend()
# plt.grid()
# plt.show()

#now TR-BDF2
def TR_BDF2_update(Tstar, Tn, Ern, max_iters=200, Lambda=2-np.sqrt(2)):
    iteration_count = 0
    converged = False
    #do a CN step to T_{n+Lambda}
    Er_CN, T_CN = CN_update(Tstar, Tn, Ern, max_iters=max_iters, dt=Lambda*Delta_t)
    alpha = (1-Lambda)/(2-Lambda)
    b1 = 1/alpha
    b2 = 1/(Lambda*(1-Lambda))
    b3 = (1-Lambda)/Lambda
    while (iteration_count < max_iters) and not(converged):
        beta = 4*a_rad*Tstar**3/C_v
        f = 1/(1 + alpha*beta*Delta_t*c*sigma)
        deltaE = b1*(C_v*Tstar) - C_v*T_CN*b2 + b3*(C_v*Tn)
        Er_new =(b2*Er_CN -b3*Ern + f*sigma*Delta_t*c*(a_rad*Tstar**4) - (1-f)*(deltaE))/(b1+f*Delta_t*c*sigma)
        T_new = (b2*C_v*T_CN - b3*C_v*Tn + f*c*sigma*Delta_t*(Er_new - a_rad*Tstar**4) + (1-f)*(deltaE))/(C_v*b1)
        # Check for convergence
        if np.abs(T_new - Tstar) < 1e-6:
            converged = True
        Tstar = T_new
        iteration_count += 1
    return Er_new, T_new

#test one step and print the result
E_test, T_test = TR_BDF2_update(T_m_initial, T_m_initial, a_rad*T_r_initial**4, max_iters=1)
print(f"Test step result: Tr = {(E_test/a_rad)**0.25}, T = {T_test}")

T_TRBDF2 = []
Er_TRBDF2 = []
t_trbdf2 = []
t = 0.
t_trbdf2.append(np.min(t_plot))
T_TRBDF2.append(T_m_initial)
Er_TRBDF2.append(a_rad*T_r_initial**4)
while t < t_plot[-1]:
    Er_new, T_new = TR_BDF2_update(T_TRBDF2[-1], T_TRBDF2[-1], Er_TRBDF2[-1])
    t += Delta_t
    t_trbdf2.append(t)
    T_TRBDF2.append(T_new)
    Er_TRBDF2.append(Er_new)
Er_TRBDF2 = np.array(Er_TRBDF2)
T_TRBDF2 = np.array(T_TRBDF2)
t_trbdf2 = np.array(t_trbdf2)

#now one iteration 
T_TRBDF2_one_iter = []
Er_TRBDF2_one_iter = []
t_trbdf2_one_iter = []
t = 0.
t_trbdf2_one_iter.append(np.min(t_plot))
T_TRBDF2_one_iter.append(T_m_initial)
Er_TRBDF2_one_iter.append(a_rad*T_r_initial**4)
while t < t_plot[-1]:
    Er_new, T_new = TR_BDF2_update(T_TRBDF2_one_iter[-1], T_TRBDF2_one_iter[-1], Er_TRBDF2_one_iter[-1], max_iters=1)
    t += Delta_t
    t_trbdf2_one_iter.append(t)
    T_TRBDF2_one_iter.append(T_new)
    Er_TRBDF2_one_iter.append(Er_new)
Er_TRBDF2_one_iter = np.array(Er_TRBDF2_one_iter)
T_TRBDF2_one_iter = np.array(T_TRBDF2_one_iter)
t_trbdf2_one_iter = np.array(t_trbdf2_one_iter)

#make a plot of all the converged solutions
plt.figure()
plt.plot(t_plot, T_plot,"k-", label=r'$T_m(t)$')
plt.plot(t_plot, Tr_plot,"k--")
plt.plot(t_trbdf2, T_TRBDF2, 'ro-', label=r'TR-BDF2')
plt.plot(t_trbdf2, np.power(Er_TRBDF2/a_rad, 0.25), 'ro--', markerfacecolor='none')
plt.plot(t_cn, T_CN, 'gs-', label=r'Crank-Nicolson')
plt.plot(t_cn, np.power(Er_CN/a_rad, 0.25), 'gs--',markerfacecolor='none')
plt.plot(t_be, T_backward_euler, 'bd-', label=r'Backward Euler')
plt.plot(t_be, np.power(Er_backward_euler/a_rad, 0.25), 'bd--', markerfacecolor='none')
plt.ylim([0.85,np.max([np.max(T_TRBDF2), np.max(np.power(Er_TRBDF2/a_rad, 0.25)), np.max(T_CN), np.max(np.power(Er_CN/a_rad, 0.25)), np.max(T_backward_euler), np.max(np.power(Er_backward_euler/a_rad, 0.25))])])
plt.xlabel('Time (ns)')
plt.xlim([0, 0.06])
plt.ylabel('Temperature (keV)')
plt.legend()
plt.grid()
show("converged_solutions.pdf")

#now make the same plot with the one iteration solutions
plt.figure()
plt.plot(t_plot, T_plot,"k-", label=r'$T_m(t)$')
plt.plot(t_plot, Tr_plot,"k--")
plt.plot(t_trbdf2_one_iter, T_TRBDF2_one_iter, 'ro-', label=r'TR-BDF2')
plt.plot(t_trbdf2_one_iter, np.power(Er_TRBDF2_one_iter/a_rad, 0.25), 'ro--', markerfacecolor='none')
plt.plot(t_cn_one_iter, T_CN_one_iter, 'gs-', label=r'Crank-Nicolson')
plt.plot(t_cn_one_iter, np.power(Er_CN_one_iter/a_rad, 0.25), 'gs--', markerfacecolor='none')
plt.plot(t_be_one_iter, T_be_one_iter, 'bd-', label=r'Backward Euler')
plt.plot(t_be_one_iter, Tr_be_one_iter, 'bd--', markerfacecolor='none')
plt.ylim([0.6,np.max([np.max(T_TRBDF2_one_iter), np.max(np.power(Er_TRBDF2_one_iter/a_rad, 0.25)), np.max(T_CN_one_iter), 
                      np.max(np.power(Er_CN_one_iter/a_rad, 0.25)), np.max(T_be_one_iter), np.max(Tr_be_one_iter)])*1.1])
plt.xlabel('Time (ns)')
plt.xlim([0, 0.06])
plt.ylabel('Temperature (keV)')
plt.legend()
plt.grid()
show("one_iteration_solutions.pdf")
