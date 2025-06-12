import numpy as np
import matplotlib.pyplot as plt
from plotfuncs import hide_spines, show

#number of samples
N = 10**3
t = 1.0
x_eval = 0.75
#sample x between -0.5 and 0.5
x = np.random.uniform(-0.5, 0.5, N)
#sample using quasi-random sampling for better distribution
#from scipy.stats import qmc
#sampler = qmc.Sobol(d=3, scramble=True)
#x = sampler.random_base2(m=int(np.log2(N))).flatten() * 0.5 - 0.25  # Rescale to [-0.5, 0.5]
#sample y between -t and t
y = np.random.uniform(-t, t, N)
#sample z between -t and t
z = np.random.uniform(-t, t, N)
VD = 1*(2*t)**2  # volume of the domain sampled
tol = 0.001  # half-width for delta-window
i4pi = 1.0 / (4.0 * np.pi)  # constant factor for Green's normalization
Gfunc = lambda d,t: i4pi/(d*t)/(2*tol) * (np.abs(t-d)<= tol)  # Gaussian function
num_in = np.sum(np.abs(t - np.sqrt(x**2 + y**2 + z**2)) <= tol)  # count points in delta-window
ds = np.sqrt((x-x_eval)**2 + y**2 + z**2)  # distances from point (x_eval, 0, 0)
# Calculate G for each distance
G = Gfunc(ds, t)  
solution = np.sum(G) * (VD / N)  # Monte Carlo estimate of the integral
# Print the result
print(f"Monte Carlo solution for G({x_eval}, {t}): {solution:.6f}")
#print number of points in delta-window
print(f"Number of points in delta-window: {num_in}, total samples: {N}, ratio: {num_in/N:.6f}")

#calculate solution 10 times and take standard deviation
solutions = []
for _ in range(10):
    x = np.random.uniform(-0., 0.5, N)
    y = np.random.uniform(-t, t, N)
    z = np.random.uniform(-t, t, N)
    ds = np.sqrt((x-x_eval)**2 + y**2 + z**2)
    dsneg = np.sqrt((-x-x_eval)**2 + y**2 + z**2)  # distances from point (-x_eval, 0, 0)
    VD = 0.5*(2*t)**2  # volume of the domain sampled
    G = Gfunc(ds, t) + Gfunc(dsneg, t)  # calculate G for both positive and negative x_eval
    solution = np.sum(G) * (VD / N)
    solutions.append(solution)
# Calculate the standard deviation of the solutions
std_dev = np.std(solutions)
print(f"Mean of solutions: {np.mean(solutions)} Standard deviation of solutions: {std_dev:.6f}")
# Plot the distribution of distances
# plt.hist(ds, bins=100, density=True, alpha=0.5, label='Distance Distribution')
# plt.axvline(x_eval, color='red', linestyle='--', label=f'Eval Point: {x_eval}')
# plt.xlabel('Distance')
# plt.ylabel('Density')   
# plt.show()

Ns = [10**2,10**3, 10**4, 10**5, 10**6, 10**7]
tols = [0.01, 0.05, 0.1]
reps = 40
values = np.zeros((3,len(Ns),reps))
for itol,tol in enumerate(tols):
    for iN,N in enumerate(Ns):
        for replicate in range(reps):
            # Sample points
            x = np.random.uniform(-0.5, 0.5, N)
            y = np.random.uniform(-t, t, N)
            z = np.random.uniform(-t, t, N)
            VD = 1*(2*t)**2
            ds = np.sqrt((x-x_eval)**2 + y**2 + z**2)
            i4pi = 1.0 / (4.0 * np.pi)
            G = Gfunc(ds, t)
            solution = np.sum(G) * (VD / N)
            values[itol, iN, replicate] = solution
mean_values = np.mean(values, axis=2)
std_values = np.std(values, axis=2)
#plot curves of fixed tol and varying N
markerlist = ['o', 's', 'D']  # Different markers for each tolerance
for itol, tol in enumerate(tols):
    eb=plt.errorbar(
        Ns, np.abs(mean_values[itol]), yerr=std_values[itol],
        label=f'$\\tau$ = {tol}', marker=markerlist[itol], capsize=5 # Set error bar opacity only
    )
    # Set alpha for error bars only
    [bar.set_alpha(0.4) for bar in eb[2]]  # eb[2] is the list of LineCollection error bars
plt.xscale('log')
plt.xlabel('Number of Samples (N)')
plt.ylabel('Monte Carlo Solution $\\phi(0.75,1)$')
plt.legend()   
plt.ylim(0.25, np.max(mean_values) * 1.1)  # Adjust y-limits for better visibility
plt.tight_layout()
show("monte_carlo_slab.pdf")


for itol, tol in enumerate([0.001, 0.01, 0.1]):
   plt.plot(Ns, np.abs(mean_values[itol]-0.375), label=f'$\\tau$ = {tol}', marker=markerlist[itol])
#add a plot of the line N^(-1/2), starting at 0.1*times the value of the first point
nplot = np.logspace(np.log10(Ns[0]), np.log10(Ns[-1]), 100)
plt.plot(nplot, 0.1 * nplot**(-0.5)*np.abs(mean_values[0,0]-0.375)/(nplot[0]**-0.5), label=r"$N^{-1/2}$", linestyle='--', color='gray')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Number of Source Samples (N)')
plt.ylabel('Error in MC Estimate')
plt.legend()   
plt.tight_layout()
show("monte_carlo_slab_error.pdf")