import numpy as np
import matplotlib.pyplot as plt
from math import gamma, pi, sqrt
from scipy.special import beta as Beta
from plotfuncs import *
# ----------------------------
# Parameters (edit as needed)
# ----------------------------
s = 4
b = .01372 #0.3
sigma0 = 300.0
n = 3.0
ac = 0.4113152524  # given as (a*c)

E0 = 1.0           # <-- choose total injected energy
M = E0 / b         # conserved "mass" of u = T^s in the source-type problem

# Times to compare
times = [0.1, 0.3, 1.0, 3.0]   # <-- edit

def S_N(N):
    """Surface area of unit sphere in R^N."""
    return 2.0 * pi**(N / 2.0) / gamma(N / 2.0)

def exponents_u(N, p):
    """Exponents for u: u = t^{-alpha_u} F(r/t^beta)."""
    beta = 1.0 / (N * (p - 1.0) + 2.0)
    alpha_u = N * beta
    return alpha_u, beta
# ----------------------------
# Derived parameters
# ----------------------------
m = 4.0 + n
p = m / s

#print beta
print(f"Using parameters: s={s}, b={b}, n={n}, m={m}, p={p}")

print(f"Derived exponents for N=1: alpha_u={exponents_u(1,p)[0]:.6f}, beta={exponents_u(1,p)[1]:.6f}")
print(f"Derived exponents for N=2: alpha_u={exponents_u(2,p)[0]:.6f}, beta={exponents_u(2,p)[1]:.6f}")
print(f"Derived exponents for N=3: alpha_u={exponents_u(3,p)[0]:.6f}, beta={exponents_u(3,p)[1]:.6f}")


# kappa = (4/(4+n)) * (ac/(3*sigma0))
kappa = (4.0 / (4.0 + n)) * (ac / (3.0 * sigma0))
D = kappa / b


def exponents_T(N, m, s):
    """Exponents for T: T = t^{-alpha} f(r/t^beta)."""
    # alpha = alpha_u / s, beta same as u
    alpha_u, beta = exponents_u(N, p=m/s)
    alpha = alpha_u / s
    return alpha, beta

def lambda_from_params(N, p, D):
    """lambda = ((p-1)*beta)/(2 p D)"""
    _, beta = exponents_u(N, p)
    return (p - 1.0) * beta / (2.0 * p * D)

def A_from_mass(N, p, lam, M):
    """
    M = (S_N/2) * lam^{-N/2} * A^{1/(p-1)+N/2} * B(N/2, 1/(p-1)+1)
    Solve for A.
    """
    SN = S_N(N)
    a = N / 2.0
    bpar = 1.0 / (p - 1.0) + 1.0
    B = Beta(a, bpar)
    power = 1.0 / (p - 1.0) + N / 2.0
    A = ((2.0 * M * (lam ** (N / 2.0))) / (SN * B)) ** (1.0 / power)
    return A

def constants_for_N(N):
    """Compute (alpha, beta, lambda, A, eta_f) for a given N."""
    alpha, beta = exponents_T(N, m=m, s=s)
    lam = lambda_from_params(N, p=p, D=D)
    A = A_from_mass(N, p=p, lam=lam, M=M)
    eta_f = sqrt(A / lam)
    return alpha, beta, lam, A, eta_f

def T_of_r_t(r, t, N):
    """Physical temperature profile T(r,t) for given N."""
    alpha, beta, lam, A, eta_f = constants_for_N(N)
    # Front radius
    R = eta_f * (t ** beta)

    # Similarity argument
    inside = 1.0 - (r ** 2) / (R ** 2)
    T = np.zeros_like(r)

    # Amplitude and exponent for T
    amp = (t ** (-alpha)) * (A ** (1.0 / (m - s)))
    expo = 1.0 / (m - s)

    mask = inside > 0.0
    T[mask] = amp * inside[mask] ** expo
    return T, R

def chi_of_r_t(r, t, N):
    """
    Compute chi = (1/(T^4 sigma_R)) |d_r T^4|
    = (4/sigma0) * T^{n-1} * |T_r|
    """
    alpha, beta, lam, A, eta_f = constants_for_N(N)
    R = eta_f * (t ** beta)

    inside = 1.0 - (r ** 2) / (R ** 2)

    T = np.zeros_like(r)
    Tr = np.zeros_like(r)

    amp = (t ** (-alpha)) * (A ** (1.0 / (m - s)))
    expo = 1.0 / (m - s)

    mask = inside > 0.0

    # T
    T[mask] = amp * inside[mask] ** expo

    # dT/dr
    Tr[mask] = (
        amp * expo
        * inside[mask] ** (expo - 1.0)
        * (-2.0 * r[mask] / (R ** 2))
    )

    chi = np.zeros_like(r)
    chi[mask] = (4.0 / sigma0) * (T[mask] ** (n - 1.0)) * np.abs(Tr[mask])
    return chi
def find_chi_crossing(r, chi):
    """
    Return the first r where chi >= 1.
    Returns None if no crossing.
    """
    idx = np.argmax(chi >= 1.0)
    if chi[idx] < 1.0:
        return None
    return r[idx]

# ----------------------------
# Main execution: Plot T(r,t) for N=1,2,3 at multiple times
# ----------------------------
def main():
    """Main function to generate plots when run as a script"""
    #create a list of lines styles to cycle through
    line_styles = [ '-','--', '-.', (0, (8, 2, 2, 2)),':']
    #create a data structure store the solutions for loading by a code to compare with numerical results

    for N in [1, 2, 3]:
        #make figure proportions wider than tall
        plt.figure(figsize=(8, 8 / 1.518))
        # Determine a plotting radius range that covers the largest front among chosen times
        _, _, _, _, eta_f = constants_for_N(N)
        alpha, beta, *_ = constants_for_N(N)
        R_max = eta_f * (max(times) ** beta)

        r = np.linspace(0.0, 1.05 * R_max, 10000)
        #keep track of max T and max R for setting plot limits
        Tmax = 0.0
        Rmax = 0.0
        for t in times:
            T, R = T_of_r_t(r, t, N)
            if np.max(T) > Tmax:
                Tmax = np.max(T)
            if R > Rmax:
                Rmax = R
            plt.plot(r, T, linestyle=line_styles[times.index(t) % len(line_styles)],label=f"t={t:g} ns  (R={R:.4g} cm)")
            # ---- compute chi and find crossing ----
            chi = chi_of_r_t(r, t, N)
            rc = find_chi_crossing(r, chi)
            if (rc is None):
                #set rc to R
                rc = R

            if (rc is not None) and ((n<1) or (s==4)):
                Tc = np.interp(rc, r, T)
                plt.vlines(
                    rc, 0.0, Tc,
                    colors="purple",
                    linestyles="solid",
                    linewidth=.75,
                    alpha=0.8, zorder=-5,
                )
        plt.ylim(0.0,1.1*Tmax)
        plt.xlim(0.0,1.1*Rmax)
        plt.xlabel("r [cm]")
        plt.ylabel("T(r,t) [keV]")
        #plt.title(f"Zel'dovich / Barenblatt solution in physical space (N={N})")
        plt.legend(loc=3)
        plt.grid(True, alpha=0.5)
        filename = f"Zeldovich_T_of_r_t_N{N}_{n}_{s}_{b}.pdf"
        show(filename)

    # ----------------------------
    # Optional: overlay geometries at a fixed time
    # ----------------------------
    t0 = 1.0  # <-- choose a time to compare geometries directly
    plt.figure()

    # pick a radius range that covers the largest front among N=1,2,3 at t0
    R_list = []
    for N in [1, 2, 3]:
        _, R = T_of_r_t(np.array([0.0]), t0, N)
        R_list.append(R)
    R_max = max(R_list)

    r = np.linspace(0.0, 1.05 * R_max, 800)
    for N in [1, 2, 3]:
        T, R = T_of_r_t(r, t0, N)
        plt.plot(r, T, line_styles[N-1], label=f"N={N}  (R={R:.4g} cm)")

    plt.xlabel("r [cm]")
    plt.ylabel("T(r,1) [keV]")
    #plt.title(f"Compare geometries at t={t0:g}")
    plt.legend()
    plt.grid(True, alpha=0.5)
    filename = f"Zeldovich_compare_geometries_t{t0}_{n}_{s}_{b}.pdf"
    show(filename)


if __name__ == "__main__":
    main()