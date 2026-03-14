"""
Su-Olson benchmark with a linear (grey) material EOS: e = alpha * T, alpha = 0.03.

Benchmark tables (x/t vs time columns [0.1, 0.31623, 1.0, 3.16228, 10.0, 31.6228, 100]):
  rad_data  — radiation energy density / a  (= T_rad^4)
  mat_data  — material energy density  e = 0.03 * T

Compared against the IMC results via:
  radiation :  radiation_temperatures[-1]**4  (= E_r / a)
  material  :  0.03 * temperatures[-1]        (= e_mat)
"""
import sys, os
_here        = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(_here)
sys.path.insert(0, _here)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import IMCSlab as imc
import numpy as np
import matplotlib.pyplot as plt
from utils.plotfuncs import font, hide_spines, show

# ── helpers ────────────────────────────────────────────────────────────────────
def last_nonzero(data, column):
    """Return index of the last row where data[i][column] != 0."""
    for i in range(len(data) - 1, -1, -1):
        if data[i][column] != 0:
            return i
    return None

a = imc.__a
c = imc.__c

# ── benchmark data ─────────────────────────────────────────────────────────────
# Columns: x/t | t=0.1 | t=0.31623 | t=1.0 | t=3.16228 | t=10.0 | t=31.6228 | t=100.0
# Dashes (wave front not yet arrived) are stored as 0.
rad_data = np.array([
    # x/t       0.1        0.31623    1.0        3.16228    10.0       31.6228    100.0
    [0.01,    0.095162,  0.271108,  0.563683,  0.765084,  1.96832,   0.267247,  0.085108],
    [0.1,     0.095162,  0.271108,  0.557609,  0.756116,  1.950367,  0.266877,  0.085054],
    [0.17783, 0.095162,  0.271108,  0.543861,  0.736106,  1.910675,  0.266071,  0.084937],
    [0.31623, 0.095162,  0.258592,  0.495115,  0.668231,  1.779896,  0.263527,  0.084565],
    [0.45,    0.08809,   0.199962,  0.396442,  0.543721,  1.558248,  0.259729,  0.084008],
    [0.5,     0.047581,  0.135554,  0.316071,  0.453151,  1.420865,  0.257976,  0.08375 ],
    [0.56234, 0.00376,   0.061935,  0.222261,  0.349209,  1.252213,  0.255538,  0.083392],
    [0.75,    0,         0.002788,  0.102348,  0.21078,   0.908755,  0.246543,  0.082061],
    [1.0,     0,         0,         0.034228,  0.124305,  0.562958,  0.230831,  0.079715],
    [1.33352, 0,         0,         0.002864,  0.067319,  0.27752,   0.203718,  0.075591],
    [1.77828, 0,         0,         0,         0.031357,  0.120054,  0.158039,  0.068419],
    [3.16228, 0,         0,         0,         0.001057,  0.013737,  0.022075,  0.036021],
    [5.62341, 0,         0,         0,         0,         0.000413,  0.000814,  0.001068],
    [10.0,    0,         0,         0,         0,         0,         5e-6,      5e-6    ],
    [17.78279,0,         0,         0,         0,         0,         0,         0       ],
])

mat_data = np.array([
    # x/t       0.1        0.31623    1.0        3.16228    10.0       31.6228    100.0
    [0.01,    0.004837,  0.045121,  0.354022,  1.613529,  2.57461,   1.592549,  1.190296],
    [0.1,     0.004837,  0.045121,  0.350958,  1.601467,  2.568476,  1.591998,  1.190108],
    [0.17783, 0.004837,  0.045121,  0.343803,  1.573757,  2.554747,  1.590795,  1.189698],
    [0.31623, 0.004837,  0.044507,  0.316063,  1.47078,   2.507772,  1.586979,  1.188398],
    [0.45,    0.004705,  0.036765,  0.249325,  1.238666,  2.421019,  1.581228,  1.186445],
    [0.5,     0.002419,  0.022562,  0.183937,  1.025219,  2.361647,  1.578549,  1.185538],
    [0.56234, 5.1e-5,    0.006779,  0.108887,  0.759317,  2.280932,  1.5748,    1.184271],
    [0.75,    0,         6.4e-5,    0.034842,  0.416175,  2.069946,  1.56071,   1.179537],
    [1.0,     0,         0,         0.006872,  0.214491,  1.68516,   1.535052,  1.171036],
    [1.33352, 0,         0,         0.000168,  0.094966,  1.028758,  1.487096,  1.155611],
    [1.77828, 0,         0,         0,         0.032116,  0.471906,  1.391456,  1.127131],
    [3.16228, 0,         0,         0,         0.000196,  0.049604,  0.471468,  0.954827],
    [5.62341, 0,         0,         0,         0,         0.001163,  0.019493,  0.082189],
    [10.0,    0,         0,         0,         0,         0,         0.000113,  0.000487],
    [17.78279,0,         0,         0,         0,         0,         0,         0       ],
])

# ── simulation parameters ──────────────────────────────────────────────────────
alpha   = 0.03           # linear material constant: e = alpha * T
Ntarget  = 10000
Nboundary = 0
NMax      = 1*10**6
Nsource   = 20000

times_data = [0.1, 0.31623, 1.0, 3.16228, 10.0, 31.62278, 100.0]
select_time = 5          # 1-based index into times_data

dt         = 0.001
final_time = times_data[select_time - 1] / c
print("final time (dimensionless) =", final_time * c)

L  = 1.25 * (0.5 + final_time * c)
if (select_time == 5):
    L = 6.0
I  = 200
mesh = np.zeros((I, 2))
dx = L / I
for i in range(I):
    mesh[i] = [i * dx, (i + 1) * dx]
mesh_midpoints = 0.5 * (mesh[:, 0] + mesh[:, 1])

Tinit  = np.zeros(I) + 1e-8
Trinit = np.zeros(I) + 1e-8
T_boundary = (0.0, 0)

source = np.zeros(I)
source[mesh_midpoints <= 0.5] = a * c   # same normalisation as nonlinear version

sigma_a_f = lambda T:  1 + 0 * T
eos       = lambda T:  alpha * T
inv_eos   = lambda u:  u / alpha
cv        = lambda T:  alpha + 0 * T

# ── run ────────────────────────────────────────────────────────────────────────
times, radiation_temperatures, temperatures = imc.run_simulation(
    Ntarget, Nboundary, Nsource, NMax, Tinit, Trinit,
    T_boundary, dt, mesh, sigma_a_f, eos, inv_eos, cv, source,
    final_time, reflect=(True, False), output_freq=100)

# ── plot ───────────────────────────────────────────────────────────────────────
print("final time check:", times[-1] * c, "vs", times_data[select_time - 1])

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def make_panel(quantity, ylabel, outfile, log=False):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for idx in range(select_time):       # idx is 0-based column index into *_data
        t_target  = times_data[idx]
        col       = colors[idx % len(colors)]
        #label_str = f"t = {t_target}"
        snap      = int(np.argmin(np.abs(times * c - t_target)))

        if quantity == 'rad':
            imc_vals        = radiation_temperatures[snap]
            bm_mask         = rad_data[:, idx + 1] > 0
            bm_x, bm_y     = rad_data[bm_mask, 0], rad_data[bm_mask, idx + 1]**.25
        else:
            imc_vals        = temperatures[snap]
            bm_mask         = mat_data[:, idx + 1] > 0
            bm_x, bm_y     = mat_data[bm_mask, 0], mat_data[bm_mask, idx + 1]/(alpha/a)

        ax.plot(mesh_midpoints, imc_vals, color=col, lw=1.4, label=f"IMC" if idx==0 else None)
        ax.plot(bm_x, bm_y, "o", color=col, fillstyle="none", ms=5, label=f"reference" if idx==0 else None)

    if log:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(1e-4, 2e0)
        ax.grid(True, alpha=0.2, which='both')
    else:
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0,4)

    ax.set_xlabel("x", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.legend(prop=font, facecolor="white", edgecolor="none", fontsize=8)
    show(outfile, close_after=True)
    print(f"Saved {outfile}")

make_panel('rad', r"$\overline{T}_\mathrm{r}$", "bennett_radiation_linear.pdf", log=False)
make_panel('rad', r"$\overline{T}_\mathrm{r}$", "bennett_radiation_loglog.pdf",  log=True)
make_panel('mat', r"$\overline{T}$",   "bennett_material_linear.pdf",  log=False)
make_panel('mat', r"$\overline{T}$",   "bennett_material_loglog.pdf",  log=True)
