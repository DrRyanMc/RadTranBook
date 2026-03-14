import matplotlib as mpl
from matplotlib.patches import Rectangle, FancyBboxPatch, RegularPolygon, Circle
import numpy as np
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

Cv = 0.01 #GJ/cm^3/ns heat capacity
sigma = 1.0 #cm^-1 opacity
C_light = 29.98 #cm/ns
A_rad = 0.01372 #GJ/(cm^3 keV^4)
T0 = 0.4 #keV
Tr0 = 1.0 #keV

ðt= 0.01 #ns
ðx = 0.1 #cm
L = 0.1 #cm
Ninit = 10
Nsource = 10
particles = []

np.random.seed(102)  # For reproducibility

#fleck factor
beta = 4*A_rad*T0**3/Cv
f = 1.0 / (1.0 + beta*sigma*C_light*ðt)
sig_a = sigma * f
sig_s = sigma * (1.0 - f)
print(f"Fleck factor: {f:.4f}")
for i in range(Ninit):
    # Sample initial particle properties (position, direction, energy)
    x = np.random.uniform(0, L)  # Uniformly distributed in the slab
    t = 0.0  # Start time
    mu = np.random.uniform(-1, 1)  # Isotropic direction cosine
    w = ðx*A_rad*Tr0**4 / Ninit  # Initial weight based on radiation energy density
    particles.append((x, mu, w, t))
print(f"Initial weight for initial condition particles is {w}")
for i in range(Nsource):
    # Sample source particle properties (position, direction, energy)
    x =np.random.uniform(0, L)  # Source at the left boundary
    t = np.random.uniform(0, ðt)  # Uniformly distributed over the timestep
    mu = np.random.uniform(-1, 1)  # Isotropic direction cosine
    w =ðt*ðx*sig_a*A_rad*T0**4 / Nsource  # Weight based on source energy density
    particles.append((x, mu, w, t))
print(f"Initial weight for sourced particles is {w}")

all_paths = []
max_steps_in_path = 0
energy_absorbed = 0.0
for x, mu, w, t in particles:
    cur_path = [(x, t)]
    alive = True
    while alive:
        # Sample free path length
        mfp = 1.0 / sigma
        s_coll = -mfp * np.log(np.random.uniform())
        s_t = C_light * (ðt - t)
        s_boundary = (L - x) / mu if mu > 0 else -x / mu

        path_length = min(s_coll, s_t, s_boundary)
        dt_travel = path_length / (C_light)
        t += dt_travel
        x += mu * path_length
        cur_path.append((x, t))
        #check if s_c was the shortest
        if path_length == s_coll:
            # Collision occurs
            if np.random.uniform() < sig_a / sigma:
                # Absorption
                alive = False
                energy_absorbed += w
            else:
                # Scattering: isotropic in the lab frame
                mu = np.random.uniform(-1, 1)
        elif path_length == s_boundary:
            # Particle hits the boundary
            #reflect at the boundary
            mu = -mu
        else:
            # Particle reaches the end of the timestep
            alive = False
    if len(cur_path) > max_steps_in_path:
        max_steps_in_path = len(cur_path)
    all_paths.append(cur_path)
print(f"Simulated {len(particles)} particles with up to {max_steps_in_path} steps each.")
print(f"Total energy absorbed: {energy_absorbed:.4e} GJ")
initial_energy_density = Cv*T0
final_energy_density = Cv*T0 + energy_absorbed / L
print(f"Initial energy density: {initial_energy_density:.4e} GJ/cm³")
print(f"Final energy density: {final_energy_density:.4e} GJ/cm³")
print(f"Final temperature: {(final_energy_density/Cv):.4f} keV")
#now we have all the paths, we can plot them with very light lines and alpha blending
import matplotlib.pyplot as plt
#create a figure and axis
fig, ax = plt.subplots(figsize=(6, 4.5))
xL = 0.0; xR = L
tn = 0.0; tnp1 = ðt
cell = FancyBboxPatch(
    (xL, tn), xR - xL, tnp1 - tn,
    boxstyle="round,pad=0.0,rounding_size=0.001",
    linewidth=0.95,
    edgecolor=INK,
    facecolor=PANEL_FILL
)
ax.add_patch(cell)
#the first step is one color, the second step is another color, and so on. We can use a colormap to assign colors based on the step number.
#use the paired colormap from matplotlib, which goes from blue to orange
#pairwise is a qualitative colormap with 12 distinct colors, so we can cycle through it if we have more than 12 steps in a path. We can use the step number modulo 12 to index into the colormap.
step_colors = plt.cm.Set1(np.arange(10))
for path in all_paths:
    curr_point = path[0]
    for step_num, next_point in enumerate(path[1:]):
        x_vals = [curr_point[0], next_point[0]]
        t_vals = [curr_point[1], next_point[1]]
        ax.plot(x_vals, t_vals, color=step_colors[step_num], alpha=0.5, linewidth=0.95, solid_capstyle='round')
        #if the step is a collision, we can add a marker at the collision point
        #a collision occurs if the path ends inside the cell
        #if first step, draw a filled circle at the starting point
        if step_num == 0:

            if np.isclose(curr_point[1], 0.0):
                #make the fill of the marker whiteand the border black
                ax.plot(curr_point[0], curr_point[1], marker='^', markersize=4, alpha=0.8, color=INK, markerfacecolor='white', markeredgewidth=0.5, zorder=5)
            else:
                ax.plot(curr_point[0], curr_point[1], marker='o', color=INK, markersize=4, alpha=0.8, markerfacecolor='white', markeredgewidth=0.5, zorder=5)
        if next_point[1] < tnp1 and (next_point[0] > xL and next_point[0] < xR):
            ax.plot(next_point[0], next_point[1], marker='*', color=INK, markersize=6, alpha=0.8,  markerfacecolor='white', markeredgewidth=0.5, zorder=5)
        #if we are on a x boundary, we can add a marker at the boundary point
        if np.isclose(next_point[0], xL) or np.isclose(next_point[0], xR):
            ax.plot(next_point[0], next_point[1], marker='s', color=INK, markersize=4, alpha=0.8, markerfacecolor='white', markeredgewidth=0.5, zorder=5)
        #if we have reached the end of the timestep, we can add a marker at the end point
        if np.isclose(next_point[1], tnp1):
            ax.plot(next_point[0], next_point[1], marker='D', color=INK, markersize=4, alpha=0.8, markerfacecolor='white', markeredgewidth=0.5,  zorder=5)
        curr_point = next_point
# add colorbar for the step number
norm = mpl.colors.Normalize(vmin=1, vmax=10)
sm = plt.cm.ScalarMappable(cmap=plt.cm.Set1, norm=norm)
sm.set_array([])
plt.ylabel("time (ns)")
plt.xlabel("position (cm)")
#center colorbar vertically on the plot
plt.colorbar(sm, label="step number", cax=ax.inset_axes([1.0, 0.1, 0.02, 0.8]))
plt.ylim(-0.25 * ðt, ðt * 1.25)
plt.xlim(-0.25 * L, L * 1.25)
#only put ticks at x=0 and x=L, and t=0 and t=ðt
plt.xticks([0, L])
plt.yticks([0, ðt])
#remove outer spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.plot([-0.25 * L, 0], [0, 0], color=PROJ, lw=1.9, ls=(0, (6, 6)))
ax.plot([-0.25 * L, 0], [ðt, ðt], color=PROJ, lw=1.9, ls=(0, (6, 6)))
ax.plot([0,0], [-0.25 * ðt, 0], color=PROJ, lw=1.9, ls=(0, (6, 6)))
ax.plot([L, L], [-0.25 * ðt, 0], color=PROJ, lw=1.9, ls=(0, (6, 6)))
#place a legend across the bottom showing what the symbols mean
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='^', color='none', markerfacecolor='white', markeredgecolor=INK, markeredgewidth=0.5, markersize=6, label='initial particle'),
    Line2D([0], [0], marker='o', color='none', markerfacecolor='white', markeredgecolor=INK, markeredgewidth=0.5, markersize=6, label='source particle'),
    Line2D([0], [0], marker='*', color='none', markerfacecolor='white', markeredgecolor=INK, markeredgewidth=0.5, markersize=8, label='collision'),
    #Line2D([0], [0], marker='s', color='none', markerfacecolor='white', markeredgecolor=INK, markeredgewidth=0.5, markersize=6, label='boundary reflection'),
    #Line2D([0], [0], marker='D', color='none', markerfacecolor='white', markeredgecolor=INK, markeredgewidth=0.5, markersize=6, label='end of timestep'),
]
#put a white box behind the legend
leg1 = ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.9745), ncol=3,
          frameon=True, facecolor='white', edgecolor='none', fontsize=10, handletextpad=0.5, columnspacing=1.0)
ax.add_artist(leg1)
legend_elements = [
    #Line2D([0], [0], marker='^', color='none', markerfacecolor='white', markeredgecolor=INK, markeredgewidth=0.5, markersize=6, label='initial particle'),
    #Line2D([0], [0], marker='o', color='none', markerfacecolor='white', markeredgecolor=INK, markeredgewidth=0.5, markersize=6, label='source particle'),
    #Line2D([0], [0], marker='*', color='none', markerfacecolor='white', markeredgecolor=INK, markeredgewidth=0.5, markersize=8, label='collision'),
    Line2D([0], [0], marker='s', color='none', markerfacecolor='white', markeredgecolor=INK, markeredgewidth=0.5, markersize=6, label='boundary reflection'),
    Line2D([0], [0], marker='D', color='none', markerfacecolor='white', markeredgecolor=INK, markeredgewidth=0.5, markersize=6, label='end of timestep'),
]
#put a white box behind the legend
ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.9275), ncol=2,
          frameon=True, facecolor='white', edgecolor='none', fontsize=10, handletextpad=0.5, columnspacing=1.0)
plt.tight_layout()
plt.savefig("IMC_demo.pdf", dpi=600)
plt.show()
