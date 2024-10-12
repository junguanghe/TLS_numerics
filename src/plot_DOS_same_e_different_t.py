import matplotlib.pyplot as plt
import numpy as np

from calc_gap_profile import solve_d

e = 0.4
T = [0.1, 0.3, 0.5, 0.7, 0.9]
teff = None
XMAX = 3 # set matplotlib x-axis limit to be (-XMAX, XMAX)
YMAX = 10 # set matplotlib y-axis limit to be (-0.5, YMAX)
figure_size = (4,3)

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "cm"
})
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

fig, ax = plt.subplots(figsize=figure_size)

for i,t in enumerate(T):
    D = solve_d(t, e, teff, 1.7, 1.71)
    c = colors[i]
    x = np.loadtxt(f"./data/dos_data/e={e}_t={t}_teff={teff}.txt")
    ax.vlines(x=[D, -D], ymin=-0.5, ymax=YMAX, ls='--', color=c, lw=0.6)
    ax.plot(x[0], x[1], label=r'$E/T_{c0}=$'+str(e)+r', $T/T_{c0}=$' + str(t), color = c, lw=0.8)
    

ax.legend(fontsize=6)
# ax.legend(loc='upper center')
ax.set_ylim(-0.5, YMAX)
ax.set_xlim(-XMAX,  XMAX)
ax.set_ylabel(r'$N/N(0)$')
ax.set_xlabel(r'$\varepsilon/T_{c0}$')
ax.grid(True)
ax.tick_params(direction="in")
fig.tight_layout()
# plt.show()

fig.savefig(f"./figures/dos/dos_same_e_different_t_teff={teff}_e={e}.pdf", bbox_inches='tight')
