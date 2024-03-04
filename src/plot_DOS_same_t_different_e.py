import matplotlib.pyplot as plt
import numpy as np

from calc_gap_profile import solve_d

# E = [0, 0.1, 0.2, 0.3, 0.4]
# t = 0.1
E = [0, 0.4, 1.2]
t = 0.9
teff = None
XMAX = 3 # set matplotlib x-axis limit to be -XMAX <= epsilon/T_c0 <= XMAX
YMAX = 10 # set matplotlib y-axis limit to be N/N(0) <= YMAX
figure_size = (4,3)

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

fig, ax = plt.subplots(figsize=figure_size)

for i,e in enumerate(E):
    D = solve_d(t, e, teff, 1.7, 1.71)
    c = colors[i]
    x = np.loadtxt(f"./data/dos_data/e={e}_t={t}_teff={teff}.txt")
    ax.plot(x[0], x[1], label=r'$E/T_{c_0}=$'+str(e)+r', $T/T_{c_0}=$' + str(t), color = c)
    ax.vlines(x=[D, -D], ymin=-0.5, ymax=YMAX, ls='--', color=c)
    

ax.legend()
# ax.legend(loc='upper center')
ax.set_ylim(-0.5, YMAX)
ax.set_xlim(-XMAX,  XMAX)
ax.set_ylabel(r'$N/N(0)$')
ax.set_xlabel(r'$\varepsilon/T_{c_0}$')
ax.grid(True)
fig.tight_layout()
# plt.show()

fig.savefig(f"./figures/dos/dos_same_t_different_e_teff={teff}_t={t}.pdf", bbox_inches='tight')