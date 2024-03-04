import matplotlib.pyplot as plt
import numpy as np

E = [0.4, 1.5]
t = 0.1
teff = 2
XMAX = 3 # set matplotlib x-axis limit to be (-XMAX, XMAX)
YMAX = 10 # set matplotlib y-axis limit to be (-0.5, YMAX)
figure_size = (4,3)

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

fig, ax = plt.subplots(figsize=figure_size)

for i,e in enumerate(E):
    c = colors[i]
    x = np.loadtxt(f"./data/dos_data/e={e}_t={t}_teff={teff}.txt")
    ax.plot(x[0], x[1], label=r'$E/T_{c_0}=$'+str(e)+r', $T_{eff}/T_{c_0}=$' + str(teff), color = c)
    x = np.loadtxt(f"./data/dos_data/e={e}_t={t}_teff={None}.txt")
    ax.plot(x[0], x[1], label=r'$E/T_{c_0}=$'+str(e)+r', $T_{eff}/T_{c_0}=$' + str(None), color = c, ls='--')
    

ax.legend()
# ax.legend(loc='upper center')
ax.set_ylim(-0.5, YMAX)
ax.set_xlim(-XMAX,  XMAX)
ax.set_ylabel(r'$N/N(0)$')
ax.set_xlabel(r'$\varepsilon/T_{c_0}$')
ax.grid(True)
fig.tight_layout()
# plt.show()

fig.savefig(f"./figures/dos/dos_with_vs_without_teff_t={t}.pdf", bbox_inches='tight')