import matplotlib.pyplot as plt
import numpy as np

E1, E2 = 0.1, 0.3  # sample energy splitting between e1 and e2
MU = 0.2
SIG = 0.05
DIST = f"gaussian_{E1}_{E2}_{MU}_{SIG}"

t = 0.1
teff = None
XMAX = 3 # set matplotlib x-axis limit to be (-XMAX, XMAX)
YMAX = 10 # set matplotlib y-axis limit to be (-0.5, YMAX)
figure_size = (4,3)

fig, ax = plt.subplots(figsize=figure_size)

x = np.loadtxt(f"./data/dos_data_with_dist_of_e/dist={DIST}_t={t}_teff={teff}.txt")
ax.plot(x[0], x[1], label=r'$T_{eff}/T_{c_0}=$' + str(teff))

ax.legend()
# ax.legend(loc='upper center')
ax.set_ylim(-0.5, YMAX)
ax.set_xlim(-XMAX,  XMAX)
ax.set_ylabel(r'$N/N(0)$')
ax.set_xlabel(r'$\varepsilon/T_{c_0}$')
ax.grid(True)
fig.tight_layout()
# plt.show()

fig.savefig(f"./figures/dos_with_dist_of_e/t={t}_teff={teff}.pdf", bbox_inches='tight')