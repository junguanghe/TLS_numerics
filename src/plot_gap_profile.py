import matplotlib.pyplot as plt
import numpy as np

E = [0, 0.5, 1.0, 1.5, 2.0]
Teff = 10

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

loc = "./data/gap_data/"

fig, ax = plt.subplots(figsize=(4, 3))

for e,c in zip(E, colors):
    x = np.loadtxt(loc + f"e={e}_Teff=None.txt")
    ax.plot(x[0], x[1], label=r"$E/T_{c_0}=$" + str(e), color=c)
    x1 = np.loadtxt(loc + f"e={e}_Teff={Teff}.txt")
    ax.plot(x1[0], x1[1], linestyle='--', color=c)

ax.legend(loc="lower left")
ax.set_ylabel(r"$\Delta/T_{c_0}$")
ax.set_xlabel(r"$T/T_{c_0}$")
ax.grid(True)
fig.tight_layout()
# plt.show()


dir = "./figures/gap_profile/"
fig.savefig(dir + f"gap_profile_Teff={Teff}.pdf", bbox_inches="tight")
