import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

from calc_gap_profile import solve_d

E1, E2 = 0.237, 3  # sample energy splitting between E1 and E2
J = 0.237
SPREAD = 2.495
def p(x):
    return x / np.sqrt(x * x - J * J) / (SPREAD * SPREAD + x * x - J * J)
NORM = quad(p, E1, E2)[0]
DIST = f"wipf_{E1}_{E2}_{J}_{SPREAD}"

t = 0.1
teff = 2
XMAX = 3 # set matplotlib x-axis limit to be (-XMAX, XMAX)
YMAX = 10 # set matplotlib y-axis limit to be (-0.5, YMAX)
figure_size = (4,3)

D = solve_d(t, J, teff, 1.7, 1.71)
print(f"Delta(T={t}, e={J}, teff={teff}) = {D}")

fig, ax = plt.subplots(figsize=figure_size)

x = np.loadtxt(f"./data/dos_data_with_dist_of_e/dist={DIST}_t={t}_teff={teff}.txt")
ax.plot(x[0], x[1], label=r'$T/T_{c_0}=$' + str(t))
ax.vlines(x=[D, -D], ymin=-0.5, ymax=YMAX, ls='--')

inset_ax = fig.add_axes([0.25, 0.45, 0.6, 0.3])
inset_x = np.linspace(E1+1e-5, E2, 10000)
inset_ax.plot(inset_x, p(inset_x) / NORM, label=r'$\varepsilon_0=$' + str(SPREAD) + r', $J$=' + str(J))
inset_ax.set_xlabel(r'$E/T_{c_0}$')
inset_ax.set_ylabel('normalized pdf')
inset_ax.set_xlim(E1*0.98, E1*1.2)
inset_ax.legend()
inset_ax.grid(True)

ax.legend()
# ax.legend(loc='upper center')
ax.set_ylim(-0.5, YMAX)
ax.set_xlim(-XMAX,  XMAX)
ax.set_ylabel(r'$N/N(0)$')
ax.set_xlabel(r'$\varepsilon/T_{c_0}$')
ax.grid(True)
# fig.tight_layout()
# plt.show()

fig.savefig(f"./figures/dos_with_dist_of_e/dist={DIST}_t={t}_teff={teff}.pdf", bbox_inches='tight')