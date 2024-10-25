
from calc_2nd_self_energy import *
from parameters import *
import matplotlib.pyplot as plt

import h5py

with h5py.File("SE-Matsubara.h5", "r") as f:
    SE_eps_now = f[f"Iter{numIter}/epsilon"][:]
    SE_delta_now = f[f"Iter{numIter}/Delta"][:]

SE = SelfEnergy(T, E, Delta, Gamma, Nge, np.array(ns), SE_eps_now, SE_delta_now)


def DOS(e):
    se = SE.eval_real_freq(e)
    # se = [0., 0.]
    res = (e+1e-10j - se[0])/np.sqrt((Delta+se[1])**2 - (e+1e-10j-se[0])**2)
    return np.imag(res)


es = np.arange(-2*Delta, 2*Delta, step=0.05)
DOSs = []
for e in es:
    DOSs.append(DOS(e))

plt.plot(es, DOSs)
plt.show()
