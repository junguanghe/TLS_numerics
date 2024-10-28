
from calc_2nd_self_energy import *
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use("tkagg")

import h5py

from parameters import *

SE_eps_now = []
SE_delta_now = []

# Get the initial SE
for n in ns:
    se = SE_fn(None, n, T, E, Delta, Gamma, Nge)
    SE_eps_now.append(se[0])
    SE_delta_now.append(se[1])

with h5py.File("SE-Matsubara.h5", "a") as f:
    f["Iter0/epsilon"] = SE_eps_now
    f["Iter0/Delta"] = SE_delta_now


for iter in range(1, numIter+1):
    print(f"Computing Iter{iter}...")
    SE_now = SelfEnergy(T, E, Delta, Gamma, Nge, np.array(ns), SE_eps_now, SE_delta_now)
    for idx, n in enumerate(ns):
        se = SE_now.eval(n)
        SE_eps_now[idx] = se[0]
        SE_delta_now[idx] = se[1] 

    # plt.plot(ns, np.imag(SE_eps_now), label=rf"$\Im\Sigma_\epsilon$, Iter{iter}")
    # plt.plot(ns, np.real(SE_delta_now), label=rf"$\Re\Sigma_\Delta$ (real), Iter{iter}")

    
    with h5py.File("SE-Matsubara.h5", "a") as f:
        f[f"Iter{iter}/epsilon"] = SE_eps_now
        f[f"Iter{iter}/Delta"] = SE_delta_now





# plt.legend()
# plt.grid()
# plt.savefig("../figures/SE-self-consistent.pdf")
