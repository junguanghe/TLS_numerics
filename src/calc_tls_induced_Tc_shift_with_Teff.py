from calc_tls_induced_Tc_shift import solve_gap_eq

from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    N = 1000  # number of Matsubara frequencies
    teffs = [None, 0.1, 1, 10, 100, 1000]
    inv_tau = 0.1
    e0 = 0.0001
    e1 = 100
    
    es = np.linspace(e0, e1, 100)
    fig, ax = plt.subplots()
    tcs = []
    for teff in teffs:
        tcs.append(Parallel(n_jobs=50, verbose=20)(delayed(solve_gap_eq)(e, inv_tau, teff) for e in es))
        ax.plot(es, tcs[-1], label=r"$T_{eff}/T_{c_0} = $" + f"{teff}")
    ax.set_xlabel(r"$E/T_{c_0}$")
    ax.set_ylabel(r"$T_c/T_{c_0}$")
    ax.grid()
    ax.legend()
    fig.savefig(f"../figures/Tc_shift_es_{e0}_{e1}_inv_tau_{inv_tau}.pdf")