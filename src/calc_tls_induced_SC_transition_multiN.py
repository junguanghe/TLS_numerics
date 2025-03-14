from calc_renormalized_self_energy import det_Mnl_minus_delta_nl

from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root_scalar
from tqdm import tqdm

EULER = 0.5772156649
CONST = 2.0 * np.exp(EULER) / np.pi

if __name__ == "__main__":

    def compute_root(e, N):
        result = root_scalar(lambda t: det_Mnl_minus_delta_nl(N, e, t), bracket=[1e-100, 0.04])
        return result.root

    def solve_gap_eq(e):
        result = root_scalar(lambda t: CONST * e * np.exp(-np.pi * e / np.tanh(e / 2 / t)) - t, x0=0.01)
        return result.root

    es = np.linspace(0.0001, 2.505, 100)
    es1 = np.linspace(0.0001, 0.3183, 100)
    es10 = np.linspace(0.0001, 1.0395, 100)
    es100 = np.linspace(0.0001, 1.75, 100)
    es500 = np.linspace(0.0001, 2.105, 100)

    # Use joblib to parallelize the root finding
    tcs1 = Parallel(n_jobs=100, verbose=20)(delayed(compute_root)(e, 1) for e in es1)
    tcs10 = Parallel(n_jobs=100, verbose=20)(delayed(compute_root)(e, 10) for e in es10)
    tcs100 = Parallel(n_jobs=100, verbose=20)(delayed(compute_root)(e, 100) for e in es100)
    tcs500 = Parallel(n_jobs=100, verbose=20)(delayed(compute_root)(e, 500) for e in es500)
    tcs1000 = Parallel(n_jobs=100, verbose=20)(delayed(compute_root)(e, 1000) for e in es)
    tcs2000 = Parallel(n_jobs=100, verbose=20)(delayed(compute_root)(e, 2000) for e in es)

    fig, ax = plt.subplots()
    ax.plot(es1, tcs1, label="Full Solution, N=1", marker='o')
    ax.plot(es10, tcs10, label="Full Solution, N=10", marker='o')
    ax.plot(es100, tcs100, label="Full Solution, N=100", marker='o')
    ax.plot(es500, tcs500, label="Full Solution, N=500", marker='o')
    ax.plot(es, tcs1000, label=f"Full Solution, N=1000", marker='o')
    ax.plot(es, tcs2000, label="Full Solution, N=2000", marker='o')
    ax.set_xlabel(r"$E/(\hbar/\tau_{in})$")
    ax.set_ylabel(r"$T_c^{tls}/(\hbar/\tau_{in})$")
    ax.legend()
    ax.grid()
    
    fig.savefig("../figures/Tc_tls_weak_coupling_multiple_N_1.pdf")

    # save data to ../data/tls_induced_SC_transition_multiN/ as txt files
    np.savetxt("../data/tls_induced_SC_transition_multiN/tcs1.txt", np.array([es1, tcs1]))
    np.savetxt("../data/tls_induced_SC_transition_multiN/tcs10.txt", np.array([es10, tcs10]))
    np.savetxt("../data/tls_induced_SC_transition_multiN/tcs100.txt", np.array([es100, tcs100]))
    np.savetxt("../data/tls_induced_SC_transition_multiN/tcs500.txt", np.array([es500, tcs500]))
    np.savetxt("../data/tls_induced_SC_transition_multiN/tcs1000.txt", np.array([es, tcs1000]))
    np.savetxt("../data/tls_induced_SC_transition_multiN/tcs2000.txt", np.array([es, tcs2000]))