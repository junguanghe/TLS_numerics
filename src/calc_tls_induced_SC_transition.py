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

    N = 1000
    es = np.linspace(0.0001, 2.505, 100)

    # Use joblib to parallelize the root finding
    tcs = Parallel(n_jobs=50, verbose=20)(delayed(compute_root)(e, N) for e in es)
    # tcs1 = Parallel(n_jobs=50, verbose=20)(delayed(solve_gap_eq)(e, N) for e in es)
    tcs1 = [solve_gap_eq(e) for e in tqdm(es)]

    fig, ax = plt.subplots()
    ax.plot(es, tcs, label="Full Solution")
    ax.plot(es, tcs1, label="Weak Coupling Solution")
    ax.text(1.0, 0.08, r"$T_c^{tls,wc} = 1.13 E \exp(-\frac{\pi E}{N_{ge}\hbar/\tau_{in}})$", fontsize=15)
    ax.set_xlabel(r"$E/(\hbar/\tau_{in})$")
    ax.set_ylabel(r"$T_c^{tls}/(\hbar/\tau_{in})$")
    ax.legend()
    ax.grid()
    
    fig.savefig("../figures/Tc_tls_weak_coupling.pdf")
