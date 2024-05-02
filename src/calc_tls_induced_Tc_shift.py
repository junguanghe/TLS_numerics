from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve
from scipy.optimize import root
from tqdm import tqdm

def i_sigma_epsilon(Nge: float, n: int, e: float, t: float, inv_tau: float) -> float:
    sign = 1 if n >= 0 else -1
    n = n if n >= 0 else -n - 1

    # -n, -n+1, ..., -1, 0, 1, ..., n-1, n
    m = np.arange(-n, n + 1)
    wm = 2 * m * np.pi * t
    return inv_tau * sign * Nge * t * e * np.sum(1 / (e * e + wm * wm))

def sigmas(tc: float, e: float, inv_tau: float, Nge: float, N: int = 1000) -> float:
    # -N, -N+1, ..., -2, -1 | 0, 1, 2, ..., N-1
    l = np.arange(-N, N)
    i_sigma_eps = np.vectorize(i_sigma_epsilon)(Nge, l, e, tc, inv_tau)
    n = np.arange(-N, N).reshape(-1, 1)
    Mnl = (
        inv_tau * Nge * e * tc
        / (e * e + 4 * (n - l) ** 2 * np.pi * np.pi * tc * tc)
        / np.abs((2 * l + 1) * np.pi * tc + i_sigma_eps)
    )
    delta_nl = np.diag(np.ones(2 * N))
    return i_sigma_eps, solve(Mnl - delta_nl, -np.sum(Mnl, axis=1))

def gap_eq(tc: float, e: float, inv_tau: float, teff: float = None, N: int = 1000) -> float:
    Nge = np.tanh(e / 2 / tc) if teff is None else np.tanh(e / 2 / teff)
    i_sig_eps, sig_del_over_delta = sigmas(tc, e, inv_tau, Nge, N)
    n = np.arange(-N, N)
    en = (2 * n + 1) * np.pi * tc
    return np.pi * tc * np.sum((1 + sig_del_over_delta) / np.abs(en + i_sig_eps) - 1 / np.abs(en)) - np.log(tc)

def solve_gap_eq(e: float, inv_tau: float, teff: float = None, N: int = 1000) -> float:
    result = root(lambda t: gap_eq(t, e, inv_tau, teff, N), x0=1.0)
    return result.x[0]


if __name__ == "__main__":
    N = 1000  # number of Matsubara frequencies
    inv_taus = [0.2, 0.4, 0.6, 0.8, 1.0]
    e0 = 0.0001
    e1 = 10
    
    es = np.linspace(e0, e1, 100)
    fig, ax = plt.subplots()
    tcs = []
    for inv_tau in inv_taus:
        tcs.append(Parallel(n_jobs=50, verbose=20)(delayed(solve_gap_eq)(e, inv_tau) for e in es))
        ax.plot(es, tcs[-1], label=r"$\hbar/\tau_{in}T_{c_0} = $" + f"{inv_tau:.2f}")
    ax.set_xlabel(r"$E/T_{c_0}$")
    ax.set_ylabel(r"$T_c/T_{c_0}$")
    ax.grid()
    ax.legend()
    fig.savefig(f"../figures/Tc_shift_es_{e0}_{e1}.pdf")