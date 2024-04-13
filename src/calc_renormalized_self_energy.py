import matplotlib.pyplot as plt
import numpy as np


def i_sigma_epsilon(Nge: float, n: int, e: float, t: float) -> float:
    sign = 1 if n >= 0 else -1
    n = n if n >= 0 else -n - 1

    # -n, -n+1, ..., -1, 0, 1, ..., n-1, n
    m = np.arange(-n, n + 1)
    wm = 2 * m * np.pi * t
    return sign * Nge * t * e * np.sum(1 / (e * e + wm * wm))


def det_Mnl_minus_delta_nl(N: int, e: float, t: float) -> float:
    Nge = np.tanh(e / (2 * t))
    
    # -N, -N+1, ..., -2, -1 | 0, 1, 2, ..., N-1
    l = np.arange(-N, N)
    i_sigma_eps = np.vectorize(i_sigma_epsilon)(Nge, l, e, t)
    n = np.arange(-N, N).reshape(-1, 1)
    Mnl = (
        Nge * e * t
        / (e * e + 4 * (n - l) ** 2 * np.pi * np.pi * t * t)
        / np.abs((2 * l + 1) * np.pi * t + i_sigma_eps)
    )
    delta_nl = np.diag(np.ones(2 * N))
    return np.linalg.det(Mnl - delta_nl)


if __name__ == "__main__":
    N = 1000  # number of Matsubara frequencies
    
    e = 0.1
    t = 0.1
    print(f"det(e={e}, t={t}) = {det_Mnl_minus_delta_nl(N, e, t)}")

    # es = np.linspace(0.1, 1.0, 100)
    # ts = np.linspace(0.1, 1.0, 100)
    # dets = np.array([[det_Mnl_minus_delta_nl(N, e, t) for e in es] for t in ts])