from joblib import Parallel, delayed, cpu_count
import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar

EPSILON = 1e-7  # the cutoff for Matsubara sum

INV_TVV = 0  # the inverse scattering times
INV_TVM = 0
INV_TMM = 4 / 15 * 1e-4
INV_TNN = 1 / 6


# integrand in Sigma_epsilon
def func_e(xi, e, D, t, en, Nge):
    y = np.sign(xi) * np.sqrt(xi * xi + D * D)
    numer = 1 - Nge * np.tanh(y / 2 / t)
    denom = (y - e) * (y - e) + en * en
    return numer / denom


# integrand in Sigma_Delta
def func_d(xi, e, D, t, en, Nge):
    y = np.sign(xi) * np.sqrt(xi * xi + D * D)
    numer = 1 - Nge * np.tanh(y / 2 / t)
    denom = (y - e) * (y - e) + en * en
    return numer / denom * (y - e) / y


# i*Sigma_omega / omega_n
def sigma_e(D, t, en, e, teff):
    Nge = np.tanh(e / 2 / teff)
    ret = (INV_TVV + 2 * Nge * INV_TVM + INV_TMM) * np.pi / np.sqrt(en * en + D * D)
    ret += INV_TNN * (
        quad(func_e, -np.inf, 0, args=(e, D, t, en, Nge))[0]
        + quad(func_e, 0, np.inf, args=(e, D, t, en, Nge))[0]
    )
    return ret


# Sigma_Delta/Delta
def sigma_d(D, t, en, e, teff):  # should have D>0
    Nge = np.tanh(e / 2 / teff)
    ret = (INV_TVV + 2 * Nge * INV_TVM + INV_TMM) * np.pi / np.sqrt(en * en + D * D)
    ret += INV_TNN * (
        quad(func_d, -np.inf, 0, args=(e, D, t, en, Nge))[0]
        + quad(func_d, 0, np.inf, args=(e, D, t, en, Nge))[0]
    )
    return ret


# the self-consistent equation
# Delta can't be 0 or too small
def SCE(t, e, D, teff):
    if teff is None:
        teff = t
    rhs = 0
    n = 0
    en = (2 * n + 1) * np.pi * t
    fe = 1 + sigma_e(D, t, en, e, teff)
    fd = 1 + sigma_d(D, t, en, e, teff)
    cur = fd / np.sqrt(en * en * fe * fe + D * D * fd * fd) - 1 / en
    cur *= 2  # even in n, so only sum over n>=0 and multiply by 2
    rhs += cur
    while np.abs(cur / rhs) > EPSILON:
        n += 1
        en = (2 * n + 1) * np.pi * t
        fe = 1 + sigma_e(D, t, en, e, teff)
        fd = 1 + sigma_d(D, t, en, e, teff)
        cur = fd / np.sqrt(en * en * fe * fe + D * D * fd * fd) - 1 / en
        cur *= 2  # even in n, so only sum over n>=0 and multiply by 2
        rhs += cur
    return np.pi * t * rhs - np.log(t)


# given temperature, solve the self-consistent equation for the gap
def solve_d(t, e, teff, x0, x1):
    res = root_scalar(lambda x: SCE(t, e, x, teff), x0=x0, x1=x1, rtol=1e-4)
    if not res.converged:
        print(f"T/Tc0={t} not converged")
    return res.root


# given gap, solve the self-consistent equation for the temperature
def solve_t(D, e, teff, x0, x1):
    res = root_scalar(lambda x: SCE(x, e, D, teff), x0=x0, x1=x1)
    if not res.converged:
        print(f"D/Tc0={D} not converged")
    return res.root


def main():
    e = 2.0  # energy splitting e = E/Tc0
    teff = 10  # set to None for teff = t
    N = 50  # number of data points for the gap profile

    num_core = cpu_count()

    x = np.zeros((2, N))
    d0, d1 = 1.7, 1.71  # initial guess
    x[0][: N // 2] = np.linspace(0.01, 0.7, N // 2)  # x[0][:] is t = Tc/Tc0
    x[1][: N // 2] = Parallel(n_jobs=num_core, verbose=20)(
        delayed(solve_d)(x[0][i], e, teff, d0, d1) for i in range(N // 2)
    )
    t0, t1 = 0.9, 0.91  # initial guess
    d_start = x[1][N // 2 - 1] * 2 - x[1][N // 2 - 2]
    x[1][N // 2 :] = np.linspace(d_start, 0.01, N - N // 2)  # x[1][:] is Delta/Tc0
    x[0][N // 2 :] = Parallel(n_jobs=num_core, verbose=20)(
        delayed(solve_t)(x[1][i], e, teff, t0, t1) for i in range(N // 2, N)
    )

    loc = "./data/gap_data/"
    np.savetxt(loc + f"e={e}_Teff={teff}.txt", x)


if __name__ == "__main__":
    main()
