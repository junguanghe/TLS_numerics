from joblib import Parallel, delayed, cpu_count
import numpy as np
from scipy.integrate import quad, dblquad
from scipy.optimize import root_scalar
from scipy.stats import norm

EPSILON = 1e-7  # the cutoff for Matsubara sum

INV_TVV = 0  # the inverse scattering times
INV_TVM = 0
INV_TMM = 4 / 15 * 1e-4
INV_TNN = 1 / 6


E1, E2 = 0.1, 0.3  # sample energy splitting between e1 and e2
MU = 0.2
SIG = 0.05
NORM = norm.cdf(E2, MU, SIG) - norm.cdf(E1, MU, SIG)
DIST = f"gaussian_{E1}_{E2}_{MU}_{SIG}"


# integrand in Sigma_epsilon
def func_e(xi, e, D, t, en):
    y = np.sign(xi) * np.sqrt(xi * xi + D * D)
    numer = 1 - np.tanh(e / 2 / t) * np.tanh(y / 2 / t)
    denom = (y - e) * (y - e) + en * en
    return numer / denom * norm.pdf(e, MU, SIG) / NORM


# integrand in Sigma_Delta
def func_d(xi, e, D, t, en):
    y = np.sign(xi) * np.sqrt(xi * xi + D * D)
    numer = 1 - np.tanh(e / 2 / t) * np.tanh(y / 2 / t)
    denom = (y - e) * (y - e) + en * en
    return numer / denom * (y - e) / y * norm.pdf(e, MU, SIG) / NORM


# i*Sigma_omega / omega_n
def sigma_e(D, t, en, teff):
    Nge = quad(lambda x: norm.pdf(x, MU, SIG) * np.tanh(x / 2 / teff), E1, E2)[0]
    ret = (INV_TVV + 2 * Nge * INV_TVM + INV_TMM) * np.pi / np.sqrt(en * en + D * D)
    ret += INV_TNN * (
        dblquad(func_e, E1, E2, -np.inf, 0, args=(D, t, en))[0]
        + dblquad(func_e, E1, E2, 0, np.inf, args=(D, t, en))[0]
    )
    return ret


# Sigma_Delta/Delta
def sigma_d(D, t, en, teff):  # should have D>0
    Nge = quad(lambda x: norm.pdf(x, MU, SIG) * np.tanh(x / 2 / teff), E1, E2)[0]
    ret = (INV_TVV + 2 * Nge * INV_TVM + INV_TMM) * np.pi / np.sqrt(en * en + D * D)
    ret += INV_TNN * (
        dblquad(func_d, E1, E2, -np.inf, 0, args=(D, t, en))[0]
        + dblquad(func_d, E1, E2, 0, np.inf, args=(D, t, en))[0]
    )
    return ret


# the self-consistent equation
# Delta can't be 0 or too small
def SCE(t, D, teff):
    if teff is None:
        teff = t
    rhs = 0
    n = 0
    en = (2 * n + 1) * np.pi * t
    fe = 1 + sigma_e(D, t, en, teff)
    fd = 1 + sigma_d(D, t, en, teff)
    cur = fd / np.sqrt(en * en * fe * fe + D * D * fd * fd) - 1 / en
    cur *= 2  # even in n, so only sum over n>=0 and multiply by 2
    rhs += cur
    while np.abs(cur / rhs) > EPSILON:
        n += 1
        en = (2 * n + 1) * np.pi * t
        fe = 1 + sigma_e(D, t, en, teff)
        fd = 1 + sigma_d(D, t, en, teff)
        cur = fd / np.sqrt(en * en * fe * fe + D * D * fd * fd) - 1 / en
        cur *= 2  # even in n, so only sum over n>=0 and multiply by 2
        rhs += cur
    return np.pi * t * rhs - np.log(t)


# given temperature, solve the self-consistent equation for the gap
def solve_d(t, teff, x0, x1):
    res = root_scalar(lambda x: SCE(t, x, teff), x0=x0, x1=x1)
    if not res.converged:
        print(f"T/Tc0={t} not converged")
    return res.root


# given gap, solve the self-consistent equation for the temperature
def solve_t(D, teff, x0, x1):
    res = root_scalar(lambda x: SCE(x, D, teff), x0=x0, x1=x1)
    if not res.converged:
        print(f"D/Tc0={D} not converged")
    return res.root


def main():
    teff = 10  # set to None for teff = t
    N = 50  # number of data points for the gap profile

    num_core = cpu_count()

    x = np.zeros((2, N))
    d0, d1 = 1.7, 1.71  # initial guess
    x[0][: N // 2] = np.linspace(0.01, 0.7, N // 2)  # x[0][:] is t = Tc/Tc0
    x[1][: N // 2] = Parallel(n_jobs=num_core, verbose=20)(
        delayed(solve_d)(x[0][i], teff, d0, d1) for i in range(N // 2)
    )
    t0, t1 = 0.9, 0.91  # initial guess
    d_start = x[1][N // 2 - 1] * 2 - x[1][N // 2 - 2]
    x[1][N // 2 :] = np.linspace(d_start, 0.01, N - N // 2)  # x[1][:] is Delta/Tc0
    x[0][N // 2 :] = Parallel(n_jobs=num_core, verbose=20)(
        delayed(solve_t)(x[1][i], teff, t0, t1) for i in range(N // 2, N)
    )

    loc = "./data/gap_data_with_dist_of_e/"
    np.savetxt(loc + f"Teff={teff}_{DIST}.txt", x)


if __name__ == "__main__":
    main()
