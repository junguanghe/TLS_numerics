from joblib import Parallel, delayed
import numpy as np
from scipy import integrate
from scipy.integrate import quad, dblquad
from scipy.stats import norm
import warnings

from calc_gap_profile import solve_d, INV_TVV, INV_TVM, INV_TMM, INV_TNN


t_ = 0.9  # calculate dos at temperature t = T/T_c0
teff_ = None  # calculate dos at effective temperature teff = T_eff/T_c0. set to None if no effective temperature
energy_range_ = 5  # -energy_range < epsilon < energy_range

E1, E2 = 0.237, 3  # sample energy splitting between E1 and E2
J = 0.237
SPREAD = 2.495
def p(x):
    if x == J:
        return np.inf
    return x / np.sqrt(x * x - J * J) / (SPREAD * SPREAD + x * x - J * J)
NORM = quad(p, E1, E2)[0]
DIST = f"wipf_{E1}_{E2}_{J}_{SPREAD}"

N = 2000  # number of data points
DELTA = 1e-5  # the small imaginary part in the retarded energy, i.e. i*en -> epsilon + i*DELTA
LIMIT = 5000  # number of subdivisions in scipy quad

D = solve_d(t_, J, teff_, 1.7, 1.71)
print(f"Delta(T={t_}, e={J}, teff={teff_}) = {D}")

original_teff_ = teff_
if teff_ is None:
    teff_ = t_
avg_Nge = quad(lambda x: p(x) / NORM * np.tanh(x / 2 / teff_), E1, E2)[0]


def func_e_real(xi, e, er):
    y = np.sign(xi) * np.sqrt(xi * xi + D * D)
    numer = 1 - np.tanh(e / 2 / teff_) * np.tanh(y / 2 / t_)
    denom = (y - e) * (y - e) - er * er
    ret = numer / denom
    return ret.real * p(e) / NORM


def func_d_real(xi, e, er):
    y = np.sign(xi) * np.sqrt(xi * xi + D * D)
    numer = 1 - np.tanh(e / 2 / teff_) * np.tanh(y / 2 / t_)
    denom = (y - e) * (y - e) - er * er
    ret = numer / denom * (y - e) / y
    return ret.real * p(e) / NORM


def rho(y):
    if np.abs(y) <= D:
        return 0
    return np.abs(y) / np.sqrt(y * y - D * D)


def sigma_d(ep):
    er = ep + 1j * DELTA
    ret = (INV_TVV + 2 * avg_Nge * INV_TVM + INV_TMM) * np.pi / np.sqrt(D * D - er * er)
    int_real1 = dblquad(func_d_real, E1, E2, -np.inf, 0, args=(er,))
    int_real2 = dblquad(func_d_real, E1, E2, 0, np.inf, args=(er,))
    to_int = lambda x: (rho(ep + x) / (ep + x) * (1 - np.tanh(x / 2 / teff_) * np.tanh((ep + x) / 2 / t_))
                        - rho(-ep + x) / (-ep + x) * (1 - np.tanh(x / 2 / teff_) * np.tanh((-ep + x) / 2 / t_))
                        ) * p(x) / NORM
    int_res = quad(to_int, E1, E2, limit=LIMIT)
    int_imag = np.pi / 2 * int_res[0]
    err = int_real1[1] + int_real2[1] + int_res[1]
    int_real = int_real1[0] + int_real2[0]
    return ret + INV_TNN * (int_real + 1j * int_imag), err


def neg_e_times_sigma_e(ep):
    er = ep + 1j * DELTA
    ret = (INV_TVV + 2 * avg_Nge * INV_TVM + INV_TMM) * np.pi * ep / np.sqrt(D * D - er * er)
    int_real1 = dblquad(func_e_real, E1, E2, -np.inf, 0, args=(er,))
    int_real2 = dblquad(func_e_real, E1, E2, 0, np.inf, args=(er,))
    to_int = lambda x: (rho(ep + x) * (1 - np.tanh(x / 2 / teff_) * np.tanh((ep + x) / 2 / t_))
                        + rho(-ep + x) * (1 - np.tanh(x / 2 / teff_) * np.tanh((-ep + x) / 2 / t_))
                        ) * p(x) / NORM
    int_res = quad(to_int, E1, E2, limit=LIMIT)
    int_imag = np.pi / 2 * int_res[0]
    err = int_real1[1] + int_real2[1] + int_res[1]
    int_real = ep * (int_real1[0] + int_real2[0])
    return ret + INV_TNN * (int_real + 1j * int_imag), err


def dos(ep):
    warnings.filterwarnings("ignore", category=integrate.IntegrationWarning)
    neg_e_se, err1 = neg_e_times_sigma_e(ep)
    sd, err2 = sigma_d(ep)
    etilde = ep + neg_e_se
    dtilde = D * (1 + sd)
    err = err1 + err2
    return np.imag(etilde / np.sqrt(dtilde**2 - (etilde + 1j * DELTA) ** 2)), err


def main():
    # x[0][:] -> epsilon, x[1][:] -> DOS, x[2][:] -> scipy quad error
    x = np.zeros((3, N))
    x[0] = np.linspace(-energy_range_, energy_range_, N)
    x[1], x[2] = zip(*Parallel(n_jobs=100, verbose=10)(delayed(dos)(x[0][i]) for i in range(N)))
    
    np.savetxt(f"./data/dos_data_with_dist_of_e/dist={DIST}_t={t_}_teff={original_teff_}.txt", x)


if __name__ == "__main__":
    main()
