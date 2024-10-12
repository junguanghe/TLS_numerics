import numpy as np
from scipy.integrate import quad
from tqdm import tqdm

from calc_gap_profile import solve_d, INV_TVV, INV_TVM, INV_TMM, INV_TNN


t_ = 0.1  # calculate dos at temperature t = T/T_c0
teff_ = None  # calculate dos at effective temperature teff = T_eff/T_c0. set to None if no effective temperature
e_ = 1.2  # the energy spliting of the TLS e = E/T_c0
energy_range_ = 5  # -energy_range < epsilon < energy_range

N = 2000  # number of data points
DELTA = 1e-5  # the small imaginary part in the retarded energy, i.e. i*en -> epsilon + i*DELTA
LIMIT = 5000  # number of subdivisions in scipy quad

if teff_ is None:
    Nge = np.tanh(e_ / 2 / t_)
else:
    Nge = np.tanh(e_ / 2 / teff_)

D = solve_d(t_, e_, teff_, 1.7, 1.71)
print(f"Delta(T={t_}, e={e_}, teff={teff_}) = {D}")


def func_e_real(xi, er):
    y = np.sign(xi) * np.sqrt(xi * xi + D * D)
    numer = 1 - Nge * np.tanh(y / 2 / t_)
    denom = (y - e_) * (y - e_) - er * er
    ret = numer / denom
    return ret.real


def func_e_imag(xi, er):
    y = np.sign(xi) * np.sqrt(xi * xi + D * D)
    numer = 1 - Nge * np.tanh(y / 2 / t_)
    denom = (y - e_) * (y - e_) - er * er
    ret = numer / denom
    return ret.imag


def func_d_real(xi, er):
    y = np.sign(xi) * np.sqrt(xi * xi + D * D)
    numer = 1 - Nge * np.tanh(y / 2 / t_)
    denom = (y - e_) * (y - e_) - er * er
    ret = numer / denom * (y - e_) / y
    return ret.real


def func_d_imag(xi, er):
    y = np.sign(xi) * np.sqrt(xi * xi + D * D)
    numer = 1 - Nge * np.tanh(y / 2 / t_)
    denom = (y - e_) * (y - e_) - er * er
    ret = numer / denom * (y - e_) / y
    return ret.imag


def rho(y):
    if np.abs(y) <= D:
        return 0
    return np.abs(y) / np.sqrt(y * y - D * D)


def sigma_d(ep):
    er = ep + 1j * DELTA
    ret = (INV_TVV + 2 * Nge * INV_TVM + INV_TMM) * np.pi / np.sqrt(D * D - er * er)
    int_real1 = quad(func_d_real, -np.inf, 0, args=(er), limit=LIMIT, epsabs=1e-4, epsrel=1e-4)
    int_real2 = quad(func_d_real, 0, np.inf, args=(er), limit=LIMIT, epsabs=1e-4, epsrel=1e-4)
    int_imag = np.pi / 2 * (rho(ep + e_) / (ep + e_) * (1 - Nge * np.tanh((ep + e_) / 2 / t_))
                            - rho(-ep + e_) / (-ep + e_) * (1 - Nge * np.tanh((-ep + e_) / 2 / t_)))
    err = int_real1[1] + int_real2[1]
    int_real = int_real1[0] + int_real2[0]
    return ret + INV_TNN * (int_real + 1j * int_imag), err


def neg_e_times_sigma_e(ep):
    er = ep + 1j * DELTA
    ret = (INV_TVV + 2 * Nge * INV_TVM + INV_TMM) * np.pi * ep / np.sqrt(D * D - er * er)
    int_real1 = quad(func_e_real, -np.inf, 0, args=(er), limit=LIMIT, epsabs=1e-4, epsrel=1e-4)
    int_real2 = quad(func_e_real, 0, np.inf, args=(er), limit=LIMIT, epsabs=1e-4, epsrel=1e-4)
    int_imag = np.pi / 2 * (rho(ep + e_) * (1 - Nge * np.tanh((ep + e_) / 2 / t_))
                            + rho(-ep + e_) * (1 - Nge * np.tanh((-ep + e_) / 2 / t_))) 
    err = int_real1[1] + int_real2[1]
    int_real = ep * (int_real1[0] + int_real2[0])
    return ret + INV_TNN * (int_real + 1j * int_imag), err


def dos(ep):
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
    for i in tqdm(range(N)):
        x[1][i], x[2][i] = dos(x[0][i])
    np.savetxt(f"./data/dos_data/e={e_}_t={t_}_teff={teff_}.txt", x)


if __name__ == "__main__":
    main()
