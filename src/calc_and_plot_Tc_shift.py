from joblib import Parallel, delayed, cpu_count
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar


E = 0.2
TEFF = [None, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
INV_TNN1 = 0.01
INV_TNN2 = 4 * np.pi
N = 100
FIGSIZE = (6, 4.5)
EPSILON = 1e-7  # the cutoff for Matsubara sum
INV_TVV = 0
INV_TVM = 0
INV_TMM = 0
LIMIT = 5000  # the limit of subintervals in quad


def func_e(x, e, tc, en):
    return np.tanh((x + e) / 2 / tc) / (x * x + en * en)


def func_d(x, e, tc, en):
    return np.tanh((x + e) / 2 / tc) / (x * x + en * en) * x / (x + e)


def neg_sigma_e_over_ien(e, tc, teff, en, inv_tvv, inv_tvm, inv_tmm, inv_tnn):
    ret = (inv_tvv + 2 * np.tanh(e / 2 / teff) * inv_tvm + inv_tmm) * np.pi / en
    ret += inv_tnn * np.pi / en
    ret -= inv_tnn * np.tanh(e / 2 / teff) * quad(func_e, -np.inf, np.inf, args=(e, tc, en), limit=LIMIT)[0]
    return ret


def sigma_d_over_delta(e, tc, teff, en, inv_tvv, inv_tvm, inv_tmm, inv_tnn):
    ret = (inv_tvv + 2 * np.tanh(e / 2 / teff) * inv_tvm + inv_tmm) * np.pi / en
    ret += inv_tnn * np.pi * en / (e * e + en * en)
    ret -= inv_tnn * np.tanh(e / 2 / teff) * quad(func_d, -np.inf, -e, args=(e, tc, en), limit=LIMIT)[0]
    ret -= inv_tnn * np.tanh(e / 2 / teff) * quad(func_d, -e, np.inf, args=(e, tc, en), limit=LIMIT)[0]
    return ret


# the self-consistent equation
def SEC(tc, teff, e, inv_tvv, inv_tvm, inv_tmm, inv_tnn):
    if teff is None:
        teff = tc
    rhs = 0
    n = 0
    en = (2 * n + 1) * np.pi * tc
    fd = 1 + sigma_d_over_delta(e, tc, teff, en, inv_tvv, inv_tvm, inv_tmm, inv_tnn)
    fe = 1 + neg_sigma_e_over_ien(e, tc, teff, en, inv_tvv, inv_tvm, inv_tmm, inv_tnn)
    cur = (
        (fd / np.abs(fe) - 1) / (2 * n + 1) * 2
    )  # only sum over en > 0 and multiply by 2
    rhs += cur
    while rhs == 0 or abs(cur / rhs) > EPSILON:
        n += 1
        en = (2 * n + 1) * np.pi * tc
        fd = 1 + sigma_d_over_delta(e, tc, teff, en, inv_tvv, inv_tvm, inv_tmm, inv_tnn)
        fe = 1 + neg_sigma_e_over_ien(
            e, tc, teff, en, inv_tvv, inv_tvm, inv_tmm, inv_tnn
        )
        cur = (fd / np.abs(fe) - 1) / (2 * n + 1) * 2
        rhs += cur
    return rhs - np.log(tc)


def calc_Tc_shift(teff, e, inv_tvv, inv_tvm, inv_tmm, inv_tnn):
    res = root_scalar(
        SEC,
        args=(teff, e, inv_tvv, inv_tvm, inv_tmm, inv_tnn),
        x0=1.00003,
        x1=1.000037,
        maxiter=1000
    )
    if not res.converged:
        print(
            f"Warning: Tc shift calculation did not converge for teff={teff}, inv_tnn={inv_tnn}\n iteration={res.iterations}, flag={res.flag}"
        )
    return res.root


def main():
    inv_tnns = np.linspace(INV_TNN1, INV_TNN2, N)
    tcs = np.zeros((len(TEFF), N))
    for i, teff in enumerate(TEFF):
        print(f"Calculating Tc shift for teff={teff}")
        tcs[i] = Parallel(n_jobs=min(50, cpu_count()), verbose=20)(
            delayed(calc_Tc_shift)(teff, E, INV_TVV, INV_TVM, INV_TMM, inv_tnn)
            for inv_tnn in inv_tnns
        )

    fig_name = f"./figures/Tc_shift/e={E}_teff={TEFF}_inv_tnn={INV_TNN1:.2f}-{INV_TNN2:.2f}.pdf"
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for i, teff in enumerate(TEFF):
        ax.plot(
            inv_tnns / 2 / np.pi,
            tcs[i],
            label=r"$E/T_{c_0}=$" + str(E) + r", $T_{eff}/T_{c_0}=$" + str(teff),
        )
    ax.set_xlabel(r"$\hbar/2\pi\tau_{nn}T_{c_0}$")
    ax.set_ylabel(r"$T_c/T_{c_0}$")
    ax.legend()
    ax.grid()
    fig.savefig(fig_name)


if __name__ == "__main__":
    main()
