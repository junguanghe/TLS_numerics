
from calc_2nd_self_energy import *
from parameters import *
import matplotlib.pyplot as plt

from scipy.integrate import quad
from scipy.integrate import trapezoid

import h5py


### Summation over the Matsubara frequencies directly

with h5py.File("SE-Matsubara.h5", "r") as f:
    SE_eps_now = f[f"Iter{numIter}/epsilon"][:]
    SE_delta_now = f[f"Iter{numIter}/Delta"][:]

SE = SelfEnergy(T, E, Delta, Gamma, Nge, np.array(ns), SE_eps_now, SE_delta_now)

es = np.arange(-3*Delta, 3*Delta, step=0.051234)
SE_eps = []
SE_delta = []

for e in es:
    se = SE.eval_real_freq(e, bare=True)
    SE_eps.append(se[0])
    SE_delta.append(se[1])


import sys
sys.path.insert(1, "../src")
from calc_DOS import *


### Old results from Jason

print(f"D={D}")


SE_eps_old = []
SE_delta_old = []
for e in es:
    nse = neg_e_times_sigma_e(e)
    SE_eps_old.append(-nse[0]) # nse[0] is the result, nse[1] is the error
    SE_delta_old.append(D*sigma_d(e)[0])



### Contour integral

i0 = 1e-6j

def Fermi(e):
    exp_fact = np.exp(e/T)
    if np.isinf(exp_fact):
        return 0.0
    return 1.0/(np.exp(e/T)+1.0)


def SE_eps_on_real_re(e, z, UHP):
    fermi = 1/(np.exp(e/T)+1)
    epm = e+i0 if UHP else e-i0
    res = E/(E**2 - (z-epm)**2)*(-epm)/np.sqrt(Delta**2 - epm**2)*fermi
    return -np.real(res/(2j*np.pi)) # The minus sign because this is from the contour integral, not residue.

def SE_eps_on_real_im(e, z):
    fermi = 1/(np.exp(e/T)+1)
    ep = e+i0 
    em = e-i0
    res = E/(E**2 - (z-e)**2)*(-e)*(1/np.sqrt(Delta**2 - ep**2) - 1/np.sqrt(Delta**2 - em**2))*fermi
    return -np.imag(res/(2j*np.pi))

def SE_eps_on_real(e, z):
    fermi = 1/(np.exp(e/T)+1)
    ep = e+i0 
    em = e-i0
    # res = E/(E**2 - (z-e)**2)*(-e)*(1/np.sqrt(Delta**2 - ep**2) - 1/np.sqrt(Delta**2 - em**2))*fermi
    # return -res/(2j*np.pi) # The minus sign because this is from the contour integral, not residue.
    res_u = E/(E**2 - (z-ep)**2) * (-ep)/np.sqrt(Delta**2 - ep**2) * Fermi(ep)
    res_d = E/(E**2 - (z-em)**2) * (-em)/np.sqrt(Delta**2 - em**2) * Fermi(em)
    return -(res_u-res_d)/(2j*np.pi)

def SE_delta_on_real_re(e, z, UHP):
    fermi = 1/(np.exp(e/T)+1)
    epm = e+i0 if UHP else e-i0
    res = E/(E**2 - (z-epm)**2)*(Delta)/np.sqrt(Delta**2 - epm**2)*fermi
    return -np.real(res/(2j*np.pi))

def SE_delta_on_real_im(e, z, UHP):
    fermi = 1/(np.exp(e/T)+1)
    epm = e+i0 if UHP else e-i0
    res = E/(E**2 - (z-epm)**2)*(Delta)/np.sqrt(Delta**2 - epm**2)*fermi
    return -np.imag(res/(2j*np.pi))

def SE_delta_on_real(e, z):
    ep = e+i0 
    em = e-i0
    # res = E/(E**2 - (z-e)**2)*(Delta)*(1/np.sqrt(Delta**2 - ep**2) - 1/np.sqrt(Delta**2 - em**2))*fermi
    # return -res/(2j*np.pi) # The minus sign because this is from the contour integral, not residue.
    res_u = E/(E**2 - (z-ep)**2) * Delta/np.sqrt(Delta**2 - ep**2) * Fermi(ep)
    res_d = E/(E**2 - (z-em)**2) * Delta/np.sqrt(Delta**2 - em**2) * Fermi(em)
    return -(res_u-res_d)/(2j*np.pi)


def u(t):
    return t/(1-t**2)
def J(t):
    return (1+t**2)/(1-t**2)**2

ts = np.linspace(-1+1e-10, 1-1e-10, num=1000000)
ys = np.exp(-u(ts)**2)*J(ts)
print(trapezoid(ys, ts))

SE_eps_contour = []
SE_delta_contour = []

for e in es:
    # res_UHP = quad(SE_f_on_real_im, -100*Delta, 100*Delta, args=(z, True),
    #                points=(Delta, -Delta, z+E, z-E),limit=5000)
    # res_LHP = quad(SE_f_on_real_im, -100*Delta, 100*Delta, args=(z, False),
    #                points=(Delta, -Delta, z+E, z-E),limit=5000)
    z = e + 3e-4j
    # res_UHP = quad(SE_eps_on_real_re, -np.inf, np.inf, args=(z, True),
    #                limit=1000)
    # res_LHP = quad(SE_eps_on_real_re, -np.inf, np.inf, args=(z, False),
    #                limit=1000)
    # res_UHP_im = quad(SE_eps_on_real_im, -np.inf, np.inf, args=(z, True),
    #                limit=1000)
    # res_LHP_im = quad(SE_eps_on_real_im, -np.inf, np.inf, args=(z, False),
    #                limit=1000)
    res1L = quad(SE_eps_on_real, -np.inf, e, args=(z), epsabs=1e-5, limit=50000, complex_func=True)
    res1R = quad(SE_eps_on_real, e, np.inf, args=(z), epsabs=1e-5, limit=50000, complex_func=True)
    # res1L_im = quad(SE_eps_on_real_im, -np.inf, e, args=(z), epsabs=1e-5, limit=50000, complex_func=True)
    # res1R_im = quad(SE_eps_on_real_im, e, np.inf, args=(z), epsabs=1e-5, limit=50000, complex_func=True)
    SE_eps_contour.append((res1L[0]+res1R[0])*Gamma*Nge)

    # Trapezoid rule
    # ys = SE_eps_on_real(u(ts), z)*J(ts)
    # res1 = trapezoid(ys, ts)
    # SE_eps_contour.append((res1)*Gamma*Nge)

    # res_UHP = quad(SE_delta_on_real_re, -np.inf, np.inf, args=(z, True),
    #                limit=1000)
    # res_LHP = quad(SE_delta_on_real_re, -np.inf, np.inf, args=(z, False),
    #                limit=1000)
    # res_UHP_im = quad(SE_delta_on_real_im, -np.inf, np.inf, args=(z, True),
    #                limit=1000)
    # res_LHP_im = quad(SE_delta_on_real_im, -np.inf, np.inf, args=(z, False),
    #                limit=1000)
    # res2 = quad(SE_delta_on_real, -np.inf, np.inf, args=(z), epsabs=1e-5, limit=50000, complex_func=True)
    res2L = quad(SE_delta_on_real, -np.inf, e, args=(z), epsabs=1e-5, limit=50000, complex_func=True)
    res2R = quad(SE_delta_on_real, e, np.inf, args=(z), epsabs=1e-5, limit=50000, complex_func=True)
    SE_delta_contour.append((res2L[0]+res2R[0])*Gamma*Nge)



def SE_eps_pole():
    eps = es + 3e-4j 
    res = (-(eps-E)/2)/np.sqrt(Delta**2 - (eps - E)**2)*1/(1-np.exp(-E/T)) \
        -(-(eps+E)/2)/np.sqrt(Delta**2 - (eps + E)**2)*1/(1-np.exp(+E/T))
    return Gamma*Nge*res

def SE_delta_pole():
    eps = es + 3e-4j 
    res = (Delta/2)/np.sqrt(Delta**2 - (eps - E)**2)*1/(1-np.exp(-E/T)) \
        -(Delta/2)/np.sqrt(Delta**2 - (eps + E)**2)*1/(1-np.exp(+E/T))
    return Gamma*Nge*res



### Bosonic sum

from calc_bosonic_sum import *


SE_eps_bosonic = []
SE_delta_bosonic = []

for e in es:
    se = SE_eps_bosonic_sum(e)
    SE_eps_bosonic.append(se)



if __name__=="__main__":
    # plt.plot(es, np.real(SE_eps), label="eps, real")
    # plt.plot(es, np.imag(SE_eps), '--', label="eps, imag")
    # plt.plot(es, np.real(SE_delta), label="Delta, real")
    # plt.plot(es, np.imag(SE_delta), '-.', label="Delta, imag")
    # plt.plot(es, np.array(SE_eps_bosonic).real, label="eps, real, Bosonic")
    # plt.plot(es, np.array(SE_eps_bosonic).imag, label="eps, imag, Bosonic")

    plt.plot(es, np.real(SE_eps_old), label="eps, real, Old", lw=1)
    plt.plot(es, np.imag(SE_eps_old), '--', label="eps, imag, Old", lw=1)
    plt.plot(es, np.array(SE_eps_contour).real+SE_eps_pole().real, label="Re, eps, Contour", lw=3, alpha=0.5)
    plt.plot(es, np.array(SE_eps_contour).imag+SE_eps_pole().imag, label="Im, eps, Contour", lw=3, alpha=0.5)

    plt.plot(es, np.real(SE_delta_old), label="Delta, real, Old", lw=1)
    plt.plot(es, np.imag(SE_delta_old), '--', label="Delta, imag, Old", lw=1)
    plt.plot(es, np.array(SE_delta_contour).real+SE_delta_pole().real, label="Re, Delta, Contour", lw=3, alpha=0.5)
    plt.plot(es, np.array(SE_delta_contour).imag+SE_delta_pole().imag, label="Im, Delta, Contour", lw=3, alpha=0.5)

    plt.grid()
    plt.legend()
    plt.show()
