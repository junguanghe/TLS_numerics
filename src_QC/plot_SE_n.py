
import matplotlib.pyplot as plt
from scipy.integrate import quad
import h5py

import sys
sys.path.insert(1, "../src")

from calc_gap_profile import *

from parameters import *

with h5py.File("SE-Matsubara.h5", 'r') as f:
    for iter in range(numIter+1):
        eps = f[f"Iter{iter}/epsilon"][:]
        delta = f[f"Iter{iter}/Delta"][:]
        #plt.plot(ns,eps.real, '+', label=f"Re(Σ_ϵ), Iter{iter}")
        plt.plot(ns,eps.imag, '+', label=f"Im(Σ_ϵ), Iter{iter}")
        plt.plot(ns,delta.real, 'x', label=f"Σ_Δ, Iter{iter}")


        
SE_eps_old = []
SE_delta_old = []

for n in ns:
    SE_eps_old.append(-sigma_e(Delta, T, (2*n+1)*np.pi*T, E, T)*(2*n+1)*np.pi*T)
    SE_delta_old.append(sigma_d(Delta, T, (2*n+1)*np.pi*T, E, T)*Delta)


plt.plot(ns, SE_eps_old, label="Σ_ϵ, Old")
plt.plot(ns, SE_delta_old, label="Σ_Δ, Old")



### Analytic expression
def SE_eps_sum():
    eps_ns = 2*np.pi*T*(np.array(ns) + 1/2)
    res = (-(1j*eps_ns-E)/2)/np.sqrt(Delta**2 - (1j*eps_ns - E)**2)*1/(1-np.exp(-E/T)) \
        -(-(1j*eps_ns+E)/2)/np.sqrt(Delta**2 - (1j*eps_ns + E)**2)*1/(1-np.exp(+E/T))
    return Gamma*Nge*res

def SE_delta_sum():
    eps_ns = 2*np.pi*T*(np.array(ns) + 1/2)
    res = (Delta/2)/np.sqrt(Delta**2 - (1j*eps_ns - E)**2)*1/(1-np.exp(-E/T)) \
        -(Delta/2)/np.sqrt(Delta**2 - (1j*eps_ns + E)**2)*1/(1-np.exp(+E/T))
    return Gamma*Nge*res

def f_eps_branch_cut(e, n, getPart):
    eps_n = 2*np.pi*T*(n+1/2)
    epos = e + 1e-10j
    eneg = e - 1e-10j
    res = E/(E**2 - (1j*eps_n - e)**2) * (-epos/np.sqrt(Delta**2 - epos**2) + eneg/np.sqrt(Delta**2-eneg**2))/(np.exp(e/T)+1)
    res = res/(2*np.pi*1j)
    return getPart(res)

def SE_eps_branch_cut(n):
    realpart = Gamma*Nge*quad(f_eps_branch_cut, -np.inf, np.inf, args=(n, np.real))[0]
    imagpart = Gamma*Nge*quad(f_eps_branch_cut, -np.inf, np.inf, args=(n, np.imag))[0]
    return realpart + 1j*imagpart

def f_delta_branch_cut(e, n, getPart):
    eps_n = 2*np.pi*T*(n+1/2)
    epos = e + 1e-10j
    eneg = e - 1e-10j
    res = E/(E**2 - (1j*eps_n - e)**2) * (Delta/np.sqrt(Delta**2 - epos**2) - Delta/np.sqrt(Delta**2-eneg**2))/(np.exp(e/T)+1)
    res = res/(2*np.pi*1j)
    return getPart(res)

def SE_delta_branch_cut(n):
    realpart = Gamma*Nge*quad(f_delta_branch_cut, -np.inf, np.inf, args=(n, np.real))[0]
    imagpart = Gamma*Nge*quad(f_delta_branch_cut, -np.inf, np.inf, args=(n, np.imag))[0]
    return realpart + 1j*imagpart

SE_eps_bc = []
SE_delta_bc = []
for n in ns:
    SE_eps_bc.append(SE_eps_branch_cut(n))
    SE_delta_bc.append(SE_delta_branch_cut(n))

SE_eps_analytic = SE_eps_sum() - np.array(SE_eps_bc)
SE_delta_analytic = SE_delta_sum() - np.array(SE_delta_bc)
plt.plot(ns, SE_eps_analytic.imag, 'x', label="eps, contour")
plt.plot(ns, SE_delta_analytic.real, '+', label="eps, contour")

plt.legend()
plt.show()
