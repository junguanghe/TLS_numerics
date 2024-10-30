
import numpy as np

from parameters import *

ms = np.arange(-1000, 1001)

def SE_eps_bosonic_sum(e):
    omega = 2*np.pi*T*ms
    summand = E/(E**2+omega**2)*(-e + 1j*omega)/np.sqrt(Delta**2 - (e-1j*omega)**2)
    return Gamma*Nge*T*np.sum(summand)

