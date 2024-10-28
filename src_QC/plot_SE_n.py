
import matplotlib.pyplot as plt
import h5py

from parameters import *

with h5py.File("SE-Matsubara.h5", 'r') as f:
    for iter in range(numIter+1):
        eps = f[f"Iter{iter}/epsilon"][:]
        delta = f[f"Iter{iter}/Delta"][:]
        plt.plot(ns,eps.imag)
        plt.plot(ns,delta.real)

plt.show()
