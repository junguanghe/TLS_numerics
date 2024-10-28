import numpy as np

ns = [*range(-500, 500)]

T = 0.9
E = 0.4
Delta = 1.764
#Gamma = 1/6 # 1/τ_in
Gamma = 2*np.pi*1/6
Nge = np.tanh(E / 2 / T)
#Nge = 0.9

numIter = 0 # numIter=0 means self-energy with bare propagators.
