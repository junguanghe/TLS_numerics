
from calc_2nd_self_energy import *
import matplotlib.pyplot as plt

ns = [*range(-500, 500)]

T = 0.3
E = 0.6
Delta = 1.764
Gamma = 1.0

SE_eps = []
SE_delta = []

for n in ns:
    se = Sigma_eps(n, T, E, Delta, Gamma)
    SE_eps.append(se[0])
    SE_delta.append(se[1])
    

plt.plot(ns, np.imag(SE_eps), label="epsilon")
plt.plot(ns, np.real(SE_delta), label="Delta")
plt.legend()
plt.grid()
plt.show()
