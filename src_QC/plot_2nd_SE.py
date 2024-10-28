
from calc_2nd_self_energy import *
import matplotlib.pyplot as plt

ns = [*range(-300, 300)]

T = 0.3
E = 0.6
Delta = 1.764
Gamma = 100.0
Nge = 0.5

SE_eps = []
SE_delta = []

for n in ns:
    se = SE_fn(None, n, T, E, Delta, Gamma, Nge)
    SE_eps.append(se[0])
    SE_delta.append(se[1])
    
SE1 = SelfEnergy(T, E, Delta, Gamma, Nge, np.array(ns), SE_eps, SE_delta)

plt.plot(ns, np.imag(SE_eps), label=r"$\Sigma_\epsilon$ (imag)")
plt.plot(ns, np.real(SE_delta), label=r"$\Sigma_\Delta$ (real)")

n_larger = np.arange(-1000, 1000)
plt.plot(n_larger, np.imag(SE1.get_SE_eps(n_larger)), lw=0.8)
plt.plot(n_larger, np.real(SE1.get_SE_delta(n_larger)), lw=0.8)

SE1_eps = []
SE1_delta = []
for n in ns:
    se = SE1.eval_with(n, T, E, Delta, Gamma, Nge)
    SE1_eps.append(se[0])
    SE1_delta.append(se[1])

plt.plot(ns, np.imag(SE1_eps), label=r"$\Im\Sigma_\epsilon$, iter1")
plt.plot(ns, np.real(SE1_delta), label=r"$\Re\Sigma_\Delta$ (real), iter1")

plt.legend()
plt.grid()
plt.show()
