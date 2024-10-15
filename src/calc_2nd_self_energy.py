# All energies are measured in T_c

import numpy as np

multp = 30
n_min = 100

def Sigma_eps(n, T, E, Delta, Gamma, Nge=0.5):
    n_limit = np.abs(n)*30 + n_min
    l = np.arange(-n_limit, n_limit)
    eps_l = 2*np.pi*T*(l+1/2)

    kernel = E/(E**2 + (2*np.pi*T)**2*(n-l)**2)
    denom = np.sqrt(Delta**2 + eps_l**2)
    prop_eps = -1j*eps_l/denom
    prop_delta = Delta/denom

    se_eps = np.sum(kernel*prop_eps)
    se_delta = np.sum(kernel*prop_delta)

    for i in range(0, 20):
        l_p = np.arange(n_limit+np.abs(n)*i, n_limit+np.abs(n)*(i+1))
        l_m = np.arange(-n_limit-np.abs(n)*(i+1), -n_limit-np.abs(n)*i)
        l_add = np.concat((l_m, l_p))


        eps_l_add = 2*np.pi*T*(l_add+1/2)

        kernel_add = E/(E**2 + (2*np.pi*T)**2*(n-l_add)**2)
        denom_add = np.sqrt(Delta**2 + eps_l_add**2)
        prop_eps_add = -1j*eps_l_add/denom_add
        prop_delta_add = Delta/denom_add

        se_eps_add = np.sum(kernel_add*prop_eps_add)
        se_delta_add = np.sum(kernel_add*prop_delta_add)


        # eps_l_p = 2*np.pi*T*(l_p+1/2)
        # eps_l_m = 2*np.pi*T*(l_m+1/2)

        # kernel_p = E/(E**2 + (2*np.pi*T)**2*(n-l_p)**2)
        # kernel_m = E/(E**2 + (2*np.pi*T)**2*(n-l_m)**2)
        # prop_p = -1j*eps_l_p/np.sqrt(Delta**2 + eps_l_p**2)
        # prop_m = -1j*eps_l_m/np.sqrt(Delta**2 + eps_l_m**2)

        # s_next = np.sum(kernel_p*prop_p) + np.sum(kernel_m*prop_m)

        se_eps += se_eps_add
        se_delta += se_delta_add

        if np.abs((se_eps_add+se_delta_add)/(se_eps+se_delta)) < 1e-4:
            print(f"Current iter: {i}")
            break

    return Gamma * Nge * T * se_eps, Gamma * Nge * T * se_delta
    

# The asymptotic value of the bare self-energy can be obtained from the Eq. (28)
# It is easy to see that the Delta part tends to zero as n tends to infinity.
# For the epsilon part, it can be computed with the formula sum(1/(m^2 + a^2), m, -inf, inf) = π·coth(πa)/a

def SE_asymptotic(T, E, Delta, Gamma, Nge=0.5):
    a = E/(2*np.pi*T)
    return Gamma*Nge*(T/E)*a**2*(np.pi/(a*np.tanh(a*np.pi)))
