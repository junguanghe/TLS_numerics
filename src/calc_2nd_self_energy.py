# All energies are measured in T_c

import numpy as np

multp = 30
n_min = 100

def SE_fn(self, n, T, E, Delta, Gamma, Nge=0.5):
    n_limit = np.abs(n)*30 + n_min
    l = np.arange(-n_limit, n_limit)
    eps_l = 2*np.pi*T*(l+1/2)

    kernel = E/(E**2 + (2*np.pi*T)**2*(n-l)**2)
    SE_eps_l = self.get_SE_eps(l) if self!=None else 0
    SE_delta_l = self.get_SE_delta(l) if self!=None else 0

    denom = np.sqrt((Delta + SE_delta_l)**2 + (eps_l + 1j*SE_eps_l)**2)
    prop_eps = (-1j*eps_l + SE_eps_l)/denom
    prop_delta = (Delta + SE_delta_l)/denom

    se_eps = np.sum(kernel*prop_eps)
    se_delta = np.sum(kernel*prop_delta)

    for i in range(0, 20):
        l_p = np.arange(n_limit+np.abs(n)*i, n_limit+np.abs(n)*(i+1))
        l_m = np.arange(-n_limit-np.abs(n)*(i+1), -n_limit-np.abs(n)*i)
        l_add = np.concat((l_m, l_p))


        eps_l_add = 2*np.pi*T*(l_add+1/2)

        kernel_add = E/(E**2 + (2*np.pi*T)**2*(n-l_add)**2)
        SE_eps_l_add = self.get_SE_eps(l_add) if self!=None else 0
        SE_delta_l_add = self.get_SE_delta(l_add) if self!=None else 0

        denom_add = np.sqrt((Delta+SE_delta_l_add)**2 + (eps_l_add+1j*SE_eps_l_add)**2)
        prop_eps_add = (-1j*eps_l_add+SE_eps_l_add)/denom_add
        prop_delta_add = (Delta+SE_delta_l_add)/denom_add

        se_eps_add = np.sum(kernel_add*prop_eps_add)
        se_delta_add = np.sum(kernel_add*prop_delta_add)


        se_eps += se_eps_add
        se_delta += se_delta_add

        if np.abs((se_eps_add+se_delta_add)/(se_eps+se_delta)) < 1e-4:
            # print(f"Current iter: {i}")
            break

    return Gamma * Nge * T * se_eps, Gamma * Nge * T * se_delta
    

# The asymptotic value of the bare self-energy can be obtained from the Eq. (28)
# It is easy to see that the Delta part tends to zero as n tends to infinity.
# For the epsilon part, it can be computed with the formula sum(1/(m^2 + a^2), m, -inf, inf) = π·coth(πa)/a

def SE_asymptotic(T, E, Delta, Gamma, Nge=0.5):
    a = E/(2*np.pi*T)
    return Gamma*Nge*(T/E)*a**2*(np.pi/(a*np.tanh(a*np.pi)))

# Create a class to save the function Σ(n)
# so that it can be evaluated for the RHS of the fixed point problem.
class SelfEnergy:
    def __init__(self, T, E, Delta, Gamma, Nge, n_grid, SE_eps, SE_delta):
        self.T = T
        self.E = E
        self.Delta = Delta
        self.Gamma = Gamma
        self.Nge = Nge

        self.n_grid = n_grid
        self.SE_eps = SE_eps
        self.SE_delta = SE_delta
        self.n_min = n_grid[0]
        self.n_max = n_grid[-1]


    def __call__(n):
        pass

    def get_SE_eps(self, ls):
        """
        ls is the array of Matsubara index. It must be sorted.  
        """
        res = np.fromiter((self.SE_eps[l-self.n_min] if self.n_min <= l <= self.n_max else
                           1j*SE_asymptotic(self.T, self.E, self.Delta, self.Gamma, self.Nge) if l < self.n_min else
                           -1j*SE_asymptotic(self.T, self.E, self.Delta, self.Gamma, self.Nge) for l in ls),
                          dtype=complex)
        return res

    def get_SE_delta(self, ls):
        """
        ls is the array of Matsubara index. It must be sorted.  
        """
        res = np.fromiter((self.SE_delta[l-self.n_min] if self.n_min <= l <= self.n_max else
                           0 if l < self.n_min else
                           0 for l in ls),
                          dtype=complex)
        return res

    # Bind the method
    eval_RHS = SE_fn
