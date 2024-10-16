# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 16:55:21 2022

@author: jungu
"""

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "cm"
})

N = 10000
size = 14 # font size
a = 2
X = 5
x = np.linspace(-X, X, N)
U = abs(x*x/a/a - 1)**1.5 * 3
Ee = 2
Eg = 0.25


fig, ax = plt.subplots(figsize = (3.5,3))
ax.plot(x, U, label=r'$U(X)$', linewidth=2)
# ax.annotate('', xy=(-a,-A/5), xytext=(a,-A/5),
#             arrowprops=dict(arrowstyle='<->'))
# ax.annotate('a', xy=(0,-A*.5), fontsize=size)


# sigma = a/2.5
# x1 = np.linspace(-1.25*X, 1.25*X, N)
# phiL = A/2*np.exp(-(x1+a)**2/sigma**2)
# phiR = A/2*np.exp(-(x1-a)**2/sigma**2)
# ax.plot(x1,phiL, label=r'$\phi_L(X)$')
# ax.plot(x1,phiR, label=r'$\phi_R(X)$')

sigma = a*0.8
x1 = np.linspace(-1.25*X, 1.25*X, N)
phiL = np.exp(-(x1+a)**2/sigma**2)
phiR = np.exp(-(x1-a)**2/sigma**2)
phig = (phiL + phiR)/np.sqrt(2) + Eg
phie = (phiL - phiR)/np.sqrt(2) + Ee
ax.plot(x1,phig, label=r'$\phi_g(X)$', lw=0.7)
ax.plot(x1,phie, label=r'$\phi_e(X)$', lw=0.7)
ax.annotate('', xy=(0,Eg), xytext=(0,Ee),
            arrowprops=dict(arrowstyle='<->'))
ax.annotate(r'$J$', xy=(0.1,(Ee+Eg)/2), fontsize=size)

ax.annotate('', xy=(-a,Ee/5), xytext=(a,Ee/5),
            arrowprops=dict(arrowstyle='<->'))
ax.annotate(r'$a$', xy=(0,0.15), fontsize=size)

ax.set_xlabel(r'$X$')
ax.grid(True)
ax.legend()
ax.set_xlim([-6,6])
ax.set_ylim([0,5])
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.tick_params(direction="in")
fig.set_tight_layout(True)

fig.savefig(f"./figures/double_well.pdf", bbox_inches='tight')
