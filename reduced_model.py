#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 20:14:17 2024

@author: Hugo Martin
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


R0 = 5.
eT = 0.9
k = 30.0
rho = 0.35

eta = 1.2

def f(e,eta):
    bias = (e**eta)/((e**eta + (1-e)**eta)**(1/eta))
    return bias
    


def Eq(Y,t,R0,eT,k,f,eta,rho):
    i,c = Y
    dYdt = [(R0*(1 - eT*c)*(1 - i) - 1)*i, k*c*(1-c)*(f(eT,eta)*i - rho)]
    return dYdt

i0 = 0.01
c0 = 0.05

Y0 = [i0,c0]

t_final = 30
Nb_pt_per_unit = 50

t = np.linspace(0,t_final,1 + Nb_pt_per_unit*t_final)

sol = odeint(Eq, Y0, t, args = (R0,eT,k,f,eta,rho))

fig = plt.figure()

plt.plot(t, sol[:,1], label = 'c(t)')
plt.plot(t, sol[:,0],'r', label = 'i(t)')
plt.xlabel('time (a.u.)')
plt.legend(loc='best')

DE = f(eT,eta)*sol[:,0] - rho

asign = np.sign(DE)
signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)
signchange = np.array(signchange, dtype=bool)

times = t[signchange]

for tc in times:
    plt.axvline(x=tc, color = 'k', linestyle = ':', linewidth = 0.8)

plt.show()
#plt.savefig('ic_dynamics.pdf') 

