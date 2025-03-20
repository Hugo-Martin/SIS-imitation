#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 08:52:45 2023

@author: Hugo Martin
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


R0 = 5.
eS = 0.25
eT = 0.9
a = 0.7
k = 3.0
rho = 0.35



def fS(e):
    return e

fT = fS



A = (1-a)*fS(eS) + a*fT(eT)
H = 1/(a/fS(eS) + (1-a)/fT(eT))

print(rho/A)
print([(1-H)*(1 - 1/(R0*(1-eS)*(1-eT))),1 - 1/R0])


def Eq(Y,t,R0,eS,eT,k,fS,fT,a,rho):
    i,c,y = Y
    dYdt = [R0*(1 - eS*c - i*(1 - eS*c*y))*(1 - eT*c*y)*i - i, k*c*(1-c)*(A*(1 - H*c)*i - rho), R0*(1 - eT*c*y)*(1 - y -eS*(1 - i*y)*(1 - c*y)) + k*(1 - y)*np.maximum(0, A*(1 - H*c)*i - rho)]
    return dYdt

i0 = 0.05
c0 = 0.1
y0 = 0.9

Y0 = [i0,c0,y0]

t_final = 30
Nb_pt_per_unit = 20

t = np.linspace(0,t_final,1 + Nb_pt_per_unit*t_final)

sol = odeint(Eq, Y0, t, args = (R0,eS,eT,k,fS,fT,a,rho))

fig = plt.figure()

plt.plot(t, sol[:,1], label = 'c(t)')
plt.plot(t, sol[:,0],'r', label = 'i(t)')
plt.plot(t, sol[:,2], 'gold', label = 'y(t)')
plt.xlabel('time (a.u.)')
plt.legend(loc='best')


DE = A*(1 - H*sol[:,1])*sol[:,0] - rho

asign = np.sign(DE)
signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)
signchange = np.array(signchange, dtype=bool)

times = t[signchange]

for tc in times:
    plt.axvline(x=tc, color = 'k', linestyle = ':', linewidth = 0.8)

plt.show()
#plt.savefig('icy_dynamics_y_to_0.pdf') 


fig2 = plt.figure() 
plt.plot(t, DE, label = r'$\Delta E$')
plt.plot(t, (eS*(sol[:,0] + sol[:,1]) - 1 + np.sqrt((eS*(sol[:,0] + sol[:,1]) - 1)**2 + 4*eS*sol[:,0]*sol[:,1]*(1 - eS)))/(2*eS*sol[:,0]*sol[:,1]), label = r'$\bar{y}$')


plt.legend(loc='best')


plt.show()

fig3 = plt.figure()

iC = sol[:,0]*sol[:,1]*sol[:,2]

plt.plot(t, sol[:,1] - iC, color= 'forestgreen', linestyle='dashed', label = r'$s_C(t)$')
plt.plot(t, iC, linestyle='dashed', color='r', label = r'$i_C(t)$')
plt.plot(t, (1-sol[:,0]) - (sol[:,1] - iC), color= 'forestgreen', label = r'$s_{NC}(t)$')
plt.plot(t, sol[:,0] - iC, color='r', label = r'$i_{NC}(t)$')
plt.xlabel('time (a.u.)')
plt.legend(loc='best')


for tc in times:
    plt.axvline(x=tc, color = 'k', linestyle = ':', linewidth = 0.8)


