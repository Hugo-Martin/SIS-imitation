#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 09:00:12 2025

@author: hmartin
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.integrate import solve_ivp
from itertools import product


R0 = 5
alpha = 0.7
eS = 0.0
eT = 0.6
kappa = 10
rho = 0.3
pC = 0.7
pS = 0.1
F = 10

obs_i_S = 1

def fS(e):
    return e

fT = fS

def AA(eS, eT, alpha):
    return (1-alpha)*fS(eS) + alpha*fT(eT)

def HH(eS, eT, alpha):
    if (eS == 0 or eT == 0):
        H = 0
    else:
        H = 1/(alpha/fS(eS) + (1-alpha)/fT(eT))
    return H



#%% Model without symptomatic infectious

def Eq(t, Y, R0, eS, eT, alpha, kappa, fS, fT, rho):
    i,c,y = Y
    A = AA(eS, eT, alpha)
    H = HH(eS, eT, alpha)
    dYdt = [R0*(1 - eS*c - i*(1 - eS*c*y))*(1 - eT*c*y)*i - i,
            kappa*c*(1-c)*(A*(1 - H*c)*i - rho),
            R0*(1 - eT*c*y)*(1 - y -eS*(1 - i*y)*(1 - c*y)) + kappa*(1 - y)*np.maximum(0, A*(1 - H*c)*i - rho)]
    return dYdt

i0 = 0.05
c0 = 0.1
y0 = 0.9

Y0 = [i0,c0,y0]

t_final = 30

sol = solve_ivp(fun = Eq, y0 = Y0, t_span=[0, t_final], args = (R0, eS, eT, alpha, kappa, fS, fT, rho), dense_output=True)

fig = plt.figure()

iC = sol.y[0,:]*sol.y[1,:]*sol.y[2,:]

plt.plot(sol.t, sol.y[1,:] - iC, color= 'forestgreen', linestyle='dashed', label = r'$s_C(t)$')
plt.plot(sol.t, iC, linestyle='dashed', color='r', label = r'$i_C(t)$')
plt.plot(sol.t, (1-sol.y[0,:]) - (sol.y[1,:] - iC), color= 'forestgreen', label = r'$s_D(t)$')
plt.plot(sol.t, sol.y[0,:] - iC, color='r', label = r'$i_D(t)$')
plt.xlabel('time (a.u.)')
plt.legend(loc='best')


#%% Model with symptomatic infectious

def DeltaE(s_D, i_D, s_C, i_C, A, H, rho):
    return A*(1 - H*(s_C + i_C))*(i_D + i_C) - rho


def Eq2(t, Y, R0, eS, eT, kappa, alpha, rho, pC, F, pS):
    s_D, i_D, s_C, i_C, i_S = Y
    
    A = AA(eS, eT, alpha)
    H = HH(eS, eT, alpha)
    
    delta_e = DeltaE(s_D, i_D, s_C, i_C, A, H, rho)
    
    switching_term_S = kappa * (
        np.maximum(0, delta_e + i_S * (rho + (1 - fT(eT)) * fS(eS) * (1 - alpha) * F - A * (i_D + i_C))) * (s_C + i_C + obs_i_S*i_S) * s_D
        + (np.minimum(0, delta_e + i_S * (rho + (1 - fT(eT)) * fS(eS) * (1 - alpha) * F - A * (i_D + i_C)))) * (s_D + i_D) * s_C
    )
    
    switching_term_I = kappa * (
        np.maximum(0, delta_e + i_S * (rho + (1 - fT(eT)) * fS(eS) * (1 - alpha) * F - A * (i_D + i_C))) * (s_C + i_C + obs_i_S*i_S) * i_D
        + (np.minimum(0, delta_e + i_S * (rho + (1 - fT(eT)) * fS(eS) * (1 - alpha) * F - A * (i_D + i_C)))) * (s_D + i_D) * i_C
    )
    
    dYdt = [
        -R0 * (i_D + (1-eT)*(i_C + i_S))*s_D + (i_D + (1-pC)*i_S) - switching_term_S,
        (1 - pS) * R0 * (i_D + (1-eT)*(i_C + i_S))*s_D - i_D - switching_term_I,
        -(1 - eS) * R0 * (i_D + (1-eT)*(i_C + i_S))*s_C + (i_C + pC*i_S) + switching_term_S,
        (1. - eS) * (1 - pS) * R0 * (i_D + (1-eT)*(i_C + i_S))*s_C - i_C + switching_term_I,
        pS * R0 * (i_D + (1 - eT)*(i_C + i_S))*(s_D + (1-eS)*s_C) - i_S
    ]
    return dYdt

i_D0 = i0 * (1 - c0 * y0)
s_C0 = c0 * (1 - i0 * y0)
i_C0 = i0 * c0 * y0
s_D0 = 1 - i_D0 - s_C0 - i_C0
i_S0 = 0.

Z0 = [s_D0, i_D0, s_C0, i_C0, i_S0]

sol2 = solve_ivp(fun = Eq2, y0 = Z0, t_span=[0, t_final], args = (R0, eS, eT, kappa, alpha, rho, pC, F, pS), dense_output=True)

fig = plt.figure()

plt.plot(sol2.t, sol2.y[0,:], color= 'forestgreen', label = r'$s_D(t)$')  # s_D
plt.plot(sol2.t, sol2.y[1,:], color='r', label = r'$i_D(t)$')  # i_D
plt.plot(sol2.t, sol2.y[2,:], color= 'forestgreen', linestyle='dashed', label = r'$s_C(t)$')  # s_C
plt.plot(sol2.t, sol2.y[3,:], linestyle='dashed', color='r', label = r'$i_C(t)$')  # i_C
plt.plot(sol2.t, sol2.y[4,:], label = r'$i^S(t)$')  # i_S
#plt.plot(sol.t, np.sum(sol.y,0), label = r'$N(t)$')  # N

plt.xlabel('time (a.u.)')
plt.legend(loc='best')


plt.show()
#plt.savefig('icy_dynamics_y_to_0.pdf') 

#%% Error

times = np.linspace(0, t_final, 500)


s = sol.sol(times)

iC = s[0,:]*s[1,:]*s[2,:]


print(np.max(abs(sol2.sol(times)[0:4,:]-np.array([(1 - s[0,:]) - (s[1,:] - iC),
                                               s[0,:] - iC,
                                               s[1,:] - iC,
                                               iC
                                               ]
                                              ))))
#%% Effect of the initial condition

i_D0 = i0 * (1 - c0 * y0)
s_C0 = c0 * (1 - i0 * y0)
i_C0 = i0 * c0 * y0
s_D0 = 1 - i_D0 - s_C0 - i_C0
i_S0 = 0.

Z0 = [s_D0, i_D0, s_C0, i_C0, i_S0]

val = [(0.01, 0.25, 0.5, 0.75, 0.99), (0.01, 0.25, 0.5, 0.75, 0.99), (0.01, 0.25, 0.5, 0.75, 1, 1.25, 1.5)]
Init_cond = list(product(*val))

SOL = []

for init in Init_cond:
    
    Z0 = [
        init[0] * (1 - init[1] * init[2]),
        init[1] * (1 - init[0] * init[2]),
        init[0] * init[1] * init[2],
        1 - init[0] - init[1] + init[0] * init[1] * init[2],
        0.
        ]
    
    if all(z <= 1. for z in Z0) and all(z >= 0. for z in Z0):
        SOL.append(solve_ivp(fun = Eq2, y0 = Z0, t_span=[0, t_final], args = (R0, eS, eT, kappa, alpha, rho, pC, F, pS), dense_output=True))
        assert abs(sum(SOL[-1].y[:,-1]) - 1) < 1e-7, "Sum on all compartments too different from 1."
    else:
        Init_cond.remove(init)
    
cmap = mpl.colormaps['plasma']

fig, axs = plt.subplots(2, 2, figsize=(12, 8))
#fig.suptitle('Effect of the initial condition', fontsize = 18)

for j in range(np.shape(SOL[0].y)[0] - 1):
    if j%2:
        h = 'i_'
    else:
        h = 's_'
    if j > 1:
        b = 'C'
    else:
        b = 'D'
    lines = [np.column_stack([times, SOL[k].sol(times)[j,:]]) for k in range(len(SOL))]
    line_collection = LineCollection(lines, colors=cmap(np.linspace(0, 1, len(lines))), linewidths=2)
    axs[j//2,j%2].add_collection(line_collection)
    axs[j//2,j%2].set_title(f"${h+b}$", fontsize = 16)

for ax in axs.flat:
    ax.set_xlabel('time (a.u.)', fontsize=14)
    ax.set_xlim(0, times[-1])
    ax.set_ylim(0, 1)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.label_outer()

    
#plt.show()
plt.savefig('no_effect_initial_cond.pdf', format="pdf", bbox_inches = 'tight') 

#%% Alternative version of Figure 5, top right

neT1, neT2, neT3 = (10, 35, 15)

npS1, npS2 = (39, 10)

ET = np.concatenate((np.linspace(0, 0.2, neT1, endpoint = False),
                    np.linspace(0.2, 0.6, neT2, endpoint = False),
                    np.linspace(0.6, 1, neT3)
                    ))


PS = np.concatenate((np.linspace(0, 0.4, npS1, endpoint = False),
                    np.linspace(0.4, 0.8, npS2)
                    ))



tol_std = 1e-4
tol_x = 1e-5

rho0 = 0.05

PC = [0.15, 0.5, 0.85]
II = []
CC = []
IIAA = []
CCAA = []

for pC in PC:
    print(pC)
    I = np.zeros((len(PS), len(ET)))
    C = np.zeros((len(PS), len(ET)))
    IA = np.zeros((len(PS), len(ET)))
    CA = np.zeros((len(PS), len(ET)))
    for i in range(len(PS)):
        for j in range(len(ET)):
            #print([i,j])
            if i == 0 and j == 0:
                x0 = np.array([0.15, 0.75, 0.05, 0.05, 0])
                t_final = 10
            else:
                if j == 0:
                    x0 = y0
                    t_final = 5
            
            eq = 0
            t_start = 0
            
            while not eq:
                #print(t_final)
                sol3 = solve_ivp(fun = Eq2, y0 = x0, t_span=[t_start, t_final], args = (R0, 0, ET[j], kappa, alpha, rho0 + 0.85*ET[j]**2, pC, F, PS[i]), dense_output=True)
                x0 = sol3.y[:,-1]
                
                if any(x0 < 0):
                    if all (x0 > -tol_x):
                        x0[x0<0] = 0
                        x0 = x0 / sum(x0)
                    else:
                        break
                
                if any(x0 > 1):
                    if all(x0 - 1 < tol_x):
                        x0[x0 > 1] = 1
                        x0 = x0 / sum(x0)
                
                x0 = x0 / sum(x0)
                m = np.mean(sol3.y[:,-20:-1], axis = 1)
                std = np.std(sol3.y[:,-20:-1] - m[np.newaxis].T, axis = 1)
                if all(std < tol_std):
                    eq = 1
                else:
                    t_start = t_final
                    t_final = 2 * t_start
            if any(x0 < 0):
                if all (x0 > -tol_x):
                    x0[x0 < 0] = 0
                    x0 = x0 / sum(x0)
                else:
                    break
            
            if any(x0 > 1):
                if all(x0 - 1 < tol_x):
                    x0[x0 > 1] = 1
                    x0 = x0 / sum(x0)
                else:
                    break
            
            I[i,j] = 1 - sum(x0[0:3:2])
            C[i,j] = sum(x0[2:])
            IA[i,j] = sum(x0[1:5:2])
            CA[i,j] = sum(x0[3:5])
            
            if j == 0:
                y0 = x0
    II.append(I)
    CC.append(C)
    IIAA.append(IA)
    CCAA.append(CA)

#%% Plots of lines depending on pA

r = range(30,-1,-3)


fig, axs = plt.subplots(3, 3, figsize=(12, 10))

for j in range(3):
    
    ## I
    axs[0,j].set_ylim(0.6, 0.9)
    lines_I = [np.column_stack([ET, II[j][k,:]]) for k in r]
    line_collection = LineCollection(lines_I, cmap='rainbow', linewidths=1.8, array=PS[r])
    axs[0,j].add_collection(line_collection)
    if j == 0:
        axs[0,j].set_ylabel('Prevalence', fontsize=14)
    axs[0,j].set_title(rf'$p_C$ = {PC[j]}', fontsize = 16)
    
    ## C
    lines_C = [np.column_stack([ET, CC[j][k,:]]) for k in r]
    line_collection = LineCollection(lines_C, cmap='rainbow', linewidths=1.8, array=PS[r])
    axs[1,j].add_collection(line_collection)
    if j == 0:
        axs[1,j].set_ylabel('Fraction of masked', fontsize = 14)
    axs[1,j].set_xlabel(r'$e_T$', fontsize = 14)
    
    ## Optimal
    argmin = [np.argmin(II[j][k,:]) for k in range(II[j].shape[0])]
    #axs[2,j].set_xlim(min(PS), max(PS))
    axs[2,j].plot(PS, ET[argmin], linewidth = 4)
    axs[2,j].set_xlabel(r'$p^*$', fontsize=14)
    axs[2,j].axvspan(0, max(PS[ET[argmin] < 1]), color='green', alpha=0.3)
    if j == 0:
        axs[2,j].set_ylabel(r"Optimal $e_T$", fontsize=14)



cbar_ax = fig.add_axes([0.96, 0.42, 0.03, 0.35])

clb = fig.colorbar(
    line_collection,
    cax=cbar_ax
)
clb.ax.set_title('Fraction of\n symptomatic' '\n' r'infections $p^*$',
                 fontsize=14,
                 pad = 12)

plt.subplots_adjust(hspace=0.35)

shift = 0.025

for ax in axs[1, :]:
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0 + shift, pos.width, pos.height])

plt.savefig("Alt_fig5.pdf", format="pdf", bbox_inches = 'tight')
#plt.show()

fig, axs = plt.subplots(2, 3, figsize=(12, 7))

for j in range(3):
    
    ## IA
    axs[0,j].set_ylim(0.4, 0.9)
    lines_IA = [np.column_stack([ET, IIAA[j][k,:]]) for k in r]
    line_collection = LineCollection(lines_IA, cmap='rainbow', linewidths=1.8, array=PS[r])
    axs[0,j].add_collection(line_collection)
    if j == 0:
        axs[0,j].set_ylabel('Prevalence', fontsize=14)
    axs[0,j].set_title(rf'$p_C$ = {PC[j]}', fontsize = 16)
    
    ## CA
    axs[1,j].set_ylim(0., 0.8)
    lines_CA = [np.column_stack([ET, CCAA[j][k,:]]) for k in r]
    line_collection = LineCollection(lines_CA, cmap='rainbow', linewidths=1.8, array=PS[r])
    axs[1,j].add_collection(line_collection)
    if j == 0:
        axs[1,j].set_ylabel('Fraction of masked', fontsize = 14)
    axs[1,j].set_xlabel(r'$e_T$', fontsize = 14)


cbar_ax = fig.add_axes([0.96, 0.42, 0.03, 0.35])

clb = fig.colorbar(
    line_collection,
    cax=cbar_ax
)
clb.ax.set_title('Fraction of\n symptomatic' '\n' r'infections $p^*$',
                 fontsize=14,
                 pad = 12)


plt.savefig("Asymptomatic_only.pdf", format="pdf", bbox_inches = 'tight')

