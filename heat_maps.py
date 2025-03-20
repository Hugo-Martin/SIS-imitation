#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 10:19:31 2023

@author: Hugo Martin
"""

import numpy as np
from scipy.optimize import fsolve


import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid


import sys


plt.rcParams['text.usetex'] = True
plt.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'

####### Functions 

def phi(x,eta):
    return (np.exp(-(-np.log(x))**eta) + 1 - np.exp(-(-np.log(1 - x))**eta))/2

def A(ES, ET, p, phiS, phiT, eta):
    return (1-p)*phiS(ES,eta) +p*phiT(ET,eta)

def H(ES, ET, p, phiS, phiT, eta):
    return 1/(p/phiS(ES,eta) + (1-p)/phiT(ET,eta))

def RHO(rho_0,q_S,q_T,ES,ET):
    return rho_0 + q_S*pow(ES,2) + q_T*pow(ET,2)

def compute_data(R0, delta, q_S, q_T,a):
    
    x = np.arange(0.01, 1.0, delta)
    y = np.arange(0.01, 1.0, delta)
    
    ES, ET = np.meshgrid(x,y)
    I = (1-1/R0)*np.ones(ES.shape)
    C = np.zeros(ES.shape)
    
    ratio = RHO(rho_0,q_S,q_T,ES,ET)/A(ES,ET,a,phi,phi,eta)

    EPC = np.logical_and((1 - H(ES,ET,a,phi,phi,eta))*(1 - 1/(R0*(1-ES)*(1-ET))) < ratio,ratio < 1 - 1/R0)
    EFC = ratio <= (1 - H(ES,ET,a,phi,phi,eta))*(1 - 1/(R0*(1-ES)*(1-ET)))
    
    def syst(Z, *data):
        c, y = Z
        eS, eT = data
        return [R0*(1 - eS*c - (RHO(rho_0,q_S,q_T,eS,eT)/(A(eS, eT, a, phi, phi, eta)*(1 - H(eS, eT, a, phi, phi, eta)*c)))*(1 - eS*c*y))*(1 - eT*c*y) - 1, 1 - y - eS*(1 - (RHO(rho_0,q_S,q_T,eS,eT)/(A(eS, eT, a, phi, phi, eta)*(1 - H(eS, eT, a, phi, phi, eta)*c)))*y)*(1 - c*y)]
    
    starting = np.array([0.95, 0.95])


    for j in range(len(x)):
        for i in range(len(y)):
            if EPC[i,j]:
                #data = (x[j], y[i])
                root = fsolve(syst, starting, args=(x[j], y[i]))
                if all(np.isclose(syst(root, x[j], y[i]), [0.0, 0.0])): # check that found values are close enough to roots
                    starting = root
                    C[i,j] = root[0]
                    I[i,j] = RHO(rho_0,q_S,q_T,x[j], y[i])/(A(x[j], y[i], a, phi, phi, eta)*(1 - H(x[j], y[i], a, phi, phi, eta)*root[0]))
                else:
                    sys.exit("Numerical approximation of EPC equilibrium did not converge")


    C[EFC] = 1.0
    I[EFC] = 1.0 - 1/(R0*(1-ES[EFC])*(1-ET[EFC]))

    ind = np.unravel_index(np.argmin(I, axis=None), I.shape)
    best_es = ES[ind]
    best_et = ET[ind]
    
    return I, C, best_es, best_et

######### Parameters

R0 = 5      # Basic reproduction number


rho_0 = 0.125


# Perception bias
eta = 1.0


######### Computations

delta = 0.01


overall_max_I = 0.
overall_min_I = 1.


figI = plt.figure(figsize=(22, 21))

gridI = AxesGrid(figI, 111,
                nrows_ncols=(3, 3),
                axes_pad=0.26,
                cbar_mode='single',
                cbar_location='right',
                cbar_pad=0.3
                )

figC = plt.figure(figsize=(22, 21))

gridC = AxesGrid(figC, 111,
                nrows_ncols=(3, 3),
                axes_pad=0.3,
                cbar_mode='single',
                cbar_location='right',
                cbar_pad=0.35
                )

param = np.meshgrid(np.linspace(start=0.2, stop=0.8, num=3),np.linspace(start=0.9, stop=0.1, num=3))

mat_I = []
mat_C = []


BEST_ES = []
BEST_ET = []


for q_S, a in zip(param[0].flat, param[1].flat):
    
    I, C, best_es, best_et = compute_data(R0, delta, q_S, 1 - q_S,a)
    
    mat_I.append(I)
    mat_C.append(C)
    
    BEST_ES.append(best_es)
    BEST_ET.append(best_et)
    
    overall_max_I = max(overall_max_I, I.max())
    overall_min_I = min(overall_min_I, I.min())
    


for axI, axC, ind in zip(gridI, gridC, range(9)):

    
    imI = axI.imshow(mat_I[ind], vmin=overall_min_I, vmax=overall_max_I, cmap='seismic')
    imC = axC.imshow(mat_C[ind], vmin=0, vmax=1, cmap='Spectral')
    
    axI.invert_yaxis()
    axC.invert_yaxis()
    
    axI.scatter(100*BEST_ES[ind], 100*BEST_ET[ind], s=800, c='limegreen', marker='x', linewidths=14)
    axC.scatter(100*BEST_ES[ind], 100*BEST_ET[ind], s=800, c='limegreen', marker='x', linewidths=14)
    
    
    if ~(ind % 3):

        axI.set_ylabel(f" {0.9 - (ind/3)*0.4:.1f}" '\n' '\n' r'$e_T$', fontsize=34)
        axI.set_yticks(np.arange(0, 101, 25))
        axI.set_yticklabels(np.arange(0, 1.1, 0.25), fontsize=24)
        
        axC.set_ylabel(r'$a = $' f" {0.9 - (ind/3)*0.4:.1f}" '\n' '\n' r'$e_T$', fontsize=34)
        axC.set_yticks(np.arange(0, 101, 25))
        axC.set_yticklabels(np.arange(0, 1.1, 0.25), fontsize=24)
        
        
    if (ind > 5):
 
        axI.set_xlabel(r'$e_S$' '\n' '\n' f' {0.2 + (ind - 6)*0.3:.1f}', fontsize=34)
        axI.set_xticks(np.arange(0, 51, 50))
        axI.set_xticklabels(np.arange(0, .51, 0.5), fontsize=24)
            
        axC.set_xlabel(r'$e_S$' '\n' '\n' r'$q_S = $' f' {0.2 + (ind - 6)*0.3:.1f}', fontsize=34)
        axC.set_xticks(np.arange(0, 51, 50))
        axC.set_xticklabels(np.arange(0, .51, 0.5), fontsize=24)

    
for ax in gridC.cbar_axes:
    ax.tick_params(labeltop=False)
    

cbarI = gridI.cbar_axes[0].colorbar(imI)
cbarI.ax.tick_params(labelsize=30)

cbarC = gridC.cbar_axes[0].colorbar(imC)
cbarC.ax.tick_params(labelsize=30)




figI.suptitle(r"$\mathrm{Prevalence\ at\ equilibrium}$", fontsize=52, weight='bold')
figC.suptitle(r"$\mathrm{Fraction\ of\ cooperators\ at\ equilibrium}$", fontsize=52, weight='bold')

figI.text(0.01, 0.5, r'$\mathrm{Degree\ of\ altruism}$ ' '$(a)$', va='center', rotation='vertical', fontsize=36, weight='bold')
figC.text(0.01, 0.5, r'$\mathrm{Degree\ of\ altruism}$ ' '$(a)$', va='center', rotation='vertical', fontsize=36, weight='bold')

figI.text(0.5, 0.01, r'$\mathrm{Marginal\ cost\ increase\ of\ self-protection} (q_S)$', ha='center', fontsize=36, weight='bold')
figC.text(0.5, 0.01, r'$\mathrm{Marginal\ cost\ increase\ of\ self-protection} (q_S)$', ha='center', fontsize=36, weight='bold')

plt.show()


figI.savefig("Prevalence.pdf")  
figC.savefig("Cooperator.pdf")  
