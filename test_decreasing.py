#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 10:29:44 2025

@author: Hugo Martin
"""


import numpy as np

import multiprocessing as mp
import itertools



def I(c, rho, A, H):
    return (rho/A)/(1 - H*c)
    
def phi(x, eta):
    return (np.exp(-(-np.log(x))**eta) + 1 - np.exp(-(-np.log(1 - x))**eta))/2

def A(eS, eT, a, phiS, phiT, etaS, etaT):
    return (1-a)*phiS(eS,etaS) +a*phiT(eT,etaT)

def H(eS, eT, a, phiS, phiT, etaS, etaT):
    return 1/(a/phiS(eS,etaS) + (1-a)/phiT(eT,etaT))

def d(c, y, eS, eT, A, H, rho):
    return (1 - eS*c - I(c, rho, A, H)*(1 - eS*c*y))*(1 - eT*c*y)

def YC(c, rho, A, H, eS):
    
    return (np.sqrt(((1 - eS*c)*(1 - H*c) - (rho/A)*eS)**2 + 4*eS*(1 - eS)*(rho/A)*c*(1 - H*c)) - (1 - eS*c)*(1 - H*c) + (rho/A)*eS)/(2*(rho/A)*eS*c)



def isdecreasing(ind):
 
    
    rho = np.arange(max(precal_A[ind]*(1 - precal_H[ind])*(1 - 1/(RR00[ind]*(1 -ES[ind])*(1 - ET[ind]))), step), precal_A[ind]*(1 - 1/RR00[ind]), step)
    
    if rho.size:
        C, RHO = np.meshgrid(c, rho)
        yc = YC(C, RHO, precal_A[ind], precal_H[ind], ES[ind])
        smaller = yc < 1
        
        D = d(C,
              yc,
              ES[ind], ET[ind],
              precal_A[ind],
              precal_H[ind],
              RHO
              )
        
        if np.any(np.diff(D)[smaller[:, :-1]] >= 0):
            return ind
        else:
            return None
    else:
        yc = YC(c, (1 - 1 /RR00[ind])/2, precal_A[ind], precal_H[ind], ES[ind])
        smaller = yc < 1
        
        D = d(c,
              yc,
              ES[ind], ET[ind],
              precal_A[ind],
              precal_H[ind],
              (1 - 1 /RR00[ind])/2
              )
        
        if np.any(np.diff(D)[smaller[:-1]] >= 0):
            return ind
        else:
            return None

step = 0.01
num = 99

ES, ET, AA, RR00, ETAS, ETAT = np.meshgrid(np.arange(step, 1, step), #eS                                      
                                           np.arange(step, 1, step),  # eT
                                           np.arange(step, 1, step),  # a
                                           np.r_[np.arange(1.1, 2, 0.1), np.arange(2, 10, 1), np.arange(10, 21, 2)],  # R0
                                           np.logspace(-1.0, 1.0, num=num, endpoint=True),  # etaS
                                           np.logspace(-1.0, 1.0, num=num, endpoint=True) # etaT
                                           )



precal_A = A(ES, ET, AA, phi, phi, ETAS, ETAT)
precal_H = H(ES, ET, AA, phi, phi, ETAS, ETAT)



c = np.arange(0.01, 1, 0.01)

combi_ind = list(itertools.product(*[range(s) for s in ES.shape]))

pool = mp.Pool(processes=mp.cpu_count())


results = pool.map(isdecreasing, combi_ind)


pool.close()
pool.join()

not_decreasing = [res for res in results if res is not None]

