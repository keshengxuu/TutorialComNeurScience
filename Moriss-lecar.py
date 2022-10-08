#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 20:45:43 2022

@author: ksxuphy
"""

## Example code for BENG 260 Homework 2, Fall 2019
## Questions to mwagner@ucsd.edu or jmford@ucsd.edu

import scipy as sp
import pylab as plt
import numpy as np
from scipy.integrate import odeint

###########################################################
## Problem 2: Morris-Lecar (barnacle muscle fiber) Model ##
###########################################################
# Constants
C_m  =   1.0 # membrane capacitance, in uF/cm^2
g_Ca =   1.1 # maximum conducances, in mS/cm^2
g_K  =   2.0
g_L  =   0.5
E_Ca = 100.0 # Nernst reversal potentials, in mV
E_K  = -70.0
E_L  = -50.0

# Channel gating kinetics
# Functions of membrane voltage
def m_infty(V): return (1.0 + sp.tanh((V + 1.0) / 15.0)) / 2.0
def w_infty(V): return (1.0 + sp.tanh(V / 30.0)) / 2.0
def tau_w(V):   return 5.0 / sp.cosh(V / 60.0)  # in ms

# Membrane currents (in uA/cm^2)
def I_Ca(V):    return g_Ca * m_infty(V) * (V - E_Ca)
def I_K(V, w):  return g_K  * w          * (V - E_K)
def I_L(V):     return g_L               * (V - E_L)

# External current
# step up 10 uA/cm^2 every 100ms
def I_ext(t): return 10*sp.floor(t/100)

# The time to integrate over and the overall current trace
t = sp.arange(0.0, 400.0, 0.1)
I = I_ext(t)


########################################
## Problem 2.1: Channel gating kinetics
Vsweep = sp.linspace(-85.0, 40.0, 500)

plt.figure()

plt.title('2.1: Channel Gating Kinetics')
# recommended style: use same colors for same state variable (m, n, h) and solid/dashed for alpha/beta
plt.plot(Vsweep, m_infty(Vsweep), 'r-', label='$m_\\infty$')
plt.plot(Vsweep, w_infty(Vsweep), 'g-', label='$w_\\infty$')
plt.plot(Vsweep, tau_w(Vsweep), 'g--', label='$\\tau_w$')
plt.xlabel('V (mV)')
plt.ylabel('Kinetics Value')
plt.xlim(Vsweep[0], Vsweep[len(Vsweep)-1])
plt.legend(loc='lower right')

plt.show()


########################################
## Problem 2.2: leaky passive neuron
def dVdt_leak(V, t): return (I_ext(t) - I_L(V)) / C_m
V_leak = odeint(dVdt_leak, E_L, t)

plt.figure()

plt.subplot(2,1,1)
plt.title('2.2: Leaky Passive Neuron')
plt.plot(t, V_leak, 'k')
plt.ylabel('V (mV)')

plt.subplot(2,1,2)
plt.plot(t, I, 'k')
plt.xlabel('t (ms)')
plt.ylabel('$I_{ext}$ ($\\mu{A}/cm^2$)')
plt.ylim(-1, I[len(I)-1]+1)

plt.show()


########################################
## Problem 2.3: Morris-Lecar neuron
def dALLdt(X,t):
    V, w = X
    dVdt = (I_ext(t) - I_Ca(V) - I_K(V, w) - I_L(V)) / C_m
    dwdt = (w_infty(V) - w) / tau_w(V)
    return dVdt, dwdt

# Call the odeint, giving 2 initial conditions, for V and w, in the same order as everything else
X = odeint(dALLdt, [-44, 0.05], t)

V = X[:,0] # the first column is the V values
w = X[:,1] # the second column is the w values

plt.figure()

plt.subplot(3,1,1)
plt.title('2.3: Morris-Lecar Neuron')
plt.plot(t, V, 'k')
plt.ylabel('V (mV)')

plt.subplot(3,1,2)
# recommended style: use the came colors as you did in part 2.1
plt.plot(t, w, 'g', label='w')
plt.ylabel('Gating Value')
plt.legend()

plt.subplot(3,1,3)
plt.plot(t, I_ext(t), 'k')
plt.xlabel('t (ms)')
plt.ylabel('$I_{ext}$ ($\\mu{A}/cm^2$)')
plt.ylim(-1, I[len(I)-1]+1)

plt.show()




