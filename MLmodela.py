# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 16:37:28 2022

@author: admin
"""

from scipy.integrate import odeint
import numpy as np
from numpy import exp
import matplotlib.pyplot as plt



c = 20.0
g_k = 10.0
g_l = 2.0
g_ca = 5.6
v_1 = -1.2
v_2 = 18.0
v_3 = 12.0
v_4 = 20.0
i_app =50
E_l = -60.0
E_ca = 120.0
E_k = -84.0
t_final = 2000.0
dt = 0.01




def tau_n(v):
    return 1 /(0.04*np.cosh((v-v_3) / (2*v_4)))


def m_inf(v):
    return 0.5 *(1 + np.tanh((v - v_1) / v_2))


def n_inf(v):
    return 0.5 * (1 + np.tanh((v-v_3) / v_4))


def derivative(x0, t):
    '''
    define Morris-Lecar Model
    '''
    v, n = x0
    
    dv = (i_app - g_l*(v - E_l)- g_k*n*(v - E_k)- g_ca * m_inf(v)*(v - E_ca)) / c
    dn = (n_inf(v) - n) / tau_n(v)
    
    return [dv, dn]



v =70.0
n = n_inf(v)
x0 = [v, n]

print(x0)

if __name__ == "__main__":

    t = np.arange(0, t_final, dt)
    
    sol = odeint(derivative, x0, t)
    
    v = sol[:, 0]
    plt.figure(figsize=(7, 3))
    plt.plot(t, v, lw=2, c="k")
    plt.xlim(min(t), max(t))
    plt.ylim(-80, 20)
    plt.xlabel("time [ms]")
    plt.ylabel("v [mV]")
    plt.yticks(range(-80, 80, 20))
    plt.tight_layout()
    plt.savefig("MLplot.png")
    plt.show()
