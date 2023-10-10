#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 14:42:48 2023

@author: ksxu
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy

###参数确定####
cM=1
gna=120
gK=36
gl=0.3
Ena=50
Ek=-77
El=-54.4
t1=3000
Q=3
T=30
v0 = 15
dt =0.01


def alpha_n(v):
    return 0.01*(v+55.)/(1.-np.exp(-(v+55.0)/10.0))
def beta_n(v):
    return 0.125 * np.exp(-(v + 65.0) / 80.0)
def alpha_m(v):
    return 0.1*(v+40.)/(1-np.exp(-(v+40.)/10.0))
def  beta_m(v):
    return 4.*np.exp(-(v+65.)/18.0)

def alpha_h(v):
    return 0.07*np.exp(-(v+65.)/20.)
def beta_h(v):
    return 1./(1.+np.exp(-(v+35.)/10.))


# state function
def h_inf(v):
    return alpha_h(v) / (alpha_h(v) + beta_h(v))


def m_inf(v):
    return alpha_m(v) / (alpha_m(v) + beta_m(v))


def n_inf(v):
    return alpha_n(v) / (alpha_n(v) + beta_n(v))



def hk_equations(y,t):
    v,m,n,h = y
    dvdt = (-gna*m**3*h*(v-Ena)-gK*n**4*(v-Ek)-gl*(v-El)+30)/cM
    dmdt = Q*(alpha_m(v)*(1-m)-beta_m(v)*m)
    dndt = Q*(alpha_n(v)*(1-n)-beta_n(v)*n)
    dhdt = Q*(alpha_h(v)*(1-h)-beta_h(v)*h)
    return   np.array([dvdt,dmdt,dndt,dhdt])




def Runge_Kutta4(x0,t,dt,f):
    
    y = x0
    h = dt  # time steps
    # the Runge_Kutta4 (below frame)
    k1 = f(y,  t)
    k2 = f(y + k1 * h / 2., t + h / 2.)
    k3 = f(y + k2 * h / 2., t + h / 2.)
    k4 = f(y + k3 * h, t + h)
    y = y + (h / 6.) * (k1 + 2*k2 + 2*k3 + k4)
    
    return y


start_steps = 6000
equ_dim = 4 # the dimensions ofequation
t = np.linspace(0, 500, start_steps)  # the span of time for the simualtion
nsteps = len(t)  # the total steps for the sumulation


# initial value for the simulation
n0 = n_inf(v0)
m0 = m_inf(v0)
h0 = h_inf(v0)
y = np.array([v0,m0,n0,h0]) # initial state value for the simulation

# the arrays for saving simulation data 
y_data  = np.zeros((equ_dim,nsteps),dtype=numpy.float64)


if __name__ == "__main__":
    

    for i in range(nsteps-1): 
        y = Runge_Kutta4(y,t,dt,hk_equations)
        y_data[:,i] = y
    
    v = y_data[0, :]
    m = y_data[1, :]
    n = y_data[2, :]
    h = y_data[3, :]

    fig, ax = plt.subplots(2, figsize=(7, 6), sharex=True)
    ax[0].plot(t, v, lw=2, c="k")
    
    ax[1].plot(t, m, lw=2, label="m", c="b")
    ax[1].plot(t, h, lw=2, label="h", c="g")
    ax[1].plot(t, n, lw=2, label="n", c="r")


    ax[0].set_xlim(min(t), max(t))
    ax[0].set_ylim(-100, 50)
    ax[1].set_xlabel("time [ms]")
    ax[0].set_ylabel("v [mV]")
    ax[0].set_yticks(range(-100, 100, 50))
    ax[1].legend()
    plt.tight_layout()
    plt.savefig("fig_4_1.png")
    # pl.show()