#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 09:49:32 2023

@author: ksxuphy
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

# initial value for the simulation
n0 = n_inf(v0)
m0 = m_inf(v0)
h0 = h_inf(v0)




def Runge_Kutta_method(x0,t,dt,f):
    nsteps = len(t)
    h = dt
    equ_dim = np.size(x0,axis=0)    # dimension of  differential equations
    y  = np.zeros((equ_dim,nsteps),dtype=numpy.float64)
    y[:,0] = x0
    
    for i in range(nsteps-1):
           k1 = f(y[:,i],  t[i])
           k2 = f(y[:,i] + k1 * h / 2., t[i] + h / 2.)
           k3 = f(y[:,i] + k2 * h / 2., t[i] + h / 2.)
           k4 = f(y[:,i] + k3 * h, t[i] + h)
           y[:,i+1] = y[:,i] + (h / 6.) * (k1 + 2*k2 + 2*k3 + k4)
    return y


start_steps = 3000
t = np.linspace(0, 2000, start_steps) 

x0 = np.array([v0,m0,n0,h0])

y = Runge_Kutta_method(x0,t,dt,hk_equations)




####V-t#######
plt.plot(t, y[0,:],'r',label='$v-t$')
plt.legend(loc='best')
plt.xlabel('t')
plt.show()