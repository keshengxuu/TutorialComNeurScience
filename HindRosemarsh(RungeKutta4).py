#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 15:51:40 2022

@author: ksxu
"""

from scipy.integrate import odeint
import numpy as np
import numpy
from numpy import exp
import matplotlib.pyplot as plt

a=1.0
b=3.0
c=1.0
d=5.0
s=4.0
r=0.006
k=-1.56
i_ext=2.0
t_fina = 2000.0
t_transient = 500
dt = 0.01
theta = 0.5
tran =150
trans_steps =  int(1000/dt)
start_steps  =  int(1000/dt)


Nsize= 4
weight = 0.0


a_ij=np.array([[1,1,1,1],
               [1,1,1,1],
               [1,1,1,1],
               [1,1,1,1]])


def HRmodel(y,t):
    # y is vector (equ_dim X Num_neruons) of time values
    x_i = y[0,:]
    y_i = y[1,:]
    z_i = y[2,:]
    
    cp = weight*np.sum(np.multiply(a_ij,x_i-x_i[:,None]),axis=1)
  
    dx_i = y_i  - a*x_i**3 + b*x_i**2 + i_ext - z_i + cp
    dy_i = c - d*x_i**2 - y_i
    dz_i= r*(s*(x_i - k) - z_i)
      
    
    return np.array([dx_i, dy_i, dz_i])



def rungekutta4(f, x0, t, args=()):
    '''
    Runge-Kutta ODE integrator.

    Params:
        f   The derivative function to integrate
        y0      Initial x-value.
        t0      Initial t-value for the integrator.
        dt      Time step.
        nsteps  Number of steps to run the integrator.

    Returns: y where
        ts      A vector (1 x nsteps) of time values.
        y      The results (len(x0) x nsteps). 
    '''
    nsteps = len(t)
    equ_dim = np.size(x0,axis=0)    # dimension of  differential equations
    Num_neruons = np.size(x0,axis=1)  # the numbers of  neurons
    y  = np.zeros((equ_dim,Num_neruons,nsteps),dtype=numpy.float64)
    
    y[:,:,0] = x0
    for i in range(nsteps-1):
        h = t[i+1] - t[i]
        k1 = f(y[:,:,i],  t[i], *args)
        k2 = f(y[:,:,i] + k1 * h / 2., t[i] + h / 2., *args)
        k3 = f(y[:,:,i] + k2 * h / 2., t[i] + h / 2., *args)
        k4 = f(y[:,:,i] + k3 * h, t[i] + h, *args)
        y[:,:,i+1] =y[:,:,i] + (h / 6.) * (k1 + 2*k2 + 2*k3 + k4)
    return y




x_i = np.random.uniform(-2,1,size=Nsize)
y_i = np.random.uniform(-2,1,size=Nsize)
z_i = np.random.uniform(-3,-2,size=Nsize)

#transient states
#initial values
x0 = np.array([x_i, y_i, z_i])
t_tran = np.linspace(0, 1000, trans_steps) 

# steady state
t = np.linspace(0, 1000, start_steps) 
ytrans = rungekutta4(HRmodel, x0, t_tran, args=( ))
x0 = ytrans[:,:,-1]

y  = rungekutta4(HRmodel, x0, t, args=( ))
    


fig = plt.figure(1, figsize=(8,6))
plt.clf()
plt.subplot(221)
plt.plot(t,y[0,0,:],c="k")
plt.subplot(222)
plt.plot(y[2,0,:],y[0,0,:],c="k")
plt.subplot(223)
plt.plot(t,y[0,2,:],c="k")
plt.subplot(224)
[plt.plot(t,y[0,i,:]) for i in range(4)]

plt.savefig('HRneuron.png')














