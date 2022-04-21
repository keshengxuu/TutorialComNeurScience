#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 21:51:35 2022

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
i_ext=3.0
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


def HRmodel(t,y):
    # y is vector (equ_dim X Num_neruons) of time values
    x_i = y[0,:]
    y_i = y[1,:]
    z_i = y[2,:]
    
    cp = weight*np.sum(np.multiply(a_ij,x_i-x_i[:,None]),axis=1)
  
    dx_i = y_i  - a*x_i**3 + b*x_i**2 + i_ext - z_i + cp
    dy_i = c - d*x_i**2 - y_i
    dz_i= r*(s*(x_i - k) - z_i)
      
    
    return np.array([dx_i, dy_i, dz_i])


def rk4(dfunc, t0, x0, dt, nsteps, **kwargs):
    '''
    Runge-Kutta ODE integrator.

    Params:
        dfunc   The derivative function to integrate, passed as a callable
                dfunc(t, x, *f_args). t is a scalar, x is a vector of the same
                shape as x0.
        t0      Initial t-value for the integrator.
        x0      Initial x-value.
        dt      Time step.
        nsteps  Number of steps to run the integrator.

    Returns: a tuple (ts, xs) where
        ts      A vector (1 x nsteps) of time values.
        xs      The results (len(x0) x nsteps). 
    '''
    ts = t0 + (numpy.array(range(nsteps+1), dtype=numpy.float64) * dt)
    equ_dim = np.size(x0,axis=0)    # dimension of  differential equations
    Num_neruons = np.size(x0,axis=1)  # the numbers of  neurons
    xs  = np.zeros((equ_dim,Num_neruons,nsteps+1),dtype=numpy.float64)
    xs[:,:,0]  = x0
    f_args = kwargs.get('f_args', tuple())
    

    for i in range(1, nsteps+1):
        xs[:,:,i] = _rk4_step(dfunc, xs[:,:,i-1], ts[i-1], dt, *f_args)
    return ts, xs

def _rk4_step(dfunc, xn, tn, dt, *args):
    '''
    Runs one step of Runge-Kutta 4th order and returns the result.

    '''
    k1 = dfunc(tn, xn, *args)
    k2 = dfunc(tn + (dt/2), xn + (k1*dt/2), *args)
    k3 = dfunc(tn + (dt/2), xn + (k2*dt/2), *args)
    k4 = dfunc(tn + dt, xn + k3*dt, *args)
    return xn + dt*((k1 + 2*k2 + 2*k3 + k4) / 6)



x_i = np.random.uniform(-2,1,size=Nsize)
y_i = np.random.uniform(-2,1,size=Nsize)
z_i = np.random.uniform(-3,-2,size=Nsize)

#transient states
#initial values
x0 = np.array([x_i, y_i, z_i])

t, y = rk4(HRmodel, 0.0, x0, 0.01, trans_steps)
x0 = y[:,:,-1]

t, y = rk4(HRmodel, 0.0, x0, 0.01, start_steps)

    


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














