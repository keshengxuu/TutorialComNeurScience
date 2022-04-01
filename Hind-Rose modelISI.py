#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 14:40:58 2022

@author: ksxuphy
"""
from scipy.integrate import odeint
import numpy as np
from numpy import exp
import matplotlib.pyplot as plt

a=1.0
b=3.0
c=1.0
d=5.0
s=4.0
r=0.006
x_1=-1.56
i_ext=3
t_final = 2000.0
t_transient = 500
dt = 0.1
theta = 0.5


def derivative(x0,t,para1):
    '''
    define Morris-Lecar Model
    '''
    x, y, z = x0
    i_ext = para1
    dx = y - a*x**3 + b*x**2 + i_ext - z
    dy = c - d*x**2 - y
    dz = r*(s*(x - x_1) - z)
    return np.array([dx, dy, dz])


x = -0.01
y = -2.0
z = -3.0
x0 = np.array([x, y, z])
print(x0) 



def ISIfunc(v,t,para):
    '''
    input variables:
    v: action pontential
    t: the simulaltion times
    para: control parameters
    return
    ISI:   interspike intervalS
    i_vec :  control parameters vector
    '''
    i_ext = para
    spikes=(np.diff(1*(x>theta))==1).nonzero()[0]  
    sptime= t[spikes]
    ISI= np.diff(sptime)
    i_vec =  np.ones(np.size(ISI))*i_ext
    
    return ISI, i_vec

ISIs = []
Ivecs = []
if __name__ == "__main__":
    iext = np.linspace(1,4,300 )
    
    for i0_ext in iext: 
    # start the transient state
        t = np.arange(0, t_transient, dt)
        sol = odeint(derivative, x0, t, args=(i0_ext,))
        x0 =  sol[-1,:]
    # start the state state
        t = np.arange(0, t_final, dt)
        sol = odeint(derivative, x0, t, args=(i0_ext,))
        x=sol[:,0]
    # calculate the interspike intervalS
        ISI, i_vec = ISIfunc(x,t,i0_ext)
        
        ISIs.append(ISI)
        Ivecs.append(i_vec)
        
        
    plt.figure(figsize=(7, 3))
    plt.clf()
    for iex,isis in zip(Ivecs,ISIs):
        plt.scatter(iex,isis,marker='.', color = 'b')
        
    plt.xlabel('I',fontsize = 'large')
    plt.ylabel('ISIs',fontsize = 'large')
    plt.savefig('ISI.png',dpi= 400)
    
    
    # plt.figure(figsize=(7, 3))
    # plt.subplot(211)
    # plt.plot(t, x, lw=1)
    # plt.subplot(212)
    # plt.scatter(i_vec,ISI,marker='.')
