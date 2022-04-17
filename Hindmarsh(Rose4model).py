# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 15:16:36 2022

@author: admin
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
k=-1.56
i_ext=3
t_final = 2000.0
t_transient = 500
dt = 0.05
theta = 0.5
tran =150
trans_steps =  int(1000/dt)
start_steps  =  int(1000/dt)
Nsize= 4

tt = np.zeros(start_steps) 
V_t = np.zeros((start_steps,Nsize))




x_i = np.random.uniform(-2,1,size=Nsize)
y_i = np.random.uniform(-2,1,size=Nsize)
z_i = np.random.uniform(-3,-2,size=Nsize)
    

def derivative(x0,t):
    '''
    define Morris-Lecar Model
    '''
    x_i, y_i, z_i = x0
    
    cp = np.sum(np.multiply(a_ij,x_i-x_i[:,None]),axis=1)
  
    dx_i = y_i  - a*x_i**3 + b*x_i**2 + i_ext - z_i  + cp
    dy_i = c - d*x_i**2 - y_i
    dz_i= r*(s*(x_i - k) - z_i)
      
    
    return np.array([dx_i, dy_i, dz_i])

       
       
a_ij=np.array([[1,1,1,1],
               [1,1,1,1],
                [1,1,1,1],
               [1,1,1,1]])



dsdt = (x_i, y_i, z_i)

if __name__ == "__main__":

# start the transient state
   for i in range(trans_steps):
       t=i*dt
       dsdt = dsdt+ dt*derivative(dsdt,t)
       #x0 = dsdt
       
   for i in range(start_steps):
       dsdt = dsdt+ dt*derivative(dsdt,t)
       tt[i] = i*dt
       V_t[i]= dsdt[0] 
       
       
     

       
# start the state state

fig = plt.figure(1, figsize=(8,6))
plt.clf()
plt.subplot(221)
plt.plot(tt,V_t[:,0],lw=2,c="k")
plt.subplot(222)
plt.plot(tt,V_t[:,1],lw=2,c="y")
plt.subplot(223)
plt.plot(tt,V_t[:,2],lw=2,c="r")
plt.subplot(224)
plt.plot(tt,V_t[:,:],lw=2)