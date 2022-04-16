#!/usr/bin/enV python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 14:56:52 2021
keshengxu
ksxu@ujs.edu.cn

@author: ksxuu
"""
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

g_L = 0.1 # oK
g_Na = 35.0 # ok
g_K = 9.0 # ok
E_L = -65.0 #ok
E_Na = 55.0 #ok 
E_K = -90.0 #ok
C_m = 1.0 #ok
fai = 5.0 # ok
I_app = 3
T_start = 0
T_stop = 200
dt = 0.01



def alpha_n(V):
    return -0.01*(V+34)/(np.exp(-0.1*(V+34))-1) # ok RH
def beta_n(V):
    return 0.125*np.exp(-(V+44)/80) # ok RH
def alpha_m(V):
    return -0.1*(V+35)/(np.exp(-0.1*(V+35))-1) # ok RH
def beta_m(V):
    return 4*np.exp(-(V+60)/18) # ok RH
def alpha_h(V):
    return 0.07*np.exp(-(V+58)/20) # ok RH
def beta_h(V):
    return 1/(np.exp(-0.1*(V+28))+1) # ok RH

def m_inf(V):
    return alpha_m(V)/(alpha_m(V) + beta_m(V))

def WBNet_3nodes(x0):
    
    V, h, n= x0
    
    I_L = g_L*(V-E_L)
    I_Na = g_Na*m_inf(V)**3*h*(V-E_Na)
    I_K = g_K*n**4*(V-E_K)
    
    Igap = 0.1*np.sum((V[:,None] - V),axis =1) # the gap jucntion currents
    
    dVdt = (-I_Na-I_K-I_L-Igap+I_app)/C_m
    dhdt = fai*(alpha_h(V)*(1.0-h)-beta_h(V)*h)
    dndt = fai*(alpha_n(V)*(1.0-n)-beta_n(V)*n)
    
    return np.array([dVdt, dhdt, dndt])



# initial Value
N=3
V = np.random.normal(-70,10,size=N)  
h = np.random.normal(1,1,size=N) 
n = np.random.normal(0.1,1,size=N) 

x0 = np.array([V, h, n])
dsdt = np.zeros((3,3)) 
Nstep = 20000
time1 = []

# Nstep and N denote total simulation time steps and numbers of neurons (network size),respectively.
tt = np.zeros(Nstep) 
V_t = np.zeros((Nstep,N))

if __name__ == "__main__":
        
     # transient state
    for i in range(1000):
#        dsdt = dsdt + WBNet_3nodes(x0)*dt
        dsdt += WBNet_3nodes(x0)*dt
        x0 = dsdt
        
        
    # start the  main simulation
    for i in range(Nstep):
#        dsdt = dsdt + WBNet_3nodes(x0)*dt
        dsdt += WBNet_3nodes(x0)*dt
        #print (dsdt )
        tt[i] = i*dt
        x0 = dsdt
        V_t[i]= dsdt[0]
          

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
    #plt.ylim([-60,20])







