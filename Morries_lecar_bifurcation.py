#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 09:06:44 2022

@author: ksxuphy
"""
from scipy.integrate import odeint
import numpy as np
from numpy import exp
import matplotlib.pyplot as plt
from scipy import optimize 


plt.rcParams['mathtext.sf'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'

#changing the xticks and  yticks fontsize for all sunplots
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('font',size=12)

c = 20.0
V_k = -84.0
g_k = 8.0
V_ca = 120.0
g_ca = 4.4
V_leak = -60.0
g_leak = 2.0



v_1 = -1.2
v_2 = 18.0
v_3 = 2
v_4 = 30
i_app =150
phi = 0.04




def tau_w(v):
    return 1 /(np.cosh((v-v_3) / (2*v_4)))

def m_inf(v):
    return 0.5 *(1 + np.tanh((v - v_1) / v_2))

def w_inf(v):
    return 0.5 * (1 + np.tanh((v-v_3) / v_4))

def Morris_lecar(x0, I0):
    '''
    define Morris-Lecar Model
    '''
    v, w = x0
    
    dv = (I0 - g_leak*(v - V_leak)- g_k*w*(v - V_k)- g_ca * m_inf(v)*(v - V_ca)) / c
    dw = phi*(w_inf(v) - w) / tau_w(v)
    
    return np.array([dv, dw])


def nullcline_Morris_lecar(vv,I0):
    FV = (I0 - g_leak*(vv - V_leak)- g_ca * m_inf(vv)*(v - V_ca))/(g_k*w*(vv - V_k))
    FW = w_inf(vv)
    return FV,FW


def MLderivs(v,w,I0):
    dvdt = (I0 - g_leak*(v - V_leak)- g_k*w*(v - V_k)- g_ca * m_inf(v)*(v - V_ca)) / c
    dwdt = phi*(w_inf(v) - w) / tau_w(v)
    
    return dvdt,dwdt

def func_ML(x,I0):
    v,w = x
    dvdt = (150 - g_leak*(v - V_leak)- g_k*w*(v - V_k)- g_ca * m_inf(v)*(v - V_ca)) / c
    dwdt = phi*(w_inf(v) - w) / tau_w(v)
    
    return np.array([dvdt,dwdt])

I0=10
xfixedpoit = optimize.fixed_point(func_ML, [-75, 0],args = (I0,))

v =30
w = w_inf(v)
x0= np.array([v, w])
timesteps = 8000
dt = 0.05
t = np.zeros(timesteps)


#varing firing state with different values of inject currents
def state_func(I0):
    dv = np.zeros([2,timesteps])
    dv[:,0] = x0
    for i in range(timesteps):
        t[i]=i*dt
        if i>=1:
            dv[:,i] = dv[:,i-1] + dt * Morris_lecar(dv[:,i-1],I0)
          
    return t,dv

def Nullcline_vari(I0):
    vv = np.linspace(-75,75,400)
    FV = np.zeros(np.size(vv))
    FW = np.zeros(np.size(vv))
    for i,v1 in enumerate(vv) :
        FV[i],FW[i] = nullcline_Morris_lecar(v1,I0)
       
    return vv,FV,FW

#calculation the minmux and maxin value of action potential
# at firing state
I_range = np.linspace(0,300,300)
Vmin = np.zeros(np.size(I_range))
Vmax = np.zeros(np.size(I_range))


for i,I_value in enumerate(I_range):
    _,dv0 = state_func(I0 = I_value)
    Vmin[i] = min(dv0[0,2000:])
    Vmax[i] = max(dv0[0,2000:])
    




t0,dv0 = state_func(I0 = 60)
t1,dv1 = state_func(I0 = 150) 
t2,dv2 = state_func(I0 = 300) 


# calcaation the v-nullcline  and w-nullcline

VV1,FV1,FW1= Nullcline_vari(I0 = 60)
VV0,FV0,FW0= Nullcline_vari(I0 = 150)
VV2,FV2,FW2= Nullcline_vari(I0 = 300)
# print(x0)



fig = plt.figure(1,figsize = (10,8)) 
plt.clf

ax1 = plt.subplot(221)
plt.plot(t0, dv0[0,:], lw=2, c="b",label = r'$\mathsf{I_{app}=60}$')
plt.plot(t1, dv1[0,:], lw=2, c="k",label = r'$\mathsf{I_{app}=150}$')
plt.plot(t2, dv2[0,:], lw=2, c="r",label = r'$\mathsf{I_{app}=300}$')
plt.legend()
plt.xlim(min(t),max(t))
plt.ylim(-80, 20)
plt.xlabel("time [ms]")
plt.ylabel("v [mV]")
plt.yticks(range(-80, 80, 20))

ax2 = plt.subplot(222)

plt.plot(VV0,FV0,'--b')
plt.plot(VV0,FW0,'-.g')
plt.plot(dv1[0,:],dv1[1,:],'r')
plt.xlim([-75,75])
plt.xlabel("v [mV]")
plt.ylabel("w")
# #Plot streamlines
x1,y1 = np.linspace(-75,75,200),np.linspace(0,1,200)
X,Y = np.meshgrid(x1,y1 )
drEdt, drIdt = MLderivs(X,Y,I0=150)

n_skip = 20
plt.quiver(X[::n_skip, ::n_skip], Y[::n_skip, ::n_skip],
             drEdt[::n_skip, ::n_skip], drIdt[::n_skip, ::n_skip],
             angles='xy', scale_units='xy', scale=1., facecolor='c')    
plt.ylim([0,1.0])


ax3 = plt.subplot(223)

plt.plot(VV0,FV0,'b')
plt.plot(VV1,FV1,'g')
plt.plot(VV2,FV2,'r')
plt.plot(VV0,FW0,'k')
plt.xlim([-75,75])
plt.ylim([0,1.0])
plt.text(-50,0.2,'60')
plt.text(-50,0.6,'150')
plt.text(-30,0.8,r'$\mathsf{I_{app}=300}$')
plt.xlabel("v [mV]")
plt.ylabel("w")
ax4 = plt.subplot(224)
plt.plot(I_range,Vmin,'ro')
plt.plot(I_range,Vmax,'g*')
plt.xlabel(r'$\mathsf{I_{app}=300}$(pA)')
plt.ylabel("v [mV]")

plt.tight_layout()
plt.savefig("Morris_lecarLplot.png",dpi = 300)
plt.savefig("Morris_lecarLplot.pdf",dpi = 300)
plt.show
