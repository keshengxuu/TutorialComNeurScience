#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 10:13:15 2022

@author: ksxu
"""

import numpy as np
import matplotlib.pyplot as plt

import csv
plt.rcParams['mathtext.sf'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'

#changing the xticks and  yticks fontsize for all sunplots
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('font',size=12)



C =10
I=0
g_l = 19
E_L = -67
G_Na = 74
v0 = 1.5
k=16
E_Na = 60


timesteps = 4000
dt = 0.1

def minfinity(v):
    m_infty = 1.0/(1.0 + np.exp((v0-v)/k))
    return m_infty 


def nullcline(v):
    
    I_l = g_l*(v-E_L)
    I_NA = G_Na * minfinity(v)*(v-E_Na)
    I_total = I_l +  I_NA +I
    Fv= -I_total/C
    
    return I_l, I_NA, I_total,Fv


def Leak_instatant(v,I0):
    I_l = g_l*(v-E_L)
    I_NA = G_Na * minfinity(v)*(v-E_Na)
    I_total = I_l +  I_NA +I0
    dvdt = -I_total/C
    
    return dvdt
    


v_range = np.linspace(-70,70,400)


I_lsave = np.zeros(np.size(v_range))
I_Nasave = np.zeros(np.size(v_range))
I_totalsave = np.zeros(np.size(v_range))
Fvsave = np.zeros(np.size(v_range))


#calculating the nullcline
for i,v in enumerate (v_range):   
   I_lsave[i],I_Nasave[i],I_totalsave[i],Fvsave[i]= nullcline(v)
   
   
# evolution of action potetial for different current I
v_list = []

Vinitial_range = np.linspace(-70,60,20)
t = np.zeros(timesteps)
vsize = np.size(Vinitial_range)



def state_func(I0):
    dv = np.zeros([vsize,timesteps])
    
    for i,v0 in enumerate(Vinitial_range):
        dv[i,0] = v0
        for j in range(timesteps):
            t[j]=j*dt
            if j>=1:
                dv[i,j] = dv[i,j-1] + dt * Leak_instatant(dv[i,j-1],I0)
          
    return t,dv

    

t,dv = state_func(I0 = 0) 
        
t1,dv1 = state_func(I0 = -60) 
   
   
   
   


plt.figure(1,figsize = (10,8)) 
plt.clf

plt.subplot(221)

plt.plot(v_range,I_lsave/1000,label= 'I_L(v)')
plt.plot(v_range,I_Nasave/1000,label= 'I_NA(v)')
plt.plot(v_range,I_totalsave/1000,'r--',label= 'I(v)')
plt.plot([-70,70],[0,0])

plt.xlabel('menbrance potential')
plt.ylabel('current(nA)')

plt.xlim([-70,70])

plt.ylim([-2.5,1.2])

plt.legend()


plt.savefig('INAcurrent.pdf',dpi = 300)

plt.subplot(222)

plt.plot(v_range,Fvsave,label= 'I_L(v)')
plt.ylim([-50,100])
plt.plot([-70,70],[0,0])
plt.savefig('INAcurrent.pdf',dpi = 300)
plt.xlabel('menbrance potential,V(mV)')
plt.ylabel('Derivation of AP (mV/ms)')


plt.subplot(223)

[plt.plot(t,dv[i,:]) for i in range(20)]
plt.xlim([0,10])
plt.xlabel('times')
plt.ylabel('menbrance potential,V(mV)')


plt.subplot(224)

[plt.plot(t1,dv1[i,:]) for i in range(20)]
plt.xlim([0,10])

plt.xlabel('times')
plt.ylabel('menbrance potential,V(mV)')

plt.savefig('leak_instan_current_INap_DynSynNeur_FIgure35_36.pdf')



plt.subplots_adjust(bottom=0.08,left=0.08,wspace = 0.30,hspace = 0.4,right=0.93, top=0.98)
