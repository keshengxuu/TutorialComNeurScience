# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 22:08:46 2021

@author: ZXJ
"""
from __future__ import division 
import numpy as np
import matplotlib.pyplot as plt


Ncells = 1
dt =  0.5#ms
#taud = 750 #ms
#tauf = 50 #ms
tuas = 20
U = 0.1
Simu_Time = 1000
taud = 10 #ms
tauf = 4000 #ms
#U =0.6



minvtaud = -1. /taud
minvtauf = -1. /tauf
mintauI = -1. /tuas

Nsteps = int(Simu_Time/dt)

lastSpike = np.zeros(Ncells)   # tmes of last spikes
interval = np.zeros(Ncells)
 # Arrays for recording
dtype=np.float32
record_x = np.zeros((Ncells,Nsteps), dtype=dtype)
record_u = np.zeros((Ncells,Nsteps), dtype=dtype)
record_ux = np.zeros((Ncells,Nsteps), dtype=dtype)
record_Isyn = np.zeros((Ncells,Nsteps), dtype=dtype)
record_tVec = np.zeros((Nsteps), dtype=dtype)



# initial conditions for ux, x,u
# begin main simulation loop
ux = np.zeros(Ncells)
x =np.ones(Ncells)
SynI = 0.0*np.ones(Ncells)
u = 0.3*np.ones(Ncells)
tmp = np.zeros(Ncells)
spiketimes = []
for ti in range(Nsteps):
    t = ti *dt
    for ci in range (Ncells): #  loop over neurons
        if t <=100:
            x[ci] = 1
            u[ci] =0.3
            ux[ci] = 0.3
        else:
            interval[ci] = t -lastSpike[ci]
            u[ci] = U + (u[ci] -U) *np.exp(interval[ci]*minvtauf)
            tmp[ci] = 1.0-u[ci]
            x[ci] = 1.0 + (x[ci]-1)*np.exp(interval[ci] * minvtaud)
            ux[ci] = u[ci] * x [ci]
            SynI[ci] = SynI[ci] * np.exp(interval[ci] * mintauI)

        if 100<t <= 300 and t% 25==0 :
            x[ci] *=tmp[ci]
            u[ci] +=U*tmp[ci]
            ux[ci] += u[ci] * x [ci]
            SynI[ci] = SynI[ci] + ux[ci]
            lastSpike[ci] = t
            spiketimes.append(t)
            
        if 300<t and t% 200==100 :
            x[ci] *=tmp[ci]
            u[ci] +=U*tmp[ci]
            ux[ci] += u[ci] * x [ci]
            SynI[ci] = SynI[ci] + ux[ci]
            lastSpike[ci] = t
            spiketimes.append(t)

        record_x[ci,ti] = x[ci]
        record_u[ci,ti] = u[ci]
        record_ux[ci,ti] = ux[ci]
        record_Isyn[ci,ti] = SynI[ci]
    
    record_tVec[ti] = t
    

x1 = np.linspace(1,5000,1000)
y1 = np.zeros(len(x1))
y1[:] = 0.5


fig=plt.figure(1,figsize=(5,6))
plt.clf()

plt.subplot(311)
for ith, trial in enumerate(np.array(spiketimes)):
    fig = plt.vlines(trial, .5, 1.8,color='blue')
plt.ylim([0.5,2.])
fig.axes.get_yaxis().set_visible(False)
plt.plot(x1,y1,color='blue')
plt.xlim([0,700])
plt.figtext(0.01,0.84, 'pre',fontsize = 'large',rotation = '90')


plt.subplot(312)
plt.plot(record_tVec,record_Isyn[0,:],label='Isyn')
plt.xlim([0,700])
plt.ylim([0,1])
plt.figtext(0.01,0.51, 'post',fontsize = 'large',rotation = '90')
plt.yticks([0,0.5,1])



plt.subplot(313)
plt.plot(record_tVec,record_ux[0,:],label='ux')
plt.xlim([0,700])
plt.figtext(0.01,0.195, 'SE',fontsize = 'large',rotation = '90')
plt.legend(fontsize='x-small')
plt.ylim([0,0.7])
plt.xlabel('Time [ms]')
plt.yticks([0,0.25,0.5,0.75])



plt.subplots_adjust(left = 0.12,bottom=0.08, right=0.95, top=0.98, wspace=0.3, hspace=0.3)

plt.savefig('short-term plasticty_faci.png',dpi = 300)
# plt.savefig('short_term_plasticty_faci.pdf',dpi = 300)

