# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 19:43:03 2021

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
taud = 400000 #ms
tauf = 100#ms
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
# plt.figtext(0.03,.15, 'pre',fontsize = 'large')

plt.subplot(312)
plt.plot(record_tVec,record_Isyn[0,:],label='Isyn')
plt.xlim([0,700])
plt.ylim([0,1])
# plt.figtext(0.02,0.85, 'post',fontsize = 'large')
plt.yticks([0,0.5,1])




plt.subplot(313)
plt.plot(record_tVec,record_ux[0,:],label='ux')
plt.xlim([0,700])
# plt.figtext(0.00,0.6, ' synaptic \n efficacy',fontsize = 'large')
plt.legend(fontsize='x-small')
plt.ylim([0,0.7])
plt.xlabel('Time [ms]')
plt.xlabel('Time [ms]')
plt.yticks([0,0.25,0.5,0.75])

# plt.subplot(425)
# plt.plot(record_tVec,record_x[0,:],color='red',label='x')
# plt.plot(record_tVec,record_u[0,:],color='blue',label='u')
# plt.legend(fontsize='6')
# plt.xlim([0,700])
# # plt.figtext(0.01,.4, ' synaptic \n variables',fontsize = 'large')



# P.arrow( x, y, dx, dy, **kwargs )
#plt.subplot(313)

# plt.figtext(0.025,0.97,r'Tsodyks-Uziel-Markram model for short term synaptic plasticity ',fontsize = 'large')

# plt.figtext(0.55,0.92,r'Short-term synaptic plasticity model ',fontsize = 'large')

# plt.figtext(0.55,0.88,r'$\mathrm{\frac{dx}{dt} = \frac{1-x}{\tau_D} - ux \delta(t-t_{sp}) }$',color ='blue',fontsize = 'x-large')
# plt.figtext(0.91,0.88,r'$\mathrm{(Depre)}$',color='red', fontsize = 'large')


# plt.figtext(0.55,0.82,r'$\mathrm{\frac{du}{dt} = \frac{U-u}{\tau_F} + U(1-u) \delta(t-t_{sp}) }$',color ='blue',fontsize = 'x-large')
# plt.figtext(0.91,0.82,r'$\mathrm{(Facil)}$',color='red', fontsize = 'large')

# #plt.figtext(0.02,0.08, ' Synaptic  efficacy',fontsize = 'large')
# #plt.figtext(0.06,0.05, ' i = W*ux',fontsize = 'large',color = 'blue')
# plt.figtext(0.55,0.74, ' Synaptic  input current',fontsize = 'large')
# plt.figtext(0.55,0.69,r'$\mathrm{\frac{dI_{syn}}{dt} = -\frac{I_{syn}}{\tau_{syn}} + ux \delta(t-t_{sp}) }$',color ='blue',fontsize = 'x-large')

# A = r'After neuronal spiking: ($u$ returns to its basline'
# B = r' value $U$ with a time constant $\tau_F$, and x recovers'
# C = r' to its maximum value $x = 1$ with a time constant $\tau_D$)'
# plt.figtext(0.535,0.55, A +'\n' + B +'\n'+ C,fontsize = 'medium')
# plt.figtext(0.58,0.52,r'$\mathrm{u=U+(u-U)*exp(-(t-lastupdate) / \tau_F)}$',color ='blue', fontsize = 'medium')
# plt.figtext(0.58,0.49,r'$\mathrm{x=1+(x-1)*exp(-(t-lastupdate)/ \tau_D)}$', color ='blue', fontsize = 'medium')
# plt.figtext(0.58,0.46,r'$\mathrm{I_{syn}=I_{syn}*exp(-(t-lastupdate)/ \tau_{syn})}$', color ='blue', fontsize = 'medium')


# plt.figtext(0.55,0.35,'Upon arrival of a spike (each presynaptic spike \n triggers modifications of the variables)',fontsize = 'medium')
# plt.figtext(0.58,0.32,r'$x\leftarrow x*(1-u)$',color ='blue',fontsize = 'medium')
# plt.figtext(0.58,0.29,r'$u\leftarrow u + U*(1-u)$', color ='blue', fontsize = 'medium')
# plt.figtext(0.58,0.26,r'$I_{syn}\leftarrow I_{syn} + ux$', color ='blue', fontsize = 'medium')

# #plt.figtext(0.55,0.09,'i+=W*u*x',color ='blue',fontsize = 'medium')

# #plt.figtext(0.01,0.03,r'The phenomenological models produces the begavior of cortical synapses,  ',fontsize = 'medium')
# plt.figtext(0.55,0.20,r'Depressing ($\tau_D > \tau_F$) and facilitating ($\tau_F > \tau_D$). ',color ='red',fontsize = 'medium')
# plt.figtext(0.55,0.14,r'References: ',color ='blue',fontsize = 'small')
# plt.figtext(0.55,0.12,r'Tsodyks et al.,J.Neurosci. 20, RC50(2000).',color ='blue',fontsize = 'small')
# plt.figtext(0.55,0.10,r'Mongillo et al.,Science,319(2008). ',color ='blue',fontsize = 'small')
# plt.figtext(0.55,0.08,r'Mi et al.,  Neuron, 93(2017). ',color ='blue',fontsize = 'small')




#plt.axis('off')

plt.subplots_adjust(left = 0.12,bottom=0.08, right=0.95, top=0.98, wspace=0.3, hspace=0.3)

plt.savefig('short-term plasticty_depre.png',dpi = 300)
# plt.savefig('short_term_plasticty_faci.pdf',dpi = 300)


