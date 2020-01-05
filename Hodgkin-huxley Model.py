# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import matplotlib.pylab as plt

"""
The Hodgkin-Huxley model is based on voltage-clamp experiments on the squid giant axon, the model incorporates 
voltage-sensitive ion channels into the circuit model of the membrane to describe the generation and propagation
of action potentials.

The two sodium (m, h) and one potassium (n) variables effect the relative conductances of their respective ion channels. 
As the channels open or close, the ionic currents contributed by each can either depolarize or hyperpolarize the membrane
potential of the cell. This will give rise to the action potential phenomena given sufficient input current.
"""


# K channel
alpha_n = np.vectorize(lambda v: 0.01*(-v + 10)/(np.exp((-v + 10)/10) - 1) if v != 10 else 0.1)
beta_n  = lambda v: 0.125*np.exp(-v/80)
n_inf   = lambda v: alpha_n(v)/(alpha_n(v) + beta_n(v))

# Na channel (activating)
alpha_m = np.vectorize(lambda v: 0.1*(-v + 25)/(np.exp((-v + 25)/10) - 1) if v != 25 else 1)
beta_m  = lambda v: 4*np.exp(-v/18)
m_inf   = lambda v: alpha_m(v)/(alpha_m(v) + beta_m(v))

# Na channel (inactivating)
alpha_h = lambda v: 0.07*np.exp(-v/20)
beta_h  = lambda v: 1/(np.exp((-v + 30)/10) + 1)
h_inf   = lambda v: alpha_h(v)/(alpha_h(v) + beta_h(v))

### channel activity ###
v = np.arange(-50,151) # mV
fig2=plt.figure()
plt.plot(v, m_inf(v), v, h_inf(v), v, n_inf(v))
plt.legend(('m','h','n'))
plt.title('Steady state values of ion channel gating variables')
plt.ylabel('Magnitude')
plt.xlabel('Voltage (mV)')

## setup parameters and state variables
T     = 55    # ms
dt    = 0.025 # ms
time  = np.arange(0,T+dt,dt)

## HH Parameters
V_rest  = 0      # mV
Cm      = 1      # uF/cm2
gbar_Na = 120    # mS/cm2
gbar_K  = 36     # mS/cm2
gbar_l  = 0.3    # mS/cm2
E_Na    = 115    # mV
E_K     = -12    # mV
E_l     = 10.613 # mV

Vm      = np.zeros(len(time)) # mV
Vm[0]   = V_rest
m       = m_inf(V_rest)      
h       = h_inf(V_rest)
n       = n_inf(V_rest)

## Stimulus
I = np.zeros(len(time))
for i, t in enumerate(time):
  if 5 <= t <= 30: I[i] = 10 # uA/cm2

## Simulate Model
for i in range(1,len(time)):
  g_Na = gbar_Na*(m**3)*h
  g_K  = gbar_K*(n**4)
  g_l  = gbar_l

  m += dt*(alpha_m(Vm[i-1])*(1 - m) - beta_m(Vm[i-1])*m)
  h += dt*(alpha_h(Vm[i-1])*(1 - h) - beta_h(Vm[i-1])*h)
  n += dt*(alpha_n(Vm[i-1])*(1 - n) - beta_n(Vm[i-1])*n)

  Vm[i] = Vm[i-1] + (I[i-1] - g_Na*(Vm[i-1] - E_Na) - g_K*(Vm[i-1] - E_K) - g_l*(Vm[i-1] - E_l)) / Cm * dt 

## plot membrane potential trace
fig=plt.figure()
plt.plot(time, Vm, time, -30+I)
plt.title('Hodgkin-Huxley')
plt.ylabel('Membrane Potential (mV)')
plt.xlabel('Time (msec)')

plt.show()
