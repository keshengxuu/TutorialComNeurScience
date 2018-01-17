#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 14:15:02 2018

@author: ksxu
"""
from __future__ import division 
import numpy as np
import matplotlib.pyplot as plt
import math

"""
Here is an excerpt from Dayan and Abbott, p30 explaining how to simulate spikes:

Spike sequences can be simulated by using some estimate of the firing rate, fr, 
predicted from knowledge of the stimulus, to drive a Poisson process. A simple 
procedure for generating spikes in a computer program is based on the fact that
 the estimated probability of firing a spike during a short interval of duration
 dt is fr*dt. The program progresses through time in small steps of size dt and 
 generates, at each time step, a random number x chosen uniformly in the range 
 between 0 and 1. If x < fr*dt at that time step, a spike is fired; otherwise
 it is not.
 
 Generate a “realistic” spike train in Python.  Read the quotation from
 Dayan and Abbott once more.  We can break it down into three steps

1. Compute the product of fr*dt. Let’s simulate a 10 ms long spike train for a neuron 
firing at 100 Hz.  Therefore, fr = 100 Hz, dt = 1 ms and fr*dt = 0.1 (remember that
 Hz is the inverse of s and 1 ms is 1/1000 s and fr*dt is therefore dimensionless).

2. Generate uniformly distributed random numbers between 0 and 1.  
To generate random numbers, use the python mumpy functon
np.random.uniform(size = (nTrials, nBins)).  np.random.uniform(size = (2, 100)) 
will generate 100 random numbers of two tirals.

3. Compare each random number to fr*dt.  If the product is < fr*dt, then 
there is a spike! We can summarize these three steps into the code below.

Modified from :
https://praneethnamburi.wordpress.com/2015/02/05/simulating-neural-spike-trains/
"""
fr = 20; # Hz
tSim = 15 #s
nTrials = 20

def  poissonSpikeGen(fr, tSim, nTrials):
    dt = 1/1000;  # s
    nBins = int(math.floor(tSim/dt));
    spikeMat = (np.random.uniform(size = (nTrials, nBins)) < fr*dt)*1;
    tVec = np.arange(0,tSim,dt);
    return spikeMat,tVec


spikeMat,tVec = poissonSpikeGen(fr, tSim , nTrials)
tVec = tVec*1000
# calculating the event times
event_times_list = []
for trialCount in range(spikeMat.shape[0]):
    event_times_list.append(tVec[np.where(spikeMat[trialCount, :]==1)[0]])

ISIs=[np.diff(T) for T in event_times_list]



# plotting the figures

plt.figure(1,figsize=(8,6))
plt.subplot(211)
for ith, trial in enumerate(event_times_list):
    plt.vlines(trial, ith + .5, ith + 1.5)
plt.ylim(.5, len(event_times_list) + .5)
plt.xlim([0,1000])
plt.yticks(np.linspace(0,20,5))
plt.xlabel('Time (ms)');
plt.ylabel('Trial Number');

plt.subplot(223)
plt.hist(ISIs[3],bins=20,  normed=1, facecolor='green', alpha=0.75)
plt.xlabel('ISI (ms)')
plt.subplot(224)
plt.hist(ISIs[10],bins=20,  normed=1, facecolor='green', alpha=0.75)
plt.xlabel('ISI (ms)')
plt.tight_layout()
plt.savefig('poissonSpikeGen.png',dpi=300)