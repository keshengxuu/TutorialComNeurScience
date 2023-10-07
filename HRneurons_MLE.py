#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 16:45:48 2022

@author: ksxu
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os.path
from matplotlib import colors

from matplotlib import rc, cm
import matplotlib.gridspec as gridspec



a=1.0
b=3.0
c=1.0
d=5.0
s=4.0
r=0.006
k=-1.56
i_ext=2.0
t_final = 2000.0
t_tran = 500
dt = 0.01
theta = 0.5
tran =150
trans_steps =  int(1000/dt)
t_steady  =  int(1000/dt)




def run_kut4(F,t,y,dt,args1):
        K0 = dt*F(y,t,args1)
        K1 = dt*F(y + K0/2.0,t + dt/2.0,args1)
        K2 = dt*F( y + K1/2.0,t + dt/2.0,args1)
        K3 = dt*F( y + K2,t + dt,args1)
        return (K0 + 2.0*K1 + 2.0*K2 + K3)/6.0


def HRmodel(y,t,i_ext):
    # y is vector (equ_dim X Num_neruons) of time values
    [x_i,y_i,z_i]= y
    
  
    dx_i = y_i  - a*x_i**3 + b*x_i**2 + i_ext - z_i 
    dy_i = c - d*x_i**2 - y_i
    dz_i= r*(s*(x_i - k) - z_i)
      
    Det = np.array([dx_i, dy_i, dz_i])
    return  Det



def MLEfunc(fun,y0,dt,t_trans,t_steady,args):
     
#     Function to calculate the maximal Lyapunov exponent of
#     an attractor of a system dy/dt=f(y, t) (it is implicitly 
#     assumed that f(y, t) is independent of t)
#     Inputs: dydt - handle to a function that calculates dy/dt
#             y0 - initial condition in the basin of attraction
#             t_trans - time for transients to disappear
#             d0 - initial separation of two orbits
#             delta_t - time step
#             t_max - length of time (after t_trans) to integrate for
#                     (steps 3-5 are repeated until this time is reached)
#     Outputs: mle - running average of the maximal Lyapunov
#                    exponent at each time step
# integrate to get rid of transient behaviour:
        delta_t = dt
        i_ext = args
        time = np.arange(0,t_trans,delta_t)
        Y=integrate.odeint(fun,y0,time,args = (i_ext,))


        d0=0.0001
        y1 = Y[-1,:] #'#;    % final value of solution is y1
        y2=y1+np.append(d0,np.zeros(len(y1)-1));   # perturb by d0 to get y2
        N_steps =int(t_steady/delta_t);   # number of steps required
        sl = np.zeros(N_steps);
        sl0=0
        sum0 = 0;
        t=0
        #   # integrate both orbits by a time delta_t:
        for I in range(N_steps):
        #    y = y + run_kut4(HyB,t,y,dt,tempF)
            y1 =y1+run_kut4(fun,t,y1,delta_t,args);
            y2 =y2+run_kut4(fun,t,y2,delta_t,args);
            t=t+delta_t
            d1 =np.linalg.norm(y2-y1);              # new separation
            lambda0 =np.log(d1/d0)/delta_t;   # Lyapunov exponent
            sum0 = sum0+lambda0;        # running sum of Lyapunov exponents
#            sl[I] = sum0/I;
            y2 = y1+(y2-y1)*d0/d1;   # renormalise y2 so separation is d0
        #end
        ## divide running sum by number of iterations to get running average:
#        mle = sl[0:I];
        mle=sum0/N_steps
        
        return mle



"""
The main function is starting from here          
"""
i_ext = 3.0
#transient states
#initial values
x0 = np.array([1.0, -1.5, -2])

# mle = MLEfunc(HRmodel,x0,dt,t_tran,t_final,args=i_ext)

# print(mle)

i_ext0 = np.linspace(1,4,500)
MLE=np.zeros(len(i_ext0))

for i_ext,i in zip(i_ext0,range(len(i_ext0))):
    mle0 = MLEfunc(HRmodel,x0,dt,t_tran,t_final,args=i_ext)
    MLE[i] = mle0
    

plt.figure(1,figsize=(8,6))

plt.plot(i_ext0,MLE,color = 'blue')

plt.savefig('HRMLE.png',doi=300)
    
    
   
    




