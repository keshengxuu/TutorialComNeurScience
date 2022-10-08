# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 10:47:27 2022

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
x_1=-1.56
i=3
t_final = 2000.0
dt = 0.1


def derivative(x0,t):
    '''
    define Morris-Lecar Model
    '''
    x, y, z = x0
    
    dx = y - a*x**3 + b*x**2 + i - z
    dy = c - d*x**2 - y
    dz = r*(s*(x - x_1) - z)
    return np.array([dx, dy, dz])


x = -0.01
y = -2.0
z = -3.0
x0 = np.array([x, y, z])
print(x0) 

if __name__ == "__main__":

    t = np.arange(0, t_final, dt)
    
    sol = odeint(derivative, x0, t)
    
    x=sol[5000:, 0]
    z=sol[5000:, 2]
    plt.figure(figsize=(7, 3))
    plt.plot(z, x, lw=1, c="k")
    plt.xlim(2.6, 3.2)
    plt.ylim(-2, 2)
   
    plt.tight_layout()
    plt.savefig("MRplot.png")
    plt.show()