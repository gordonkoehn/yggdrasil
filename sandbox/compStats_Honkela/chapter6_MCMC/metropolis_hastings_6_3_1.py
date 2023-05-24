#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extended on example of an inplementation of the Metropolis-Haistings Algorithm as presented
in chapter 6.3.1 in Computational Statistics by Antti Honkela. 

Used - target distribution pi(theta) = - abs(theta) [note the missing exponent]
     - proposal density q(theta) =  theta + normal()
     
     -->  Testing Laplace target with normal proposal centred around the previous value

Created on Wed Feb 22 15:41:56 2023

@author: gordonkoehn

"""

 eimport numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
 
# Metropolis sampling (symmetric proposal) for given log-target distribution
def mhsample0(theta0, n, logtarget, drawproposal):
    theta = theta0
    thetas = np.zeros(n)
    for i in range(n):
        theta_prop = drawproposal(theta)
        if np.log(npr.rand()) < logtarget(theta_prop) - logtarget(theta):
            theta = theta_prop
        thetas[i] = theta
    return thetas
 
# Testing Laplace target with normal proposal centred around the previous value
npr.seed(42)
theta = mhsample0(0.0, 10000, lambda theta: -np.abs(theta),
                  lambda theta: theta+npr.normal())
 
# Discard the first half of samples as warm-up
theta = theta[len(theta)//2:]
fig, ax = plt.subplots(1, 3)


#NB: [::10] takes every 10-th element
ax[0].plot(theta[::10])
ax[0].set_title("theta")
ax[0].set_xlabel("step (every 10-th)")
ax[0].set_ylabel("theta value")

h = ax[1].hist(theta[::10], 50, density=True)
ax[1].set_title("hist of theta")
ax[1].set_xlabel("theta value")
ax[1].set_ylabel("frequency")

ax[2].plot(theta, np.exp(-theta), ",", marker=11)
ax[2].set_title("proposed theta against laplace target")
ax[2].set_ylabel("laplacian target value")
ax[2].set_xlabel("proposed thetas")
plt.show()

print("The final theta is :" + str(theta[-1]))