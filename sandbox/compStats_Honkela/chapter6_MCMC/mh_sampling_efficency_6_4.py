#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extended on example of an inplementation of the Metropolis-Haistings Algorithm as presented
in chapter 6.4 in Computational Statistics by Antti Honkela. 

Basically this proposal distribution makes smaller steps - hence is may converge
to a value.

Created on Wed Feb 22 16:21:10 2023

@author: gordonkoehn
"""

import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
 
# Metropolis sampling (symmetric proposal) for given log-target distribution
def mhsample1(theta0, n, logtarget, drawproposal):
    theta = theta0
    thetas = np.zeros(n)
    accepts = 0
    for i in range(n):
        theta_prop = drawproposal(theta)
        if np.log(npr.rand()) < logtarget(theta_prop) - logtarget(theta):
            theta = theta_prop
            accepts += 1
        thetas[i] = theta
    print("Sampler acceptance rate:", accepts/n)
    return thetas
 
# Testing Laplace target with a narrow normal proposal
theta = mhsample1(0.0, 10000, lambda theta: -np.abs(theta),
                  lambda theta: theta+0.1*npr.normal())
 
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