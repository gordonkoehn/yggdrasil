#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 14:15:37 2023

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
 

# Define the normal pdf. Note parameters: mean, standard deviation
# Note the sum to allow evaluation for a data set at once
def lnormpdf(x, mu, sigma):
    return np.sum(-0.5*np.log(2*np.pi) - np.log(sigma) - 0.5 * (x-mu)**2/sigma**2)
 
# Define the target log-pdf as a sum of likelihood and prior terms
def target(mu, data, sigma_x, mu0, sigma0):
    return lnormpdf(data, mu, sigma_x) + lnormpdf(mu, mu0, sigma0)
 
# Simulate n=100 points of normally distributed data about mu=0.5
n = 100
data = 0.5 + npr.normal(size=n)
 
# Set the prior parameters
sigma_x = 1.0
mu0 = 0.0
sigma0 = 3.0
 
# Run the sampler
theta = mhsample1(0.0, 10000,
                  lambda mu: target(mu, data, sigma_x, mu0, sigma0),
                  lambda theta: theta+0.2*npr.normal())
 
# Discard the first half of samples as warm-up
theta = theta[len(theta)//2:]
fig, ax = plt.subplots(1, 2)


#NB: [::10] takes every 10-th element
ax[0].plot(theta[::10])
ax[0].set_title("theta")
ax[0].set_xlabel("step (every 10-th)")
ax[0].set_ylabel("theta value")

h = ax[1].hist(theta[::10], 50, density=True)
ax[1].set_title("hist of theta")
ax[1].set_xlabel("theta value")
ax[1].set_ylabel("frequency")

#plt.show()


tt = np.linspace(0.1, 0.9, 50)
m_post = sigma0**2 * np.sum(data) / (n*sigma0**2 + sigma_x**2)
s2_post = 1/(n/sigma_x**2 + 1/sigma0**2)
y = np.array([np.exp(lnormpdf(t, m_post, np.sqrt(s2_post))) for t in tt])
plt.plot(tt, y)
plt.show()

print("The final theta is :" + str(theta[-1]))