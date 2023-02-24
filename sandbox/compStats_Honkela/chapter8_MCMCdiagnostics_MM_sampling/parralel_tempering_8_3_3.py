#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 11:54:45 2023

Example in 8.3.3  Parralel tempering

@author: gordonkoehn
"""

from sandbox.compStats_Honkela.mcmc_toolbox import mhsample1 


import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
 


import scipy.integrate
 
def ltarget(theta, gamma):
    return -gamma*(theta**2-1)**2
 
# Find the normaliser of the target for visualisation
Z = scipy.integrate.quad(lambda theta: np.exp(ltarget(theta, 64.0)), -2, 2)
 
npr.seed(42)
theta = mhsample1(1.0, 10000, lambda theta: ltarget(theta, 64.0),
              lambda theta: theta + 0.1*npr.normal())

theta = theta[len(theta)//2:]
h = plt.hist(theta, 50, density=True)
plt.xlabel("theta")
plt.ylabel("freq")
t = np.linspace(-1.2, 1.2, 200)
plt.plot(t, np.exp(ltarget(t, 64.0)) / Z[0])
plt.show()

print(np.mean(theta))



#### well simple smapling does not work

### use 5 Chains

betas = np.logspace(-3, 0, 5)
print(betas)


import scipy.integrate
 
def ltarget(theta, gamma):
    return -gamma*(theta**2-1)**2
 
t = np.linspace(-3, 3, 200)
for b in betas:
    # Find the normaliser of the target for visualisation
    Z = scipy.integrate.quad(lambda theta: np.exp(b * ltarget(theta, 64.0)), -4, 4)
    plt.plot(t, np.exp(b * ltarget(t, 64.0)) / Z[0], label=r"""$\beta = %.3f$""" % b)
plt.legend()
plt.show()

###################
# The remaining challenge is to tune the samplers for each  β to achieve a good acceptance.


def pt_target(theta, beta, gamma):
    return beta * ltarget(theta, gamma)
 
def pt_msample(theta0, n, betas, target, drawproposal):
    CHAINS = len(betas)
    accepts = np.zeros(CHAINS)
    swapaccepts = np.zeros(CHAINS-1)
    swaps = np.zeros(CHAINS-1)
    # All variables are duplicated for all the chains
    theta = theta0 * np.ones(CHAINS)
    lp = np.zeros(CHAINS)
    thetas = np.zeros((n, CHAINS))
    for j in range(CHAINS):
        lp[j] = target(theta[j], betas[j])
    for i in range(n):
        # Independent moves for every chain, MH acceptance
        for j in range(CHAINS):
            theta_prop = drawproposal(theta[j], betas[j])
            l_prop = target(theta_prop, betas[j])
            if np.log(npr.rand()) < l_prop - lp[j]:
                theta[j] = theta_prop
                lp[j] = l_prop
                accepts[j] += 1
        # Swap move for two chains, MH acceptance:
        j = npr.randint(CHAINS-1)
        h = target(theta[j+1],betas[j])+target(theta[j],betas[j+1]) - lp[j] - lp[j+1]
        swaps[j] += 1
        if np.log(npr.rand()) < h:
            # Swap theta[j] and theta[j+1]
            temp = theta[j]
            theta[j] = theta[j+1]
            theta[j+1] = temp
            lp[j] = target(theta[j], betas[j])
            lp[j+1] = target(theta[j+1], betas[j+1])
            swapaccepts[j] += 1
        thetas[i,:] = theta
    print('Acceptance rates:', accepts/n)
    print('Swap acceptance rates:', swapaccepts/swaps)
    return thetas
 
npr.seed(42)
betas = np.logspace(-3, 0, 5)
theta = pt_msample(1.0, 10000, betas,
                   lambda theta, beta: pt_target(theta, beta, 64.0),
                   lambda theta, beta: theta + 0.1/np.sqrt(beta)*npr.normal())

theta = theta[len(theta)//2:]
h = plt.hist(theta[:,-1], 50, density=True)
t = np.linspace(-1.2, 1.2, 200)
plt.plot(t, np.exp(ltarget(t, 64.0)) / Z[0])
plt.show()

#### Trace plots of parallel tempering chains

N = len(betas)
fix, ax = plt.subplots(N, 1, figsize=[6.4, 9.6])
for i in range(N):
    ax[i].set_title('beta = %.2f' % betas[i])
    ax[i].plot(theta[:,i])
    plt.subplots_adjust(hspace=0.4)
plt.show()