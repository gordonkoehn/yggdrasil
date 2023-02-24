#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions that appeared often in the book and are shared between the scripts.


Created on Fri Feb 24 11:56:38 2023

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