#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Following this case study in the introduction of PyMc

https://www.pymc.io/projects/docs/en/latest/learn/core_notebooks/pymc_overview.html

Case Studie 2: Coal Mining Disaster

Created on Tue Feb 28 11:47:19 2023

@author: gordonkoehn
"""
###############################################################################
# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az



if __name__ == '__main__':
    ###############################################################################
    # fmt: off
    disaster_data = pd.Series(
        [4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,
        3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5,
        2, 2, 3, 4, 2, 1, 3, np.nan, 2, 1, 1, 1, 1, 3, 0, 0,
        1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
        0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
        3, 3, 1, np.nan, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,
        0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1]
    )
    # fmt: on
    years = np.arange(1851, 1962)
    
    #plt.plot(years, disaster_data, "o", markersize=8, alpha=0.4)
    #plt.ylabel("Disaster count")
    #plt.xlabel("Year");
    
    ###############################################################################
    with pm.Model() as disaster_model:
        switchpoint = pm.DiscreteUniform("switchpoint", lower=years.min(), upper=years.max())
    
        # Priors for pre- and post-switch rates number of disasters
        early_rate = pm.Exponential("early_rate", 1.0)
        late_rate = pm.Exponential("late_rate", 1.0)
    
        # Allocate appropriate Poisson rates to years before and after current
        rate = pm.math.switch(switchpoint >= years, early_rate, late_rate)
    
        disasters = pm.Poisson("disasters", rate, observed=disaster_data)
    ###############################################################################
    with disaster_model:
        idata = pm.sample(10000)
    ###############################################################################
    axes_arr = az.plot_trace(idata)
    plt.draw()
    for ax in axes_arr.flatten():
        if ax.get_title() == "switchpoint":
            labels = [label.get_text() for label in ax.get_xticklabels()]
            ax.set_xticklabels(labels, rotation=45, ha="right")
            break
    plt.draw()