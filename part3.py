# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 13:00:29 2026

@author: u126250
"""

import numpy as np

#New supplier data for 10 instances
np.random.seed(3003)
N = 200

Demand = np.random.normal(mu, sigma, N)

Demand[Demand < 0] = 0 

yields = np.random.binomial(n=1, p=1-p_i_new, size = (N, 10))

y_matrix = np.tile(y_i_new, (N, 1))
trueyields = np.where(yields == 0, y_matrix, yields)

quantities = mu/10*np.ones(10)
real_supply = trueyields @ quantities
real_supply_i = trueyields * quantities
c_matrix = np.tile(c_i_new, (N,1))
k = 15
profit = (s - c_matrix + k)@ np.transpose(real_supply_i) - k * Demand - (s - c_matrix + k) @ np.transpose(np.maximum(np.zeros(N), np.sum(quantities) - Demand))
