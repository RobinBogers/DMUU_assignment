# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 15:07:59 2026

@author: maike
"""
import numpy as np
# Making an array of all possible scenarios
# Given number of suppliers: 10
N = 10
from itertools import product

scenarios = np.array(list(product([0, 1], repeat=N)))



# stockout probabilities p

p = np.array([0.05, 0.25, 0.15, 0.10, 0.20, 0.08, 0.12, 0.04, 0.17, 0.22])

probabilities = np.prod(scenarios * p + (1 - scenarios) * (1 - p), axis=1)