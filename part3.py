# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 13:00:29 2026

@author: u126250
"""
import matplotlib.pyplot as plt
import scipy.special as special



#New supplier data for 10 instances
np.random.seed(3003)
N = 100000

Demand = np.random.normal(mu, sigma, N)

Demand[Demand < 0] = 0 

yields = np.random.binomial(n=1, p=1-p_i_new, size = (N, 10))

y_matrix = np.tile(y_i_new, (N, 1))
trueyields = np.where(yields == 0, y_matrix, yields)
#pick your choice of quantities --> reverse way to find optimal Q
#quantities = np.array([10, 100, 20, 40, 260, 0, 0, 10, 0, 0])
quantities = np.array([0 ,   0,  0,  0, 411, 0, 0,  0, 0, 0])
#quantities = mu/10*np.ones(10)

real_supply = trueyields @ quantities

profit = s * real_supply - (s - r)* np.maximum(real_supply - Demand, np.zeros(N)) - k * np.maximum(Demand - real_supply, np.zeros(N)) - c_i_new @ quantities
mean = np.mean(profit)
stdev = np.std(profit)
print(f'mean profit: {mean}')
print(f'standard deviation of profit: {stdev}')
plt.hist(profit, bins = 100)
df = N - 1 
conf_95 = special.stdtrit(df, 0.975)
print(f'confidence interval of profit: [{mean - conf_95 * stdev / np.sqrt(N)}, {mean + conf_95 * stdev / np.sqrt(N)} ]')
