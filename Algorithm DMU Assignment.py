# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 15:07:59 2026

@author: maike
"""
# Remember to distinguish the distributions of mu and sigma
import numpy as np
from scipy.optimize import fsolve
from scipy.stats import norm
from itertools import product

# Given number of suppliers: 10
N = 10


b_u = np.array(list(product([0, 1], repeat=N)))
#stockout probabilities
P_u = np.prod(b_u * p_i + (1 - b_u) * (1 - p_i), axis=1)
# implementing given parameters
s = 45
k = 15
#%%
order = np.argsort(c_i, descending=False) 
CR_ordered = CR_i[order] 
y_ordered = y_i[order]
p_ordered = p_i[order]
b_ordered = b_u[:, order]
#%%
order = np.argsort(c_i_new, descending=False) 
CR_ordered = CR_i_new[order] 
y_ordered = y_i_new[order]
p_ordered = p_i_new[order]
b_ordered = b_u[:, order]
#%% Test instance
CR_i_2 = 0.99*np.ones(10)

order = np.argsort(c_i_new, descending=False) 
CR_ordered = CR_i_2[order] 
y_ordered = y_i_new[order]
p_ordered = p_i_new[order]
b_ordered = b_u[:, order]
#%%
def optimalcond(Q, active):
    i = len(active)
    onesi = np.ones(i)
    ones1024 = np.ones((1024, i))
    LHS = CR_ordered[0:i]*(onesi - (onesi-y_ordered[0:i])*p_ordered[0:i])
    Q_u = np.sum(Q * (ones1024 - (b_ordered[:,0:i])*(onesi - y_ordered[0:i])),
                 axis = 1)
    normal = norm(440, 110)
    FQ_u = normal.cdf(Q_u)[:, None]
    P = P_u[:, None]
    return LHS - np.sum(P*(ones1024 - (b_ordered[:,0:i])*
                           (onesi - y_ordered[0:i]))*FQ_u, axis = 0)
def acceptor(Q, active, bound = 0.01):
    resid = optimalcond(Q, active)
    
    bound = (np.abs(resid) <= bound)
    accept_resid = np.all(bound)
    
    nonneg = (Q >= 0)
    accept_nonneg = np.all(nonneg)
    
    return accept_resid, accept_nonneg, nonneg



def kkt_inactivity(root, active, order):
    # i: the count of remaining candidate suppliers, j count of active suppliers
    i = len(order)
    j = len(active)
    onesi = np.ones(i)
    onesj = np.ones(j)
    ones1024 = np.ones((1024, i))
    LHS = CR_ordered[(N-i):]*(onesi - (onesi-y_ordered[(N-i):])*
                              p_ordered[(N-i):])
    Q_u = np.sum(root * (ones1024 - (b_ordered[:,0:j])*
                         (onesj - y_ordered[0:j])), axis = 1)
    normal = norm(440, 110)
    FQ_u = normal.cdf(Q_u)[:, None]
    P = P_u[:, None]
    return LHS - np.sum(P*(ones1024 - (b_ordered[:,(N-i):])*
                           (onesi - y_ordered[(N-i):]))*FQ_u, axis = 0), LHS

order = list(order)  
active = []
inactive = []  
    
while order:
    # Look at the next supplier in the ordering
    active.append(order.pop(0))
    # guess value in line with the paper
    guess = (440/(len(active)))*np.ones(len(active))
    root = fsolve(optimalcond, x0=guess, args = (active), xtol = 1*(10**-4))
    # check if answer meets conditions
    accept_resid, accept_nonneg, nonneg = acceptor(root, active)
    if (accept_resid & accept_nonneg) == True:
        # valid solution --> check KKT conditions
        diff, test = kkt_inactivity(root, active, order)
        print(test)
        for k in range(len(diff)):
            if diff[k] <= 0:
                # LHS <= RHS --> Q_i = 0 for all i in the ordering after k
                inactive.extend(order[k:])
                order = order[:k]
                print('I am over here')
                break
            else:
                # LHS > RHS --> Q_k is still potentially active
                print('I am over there')
                continue          
                
    elif accept_nonneg == False:
        # if Q_i becomes 0 in the solution, then add to inactive
        inactive.extend(np.array(active)[(~nonneg)])
        active = list(np.array(active)[nonneg])
        print('I am over yonder')
        continue
    elif accept_resid == False:
        print('Residuals imply solution does not adequately solve opt. conditions')
        break

