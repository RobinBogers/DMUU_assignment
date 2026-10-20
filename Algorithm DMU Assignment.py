# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 13:09:12 2026

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
    ones_matr = np.ones((i, np.size(b_ordered, 0)))
    LHS = np.diag(CR_ordered[0:i])@(onesi - (np.diag(onesi-y_ordered[0:i]))@
                                    p_ordered[0:i])
    Q_u = Q @ (ones_matr - (np.diag(onesi - y_ordered[0:i])@
                            np.transpose(b_ordered[:, 0:i])))
    normal = norm(mu, sigma)
    FQ_u = normal.cdf(Q_u)
    P = P_u
    return (LHS - (ones_matr - (np.diag(onesi - y_ordered[0:i])@
                               np.transpose(b_ordered[:, 0:i])))@np.diag(FQ_u)@P).flatten()
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
    ones_matrj = np.ones((j, np.size(b_ordered, 0)))
    ones_matri = np.ones((i, np.size(b_ordered, 0)))
    LHS = np.diag(CR_ordered[N-i:])@(onesi - (np.diag(onesi-y_ordered[N-i:]))@
                                    p_ordered[N-i:])
    Q_u = root @ (ones_matrj - (np.diag(onesj - y_ordered[0:j])@
                            np.transpose(b_ordered[:, 0:j])))
    normal = norm(mu, sigma)
    FQ_u = normal.cdf(Q_u)
    P = P_u
    return LHS - (ones_matri - (np.diag(onesi - y_ordered[N-i:])@
                               np.transpose(b_ordered[:, N-i:])))@np.diag(FQ_u)@P, LHS

order = list(order)  
active = []
    
while order:
    # Look at the next supplier in the ordering
    active.append(order.pop(0))
    # guess value in line with the paper
    guess = (440/(len(active)))*np.ones(len(active))
    root = fsolve(optimalcond, x0=guess, args = (active,), xtol = 1*(10**-4))
    # check if answer meets conditions
    accept_resid, accept_nonneg, nonneg = acceptor(root, active)
    if (accept_resid & accept_nonneg) == True:
        # valid solution --> check KKT conditions
        diff, test = kkt_inactivity(root, active, order)
        print(test)
        for k in range(len(diff)):
            if diff[k] <= 0:
                # LHS <= RHS --> Q_i = 0 for all i in the ordering after k
                order = order[:k]
                print('I am over here')
                break
            else:
                # LHS > RHS --> Q_k is still potentially active
                print('I am over there')
                continue          
                
    elif accept_nonneg == False:
        # if Q_i becomes 0 in the solution, then add to inactive
        active = list(np.array(active)[nonneg])
        print('I am over yonder')
        continue
    elif accept_resid == False:
        print('Residuals imply solution does not adequately solve opt. conditions')
        break
