# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 15:07:59 2026

@author: maike
"""
import numpy as np
from scipy.optimize import fsolve
from scipy.stats import norm


# Given number of suppliers: 10
N = 10



#stockout probabilities
P_u = np.prod(b_u * p + (1 - b_u) * (1 - p), axis=1)
# implementing given parameters
s = 45
k = 15
order = np.argsort(c_i, descending=False)
CR_ordered = CR[order]
y_ordered = y[order]
p_ordered = p[order]
b_ordered = b_u[:, order]

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
                           (onesi - y_ordered[(N-i):]))*FQ_u, axis = 0)

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
        diff = kkt_inactivity(root, active, order)
        for k in range(len(diff)):
            if diff[k] <= 0:
                # LHS <= RHS --> Q_i = 0 for all i in the ordering after k
                inactive.extend(order[k:])
                order = order[:k]
                break
            else:
                # LHS > RHS --> Q_k is still potentially active
                continue          
                
    elif accept_nonneg == False:
        # if Q_i becomes 0 in the solution, then add to inactive
        inactive.append(active[(-nonneg)])
        break
    elif accept_resid == False:
        print('Residuals imply solution does not adequately solve opt. conditions')
