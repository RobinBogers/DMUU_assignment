"""Preference-order-based supplier selection (risk-neutral, article Sec. 5.3).

Copied from the team's implementation in "Algorithm DMU Assignment.py"
(lines 14-121, author: maike). The mathematics is unchanged. Changes:
- the data are passed as arguments instead of being taken from kernel globals;
- P_u uses the p of the instance being solved (the original always used the
  benchmark p_i);
- the test cell with CR = 0.99 is left out;
- the loop variable k is renamed to j so it no longer overwrites the cost k;
- debug prints are removed, a failed residual check raises instead of breaking,
  and the result is returned as a full vector Q.
"""
from itertools import product

import numpy as np
from scipy.optimize import fsolve
from scipy.stats import norm

from data import k, s, r


def preference_order(p, y, c, mu, sigma):
    """Return Q (one entry per supplier) for demand D ~ N(mu, sigma^2)."""
    N = len(p)
    CR = (s + k - c) / (s + k - r)

    # All 2^N disruption scenarios b_u (1 = disrupted) and their probabilities
    b_u = np.array(list(product([0, 1], repeat=N)))
    P_u = np.prod(b_u * p + (1 - b_u) * (1 - p), axis=1)

    # Suppliers are considered in increasing order of unit cost c_i
    order = np.argsort(c)
    CR_ordered, y_ordered, p_ordered = CR[order], y[order], p[order]
    b_ordered = b_u[:, order]
    normal = norm(mu, sigma)

    def optimalcond(Q, active):
        # First-order conditions CR_i E[R_i] = sum_u P_u R_iu F(X_u) for the active suppliers
        i = len(active)
        onesi = np.ones(i)
        ones_matr = np.ones((i, np.size(b_ordered, 0)))
        LHS = np.diag(CR_ordered[0:i]) @ (onesi - (np.diag(onesi - y_ordered[0:i])) @ p_ordered[0:i])
        Q_u = Q @ (ones_matr - (np.diag(onesi - y_ordered[0:i]) @ np.transpose(b_ordered[:, 0:i])))
        FQ_u = normal.cdf(Q_u)
        return (LHS - (ones_matr - (np.diag(onesi - y_ordered[0:i]) @ np.transpose(b_ordered[:, 0:i])))
                @ np.diag(FQ_u) @ P_u).flatten()

    def acceptor(Q, active, bound=0.01):
        # Accept a root if the residuals are small and all quantities are non-negative
        resid = optimalcond(Q, active)
        accept_resid = np.all(np.abs(resid) <= bound)
        nonneg = Q >= 0
        return accept_resid, np.all(nonneg), nonneg

    def kkt_inactivity(root, active, order):
        # Marginal value of the remaining suppliers at the current solution (LHS - RHS)
        i = len(order)
        j = len(active)
        onesi = np.ones(i)
        onesj = np.ones(j)
        ones_matrj = np.ones((j, np.size(b_ordered, 0)))
        ones_matri = np.ones((i, np.size(b_ordered, 0)))
        LHS = np.diag(CR_ordered[N - i:]) @ (onesi - (np.diag(onesi - y_ordered[N - i:])) @ p_ordered[N - i:])
        Q_u = root @ (ones_matrj - (np.diag(onesj - y_ordered[0:j]) @ np.transpose(b_ordered[:, 0:j])))
        FQ_u = normal.cdf(Q_u)
        return LHS - (ones_matri - (np.diag(onesi - y_ordered[N - i:]) @ np.transpose(b_ordered[:, N - i:]))) \
            @ np.diag(FQ_u) @ P_u

    order = list(order)
    active = []
    while order:
        # Add the next-cheapest supplier and solve the first-order conditions
        active.append(order.pop(0))
        guess = (440 / len(active)) * np.ones(len(active))
        root = fsolve(optimalcond, x0=guess, args=(active,), xtol=1e-4)
        accept_resid, accept_nonneg, nonneg = acceptor(root, active)
        if accept_resid and accept_nonneg:
            # Valid solution: from the first remaining supplier with LHS <= RHS on,
            # all remaining suppliers stay inactive
            diff = kkt_inactivity(root, active, order)
            for j in range(len(diff)):
                if diff[j] <= 0:
                    order = order[:j]
                    break
        elif not accept_nonneg:
            # A negative quantity: drop that supplier from the active set
            active = list(np.array(active)[nonneg])
            continue
        else:
            raise RuntimeError("residuals imply the solution does not solve the optimality conditions")

    Q = np.zeros(N)
    Q[active] = root
    return Q
