"""Realized stress-test profit per scenario for a fixed order vector Q."""
import numpy as np

from data import k, s, r


def profit(Q, R, D, SP, inst):
    """Pi = (s-r)D + rX - sum_i c_i R_i Q_i - min(s-r+k, SP-r) * (D-X)^+

    Each unit short costs either SP - r (buy on the spot market) or s - r + k
    (lost sale), whichever is cheaper.
    """
    X = R @ Q                       # realized supply per scenario
    purchase = R @ (inst["c"] * Q)  # pay only for delivered units
    shortage = np.maximum(0.0, D - X)
    return (s - r) * D + r * X - purchase - np.minimum(s - r + k, SP - r) * shortage
