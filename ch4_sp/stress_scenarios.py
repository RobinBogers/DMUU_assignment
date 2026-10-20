"""Scenario generation for the stress-test environment (assignment 3.3 / 4)."""
import numpy as np

from data import GROUPS


def generate(N, seed, inst, E_fixed=None):
    """Generate N stress-test scenarios, independent of Q.

    Returns H (N,), R (N x 10), D (N,), SP (N,).
    E_fixed = (E1, E2, E3) fixes the group events in every scenario (for 4.6).
    """
    rng = np.random.default_rng(seed)
    n = len(inst["p"])

    # Group events E_g ~ Bern(0.15), independent across groups; H = number of hit groups
    E = (rng.random((N, 3)) < 0.15).astype(int)
    if E_fixed is not None:
        E = np.tile(np.asarray(E_fixed, dtype=int), (N, 1))
    H = E.sum(axis=1)

    # Disruption probability per supplier: p_i, or min(0.90, p_i + 0.25) if its group is hit
    p_mod = np.tile(inst["p"], (N, 1))
    for g, members in enumerate(GROUPS):
        hit = E[:, g] == 1
        p_mod[np.ix_(hit, members)] = np.minimum(0.90, inst["p"][members] + 0.25)

    # Z_i = 1{U_i < p_i'}; yield factor R_i = 1 if not disrupted, y_i if disrupted
    U = rng.random((N, n))
    Z = U < p_mod
    R = np.where(Z, inst["y"], 1.0)

    # Stressed demand: each hit group raises mean and sd by 10%
    D0 = rng.normal(inst["mu"], inst["sigma"], N)
    D = np.maximum(0.0, (1 + 0.1 * H) * D0)

    # Spot price SP ~ U(30 + 15H, 60 + 15H)
    SP = 30 + 15 * H + 30 * rng.random(N)

    return H, R, D, SP
