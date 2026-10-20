"""Instance l* = 3 (group 3, seed 3003) with the exact generated values."""
import numpy as np

# Cost parameters (same for every instance)
k, s, r = 15, 45, 10

# Supplier groups for the correlated disruptions (0-based: suppliers 1-3, 4-6, 7-10)
GROUPS = [[0, 1, 2], [3, 4, 5], [6, 7, 8, 9]]

# Comparison sample shared by compare_alg.py and vss.py (common random numbers)
COMPARE_SEED = 5000
N_COMPARE = 10_000

# Benchmark data (Table 1 of the assignment)
p_bench = np.array([0.05, 0.25, 0.15, 0.10, 0.20, 0.08, 0.12, 0.04, 0.17, 0.22])
y_bench = np.array([0.80, 0.40, 0.50, 0.70, 0.40, 0.60, 0.75, 0.90, 0.65, 0.85])
c_bench = np.array([40, 25, 35, 30, 21, 55, 45, 37, 52, 42])

# Rounded table of instance 3 (as in AItest.py:44-53), used only as a check
P_TABLE = [0.05, 0.27, 0.11, 0.07, 0.20, 0.10, 0.11, 0.05, 0.16, 0.19]
Y_TABLE = [0.71, 0.36, 0.58, 0.72, 0.43, 0.67, 0.67, 0.87, 0.71, 0.91]
C_TABLE = [38.6, 24.8, 35.0, 31.4, 20.1, 50.1, 47.6, 39.4, 51.6, 43.6]


def load_instance(group_number=3, instance=3):
    """Regenerate instance l with seed 1000g + l.

    The order of the draws (p, y, c, mu, v) is copied from
    Introduction_data_latex.py:30-40, so the values are identical.
    """
    np.random.seed(1000 * group_number + instance)
    p = np.random.uniform(np.maximum(0.02, p_bench - 0.05), np.minimum(0.35, p_bench + 0.05))
    y = np.random.uniform(np.maximum(0.30, y_bench - 0.15), np.minimum(0.95, y_bench + 0.15))
    c = np.random.uniform(np.maximum(r + 1, 0.90 * c_bench), np.minimum(s + k - 1, 1.10 * c_bench))
    mu = np.random.uniform(350, 450)
    v = np.random.uniform(0.20, 0.29)
    sigma = v * mu

    # Check against the rounded table (p, y: 2 decimals, c: 1 decimal) and mu, sigma
    if (group_number, instance) == (3, 3):
        assert np.allclose(np.round(p, 2), P_TABLE)
        assert np.allclose(np.round(y, 2), Y_TABLE)
        assert np.allclose(np.round(c, 1), C_TABLE)
        assert round(mu, 2) == 353.12 and round(sigma, 2) == 76.03

    return {"p": p, "y": y, "c": c, "mu": mu, "sigma": sigma}
