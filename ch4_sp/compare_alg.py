"""Compare the SAA candidates with Q_alg in the stress test (assignment 4.4)."""
import os

import numpy as np
import pandas as pd

from data import load_instance, p_bench, y_bench, c_bench, COMPARE_SEED, N_COMPARE
from stress_scenarios import generate
from profit import profit
from team_algorithm import preference_order

RESULTS = os.path.join(os.path.dirname(__file__), "results")
Q_COLS = [f"Q{i}" for i in range(1, 11)]

# Step 1: check the copied algorithm against Table 5 of the article
# (benchmark, mu = 400, sigma = 130: Q5 = 516.1, Q2 = 21.5)
Q_article = preference_order(p_bench, y_bench, c_bench.astype(float), 400, 130)
print("benchmark, sigma = 130 (article):", np.round(Q_article, 2))
assert np.allclose(np.round(Q_article, 1), [0, 21.5, 0, 0, 516.1, 0, 0, 0, 0, 0])
# The assignment specifies sigma = 110 for the benchmark; shown for information
Q_bench = preference_order(p_bench, y_bench, c_bench.astype(float), 400, 110)
print("benchmark, sigma = 110 (assignment):", np.round(Q_bench, 2))

# Step 2: Q_alg for instance l* = 3 (risk-neutral model, no stress)
inst = load_instance()
Q_alg = preference_order(inst["p"], inst["y"], inst["c"], inst["mu"], inst["sigma"])

# Step 3: the selected SAA candidates from run_saa.py
summary = pd.read_csv(os.path.join(RESULTS, "saa_summary.csv"))
solutions = {"alg": Q_alg}
for _, row in summary.iterrows():
    solutions[f"SAA_N{int(row['N'])}"] = row[Q_COLS].to_numpy(dtype=float)

# Step 4: one comparison sample, the same for every solution (CRN)
_, R, D, SP = generate(N_COMPARE, COMPARE_SEED, inst)
print(f"\ncomparison seed: {COMPARE_SEED} (N = {N_COMPARE})")
Pi_alg = profit(Q_alg, R, D, SP, inst)

rows = []
for name, Q in solutions.items():
    Pi = profit(Q, R, D, SP, inst)
    # Paired difference d_j = Pi_j(Q_SAA) - Pi_j(Q_alg) with a 95% CI
    if name == "alg":
        d_mean = d_lo = d_hi = np.nan
    else:
        d = Pi - Pi_alg
        half = 1.96 * d.std(ddof=1) / np.sqrt(N_COMPARE)
        d_mean, d_lo, d_hi = d.mean(), d.mean() - half, d.mean() + half
    rows.append({"solution": name, **dict(zip(Q_COLS, Q)),
                 "active": " ".join(str(i + 1) for i in np.flatnonzero(Q > 1e-6)),
                 "mean": Pi.mean(), "sd": Pi.std(ddof=1),
                 "d_mean": d_mean, "d_lo": d_lo, "d_hi": d_hi})
table = pd.DataFrame(rows)

with pd.option_context("display.width", 250, "display.precision", 2):
    print(table.to_string(index=False))
table.to_csv(os.path.join(RESULTS, "compare_alg.csv"), index=False)
