"""Value of the stochastic solution in the stress test (assignment 4.5)."""
import os

import numpy as np
import pandas as pd

from data import load_instance, COMPARE_SEED, N_COMPARE
from stress_scenarios import generate
from profit import profit
from saa_model import solve_saa

RESULTS = os.path.join(os.path.dirname(__file__), "results")
Q_COLS = [f"Q{i}" for i in range(1, 11)]

inst = load_instance()
p, y = inst["p"], inst["y"]

# Step 1: expectations under the stress distribution.
# E[H] = 3 * 0.15 = 0.45, so E[D] = (1 + 0.1 * 0.45) mu = 1.045 mu (truncation at 0
# ignored) and E[SP] = 45 + 15 E[H] = 51.75. A supplier's group is hit with prob. 0.15.
ED = 1.045 * inst["mu"]
ESP = 51.75
p_bar = 0.85 * p + 0.15 * np.minimum(0.90, p + 0.25)
ER = 1 - p_bar * (1 - y)

# Step 2: check against the sample means of the comparison sample
_, R, D, SP = generate(N_COMPARE, COMPARE_SEED, inst)
print(f"comparison seed: {COMPARE_SEED} (N = {N_COMPARE})")
print("check of expectations (analytic vs sample mean):")
checks = [("E[D]", ED, D), ("E[SP]", ESP, SP)] + [(f"E[R{i + 1}]", ER[i], R[:, i]) for i in range(10)]
for name, exact, x in checks:
    se = x.std(ddof=1) / np.sqrt(len(x))
    print(f"  {name:7s} analytic {exact:9.4f}  sample {x.mean():9.4f}  "
          f"diff = {(x.mean() - exact) / se:+.2f} se")

# Step 3: expected-value solution = SAA LP with the single expected scenario
Q_EV, v_EV, _ = solve_saa(ER[None, :], np.array([ED]), np.array([ESP]), inst)

# Step 4: evaluate Q_EV and Q_SAA (N = 2000 candidate) on the comparison sample (CRN)
summary = pd.read_csv(os.path.join(RESULTS, "saa_summary.csv"))
Q_SAA = summary.loc[summary["N"] == 2000, Q_COLS].to_numpy(dtype=float)[0]
z_EEV = profit(Q_EV, R, D, SP, inst).mean()
z_SAA = profit(Q_SAA, R, D, SP, inst).mean()
VSS = z_SAA - z_EEV

print(f"\nQ_EV  = {np.round(Q_EV, 2)}  (EV objective {v_EV:.2f})")
print(f"Q_SAA = {np.round(Q_SAA, 2)}")
print(f"z_EEV = {z_EEV:.2f}  z_SAA = {z_SAA:.2f}  VSS = {VSS:.2f}")

table = pd.DataFrame([{"solution": "EV", **dict(zip(Q_COLS, Q_EV)), "z": z_EEV},
                      {"solution": "SAA_N2000", **dict(zip(Q_COLS, Q_SAA)), "z": z_SAA},
                      {"solution": "VSS", "z": VSS}])
table.to_csv(os.path.join(RESULTS, "vss.csv"), index=False)
