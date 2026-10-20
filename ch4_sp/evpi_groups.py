"""Value of perfect information on group disruptions (assignment 4.6)."""
import os
from itertools import product

import numpy as np
import pandas as pd

from data import load_instance
from stress_scenarios import generate
from profit import profit
from saa_model import solve_saa

RESULTS = os.path.join(os.path.dirname(__file__), "results")
Q_COLS = [f"Q{i}" for i in range(1, 11)]
N, M, N_VAL = 2000, 5, 10_000

inst = load_instance()

# Final SAA solution (N = 2000 candidate) and its unconditional z_SAA from vss.py
summary = pd.read_csv(os.path.join(RESULTS, "saa_summary.csv"))
Q_SAA = summary.loc[summary["N"] == 2000, Q_COLS].to_numpy(dtype=float)[0]
vss = pd.read_csv(os.path.join(RESULTS, "vss.csv"))
z_SAA = vss.loc[vss["solution"] == "SAA_N2000", "z"].iloc[0]

rows = []
for e in product([0, 1], repeat=3):
    idx = 4 * e[0] + 2 * e[1] + e[2]
    H = sum(e)
    P_e = 0.15 ** H * 0.85 ** (3 - H)
    seeds = [6000 + 100 * idx + m for m in range(1, M + 1)]
    val_seed = 6000 + 100 * idx + 99
    print(f"e = {e}: batch seeds {seeds}, validation seed {val_seed}")

    # Conditional validation sample with the group events fixed at e
    _, R_val, D_val, SP_val = generate(N_VAL, val_seed, inst, E_fixed=e)

    # Same SAA procedure as run_saa.py: M batches, keep the candidate with the highest g_hat
    z_star, Q_star = -np.inf, None
    for seed in seeds:
        _, R, D, SP = generate(N, seed, inst, E_fixed=e)
        Q_hat, _, _ = solve_saa(R, D, SP, inst)
        g_hat = profit(Q_hat, R_val, D_val, SP_val, inst).mean()
        if g_hat > z_star:
            z_star, Q_star = g_hat, Q_hat

    # Q_SAA on the same conditional sample (CRN)
    g_SAA = profit(Q_SAA, R_val, D_val, SP_val, inst).mean()

    rows.append({"E1": e[0], "E2": e[1], "E3": e[2], "H": H, "P_e": P_e,
                 **dict(zip(Q_COLS, Q_star)),
                 "active": " ".join(str(i + 1) for i in np.flatnonzero(Q_star > 1e-6)),
                 "z_star": z_star, "g_SAA": g_SAA})
table = pd.DataFrame(rows)
assert np.isclose(table["P_e"].sum(), 1.0)

# z_PI-G = sum_e P(e) z*_e, compared with Q_SAA on the same conditional samples
P = table["P_e"]
z_PIG = (P * table["z_star"]).sum()
z_SAA_strat = (P * table["g_SAA"]).sum()
EVPI_G = (P * (table["z_star"] - table["g_SAA"])).sum()

with pd.option_context("display.width", 250, "display.precision", 2):
    print()
    print(table.to_string(index=False))
print(f"\nz_PI-G = {z_PIG:.2f}")
print(f"z_SAA,strat = {z_SAA_strat:.2f}   (z_SAA from vss.csv: {z_SAA:.2f})")
print(f"EVPI_G = {EVPI_G:.2f}")

table.to_csv(os.path.join(RESULTS, "evpi_groups.csv"), index=False)
pd.DataFrame([{"z_PIG": z_PIG, "z_SAA_strat": z_SAA_strat, "z_SAA_vss": z_SAA,
               "EVPI_G": EVPI_G}]).to_csv(os.path.join(RESULTS, "evpi_summary.csv"), index=False)
