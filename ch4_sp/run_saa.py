"""SAA for the stress-test problem: candidates, bounds, gap (assignment 4.3)."""
import os
import time

import numpy as np
import pandas as pd

from data import load_instance
from stress_scenarios import generate
from profit import profit
from saa_model import solve_saa

SETTINGS = [(50, 5), (1000, 5), (2000, 5)]
N_VAL = 10_000
VAL_SEED = 4999      # one validation sample for all settings (common random numbers)
T_4 = 2.776          # t_{0.975, M-1} for M = 5
Z = 1.96

RESULTS = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS, exist_ok=True)

inst = load_instance()

# Validation sample, generated once and shared by all candidates
_, R_val, D_val, SP_val = generate(N_VAL, VAL_SEED, inst)
print(f"validation seed: {VAL_SEED} (N' = {N_VAL})")

summary = []
for idx, (N, M) in enumerate(SETTINGS, start=1):
    start = time.perf_counter()
    seeds = [4000 + 100 * idx + m for m in range(1, M + 1)]
    print(f"\n=== N = {N}, M = {M}, batch seeds: {seeds} ===")

    # Step 1: solve M independent SAA batches -> candidates Q_m and values v_hat_m
    rows = []
    solve_time = eval_time = 0.0
    for m, seed in enumerate(seeds, start=1):
        _, R, D, SP = generate(N, seed, inst)
        Q_hat, v_hat, t_solve = solve_saa(R, D, SP, inst)
        solve_time += t_solve
        # Step 2: evaluate every candidate on the common validation sample
        t0 = time.perf_counter()
        Pi = profit(Q_hat, R_val, D_val, SP_val, inst)
        eval_time += time.perf_counter() - t0
        rows.append({"m": m, "seed": seed,
                     **{f"Q{i + 1}": q for i, q in enumerate(Q_hat)},
                     "v_hat": v_hat, "g_hat": Pi.mean(), "sd_Pi": Pi.std(ddof=1)})
    cand = pd.DataFrame(rows)

    # Step 3: upper bound on v* (maximization): mean of v_hat + t * sd / sqrt(M)
    v = cand["v_hat"]
    U = v.mean() + T_4 * v.std(ddof=1) / np.sqrt(M)

    # Step 4: pick the candidate with the highest g_hat; lower bound from its profits
    best = cand.loc[cand["g_hat"].idxmax()]
    L = best["g_hat"] - Z * best["sd_Pi"] / np.sqrt(N_VAL)
    gap = U - L
    total_time = time.perf_counter() - start

    with pd.option_context("display.width", 200, "display.precision", 2):
        print(cand.drop(columns="sd_Pi").to_string(index=False))
    print(f"chosen: m = {int(best['m'])}  U = {U:.2f}  L = {L:.2f}  gap = {gap:.2f}")
    print(f"time: Gurobi solve {solve_time:.3f} s, evaluation {eval_time:.3f} s, "
          f"total {total_time:.2f} s")

    cand.to_csv(os.path.join(RESULTS, f"saa_N{N}_M{M}.csv"), index=False)
    summary.append({"N": N, "M": M, "chosen_m": int(best["m"]),
                    **{f"Q{i + 1}": best[f"Q{i + 1}"] for i in range(10)},
                    "g_hat": best["g_hat"], "U": U, "L": L, "gap": gap,
                    "solve_time_s": solve_time, "eval_time_s": eval_time,
                    "total_time_s": total_time})

summary = pd.DataFrame(summary)
summary.to_csv(os.path.join(RESULTS, "saa_summary.csv"), index=False)

# Checks (printed only; these are statistical outcomes, not guarantees)
print("\nchecks:")
print("  U >= L for every setting:", bool((summary["U"] >= summary["L"]).all()))
print("  gap decreases with N:    ", bool(summary["gap"].is_monotonic_decreasing))
