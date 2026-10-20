"""Tests: (a) LP with fixed Q equals profit.py, (b) sanity check with a test Q."""
import numpy as np

from data import load_instance, k, s, r
from stress_scenarios import generate
from profit import profit
from saa_model import solve_saa

inst = load_instance()
print(f"instance 3: mu = {inst['mu']:.2f}, sigma = {inst['sigma']:.2f} (table check passed)")

Q_test = np.zeros(10)
Q_test[1], Q_test[4] = 21.5, 516.1   # Q2 = 21.5, Q5 = 516.1

# (a) With Q fixed, the LP only optimizes the recourse, so its value must equal
#     the average of profit.py on the same scenarios.
_, R, D, SP = generate(200, 1, inst)
for Q in [Q_test, np.full(10, 40.0)]:
    _, v_lp, _ = solve_saa(R, D, SP, inst, Q_fixed=Q)
    v_pf = profit(Q, R, D, SP, inst).mean()
    print(f"(a) LP = {v_lp:.6f}, profit.py = {v_pf:.6f}, diff = {abs(v_lp - v_pf):.2e}")
    assert abs(v_lp - v_pf) < 1e-6

# (b) Test Q on 10,000 scenarios. Compare profit.py per scenario with the
#     explicit formula (revenue + salvage - penalty - purchase - spot cost).
_, R, D, SP = generate(10_000, 2, inst)
Pi = profit(Q_test, R, D, SP, inst)

X = R @ Q_test
B = np.maximum(0, D - X) * (SP < s + k)
lost = np.maximum(0, D - X) - B
Pi_explicit = (s * np.minimum(D, X + B) + r * np.maximum(0, X + B - D) - k * lost
               - R @ (inst["c"] * Q_test) - SP * B)
assert np.max(np.abs(Pi - Pi_explicit)) < 1e-9

print(f"(b) test Q: mean = {Pi.mean():.2f}, sd = {Pi.std(ddof=1):.2f}, "
      f"spot used = {(B > 0).mean():.3f}, mean lost sales = {lost.mean():.2f}, "
      f"max diff vs explicit = {np.max(np.abs(Pi - Pi_explicit)):.1e}")
print("all tests passed")
