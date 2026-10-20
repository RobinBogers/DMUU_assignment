"""Check that a full Gurobi license is active.

Solves an LP with 3000 variables; the size-limited pip license allows at most
2000, so this only succeeds with a full (e.g. academic) license. Gurobi output
is suppressed so that no license details are printed.
"""
import gurobipy as gp
from gurobipy import GRB

n = 3000

env = gp.Env(empty=True)
env.setParam("OutputFlag", 0)
env.start()

with gp.Model("license_check", env=env) as m:
    x = m.addVars(n, lb=0, ub=1, name="x")
    m.addConstr(x.sum() <= 2500, name="budget")
    m.setObjective(x.sum(), GRB.MAXIMIZE)
    m.optimize()

    print(f"gurobipy {gp.gurobi.version()}")
    print(f"variables: {m.NumVars}, status: {m.Status} (2 = OPTIMAL)")
    print(f"objective: {m.ObjVal:.1f} (expected 2500.0)")
    print(f"runtime: {m.Runtime:.3f} s")

env.dispose()
