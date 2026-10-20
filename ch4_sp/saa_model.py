"""SAA model: two-stage LP with spot-market recourse, solved with Gurobi."""
import gurobipy as gp
from gurobipy import GRB

from data import k, s, r


def solve_saa(R, D, SP, inst, Q_fixed=None):
    """Solve the SAA LP on the given scenarios and return (Q_hat, v_hat, runtime).

    runtime is Gurobi's solve time (Model.Runtime) in seconds.

    max 1/N sum_w [s z_w + r v_w - SP_w B_w - k L_w - sum_i c_i R_iw Q_i]
    s.t. z_w + v_w = sum_i R_iw Q_i + B_w   (supply + spot = sales + leftover)
         z_w + L_w = D_w                    (demand = sales + lost sales)
         Q, z, v, B, L >= 0
    Q_fixed fixes the first stage (lb = ub = Q), used for testing.
    """
    N, n = R.shape

    # Quiet environment so no license details are printed
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    m = gp.Model("saa", env=env)

    # First stage: order quantities; second stage per scenario: sales z,
    # leftover v, spot purchase B, lost sales L
    Q = m.addMVar(n, lb=0.0, name="Q")
    if Q_fixed is not None:
        Q.lb = Q_fixed
        Q.ub = Q_fixed
    z = m.addMVar(N, lb=0.0, name="z")
    v = m.addMVar(N, lb=0.0, name="v")
    B = m.addMVar(N, lb=0.0, name="B")
    L = m.addMVar(N, lb=0.0, name="L")

    m.addConstr(z + v - R @ Q - B == 0, name="balance")
    m.addConstr(z + L == D, name="demand")

    # Average purchase cost: 1/N sum_w sum_i c_i R_iw Q_i = (c * mean_w R_w) @ Q
    avg_cost = inst["c"] * R.mean(axis=0)
    m.setObjective((s * z.sum() + r * v.sum() - SP @ B - k * L.sum()) / N - avg_cost @ Q,
                   GRB.MAXIMIZE)
    m.optimize()
    assert m.Status == GRB.OPTIMAL

    Q_hat, v_hat, runtime = Q.X.copy(), m.ObjVal, m.Runtime
    m.dispose()
    env.dispose()
    return Q_hat, v_hat, runtime
