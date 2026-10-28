#!/usr/bin/env python3
"""
DMUU Assignment 2026-27 -- Newsvendor with unreliable suppliers
Group 3, selected random instance l* = 3

Usage:
    python dmuu_assignment.py --part all            # everything (full size)
    python dmuu_assignment.py --part 3              # a single part (1..6)
    python dmuu_assignment.py --part all --fast     # small/quick sanity run

Requires: numpy, scipy (>=1.9), pandas, matplotlib.  cvxpy is optional (Part 6.3 SOCP;
without it a smooth concave maximisation with scipy is used, which is equivalent).
Outputs (tables as CSV, figures as PNG) are written to ./output
"""
import argparse, itertools, os, time, math
import numpy as np
import pandas as pd
from scipy import sparse, stats, optimize
from scipy.stats import norm
from scipy.optimize import linprog, milp, LinearConstraint, Bounds
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Global settings
# ----------------------------------------------------------------------------
GROUP, LSTAR = 3, 3
K, S, R = 15.0, 45.0, 10.0                  # k (stockout), s (price), r (salvage)
SEED = 1000 * GROUP + LSTAR                 # base seed for all simulations (=3003)
OUT = "output"
GROUPS = [[0, 1, 2], [3, 4, 5], [6, 7, 8, 9]]   # supplier groups for the stress test (0-indexed)
np.set_printoptions(precision=3, suppress=True, linewidth=150)
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)


def savefig(name):
    os.makedirs(OUT, exist_ok=True)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, name), dpi=150); plt.close()


def savecsv(df, name, index=False):
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, name), index=index)


def header(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


# ----------------------------------------------------------------------------
# Part 1: data
# ----------------------------------------------------------------------------
P0 = np.array([.05, .25, .15, .10, .20, .08, .12, .04, .17, .22])
Y0 = np.array([.80, .40, .50, .70, .40, .60, .75, .90, .65, .85])
C0 = np.array([40, 25, 35, 30, 21, 55, 45, 37, 52, 42], float)


class Instance:
    def __init__(self, p, y, c, mu, sigma, name):
        self.p, self.y, self.c = np.asarray(p, float), np.asarray(y, float), np.asarray(c, float)
        self.mu, self.sigma, self.name, self.n = float(mu), float(sigma), name, len(p)
        self.ER = 1 - self.p * (1 - self.y)                       # E[R_i]
        self.VR = self.p * (1 - self.p) * (1 - self.y) ** 2        # Var(R_i)
        self.ceff = self.c * self.ER
        self.CR = (S + K - self.c) / (S + K - R)

    def table(self):
        return pd.DataFrame({"supplier": np.arange(1, self.n + 1), "p": self.p, "y": self.y, "c": self.c,
                             "c_eff": self.ceff, "CR": self.CR, "E[R]": self.ER, "Var(R)": self.VR})


def benchmark():
    return Instance(P0, Y0, C0, 400.0, 110.0, "benchmark")


def gen_instances(group=GROUP):
    """seed_{g,l} = 1000 g + l.  Draw order: p (10), y (10), c (10), mu, v."""
    inst = []
    for l in range(1, 11):
        rng = np.random.default_rng(1000 * group + l)
        p = rng.uniform(np.maximum(.02, P0 - .05), np.minimum(.35, P0 + .05))
        y = rng.uniform(np.maximum(.30, Y0 - .15), np.minimum(.95, Y0 + .15))
        c = rng.uniform(np.maximum(R + 1, .9 * C0), np.minimum(S + K - 1, 1.1 * C0))
        mu = rng.uniform(350, 450)
        v = rng.uniform(.20, .29)
        inst.append(Instance(p, y, c, mu, v * mu, f"random_{l}"))
        inst[-1].v = v
    return inst


def part1(instances):
    header("PART 1: random instances")
    rows = []
    for l, I in enumerate(instances, 1):
        t = I.table().round({"p": 3, "y": 3, "c": 2, "c_eff": 2, "CR": 3, "E[R]": 3, "Var(R)": 3})
        savecsv(t, f"part1_instance_{l}.csv")
        rows.append({"instance": l, "mu": round(I.mu, 2), "sigma": round(I.sigma, 2), "v": round(I.v, 3)})
        print(f"\nInstance {l}:  mu={I.mu:.2f}  sigma={I.sigma:.2f}  v={I.v:.3f}"); print(t.to_string(index=False))
    savecsv(pd.DataFrame(rows), "part1_demand_params.csv")
    print("\nDemand parameters:\n", pd.DataFrame(rows).to_string(index=False))


# ----------------------------------------------------------------------------
# Normal loss functions and analytical profit measures (Section 2)
# ----------------------------------------------------------------------------
def n_loss(x, mu, sg):       # E[(D-x)^+]
    z = (x - mu) / sg; return sg * (norm.pdf(z) - z * norm.sf(z))


def n_comp(x, mu, sg):       # E[(x-D)^+]
    return (x - mu) + n_loss(x, mu, sg)


def n_loss2(x, mu, sg):      # E[((D-x)^+)^2]
    z = (x - mu) / sg; return sg ** 2 * ((1 + z ** 2) * norm.sf(z) - z * norm.pdf(z))


def n_comp2(x, mu, sg):      # E[((x-D)^+)^2]
    z = (x - mu) / sg; return sg ** 2 * ((1 + z ** 2) * norm.cdf(z) + z * norm.pdf(z))


def enum_scen(I):
    """All 2^n supplier-yield scenarios: yield-factor matrix (U x n) and probabilities."""
    B = np.array(list(itertools.product([0, 1], repeat=I.n)), float)
    Pu = np.prod(np.where(B == 1, I.p, 1 - I.p), axis=1)
    return 1 - (1 - I.y) * B, Pu


def analytic_stats(Q, I, Ru, Pu):
    """E[G(Q)], variance G_V(Q) (law of total variance), weighted variance G_WV(Q)."""
    X = Ru @ Q
    GEu = (S + K) * X - (Ru * I.c) @ Q - K * I.mu - (S - R + K) * n_comp(X, I.mu, I.sigma)
    nn, ll = n_comp(X, I.mu, I.sigma), n_loss(X, I.mu, I.sigma)
    GVu = (S - R) ** 2 * n_comp2(X, I.mu, I.sigma) + K ** 2 * n_loss2(X, I.mu, I.sigma) \
        - ((S - R) * nn + K * ll) ** 2
    GE = Pu @ GEu; GWV = Pu @ GVu; GV = GWV + Pu @ GEu ** 2 - GE ** 2
    return GE, GV, GWV


# ----------------------------------------------------------------------------
# Part 2: Preference-Order-Based Supplier Selection Algorithm (Algorithm 1)
# ----------------------------------------------------------------------------
def cond_rhs(Q, I, Ru, Pu):
    """vector_i = sum_u P_u R_iu F(X_u(Q))   (right-hand side of optimality conditions (6)/(7))"""
    F = norm.cdf((Ru @ Q - I.mu) / I.sigma)
    return (Pu * F) @ Ru


def _neg_GE_and_grad(q, idx, I, Ru, Pu):
    Q = np.zeros(I.n); Q[idx] = q
    GE = analytic_stats(Q, I, Ru, Pu)[0]
    g = (S - R + K) * (I.CR * I.ER - cond_rhs(Q, I, Ru, Pu))     # dGE/dQ_i
    return -GE, -g[idx]


def algorithm1(I, Ru=None, Pu=None, verbose=True):
    """Suppliers sorted by unit cost c_i; sets A (active), I (inactive), R (remaining).
    Move next supplier R->A, solve FOC (6) for A with fsolve (start mu/|A|); then test inactivity
    condition (7) for the next supplier in R; if it holds, all remaining suppliers go to I."""
    if Ru is None: Ru, Pu = enum_scen(I)
    n = I.n; order = list(np.argsort(I.c, kind="stable"))
    lhs = I.CR * I.ER
    A, Rem, Inact, Q, log = [], order.copy(), [], np.zeros(n), []
    while Rem:
        j = Rem.pop(0); A.append(j)
        def f(qa):
            Qt = np.zeros(n); Qt[A] = qa
            return (cond_rhs(Qt, I, Ru, Pu) - lhs)[A]
        q0 = np.full(len(A), I.mu / len(A))
        qa, info, ier, _ = optimize.fsolve(f, q0, xtol=1e-10, full_output=True)
        if ier != 1 or np.max(np.abs(f(qa))) > 1e-6 or np.any(qa < -1e-6):     # fall back (problem is concave)
            r = optimize.minimize(_neg_GE_and_grad, q0, args=(A, I, Ru, Pu), jac=True, method="L-BFGS-B",
                                  bounds=[(0, None)] * len(A), options=dict(ftol=1e-15, gtol=1e-9, maxiter=2000))
            qa = r.x
        Q = np.zeros(n); Q[A] = np.maximum(qa, 0)
        # drop suppliers whose quantity collapsed to zero (re-elimination)
        zero = [i for i in A if Q[i] <= 1e-6]
        for i in zero: A.remove(i); Inact.append(i)
        rhs = cond_rhs(Q, I, Ru, Pu)
        log.append(dict(step=len(log) + 1, added=int(j) + 1, active=[int(a) + 1 for a in sorted(A)], Q=Q.round(2).copy()))
        if Rem and lhs[Rem[0]] <= rhs[Rem[0]] + 1e-9:      # inactivity condition (7) holds for next supplier
            Inact += Rem; Rem = []
    if verbose:
        for L in log: print(f"  iter {L['step']}: add supplier {L['added']} -> active {L['active']}")
    return Q, sorted(A), log


def global_check(I, Ru, Pu, Qalg):
    """Independent check: maximise the concave expected profit over all suppliers directly."""
    r = optimize.minimize(_neg_GE_and_grad, np.full(I.n, 20.), args=(list(range(I.n)), I, Ru, Pu), jac=True,
                          method="L-BFGS-B", bounds=[(0, None)] * I.n, options=dict(ftol=1e-15, gtol=1e-9, maxiter=5000))
    return r.x, -r.fun


def part2(bench, sel):
    header("PART 2: risk-neutral solution (Algorithm 1)")
    res, rows = {}, []
    for I in (bench, sel):
        Ru, Pu = enum_scen(I)
        Q, A, _ = algorithm1(I, Ru, Pu)
        GE, GV, GWV = analytic_stats(Q, I, Ru, Pu)
        Qg, GEg = global_check(I, Ru, Pu, Q)
        print(f"[{I.name}] active={[int(a) + 1 for a in A]}  Q={Q.round(2)}  E[G]={GE:.2f}  G_V={GV:.2f}  G_WV={GWV:.2f}")
        print(f"   check: direct bounded maximisation gives E[G]={GEg:.2f} (diff {GEg - GE:.2e}); max|Q-Qdirect|={np.abs(Q - Qg).max():.3f}")
        res[I.name] = Q
        rows.append(dict(instance=I.name, active=[int(a) + 1 for a in A],
                         **{f"Q{i + 1}": round(Q[i], 2) for i in range(I.n)},
                         E_G=round(GE, 2), G_V=round(GV, 2), G_WV=round(GWV, 2), sd_G=round(math.sqrt(GV), 2)))
    df = pd.DataFrame(rows); savecsv(df, "part2_results.csv"); print(df.to_string(index=False))
    return res


# ----------------------------------------------------------------------------
# Part 3: Monte Carlo
# ----------------------------------------------------------------------------
def profit_basic(D, X, cost):
    return S * np.minimum(D, X) + R * np.maximum(X - D, 0) - K * np.maximum(D - X, 0) - cost


def sim_basic(Q, I, N, rng):
    D = np.maximum(0.0, rng.normal(I.mu, I.sigma, N))
    RY = np.where(rng.random((N, I.n)) < I.p, I.y, 1.0)
    X = RY @ Q; cost = (RY * I.c) @ Q
    return D, X, profit_basic(D, X, cost)


def ci95(x):
    m, s = x.mean(), x.std(ddof=1); h = 1.96 * s / math.sqrt(len(x)); return m, s, m - h, m + h


def stress_gen(I, N, rng, E=None):
    """Correlated disruptions + stressed demand + spot price (Section 3.3)."""
    if E is None: E = rng.random((N, 3)) < 0.15
    H = E.sum(1)
    gidx = np.zeros(I.n, int)
    for g, idx in enumerate(GROUPS): gidx[idx] = g
    pj = np.where(E[:, gidx], np.minimum(0.9, I.p + 0.25), I.p)
    RY = np.where(rng.random((N, I.n)) < pj, I.y, 1.0)
    D = np.maximum(0.0, (1 + 0.1 * H) * rng.normal(I.mu, I.sigma, N))
    SP = rng.uniform(30 + 15 * H, 60 + 15 * H)
    return dict(D=D, RY=RY, SP=SP, H=H)


def stress_profit(Q, I, sc, detail=False):
    """Realised stress profit: spot purchase B covers the whole shortage iff SP < s+k.
    Profit = s*(min(D,X)+B) + r*(X-D)^+ - k*L - sum c_i R_i Q_i - SP*B."""
    D, SP = sc["D"], sc["SP"]
    X = sc["RY"] @ Q; cost = (sc["RY"] * I.c) @ Q
    short = np.maximum(D - X, 0); B = short * (SP < S + K); L = short - B
    pi = S * (np.minimum(D, X) + B) + R * np.maximum(X - D, 0) - K * L - cost - SP * B
    return (pi, B, L, SP * B) if detail else pi


def part3(I, Q, GE_an, fast):
    header("PART 3: Monte Carlo simulation (selected random instance)")
    out = {}
    # ---- 3.1 choose N via pilot: half-width eps = 0.25% of the analytical mean
    pilot = sim_basic(Q, I, 10000, np.random.default_rng(SEED - 1))[2]
    eps = 0.0025 * abs(GE_an)
    N = int(math.ceil((1.96 * pilot.std(ddof=1) / eps) ** 2 / 10000) * 10000)
    N = max(N, 100000) if not fast else 20000
    print(f"pilot sd={pilot.std(ddof=1):.1f}; target half-width {eps:.1f} -> N = {N}  (seed {SEED})")
    D, X, Pi = sim_basic(Q, I, N, np.random.default_rng(SEED))
    m, s, lo, hi = ci95(Pi); varPi = s ** 2 / N
    print(f"[3.1] mean={m:.2f} sd={s:.2f} CI95=[{lo:.2f},{hi:.2f}]  analytical E[G]={GE_an:.2f}  diff={m - GE_an:.2f} "
          f"(inside CI: {lo <= GE_an <= hi})   skew={stats.skew(Pi):.3f}")
    plt.figure(figsize=(7, 4)); plt.hist(Pi, bins=80, color="steelblue", edgecolor="white")
    plt.axvline(m, color="r", label=f"mean {m:.0f}"); plt.xlabel("profit"); plt.ylabel("frequency")
    plt.title(f"Simulated profit of Q_alg (instance {LSTAR}, N={N})"); plt.legend(); savefig("part3_hist_plain.png")
    # coverage experiment
    M, cover = 100, 0
    for mm in range(M):
        pm = sim_basic(Q, I, N, np.random.default_rng(SEED + 1000 + mm))[2]
        _, _, a, b = ci95(pm); cover += (a <= GE_an <= b)
    print(f"[3.1] CI coverage over {M} runs: {cover / M:.2f}")
    # ---- 3.2.1 control variate  C = D - X, E[C] = mu - sum E[R_i] Q_i
    C = D - X; EC = I.mu - I.ER @ Q
    beta = -np.cov(Pi, C)[0, 1] / np.var(C, ddof=1)
    Pcv = Pi + beta * (C - EC); vcv = Pcv.var(ddof=1) / N
    vr_cv = 100 * (varPi - vcv) / varPi
    print(f"[3.2.1] CV: beta={beta:.4f} corr(Pi,C)={np.corrcoef(Pi, C)[0, 1]:.4f} mean={Pcv.mean():.2f} "
          f"Var(plain)={varPi:.4f} Var(cv)={vcv:.4f}  %VR={vr_cv:.2f}")
    # ---- 3.2.2 antithetic demand (N/2 pairs, independent yields)
    rng = np.random.default_rng(SEED); h = N // 2
    Z = rng.standard_normal(h)
    Dp, Dm = np.maximum(0, I.mu + I.sigma * Z), np.maximum(0, I.mu - I.sigma * Z)
    def one(Dv):
        RY = np.where(rng.random((h, I.n)) < I.p, I.y, 1.0); return profit_basic(Dv, RY @ Q, (RY * I.c) @ Q)
    Pe = 0.5 * (one(Dp) + one(Dm)); van = Pe.var(ddof=1) / h
    vr_an = 100 * (varPi - van) / varPi
    print(f"[3.2.2] AV: mean={Pe.mean():.2f} Var(anti)={van:.4f} %VR={vr_an:.2f}")
    pd.DataFrame([dict(method="plain", mean=m, var_of_mean=varPi, VR_pct=0), dict(method="control variate", mean=Pcv.mean(), var_of_mean=vcv, VR_pct=vr_cv),
                  dict(method="antithetic", mean=Pe.mean(), var_of_mean=van, VR_pct=vr_an)]).round(4).to_csv(os.path.join(OUT, "part3_variance_reduction.csv"), index=False)
    # ---- 3.3 stress test
    Ns = 10000 if not fast else 5000; sseed = SEED + 5
    sc = stress_gen(I, Ns, np.random.default_rng(sseed))
    pi, B, L, spotcost = stress_profit(Q, I, sc, detail=True)
    m2, s2, lo2, hi2 = ci95(pi)
    print(f"[3.3] N={Ns} seed={sseed}: mean={m2:.2f} sd={s2:.2f} CI95=[{lo2:.2f},{hi2:.2f}]")
    print(f"      spot-use freq={np.mean(B > 0):.4f} mean spot purchase={B.mean():.2f} mean spot cost={spotcost.mean():.2f} "
          f"mean lost sales={L.mean():.2f}  P(profit<0)={np.mean(pi < 0):.4f}  5% quantile={np.quantile(pi, .05):.1f}")
    plt.figure(figsize=(7, 4)); plt.hist(pi, bins=80, color="indianred", edgecolor="white")
    plt.xlabel("profit"); plt.ylabel("frequency"); plt.title("Stress-test profit of Q_alg"); savefig("part3_hist_stress.png")
    out.update(stress_mean=m2, stress_sd=s2)
    return out


# ----------------------------------------------------------------------------
# Second-stage LP building blocks (Parts 4 and 5)
#   variables  x = [Q (n), b (N), o (N), L (N)]   (b: spot purchase, o: leftover, L: lost sales)
#   o - L - b - sum_i R_i Q_i = -D        profit_w = s D - (s+k) L + r o - SP b - sum_i c_i R_iw Q_i
# ----------------------------------------------------------------------------
def build_ss(I, sc):
    N, n = len(sc["D"]), I.n; RY, D, SP = sc["RY"], sc["D"], sc["SP"]
    Id = sparse.identity(N, format="csr")
    Aeq = sparse.hstack([sparse.csr_matrix(-RY), -Id, Id, -Id], format="csr")
    PI = sparse.hstack([sparse.csr_matrix(-(RY * I.c)), -sparse.diags(SP), R * Id, -(S + K) * Id], format="csr")
    return dict(N=N, n=n, Aeq=Aeq, beq=-D, PI=PI, const=S * D)


def solve_saa(I, sc, Qcap=None):
    """Risk-neutral two-stage SAA LP with spot-market recourse (maximise mean profit)."""
    m = build_ss(I, sc); N, n = m["N"], m["n"]
    cvec = -np.asarray(m["PI"].mean(axis=0)).ravel()
    A_ub = b_ub = None
    if Qcap is not None:
        A_ub = np.r_[np.ones(n), np.zeros(3 * N)][None, :]; b_ub = [Qcap]
    r = linprog(cvec, A_ub=A_ub, b_ub=b_ub, A_eq=m["Aeq"], b_eq=m["beq"], bounds=(0, None), method="highs")
    assert r.status == 0, r.message
    return r.x[:n], -r.fun + m["const"].mean()


def part4(I, Qalg, fast):
    header("PART 4: SAA with spot-market recourse")
    settings = [(50, 5), (1000, 5), (2000, 5)] if not fast else [(50, 3), (200, 3), (400, 3)]
    Nprime = 10000 if not fast else 3000
    oos = stress_gen(I, Nprime, np.random.default_rng(SEED + 100))            # common out-of-sample set (CRN)
    zq = norm.ppf(0.95); rows, cands = [], {}
    for (N, M) in settings:
        t0 = time.time(); Qs, vs = [], []
        for mm in range(M):
            sc = stress_gen(I, N, np.random.default_rng(SEED + 10000 * N + mm))
            q, v = solve_saa(I, sc); Qs.append(q); vs.append(v)
        rt = time.time() - t0
        Qs, vs = np.array(Qs), np.array(vs)
        oosp = [stress_profit(q, I, oos) for q in Qs]
        means = np.array([p.mean() for p in oosp]); ses = np.array([p.std(ddof=1) / math.sqrt(Nprime) for p in oosp])
        b = int(np.argmax(means))
        vbar = vs.mean(); sv = vs.std(ddof=1) / math.sqrt(M); tq = stats.t.ppf(0.95, M - 1)
        U_opt = vbar + tq * sv                     # statistical UPPER bound on v* (max problem)
        L_cand = means[b] - zq * ses[b]            # statistical LOWER bound on g(x_hat) (and hence on v*)
        gap = U_opt - L_cand
        cands[N] = Qs[b]
        rows.append(dict(N=N, M=M, best_batch=b + 1, saa_obj_best=vs[b], saa_obj_mean=vbar, oos_mean=means[b], oos_se=ses[b],
                         upper_bound_vstar=U_opt, lower_bound_g=L_cand, gap_95=gap, point_gap=vbar - means[b],
                         Q_spread=np.mean(Qs.std(axis=0)), runtime_s=rt, **{f"Q{i + 1}": Qs[b][i] for i in range(I.n)}))
        print(f"(N={N},M={M}) best batch {b + 1}: Q={Qs[b].round(1)}\n   SAA obj={vs[b]:.1f} (mean {vbar:.1f}), OOS={means[b]:.1f}+-{ses[b]:.1f}, "
              f"U(v*)={U_opt:.1f}, L(g)={L_cand:.1f}, gap<= {gap:.1f}, mean sd of Q over batches={rows[-1]['Q_spread']:.2f}, time {rt:.1f}s")
    tab = pd.DataFrame(rows); savecsv(tab.round(3), "part4_saa_analysis.csv"); print(tab.round(2).to_string(index=False))
    # ---- 4.4 select Q_SAA (best OOS performance; ties broken by smaller gap / runtime) and compare with Q_alg via CRN
    best = tab.sort_values(["oos_mean"], ascending=False).iloc[0]; Nsel = int(best["N"]); Qsaa = cands[Nsel]
    print(f"\n[4.4] selected Q_SAA from N={Nsel}: {Qsaa.round(2)}")
    Ne = 100000 if not fast else 10000
    ev = stress_gen(I, Ne, np.random.default_rng(SEED + 200))
    pa, ps = stress_profit(Qalg, I, ev), stress_profit(Qsaa, I, ev); d = ps - pa
    md, sd, lo, hi = ci95(d)
    print(f"   Q_alg: mean={pa.mean():.2f} sd={pa.std(ddof=1):.2f}   Q_SAA: mean={ps.mean():.2f} sd={ps.std(ddof=1):.2f}")
    print(f"   paired diff mean={md:.2f} CI95=[{lo:.2f},{hi:.2f}] -> SAA {'significantly better' if lo > 0 else 'not significantly better'}")
    pd.DataFrame([dict(sol="Q_alg", mean=pa.mean(), sd=pa.std(ddof=1), **{f"Q{i + 1}": Qalg[i] for i in range(I.n)}),
                  dict(sol="Q_SAA", mean=ps.mean(), sd=ps.std(ddof=1), **{f"Q{i + 1}": Qsaa[i] for i in range(I.n)})]).round(3).to_csv(os.path.join(OUT, "part4_comparison.csv"), index=False)
    # ---- 4.5 VSS : expected-value problem (expected demand, expected yields, expected spot price)
    Eh = 3 * 0.15
    ep = I.p + 0.15 * (np.minimum(0.9, I.p + 0.25) - I.p)
    ev_sc = dict(D=np.array([I.mu * (1 + 0.1 * Eh)]), RY=(1 - (1 - I.y) * ep)[None, :], SP=np.array([45 + 15 * Eh]))
    Qev, _ = solve_saa(I, ev_sc)
    zEEV, zSAA = stress_profit(Qev, I, ev).mean(), ps.mean()
    print(f"[4.5] Q_EV={Qev.round(2)}  zEEV={zEEV:.2f}  zSAA={zSAA:.2f}  VSS={zSAA - zEEV:.2f}")
    # ---- 4.6 EVPI on group disruptions
    Mb, Nb, Np2 = (3, 1000, 10000) if not fast else (2, 200, 2000)
    zstar = 0.0; rows6 = []
    for e in itertools.product([0, 1], repeat=3):
        H = sum(e); Pe = 0.15 ** H * 0.85 ** (3 - H)
        Ee = np.tile(np.array(e, bool), (Np2, 1))
        osc = stress_gen(I, Np2, np.random.default_rng(SEED + 300 + int("".join(map(str, e)), 2)), E=Ee)
        bestz, bestq = -np.inf, None
        for mm in range(Mb):
            sc = stress_gen(I, Nb, np.random.default_rng(SEED + 50000 + 100 * int("".join(map(str, e)), 2) + mm), E=np.tile(np.array(e, bool), (Nb, 1)))
            q, _ = solve_saa(I, sc); z = stress_profit(q, I, osc).mean()
            if z > bestz: bestz, bestq = z, q
        zstar += Pe * bestz
        rows6.append(dict(state=e, prob=Pe, z_star=bestz, **{f"Q{i + 1}": bestq[i] for i in range(I.n)}))
    print(pd.DataFrame(rows6).round(2).to_string(index=False))
    print(f"[4.6] z_PI-G={zstar:.2f}  z_SAA={zSAA:.2f}  EVPPI_G={zstar - zSAA:.2f}")
    savecsv(pd.DataFrame(rows6).round(3), "part4_evppi_states.csv")
    return dict(Qsaa=Qsaa, Qev=Qev)


# ----------------------------------------------------------------------------
# Part 5: risk-averse models (fixed scenario set N=1000)
# ----------------------------------------------------------------------------
def risk_stats(pi, alpha):
    N = len(pi); k = max(1, int(math.ceil(alpha * N - 1e-9))); srt = np.sort(pi)
    return dict(E=pi.mean(), sd=pi.std(ddof=1), VaR=srt[k - 1], CVaR=srt[:k].mean())


def solve_risk(I, sc, kind, alpha=1.0, tau=None, Qcap=1500.0, tl=60.0):
    m = build_ss(I, sc); N, n = m["N"], m["n"]; PI, const = m["PI"], m["const"]; nx = n + 3 * N
    qrow = np.r_[np.ones(n), np.zeros(nx - n)]
    cx = -np.asarray(PI.mean(axis=0)).ravel()
    if kind == "RN":
        r = linprog(cx, A_ub=qrow[None, :], b_ub=[Qcap], A_eq=m["Aeq"], b_eq=m["beq"], bounds=(0, None), method="highs")
        return r.x[:n], r.status == 0, 0.0
    if kind == "CVaR":                        # max t - 1/(alpha N) sum eta ;  eta_w >= t - Pi_w
        Id = sparse.identity(N, format="csr")
        c = np.r_[np.zeros(nx), -1.0, np.full(N, 1.0 / (alpha * N))]
        A_ub = sparse.vstack([sparse.hstack([-PI, sparse.csr_matrix(np.ones((N, 1))), -Id]),
                              sparse.csr_matrix(np.r_[qrow, 0, np.zeros(N)][None, :])], format="csr")
        b_ub = np.r_[const, Qcap]
        A_eq = sparse.hstack([m["Aeq"], sparse.csr_matrix((N, 1 + N))], format="csr")
        lb = np.r_[np.zeros(nx), -np.inf, np.zeros(N)]
        r = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=m["beq"], bounds=list(zip(lb, [None] * len(lb))), method="highs")
        return (r.x[:n] if r.status == 0 else None), r.status == 0, 0.0
    if kind == "VaR":                         # max E[Pi] s.t. P(Pi < tau) <= alpha  (binary phi_w = 1 if scenario violates)
        cmax = I.c.max(); lbp = -cmax * Qcap - (K + sc["SP"]) * sc["D"]
        Mbig = np.maximum(tau - lbp, 0.0)
        c = np.r_[cx, np.zeros(N)]
        A_ub = sparse.vstack([sparse.hstack([-PI, -sparse.diags(Mbig)]),
                              sparse.csr_matrix(np.r_[np.zeros(nx), np.ones(N)][None, :]),
                              sparse.csr_matrix(np.r_[qrow, np.zeros(N)][None, :])], format="csr")
        b_ub = np.r_[const - tau, alpha * N, Qcap]
        A_eq = sparse.hstack([m["Aeq"], sparse.csr_matrix((N, N))], format="csr")
        integ = np.r_[np.zeros(nx), np.ones(N)]
        cons = [LinearConstraint(A_ub, -np.inf, b_ub), LinearConstraint(A_eq, m["beq"], m["beq"])]
        r = milp(c, constraints=cons, integrality=integ, bounds=Bounds(np.zeros(nx + N), np.r_[np.full(nx, np.inf), np.ones(N)]),
                 options=dict(time_limit=tl, mip_rel_gap=1e-3, disp=False))
        if r.x is None: return None, False, np.nan
        return r.x[:n], True, getattr(r, "mip_gap", np.nan)
    raise ValueError(kind)


def part5(I, fast):
    header("PART 5: risk-averse models (CVaR and VaR-constrained)")
    N = 1000 if not fast else 200; tl = 60.0 if not fast else 10.0
    sc = stress_gen(I, N, np.random.default_rng(SEED + 400))
    alphas = [0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]
    def row(tag, a, Q, extra=None):
        pi = stress_profit(Q, I, sc); st = risk_stats(pi, a)
        act = [i + 1 for i in range(I.n) if Q[i] > 0.5]
        d = dict(model=tag, alpha=a, active=act, **{f"Q{i + 1}": round(Q[i], 1) for i in range(I.n)},
                 total_Q=round(Q.sum(), 1), E_profit=round(st["E"], 1), sd=round(st["sd"], 1), VaR=round(st["VaR"], 1), CVaR=round(st["CVaR"], 1))
        if extra: d.update(extra)
        return d
    Qrn, _, _ = solve_risk(I, sc, "RN"); rows_rn = [row("risk-neutral", a, Qrn) for a in alphas]
    print("Risk-neutral reference (stats depend on alpha only through VaR/CVaR):"); print(pd.DataFrame(rows_rn).to_string(index=False))
    savecsv(pd.DataFrame(rows_rn), "part5_risk_neutral.csv")
    # 5.1 CVaR
    rc, Qc = [], {}
    for a in alphas:
        t0 = time.time(); q, ok, _ = solve_risk(I, sc, "CVaR", alpha=a); Qc[a] = q
        rc.append(row("CVaR", a, q, dict(time_s=round(time.time() - t0, 2))))
    dfc = pd.DataFrame(rc); savecsv(dfc, "part5_cvar.csv"); print("\nCVaR models:"); print(dfc.to_string(index=False))
    # 5.2 VaR-constrained: tau_alpha = VaR_RN + 0.5 (VaR_CVaR - VaR_RN)  (>= VaR_RN, feasible by the CVaR solution)
    rv = []
    for a in alphas:
        if a >= 1.0:
            rv.append(row("VaR-constr.", a, Qrn, dict(tau=-np.inf, mip_gap=0.0, time_s=0.0, status='no constraint (= RN)'))); continue
        vr_rn = risk_stats(stress_profit(Qrn, I, sc), a)["VaR"]; vr_c = risk_stats(stress_profit(Qc[a], I, sc), a)["VaR"]
        tau = vr_rn + 0.5 * max(vr_c - vr_rn, 0.0)
        t0 = time.time(); q, ok, gap = solve_risk(I, sc, "VaR", alpha=a, tau=tau, tl=tl)
        if q is None: q = Qc[a]; flag = "MILP found no incumbent in time limit -> CVaR solution (feasible by construction) reported"
        else: flag = "ok"
        rv.append(row("VaR-constr.", a, q, dict(tau=round(tau, 1), mip_gap=round(float(gap), 4), time_s=round(time.time() - t0, 1), status=flag)))
    dfv = pd.DataFrame(rv); savecsv(dfv, "part5_var_constrained.csv"); print("\nVaR-constrained models:"); print(dfv.to_string(index=False))
    # frontier plot
    plt.figure(figsize=(6, 4))
    plt.plot(dfc["E_profit"], dfc["CVaR"], "o-", label="CVaR model"); plt.plot(dfv["E_profit"], dfv["CVaR"], "s--", label="VaR-constr.")
    plt.xlabel("expected profit"); plt.ylabel("CVaR$_{1-\\alpha}$ of own alpha"); plt.legend(); plt.title("Expected profit vs tail"); savefig("part5_frontier.png")


# ----------------------------------------------------------------------------
# Part 6: Distributionally robust optimisation (no spot market, original model)
# ----------------------------------------------------------------------------
def solve_scen_lp(I, Rs, probs, Ds):
    """max sum_m prob_m [ s w_m + r e_m - k l_m - sum_i c_i R_im Q_i ];  w+e = X_m ; w+l = D_m."""
    m, n = Rs.shape; Id = sparse.identity(m, format="csr"); Z = sparse.csr_matrix((m, m))
    c = np.r_[(probs[:, None] * Rs * I.c).sum(0), -S * probs, -R * probs, K * probs]
    Aeq = sparse.vstack([sparse.hstack([sparse.csr_matrix(-Rs), Id, Id, Z]), sparse.hstack([sparse.csr_matrix((m, n)), Id, Z, Id])], format="csr")
    beq = np.r_[np.zeros(m), Ds]
    t0 = time.time(); r = linprog(c, A_eq=Aeq, b_eq=beq, bounds=(0, None), method="highs"); rt = time.time() - t0
    assert r.status == 0, r.message
    return r.x[:n], -r.fun, rt


def check_convexity(I, Ru, ntest=200):
    """Numerical check that D -> -Pi_u(Q;D) has non-negative second differences."""
    rng = np.random.default_rng(0); worst = np.inf
    for _ in range(ntest):
        Q = rng.uniform(0, 300, I.n); u = rng.integers(len(Ru)); X = Ru[u] @ Q; cost = (Ru[u] * I.c) @ Q
        Dg = np.linspace(0, 2 * I.mu, 401)
        f = -profit_basic(Dg, X, cost); worst = min(worst, np.diff(f, 2).min())
    return worst


def scarf_socp(I, Ru, Pu):
    """max sum_u p_u [ r X_u - c.R_u Q + (s-r) mu - (s-r+k) (mu - X_u + t_u)/2 ],  t_u >= sqrt((X_u-mu)^2+sigma^2)."""
    t0 = time.time()
    try:
        import cvxpy as cp
        Q = cp.Variable(I.n, nonneg=True); t = cp.Variable(len(Pu)); X = Ru @ Q
        obj = Pu @ (R * X - (Ru * I.c) @ Q + (S - R) * I.mu - (S - R + K) * (I.mu - X + t) / 2)
        cons = [cp.norm(cp.vstack([X - I.mu, I.sigma * np.ones(len(Pu))]), 2, axis=0) <= t]
        pr = cp.Problem(cp.Maximize(obj), cons); pr.solve()
        return Q.value, pr.value, time.time() - t0, "cvxpy"
    except ImportError:
        def f(q):
            X = Ru @ q; tt = np.sqrt((X - I.mu) ** 2 + I.sigma ** 2)
            val = Pu @ (R * X - (Ru * I.c) @ q + (S - R) * I.mu - (S - R + K) * (I.mu - X + tt) / 2)
            dt = (X - I.mu) / tt
            g = (Pu[:, None] * (R * Ru - Ru * I.c - (S - R + K) * (-Ru + dt[:, None] * Ru) / 2)).sum(0)
            return -val, -g
        r = optimize.minimize(f, np.full(I.n, 20.), jac=True, method="L-BFGS-B", bounds=[(0, None)] * I.n,
                              options=dict(ftol=1e-15, gtol=1e-10, maxiter=5000))
        return r.x, -r.fun, time.time() - t0, "scipy (smooth concave form of the SOCP)"


def part6(I):
    header("PART 6: Distributionally robust optimisation")
    Ru, Pu = enum_scen(I); mu, sg = I.mu, I.sigma
    # 6.1 deterministic demand
    Q1, v1, t1 = solve_scen_lp(I, Ru, Pu, np.full(len(Pu), mu))
    print(f"[6.1] Deterministic-demand SLP: Q={Q1.round(2)}  profit={v1:.2f}  time={t1:.2f}s")
    # 6.2 mean-MAD
    d = math.sqrt(2 / math.pi) * sg; b = 2 * mu; sup = np.array([0.0, mu, b])
    pa, pb = d / (2 * mu), d / (2 * (b - mu)); q3 = np.array([pa, 1 - pa - pb, pb])
    print(f"[6.2] convexity check (min 2nd difference of -Pi in D, should be >=0): {check_convexity(I, Ru):.2e}")
    print(f"      d={d:.2f}  worst-case demand law: P(0)={q3[0]:.4f}, P(mu={mu:.1f})={q3[1]:.4f}, P(2mu={b:.1f})={q3[2]:.4f}  (feasible: {q3[1] >= 0})")
    U = len(Pu)
    Rs = np.repeat(Ru, 3, axis=0); probs = (Pu[:, None] * q3[None, :]).ravel(); Ds = np.tile(sup, U)
    Q2, v2, t2 = solve_scen_lp(I, Rs, probs, Ds)
    print(f"      DRO-LP: Q={Q2.round(2)}  worst-case expected profit={v2:.2f}  time={t2:.2f}s")
    # 6.3 mean-variance (scenario-wise Scarf)
    Q3, v3, t3, solver = scarf_socp(I, Ru, Pu)
    print(f"[6.3] Scarf SOCP ({solver}): Q={Q3.round(2)}  conservative worst-case profit bound={v3:.2f}  time={t3:.2f}s")
    # 6.4 comparison  (+ analytical expected profit under N(mu,sigma^2) for reference)
    rows = []
    for name, Q, val, meas, t in [("Deterministic SLP", Q1, v1, "Expected profit (D=mu)", t1), ("Mean-MAD DRO-LP", Q2, v2, "Worst-case profit", t2),
                                  ("Mean-variance Scarf SOCP", Q3, v3, "Conservative worst-case bound", t3)]:
        GE = analytic_stats(Q, I, Ru, Pu)[0]
        rows.append(dict(model=name, active=[i + 1 for i in range(I.n) if Q[i] > 0.5], total_Q=Q.sum(),
                         **{f"Q{i + 1}": Q[i] for i in range(I.n)}, measure=meas, value=val, E_profit_under_Normal=GE, runtime_s=t))
    df = pd.DataFrame(rows).round(2); savecsv(df, "part6_comparison.csv"); print("\n", df.to_string(index=False))


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--part", default="all"); ap.add_argument("--fast", action="store_true")
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    inst = gen_instances(GROUP); sel = inst[LSTAR - 1]; bench = benchmark()
    parts = {"1", "2", "3", "4", "5", "6"} if a.part == "all" else {a.part}
    if "1" in parts: part1(inst)
    Qs = part2(bench, sel) if parts & {"2", "3", "4"} else None
    Qalg = Qs[sel.name] if Qs else None
    if "3" in parts:
        Ru, Pu = enum_scen(sel); part3(sel, Qalg, analytic_stats(Qalg, sel, Ru, Pu)[0], a.fast)
    if "4" in parts: part4(sel, Qalg, a.fast)
    if "5" in parts: part5(sel, a.fast)
    if "6" in parts: part6(sel)


if __name__ == "__main__":
    main()
