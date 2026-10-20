import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

np.random.seed(3003)

suppliers = pd.DataFrame({
    "Supplier":[1,2,3,4,5,6,7,8,9,10],
    "p":[0.05,0.25,0.15,0.10,0.20,0.08,0.12,0.04,0.17,0.22],
    "y":[0.80,0.40,0.50,0.70,0.40,0.60,0.75,0.90,0.65,0.85],
    "c":[40,25,35,30,21,55,45,37,52,42]
})

suppliers = suppliers.astype({
    "p": float,
    "y": float,
    "c": float
})

s = 45
r = 10
k = 15

seed = 3003

instance = suppliers.copy()

import pandas as pd
import numpy as np

cols = [
    "Supplier",
    "p",
    "y",
    "c",
    "ceff",
    "CR",
    "ER",
    "VarR"
]

data = np.array([
 [1,0.05,0.71,38.6,38.0,0.43,0.99,0.00],
 [2,0.27,0.36,24.8,20.5,0.70,0.83,0.08],
 [3,0.11,0.58,35.0,33.3,0.50,0.95,0.02],
 [4,0.07,0.72,31.4,30.8,0.57,0.98,0.00],
 [5,0.20,0.43,20.1,17.8,0.80,0.88,0.05],
 [6,0.10,0.67,50.1,48.4,0.20,0.97,0.01],
 [7,0.11,0.67,47.6,45.8,0.25,0.96,0.01],
 [8,0.05,0.87,39.4,39.2,0.41,0.99,0.00],
 [9,0.16,0.71,51.6,49.1,0.17,0.95,0.01],
 [10,0.19,0.91,43.6,42.9,0.33,0.98,0.00]
])

instance = pd.DataFrame(data, columns=cols)

mu = np.random.uniform(350,450)

cv = np.random.uniform(0.20,0.29)

sigma = mu * cv

instance["ER"] = 1 - instance["p"]*(1-instance["y"])

instance["VarR"] = (
    instance["p"] *
    (1-instance["p"]) *
    (1-instance["y"])**2
)

instance["ceff"] = instance["c"] * instance["ER"]

instance["CR"] = (
    (s+k-instance["c"])
    /
    (s+k-r)
)

print(instance.round(4))
print("\nmu =",round(mu,2))
print("sigma =",round(sigma,2))
print("cv =",round(cv,4))

def profit(D, X, procurement_cost):

    sales = min(D,X)

    leftover = max(0,X-D)

    shortage = max(0,D-X)

    return (
        s*sales
        + r*leftover
        - k*shortage
        - procurement_cost
    )

def realized_supply(Q, p, y):

    yields = np.where(
        np.random.rand(len(Q)) < p,
        y,
        1
    )

    X = np.sum(yields * Q)

    procurement = np.sum(
        yields * Q * instance["c"].values
    )

    return X, procurement





Q_alg = np.array([
    0,
    21.5,
    0,
    0,
    516.1,
    0,
    0,
    0,
    0,
    0
])


N = 100000

profits = []

for j in range(N):

    D = max(
        0,
        np.random.normal(mu,sigma)
    )

    X, cost = realized_supply(
        Q_alg,
        instance["p"].values,
        instance["y"].values
    )

    profits.append(
        profit(D,X,cost)
    )

profits = np.array(profits)

mean_profit = np.mean(profits)

std_profit = np.std(
    profits,
    ddof=1
)

CI_low = (
    mean_profit
    - 1.96*std_profit/np.sqrt(N)
)

CI_high = (
    mean_profit
    + 1.96*std_profit/np.sqrt(N)
)

print("Mean =",mean_profit)
print("Std =",std_profit)
print("95% CI =",CI_low,CI_high)

plt.figure(figsize=(8,5))

plt.hist(
    profits,
    bins=50,
    color="skyblue",
    edgecolor="black"
)

plt.title("Profit Distribution")

plt.xlabel("Profit")

plt.ylabel("Frequency")

plt.show()

M = 100

coverage = 0

true_mean = mean_profit

for m in range(M):

    np.random.seed(1000+m)

    sample = []

    for j in range(10000):

        D = max(
            0,
            np.random.normal(mu,sigma)
        )

        X,cost = realized_supply(
            Q_alg,
            instance["p"].values,
            instance["y"].values
        )

        sample.append(
            profit(D,X,cost)
        )

    sample = np.array(sample)

    xbar = np.mean(sample)

    sdev = np.std(sample,ddof=1)

    lower = (
        xbar
        - 1.96*sdev/np.sqrt(len(sample))
    )

    upper = (
        xbar
        + 1.96*sdev/np.sqrt(len(sample))
    )

    if lower <= true_mean <= upper:
        coverage += 1

coverage_rate = coverage/M

print("Coverage =",coverage_rate)

profits = []
controls = []

EC = (
    mu
    - np.sum(
        instance["ER"].values * Q_alg
    )
)

for j in range(N):

    D = max(
        0,
        np.random.normal(mu,sigma)
    )

    X,cost = realized_supply(
        Q_alg,
        instance["p"].values,
        instance["y"].values
    )

    Pi = profit(D,X,cost)

    C = D - X

    profits.append(Pi)

    controls.append(C)


profits = np.array(profits)
controls = np.array(controls)

beta = -np.cov(
    profits,
    controls
)[0,1] / np.var(controls)

profits_cv = (
    profits
    + beta*(controls-EC)
)

var_mc = np.var(profits)/N

var_cv = np.var(profits_cv)/N

VR_cv = (
    100
    *
    (var_mc-var_cv)
    /
    var_mc
)

print("Variance reduction =",VR_cv,"%")

pair_profits = []

for j in range(N//2):

    z = np.random.normal()

    D_plus = max(
        0,
        mu + sigma*z
    )

    D_minus = max(
        0,
        mu - sigma*z
    )

    X1,c1 = realized_supply(
        Q_alg,
        instance["p"].values,
        instance["y"].values
    )

    X2,c2 = realized_supply(
        Q_alg,
        instance["p"].values,
        instance["y"].values
    )

    p1 = profit(
        D_plus,
        X1,
        c1
    )

    p2 = profit(
        D_minus,
        X2,
        c2
    )

    pair_profits.append(
        0.5*(p1+p2)
    )

pair_profits = np.array(pair_profits)

anti_mean = np.mean(pair_profits)

anti_var = (
    np.var(pair_profits)
    /(N/2)
)

VR_anti = (
    100
    *
    (var_mc-anti_var)
    /
    var_mc
)

print("Antithetic VR =",VR_anti,"%")

G1 = [0,1,2]
G2 = [3,4,5]
G3 = [6,7,8,9]

N_stress = 10000

stress_profit = []

spot_used = []
spot_cost = []
spot_qty = []
lost_sales = []

for rep in range(N_stress):

    E1 = np.random.binomial(1,0.15)
    E2 = np.random.binomial(1,0.15)
    E3 = np.random.binomial(1,0.15)

    H = E1+E2+E3

    p_mod = instance["p"].values.copy()

    for i in G1:
        if E1:
            p_mod[i] = min(
                0.90,
                p_mod[i]+0.25
            )

    for i in G2:
        if E2:
            p_mod[i] = min(
                0.90,
                p_mod[i]+0.25
            )

    for i in G3:
        if E3:
            p_mod[i] = min(
                0.90,
                p_mod[i]+0.25
            )

    yields = np.where(
        np.random.rand(10) < p_mod,
        instance["y"].values,
        1
    )

    X = np.sum(yields*Q_alg)

    procurement = np.sum(
        yields
        *Q_alg
        *instance["c"].values
    )

    D0 = np.random.normal(
        mu,
        sigma
    )

    D_stress = max(
        0,
        (1+0.1*H)*D0
    )

    SP = np.random.uniform(
        30+15*H,
        60+15*H
    )

    shortage = max(
        0,
        D_stress-X
    )

    Bspot = (
        shortage
        if SP < s+k
        else 0
    )

    lost = shortage-Bspot

    revenue = (
        s * min(
            D_stress,
            X+Bspot
        )
    )

    salvage = (
        r *
        max(
            0,
            X+Bspot-D_stress
        )
    )

    penalty = k*lost

    spot_purchase_cost = SP*Bspot

    pi = (
        revenue
        + salvage
        - penalty
        - procurement
        - spot_purchase_cost
    )

    stress_profit.append(pi)

    spot_used.append(Bspot>0)
    spot_qty.append(Bspot)
    spot_cost.append(spot_purchase_cost)
    lost_sales.append(lost)

stress_profit = np.array(stress_profit)

mean_stress = np.mean(stress_profit)

std_stress = np.std(
    stress_profit,
    ddof=1
)

ci_stress = (
    mean_stress
    - 1.96*std_stress/np.sqrt(N_stress),
    mean_stress
    + 1.96*std_stress/np.sqrt(N_stress)
)

print(mean_stress)
print(std_stress)
print(ci_stress)

print(
    "Spot market frequency:",
    np.mean(spot_used)
)

print(
    "Mean spot quantity:",
    np.mean(spot_qty)
)

print(
    "Mean spot cost:",
    np.mean(spot_cost)
)

print(
    "Mean lost sales:",
    np.mean(lost_sales)
)






from itertools import product
import numpy as np

def generate_scenarios(instance):

    n = len(instance)

    scenarios = []

    for b in product([0,1], repeat=n):

        prob = 1.0

        for i in range(n):

            p = instance.loc[i,"p"]

            if b[i] == 1:
                prob *= p
            else:
                prob *= (1-p)

        scenarios.append({
            "b": np.array(b),
            "prob": prob
        })

    return scenarios

scenarios = generate_scenarios(instance)

print(len(scenarios))
# 1024


print(np.sqrt(var_mc))
print(np.sqrt(var_cv))