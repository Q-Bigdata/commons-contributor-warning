"""Minimal generative models of the commons that differ only in the ROUTE to collapse.

S1 forcing : the group's mean return fraction declines slowly through the replacement threshold 1/(1+r)
             (classical bifurcation-approach scenario; critical slowing down expected).
S2 neutral : mean return fluctuates around the replacement threshold with persistent group shocks
             (self-organised marginal stability; collapse is a diffusive first passage).
S3 allee   : return fraction increases with pool size (positive feedback), giving a stable full-pool
             state and an unstable threshold; collapse is a noise-induced escape without parameter change.
All scenarios use equal allocation among four players and the exact pool law of the human game.
Output: data_toy/rounds.csv.gz in the same schema as the human data (source='toy').
"""
import os
import numpy as np
import pandas as pd

rng = np.random.default_rng(11)
NG, T, P = 3000, 40, 4
RSTAR = 1 / 1.4
rows = []


def play(scen, g):
    R = 200.0
    m_prev = 0.0
    if scen == "S1-forcing":
        m0 = rng.uniform(0.74, 0.84); kappa = rng.uniform(0.0, 0.008)
    if scen == "S2-neutral":
        u = rng.normal(0.0, 0.015); eta = 0.0
    for t in range(T):
        if scen == "S1-forcing":
            m = m0 - kappa * t + rng.normal(0, 0.035)
        elif scen == "S2-neutral":
            eta = 0.6 * eta + rng.normal(0, 0.09)
            m = RSTAR + u + eta
        else:  # S3-allee
            m = RSTAR + 0.035 * (np.log(max(R, 1)) - np.log(60.0)) + rng.normal(0, 0.07)
            m = min(m, 0.80)
        e = np.full(P, R / P)
        f = np.clip(m + rng.normal(0, 0.30, P), 0, 1)
        c = np.round(f * e)
        nxt = min(200.0, R - e.sum() + 1.4 * c.sum())
        rows.append(dict(source="toy", game=f"{scen}_{g}", group=f"{scen}_{g}", mech_label=scen, round=t,
                         pool=R, next_pool=nxt, **{f"e{i}": e[i] for i in range(P)}, **{f"c{i}": c[i] for i in range(P)},
                         mech=scen, exp=0, bc=0))
        R = nxt
        if R < 1:
            # keep schema identical to the human data: rounds after depletion carry an empty pool
            for tt in range(t + 1, T):
                rows.append(dict(source="toy", game=f"{scen}_{g}", group=f"{scen}_{g}", mech_label=scen, round=tt,
                                 pool=0.01, next_pool=0.01, **{f"e{i}": 0.0 for i in range(P)},
                                 **{f"c{i}": 0.0 for i in range(P)}, mech=scen, exp=0, bc=0))
            break


for scen in ["S1-forcing", "S2-neutral", "S3-allee"]:
    for g in range(NG):
        play(scen, g)
df = pd.DataFrame(rows)
os.makedirs("data_toy", exist_ok=True)
df.to_csv("data_toy/rounds.csv.gz", index=False, compression="gzip")
dep = df.groupby("game").next_pool.min().lt(1).groupby(df.groupby("game").mech.first()).mean()
print(dep)
