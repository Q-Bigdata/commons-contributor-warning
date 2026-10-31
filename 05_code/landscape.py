"""Empirical stability landscapes (Kramers-Moyal drift and diffusion of log pool), a state-only Markov
first-passage model of collapse, point-of-no-return curves and institution Peclet numbers.

usage: python landscape.py --out DIR
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import ews_lib as L

p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument("--nsim", type=int, default=20000)
p.add_argument("--boot", type=int, default=500)
p.add_argument("--seed", type=int, default=7)
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
rng = np.random.default_rng(a.seed)

R = pd.read_csv("data/rounds.csv.gz")
D = L.prepare(R)
D = D[D.pool >= 1].copy()
D["x"] = np.log(D.pool)
D["dx"] = np.log(np.maximum(D.next_pool, 0.01)) - D.x
D["cond"] = D.source.map({"human": "H", "clone": "C"}).fillna("T") + ":" + D.mech
D.loc[D.source == "clone", "cond"] = D.loc[D.source == "clone", "cond"] + ":BC" + D.loc[D.source == "clone", "bc"].astype(int).astype(str)
EDGES = np.log(np.array([1, 5, 10, 25, 50, 75, 100, 125, 150, 175, 190, 199.5, 200.01]))
D["bin"] = np.clip(np.digitize(D.x, EDGES) - 1, 0, len(EDGES) - 2)
PLATEAU = (np.log(25), np.log(190))

# ------------------------------------------------ drift / diffusion per condition and bin
km = D.groupby(["cond", "bin"]).dx.agg(mu="mean", var="var", n="size").reset_index()
km["x_mid"] = [(EDGES[b] + EDGES[b + 1]) / 2 for b in km.bin]
km.to_csv(f"{a.out}/kramers_moyal.csv", index=False)

# ------------------------------------------------ Markov first-passage model (state-only, nonparametric)
def simulate(sub, horizon_sampler, nsim):
    inc = {b: sub.loc[sub.bin == b, "dx"].to_numpy() for b in range(len(EDGES) - 1)}
    # borrow increments from nearest populated bin if empty
    pop = [b for b in inc if len(inc[b]) >= 5]
    for b in inc:
        if len(inc[b]) < 5:
            nb = min(pop, key=lambda q: abs(q - b))
            inc[b] = inc[nb]
    H = horizon_sampler(nsim)
    x = np.full(nsim, np.log(200.0))
    alive = np.ones(nsim, bool)
    collapsed = np.zeros(nsim, bool)
    for t in range(int(H.max())):
        act = alive & (t < H)
        if not act.any():
            break
        b = np.clip(np.digitize(x, EDGES) - 1, 0, len(EDGES) - 2)
        for bb in np.unique(b[act]):
            m = act & (b == bb)
            x[m] = np.minimum(np.log(200.0), x[m] + rng.choice(inc[bb], m.sum()))
        newc = act & (x < 0)
        collapsed |= newc
        alive &= ~newc
    return collapsed.mean()


games = D.groupby("game").agg(source=("source", "first"), mech=("mech", "first"), exp=("exp", "first"),
                              cond=("cond", "first"), tau=("tau", "first"), T=("T", "first"), group=("group", "first"))
games["collapsed"] = games.tau.notna()
exp4_T = games.loc[(games.source == "human") & (games.exp == 4), "T"].to_numpy()
rows = []
for (src, exp, mech), gg in games.groupby(["source", "exp", "mech"]):
    cond = gg.cond.iloc[0]
    sub = D[D.cond == cond]
    if src == "human" and exp == 4:
        hs = lambda n: rng.choice(exp4_T, n)
    else:
        hs = lambda n: np.full(n, 40)
    # fit landscape on the SAME mechanism pooled over experiments (humans) -> out-of-condition check below
    pred = simulate(sub, hs, a.nsim)
    # leave-condition-out: fit increments on other experiments of the same mechanism (humans only)
    other = D[(D.source == src) & (D.mech == mech) & (D.exp != exp)]
    pred_loo = simulate(other, hs, a.nsim) if (src == "human" and len(other) > 500) else np.nan
    # bootstrap CI of observed collapse fraction by group
    obs = gg.collapsed.to_numpy()
    grp = gg.group.to_numpy(); ug = np.unique(grp)
    bs = []
    for _ in range(a.boot):
        pick = rng.choice(ug, len(ug))
        bs.append(np.concatenate([obs[grp == q] for q in pick]).mean())
    rows.append(dict(source=src, exp=exp, mech=mech, n_games=len(gg), obs=obs.mean(),
                     obs_lo=np.percentile(bs, 2.5), obs_hi=np.percentile(bs, 97.5), pred_markov=pred,
                     pred_markov_other_exps=pred_loo))
FP = pd.DataFrame(rows)
FP.to_csv(f"{a.out}/markov_first_passage.csv", index=False)

# ------------------------------------------------ Peclet numbers on the plateau
pe = []
for cond, sub in D.groupby("cond"):
    pl = sub[(sub.x >= PLATEAU[0]) & (sub.x < PLATEAU[1])]
    if len(pl) < 50:
        continue
    mu, var = pl.dx.mean(), pl.dx.var()
    Lw = PLATEAU[1] - PLATEAU[0]
    # cluster bootstrap by game
    gpos = pl.groupby("game").dx.apply(np.array)
    arrs = list(gpos.values)
    bmu, bpe = [], []
    for _ in range(a.boot):
        pick = rng.integers(0, len(arrs), len(arrs))
        z = np.concatenate([arrs[i] for i in pick])
        bmu.append(z.mean()); bpe.append(z.mean() * Lw / z.var())
    below = sub[sub.x < PLATEAU[0]]
    atcap = sub[sub.x >= np.log(199.5)]
    pe.append(dict(cond=cond, n=len(pl), mu_plateau=mu, var_plateau=var, Pe=mu * Lw / var,
                   Pe_lo=np.percentile(bpe, 2.5), Pe_hi=np.percentile(bpe, 97.5),
                   mu_lo=np.percentile(bmu, 2.5), mu_hi=np.percentile(bmu, 97.5),
                   mu_below25=below.dx.mean() if len(below) else np.nan,
                   mu_atcap=atcap.dx.mean() if len(atcap) else np.nan,
                   frac_time_atcap=float((sub.x >= np.log(199.5)).mean())))
PE = pd.DataFrame(pe)
PE.to_csv(f"{a.out}/peclet.csv", index=False)

# ------------------------------------------------ point of no return (humans and clones)
levels = np.array([150, 125, 100, 75, 50, 40, 30, 25, 20, 15, 10, 5])
minpool = D.groupby("game").pool.min()
pnr = []
for src in ["human", "clone"]:
    for cls in ["unconditional", "contingent", "all"]:
        gg = games[(games.source == src) & ((games.mech.map(L.MECH_CLASS) == cls) | (cls == "all"))]
        for lv in levels:
            hit = gg[minpool.reindex(gg.index) < lv]
            if len(hit) == 0:
                continue
            pnr.append(dict(source=src, mclass=cls, level=lv, n_reached=len(hit), p_collapse=hit.collapsed.mean()))
pd.DataFrame(pnr).to_csv(f"{a.out}/point_of_no_return.csv", index=False)

# ------------------------------------------------ history dependence: does the last increment predict the next one given the state bin?
Dh = D.sort_values(["game", "round"]).copy()
Dh["dx_lag"] = Dh.groupby("game").dx.shift(1)
hist = {}
for src in ["human", "clone"]:
    z = Dh[(Dh.source == src) & Dh.dx_lag.notna() & (Dh.x < np.log(199.5))].copy()
    cell = z["bin"].astype(str) + "|" + z["mech"]
    # Frisch-Waugh-Lovell: absorb bin x mechanism fixed effects, then OLS slope with game-clustered SE
    ry = z.dx - z.groupby(cell).dx.transform("mean")
    rx = z.dx_lag - z.groupby(cell).dx_lag.transform("mean")
    beta = float((rx * ry).sum() / (rx ** 2).sum())
    e = ry - beta * rx
    sc = (rx * e).groupby(z.game).sum()
    se = float(np.sqrt((sc ** 2).sum()) / (rx ** 2).sum())
    hist[src] = dict(beta_lag=beta, se=se, n=int(len(z)))
json.dump({"history": hist, "plateau": PLATEAU, "edges": list(EDGES)}, open(f"{a.out}/landscape_meta.json", "w"), indent=1)
print(FP.to_string()); print(PE.to_string()); print(hist)
