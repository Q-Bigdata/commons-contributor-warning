"""Revision analysis 4: first-passage validation of the state-only Markov model beyond terminal collapse fractions.

(1) Observed vs simulated survival curves S(t) (fraction not yet collapsed by round t) for each human condition,
    with the state-only kernel (pool bin x institution) and a short-memory kernel (pool bin x sign class of the
    previous change in log pool). Kernel-estimation uncertainty by bootstrapping games within condition.
(2) Exp 4 horizons: administrative game lengths are used (collapse does not shorten the record, verified).
(3) Plateau drift/diffusion: capped vs uncapped, one-step vs long-run variance (including lag-1 serial covariance),
    Peclet numbers on alternative plateau widths.
usage: python rev_markov.py --res OUT --nsim 20000 --boot 200 --procs 32
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multiprocessing import Pool


class _Dummy:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def map(self, f, it):
        return list(map(f, it))

import numpy as np
import pandas as pd

p = argparse.ArgumentParser(); p.add_argument("--res", required=True); p.add_argument("--nsim", type=int, default=20000)
p.add_argument("--boot", type=int, default=200); p.add_argument("--procs", type=int, default=32)
a = p.parse_args()
H = pd.read_csv(f"{a.res}/rounds_ext_human.csv.gz")
EDGES = np.log(np.array([1, 5, 10, 25, 50, 75, 100, 125, 150, 175, 190, 199.5, 200.01]))
NB = len(EDGES) - 1
D = H[H.pool >= 1].sort_values(["game", "round"]).copy()
D["x"] = np.log(D.pool)
D["bin"] = np.clip(np.digitize(D.x, EDGES) - 1, 0, NB - 1)
D["dx_lag"] = D.groupby("game").dx.shift(1)
D["mem"] = np.select([D.dx_lag.isna(), D.dx_lag < -0.05, D.dx_lag > 0.05], [1, 0, 2], 1)   # down / flat-or-first / up
D["cond"] = D.mech + "|" + D.exp.astype(str)
G = H.groupby("game").agg(cond=("mech", "first"), exp=("exp", "first"), tau=("tau", "first"), T=("T", "first"))
G["cond"] = G.cond + "|" + G.exp.astype(str)
exp4T = G[G.exp == 4]["T"].to_numpy()


def kernel(sub, memory):
    K = {}
    for (b, m), x in sub.groupby(["bin", "mem"] if memory else ["bin", "mem"]):
        K.setdefault(b, {}).setdefault(m if memory else 0, []).extend(x.dx.tolist())
    if not memory:
        K = {b: {0: sum(v.values(), [])} for b, v in K.items()}
    # fill: memory cells with <5 obs borrow the bin-pooled distribution; empty bins borrow nearest bin
    pooled = {b: np.array(sum(v.values(), [])) for b, v in K.items()}
    pop = [b for b in pooled if len(pooled[b]) >= 5]
    F = {}
    for b in range(NB):
        bb = b if b in pooled and len(pooled[b]) >= 5 else min(pop, key=lambda q: abs(q - b))
        F[b] = {}
        for m in ([0, 1, 2] if memory else [0]):
            v = np.array(K.get(bb, {}).get(m, []))
            F[b][m] = v if len(v) >= 5 else pooled[bb]
    return F


def simulate(F, horizons, memory, rng):
    n = len(horizons); Tm = int(horizons.max())
    x = np.full(n, np.log(200.0)); mem = np.ones(n, int); alive = np.ones(n, bool)
    tau = np.full(n, np.nan)
    for t in range(Tm):
        act = alive & (t < horizons)
        if not act.any():
            break
        b = np.clip(np.digitize(x, EDGES) - 1, 0, NB - 1)
        dx = np.zeros(n)
        for bb in np.unique(b[act]):
            for m in ([0, 1, 2] if memory else [0]):
                sel = act & (b == bb) & ((mem == m) if memory else True)
                if sel.any():
                    dx[sel] = rng.choice(F[bb][m], sel.sum())
        x = np.where(act, np.minimum(np.log(200.0), x + dx), x)
        mem = np.where(act, np.select([dx < -0.05, dx > 0.05], [0, 2], 1), mem)
        newc = act & (x < 0)
        tau[newc] = t; alive &= ~newc
    return tau


def surv_curve(tau, horizons, Tmax=40):
    return [float(np.mean(~(tau <= t))) for t in range(Tmax)]


def run_cond(args):
    cond, seed = args
    rng = np.random.default_rng(seed)
    sub = D[D.cond == cond]
    games = sub.game.unique()
    parts = {g: x for g, x in sub.groupby("game")}
    hor = rng.choice(exp4T, a.nsim) if cond.endswith("|4") else np.full(a.nsim, 40)
    obs_tau = G.loc[games, "tau"].to_numpy()
    obs_T = G.loc[games, "T"].to_numpy()
    out = {"cond": cond, "n_games": int(len(games)), "obs_collapse": float(np.mean(np.isfinite(obs_tau))),
           "obs_surv": [float(np.mean(~(obs_tau <= t)[obs_T > t])) if (obs_T > t).any() else np.nan for t in range(40)]}
    for memory in [False, True]:
        F = kernel(sub, memory)
        tau = simulate(F, hor, memory, rng)
        key = "mem" if memory else "state"
        out[f"{key}_collapse"] = float(np.mean(np.isfinite(tau)))
        out[f"{key}_surv"] = surv_curve(tau, hor)
        out[f"{key}_median_tau"] = float(np.nanmedian(tau)) if np.isfinite(tau).any() else np.nan
        # kernel uncertainty: bootstrap games
        bc = []
        for _ in range(a.boot if not memory else max(50, a.boot // 2)):
            pick = rng.choice(games, len(games))
            sb = pd.concat([parts[g] for g in pick])
            Fb = kernel(sb, memory)
            tb = simulate(Fb, hor[:4000], memory, rng)
            bc.append(np.mean(np.isfinite(tb)))
        out[f"{key}_collapse_lo"], out[f"{key}_collapse_hi"] = float(np.percentile(bc, 2.5)), float(np.percentile(bc, 97.5))
    out["obs_median_tau"] = float(np.nanmedian(obs_tau)) if np.isfinite(obs_tau).any() else np.nan
    # Kolmogorov distance between observed and simulated survival curves (rounds with observations)
    for key in ["state", "mem"]:
        d = [abs(o - s) for o, s in zip(out["obs_surv"], out[f"{key}_surv"]) if o == o]
        out[f"{key}_maxdiff_surv"] = float(max(d))
    return out


if __name__ == "__main__":
    # Exp 4: verify that collapse does not shorten recorded games (administrative horizons)
    e4 = G[G.exp == 4]
    chk = dict(exp4_T_min=int(e4["T"].min()), exp4_T_collapsed_mean=float(e4[e4.tau.notna()]["T"].mean()),
               exp4_T_survived_mean=float(e4[e4.tau.isna()]["T"].mean()))
    conds = sorted(D.cond.unique())
    with (Pool(min(a.procs, len(conds))) if a.procs > 1 else _Dummy()) as pool:
        rows = pool.map(run_cond, [(c, 500 + i) for i, c in enumerate(conds)])
    # plateau statistics
    pl = []
    for (mech), x in D.groupby("mech"):
        for lo_, hi_ in [(25, 190), (50, 190), (25, 175), (10, 190)]:
            y = x[(x.pool >= lo_) & (x.pool < hi_)]
            if len(y) < 50:
                continue
            Lw = np.log(hi_ / lo_)
            r1 = y.dropna(subset=["dx_lag"])
            phi = float(np.corrcoef(r1.dx, r1.dx_lag)[0, 1])
            s2 = float(y.dx.var()); s2lr = s2 * (1 + phi) / (1 - phi) if phi < 0.95 else np.nan
            mu = float(y.dx.mean()); mu_unc = float(y.logG.mean())
            pl.append(dict(mech=mech, lo=lo_, hi=hi_, n=int(len(y)), mu_capped=mu, mu_uncapped=mu_unc, s2=s2, phi=phi, s2_longrun=s2lr,
                           Pe=mu * Lw / s2, Pe_longrun=mu * Lw / s2lr if s2lr == s2lr else np.nan,
                           q_mean=float(y.q.mean()), frac_q_lt_099=float((y.q < 0.99).mean())))
    json.dump(dict(check=chk, conds=rows, plateau=pl), open(f"{a.res}/markov_revision.json", "w"), indent=1, default=float)
    print(chk)
    print(pd.DataFrame([{k: r[k] for k in ["cond", "n_games", "obs_collapse", "state_collapse", "state_collapse_lo", "state_collapse_hi",
                                           "mem_collapse", "obs_median_tau", "state_median_tau", "mem_median_tau", "state_maxdiff_surv", "mem_maxdiff_surv"]} for r in rows]).round(3).to_string())
    print(pd.DataFrame(pl).round(3).to_string())
