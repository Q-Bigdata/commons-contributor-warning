"""Detectability phase diagram for generic early-warning indicators in the capped commons.

Each game g has a persistent group mean return m_g ~ N(mu, 0.02) (between-group heterogeneity in resilience),
optionally a slow within-game decline (forcing: m_g,t = m_g - kappa_g t, kappa_g ~ U(0, 0.004)), and a
group-level AR(1) shock with innovation s.d. sigma (phi = 0.35, matching the momentum estimated in the human data).
Individual return fractions add N(0, 0.10) noise; allocation is equal; the pool obeys the exact law of the human game.
For every (margin, sigma, forcing) cell we run the SAME landmark pipeline as for the human data (W = 8, h = 10) and
report univariate AUCs and the cross-validated incremental AUC of the six generic indicators over the state model.
usage: python theory_phase.py --out DIR --ngames 2000 --procs 64
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import itertools
from multiprocessing import Pool
import numpy as np
import pandas as pd
import ews_lib as L

p = argparse.ArgumentParser()
p.add_argument("--out", required=True); p.add_argument("--ngames", type=int, default=2000)
p.add_argument("--procs", type=int, default=64)
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
MARGINS = [-0.03, -0.01, 0.0, 0.01, 0.03, 0.06]     # (1+r) * mu - 1
SIGMAS = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40]
FORCING = [0, 1]


def simulate(margin, sigma, forcing, n, seed):
    rng = np.random.default_rng(seed)
    mu = (1 + margin) / 1.4
    m_g = rng.normal(mu, 0.02, n)
    kap = rng.uniform(0, 0.004, n) if forcing else np.zeros(n)
    R = np.full(n, 200.0); alive = np.ones(n, bool); eta = np.zeros(n)
    recs = []
    for t in range(40):
        eta = 0.35 * eta + rng.normal(0, sigma, n)
        m = m_g - kap * t + eta
        e = np.where(alive, R / 4, 0.0)
        f = np.clip(m[:, None] + rng.normal(0, 0.10, (n, 4)), 0, 1)
        c = np.round(f * e[:, None])
        nxt = np.where(alive, np.minimum(200.0, R - 4 * e + 1.4 * c.sum(1)), 0.01)
        recs.append(pd.DataFrame({"game": np.arange(n), "round": t, "pool": np.where(alive, R, 0.01), "next_pool": nxt,
                                  **{f"e{i}": e for i in range(4)}, **{f"c{i}": c[:, i] for i in range(4)}}))
        alive = alive & (nxt >= 1)
        R = np.where(alive, nxt, 0.01)
    D = pd.concat(recs, ignore_index=True)
    D["game"] = "g" + D.game.astype(str); D["group"] = D.game
    D["source"] = "phase"; D["mech"] = "phase"; D["exp"] = 0; D["bc"] = 0
    return D


def cv_delta(df, rng):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    ug = df.game.unique(); fold = dict(zip(rng.permutation(ug), np.arange(len(ug)) % 5))
    f = df.game.map(fold).to_numpy(); y = df.y.to_numpy()
    out = {}
    for name, feats in [("state", L.FEATURES_STATE), ("state+ews", L.FEATURES_STATE + L.FEATURES_EWS), ("ews", L.FEATURES_EWS)]:
        pr = np.zeros(len(df))
        for k in range(5):
            tr, te = f != k, f == k
            if y[tr].sum() < 3:
                pr[te] = np.nan; continue
            m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000)).fit(df.loc[tr, feats], y[tr])
            pr[te] = m.predict_proba(df.loc[te, feats])[:, 1]
        out[name] = L.auc(y, pr)
    return out


def cell(args):
    margin, sigma, forcing, seed = args
    D = L.prepare(simulate(margin, sigma, forcing, a.ngames, seed))
    res = dict(margin=margin, sigma=sigma, forcing=forcing, collapse=float(D.groupby("game").tau.first().notna().mean()))
    LM = L.landmarks(D, W=8, h=10)
    rng = np.random.default_rng(seed + 1)
    for regime, df in [("all", LM), ("early", LM[LM.pool_now >= 100])]:
        df = df.reset_index(drop=True)
        res[f"{regime}_n"] = len(df); res[f"{regime}_events"] = int(df.y.sum())
        if df.y.sum() < 10:
            continue
        for fch in ["logR", "mean_logG", "var_pool", "ar1_pool", "var_logpool", "var_rho", "ar1_rho", "cap_frac", "flicker"]:
            res[f"{regime}_auc_{fch}"] = L.auc(df.y, df[fch])
        cv = cv_delta(df, rng)
        for k, v in cv.items():
            res[f"{regime}_cv_{k}"] = v
        res[f"{regime}_delta"] = cv["state+ews"] - cv["state"]
    return res


if __name__ == "__main__":
    grid = [(m, s, fo, 1000 + i) for i, (m, s, fo) in enumerate(itertools.product(MARGINS, SIGMAS, FORCING))]
    with Pool(a.procs) as pool:
        rows = pool.map(cell, grid)
    pd.DataFrame(rows).to_csv(f"{a.out}/phase_diagram.csv", index=False)
    print(pd.DataFrame(rows)[["margin", "sigma", "forcing", "collapse", "early_events", "early_auc_var_pool", "early_auc_ar1_pool", "early_delta", "all_delta"]].round(3).to_string())
