"""Revision analysis 5: theory checks requested by the stochastic-dynamics reviewers.

(a) Stationary capped commons R_{t+1} = min(K, G_t R_t), G_t = 1.4 * clip(rho_bar + sigma*eps, 0, 1):
    report the true log-growth margin E[log G_t] next to the arithmetic margin, long-run lag-1 autocorrelation,
    and the SHORT-WINDOW estimator used on the human data (8-round windows, linear detrending), plus the
    fraction of windows pinned at the cap. Collapsed runs are restarted at K (stationarity device); this is a
    benchmark for relaxation, not a model of absorbed games.
(b) Phase-diagram sensitivity: the equal-allocation, Gaussian-shock simulator is compared with (i) proportional
    allocation (shares equal last-round return shares; zero allocation -> zero return), (ii) heavy-tailed
    (Student-t, 3 d.f., rescaled to the same s.d.) group shocks, (iii) both. Slow forcing, 3,000 games per cell,
    identical landmark pipeline (W=8, h=10, pool >= 100).
usage: python rev_theory.py --out OUT --procs 64 --ngames 3000
"""
import argparse, itertools, json, os, sys
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
import ews_lib as L

p = argparse.ArgumentParser(); p.add_argument("--out", required=True); p.add_argument("--procs", type=int, default=64)
p.add_argument("--ngames", type=int, default=3000); p.add_argument("--smoke", type=int, default=0)
a = p.parse_args(); os.makedirs(a.out, exist_ok=True)
K = 200.0


def stationary(args):
    rho_bar, sigma, seed = args
    rng = np.random.default_rng(seed)
    n, T, W = 300, 400, 8
    R = np.full(n, K); X = []; lg = []
    for t in range(T):
        G = 1.4 * np.clip(rho_bar + sigma * rng.standard_normal(n), 0, 1)
        lg.append(np.log(np.maximum(G, 1e-12)))
        R = np.minimum(K, G * R); R = np.where(R < 1, K, R)
        if t >= 100:
            X.append(R.copy())
    X = np.array(X); lg = np.array(lg)
    def ar1(v):
        v = v - v.mean(); d = (v[:-1] ** 2).sum(); return (v[:-1] * v[1:]).sum() / d if d > 0 else np.nan
    lr = np.nanmean([ar1(X[:, i]) for i in range(n)])
    sw, sv, pinned = [], [], 0; tot = 0
    tt = np.arange(W)
    for i in range(0, n, 3):
        for s0 in range(0, X.shape[0] - W, 4):
            w = X[s0:s0 + W, i]; tot += 1
            if np.all(w >= K - 1e-9):
                pinned += 1; continue
            r = w - np.polyval(np.polyfit(tt, w, 1), tt)
            sv.append(r.var())
            if r[:-1].std() > 1e-9 and r[1:].std() > 1e-9:
                sw.append(np.corrcoef(r[:-1], r[1:])[0, 1])
    return dict(rho_bar=rho_bar, sigma=sigma, arith_margin=1.4 * rho_bar - 1, log_margin=float(np.mean(lg)),
                ar1_longrun=float(lr), ar1_window8_median=float(np.median(sw)) if sw else np.nan,
                var_window8_median=float(np.median(sv)) if sv else np.nan, frac_windows_pinned=pinned / tot)


def simulate(margin, sigma, alloc, tails, n, seed):
    rng = np.random.default_rng(seed)
    mu = (1 + margin) / 1.4
    m_g = rng.normal(mu, 0.02, n); kap = rng.uniform(0, 0.004, n)
    R = np.full(n, K); alive = np.ones(n, bool); eta = np.zeros(n); share = np.full((n, 4), 0.25)
    recs = []
    for t in range(40):
        z = rng.standard_t(3, n) / np.sqrt(3.0) if tails == "t3" else rng.standard_normal(n)
        eta = 0.35 * eta + sigma * z
        m = m_g - kap * t + eta
        e = np.where(alive[:, None], R[:, None] * share, 0.0)
        f = np.clip(m[:, None] + rng.normal(0, 0.10, (n, 4)), 0, 1)
        c = np.round(f * e)
        nxt = np.where(alive, np.minimum(K, R - e.sum(1) + 1.4 * c.sum(1)), 0.01)
        recs.append(pd.DataFrame({"game": np.arange(n), "round": t, "pool": np.where(alive, R, 0.01), "next_pool": nxt,
                                  **{f"e{i}": e[:, i] for i in range(4)}, **{f"c{i}": c[:, i] for i in range(4)}}))
        if alloc == "proportional":
            cs = c.sum(1, keepdims=True)
            share = np.where(cs > 0, c / np.maximum(cs, 1e-9), 0.25)
        alive = alive & (nxt >= 1); R = np.where(alive, nxt, 0.01)
    D = pd.concat(recs, ignore_index=True)
    D["game"] = "g" + D.game.astype(str); D["group"] = D.game
    D["source"] = "phase"; D["mech"] = "phase"; D["exp"] = 0; D["bc"] = 0
    return D


def cell(args):
    margin, sigma, alloc, tails, seed = args
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    D = L.prepare(simulate(margin, sigma, alloc, tails, a.ngames, seed))
    LM = L.landmarks(D, W=8, h=10)
    E = LM[LM.pool_now >= 100].reset_index(drop=True)
    res = dict(margin=margin, sigma=sigma, alloc=alloc, tails=tails, collapse=float(D.groupby("game").tau.first().notna().mean()),
               early_events=int(E.y.sum()), early_n=int(len(E)))
    if E.y.sum() >= 10:
        for f in ["var_pool", "ar1_pool", "var_rho", "gini_mean", "active_mean"]:
            res[f"auc_{f}"] = L.auc(E.y, E[f]) if f != "active_mean" else 1 - L.auc(E.y, E[f])
        rng = np.random.default_rng(seed + 1)
        ug = E.game.unique(); fold = dict(zip(rng.permutation(ug), np.arange(len(ug)) % 5)); fo = E.game.map(fold).to_numpy()
        y = E.y.to_numpy()
        for name, feats in [("state", L.FEATURES_STATE), ("state+ews", L.FEATURES_STATE + L.FEATURES_EWS)]:
            pr = np.zeros(len(E))
            for k in range(5):
                tr, te = fo != k, fo == k
                if y[tr].sum() < 3:
                    pr[te] = np.nan; continue
                m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000)).fit(E.loc[tr, feats], y[tr])
                pr[te] = m.predict_proba(E.loc[te, feats])[:, 1]
            res[f"cv_{name}"] = L.auc(y, pr)
        res["delta"] = res["cv_state+ews"] - res["cv_state"]
    return res


if __name__ == "__main__":
    st_jobs = [(rb, s, 10 + i) for i, (rb, s) in enumerate(itertools.product(np.round(np.arange(0.715, 0.8201, 0.015), 3), [0.02, 0.05, 0.10, 0.20]))]
    ph_jobs = [(m, s, al, tl, 2000 + i) for i, (m, s, al, tl) in enumerate(itertools.product(
        [-0.01, 0.0, 0.01, 0.03], [0.10, 0.20, 0.30, 0.40], ["equal", "proportional"], ["gauss", "t3"]))]
    if a.smoke:
        st_jobs, ph_jobs = st_jobs[:2], [j for j in ph_jobs if j[1] == 0.30][:4]
    with (Pool(a.procs) if a.procs > 1 else _Dummy()) as pool:
        st = pool.map(stationary, st_jobs)
        ph = pool.map(cell, ph_jobs)
    pd.DataFrame(st).to_csv(f"{a.out}/stationary_margins.csv", index=False)
    pd.DataFrame(ph).to_csv(f"{a.out}/phase_sensitivity.csv", index=False)
    print(pd.DataFrame(st).round(3).to_string())
    print(pd.DataFrame(ph).round(3).to_string())
