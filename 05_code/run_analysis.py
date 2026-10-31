"""Prospective (landmark) and retrospective early-warning analysis for one data source.

usage: python run_analysis.py --source clone|human --out DIR [--W 8 --h 10 --boot 1000]
"""
import argparse, json, os, pickle, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import ews_lib as L

p = argparse.ArgumentParser()
p.add_argument("--source", required=True)
p.add_argument("--out", required=True)
p.add_argument("--W", type=int, default=8)
p.add_argument("--h", type=int, default=10)
p.add_argument("--boot", type=int, default=1000)
p.add_argument("--early", type=float, default=100.0)
p.add_argument("--seed", type=int, default=20260930)
p.add_argument("--data", default="data/rounds.csv.gz")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
rng = np.random.default_rng(a.seed)
t0 = time.time()

R = pd.read_csv(a.data)
R = R[R.source == a.source]
D = L.prepare(R)
LM = L.landmarks(D, W=a.W, h=a.h)
LM.to_csv(f"{a.out}/landmarks_W{a.W}_h{a.h}.csv.gz", index=False, compression="gzip")
print("landmarks", LM.shape, "events", LM.y.sum(), "time", round(time.time() - t0, 1), flush=True)

FEATS = {"state": L.FEATURES_STATE, "ews": L.FEATURES_EWS, "struct": L.FEATURES_STRUCT,
         "state+ews": L.FEATURES_STATE + L.FEATURES_EWS,
         "full": L.FEATURES_STATE + L.FEATURES_EWS + L.FEATURES_STRUCT}
SINGLE = ["logR", "mean_logG", "last3_logG"] + L.FEATURES_EWS + L.FEATURES_STRUCT


def model():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=1.0, max_iter=2000))


def oof_predictions(df, feats, n_splits=5, reps=5):
    """Out-of-fold predictions with folds defined on groups (clusters); averaged over repeats."""
    groups = df.group.to_numpy()
    ug = np.unique(groups)
    pred = np.zeros(len(df))
    for r in range(reps):
        perm = rng.permutation(ug)
        fold_of = {g: i % n_splits for i, g in enumerate(perm)}
        f = np.array([fold_of[g] for g in groups])
        for k in range(n_splits):
            tr, te = f != k, f == k
            if df.y.to_numpy()[tr].sum() == 0:
                continue
            m = model().fit(df.loc[tr, feats], df.y.to_numpy()[tr])
            pred[te] += m.predict_proba(df.loc[te, feats])[:, 1] / reps
    return pred


def boot_auc(y, scores: dict, groups, B):
    """Cluster bootstrap of AUCs and pairwise differences."""
    pos = pd.Series(np.arange(len(groups))).groupby(groups).apply(np.array).to_dict()
    ug = np.array(list(pos.keys()))
    out = {k: [] for k in scores}
    for _ in range(B):
        pick = rng.choice(ug, size=len(ug), replace=True)
        idx = np.concatenate([pos[g] for g in pick])
        for k, s in scores.items():
            out[k].append(L.auc(y[idx], s[idx]))
    return {k: np.array(v) for k, v in out.items()}


def summarise(est, bs):
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return {"est": float(est), "lo": float(lo), "hi": float(hi)}


res = {"source": a.source, "W": a.W, "h": a.h, "early_threshold": a.early,
       "n_games": int(D.game.nunique()), "n_collapsed": int(D.groupby("game").tau.first().notna().sum()),
       "n_groups": int(D.group.nunique())}
regimes = {"all": LM, "early": LM[LM.pool_now >= a.early]}
uni_rows = []
for rname, df in regimes.items():
    df = df.reset_index(drop=True)
    res[f"{rname}_n"] = int(len(df)); res[f"{rname}_events"] = int(df.y.sum())
    # univariate AUCs by mechanism and pooled
    for mech, sub in [("pooled", df)] + list(df.groupby("mech")) + list(df.groupby("mclass")):
        for f in SINGLE:
            uni_rows.append(dict(regime=rname, mech=mech, feature=f, auc=L.auc(sub.y, sub[f]),
                                 n=len(sub), events=int(sub.y.sum())))
    # cross-validated models
    preds = {k: oof_predictions(df, v) for k, v in FEATS.items()}
    y = df.y.to_numpy(); g = df.group.to_numpy()
    bs = boot_auc(y, preds, g, a.boot)
    res[f"{rname}_cv"] = {k: summarise(L.auc(y, preds[k]), bs[k]) for k in preds}
    for k1, k0 in [("state+ews", "state"), ("full", "state"), ("ews", "state"), ("full", "state+ews")]:
        d = bs[k1] - bs[k0]
        res[f"{rname}_cv"][f"delta_{k1}_minus_{k0}"] = summarise(L.auc(y, preds[k1]) - L.auc(y, preds[k0]), d)
    # moderation: CV AUC by mechanism class and by mechanism
    for col in ["mclass", "mech"]:
        for lev, sub in df.groupby(col):
            if sub.y.sum() < 5 or (len(sub) - sub.y.sum()) < 5:
                continue
            ii = sub.index.to_numpy()
            bsl = boot_auc(y[ii], {k: preds[k][ii] for k in ["state", "ews", "state+ews"]}, g[ii], max(200, a.boot // 2))
            res[f"{rname}_cv_{col}_{lev}"] = {k: summarise(L.auc(y[ii], preds[k][ii]), bsl[k]) for k in bsl}
            res[f"{rname}_cv_{col}_{lev}"]["delta_state+ews_minus_state"] = summarise(
                L.auc(y[ii], preds["state+ews"][ii]) - L.auc(y[ii], preds["state"][ii]), bsl["state+ews"] - bsl["state"])
            res[f"{rname}_cv_{col}_{lev}"]["n"] = int(len(ii)); res[f"{rname}_cv_{col}_{lev}"]["events"] = int(y[ii].sum())
    pd.DataFrame({"game": df.game, "t": df.t, "y": y, "mech": df.mech, **{f"p_{k}": v for k, v in preds.items()}}).to_csv(
        f"{a.out}/oof_{rname}.csv.gz", index=False, compression="gzip")
    # full-data fits for transfer (fit on this source, apply to another)
    fits = {k: model().fit(df[v], y) for k, v in FEATS.items()}
    with open(f"{a.out}/fits_{rname}.pkl", "wb") as fh:
        pickle.dump({"feats": FEATS, "fits": fits}, fh)
    print(rname, "done", round(time.time() - t0, 1), flush=True)
pd.DataFrame(uni_rows).to_csv(f"{a.out}/univariate_auc.csv", index=False)


# ---------------------------------------------------------------- retrospective (classical) analysis
def rolling_ind(y, w):
    v, ac = [], []
    for j in range(w - 1, len(y)):
        vv, aa = L._var_ar1(y[j - w + 1:j + 1])
        v.append(vv); ac.append(aa)
    return np.array(v), np.array(ac)


def ktau(z):
    z = np.asarray(z, float)
    ok = np.isfinite(z)
    if ok.sum() < 4 or np.nanstd(z[ok]) < 1e-12:
        return np.nan
    return kendalltau(np.arange(ok.sum()), z[ok])[0]


games = {g: x.sort_values("round") for g, x in D.groupby("game")}
retro = []
surv = {g: x for g, x in games.items() if not np.isfinite(x.tau.iloc[0])}
for g, x in games.items():
    tau = x.tau.iloc[0]
    if not np.isfinite(tau):
        continue
    pn = x.next_pool.to_numpy()
    below = pn < 100
    # onset of the final descent: first round after which the pool never returns to >= 100
    tc = None
    for j in range(len(pn)):
        if below[j:].all():
            tc = j; break
    if tc is None or tc < 12:
        continue
    seg_len = tc  # rounds 0..tc-1
    w = max(6, seg_len // 2)
    rec = {"game": g, "mech": x.mech.iloc[0], "tau": tau, "tc": tc, "seg_len": seg_len, "w": w}
    series = {"pool": pn[:tc], "rho": x.rho.to_numpy()[:tc]}
    for sname, s in series.items():
        v, ac = rolling_ind(s, w)
        rec[f"kt_var_{sname}"] = ktau(v); rec[f"kt_ar1_{sname}"] = ktau(ac)
        # surrogate p-value (shuffle within segment)
        nulls_v, nulls_a = [], []
        for _ in range(200):
            sp = rng.permutation(s)
            vv, aa = rolling_ind(sp, w)
            nulls_v.append(ktau(vv)); nulls_a.append(ktau(aa))
        rec[f"p_var_{sname}"] = float(np.nanmean(np.array(nulls_v) >= rec[f"kt_var_{sname}"])) if np.isfinite(rec[f"kt_var_{sname}"]) else np.nan
        rec[f"p_ar1_{sname}"] = float(np.nanmean(np.array(nulls_a) >= rec[f"kt_ar1_{sname}"])) if np.isfinite(rec[f"kt_ar1_{sname}"]) else np.nan
    # time-matched controls: surviving games of the same mechanism, same segment
    ctrl = []
    elig = [gs for gs, xs in surv.items() if xs.mech.iloc[0] == rec["mech"] and len(xs) >= tc]
    if len(elig) > 50:
        elig = list(rng.choice(elig, 50, replace=False))
    for gs in elig:
        xs = surv[gs]
        cs = {"pool": xs.next_pool.to_numpy()[:tc], "rho": xs.rho.to_numpy()[:tc]}
        cr = {}
        for sname, s in cs.items():
            v, ac = rolling_ind(s, w)
            cr[f"kt_var_{sname}"] = ktau(v); cr[f"kt_ar1_{sname}"] = ktau(ac)
        ctrl.append(cr)
    if ctrl:
        C = pd.DataFrame(ctrl)
        for c in C.columns:
            rec[f"ctrl_{c}"] = float(np.nanmean(C[c]))
            rec[f"ctrl_pct_{c}"] = float(np.nanmean(C[c] < rec[c])) if np.isfinite(rec[c]) else np.nan
    retro.append(rec)
RT = pd.DataFrame(retro)
RT.to_csv(f"{a.out}/retrospective.csv", index=False)
if len(RT):
    rs = {"n": int(len(RT))}
    for c in ["kt_var_pool", "kt_ar1_pool", "kt_var_rho", "kt_ar1_rho"]:
        pc = c.replace("kt_", "p_")
        rs[c] = {"median": float(np.nanmedian(RT[c])), "frac_pos": float(np.nanmean(RT[c] > 0)),
                 "frac_sig05": float(np.nanmean(RT[pc] < 0.05)),
                 "median_ctrl": float(np.nanmedian(RT.get(f"ctrl_{c}", pd.Series(dtype=float)))),
                 "mean_pct_vs_ctrl": float(np.nanmean(RT.get(f"ctrl_pct_{c}", pd.Series(dtype=float))))}
    res["retrospective"] = rs
json.dump(res, open(f"{a.out}/results.json", "w"), indent=1)
print("finished", round(time.time() - t0, 1))
