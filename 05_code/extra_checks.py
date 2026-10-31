"""Exploratory checks requested by the pattern in the per-mechanism AUCs (labelled exploratory in the manuscript).
(1) Mechanism-stratified AUC of each indicator (only event/non-event pairs within the same mechanism are compared).
(2) Does the generic-indicator set add to the contributor-base (structural) set? CV AUC of state+struct vs full,
    pooled and within the Proportional mechanism, with cluster-bootstrap CIs.
usage: python extra_checks.py --res results/human --out out/extra --boot 2000
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import ews_lib as L

p = argparse.ArgumentParser()
p.add_argument("--res", required=True); p.add_argument("--out", required=True); p.add_argument("--boot", type=int, default=2000)
a = p.parse_args(); os.makedirs(a.out, exist_ok=True)
rng = np.random.default_rng(2027)
LM = pd.read_csv(f"{a.res}/landmarks_W8_h10.csv.gz")
E = LM[LM.pool_now >= 100].reset_index(drop=True)
FE = ["logR", "mean_logG", "last3_logG"] + L.FEATURES_EWS + L.FEATURES_STRUCT


def strat_auc(y, s, strata):
    num, den = 0.0, 0.0
    for st in np.unique(strata):
        m = strata == st
        yy, ss = y[m], s[m]
        ok = np.isfinite(ss); yy, ss = yy[ok], ss[ok]
        n1 = yy.sum(); n0 = len(yy) - n1
        if n1 == 0 or n0 == 0:
            continue
        num += L.auc(yy, ss) * n1 * n0; den += n1 * n0
    return num / den if den else np.nan


pos = pd.Series(np.arange(len(E))).groupby(E.group.astype(str).to_numpy()).apply(np.array).to_dict()
ug = np.array(list(pos.keys()))
boots = [np.concatenate([pos[g] for g in rng.choice(ug, len(ug))]) for _ in range(a.boot)]
y = E.y.to_numpy(); st = E.mech.to_numpy()
rows = []
for f in FE:
    s = E[f].to_numpy(float)
    est = strat_auc(y, s, st)
    bs = np.array([strat_auc(y[i], s[i], st[i]) for i in boots])
    rows.append(dict(feature=f, strat_auc=est, lo=np.nanpercentile(bs, 2.5), hi=np.nanpercentile(bs, 97.5),
                     p_le_half=float(np.nanmean(bs <= 0.5))))
pd.DataFrame(rows).to_csv(f"{a.out}/stratified_auc_early.csv", index=False)


def oof(df, feats, reps=5):
    groups = df.group.astype(str).to_numpy(); ugg = np.unique(groups); pred = np.zeros(len(df)); yv = df.y.to_numpy()
    for r in range(reps):
        fold = dict(zip(rng.permutation(ugg), np.arange(len(ugg)) % 5)); f = np.array([fold[g] for g in groups])
        for k in range(5):
            tr, te = f != k, f == k
            m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000)).fit(df.loc[tr, feats], yv[tr])
            pred[te] += m.predict_proba(df.loc[te, feats])[:, 1] / reps
    return pred


out = {}
for scope, df in [("pooled", E), ("Proportional", E[E.mech == "Proportional"].reset_index(drop=True))]:
    sets = {"state+struct": L.FEATURES_STATE + L.FEATURES_STRUCT, "full": FE, "struct": L.FEATURES_STRUCT,
            "struct+ews": L.FEATURES_STRUCT + L.FEATURES_EWS}
    pr = {k: oof(df, v) for k, v in sets.items()}
    yv = df.y.to_numpy()
    pp = pd.Series(np.arange(len(df))).groupby(df.group.astype(str).to_numpy()).apply(np.array).to_dict(); uu = np.array(list(pp.keys()))
    bsi = [np.concatenate([pp[g] for g in rng.choice(uu, len(uu))]) for _ in range(a.boot)]
    res = {}
    for k in pr:
        b = np.array([L.auc(yv[i], pr[k][i]) for i in bsi])
        res[k] = dict(est=L.auc(yv, pr[k]), lo=float(np.nanpercentile(b, 2.5)), hi=float(np.nanpercentile(b, 97.5)))
    for k1, k0 in [("full", "state+struct"), ("struct+ews", "struct")]:
        d = np.array([L.auc(yv[i], pr[k1][i]) - L.auc(yv[i], pr[k0][i]) for i in bsi])
        res[f"delta_{k1}_minus_{k0}"] = dict(est=L.auc(yv, pr[k1]) - L.auc(yv, pr[k0]), lo=float(np.nanpercentile(d, 2.5)),
                                             hi=float(np.nanpercentile(d, 97.5)), p_one_sided=float(np.nanmean(d <= 0)))
    res["n"] = int(len(df)); res["events"] = int(yv.sum())
    out[scope] = res
json.dump(out, open(f"{a.out}/incremental_over_struct.json", "w"), indent=1)
print(pd.DataFrame(rows).round(3).to_string()); print(json.dumps(out, indent=1))
