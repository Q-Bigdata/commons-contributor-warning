"""Confirmatory statistics for H1-H3: cluster-bootstrap CIs and p-values of univariate AUCs, of Delta AUC,
and of the mechanism-class difference in indicator-only AUC. Reads the outputs of run_analysis.py (human).
usage: python h1_boot.py --res results/human --out results/confirm --boot 2000
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import ews_lib as L

p = argparse.ArgumentParser()
p.add_argument("--res", required=True); p.add_argument("--out", required=True)
p.add_argument("--boot", type=int, default=2000); p.add_argument("--W", type=int, default=8); p.add_argument("--h", type=int, default=10)
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
rng = np.random.default_rng(99)
LM = pd.read_csv(f"{a.res}/landmarks_W{a.W}_h{a.h}.csv.gz")
FEATS = ["logR", "mean_logG", "last3_logG"] + L.FEATURES_EWS + L.FEATURES_STRUCT


def cluster_index(groups):
    pos = pd.Series(np.arange(len(groups))).groupby(groups).apply(np.array).to_dict()
    return pos, np.array(list(pos.keys()))


out_rows, summary = [], {}
for regime in ["early", "all"]:
    df = (LM[LM.pool_now >= 100] if regime == "early" else LM).reset_index(drop=True)
    oof = pd.read_csv(f"{a.res}/oof_{regime}.csv.gz")
    assert (oof.game.to_numpy() == df.game.to_numpy()).all() and (oof.t.to_numpy() == df.t.to_numpy()).all()
    for scope, sub in [("pooled", df)] + list(df.groupby("mech")) + list(df.groupby("mclass")):
        sub = sub.reset_index(drop=True)
        if sub.y.sum() < 5:
            continue
        y = sub.y.to_numpy(); pos, ug = cluster_index(sub.group.astype(str).to_numpy())
        bsidx = [np.concatenate([pos[g] for g in rng.choice(ug, len(ug))]) for _ in range(a.boot)]
        for f in FEATS:
            s = sub[f].to_numpy(float)
            est = L.auc(y, s)
            bs = np.array([L.auc(y[i], s[i]) for i in bsidx])
            out_rows.append(dict(regime=regime, scope=scope, feature=f, auc=est, lo=np.nanpercentile(bs, 2.5),
                                 hi=np.nanpercentile(bs, 97.5), p_le_half=float(np.nanmean(bs <= 0.5)),
                                 p_ge_half=float(np.nanmean(bs >= 0.5)), n=len(sub), events=int(y.sum())))
    # Delta AUC (H2) with bootstrap p
    y = df.y.to_numpy(); pos, ug = cluster_index(df.group.astype(str).to_numpy())
    d_se, d_fs = [], []
    for _ in range(a.boot):
        i = np.concatenate([pos[g] for g in rng.choice(ug, len(ug))])
        d_se.append(L.auc(y[i], oof["p_state+ews"].to_numpy()[i]) - L.auc(y[i], oof["p_state"].to_numpy()[i]))
        d_fs.append(L.auc(y[i], oof["p_full"].to_numpy()[i]) - L.auc(y[i], oof["p_state"].to_numpy()[i]))
    d_se, d_fs = np.array(d_se), np.array(d_fs)
    summary[f"{regime}_H2_delta_stateews_minus_state"] = dict(
        est=L.auc(y, oof["p_state+ews"]) - L.auc(y, oof["p_state"]), lo=float(np.nanpercentile(d_se, 2.5)),
        hi=float(np.nanpercentile(d_se, 97.5)), p_one_sided=float(np.nanmean(d_se <= 0)))
    summary[f"{regime}_delta_full_minus_state"] = dict(
        est=L.auc(y, oof["p_full"]) - L.auc(y, oof["p_state"]), lo=float(np.nanpercentile(d_fs, 2.5)),
        hi=float(np.nanpercentile(d_fs, 97.5)), p_one_sided=float(np.nanmean(d_fs <= 0)))
    # H3: indicator-only AUC, contingent minus unconditional
    cls = df.mclass.to_numpy(); pe = oof["p_ews"].to_numpy()
    if (y[cls == "unconditional"].sum() >= 5):
        diffs = []
        for _ in range(a.boot):
            i = np.concatenate([pos[g] for g in rng.choice(ug, len(ug))])
            ci, yi, pi = cls[i], y[i], pe[i]
            diffs.append(L.auc(yi[ci == "contingent"], pi[ci == "contingent"]) - L.auc(yi[ci == "unconditional"], pi[ci == "unconditional"]))
        diffs = np.array(diffs)
        est = L.auc(y[cls == "contingent"], pe[cls == "contingent"]) - L.auc(y[cls == "unconditional"], pe[cls == "unconditional"])
        summary[f"{regime}_H3_ews_auc_contingent_minus_unconditional"] = dict(
            est=est, lo=float(np.nanpercentile(diffs, 2.5)), hi=float(np.nanpercentile(diffs, 97.5)),
            p_one_sided=float(np.nanmean(diffs >= 0)))
    else:
        summary[f"{regime}_H3_ews_auc_contingent_minus_unconditional"] = dict(
            note="not testable: fewer than 5 events in unconditional mechanisms", events_unconditional=int(y[cls == "unconditional"].sum()))
U = pd.DataFrame(out_rows)
U.to_csv(f"{a.out}/univariate_auc_ci.csv", index=False)
json.dump(summary, open(f"{a.out}/confirm_summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
print(U[(U.scope == "pooled") & (U.regime == "early")].round(3).to_string())
