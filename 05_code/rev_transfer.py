"""Revision analysis 3: clone-to-human transfer - provenance-aware subsets, institution strata, calibration, alerts.

Clone ensembles (Koster et al. 2025, Methods): BC1 was trained on 537 pre-Experiment-1 games (random/baseline
mechanisms and pilot agent M0), disjoint from all evaluation games; BC2 was trained on 990 games that include the
Experiment 1 and 2 evaluation games but precede Experiments 3 and 4. Experiments 3-4 are therefore disjoint from
the training data of every clone and of every detector fitted here.
Detectors: logistic models (fixed configuration, C=1, median imputation, standardisation, all fitted on clone
landmarks only); hyperparameters were not tuned. Models are trained (i) on all clone episodes, (ii) on BC1 episodes only.
Outputs: AUC (landmark-weighted, game-weighted, institution-stratified), by institution and by experiment subset;
calibration-in-the-large, calibration slope, Brier score and reliability bins; group-cross-fitted intercept
recalibration (reported separately); an alert rule with a threshold fixed on clone data.
usage: python rev_transfer.py --res OUT --boot 1000
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
import rev_lib as RL

p = argparse.ArgumentParser(); p.add_argument("--res", required=True); p.add_argument("--boot", type=int, default=1000)
a = p.parse_args()
rng = np.random.default_rng(77)
LH = pd.read_csv(f"{a.res}/landmarks_ext_human.csv.gz"); LC = pd.read_csv(f"{a.res}/landmarks_ext_clone.csv.gz")
FS = {"state": L.FEATURES_STATE, "state+ews": L.FEATURES_STATE + L.FEATURES_EWS, "alloc": RL.FEATURES_ALLOC,
      "struct": L.FEATURES_STRUCT, "state+alloc": L.FEATURES_STATE + RL.FEATURES_ALLOC,
      "full": L.FEATURES_STATE + L.FEATURES_EWS + L.FEATURES_STRUCT}


def model():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))


def calib(y, p):
    """Calibration-in-the-large (intercept with slope fixed at 1, via offset logistic) and slope."""
    import statsmodels.api as sm
    lp = logit(p)
    m1 = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=lp).fit()
    m2 = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit()
    return dict(mean_pred=float(p.mean()), obs=float(y.mean()), citl=float(m1.params[0]), slope=float(m2.params[1]),
                brier=float(np.mean((p - y) ** 2)), brier_null=float(y.mean() * (1 - y.mean())))


res = {}
for regime in ["early", "all"]:
    lh = (LH if regime == "all" else LH[LH.pool_now >= 100]).reset_index(drop=True)
    lc = (LC if regime == "all" else LC[LC.pool_now >= 100]).reset_index(drop=True)
    R = {"clone_train": dict(landmarks=int(len(lc)), pos=int(lc.y.sum()), pos_games=int(lc[lc.y == 1].game.nunique()),
                             bc1_landmarks=int((lc.bc == 1).sum()), bc1_pos=int(lc[lc.bc == 1].y.sum())),
         "human": dict(landmarks=int(len(lh)), pos=int(lh.y.sum()), pos_games=int(lh[lh.y == 1].game.nunique()))}
    scores = {}
    for src, lcs in [("all", lc), ("bc1", lc[lc.bc == 1])]:
        for k, f in FS.items():
            scores[f"{src}:{k}"] = model().fit(lcs[f], lcs.y).predict_proba(lh[f])[:, 1]
    S = pd.concat([lh[["game", "group", "t", "y", "mech", "exp", "episode", "tau", "pool_now"]], pd.DataFrame(scores)], axis=1)
    y = S.y.to_numpy(); w = RL.game_weights(S)
    subsets = {"all": np.ones(len(S), bool), "Exp3-4 (clone-independent)": (S.exp >= 3).to_numpy(), "Exp1-2": (S.exp <= 2).to_numpy()}
    R["auc"] = {}
    for sname, m in subsets.items():
        Ssub = S[m]
        R["auc"][sname] = {}
        for k in scores:
            def st(q, k=k):
                return dict(landmark=RL.wauc(q.y, q[k]), game=RL.wauc(q.y, q[k], RL.game_weights(q)), inst=RL.strat_auc(q.y, q[k], q.mech.to_numpy()))
            e = st(Ssub)
            B = a.boot if (sname == "all" or k.startswith("all:")) else 300
            bs = RL.cluster_boot(Ssub, st, B, rng)
            R["auc"][sname][k] = {m_: dict(est=e[m_], lo=RL.ci([b[m_] for b in bs])[0], hi=RL.ci([b[m_] for b in bs])[1]) for m_ in e}
            R["auc"][sname][k]["n"] = int(len(Ssub)); R["auc"][sname][k]["pos"] = int(Ssub.y.sum())
            R["auc"][sname][k]["pos_games"] = int(Ssub[Ssub.y == 1].game.nunique())
    R["by_mech"] = {mech: {k: RL.wauc(x.y, x[k]) for k in scores} | {"pos": int(x.y.sum()), "pos_games": int(x[x.y == 1].game.nunique()), "n": int(len(x))}
                    for mech, x in S.groupby("mech") if x.y.sum() >= 3}
    # calibration of clone-trained models
    R["calibration"] = {k: calib(y, S[k].to_numpy()) for k in scores if k.startswith("all:")}
    R["calibration_by_mech"] = {mech: {k: dict(mean_pred=float(x[k].mean()), obs=float(x.y.mean())) for k in ["all:state", "all:full", "all:alloc"]}
                                for mech, x in S.groupby("mech")}
    R["reliability"] = {}
    for k in ["all:state", "all:full", "all:alloc"]:
        q = pd.qcut(S[k].rank(method="first"), 10, labels=False)
        R["reliability"][k] = [dict(bin=int(b), mean_pred=float(x[k].mean()), obs=float(x.y.mean()), n=int(len(x))) for b, x in S.groupby(q)]
    # group-cross-fitted intercept recalibration on human data (5 folds by group)
    ug = S.group.astype(str).unique(); fold = dict(zip(rng.permutation(ug), np.arange(len(ug)) % 5))
    f = S.group.astype(str).map(fold).to_numpy()
    rec = {}
    import statsmodels.api as sm
    for k in ["all:state", "all:full", "all:alloc"]:
        lp = logit(S[k].to_numpy()); pr = np.zeros(len(S))
        for j in range(5):
            tr, te = f != j, f == j
            b0 = sm.GLM(y[tr], np.ones((tr.sum(), 1)), family=sm.families.Binomial(), offset=lp[tr]).fit().params[0]
            pr[te] = 1 / (1 + np.exp(-(lp[te] + b0)))
        rec[k] = calib(y, pr)
    R["recalibrated_intercept_crossfit"] = rec
    # alert rule: threshold on the clone-trained allocation model fixed at the clone early-regime 80% sensitivity point
    pc = model().fit(lc[FS["alloc"]], lc.y).predict_proba(lc[FS["alloc"]])[:, 1]
    thr = float(np.quantile(pc[lc.y.to_numpy() == 1], 0.20))
    al = S["all:alloc"].to_numpy() >= thr
    tp = int((al & (y == 1)).sum()); fp = int((al & (y == 0)).sum()); fn = int((~al & (y == 1)).sum()); tn = int((~al & (y == 0)).sum())
    # game-level: collapsed games alerted before collapse (first alert lead), surviving games with any alert
    g = S.assign(al=al)
    coll = g[g.tau.notna()]; surv = g[g.tau.isna()]
    first = coll[coll.al].groupby("game").agg(t=("t", "min"), tau=("tau", "first"))
    lead = (first.tau - first.t).to_numpy()
    R["alert"] = dict(threshold=thr, sensitivity=tp / max(tp + fn, 1), specificity=tn / max(tn + fp, 1), ppv=tp / max(tp + fp, 1),
                      alert_rate=float(al.mean()), collapsed_games=int(coll.game.nunique()), collapsed_games_alerted=int(first.shape[0]),
                      median_lead=float(np.median(lead)) if len(lead) else np.nan, lead_q25=float(np.percentile(lead, 25)) if len(lead) else np.nan,
                      lead_q75=float(np.percentile(lead, 75)) if len(lead) else np.nan,
                      surviving_games=int(surv.game.nunique()), surviving_games_alerted=int(surv[surv.al].game.nunique()))
    res[regime] = R
    S.to_csv(f"{a.res}/transfer_scores_{regime}.csv.gz", index=False, compression="gzip")
json.dump(res, open(f"{a.res}/transfer_revision.json", "w"), indent=1, default=float)
for rg in res:
    print(rg, {s: {k: round(v["landmark"]["est"], 3) for k, v in res[rg]["auc"][s].items() if k.startswith("all:")} for s in res[rg]["auc"]})
    print(rg, "calib", {k: {m: round(v[m], 3) for m in ["mean_pred", "obs", "citl", "slope"]} for k, v in res[rg]["calibration"].items()})
    print(rg, "alert", res[rg]["alert"])
