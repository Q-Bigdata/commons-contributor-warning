"""Revision analysis 1: landmark prediction under alternative eligibility, weighting and subsets.

Builds extended landmarks for humans and clones (saved for the transfer analysis), then evaluates
cross-validated feature sets under: primary eligibility; full-horizon (outcome-independent) eligibility;
subsets excluding Experiment 4, using only the first Experiment-4 game per group, and using only the
clone-independent Experiments 3-4. Metrics: landmark-weighted AUC (primary estimand), game-weighted AUC,
institution-stratified AUC and landmark-round-stratified AUC, with group cluster bootstrap holding the
out-of-fold predictions fixed (paired resampling across models).
usage: python rev_predict.py --out OUT --boot 1000 --procs 32
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
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import ews_lib as L
import rev_lib as RL

p = argparse.ArgumentParser()
p.add_argument("--out", required=True); p.add_argument("--boot", type=int, default=1000)
p.add_argument("--procs", type=int, default=32)
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)

FS = {"state": L.FEATURES_STATE, "ews": L.FEATURES_EWS, "state+ews": L.FEATURES_STATE + L.FEATURES_EWS,
      "alloc": RL.FEATURES_ALLOC, "ret": RL.FEATURES_RET, "struct": L.FEATURES_STRUCT,
      "state+alloc": L.FEATURES_STATE + RL.FEATURES_ALLOC,
      "state+alloc+ews": L.FEATURES_STATE + RL.FEATURES_ALLOC + L.FEATURES_EWS,
      "state+struct": L.FEATURES_STATE + L.FEATURES_STRUCT,
      "full": L.FEATURES_STATE + L.FEATURES_EWS + L.FEATURES_STRUCT}
UNI = ["var_pool", "ar1_pool", "var_logpool", "var_rho", "ar1_rho", "logR", "mean_logG", "gini_mean", "active_mean",
       "neff_alloc_mean", "npos_mean", "neff_ret_mean", "gini_ret_mean", "disp_slope"]
TBINS = [6, 12, 18, 24, 30, 60]


def model():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))


def oof(df, feats, seed, reps=5):
    rng = np.random.default_rng(seed)
    g = df.group.astype(str).to_numpy(); ug = np.unique(g); y = df.y.to_numpy()
    pred = np.zeros(len(df))
    for r in range(reps):
        fold = dict(zip(rng.permutation(ug), np.arange(len(ug)) % 5)); f = np.array([fold[k] for k in g])
        for k in range(5):
            tr, te = f != k, f == k
            pred[te] += model().fit(df.loc[tr, feats], y[tr]).predict_proba(df.loc[te, feats])[:, 1] / reps
    return pred


def metrics(d, cols):
    y = d.y.to_numpy(); w = RL.game_weights(d); tb = np.digitize(d.t, TBINS)
    out = {}
    for c in cols:
        s = d[c].to_numpy()
        out[c] = dict(landmark=RL.wauc(y, s), game=RL.wauc(y, s, w), inst=RL.strat_auc(y, s, d.mech.to_numpy()),
                      time=RL.strat_auc(y, s, tb))
    return out


def counts(d):
    pos = d[d.y == 1]
    return dict(landmarks=int(len(d)), pos_landmarks=int(len(pos)), games=int(d.game.nunique()), groups=int(d.group.nunique()),
                pos_games=int(pos.game.nunique()), pos_groups=int(pos.group.nunique()),
                by_mech={m: dict(landmarks=int(len(x)), pos_landmarks=int(x.y.sum()), pos_games=int(x[x.y == 1].game.nunique()),
                                 pos_groups=int(x[x.y == 1].group.nunique()), games=int(x.game.nunique()))
                         for m, x in d.groupby("mech")})


def run_variant(args):
    name, d, B, seed = args
    d = d.reset_index(drop=True)
    preds = pd.DataFrame({k: oof(d, v, seed + i) for i, (k, v) in enumerate(FS.items())})
    P = pd.concat([d[["game", "group", "t", "y", "mech", "exp"] + UNI], preds.add_prefix("p_")], axis=1)
    cols = [f"p_{k}" for k in FS] + UNI
    est = metrics(P, cols)
    for c in ["active_mean", "neff_alloc_mean", "npos_mean", "neff_ret_mean"]:  # fewer -> riskier: report reversed
        for k in est[c]:
            est[c][k] = 1 - est[c][k] if est[c][k] == est[c][k] else est[c][k]
    rng = np.random.default_rng(seed + 99)
    key = ["p_state", "p_ews", "p_state+ews", "p_alloc", "p_ret", "p_struct", "p_state+alloc", "p_state+alloc+ews", "p_full", "p_state+struct",
           "var_pool", "ar1_pool", "var_rho", "gini_mean", "active_mean", "neff_alloc_mean", "npos_mean", "gini_ret_mean"]
    bs = RL.cluster_boot(P, lambda q: metrics(q, key), B, rng)
    res = {"counts": counts(d), "est": est, "ci": {}}
    for c in key:
        res["ci"][c] = {}
        for m in ["landmark", "game", "inst", "time"]:
            v = np.array([b[c][m] for b in bs])
            if c in ["active_mean", "neff_alloc_mean", "npos_mean"]:
                v = 1 - v
            res["ci"][c][m] = RL.ci(v)
    for c1, c0 in [("p_state+ews", "p_state"), ("p_state+alloc", "p_state"), ("p_state+alloc+ews", "p_state+alloc"),
                   ("p_alloc", "p_ret"), ("p_full", "p_state+struct"), ("p_alloc", "p_struct")]:
        res["ci"][f"{c1}-{c0}"] = {}
        for m in ["landmark", "game", "inst", "time"]:
            v = np.array([b[c1][m] - b[c0][m] for b in bs])
            res["ci"][f"{c1}-{c0}"][m] = dict(est=est[c1][m] - est[c0][m], lo=RL.ci(v)[0], hi=RL.ci(v)[1],
                                              p_le0=float(np.nanmean(v <= 0)))
    # AUC by landmark round bin (descriptive)
    tb = np.digitize(P.t, TBINS)
    res["by_round"] = {int(b): {c: RL.wauc(x.y, x[c]) for c in ["p_state", "p_ews", "p_alloc", "p_state+alloc"]} | {"n": int(len(x)), "pos": int(x.y.sum())}
                       for b, x in P.groupby(tb)}
    # univariate AUC by institution
    res["by_mech_uni"] = {m: {c: (RL.wauc(x.y, x[c]) if c not in ["active_mean", "neff_alloc_mean", "npos_mean", "neff_ret_mean"]
                                  else 1 - RL.wauc(x.y, x[c])) for c in UNI} for m, x in P.groupby("mech") if x.y.sum() >= 5}
    P.to_csv(f"{a.out}/oof_{name}.csv.gz", index=False, compression="gzip")
    return name, res


if __name__ == "__main__":
    R = pd.read_csv("data/rounds.csv.gz")
    H = RL.prepare_ext(R[R.source == "human"]); C = RL.prepare_ext(R[R.source == "clone"])
    H.to_csv(f"{a.out}/rounds_ext_human.csv.gz", index=False, compression="gzip")
    LH = RL.landmarks_ext(H); LC = RL.landmarks_ext(C)
    LH.to_csv(f"{a.out}/landmarks_ext_human.csv.gz", index=False, compression="gzip")
    LC.to_csv(f"{a.out}/landmarks_ext_clone.csv.gz", index=False, compression="gzip")
    print("landmarks", LH.shape, LC.shape, flush=True)
    E = LH[LH.pool_now >= 100]
    V = {"early_primary": E, "all_primary": LH,
         "early_fullhorizon": E[E.full_horizon], "all_fullhorizon": LH[LH.full_horizon],
         "early_noExp4": E[E.exp != 4], "early_Exp4first": E[(E.exp != 4) | (E.episode == 0)],
         "early_Exp34": E[E.exp >= 3], "early_Exp12": E[E.exp <= 2]}
    jobs = [(k, v, a.boot if k in ("early_primary", "early_fullhorizon", "all_primary") else max(300, a.boot // 2), 1000 + i)
            for i, (k, v) in enumerate(V.items())]
    with (Pool(min(a.procs, len(jobs))) if a.procs > 1 else _Dummy()) as pool:
        out = dict(pool.map(run_variant, jobs))
    json.dump(out, open(f"{a.out}/predict_variants.json", "w"), indent=1, default=float)
    for k, v in out.items():
        c = v["counts"]; e = v["est"]
        print(k, c["landmarks"], c["pos_landmarks"], c["pos_games"], c["pos_groups"],
              {m: round(e[m]["landmark"], 3) for m in ["p_state", "p_ews", "p_state+ews", "p_alloc", "p_ret", "p_full"]}, flush=True)
