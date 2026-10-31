"""Revision analysis 2: does allocation concentration go with larger fluctuations of the commons?

Uses rounds_ext_human.csv.gz from rev_predict.py. Plateau rounds: start-of-round pool in [25, 190), E >= 1.
(1) Variance of the one-round change in log pool by number of allocation recipients: rounds, games, groups,
    mean, variance, cluster-bootstrap CI; pooled, by institution, uncapped vs capped, excluding terminal jumps.
(2) Residual variance: residuals of dx after removing institution x experiment x pool-bin means; squared
    residuals regressed on log n_eff (allocation shares) with institution x pool-bin fixed effects, distributed
    fraction q and lagged dx; game-clustered SEs (FWL). Same for the NEXT round's squared residual (temporal order),
    and for the mean (drift) of dx.
(3) Return-share variance: Var(rho) by n_eff bin versus the homogeneous prediction v[(1-c)/n_eff + c].
(4) Cross-tabulation of allocation recipients versus positive-return contributors.
(5) Clones: same table (1).
usage: python rev_concentration.py --res OUT --boot 2000
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import rev_lib as RL

p = argparse.ArgumentParser(); p.add_argument("--res", required=True); p.add_argument("--boot", type=int, default=2000)
a = p.parse_args()
rng = np.random.default_rng(31)
H = pd.read_csv(f"{a.res}/rounds_ext_human.csv.gz")
R = pd.read_csv("data/rounds.csv.gz")
C = RL.prepare_ext(R[R.source == "clone"])
EDGES = np.array([25, 50, 100, 150, 190])


def plateau(D):
    P = D[(D.pool >= 25) & (D.pool < 190) & (D.E >= 1)].copy()
    P["pbin"] = np.digitize(P.pool, EDGES)
    P["terminal"] = P.next_pool < 1
    return P


def var_table(P, col="dx", by=None, B=2000):
    rows = []
    groups = [("pooled", P)] if by is None else list(P.groupby(by))
    for lab, Q in groups:
        for k, x in Q.groupby("active"):
            if k < 1 or len(x) < 10:
                continue
            pos = x.groupby("group").indices; keys = list(pos.keys())
            bv = []
            for _ in range(B):
                pick = rng.integers(0, len(keys), len(keys))
                v = x[col].to_numpy()[np.concatenate([pos[keys[i]] for i in pick])]
                bv.append(v.var(ddof=1))
            rows.append(dict(scope=lab, recipients=int(k), rounds=len(x), games=x.game.nunique(), groups=x.group.nunique(),
                             mean=x[col].mean(), var=x[col].var(ddof=1), lo=np.percentile(bv, 2.5), hi=np.percentile(bv, 97.5),
                             terminal=int(x.terminal.sum()),
                             share_prop=float((x.mech == "Proportional").mean())))
    return pd.DataFrame(rows)


def fwl(y, x, cells, controls, cluster):
    """OLS slope of y on x absorbing cell fixed effects and linear controls; cluster-robust SE."""
    df = pd.DataFrame({"y": y, "x": x, "cell": cells, "cl": cluster})
    for i, c in enumerate(controls.T):
        df[f"c{i}"] = c
    df = df.dropna()
    cols = ["y", "x"] + [c for c in df.columns if c.startswith("c") and c not in ("cell", "cl")]
    Z = df[cols] - df.groupby("cell")[cols].transform("mean")
    Xm = Z[[c for c in cols if c != "y"]].to_numpy(); Y = Z["y"].to_numpy()
    beta = np.linalg.lstsq(Xm, Y, rcond=None)[0]
    e = Y - Xm @ beta
    XtX_inv = np.linalg.inv(Xm.T @ Xm)
    S = np.zeros((Xm.shape[1], Xm.shape[1]))
    for _, idx in df.groupby("cl").indices.items():
        sc = Xm[idx].T @ e[idx]; S += np.outer(sc, sc)
    V = XtX_inv @ S @ XtX_inv
    return dict(beta=float(beta[0]), se=float(np.sqrt(V[0, 0])), n=int(len(df)), clusters=int(df.cl.nunique()))


out = {}
PH = plateau(H)
T1 = var_table(PH, "dx", None, a.boot); T1["series"] = "capped dx, all"
T2 = var_table(PH[~PH.terminal], "dx", None, a.boot); T2["series"] = "capped dx, excl. terminal rounds"
T3 = var_table(PH, "logG", None, a.boot); T3["series"] = "uncapped log growth"
T4 = var_table(PH, "dx", "mech", max(500, a.boot // 4)); T4["series"] = "capped dx, by institution"
PC = plateau(C)
T5 = var_table(PC, "dx", None, max(500, a.boot // 4)); T5["series"] = "clones, capped dx"
T6 = var_table(PC, "dx", "mech", 300); T6["series"] = "clones, by institution"
TV = pd.concat([T1, T2, T3, T4, T5, T6]); TV.to_csv(f"{a.res}/variance_by_recipients.csv", index=False)

# residual analysis (humans)
PH = PH.sort_values(["game", "round"]).copy()
PH["cell"] = PH.mech + "|" + PH.exp.astype(str) + "|" + PH.pbin.astype(str)
PH["resid"] = PH.dx - PH.groupby("cell").dx.transform("mean")
PH["r2"] = PH.resid ** 2
PH["lneff"] = np.log(PH.neff_alloc)
H2 = H.sort_values(["game", "round"]).copy()
H2["dx_lag"] = H2.groupby("game").dx.shift(1)
H2["dx_next"] = H2.groupby("game").dx.shift(-1)
H2["pool_next_start"] = H2.groupby("game").pool.shift(-1)
PH = PH.merge(H2[["game", "round", "dx_lag", "dx_next", "pool_next_start"]], on=["game", "round"], how="left")
PH["nbin"] = np.digitize(PH.pool_next_start.fillna(-1), EDGES)
PH["cell_next"] = PH.mech + "|" + PH.exp.astype(str) + "|" + PH.nbin.astype(str)
PH["resid_next"] = PH.dx_next - PH.groupby("cell_next").dx_next.transform("mean")
PH["r2_next"] = PH.resid_next ** 2
ctrl = np.c_[PH.q.to_numpy(), PH.dx_lag.to_numpy()]
reg = {}
for sub, Q in [("all", PH), ("no_terminal", PH[~PH.terminal]), ("contingent", PH[PH.mclass == "contingent"]),
               ("Proportional", PH[PH.mech == "Proportional"]), ("RL", PH[PH.mech.isin(["RL-M1", "RL-M2"])]),
               ("Interpolating", PH[PH.mech == "Interpolating"])]:
    c = np.c_[Q.q.to_numpy(), Q.dx_lag.to_numpy()]
    reg[sub] = dict(sq_resid_same=fwl(Q.r2.to_numpy(), Q.lneff.to_numpy(), Q.cell.to_numpy(), c, Q.group.to_numpy()),
                    sq_resid_next=fwl(Q.r2_next.to_numpy(), Q.lneff.to_numpy(), Q.cell.to_numpy(), c, Q.group.to_numpy()),
                    abs_resid_same=fwl(np.abs(Q.resid.to_numpy()), Q.lneff.to_numpy(), Q.cell.to_numpy(), c, Q.group.to_numpy()),
                    drift_same=fwl(Q.dx.to_numpy(), Q.lneff.to_numpy(), Q.cell.to_numpy(), c, Q.group.to_numpy()),
                    sq_resid_recipients=fwl(Q.r2.to_numpy(), -Q.active.to_numpy().astype(float), Q.cell.to_numpy(), c, Q.group.to_numpy()))
out["regressions"] = reg

# return-ratio variance by n_eff bin, and the homogeneous-diversification prediction
PH["nb"] = pd.cut(PH.neff_alloc, [0.99, 1.5, 2.5, 3.5, 4.01], labels=["~1", "~2", "~3", "~4"])
E = H[["e0", "e1", "e2", "e3"]].to_numpy(); Cc = H[["c0", "c1", "c2", "c3"]].to_numpy()
with np.errstate(all="ignore"):
    F = np.where(E >= 1, Cc / np.maximum(E, 1e-9), np.nan)
Hf = pd.DataFrame(F, columns=["f0", "f1", "f2", "f3"], index=H.index)
Hf["game"] = H.game.values; Hf["round"] = H["round"].values
PHf = PH.merge(Hf, on=["game", "round"], how="left")
fr = PHf[["f0", "f1", "f2", "f3"]].to_numpy()
v_ind = np.nanmean(np.nanvar(fr, axis=1, ddof=0)[np.sum(np.isfinite(fr), 1) >= 2])
rows = []
for k, x in PHf.groupby("nb", observed=True):
    rows.append(dict(neff_bin=str(k), rounds=len(x), mean_neff=x.neff_alloc.mean(), var_rho=x.rho.var(ddof=1),
                     var_rho_resid=(x.rho - x.groupby("cell").rho.transform("mean")).var(ddof=1)))
out["rho_by_neff"] = rows
out["within_round_return_fraction_variance"] = float(v_ind)
# crosstab recipients x positive-return contributors (all human rounds with E>=1)
Hx = H[H.E >= 1]
out["crosstab_recipients_positive"] = pd.crosstab(Hx.active, Hx.npos).to_dict()
out["corr_measures_plateau"] = PH[["active", "neff_alloc", "gini", "npos", "neff_ret"]].corr(method="spearman").round(3).to_dict()
out["retention_by_mech_bin"] = PH.groupby(["mech", "pbin"]).q.agg(["mean", "median", lambda s: float((s < 0.99).mean())]).round(3).reset_index().to_dict("records")
json.dump(out, open(f"{a.res}/concentration.json", "w"), indent=1, default=str)
print(TV[TV.series.isin(["capped dx, all", "capped dx, excl. terminal rounds", "uncapped log growth"])].round(3).to_string())
print(json.dumps(reg, indent=1)[:3000])
