"""Core library for the early-warning analysis of human commons collapse.

All quantities follow the game of Koster, Pislar et al. (2025, Nat Commun 16:2824):
R_{t+1} = min(200, R_t - sum_i e_{i,t} + 1.4 * sum_i c_{i,t}); the game is lost when the pool
falls below 1. Round t is indexed from 0. A landmark at round t uses information up to and including
the reciprocations of round t (i.e. the pool entering round t+1 is known).
"""
import numpy as np
import pandas as pd

CAP = 200.0
GROWTH = 1.4
COLLAPSE_LEVEL = 1.0

MECH_CLASS = {"Equal": "unconditional", "Mixed": "unconditional",
              "Proportional": "contingent", "Interpolating": "contingent",
              "RL-M1": "contingent", "RL-M2": "contingent"}


def gini4(x):
    x = np.asarray(x, float)
    s = x.sum()
    if s <= 0:
        return np.nan
    diff = np.abs(x[:, None] - x[None, :]).sum()
    return diff / (2 * len(x) * s)


def prepare(rounds: pd.DataFrame) -> pd.DataFrame:
    """Per-round derived variables, truncated at collapse (collapse round kept, later rows dropped)."""
    D = rounds.sort_values(["game", "round"]).copy()
    E = D[[f"e{i}" for i in range(4)]].to_numpy()
    C = D[[f"c{i}" for i in range(4)]].to_numpy()
    D["E"] = E.sum(1)
    D["C"] = C.sum(1)
    D["rho"] = np.where(D.E >= 1, D.C / np.maximum(D.E, 1e-9), np.nan)
    uncapped = D["pool"] - D["E"] + GROWTH * D["C"]
    D["logG"] = np.log(np.maximum(uncapped, 1e-3) / np.maximum(D["pool"], 1e-3))
    frac = np.where(E >= 1, C / np.maximum(E, 1e-9), np.nan)
    with np.errstate(all="ignore"):
        D["disp"] = np.nanstd(frac, axis=1)
    D.loc[(E >= 1).sum(1) < 2, "disp"] = np.nan
    D["active"] = (E >= 1).sum(1)
    D["gini"] = [gini4(r) for r in E]
    D["retain"] = 1 - D["E"] / np.maximum(D["pool"], 1e-9)
    D["logR_next"] = np.log(np.maximum(D["next_pool"], 0.01))
    # collapse round tau: first round whose END pool (next_pool) < 1
    keep, taus, lens = [], {}, {}
    for g, x in D.groupby("game", sort=False):
        idx = np.where(x["next_pool"].to_numpy() < COLLAPSE_LEVEL)[0]
        if len(idx):
            tau = int(x["round"].to_numpy()[idx[0]])
            taus[g] = tau
            keep.append(x.index[x["round"] <= tau])
        else:
            taus[g] = np.nan
            keep.append(x.index)
        lens[g] = int(x["round"].max()) + 1
    D = D.loc[np.concatenate(keep)].copy()
    D["tau"] = D["game"].map(taus)
    D["T"] = D["game"].map(lens)
    D["mclass"] = D["mech"].map(MECH_CLASS)
    return D


def _detrended(y):
    t = np.arange(len(y), dtype=float)
    ok = np.isfinite(y)
    if ok.sum() < 4:
        return None
    b = np.polyfit(t[ok], y[ok], 1)
    r = y - (b[0] * t + b[1])
    r[~ok] = np.nan
    return r


def _var_ar1(y):
    r = _detrended(np.asarray(y, float))
    if r is None:
        return np.nan, np.nan
    ok = np.isfinite(r)
    v = np.nanvar(r)
    a, b = r[:-1], r[1:]
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or np.nanstd(a[m]) < 1e-9 or np.nanstd(b[m]) < 1e-9:
        return v, np.nan
    return v, float(np.corrcoef(a[m], b[m])[0, 1])


def _slope(y):
    y = np.asarray(y, float)
    ok = np.isfinite(y)
    if ok.sum() < 3:
        return np.nan
    t = np.arange(len(y), dtype=float)
    return float(np.polyfit(t[ok], y[ok], 1)[0])


FEATURES_STATE = ["logR", "mean_logG", "last3_logG"]
FEATURES_EWS = ["var_pool", "ar1_pool", "var_logpool", "ar1_logpool", "var_rho", "ar1_rho"]
FEATURES_STRUCT = ["disp_mean", "disp_slope", "active_mean", "gini_mean", "cap_frac", "flicker", "retain_mean"]


def landmarks(D: pd.DataFrame, W: int = 8, h: int = 10) -> pd.DataFrame:
    """One row per (game, landmark round t) in the risk set with complete follow-up."""
    rows = []
    for g, x in D.groupby("game", sort=False):
        x = x.sort_values("round")
        tau = x["tau"].iloc[0]
        T = x["T"].iloc[0]
        r = x["round"].to_numpy()
        pool_next = x["next_pool"].to_numpy()
        logp = np.log(np.maximum(pool_next, 0.01))
        logG = x["logG"].to_numpy(); rho = x["rho"].to_numpy(); disp = x["disp"].to_numpy()
        act = x["active"].to_numpy(); gin = x["gini"].to_numpy(); ret = x["retain"].to_numpy()
        start_pool = x["pool"].to_numpy()
        meta = {k: x[k].iloc[0] for k in ["source", "group", "mech", "mclass", "exp", "bc"]}
        for j in range(W - 1, len(x)):
            t = r[j]
            if np.isfinite(tau) and t >= tau:
                break
            y = int(np.isfinite(tau) and (t < tau <= t + h))
            if not y and (t + h > T - 1):
                continue  # incomplete follow-up without an event
            sl = slice(j - W + 1, j + 1)
            vp, ap = _var_ar1(pool_next[sl])
            vlp, alp = _var_ar1(logp[sl])
            vr, ar = _var_ar1(rho[sl])
            hi = start_pool[sl] >= 190
            rows.append(dict(game=g, t=int(t), y=y, pool_now=float(pool_next[j]), logR=float(logp[j]),
                             mean_logG=float(np.nanmean(logG[sl])), last3_logG=float(np.nanmean(logG[max(0, j - 2):j + 1])),
                             var_pool=vp, ar1_pool=ap, var_logpool=vlp, ar1_logpool=alp, var_rho=vr, ar1_rho=ar,
                             disp_mean=float(np.nanmean(disp[sl])) if np.isfinite(disp[sl]).any() else np.nan,
                             disp_slope=_slope(disp[sl]), active_mean=float(np.mean(act[sl])),
                             gini_mean=float(np.nanmean(gin[sl])) if np.isfinite(gin[sl]).any() else np.nan,
                             cap_frac=float(np.mean(hi)), flicker=int(np.sum(hi[1:] != hi[:-1])),
                             retain_mean=float(np.nanmean(ret[sl])), tau=tau, **meta))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- evaluation helpers
def auc(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    m = np.isfinite(s)
    y, s = y[m], s[m]
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    ranks = pd.Series(s).rank().to_numpy()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def group_boot_idx(groups, rng):
    u = np.unique(groups)
    pick = rng.choice(u, size=len(u), replace=True)
    pos = pd.Series(np.arange(len(groups))).groupby(groups).apply(list)
    return np.concatenate([pos[p] for p in pick])
