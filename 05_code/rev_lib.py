"""Revision helpers: extended allocation/return features, landmark variants, weighted and stratified AUCs.

The primary landmark features are computed exactly as in ews_lib.landmarks; this module adds
(i) allocation-share effective number n_eff = 1/sum(a_i^2), (ii) the number of players making a positive
return (c_i >= 1), (iii) return-share effective number and return Gini, and (iv) bookkeeping needed for
censoring-robust and re-weighted analyses (guaranteed horizon, episode index, terminal flag).
"""
import numpy as np
import pandas as pd
import ews_lib as L

FEATURES_ALLOC = ["active_mean", "gini_mean", "neff_alloc_mean"]
FEATURES_RET = ["npos_mean", "neff_ret_mean", "gini_ret_mean"]


def gini_rows(M):
    M = np.asarray(M, float)
    s = M.sum(1)
    d = np.abs(M[:, :, None] - M[:, None, :]).sum((1, 2))
    with np.errstate(all="ignore"):
        g = d / (2 * M.shape[1] * s)
    g[s <= 0] = np.nan
    return g


def prepare_ext(rounds):
    D = L.prepare(rounds)
    E = D[[f"e{i}" for i in range(4)]].to_numpy()
    C = D[[f"c{i}" for i in range(4)]].to_numpy()
    Es = E.sum(1, keepdims=True)
    Cs = C.sum(1, keepdims=True)
    with np.errstate(all="ignore"):
        a = np.where(Es >= 1, E / Es, np.nan)
        b = np.where(Cs >= 1, C / Cs, np.nan)
        D["neff_alloc"] = 1.0 / np.nansum(a ** 2, 1)
        D["neff_ret"] = 1.0 / np.nansum(b ** 2, 1)
    D.loc[Es[:, 0] < 1, "neff_alloc"] = np.nan
    D.loc[Cs[:, 0] < 1, "neff_ret"] = 0.0
    D["npos"] = (C >= 1).sum(1)
    D["gini_ret"] = gini_rows(C)
    D["q"] = D["E"] / np.maximum(D["pool"], 1e-9)          # distributed fraction
    D["dx"] = np.log(np.maximum(D["next_pool"], 0.01)) - np.log(np.maximum(D["pool"], 0.01))
    D["episode"] = D["game"].astype(str).str.rsplit("_", n=1).str[-1]
    D.loc[D.source != "human", "episode"] = "0"
    D["episode"] = D["episode"].astype(int)
    D["Tg"] = np.where((D.source == "human") & (D.exp == 4), 25, D["T"])  # guaranteed administrative length
    return D


def landmarks_ext(D, W=8, h=10):
    base = L.landmarks(D, W=W, h=h)
    extra = []
    for g, x in D.groupby("game", sort=False):
        x = x.sort_values("round")
        r = x["round"].to_numpy()
        na = x["neff_alloc"].to_numpy(); npz = x["npos"].to_numpy(); nr = x["neff_ret"].to_numpy(); gr = x["gini_ret"].to_numpy()
        for j in range(W - 1, len(x)):
            sl = slice(j - W + 1, j + 1)
            extra.append((g, int(r[j]), np.nanmean(na[sl]) if np.isfinite(na[sl]).any() else np.nan,
                          float(np.mean(npz[sl])), np.nanmean(nr[sl]) if np.isfinite(nr[sl]).any() else np.nan,
                          np.nanmean(gr[sl]) if np.isfinite(gr[sl]).any() else np.nan,
                          na[j], int(x["Tg"].iloc[0]), int(x["episode"].iloc[0]), int(x["T"].iloc[0])))
    X = pd.DataFrame(extra, columns=["game", "t", "neff_alloc_mean", "npos_mean", "neff_ret_mean", "gini_ret_mean",
                                     "neff_alloc_last", "Tg", "episode", "T"])
    out = base.merge(X, on=["game", "t"], how="left")
    out["full_horizon"] = (out["t"] + h <= out["Tg"] - 1)
    return out


# ------------------------------------------------------------------ AUC variants
def wauc(y, s, w=None):
    """Weighted AUC = sum_{i:y=1, j:y=0} w_i w_j [1(s_i>s_j)+0.5*1(s_i=s_j)] / (sum w_pos * sum w_neg)."""
    y = np.asarray(y).astype(int); s = np.asarray(s, float)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    m = np.isfinite(s); y, s, w = y[m], s[m], w[m]
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan
    u, inv = np.unique(s, return_inverse=True)
    wp = np.bincount(inv, weights=w * (y == 1), minlength=len(u))
    wn = np.bincount(inv, weights=w * (y == 0), minlength=len(u))
    below = np.cumsum(wn) - wn
    return float((wp * (below + 0.5 * wn)).sum() / (wp.sum() * wn.sum()))


def strat_auc(y, s, strata, w=None):
    y = np.asarray(y).astype(int); s = np.asarray(s, float); strata = np.asarray(strata)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    num = den = 0.0
    for st in pd.unique(strata):
        k = strata == st
        yk, sk, wk = y[k], s[k], w[k]
        ok = np.isfinite(sk); yk, sk, wk = yk[ok], sk[ok], wk[ok]
        wp, wn = wk[yk == 1].sum(), wk[yk == 0].sum()
        if wp == 0 or wn == 0:
            continue
        num += wauc(yk, sk, wk) * wp * wn; den += wp * wn
    return num / den if den else np.nan


def game_weights(df):
    return 1.0 / df.groupby("game")["t"].transform("size").to_numpy()


def cluster_boot(df, stat, B, rng, group_col="group"):
    """Cluster bootstrap over groups. stat(dfb) -> dict or float."""
    pos = df.groupby(group_col).indices
    keys = np.array(list(pos.keys()), dtype=object)
    out = []
    for _ in range(B):
        pick = rng.integers(0, len(keys), len(keys))
        idx = np.concatenate([pos[keys[i]] for i in pick])
        out.append(stat(df.iloc[idx]))
    return out


def ci(v):
    v = np.asarray(v, float)
    return float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))
