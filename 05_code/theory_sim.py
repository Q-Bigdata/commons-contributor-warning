"""Hard versus soft saturation: why a capped commons loses its sustainable state without critical slowing down.

Map: R_{t+1} = F(G_t R_t), G_t = (1+r) * (rho_bar + sigma * eps_t), r = 0.4, K = 200.
  hard cap (the human game):   F(x) = min(K, x)
  soft cap (Beverton-Holt):    R_{t+1} = G R / (1 + (G - 1) R / K)   (same carrying capacity; smooth)
Linearisation at the full-pool state: hard cap multiplier 0 (superstable; any deficit d < K(1 - 1/G) is erased
in one round); soft cap multiplier 1/G -> 1 as G -> 1 (critical slowing down).
Outputs: theory_stationary.csv (indicators versus rho_bar), theory_forcing.csv (Kendall trends under slow forcing).
"""
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

rng = np.random.default_rng(3)
K, r = 200.0, 0.4


def step(R, G, kind):
    if kind == "hard":
        return np.minimum(K, G * R)
    return G * R / (1 + (G - 1) * R / K)


def ar1(x):
    x = x - x.mean()
    d = (x[:-1] ** 2).sum()
    return float((x[:-1] * x[1:]).sum() / d) if d > 0 else np.nan


rows = []
for rho_bar in np.round(np.arange(0.715, 0.8201, 0.005), 3):
    for kind, sigma in [("hard", 0.02), ("hard", 0.05), ("hard", 0.10), ("hard", 0.20)]:
        n, T = 400, 400
        R = np.full(n, K)
        rec = []
        for t in range(T):
            G = (1 + r) * np.clip(rho_bar + sigma * rng.standard_normal(n), 0, 1)
            R = step(R, G, kind)
            R = np.where(R < 1, K, R)  # restart collapsed runs so the ensemble stays stationary
            if t >= 100:
                rec.append(R.copy())
        X = np.array(rec)  # time x runs
        a = np.nanmean([ar1(X[:, i]) for i in range(n)])
        v = float(X.var(0).mean())
        rows.append(dict(kind=kind, sigma=sigma, rho_bar=rho_bar, G=(1 + r) * rho_bar, ar1=a, var=v,
                         frac_at_cap=float((X >= K - 1e-9).mean()), mean_R=float(X.mean()),
                         lambda_theory=(0.0 if kind == "hard" else 1 / ((1 + r) * rho_bar))))
pd.DataFrame(rows).to_csv("theory_stationary.csv", index=False)

# slow forcing through the threshold: rho_bar declines linearly from 0.80 to 0.66 over 60 rounds
fr = []
T = 60
rb = np.linspace(0.80, 0.66, T)
cross = int(np.argmax(rb < 1 / (1 + r)))
for kind, sigma in [("hard", 0.02), ("hard", 0.05), ("hard", 0.10), ("hard", 0.20)]:
    n = 1000
    R = np.full(n, K)
    traj = np.zeros((T, n))
    for t in range(T):
        G = (1 + r) * np.clip(rb[t] + sigma * rng.standard_normal(n), 0, 1)
        R = step(R, G, kind)
        traj[t] = R
    seg = traj[:cross]  # before the deterministic threshold crossing
    W = cross // 2
    for i in range(n):
        s = seg[:, i]
        v = [np.var(s[j - W + 1:j + 1] - np.polyval(np.polyfit(np.arange(W), s[j - W + 1:j + 1], 1), np.arange(W)))
             for j in range(W - 1, len(s))]
        a = []
        for j in range(W - 1, len(s)):
            w = s[j - W + 1:j + 1]
            w = w - np.polyval(np.polyfit(np.arange(W), w, 1), np.arange(W))
            a.append(ar1(w))
        kv = kendalltau(np.arange(len(v)), v)[0] if np.std(v) > 0 else np.nan
        ka = kendalltau(np.arange(len(a)), a)[0] if np.nanstd(a) > 0 else np.nan
        fr.append(dict(kind=kind, sigma=sigma, run=i, kt_var=kv, kt_ar1=ka, collapsed=bool((traj[:, i] < 1).any()),
                       frac_at_cap_pre=float((seg[:, i] >= K - 1e-9).mean())))
F = pd.DataFrame(fr)
F.to_csv("theory_forcing.csv", index=False)
print(pd.DataFrame(rows).pivot(index="rho_bar", columns="sigma", values=["ar1", "var", "frac_at_cap"]).round(3).to_string())
print(F.groupby("sigma")[["kt_var", "kt_ar1", "frac_at_cap_pre"]].agg(["median", lambda x: np.nanmean(x > 0)]).round(3))
print("threshold crossing round", cross)
