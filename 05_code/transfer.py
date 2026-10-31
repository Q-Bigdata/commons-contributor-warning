"""Sim-to-real transfer: early-warning models trained ONLY on behavioural-clone games, evaluated on humans.
Also a 1D convolutional network on raw windows (the deep-learning detector), trained on clones.

usage: python transfer.py --clone DIR --human DIR --out DIR [--cnn 1]
"""
import argparse, json, os, pickle, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
import ews_lib as L

p = argparse.ArgumentParser()
p.add_argument("--clone", required=True)
p.add_argument("--human", required=True)
p.add_argument("--out", required=True)
p.add_argument("--W", type=int, default=8)
p.add_argument("--h", type=int, default=10)
p.add_argument("--boot", type=int, default=1000)
p.add_argument("--cnn", type=int, default=1)
p.add_argument("--early", type=float, default=100.0)
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
rng = np.random.default_rng(5)

LC = pd.read_csv(f"{a.clone}/landmarks_W{a.W}_h{a.h}.csv.gz")
LH = pd.read_csv(f"{a.human}/landmarks_W{a.W}_h{a.h}.csv.gz")


def boot(y, scores, groups, B):
    pos = pd.Series(np.arange(len(groups))).groupby(groups).apply(np.array).to_dict()
    ug = np.array(list(pos.keys()))
    out = {k: [] for k in scores}
    for _ in range(B):
        idx = np.concatenate([pos[g] for g in rng.choice(ug, len(ug))])
        for k, s in scores.items():
            out[k].append(L.auc(y[idx], s[idx]))
    return {k: np.array(v) for k, v in out.items()}


res = {}
full = L.FEATURES_STATE + L.FEATURES_EWS + L.FEATURES_STRUCT
for regime in ["all", "early"]:
    lc = LC if regime == "all" else LC[LC.pool_now >= a.early]
    lh = (LH if regime == "all" else LH[LH.pool_now >= a.early]).reset_index(drop=True)
    F = pickle.load(open(f"{a.clone}/fits_{regime}.pkl", "rb"))
    scores = {f"logit_{k}": F["fits"][k].predict_proba(lh[v])[:, 1] for k, v in F["feats"].items()}
    for name, feats in [("hgb_state", L.FEATURES_STATE), ("hgb_full", full)]:
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                           l2_regularization=1.0, random_state=0).fit(lc[feats], lc.y)
        scores[name] = m.predict_proba(lh[feats])[:, 1]
    y = lh.y.to_numpy(); g = lh.group.astype(str).to_numpy()
    bs = boot(y, scores, g, a.boot)
    r = {}
    for k, s in scores.items():
        lo, hi = np.nanpercentile(bs[k], [2.5, 97.5])
        r[k] = {"est": L.auc(y, s), "lo": float(lo), "hi": float(hi)}
    for k1, k0 in [("logit_state+ews", "logit_state"), ("logit_full", "logit_state"), ("hgb_full", "hgb_state")]:
        d = bs[k1] - bs[k0]
        r[f"delta_{k1}_minus_{k0}"] = {"est": L.auc(y, scores[k1]) - L.auc(y, scores[k0]),
                                       "lo": float(np.nanpercentile(d, 2.5)), "hi": float(np.nanpercentile(d, 97.5))}
    # calibration-in-the-large of the clone-trained state model on humans
    r["mean_pred_state"] = float(scores["logit_state"].mean()); r["obs_rate"] = float(y.mean())
    res[regime] = r
    pd.DataFrame({"game": lh.game, "t": lh.t, "y": y, "mech": lh.mech, **scores}).to_csv(
        f"{a.out}/transfer_scores_{regime}.csv.gz", index=False, compression="gzip")

# ---------------------------------------------------------------- CNN detector on raw windows
if a.cnn:
    import torch, torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    R = pd.read_csv("data/rounds.csv.gz")
    D = L.prepare(R)
    D["logR"] = np.log(np.maximum(D.next_pool, 0.01)) / np.log(200)
    D["cap"] = (D.pool >= 190).astype(float)
    D["act4"] = D.active / 4
    CH = ["logR", "logG", "rho", "disp", "act4", "gini", "cap"]
    D[CH] = D[CH].fillna(0.0)
    arr = {g: x.sort_values("round")[CH].to_numpy(np.float32) for g, x in D.groupby("game")}

    def windows(lm):
        X = np.stack([arr[g][t - a.W + 1:t + 1] for g, t in zip(lm.game, lm.t)])
        return torch.tensor(X.transpose(0, 2, 1))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.c = nn.Sequential(nn.Conv1d(len(CH), 32, 3, padding=1), nn.ReLU(), nn.Conv1d(32, 32, 3, padding=1),
                                   nn.ReLU(), nn.AdaptiveAvgPool1d(1))
            self.l = nn.LSTM(len(CH), 32, batch_first=True)
            self.o = nn.Linear(64, 1)

        def forward(self, x):
            h1 = self.c(x).squeeze(-1)
            _, (h2, _) = self.l(x.transpose(1, 2))
            return self.o(torch.cat([h1, h2[-1]], 1)).squeeze(-1)

    cres = {}
    for regime in ["all", "early"]:
        lc = (LC if regime == "all" else LC[LC.pool_now >= a.early]).reset_index(drop=True)
        lh = (LH if regime == "all" else LH[LH.pool_now >= a.early]).reset_index(drop=True)
        Xc, yc = windows(lc), torch.tensor(lc.y.to_numpy(np.float32))
        Xh, yh = windows(lh), lh.y.to_numpy()
        ug = lc.group.unique()
        preds = []
        for seed in range(5):
            torch.manual_seed(seed)
            val_g = set(np.random.default_rng(seed).choice(ug, len(ug) // 5, replace=False))
            va = lc.group.isin(val_g).to_numpy()
            net = Net().to(dev)
            opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
            pw = torch.tensor((1 - yc[~va].mean()) / yc[~va].mean()).to(dev)
            lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
            Xtr, ytr = Xc[~va].to(dev), yc[~va].to(dev)
            Xva, yva = Xc[va].to(dev), yc[va].numpy()
            best, best_state, bad = -1, None, 0
            for ep in range(60):
                net.train()
                perm = torch.randperm(len(Xtr), device=dev)
                for i in range(0, len(perm), 512):
                    b = perm[i:i + 512]
                    opt.zero_grad(); l = lossf(net(Xtr[b]), ytr[b]); l.backward(); opt.step()
                net.eval()
                with torch.no_grad():
                    av = L.auc(yva, net(Xva).cpu().numpy())
                if av > best:
                    best, bad = av, 0
                    best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= 8:
                        break
            net.load_state_dict(best_state); net.eval()
            with torch.no_grad():
                preds.append(torch.sigmoid(net(Xh.to(dev))).cpu().numpy())
            print(regime, "seed", seed, "val", round(best, 3), "human", round(L.auc(yh, preds[-1]), 3), flush=True)
        ens = np.mean(preds, 0)
        g = lh.group.astype(str).to_numpy()
        bs = boot(yh, {"cnn": ens}, g, a.boot)["cnn"]
        cres[regime] = {"est": L.auc(yh, ens), "lo": float(np.nanpercentile(bs, 2.5)), "hi": float(np.nanpercentile(bs, 97.5)),
                        "seed_aucs": [L.auc(yh, q) for q in preds], "device": dev}
        pd.DataFrame({"game": lh.game, "t": lh.t, "y": yh, "cnn": ens}).to_csv(
            f"{a.out}/cnn_scores_{regime}.csv.gz", index=False, compression="gzip")
    res["cnn"] = cres
json.dump(res, open(f"{a.out}/transfer_results.json", "w"), indent=1)
print(json.dumps(res, indent=1)[:3000])
