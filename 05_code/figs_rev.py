"""Revised main figures 1-4 and Extended Data figures 1-7 (revision R1).
All inputs are files: original derived data (submission package 06_derived_data) and revision outputs (revres/).
Run from the repository root: python 05_code/figs_rev.py. Output names match the manuscript. Set EWS_FIGURE_DIR to choose an output directory."""
import json, os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

def apply_figure_style(sizes=(7, 6, 5.5)):
    """Self-contained typography; replaces the unavailable external style helper."""
    mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": sizes[0],
                         "axes.labelsize": sizes[1], "xtick.labelsize": sizes[2],
                         "ytick.labelsize": sizes[2], "pdf.fonttype": 42,
                         "ps.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False})


apply_figure_style(sizes=(7, 6, 5.5))
mpl.rcParams.update({"axes.titlesize": 7, "axes.labelsize": 7, "legend.fontsize": 6, "xtick.labelsize": 5.5, "ytick.labelsize": 5.5})
B = "06_derived_data/"; RV = "06_derived_data/revision/"
OUT = os.environ.get("EWS_FIGURE_DIR", "figs_rev"); os.makedirs(OUT, exist_ok=True)
C_COL, C_SUR = "#D55E00", "#0072B2"
C_GEN, C_STA, C_STR, C_RET = "#7F7F7F", "#0072B2", "#009E73", "#CC79A7"
MECH_ORDER = ["Equal", "Mixed", "Proportional", "Interpolating", "RL-M1", "RL-M2"]
MECH_LAB = {"Equal": "Equal", "Mixed": "Mixed", "Proportional": "Proportional", "Interpolating": "Interpolating",
            "RL-M1": "RL planner M1", "RL-M2": "RL planner M2"}
MECH_C = {"Equal": "#999999", "Mixed": "#E69F00", "Proportional": "#CC79A7", "Interpolating": "#56B4E9",
          "RL-M1": "#009E73", "RL-M2": "#0072B2"}
MW = 183 / 25.4


def pl(ax, s, x=-0.18):
    ax.text(x, 1.07, s, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left")


def save(fig, name):
    name = {"ED_Fig4": "SI_Fig1", "ED_Fig5": "ED_Fig4", "ED_Fig6": "ED_Fig5", "ED_Fig7": "ED_Fig6"}.get(name, name)
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def band(ax, df, col, color, label, stat="median"):
    g = df.groupby("k")[col]
    mid = g.median() if stat == "median" else g.mean()
    ax.fill_between(mid.index, g.quantile(0.25), g.quantile(0.75), color=color, alpha=0.18, lw=0)
    ax.plot(mid.index, mid, color=color, lw=1.3, label=label)


H = pd.read_csv(RV + "rounds_ext_human.csv.gz")
AL = pd.read_csv(RV + "aligned_expmatched.csv.gz")
PV = json.load(open(RV + "predict_variants.json"))
TRV = json.load(open(RV + "transfer_revision.json"))
TJ = json.load(open(B + "transfer/transfer_results.json"))
MK = json.load(open(RV + "markov_revision_fixedobs.json"))
CR = pd.read_csv(RV + "crash_by_recipients.csv")
UCI = pd.read_csv(B + "confirm/univariate_auc_ci.csv")
J = json.load(open(B + "human/results.json"))
RT = pd.read_csv(B + "human/retrospective.csv")
FP = pd.read_csv(B + "landscape/markov_first_passage.csv")
KM = pd.read_csv(B + "landscape/kramers_moyal.csv")
PH = pd.read_csv(B + "phase/phase_diagram.csv")
TF = pd.read_csv(B + "theory/theory_forcing.csv")
SM = pd.read_csv(RV + "theory/stationary_margins.csv")
PS = pd.read_csv(RV + "theory/phase_sensitivity.csv")
PNR = pd.read_csv(B + "landscape/point_of_no_return.csv")
UM = pd.read_csv(RV + "univariate_by_institution.csv")
rng = np.random.default_rng(4)

# ================================================================== Figure 1
fig = plt.figure(figsize=(MW, 2.45))
gs = fig.add_gridspec(1, 4, width_ratios=[1.2, 1.2, 1.3, 1.15], wspace=0.8)
axA, axB, axC, axD = [fig.add_subplot(gs[i]) for i in range(4)]
for ax, mech, title in [(axA, "Proportional", "Proportional allocation"), (axB, "RL-M2", "Deep-RL planner M2")]:
    gm = H[H.mech == mech].groupby("game").tau.first()
    for kind, color in [("sur", C_SUR), ("col", C_COL)]:
        ids = gm[gm.isna()].index if kind == "sur" else gm[gm.notna()].index
        for g in rng.choice(ids, min(12, len(ids)), replace=False):
            x = H[H.game == g]
            ax.plot(x["round"] + 1, x.next_pool, color=color, lw=0.6, alpha=0.55)
    ax.set_xlabel("Round"); ax.set_ylim(-5, 240); ax.set_yticks([0, 50, 100, 150, 200]); ax.set_xlim(0, 41); ax.set_title(title, loc="left")
axA.set_ylabel("Pool (end of round)")
axA.text(40, 238, "sustained", color=C_SUR, ha="right", va="top", fontsize=6)
axA.text(40, 222, "collapsed", color=C_COL, ha="right", va="top", fontsize=6)
pl(axA, "a"); pl(axB, "b")
Hf = FP[FP.source == "human"]
for _, r in Hf.iterrows():
    axC.errorbar(r.pred_markov, r.obs, yerr=[[r.obs - r.obs_lo], [r.obs_hi - r.obs]], fmt="o", ms=3.3, color=MECH_C[r.mech],
                 mec="white", mew=0.4, elinewidth=0.8, capsize=0)
    if np.isfinite(r.pred_markov_other_exps):
        axC.plot(r.pred_markov_other_exps, r.obs, marker="D", ms=2.7, mfc="none", mec=MECH_C[r.mech], mew=0.7, ls="none")
axC.plot([0, 1], [0, 1], color="0.6", lw=0.7, ls="--")
axC.set_xlim(-0.03, 1.03); axC.set_ylim(-0.03, 1.05)
axC.set_xlabel("Predicted by pool-state Markov model"); axC.set_ylabel("Observed collapse fraction")
axC.set_title("Pool state reproduces\ncollapse frequencies", loc="left")
for i, m in enumerate(MECH_ORDER):
    axC.text(0.02, 1.0 - 0.075 * i, MECH_LAB[m], color=MECH_C[m], fontsize=5.5, transform=axC.transAxes, va="top")
axC.text(1.0, 0.03, "filled: in-condition\nopen: other experiment", fontsize=5, ha="right", multialignment="right", transform=axC.transAxes, color="0.3")
pl(axC, "c")
band(axD, AL[AL.kind == "matched survivor"], "pool", C_SUR, "matched survivors")
band(axD, AL[AL.kind == "collapsed"], "pool", C_COL, "collapsed")
axD.set_xlim(25.5, -0.5); axD.set_xlabel("Rounds before collapse"); axD.set_ylabel("Pool (start of round)")
axD.set_title("Collapsing groups drain\nover 10-20 rounds", loc="left"); axD.set_ylim(0, 205)
axD.legend(frameon=False, loc="lower left", fontsize=5.5)
pl(axD, "d")
save(fig, "Fig1")

# ================================================================== Figure 2
fig = plt.figure(figsize=(MW, 2.75))
gs = fig.add_gridspec(1, 3, width_ratios=[1.4, 1.15, 1.0], wspace=0.62)
axA, axB, axC = [fig.add_subplot(gs[i]) for i in range(3)]
ep = PV["early_primary"]
feat_lab = [("var_pool", "Variance, pool", C_GEN), ("ar1_pool", "Autocorrelation, pool", C_GEN),
            ("var_logpool", "Variance, log pool", C_GEN), ("ar1_logpool", "Autocorrelation, log pool", C_GEN),
            ("var_rho", "Variance, return ratio", C_GEN), ("ar1_rho", "Autocorrelation, return ratio", C_GEN),
            ("mean_logG", "Mean log growth (state)", C_STA), ("logR", "Log pool (state)", C_STA),
            ("gini_mean", "Allocation Gini", C_STR), ("active_mean", "Allocation recipients*", C_STR),
            ("npos_mean", "Positive-return contributors*", C_RET)]
E = UCI[(UCI.regime == "early") & (UCI.scope == "pooled")].set_index("feature")
for i, (f, lab, col) in enumerate(feat_lab[::-1]):
    if f in E.index:
        est, lo, hi = E.loc[f, "auc"], E.loc[f, "lo"], E.loc[f, "hi"]
        if f == "active_mean":
            est, lo, hi = 1 - est, 1 - hi, 1 - lo
    else:
        est = ep["est"][f]["landmark"]; lo, hi = ep["ci"][f]["landmark"]
    axA.plot([lo, hi], [i, i], color=col, lw=1.2); axA.plot(est, i, "o", color=col, ms=3.3, mec="white", mew=0.4)
axA.axvline(0.5, color="0.5", lw=0.7, ls="--")
axA.set_yticks(range(len(feat_lab))); axA.set_yticklabels([l for _, l, _ in feat_lab[::-1]])
for t, (_, _, col) in zip(axA.get_yticklabels(), feat_lab[::-1]):
    t.set_color(col)
axA.set_xlabel("AUC, collapse within 10 rounds"); axA.set_xlim(0.35, 0.95)
axA.set_title("Pool at least half full\n(296 positive landmarks, 40 games)", loc="left")
pl(axA, "a", -0.62)
models = [("p_state", "State"), ("p_state+ews", "State + generic"), ("p_alloc", "Allocation"), ("p_ret", "Returns"),
          ("p_full", "All")]
cols = [C_STA, "#56B4E9", C_STR, C_RET, "#000000"]
wts = [("landmark", "o", "landmarks"), ("game", "s", "games equally weighted"), ("inst", "^", "within institution")]
for j, (w, mk, lab) in enumerate(wts):
    for i, ((k, _), col) in enumerate(zip(models, cols)):
        e = ep["est"][k][w]; lo, hi = ep["ci"][k][w]
        axB.errorbar(i + (j - 1) * 0.22, e, yerr=[[e - lo], [hi - e]], fmt=mk, ms=3.0, color=col,
                     mfc=col if j == 0 else "white", elinewidth=0.7, capsize=0)
for w, mk, lab in wts:
    axB.plot([], [], mk, color="0.25", mfc="0.25" if w == "landmark" else "white", ms=3, label=lab)
axB.axhline(0.5, color="0.5", lw=0.7, ls="--")
axB.set_xticks(range(len(models))); axB.set_xticklabels([l for _, l in models], rotation=35, ha="right")
axB.set_ylabel("Cross-validated AUC"); axB.set_ylim(0.3, 1.0)
axB.set_title("Concentration, not generic\nindicators, adds discrimination", loc="left")
axB.legend(frameon=False, loc="lower right", fontsize=5.5)
pl(axB, "b")
labs = [("kt_var_pool", "Var, pool"), ("kt_ar1_pool", "AC, pool"), ("kt_var_rho", "Var, return"), ("kt_ar1_rho", "AC, return")]
for i, (c, lab) in enumerate(labs):
    v = RT[f"ctrl_pct_{c}"].dropna().to_numpy()
    axC.scatter(i + rng.uniform(-0.16, 0.16, len(v)), v, s=5, color=C_GEN, alpha=0.6, lw=0)
    axC.plot([i - 0.25, i + 0.25], [np.mean(v)] * 2, color="black", lw=1.4)
axC.axhline(0.5, color="0.5", lw=0.7, ls="--")
axC.set_xticks(range(4)); axC.set_xticklabels([l for _, l in labs], rotation=35, ha="right")
axC.set_ylabel("Percentile of Kendall's τ among\ntime-matched surviving games"); axC.set_ylim(-0.03, 1.03)
axC.set_title(f"Retrospective trends\n({len(RT)} collapsed games)", loc="left")
pl(axC, "c")
save(fig, "Fig2")

# ================================================================== Figure 3
fig = plt.figure(figsize=(MW * 0.84, 5.0))
gs = fig.add_gridspec(2, 2, wspace=0.55, hspace=0.78)
axA, axB, axC, axD = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
sig_c = {0.02: "#c6dbef", 0.05: "#6baed6", 0.10: "#2171b5", 0.20: "#08306b"}
for s, g in SM.groupby("sigma"):
    g = g.sort_values("log_margin")
    axA.plot(g.log_margin, g.ar1_longrun, color=sig_c[s], lw=1.3, label=f"σ = {s:.2f}")
    axA.plot(g.log_margin, g.ar1_window8_median, color=sig_c[s], lw=1.0, ls="--")
axA.axvline(0, color="0.6", lw=0.6, ls=":")
axA.set_xlabel("Log-growth margin E[log G]"); axA.set_ylabel("Lag-1 autocorrelation of pool")
axA.set_title("Long records slow down; 8-round\nwindows (dashed) do not", loc="left")
axA.legend(frameon=False, fontsize=5.5, loc="upper right", ncol=2); axA.set_ylim(-0.5, 1.4); axA.set_yticks([-0.4, 0, 0.4, 0.8, 1.2])
pl(axA, "a")
fr = TF.groupby("sigma")[["kt_var", "kt_ar1"]].apply(lambda d: (d > 0).mean())
axB.plot(fr.index, fr.kt_var, "o-", color="#000000", ms=3, lw=1.1, label="variance")
axB.plot(fr.index, fr.kt_ar1, "s--", color="0.5", ms=3, lw=1.1, label="autocorrelation")
axB.axhline(0.5, color="0.6", lw=0.6, ls=":")
axB.set_xlabel("Noise σ (group return ratio)"); axB.set_ylabel("Runs with rising trend"); axB.set_xticks([0.02, 0.05, 0.10, 0.20])
axB.set_title("Retrospective trends fade as\nnoise grows (slow forcing)", loc="left"); axB.set_ylim(0, 1.05)
axB.legend(frameon=False, fontsize=5.5, loc="lower left")
pl(axB, "b")
P1 = PH[PH.forcing == 1].copy()
margins = sorted(P1.margin.unique()); sigmas = sorted(P1.sigma.unique())
M = np.full((len(sigmas), len(margins)), np.nan)
for _, r in P1.iterrows():
    M[sigmas.index(r.sigma), margins.index(r.margin)] = r.early_auc_var_pool if r.early_events >= 10 else np.nan
im = axC.imshow(M, origin="lower", cmap="RdBu_r", norm=TwoSlopeNorm(0.5, 0.3, 0.7), aspect="auto")
for i in range(len(sigmas)):
    for j in range(len(margins)):
        if np.isnan(M[i, j]):
            axC.add_patch(mpl.patches.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="////", ec="0.75", lw=0))
        else:
            axC.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5)
axC.set_xticks(range(len(margins))); axC.set_xticklabels([f"{m:+.2f}" for m in margins], rotation=45)
axC.set_yticks(range(len(sigmas))); axC.set_yticklabels([f"{s:.2f}" for s in sigmas])
axC.set_xlabel("Arithmetic margin (1+r)μ − 1"); axC.set_ylabel("Noise σ")
axC.set_title("From a pool at least half full,\nvariance does not discriminate", loc="left")
cb = fig.colorbar(im, ax=axC, fraction=0.05, pad=0.03); cb.set_label("AUC, variance of pool", fontsize=5.5); cb.ax.tick_params(labelsize=5)
pl(axC, "c")
for m in MECH_ORDER:
    g = KM[(KM.cond == f"H:{m}") & (KM.n >= 30)]
    axD.plot(np.exp(g.x_mid), g.mu, "o-", color=MECH_C[m], ms=2.2, lw=1.0, label=MECH_LAB[m])
axD.axhline(0, color="0.5", lw=0.7, ls="--")
axD.set_xscale("log"); axD.set_xticks([2, 10, 50, 200]); axD.set_xticklabels(["2", "10", "50", "200"])
axD.set_xlabel("Pool (start of round)"); axD.set_ylabel("Mean change in log pool")
axD.set_ylim(-2.2, 0.4); axD.set_title("Exclusion-capable rules hold groups\nnear zero drift", loc="left")
axD.legend(frameon=False, fontsize=5, loc="lower right", ncol=2)
pl(axD, "d")
save(fig, "Fig3")

# ================================================================== Figure 4
fig = plt.figure(figsize=(MW, 2.75))
gs = fig.add_gridspec(1, 4, width_ratios=[1.05, 1.05, 1.15, 1.35], wspace=0.78)
axA, axB, axC, axD = [fig.add_subplot(gs[i]) for i in range(4)]
band(axA, AL[AL.kind == "matched survivor"], "gini", C_SUR, "matched survivors")
band(axA, AL[AL.kind == "collapsed"], "gini", C_COL, "collapsed")
axA.set_xlim(25.5, -0.5); axA.set_xlabel("Rounds before collapse"); axA.set_ylabel("Gini of allocations")
axA.set_title("Allocations\nconcentrate", loc="left"); axA.legend(frameon=False, fontsize=5.5, loc="upper left")
pl(axA, "a")
for kind, col in [("matched survivor", C_SUR), ("collapsed", C_COL)]:
    g = AL[AL.kind == kind].groupby("k")
    axB.plot(g.active.mean().index, g.active.mean().values, color=col, lw=1.3)
    axB.plot(g.npos.mean().index, g.npos.mean().values, color=col, lw=1.0, ls="--")
axB.plot([], [], color="0.3", lw=1.3, label="allocation recipients"); axB.plot([], [], color="0.3", lw=1.0, ls="--", label="positive-return contributors")
axB.set_xlim(25.5, -0.5); axB.set_xlabel("Rounds before collapse"); axB.set_ylabel("Mean number of players")
axB.set_ylim(0.8, 4.4); axB.set_title("The contributor\nbase narrows", loc="left"); axB.legend(frameon=False, fontsize=5, loc="upper left")
pl(axB, "b")
for src, col, mk, off in [("humans", "#000000", "o", -0.08), ("clones", "0.55", "s", 0.08)]:
    d = CR[CR.source == src].sort_values("recipients")
    nz = d[d.crashes > 0]
    axC.errorbar(nz.recipients + off, nz.p_crash, yerr=[nz.p_crash - nz.p_lo.clip(lower=1.2e-4), nz.p_hi - nz.p_crash], fmt=mk + "-",
                 color=col, ms=3.2, lw=1.1, elinewidth=0.7, label=src)
    z = d[d.crashes == 0]
    axC.plot(z.recipients + off, np.full(len(z), 1e-4), marker="v", color=col, ms=3.5, ls="none")
axC.set_yscale("log"); axC.set_ylim(6e-5, 0.05); axC.set_xticks([1, 2, 3, 4])
axC.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{100 * v:g}%"))
axC.set_xlabel("Allocation recipients"); axC.set_ylabel("Single-round crashes per round")
axC.set_title("One recipient: the pool can\nvanish in one round", loc="left"); axC.legend(frameon=False, fontsize=5.5, loc="upper right")
axC.text(0.03, 0.10, "▼ no crash observed", transform=axC.transAxes, fontsize=5, color="0.35")
pl(axC, "c")
ear = TRV["early"]["auc"]
items = [("all:state", "State", C_STA), ("all:state+ews", "State + generic", "#56B4E9"), ("all:alloc", "Allocation", C_STR),
         ("all:full", "All features", "#000000")]
for i, (k, lab, col) in enumerate(items):
    for sub, off, mk, filled in [("all", -0.2, "o", True), ("Exp3-4 (clone-independent)", 0.0, "D", False)]:
        r = ear[sub][k]["landmark"]
        axD.errorbar(i + off, r["est"], yerr=[[r["est"] - r["lo"]], [r["hi"] - r["est"]]], fmt=mk, ms=3.0, color=col,
                     mfc=col if filled else "white", elinewidth=0.7)
    r = TRV["all"]["auc"]["all"][k]["landmark"]
    axD.errorbar(i + 0.2, r["est"], yerr=[[r["est"] - r["lo"]], [r["hi"] - r["est"]]], fmt="s", ms=3.0, color=col, mfc="white", elinewidth=0.7, alpha=0.7)
for reg, off, mk, filled in [("early", -0.2, "o", True), ("all", 0.2, "s", False)]:
    r = TJ["cnn"][reg]
    axD.errorbar(4 + off, r["est"], yerr=[[r["est"] - r["lo"]], [r["hi"] - r["est"]]], fmt=mk, ms=3.0, color="#E69F00",
                 mfc="#E69F00" if filled else "white", elinewidth=0.7)
axD.axhline(0.5, color="0.5", lw=0.7, ls="--")
axD.set_xticks(range(5)); axD.set_xticklabels([l for _, l, _ in items] + ["Neural net"], rotation=35, ha="right")
axD.set_ylabel("AUC on human games"); axD.set_ylim(0.4, 1.0)
axD.set_title("Detectors trained on\nbehavioural clones only", loc="left")
axD.plot([], [], "o", color="0.2", ms=3, label="pool ≥ 100, all games"); axD.plot([], [], "D", mfc="white", color="0.2", ms=3, label="pool ≥ 100, Exp. 3-4")
axD.plot([], [], "s", mfc="white", color="0.2", ms=3, label="all landmarks")
axD.legend(frameon=False, loc="lower right", fontsize=5)
pl(axD, "d")
save(fig, "Fig4")

# ================================================================== Extended Data Fig. 1: indicators by institution
FE = [("var_pool", "Variance, pool"), ("ar1_pool", "Autocorrelation, pool"), ("var_logpool", "Variance, log pool"),
      ("ar1_logpool", "Autocorrelation, log pool"), ("var_rho", "Variance, return ratio"), ("ar1_rho", "Autocorrelation, return ratio"),
      ("logR", "Log pool"), ("mean_logG", "Mean log growth"), ("disp_slope", "Dispersion trend"), ("gini_mean", "Allocation Gini"),
      ("neff_alloc_mean", "Effective recipients"), ("active_mean", "Allocation recipients"), ("npos_mean", "Positive-return contributors"),
      ("gini_ret_mean", "Return Gini")]
fig, axs = plt.subplots(1, 2, figsize=(MW, 3.4), gridspec_kw={"width_ratios": [4, 6], "wspace": 0.08})
for ax, reg in zip(axs, ["early", "all"]):
    sub = UM[UM.regime == reg]
    ms = [m for m in MECH_ORDER if m in sub.mech.unique()]
    A = np.array([[sub[(sub.mech == m) & (sub.feature == f)].auc.iloc[0] for m in ms] for f, _ in FE])
    ax.imshow(A, cmap="RdBu_r", norm=TwoSlopeNorm(0.5, 0.0, 1.0), aspect="auto")
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center", fontsize=5, color="white" if abs(A[i, j] - 0.5) > 0.3 else "black")
    lab = [f"{MECH_LAB[m]}\n({int(sub[sub.mech == m].events.iloc[0])}; {int(sub[sub.mech == m].pos_games.iloc[0])} games)" for m in ms]
    ax.set_xticks(range(len(ms))); ax.set_xticklabels(lab, rotation=40, ha="right")
    ax.set_yticks(range(len(FE))); ax.set_yticklabels([l for _, l in FE] if reg == "early" else [])
    ax.set_title("Pool at least half full" if reg == "early" else "All landmarks", loc="left")
save(fig, "ED_Fig1")

# ================================================================== Extended Data Fig. 2: robustness of model comparison
VAR = [("early_primary", "Primary (pool ≥ 100)"), ("early_fullhorizon", "Full 10-round horizon guaranteed"),
       ("early_noExp4", "Without Experiment 4"), ("early_Exp4first", "Experiment 4, first game only"),
       ("early_Exp12", "Experiments 1-2"), ("early_Exp34", "Experiments 3-4"), ("all_primary", "All landmarks"),
       ("all_fullhorizon", "All landmarks, full horizon")]
MODS = [("p_state", "State", C_STA), ("p_state+ews", "State + generic", "#56B4E9"), ("p_alloc", "Allocation", C_STR),
        ("p_ret", "Returns", C_RET), ("p_full", "All", "#000000")]
WTS = [("landmark", "Landmark-weighted"), ("game", "Games equally weighted"), ("inst", "Within institution"), ("time", "Within landmark-round bands")]
fig, axs = plt.subplots(1, 4, figsize=(MW, 3.3), sharey=True, gridspec_kw={"wspace": 0.08})
for ax, (w, wl) in zip(axs, WTS):
    for i, (v, vl) in enumerate(VAR):
        for j, (k, ml, col) in enumerate(MODS):
            e = PV[v]["est"][k][w]; lo, hi = PV[v]["ci"][k][w]
            y = i + (j - 2) * 0.14
            ax.plot([lo, hi], [y, y], color=col, lw=0.8); ax.plot(e, y, "o", color=col, ms=2.4, mec="white", mew=0.3)
    ax.axvline(0.5, color="0.6", lw=0.6, ls="--"); ax.set_xlim(0.3, 1.0); ax.set_title(wl, loc="left")
    ax.set_xlabel("Cross-validated AUC")
axs[0].set_yticks(range(len(VAR)))
axs[0].set_yticklabels([f"{vl}\n({PV[v]['counts']['pos_landmarks']} pos.; {PV[v]['counts']['pos_games']} games)" for v, vl in VAR])
axs[0].invert_yaxis()
for k, ml, col in MODS:
    axs[-1].plot([], [], "o", color=col, ms=3, label=ml)
axs[-1].legend(frameon=False, fontsize=5.5, loc="center left", bbox_to_anchor=(1.02, 0.5))
save(fig, "ED_Fig2")

# ================================================================== Extended Data Fig. 3: no point of no return; collapse routes
fig, axs = plt.subplots(1, 3, figsize=(MW, 2.5), gridspec_kw={"wspace": 0.75})
lab = {"contingent": "exclusion-capable", "unconditional": "non-excluding"}
for (src, mc), g in PNR[PNR.mclass.isin(["contingent", "unconditional"])].groupby(["source", "mclass"]):
    g = g.sort_values("level")
    axs[0].plot(g.level, g.p_collapse, "o-" if src == "human" else "o--", ms=2.5, lw=1.1,
                color="#000000" if mc == "contingent" else "#E69F00")
axs[0].set_xscale("log"); axs[0].set_xticks([5, 10, 25, 50, 100]); axs[0].set_xticklabels(["5", "10", "25", "50", "100"])
axs[0].set_xlabel("Lowest pool reached"); axs[0].set_ylabel("Probability game collapses"); axs[0].set_ylim(0, 1.05)
axs[0].text(150, 0.10, "non-excluding", color="#E69F00", ha="right", va="bottom", fontsize=5.5)
axs[0].text(150, 0.02, "exclusion-capable", color="#000000", ha="right", va="bottom", fontsize=5.5)
axs[0].plot([], [], "-", color="0.3", label="humans"); axs[0].plot([], [], "--", color="0.3", label="clones")
axs[0].set_title("No sharp point of no return\nunder exclusion-capable rules", loc="left"); axs[0].legend(frameon=False, fontsize=5.5, loc="lower left")
pl(axs[0], "a")
TR = pd.read_csv(RV + "terminal_rounds.csv")
TR["route"] = np.where(TR.pool >= 100, "crash from ≥ 100", np.where(TR.pool >= 25, "crash from 25-100", "descent below 25"))
ct = pd.crosstab(TR.mech, TR.route).reindex(MECH_ORDER).fillna(0)
cc = {"crash from ≥ 100": "#D55E00", "crash from 25-100": "#F0A070", "descent below 25": "0.75"}
left = np.zeros(len(ct))
for r_ in ["crash from ≥ 100", "crash from 25-100", "descent below 25"]:
    v = ct[r_].to_numpy() if r_ in ct else np.zeros(len(ct))
    axs[1].barh(range(len(ct)), v, left=left, color=cc[r_], label=r_, height=0.7); left += v
axs[1].set_yticks(range(len(ct))); axs[1].set_yticklabels([MECH_LAB[m] for m in ct.index]); axs[1].invert_yaxis()
axs[1].set_xlabel("Collapsed games"); axs[1].set_title("Routes to collapse:\nsingle-round crashes", loc="left")
axs[1].set_xlim(0, 100); axs[1].legend(frameon=False, fontsize=5, loc="lower right")
pl(axs[1], "b", -0.55)
for src, col, mk in [("humans", "#000000", "o"), ("clones", "0.55", "s")]:
    d = CR[CR.source == src].sort_values("recipients")
    axs[2].errorbar(d.recipients, d.var_nocrash, yerr=[d.var_nocrash - d.v_lo, d.v_hi - d.var_nocrash], fmt=mk + "-", color=col, ms=3, lw=1.1, elinewidth=0.7, label=f"{src}, crash rounds excluded")
    axs[2].plot(d.recipients, d.var_all, mk + ":", color=col, ms=2.5, lw=0.9, mfc="white", label=f"{src}, all rounds")
axs[2].set_yscale("log"); axs[2].set_xticks([1, 2, 3, 4]); axs[2].set_xlabel("Allocation recipients")
axs[2].set_ylabel("Variance of change in log pool"); axs[2].set_title("Without crashes, fluctuations barely\ndepend on recipients", loc="left")
axs[2].set_ylim(0.04, 8); axs[2].legend(frameon=False, fontsize=4.8, loc="upper right")
pl(axs[2], "c")
save(fig, "ED_Fig3")

# ================================================================== Extended Data Fig. 4: phase diagrams (unchanged content)
fig, axs = plt.subplots(1, 3, figsize=(MW, 2.4), gridspec_kw={"wspace": 0.45})
for ax, (forcing, col, ttl, norm) in zip(axs, [(0, "early_auc_var_pool", "AUC of pool variance,\nno forcing", TwoSlopeNorm(0.5, 0.3, 0.7)),
                                              (0, "early_delta", "ΔAUC (state + generic − state),\nno forcing", TwoSlopeNorm(0, -0.1, 0.1)),
                                              (1, "early_delta", "ΔAUC (state + generic − state),\nslow forcing", TwoSlopeNorm(0, -0.1, 0.1))]):
    P_ = PH[PH.forcing == forcing]
    Mx = np.full((len(sigmas), len(margins)), np.nan)
    for _, r in P_.iterrows():
        Mx[sigmas.index(r.sigma), margins.index(r.margin)] = r[col] if r.early_events >= 10 else np.nan
    im = ax.imshow(Mx, origin="lower", cmap="RdBu_r", norm=norm, aspect="auto")
    for i in range(len(sigmas)):
        for j in range(len(margins)):
            if np.isnan(Mx[i, j]):
                ax.add_patch(mpl.patches.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="////", ec="0.75", lw=0))
            else:
                ax.text(j, i, f"{Mx[i, j]:.2f}", ha="center", va="center", fontsize=4.6)
    ax.set_xticks(range(len(margins))); ax.set_xticklabels([f"{m:+.2f}" for m in margins], rotation=45)
    ax.set_yticks(range(len(sigmas))); ax.set_yticklabels([f"{s:.2f}" for s in sigmas]); ax.set_title(ttl, loc="left")
    ax.set_xlabel("Arithmetic margin")
    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03).ax.tick_params(labelsize=5)
axs[0].set_ylabel("Noise σ")
for a_, l_ in zip(axs, "abc"):
    pl(a_, l_)
save(fig, "ED_Fig4")

# ================================================================== Extended Data Fig. 5: theory sensitivity
fig, axs = plt.subplots(1, 3, figsize=(MW, 2.4), gridspec_kw={"wspace": 0.5})
for s, g in SM.groupby("sigma"):
    g = g.sort_values("rho_bar")
    axs[0].plot(1.4 * g.rho_bar - 1, g.log_margin, "o-", ms=2.3, color=sig_c[s], lw=1.0, label=f"σ = {s:.2f}")
axs[0].axhline(0, color="0.6", lw=0.6, ls=":"); axs[0].set_xlabel("Arithmetic margin (1+r)ρ̄ − 1"); axs[0].set_ylabel("Log-growth margin E[log G]")
axs[0].set_title("Noise erodes the log-growth\nmargin (volatility drag)", loc="left"); axs[0].legend(frameon=False, fontsize=5)
pl(axs[0], "a")
PSv = PS[PS.early_events >= 100]
for j, (al, mk) in enumerate([("equal", "o"), ("proportional", "s")]):
    for k, (tl, col) in enumerate([("gauss", "#0072B2"), ("t3", "#D55E00")]):
        d = PSv[(PSv.alloc == al) & (PSv.tails == tl)]
        axs[1].plot(d.sigma + (2 * j + k - 1.5) * 0.012, d.auc_var_pool, mk, ms=3, color=col, mfc=col if j == 0 else "white", ls="none",
                    label=f"{al}, {'Gaussian' if tl == 'gauss' else 'Student-t'}")
axs[1].axhline(0.5, color="0.6", lw=0.6, ls="--"); axs[1].set_ylim(0.35, 0.75); axs[1].set_xlabel("Noise σ"); axs[1].set_ylabel("AUC, variance of pool (pool ≥ 100)")
axs[1].set_title("Variance does not discriminate under\nother allocation rules or heavy tails", loc="left"); axs[1].legend(frameon=False, fontsize=4.8, loc="upper right")
pl(axs[1], "b")
for j, (al, mk) in enumerate([("equal", "o"), ("proportional", "s")]):
    for k, (tl, col) in enumerate([("gauss", "#0072B2"), ("t3", "#D55E00")]):
        d = PSv[(PSv.alloc == al) & (PSv.tails == tl)]
        axs[2].plot(d.sigma + (2 * j + k - 1.5) * 0.012, d.delta, mk, ms=3, color=col, mfc=col if j == 0 else "white", ls="none")
axs[2].axhline(0, color="0.6", lw=0.6, ls="--"); axs[2].set_xlabel("Noise σ"); axs[2].set_ylabel("ΔAUC (state + generic − state)")
axs[2].set_title("Generic indicators add little\nto the state model", loc="left"); axs[2].set_ylim(-0.12, 0.12)
pl(axs[2], "c")
save(fig, "ED_Fig5")

# ================================================================== Extended Data Fig. 6: survival curves, observed vs Markov kernels
conds = MK["conds"]
fig, axs = plt.subplots(3, 4, figsize=(MW, 4.6), sharex=True, sharey=True, gridspec_kw={"wspace": 0.08, "hspace": 0.45})
for ax, c in zip(axs.flat, conds):
    t = np.arange(1, 41)
    ax.step(t, c["obs_surv"], where="post", color="#000000", lw=1.2, label="observed")
    ax.plot(t, c["state_surv"], color="#0072B2", lw=1.0, label="pool-state kernel")
    ax.plot(t, c["mem_surv"], color="#D55E00", lw=1.0, ls="--", label="kernel with one-round memory")
    m, e = c["cond"].split("|")
    ax.set_title(f"{MECH_LAB[m]}, Exp. {e} (n = {c['n_games']})", loc="left", fontsize=6)
    ax.set_ylim(-0.02, 1.02); ax.set_xlim(0, 40)
for ax in axs[-1]:
    ax.set_xlabel("Round")
for ax in axs[:, 0]:
    ax.set_ylabel("Not yet collapsed")
axs.flat[-1].axis("off")
h_, l_ = axs.flat[0].get_legend_handles_labels(); axs.flat[-1].legend(h_, l_, frameon=False, fontsize=6, loc="center")
save(fig, "ED_Fig6")

# ================================================================== Extended Data Fig. 7: transfer calibration and alerts
fig, axs = plt.subplots(1, 3, figsize=(MW, 2.45), gridspec_kw={"wspace": 0.5})
rel = TRV["early"]["reliability"]
for k, lab, col in [("all:state", "State", C_STA), ("all:alloc", "Allocation", C_STR), ("all:full", "All features", "#000000")]:
    d = pd.DataFrame(rel[k]); axs[0].plot(d.mean_pred, d.obs, "o-", ms=2.6, lw=1.0, color=col, label=lab)
axs[0].plot([0, 0.07], [0, 0.07], color="0.6", lw=0.6, ls="--"); axs[0].set_xlim(0, 0.07); axs[0].set_ylim(0, 0.25)
axs[0].set_xlabel("Mean predicted risk (decile)"); axs[0].set_ylabel("Observed fraction positive")
axs[0].set_title("Clone-trained models\nunder-predict human risk", loc="left"); axs[0].legend(frameon=False, fontsize=5.5, loc="lower right")
pl(axs[0], "a")
cbm = TRV["early"]["calibration_by_mech"]
ms_ = [m for m in MECH_ORDER if m in cbm and m != "Equal"]
x = np.arange(len(ms_))
axs[1].bar(x - 0.2, [cbm[m]["all:alloc"]["mean_pred"] for m in ms_], 0.38, color=C_STR, label="predicted")
axs[1].bar(x + 0.2, [cbm[m]["all:alloc"]["obs"] for m in ms_], 0.38, color="0.3", label="observed")
axs[1].set_xticks(x); axs[1].set_xticklabels([MECH_LAB[m] for m in ms_], rotation=35, ha="right")
axs[1].set_ylabel("Fraction of landmarks positive"); axs[1].set_title("Under-prediction is concentrated\nin proportional allocation", loc="left")
axs[1].legend(frameon=False, fontsize=5, loc="upper right")
pl(axs[1], "b")
al = TRV["early"]["alert"]
vals = [al["sensitivity"], al["specificity"], al["ppv"], al["collapsed_games_alerted"] / al["collapsed_games"], al["surviving_games_alerted"] / al["surviving_games"]]
labs_ = ["Sensitivity", "Specificity", "PPV", "Collapsed\ngames", "Surviving\ngames"]
axs[2].bar(range(5), vals, color=["#009E73", "#009E73", "#009E73", C_COL, C_SUR], width=0.65)
for i, v in enumerate(vals):
    axs[2].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=5.5)
axs[2].set_xticks(range(5)); axs[2].set_xticklabels(labs_, rotation=40, ha="right"); axs[2].set_ylabel("Fraction"); axs[2].set_ylim(0, 1.05)
axs[2].set_title("Alert at a threshold fixed on\nclones: many false alerts", loc="left")
pl(axs[2], "c")
save(fig, "ED_Fig7")
print("revised figures written")
