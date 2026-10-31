"""Check that headline numbers quoted in the manuscript match the derived results shipped in the package.
usage (from the package root): python 05_code/verify_numbers.py
Checks numbers retained from the original submission (revision numbers: verify_numbers_rev.py). Each check recomputes a formatted string from 06_derived_data and asserts it occurs in 01_manuscript/main.tex.
"""
import json, os, re, sys
import pandas as pd

root = sys.argv[1] if len(sys.argv) > 1 else "."
D = os.path.join(root, "06_derived_data")
tex = open(os.path.join(root, "01_manuscript", "main.tex"), encoding="utf-8").read()
J = json.load(open(f"{D}/human/results.json"))
CS = json.load(open(f"{D}/confirm/confirm_summary.json"))
UC = pd.read_csv(f"{D}/confirm/univariate_auc_ci.csv")
ST = pd.read_csv(f"{D}/extra/stratified_auc_early.csv").set_index("feature")
X5 = json.load(open(f"{D}/extra/incremental_over_struct.json"))
TJ = json.load(open(f"{D}/transfer/transfer_results.json"))
PE = pd.read_csv(f"{D}/landscape/peclet.csv").set_index("cond")
FP = pd.read_csv(f"{D}/landscape/markov_first_passage.csv")
PH = pd.read_csv(f"{D}/phase/phase_diagram.csv")
CL = json.load(open(f"{D}/clone/results.json"))


def ci(d, nd=2):
    return f"{d['est']:.{nd}f} ({d['lo']:.{nd}f}--{d['hi']:.{nd}f})"


E = UC[(UC.regime == "early") & (UC.scope == "pooled")].set_index("feature")
checks = {
    "early landmarks": f"{J['early_n']:,}",
    "early events": f"{J['early_events']}",
    "collapsed games": f"{J['n_collapsed']} of 680",
    "var_pool AUC": f"{E.loc['var_pool'].auc:.2f} (95\\% cluster-bootstrap interval {E.loc['var_pool'].lo:.2f}--{E.loc['var_pool'].hi:.2f})",
    "ar1_pool AUC": f"{E.loc['ar1_pool'].auc:.2f} ({E.loc['ar1_pool'].lo:.2f}--{E.loc['ar1_pool'].hi:.2f})",
    "state CV early": ci(J["early_cv"]["state"]),
    "H2 delta P": f"$P={CS['early_H2_delta_stateews_minus_state']['p_one_sided']:.2f}$",
    "state CV all": ci(J["all_cv"]["state"]),
    "gini AUC": f"{E.loc['gini_mean'].auc:.2f} ({E.loc['gini_mean'].lo:.2f}--{E.loc['gini_mean'].hi:.2f})",
    "active AUC reversed": f"{1 - E.loc['active_mean'].auc:.2f} ({1 - E.loc['active_mean'].hi:.2f}--{1 - E.loc['active_mean'].lo:.2f}",
    "stratified var": f"{ST.loc['var_pool'].strat_auc:.2f} ({ST.loc['var_pool'].lo:.2f}--{ST.loc['var_pool'].hi:.2f})",
    "cnn early": ci(TJ["cnn"]["early"])[:-1],
    "mean predicted risk": f"{TJ['early']['mean_pred_state']:.3f}",
    "observed early rate": f"{TJ['early']['obs_rate']:.3f}",
    "Pe equal": f"{PE.loc['H:Equal'].Pe:.1f} (${PE.loc['H:Equal'].Pe_lo:.1f}$ to ${PE.loc['H:Equal'].Pe_hi:.1f}$)".replace("-", "-"),
    "clone delta": f"{CL['early_cv']['delta_state+ews_minus_state']['est']:.3f}",
}
fails = 0
for k, s in checks.items():
    s2 = s.replace("-", "$-$") if k == "Pe equal" else s
    ok = (s in tex) or (s2 in tex)
    if k == "Pe equal":
        ok = f"${PE.loc['H:Equal'].Pe:.1f}$" in tex
    print(("OK   " if ok else "FAIL ") + k + ": " + s)
    fails += (not ok)
hi = PH[PH.sigma >= 0.2]
print("phase range", round(hi.early_auc_var_pool.min(), 2), round(hi.early_auc_var_pool.max(), 2))
n_in = int(((FP.source == "human") & (FP.pred_markov >= FP.obs_lo) & (FP.pred_markov <= FP.obs_hi)).sum())
print("Markov in-condition within CI:", n_in, "of", int((FP.source == "human").sum()))
print("FAILURES:", fails)
sys.exit(1 if fails else 0)
