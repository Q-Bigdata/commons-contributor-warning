"""Traceability check for numbers introduced in revision R1.
Each check recomputes a value from a derived-data file and asserts that the formatted string appears in the
manuscript or SI source. Run from the package root:  python 05_code/verify_numbers_rev.py
"""
import json, re, sys
import numpy as np
import pandas as pd

D = "06_derived_data/revision/"
MS = open("01_manuscript/main.tex", encoding="utf-8").read() + open("02_supplementary/si.tex", encoding="utf-8").read() \
    + open("02_supplementary/si_tables.tex", encoding="utf-8").read() + open("01_manuscript/ed_tables.tex", encoding="utf-8").read()
PV = json.load(open(D + "predict_variants.json"))
TR = json.load(open(D + "transfer_revision.json"))
MK = json.load(open(D + "markov_revision_fixedobs.json"))
CR = pd.read_csv(D + "crash_by_recipients.csv")
TM = pd.read_csv(D + "terminal_rounds.csv")
ok = fail = 0


def chk(label, s):
    global ok, fail
    if s in MS:
        ok += 1
    else:
        fail += 1; print("NOT FOUND:", label, repr(s))


def f2(x): return f"{x:.2f}"
def ci2(e, lo, hi): return f"{e:.2f} ({lo:.2f}--{hi:.2f})"


ep = PV["early_primary"]
chk("positive games", f"{ep['counts']['pos_games']} collapsed games in {ep['counts']['pos_groups']} groups")
for k, w, lab in [("p_alloc", "game", "alloc game"), ("p_alloc", "inst", "alloc inst"), ("p_alloc", "time", "alloc time")]:
    e = ep["est"][k][w]; lo, hi = ep["ci"][k][w]; chk(lab, ci2(e, lo, hi))
for v, lab in [("early_fullhorizon", "full horizon"), ("early_noExp4", "noExp4"), ("early_Exp4first", "Exp4 first"),
               ("early_Exp12", "Exp12"), ("early_Exp34", "Exp34")]:
    e = PV[v]["est"]["p_alloc"]["landmark"]; lo, hi = PV[v]["ci"]["p_alloc"]["landmark"]; chk(lab, ci2(e, lo, hi))
d = ep["ci"]["p_state+alloc-p_state"]["landmark"]; chk("alloc increment", f"{d['est']:.2f} ({d['lo']:.2f}--{d['hi']:.2f})")
d = ep["ci"]["p_state+ews-p_state"]["inst"]; chk("generic increment inst", f"{d['est']:.2f} ({d['lo']:.2f}--{d['hi']:.2f})")
crashes = TM[TM.pool >= 25]; chk("crash count", f"{len(crashes)} occurred in a single round"); chk("crash >=100", f"{int((TM.pool >= 100).sum())} of these from at least 100")
h = CR[CR.source == "humans"].set_index("recipients")
chk("crash rate 1", f"{100 * h.loc[1, 'p_crash']:.1f}\\% ({100 * h.loc[1, 'p_lo']:.1f}--{100 * h.loc[1, 'p_hi']:.1f}\\%)")
for k in [1, 2, 3, 4]:
    chk(f"var nocrash {k}", f"{h.loc[k, 'var_nocrash']:.3f} ({h.loc[k, 'v_lo']:.3f}--{h.loc[k, 'v_hi']:.3f})")
c = CR[CR.source == "clones"].set_index("recipients"); chk("clone crash", f"{100 * c.loc[1, 'p_crash']:.2f}\\%")
ea = TR["early"]["auc"]
for sub, k, lab in [("all", "all:alloc", "transfer alloc"), ("Exp3-4 (clone-independent)", "all:alloc", "transfer alloc E34"),
                    ("Exp3-4 (clone-independent)", "bc1:alloc", "transfer bc1 E34"), ("all", "all:full", "transfer full")]:
    x = ea[sub][k]["landmark"]; chk(lab, ci2(x["est"], x["lo"], x["hi"]))
al = TR["early"]["alert"]; chk("alert games", f"{al['collapsed_games_alerted']} of {al['collapsed_games']} collapsed games")
gaps = [cnd["state_maxdiff_surv"] for cnd in MK["conds"]]; chk("timing gap", f"{min(gaps):.2f}--{max(gaps):.2f}")
print(f"{ok} checks passed, {fail} failed")
sys.exit(1 if fail else 0)
