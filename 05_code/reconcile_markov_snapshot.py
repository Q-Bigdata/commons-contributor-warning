"""Reconstruct the archived observed-curve reporting transformation, without resimulation.

This preserves simulated curves and parameters from markov_revision.json. The
archived final curves count games without a recorded collapse by round t over
the full condition cohort, including games whose recording ended earlier.
They are descriptive recorded-outcome fractions, not a censoring-adjusted
survival estimator. No new scientific choice is introduced by this utility.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def reconcile(root):
    base = Path(root) / "06_derived_data" / "revision"
    result = json.loads((base / "markov_revision.json").read_text())
    rounds = pd.read_csv(base / "rounds_ext_human.csv.gz")
    games = rounds.groupby("game").agg(
        mech=("mech", "first"), exp=("exp", "first"), tau=("tau", "first")
    )
    games["cond"] = games.mech + "|" + games.exp.astype(str)
    for condition in result["conds"]:
        tau = games.loc[games.cond == condition["cond"], "tau"].to_numpy()
        assert len(tau) == condition["n_games"]
        observed = [float(np.mean(~(tau <= t))) for t in range(40)]
        condition["obs_surv"] = observed
        for key in ("state", "mem"):
            condition[f"{key}_maxdiff_surv"] = float(
                max(abs(o - s) for o, s in zip(observed, condition[f"{key}_surv"]))
            )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = reconcile(args.root)
    reference = json.loads(
        (args.root / "06_derived_data/revision/markov_revision_fixedobs.json").read_text()
    )
    # Compare all archived fields, including simulations that must be preserved.
    assert json.dumps(result, sort_keys=True) == json.dumps(reference, sort_keys=True), (
        "Reconstructed values differ from the archived final snapshot"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print("All final Markov snapshot fields match; simulation outputs were preserved.")
