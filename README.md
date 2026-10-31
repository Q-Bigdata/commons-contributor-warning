# Commons contributor warning

Analysis code and derived inputs for **Reliance on a narrowing contributor base, not critical slowing down, forecasts collapse in governed human commons**, by Tianyu Shi, Hanqing Qian and Chengyi Tu (R1 manuscript).

The package corresponds to manuscript revision R1. The code is released under the MIT License, with original data attribution and third-party terms retained. Fixed archived version: **v1.0.0**, DOI: [10.5281/zenodo.23836262](https://doi.org/10.5281/zenodo.23836262). The archive is fixed to source commit `3860e885ded12d8c08044e74cf8884295b6a1f7d`; the tagged research files remain unchanged.

## Contents

- `05_code/`: original analysis sources, a repaired standalone R1 figure entry point and an explicit reconstruction of the archived Markov reporting transformation.
- `06_derived_data/`: compact rounds, landmarks, estimates, out-of-fold predictions, simulation summaries and model snapshots consumed by the existing transfer code. Logs are excluded.
- `04_source_data/Source_Data.xlsx`: figure source-data workbook.
- `07_preregistration/`: frozen analysis plan, amendment hash and reconciliation note; this was not a public preregistration.
- `FIGURE_MAP.csv`: manuscript figure names and generating script sections.
- `ATTRIBUTION.md`: original data provenance and terms.

Manuscript source/PDF, cover letter, reviewer reports, author contact lists, signatures and internal workflow records are excluded. Legacy plotting scripts and server-specific shell drivers are excluded; their analysis sources are preserved in the submission baseline.

## Environment

The manuscript reports Python 3.11 on the original Linux compute server. This audit ran Python 3.13.5 on Windows, with the package versions pinned in `requirements.txt`. Figure regeneration and deterministic checks passed in that environment. The original complete training environment has not been reconstructed. Model pickle files are supplied for provenance/consumption by `transfer.py`; compatibility with another scikit-learn version is not guaranteed, and they were not deserialized during this audit.

```sh
python -m pip install -r requirements.txt
python 05_code/figs_rev.py
```

Run commands from the repository root. Figures are written as PDF and 300-dpi PNG to `figs_rev/`. Set `EWS_FIGURE_DIR` to choose another output directory and `MPLBACKEND=Agg` for a display-free run. The script produces the 4 main, 6 Extended Data and 1 supplementary figures with manuscript filenames. The external style helper was replaced with self-contained typography; numerical plotting calculations and archived inputs were preserved. Typography and image bytes need not match the submission figures exactly.

## Analysis inputs and entry points

The compact round table is the analysis input, not the raw original CSV. Prepare the existing expected input path once:

```sh
python -c "from pathlib import Path; import shutil; Path('data').mkdir(exist_ok=True); shutil.copyfile('06_derived_data/rounds_compact.csv.gz', 'data/rounds.csv.gz')"
```

The raw-to-compact conversion script was not present in the supplied package. The compact snapshot is provided with hashes and original-data attribution; this release does not claim independently verified reconstruction of that conversion.

The following are full analysis commands, not commands that were all rerun in this audit. They can be expensive. Outputs go under `results/`; the figure entry point intentionally reads the immutable archived inputs in `06_derived_data/`.

```sh
python 05_code/run_analysis.py --source human --out results/human --W 8 --h 10 --boot 1000
python 05_code/run_analysis.py --source clone --out results/clone --W 8 --h 10 --boot 1000
python 05_code/h1_boot.py --res results/human --out results/confirm --boot 2000
python 05_code/landscape.py --out results/landscape --nsim 20000 --boot 500
python 05_code/rev_predict.py --out results/revision --boot 1000 --procs 1
python 05_code/rev_concentration.py --res results/revision --boot 2000
python 05_code/rev_transfer.py --res results/revision --boot 1000
python 05_code/rev_markov.py --res results/revision --nsim 20000 --boot 200 --procs 1
python 05_code/rev_theory.py --out results/revision/theory --ngames 3000 --procs 1
python 05_code/theory_phase.py --out results/phase --ngames 3000 --procs 1
```

`theory_phase.py` defaults to 2,000 games; use the explicit 3,000 argument for the manuscript's phase grid. Window length 8, forecast horizon 10 and early-pool threshold 100 agree with the primary analysis. Other original entry points (`toy.py`, `theory_sim.py`, `extra_checks.py`, `transfer.py`) are retained; consult their in-source arguments. `theory_sim.py` writes CSVs to its working directory. `transfer.py` first requires the original human/clone analysis outputs and their fitted-model pickle files. Retraining its CNN/LSTM additionally requires `requirements-neural.txt` (PyTorch); no neural training or full Monte Carlo/bootstrap rerun was performed here.

## Markov reporting snapshot

`rev_markov.py` emits `markov_revision.json`; the final manuscript uses `markov_revision_fixedobs.json`. These differ in observed-curve denominators and two derived curve-distance summaries for Experiment 4. All simulation arrays are the same. The following reconstructs every field of the archived final snapshot from the archived pre-transformation result and round table, and fails if any field differs:

```sh
python 05_code/reconcile_markov_snapshot.py --output results/markov_revision_fixedobs.json
```

The final observed curves describe the fraction of the entire condition cohort without a recorded collapse by round t, including games whose administrative record ended earlier. They are not a censoring-adjusted survival estimator. The script documents and verifies the existing transformation; it does not make a new inference about censored observations. Further scientific interpretation belongs in manuscript review.

## Validation

In this preparation run: all 11 figures regenerated; both existing manuscript-number verification scripts passed (including 25 revision checks); recomputation from the compact human rows recovered 680 games, 520 groups, 197 collapsed games and 7,192 early landmarks with 296 positive outcomes. All 10,744 human landmark outcomes and pool states matched the archived table. The Markov final snapshot reconstruction matched every archived field. See `VALIDATION.md` for the scope.

The original `verify_numbers.py` and `verify_numbers_rev.py` compare against manuscript/SI TeX files. Those files are intentionally not included in this repository. To rerun the text checks, change the working directory to the separately supplied submission package, then invoke this repository's absolute verifier path. Do not interpret their absence of errors as full experimental replication.

## License

The authors' software is licensed under the MIT License (`LICENSE`). Third-party data terms and attribution are documented in `ATTRIBUTION.md`. The original-data CC BY 4.0 attribution obligations are retained.
