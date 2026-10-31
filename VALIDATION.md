# Validation scope — 31 October 2026

- Python 3.13.5; exact installed numerical/plotting versions in requirements.txt.
- Existing number verifiers: both exit 0; revision verifier reports 25 passed, 0 failed.
- Compact human-data preparation and all landmark computations rerun with the original ews_lib.py: 680 games, 520 groups, 197 collapses, 10,744 landmarks, 7,192 early landmarks and 296 early positive landmarks; zero joined outcome mismatches and zero pool-state differences.
- Pool-law absolute error at most 0.010010 units in the rounded compact data (approximately 0.01).
- Original figure entry point reproduced its missing-style NameError; the corrected copy regenerated all 11 figure pairs. Archived/generated main and extended-data comparison sheets were visually inspected: curves, heatmap values and qualitative trends agree; styling differs.
- Original final Markov reporting transformation reconstructed exactly, including all simulation fields preserved from the archive.
- Source parsing passed for all supplied analysis sources; an additional reconstruction utility is included and executed successfully.
- Not rerun: full cross-validation/model fitting, bootstrap distributions, theory Monte Carlo grids, CNN/LSTM training and original raw-to-compact conversion. PyTorch is not installed in the audit environment. Fitted-model pickles were copied but not loaded.
- Remaining warnings: NumPy degrees-of-freedom warnings for sparse windows during landmark creation; Matplotlib/pandas single-element-Series conversion FutureWarning during plotting. Commands completed successfully; no future-version compatibility claim is made.
