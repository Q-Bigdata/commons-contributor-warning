# Pre-specified analysis plan: early-warning signals of collapse in replicated human commons

Frozen on 2026-09-30, before the full-sample confirmatory analyses were run on the 680 human games. Code smoke tests on a 120-game human subset preceded the freeze and are disclosed below (amendment A1). The SHA-256 hash of this file is recorded in the submission package (`MANIFEST.txt`) and in the manuscript's Methods.

## What had already been seen

Before freezing, the authors inspected the following descriptive quantities in the human data: the number of games and collapses per experiment and mechanism (191 of 680 games reach a pool below 1), the median pool, group return ratio, within-group dispersion and number of active players as a function of time-to-collapse in the collapsed games, and a table of the mean one-round growth factor 1.4 times the group return ratio binned by pool size and mechanism. No indicator of variance, autocorrelation, trend, or predictive accuracy had been computed on human data. Toy-model parameters were tuned only to give collapse rates of about one third within 40 rounds.

**Amendment A1 (disclosure of pre-freeze code smoke tests, added 2026-09-30 after an internal audit; analysis choices unchanged).** To debug the pipeline, the scripts were executed on a subset of 120 human games (the first 120 game identifiers in sorted order; about 18% of the human sample). This printed (i) summary statistics (mean, minimum, maximum) of the landmark features on a 60-game subset, including var_pool, ar1_pool, var_rho and ar1_rho; (ii) the landmark count and number of events of `run_analysis.py` on the 120-game subset (the AUC results file was written but not inspected); and (iii) the output of `landscape.py` on the 120-game subset, i.e. observed versus Markov-predicted collapse fractions per condition, plateau drift and Péclet numbers per human mechanism, and the lagged-log-growth coefficient (0.103, s.e. 0.094). None of the pre-specified parameters (W = 8, h = 10, early threshold 100, indicator set, hypothesis family, decision rules) was changed after these outputs. H5 and the landscape analyses should therefore be read as partially unblinded, and H1–H4 as blinded only with respect to AUC and trend statistics.

## Data

The public per-round file `sustainable_behavior.csv` (214,185,378 bytes, MD5 base64 `dMl+sTp8gl3Y7gOvEE1Ymg==`) of Koster, Pîslar et al. (2025). Human games are the 680 evaluation games of Experiments 1–4 (Exp. 1: Equal, Mixed, Proportional, RL-M1, 40 each; Exp. 2: Interpolating, Proportional, RL-M1, 40 each; Exp. 3: Interpolating and RL-M2, 80 each; Exp. 4: 80 groups × 3 consecutive games under Proportional or RL-M2). Behavioural-clone games are the 3,584 simulated episodes in the same file (512 per mechanism-clone combination). Collapse is the first round whose end-of-round pool falls below 1 (the definition of the original paper). Rows after collapse are discarded. The cluster unit for all resampling is the human group (the launch identifier, which links the three games of an Exp. 4 group).

## Landmark design (primary)

For every game and every round t ≥ 7 still at risk, a landmark uses only information available at the end of round t. The window is W = 8 rounds (t−7 … t). The outcome is collapse within the next h = 10 rounds (t < τ ≤ t + 10). Landmarks without an event and with fewer than 10 rounds of remaining follow-up are dropped. The primary regime is the **early regime**, landmarks at which the pool is still at least 100 (half of capacity); the secondary regime includes all landmarks.

Generic indicators (window, after linear detrending): variance and lag-1 autocorrelation of the end-of-round pool, of the log pool, and of the group return ratio ρ = Σc/Σe. State variables: log pool, mean log growth factor over the window and over the last three rounds. Structural indicators: mean and slope of within-group dispersion of individual return fractions, mean number of active players, mean offer Gini, fraction of window rounds at the cap (pool ≥ 190), number of cap crossings (flickering), mean fraction of the pool retained by the mechanism.

## Hypotheses and primary tests (family of five, Holm-corrected)

H1 (generic warning signals). In the early regime, at least one of the pool variance or pool lag-1 autocorrelation has a pooled univariate AUC above 0.5 with a cluster-bootstrap 95% CI that excludes 0.5.

H2 (incremental value). Adding the six generic indicators to the state model improves the cross-validated AUC (group-wise five-fold CV, five repeats) in the early regime; test: cluster-bootstrap CI of ΔAUC excludes 0.

H3 (governance moderation). The cross-validated AUC of the indicator-only model is lower in state-contingent mechanisms (Proportional, Interpolating, RL-M1, RL-M2) than in unconditional mechanisms (Equal, Mixed); test: bootstrap difference.

H4 (classical retrospective test). In collapsed games with at least 12 pre-descent rounds (the final descent begins at the first round after which the pool never returns to ≥100), Kendall's τ of rolling variance and autocorrelation of the pool (window = half the segment, at least 6) is higher than in time-matched surviving games of the same mechanism; statistic: mean percentile of each collapsed game among its matched controls, compared with 0.5 by a sign test.

H5 (state-only sufficiency). A Markov model in which the next change of log pool depends only on the current pool bin (nonparametric, mechanism-specific) predicts the collapse probability of each experimental condition within the 95% CI of the observed fraction in at least 9 of 11 conditions.

## Secondary and robustness analyses

W ∈ {6, 12}, h = 5; all-landmark regime; per-mechanism AUCs; the same pipeline on the behavioural clones and on three toy models that differ only in the route to collapse (slow forcing, neutral marginal stability, Allee-type noise-induced escape); transfer of models trained only on clone games to humans (logistic and gradient-boosted trees); a convolutional–recurrent network trained on clone windows and applied to human windows; stability landscapes (Kramers–Moyal drift and diffusion of log pool) and plateau Péclet numbers Pe = μL/σ² for L = log(190/25); point-of-no-return curves; test of history dependence (lagged log-growth coefficient after absorbing pool-bin × mechanism fixed effects).

## Decision rules

A null result for H1–H2 is reported as such. Exploratory analyses added after unblinding are labelled as exploratory in the manuscript.
