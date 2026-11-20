# ELEPHANT-PREREG: do elephant "name" calls survive a receiver-label null blocked by caller, day and relationship?

Pre-registration, MV2 of Munim, Inc. (mv2@munimtech.com), 2026-11-20. Made with AI agents; no
human expert has reviewed this design yet. Text CC BY 4.0, code Apache-2.0.
Registered before we computed any statistic on these data. A label-only description job
(elephant-names-describe-v1: caller, receiver, date, site and certainty counts, no acoustics)
was filed earlier, but it had not returned when this was written. We have read only the
reply's R script and the column headers.
Code: `names_null.py` (tests: `test_names_null.py`), job `elephant-names-null-v1`.

## 1. The claim and the open dispute
* Pardo et al. 2024, Nature Ecology & Evolution 8:1353-1364 (doi 10.1038/s41559-024-02420-w):
  in 469 rumbles from Samburu and Amboseli, a random forest predicts the receiver from call
  acoustics above chance, and calls from one caller to the same receiver are more similar than
  calls to different receivers. Read as evidence that African elephants address each other with
  individually specific, name-like calls.
* Dharmarajan 2026, Animal Behaviour 235:123547 (doi 10.1016/j.anbehav.2026.123547): the
  evidence is inconclusive. Receiver prediction is driven by caller identity and temporal
  autocorrelation, and the same-receiver similarity by autocorrelation and pseudoreplication.
  Code and data: figshare doi 10.6084/m9.figshare.30704276.
* Pardo et al., reply (submitted to Animal Behaviour, October 2026; data and code Zenodo
  record 23284924, CC BY 4.0): corrects the Amboseli timestamps, tests autocorrelation by
  lag bins, and models the RF-GAP proximity rank of same-caller pairs as
  `pairType + sameCont + sameDate + (1|pairID) + (1|mm(callA, callB))`, excluding pairs in the
  same 60-s temporal group. The forest that produces the proximities is trained on the TRUE
  receiver labels. Date enters only as a fixed effect.

## 2. What is new here
Nobody has refitted the whole pipeline (forest, proximities and contrast) under receiver labels
permuted only WITHIN blocks of one caller, one date and one relationship category. Such a null
keeps every day-level and caller-level confound and the relationship type in place, and breaks
only receiver identity. A positive result could not come from day-level drift, caller identity,
label imbalance, relationship type or a forest trained on confounded labels. A negative result
is read only together with the measured power at an effect that reproduces the observed
contrast. Both camps in the dispute have proposed weaker controls, so either answer is new.

## 3. Data and gates
Zenodo record 23284924: `20231225_dfall_obs_data_and_spectral_features_revision1_n469.csv`
(md5 f9024ced42a489e82c4e41d65ba70aa4) and
`20260731_Amboseli_sound_files_used_in_names_paper_for_upload.csv` (md5
4ea38110eb7a48078fe9a45b5e667181).
* G1: both md5 match, or the job stops with DATA_MISMATCH and computes nothing.
* G2: usable calls (all 94 features numeric; caller, receiver and date present). If fewer than
  460, the result is graded but marked provisional.
* G3: at least 100 eligible pairs (section 4), at least one informative (caller, date,
  relationship) block, and both same- and different-receiver pairs present. Otherwise the
  verdict is NO_INFORMATIVE_BLOCKS.
Timestamps follow the reply's code: Samburu = YYYYMMDD_HHMM from the file name + `start`.
Amboseli = the timestamp file joined on soundID, with Date as d/m/Y, then the Call_start fields
if all three are present, else track start + tape-to-call - tape-to-track. Calls without a
timestamp form their own temporal group, and their count is reported.

## 4. Statistic (mirrors the reply's model set)
* Forest: one classification tree per draw (scikit-learn DecisionTreeClassifier, gini, mtry 6,
  min leaf 1, fully grown), 300 trees, each on a fixed 60% subsample without replacement.
  The response is the receiver (Elicitor1_ID) and the predictors are the 94 acoustic features.
  All usable calls are used. The in-bag sets and tree seeds are fixed (seed 12345) and
  identical for every fit; only labels or features change. The reply uses ranger with 8000
  trees; 300 are used here so that about 7,500 refits fit the budget.
* Proximity: symmetrised RF-GAP (Rhodes, Cutler & Moon 2023), computed from out-of-bag calls.
* Eligible pairs: same caller; same relationship category (newSocCat); both Cert_ID_Bin = 1
  and Cert_Con_Bin = 1; both contexts known; different 60-s temporal groups (chained within
  caller; a gap of exactly 60 s stays in the group).
* T = OLS coefficient of sameReceiver in `rank(proximity)/n_pairs ~ 1 + sameReceiver +
  sameContext + sameDate`. T > 0 means calls to the same receiver are more similar.
* Model-free secondary: the same T with rank(-Euclidean distance on z-scored features), no
  forest.

## 5. Nulls
Receiver labels are permuted within blocks, and the whole pipeline is refitted for each
permutation. p = (1 + #{T_null >= T_obs, or T_null undefined}) / (B + 1), one-sided.
* PRIMARY, cdr: blocks = (caller, date, relationship category), B = 999, seed 20261120.
* Side, cd: blocks = (caller, date), B = 199, seed 20261121.
* Side, cr: blocks = (caller, relationship category), all dates pooled. This is the naive
  within-caller null. B = 199, seed 20261122.
* Model-free T under cdr and cr: B = 999 each, seeds 20261123 and 20261124.

## 6. Calibration: two synthetic worlds on the real label structure
Each synthetic call has the same caller, receiver, date, relationship, context, certainty and
temporal group as the real one. Its 94 features are caller effect + noise (both unit variance)
plus one of:
* Receiver world: size x (one random vector per caller-receiver dyad). A pure "names" effect.
* Drift world: size x (one random vector per (caller, date), an Ornstein-Uhlenbeck chain over
  each caller's dates with time scale 60 days). Slow drift with no receiver effect; the
  autocorrelation story.
Matching: for size in 0.25, 0.50, ..., 4.00 (4 replicates each, seeds 5000000 and 6000000
bases), T is computed with real labels. The matched size is the grid value whose mean T is
closest to T_obs (ties go to the smaller value). If T_obs <= 0, both worlds use size 1.0.
"Reached" = the largest mean T on the grid is at least T_obs.
Calibration at the matched sizes, R = 20 synthetic data sets per world, B = 99 per test:
* power = share of receiver-world sets with p_cdr <= 0.05;
* size = share of drift-world sets with p_cdr <= 0.05 (should be about 0.05, because labels
  are exchangeable within the blocks in that world);
* fooled = share of drift-world sets with p_cr <= 0.05 (how often the naive within-caller
  null mistakes drift for names when drift alone reproduces the observed contrast).

## 7. Decision rules (applied in this order, coded in verdict_of)
1. size > 0.20 (more than 4 of 20), or undefined: CALIBRATION_FAILED.
2. T_obs <= 0 or undefined: NO_POSITIVE_CONTRAST.
3. p_cdr <= 0.05: RECEIVER_SIGNAL_SURVIVES_DAY_BLOCKING.
4. power >= 0.80: NOT_DETECTED_WITH_POWER.
5. otherwise: UNDERPOWERED.
Readings fixed now:
* SURVIVES: receiver-specific acoustic similarity that no day-level confound, caller identity
  or relationship type explains. This is the strongest support yet for name-like calls from
  these data, and it would be stated as such.
* NOT_DETECTED_WITH_POWER: a receiver effect big enough to produce the published contrast
  would very likely have been seen inside days. The contrast is better explained by structure
  across days (drift, state, pseudoreplication) than by names. This is not proof that elephants
  lack names.
* UNDERPOWERED: these data cannot decide the question with a day-blocked null. The share of
  calls in informative blocks is the reason, and that is itself a design lesson for future
  recordings.
* Side readings, never verdicts: p_cr versus p_cdr shows how much the conclusion depends on
  blocking by day. fooled >= 0.5 means the naive null cannot be trusted in this label structure.
  Reached = false means slow drift of this form cannot by itself produce the observed contrast.
  That is evidence against the pure-drift story, reported as exploratory.

## 8. Predictions (priors, Brier-scored)
Q1 T_obs > 0: 0.90. Q2 p_cr <= 0.05: 0.75. Q3 p_cdr <= 0.05: 0.30. Q4 power >= 0.80: 0.35.
Q5 size <= 0.20: 0.95. Q6 fooled >= 0.50: 0.50. Q7 drift world reaches T_obs: 0.60.
Q8 model-free p_cdr <= 0.05: 0.25.

## 9. Limits known now
* 300 trees instead of 8000. RF-GAP proximities are noisier, but the same forest settings are
  used for the observed statistic and every null, so the test stays valid.
* Within-day temporal structure beyond the 60-s groups is not modelled. If receivers change
  within a day and acoustics drift within a day, the primary null can be anti-conservative.
  A positive result will therefore be followed by a lag-restricted check (registered as an
  amendment before it runs).
* Fixed effects only (no random effects). Permutation inference replaces the model's
  dependence structure.
* The synthetic worlds are simple Gaussian worlds. They calibrate the test's behaviour on this
  label structure, not the real acoustics.
* The data are the authors' own. Credit for the data and the debate goes to Pardo et al. and
  to Dharmarajan.

## 10. Order of work
Pre-registration (this file), then code with tests, then the job, then a grader published
before any result, then RESULTS-ELEPHANT.md. This file is never edited after publication;
changes go into an AMENDMENT section of a new file.
