# RESULTS-ELEPHANT: do elephant "name" calls survive a receiver-label null blocked by caller, day and relationship?

MV2 of Munim, Inc. (mv2@munimtech.com), 2026-11-20. Made with AI agents; no human expert has
reviewed this result yet. Text CC BY 4.0, code Apache-2.0.

Design, rules and predictions: `ELEPHANT-PREREG.md` (registered 2026-11-20, before any
statistic). Job `elephant-names-null-v1` ran 2026-11-20 06:47-10:20 UTC (python 3.12.16, numpy
2.1.3, scikit-learn 1.5.2). Raw output: `results/elephant/elephant_results.json`. Graded by
`test_elephant_results.py`, which was published before the job returned. Every fixed line below
is the grader's `render_md_lines` output and is checked by `check_md`.

## Verdict

Verdict: **`UNDERPOWERED`**

In plain words: these public data cannot decide whether the receiver signal survives day-level
confounds. Only 4 (caller, date, relationship) blocks hold calls to more than one receiver, and
they contain 11 of the 469 calls. So a null that keeps each caller's day and relationship type
fixed can move almost nothing. At an effect size that reproduces the observed contrast, a pure
"names" effect was detected in 1 of 20 synthetic data sets. Under section 7 of the
pre-registration, this is the UNDERPOWERED reading: the share of calls in informative blocks is
the reason, and that is a design lesson for future recordings. It is not evidence for or
against elephant names.

## Data and label structure

Code version `elephant-names-null-1`; data md5 match: main file yes, Amboseli timestamp file yes

Usable calls: **469** (provisional, fewer than 460: no); dropped for incomplete features 0, for missing caller, receiver or date 0; with a timestamp: Samburu 371, Amboseli 98; without: Samburu 0, Amboseli 0

101 callers, 117 receivers, 154 dates, 379 temporal groups; 1,181 eligible pairs, of which 991 same receiver, 274 same date, 911 same context

| blocks | n | informative | calls in informative | log10 relabelings | movable pairs |
|---|---|---|---|---|---|
| caller, date, relationship (primary) | 301 | 4 | 11 | 1.7 | 36 |
| caller, date | 272 | 29 | 109 | 21.2 | 248 |
| caller, relationship (naive) | 191 | 22 | 131 | 25.5 | 391 |

## Observed contrast and nulls

Observed contrast T_obs = **+0.11584** (OLS coefficient of same receiver on rank(RF-GAP proximity)/n, with same context and same date)

| null | B | null mean | null sd | null 95% quantile | p |
|---|---|---|---|---|---|
| caller, date, relationship (primary) | 999 | +0.11043 | 0.01254 | +0.12782 | **0.3830** |
| caller, date | 199 | +0.09834 | 0.01943 | +0.12926 | **0.1950** |
| caller, relationship (naive) | 199 | +0.07641 | 0.02663 | +0.11410 | **0.0550** |

Model-free contrast (z-scored Euclidean distance, no forest): T = +0.03001; p within caller, date, relationship = 0.0740 (B = 999); p within caller, relationship = 0.1160 (B = 999)

## Effect matching (synthetic worlds on the real label structure)

Receiver (names) world: matched size **0.25** (mean T there +0.08318); largest mean T on the grid +0.58878; reaches T_obs: yes

Drift world: matched size **0.75** (mean T there +0.13052); largest mean T on the grid +0.22416; reaches T_obs: yes

| synthetic size | receiver-world mean T | drift-world mean T |
|---|---|---|
| 0.25 | +0.08318 | +0.02631 |
| 0.50 | +0.25019 | +0.09662 |
| 0.75 | +0.41692 | +0.13052 |
| 1.00 | +0.51001 | +0.17781 |
| 1.25 | +0.54291 | +0.19705 |
| 1.50 | +0.57223 | +0.19327 |
| 1.75 | +0.57950 | +0.20358 |
| 2.00 | +0.58164 | +0.21349 |
| 2.25 | +0.57969 | +0.21070 |
| 2.50 | +0.57962 | +0.22177 |
| 2.75 | +0.58545 | +0.21582 |
| 3.00 | +0.58821 | +0.21430 |
| 3.25 | +0.58604 | +0.22245 |
| 3.50 | +0.58878 | +0.22090 |
| 3.75 | +0.58430 | +0.21011 |
| 4.00 | +0.58658 | +0.22416 |

## Calibration (R = 20 synthetic data sets per world)

Power (receiver world at size 0.25, primary null, B = 99): **1 of 20** rejected; rate 0.05

Size (drift world at size 0.75, primary null, B = 99): **0 of 20** rejected; rate 0.00

Fooled (drift world at size 0.75, naive within-caller null, B = 99): **20 of 20** rejected; rate 1.00

## Predictions (priors fixed in ELEPHANT-PREREG.md section 8)

4 of 8 graded predictions true

Brier score **0.158**

| id | prediction | prior | outcome |
|---|---|---|---|
| Q1 | the observed contrast is positive (T_obs > 0) | 0.90 | **true** |
| Q2 | the naive within-caller null rejects (p_cr <= 0.05) | 0.75 | **false** |
| Q3 | the primary day-blocked null rejects (p_cdr <= 0.05) | 0.30 | **false** |
| Q4 | power at the matched receiver effect is at least 0.80 | 0.35 | **false** |
| Q5 | the size of the primary null in the drift world is at most 0.20 | 0.95 | **true** |
| Q6 | the naive null is fooled by drift in at least half of the drift-world sets | 0.50 | **true** |
| Q7 | the drift world reaches the observed contrast on the grid | 0.60 | **true** |
| Q8 | the model-free contrast rejects under the primary null (p <= 0.05) | 0.25 | **false** |

## Readings fixed by the pre-registration (section 7)

* **Primary (UNDERPOWERED):** the day-blocked null has almost nothing to permute (11 of 469
  calls, about 48 distinct relabelings), so its power at the matched effect is 0.05. The
  data cannot tell names apart from day-level structure with this test.
* **Side reading, p_cr versus p_cdr:** even the naive within-caller null, which pools all
  dates, gives p = 0.0550 with this pipeline (prior 0.75 that it would reject). In our
  300-tree, fixed-effects version, the same-receiver contrast does not clear 0.05 even without
  day blocking.
* **Side reading, fooled = 1.00 (>= 0.5):** in a world with no receiver effect, slow
  caller-level drift (60-day time scale) that reproduces the observed contrast made the naive
  within-caller null reject in 20 of 20 data sets. In this label structure, the naive null
  cannot be trusted to separate names from drift.
* **Side reading, reached = yes:** slow drift of this form can by itself produce a contrast
  as large as the observed one (mean T 0.13052 at size 0.75 against 0.11584 observed). The
  pure-drift story is therefore not ruled out.
* The primary null held its size in the drift world (0 of 20), so the test is valid. It is
  just not able to see a matched effect in these data.

## Exploratory notes (not pre-registered; read with care)

* 991 of the 1,181 eligible same-caller pairs are same-receiver pairs, so most callers in
  these data address one partner of a given kind. Within one caller, day and relationship
  type, two or more receivers occur in only 4 blocks. Receiver identity is therefore nearly
  inseparable from caller-day structure. Separating the two needs recordings in which the
  same caller addresses two or more partners of the same kind on the same day.
* The receiver world reaches the observed contrast already at the smallest grid size, 0.25.
  The power figure therefore refers to a small effect, the one that matches T_obs.

## Limits

300 trees rather than the authors' 8,000; fixed effects only; simple Gaussian synthetic worlds;
no within-day temporal model beyond 60-s groups. The data and the debate are the work of
Pardo et al. (2024; reply 2026, Zenodo record 23284924) and Dharmarajan (2026). This result was
produced and graded with AI agents working in MV2's name, and no human expert has checked it
yet.
