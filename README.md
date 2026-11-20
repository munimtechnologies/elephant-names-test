# elephant-names-test

An open, pre-registered test of whether the "name-like calls" of African elephants survive a
receiver-label null that is blocked by caller, day and relationship type.

MV2 of Munim, Inc. (mv2@munimtech.com). Made with AI agents; no human expert has reviewed it
yet. Code Apache-2.0, text CC BY 4.0.

## Result (2026-11-20): UNDERPOWERED
See `RESULTS-ELEPHANT.md`, graded by the grader published before the job returned.
* The authors' public data hold 469 calls from 101 callers. Only **4** (caller, date,
  relationship) blocks contain calls to more than one receiver, with **11** calls in them. A
  null that keeps each caller's day and relationship type fixed can move almost nothing.
* Observed contrast T = +0.11584. Day-blocked null p = 0.3830, caller-date null p = 0.1950,
  naive within-caller null p = 0.0550 (it does not clear 0.05 either).
* Power at a receiver effect that reproduces the contrast: **1 of 20**. Size in a pure-drift
  world: 0 of 20 (the test is valid but nearly blind).
* A pure slow-drift world with no receiver effect reproduces the observed contrast and fools
  the naive within-caller null in **20 of 20** synthetic data sets.
* Reading (fixed before the result): these data cannot decide whether elephant "names"
  survive day-level confounds. Deciding it needs recordings where one caller addresses two or
  more partners of the same kind on the same day. This is not evidence for or against names.

## Why
Pardo et al. (2024, Nature Ecology & Evolution) reported that elephants address each other
with individually specific calls. Dharmarajan (2026, Animal Behaviour) argued that the evidence
is driven by caller identity, temporal autocorrelation and pseudoreplication. Pardo et al.
replied with corrected timestamps, lag-binned permutation tests and a model with date as a
fixed effect (Zenodo record 23284924).
This repo asks a sharper question with the authors' own public data. If receiver labels are
shuffled only among one caller's calls on one day to the same kind of partner, and the whole
pipeline (forest, RF-GAP proximities, contrast) is refitted every time, is the same-receiver
similarity still there? Two synthetic worlds, each tuned to reproduce the observed contrast,
measure the test's power (pure receiver effect) and its size (pure slow drift).

## Files
* `ELEPHANT-PREREG.md`: design, decision rules and predictions, registered 2026-11-20 before
  any statistic was computed.
* `names_null.py` + `test_names_null.py`: the analysis (job `elephant-names-null-v1`).
* `describe_elephant.py` + `test_describe_elephant.py`: label-only description.
* `test_elephant_results.py`: the grader, published before the job returned. Stdlib only:
  `python -m unittest -v test_elephant_results`.
* `results/elephant/elephant_results.json`: the job's raw output, unchanged.
* `RESULTS-ELEPHANT.md`: the graded result.

## Reproduce
```
pip install numpy==2.1.3 scikit-learn==1.5.2
python -m unittest -v test_names_null
python names_null.py --data-dir ./data --out-dir ./out --workers 2
python -m unittest -v test_elephant_results
```
The data download from Zenodo record 23284924 (CC BY 4.0) and are checked by md5. The full run
took about 3.5 hours on 2 CPU cores.

## Credit
Data: Pardo et al. (2024, doi 10.1038/s41559-024-02420-w; Dryad doi
10.5061/dryad.hmgqnk9nj) and the authors of the 2026 reply (Zenodo record 23284924). Critique and controls: Dharmarajan (2026).
RF-GAP proximity: Rhodes, Cutler and Moon (2023).
