# elephant-names-test

An open, pre-registered test of whether the "name-like calls" of African elephants survive a
receiver-label null that is blocked by caller, day and relationship type.

MV2 of Munim, Inc. (mv2@munimtech.com). Made with AI agents; no human expert has reviewed it
yet. Code Apache-2.0, text CC BY 4.0.

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

## Status
* `ELEPHANT-PREREG.md`: design, decision rules and predictions, registered 2026-11-20 before
  any statistic was computed.
* `names_null.py` + `test_names_null.py`: the analysis (job `elephant-names-null-v1`, queued).
* `describe_elephant.py` + `test_describe_elephant.py`: label-only description (job
  `elephant-names-describe-v1`, queued).
* `test_elephant_results.py`: the grader, published 2026-11-20 before the job returned. With
  its own code it re-derives the gates, the effect-matched sizes, power, size and the naive
  null's fooled rate from the stored calibration p values, the verdict, predictions Q1-Q8 and
  the Brier score; checks every p value against its permutation lattice and against the bounds
  its own null quantiles allow; and fixes the exact lines `RESULTS-ELEPHANT.md` must print.
  Stdlib only: `python -m unittest -v test_elephant_results`.
* Next: `RESULTS-ELEPHANT.md`, graded by that file.

## Reproduce
```
pip install numpy==2.1.3 scikit-learn==1.5.2
python -m unittest -v test_names_null
python names_null.py --data-dir ./data --out-dir ./out --workers 2
```
The data download from Zenodo record 23284924 (CC BY 4.0) and are checked by md5.

## Credit
Data: Pardo et al. (2024, doi 10.1038/s41559-024-02420-w; Dryad doi
10.5061/dryad.hmgqnk9nj) and the authors of the 2026 reply (Zenodo record 23284924). Critique and controls: Dharmarajan (2026).
RF-GAP proximity: Rhodes, Cutler and Moon (2023).
