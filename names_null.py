# SPDX-License-Identifier: Apache-2.0
# names_null.py: do elephant "name" calls survive a receiver-label permutation null blocked by
# caller, day and relationship category? Pre-registered in ELEPHANT-PREREG.md (repo
# munimtechnologies/elephant-names-test) before this code was run on the data.
#
# Data: Pardo et al., reply to Dharmarajan (2026), Zenodo record 23284924 (CC BY 4.0): the 469
# rumbles of Pardo et al. 2024 (Nature Ecology & Evolution) with caller, receiver, date, context,
# relationship category, certainty flags and 94 acoustic features, plus the Amboseli timestamp
# fix. The statistic mirrors the reply's own analysis (RF-GAP proximity from a random forest that
# predicts the receiver from the 94 features; same-caller, same-relationship pairs; pairs within
# the same 60-s temporal group and pairs with uncertain IDs or contexts excluded; context and
# date entered as fixed effects). What is new is the null: the receiver labels are permuted
# WITHIN (caller, date, relationship category) blocks and the WHOLE pipeline, forest included, is
# refitted for every permutation. Two synthetic worlds built on the real label structure, each
# tuned to reproduce the observed contrast, measure the test: a pure receiver-effect world
# (power) and a pure slow-drift world with no receiver effect (size; and how often the naive
# within-caller null is fooled).
# MV2, Munim, Inc. (mv2@munimtech.com). Made with AI agents. Apache-2.0.
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import calendar  # noqa: E402
import csv  # noqa: E402
import datetime  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import urllib.request  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402

VERSION = "elephant-names-null-1"
RECORD = "23284924"
URL = "https://zenodo.org/records/%s/files/%s?download=1"
MAIN = "20231225_dfall_obs_data_and_spectral_features_revision1_n469.csv"
AMB = "20260731_Amboseli_sound_files_used_in_names_paper_for_upload.csv"
MD5 = {MAIN: "f9024ced42a489e82c4e41d65ba70aa4", AMB: "4ea38110eb7a48078fe9a45b5e667181"}
NA = {"", "NA", "na", "N/A", "NaN", "nan"}
FEATURES = (["sprsMed", "sprsMbw", "sprsEqbw", "sprsMc"]
            + ["V%d" % i for i in range(1, 75)]
            + ["%s%d" % (p, i) for p in ("Fw", "Mw", "F", "M") for i in range(1, 5)])

GROUP_GAP_SEC = 60.0          # reply: new temporal group when the gap to the caller's previous call > 60 s
N_TREES = 300                 # reply: 8000 (ranger); 300 here so that ~7,000 refits fit the budget
MTRY = 6                      # reply: mtry = 6
SAMPLE_FRACTION = 0.6         # reply: replace = FALSE, sample.fraction = 0.6
FOREST_SEED = 12345           # one forest seed for every fit (common random numbers)
B_PRIMARY = 999               # primary null: within (caller, date, relationship)
B_SIDE = 199                  # side nulls: within (caller, date) and within (caller, relationship)
B_MF = 999                    # model-free statistic, both nulls
GRID = [0.25 * k for k in range(1, 17)]   # 0.25 .. 4.00, synthetic effect sizes for matching
MATCH_REPS = 4
FALLBACK_SIZE = 1.0           # used when the observed contrast is not positive
R_CAL = 20
B_CAL = 99
ALPHA = 0.05
POWER_MIN = 0.80
SIZE_MAX = 0.20               # at most 4 of 20 drift-world rejections under the primary null
MIN_PAIRS = 100
TAU_DAYS = 60.0               # drift-world time scale (Ornstein-Uhlenbeck across dates)
SEEDS = {"cdr": 20261120, "cd": 20261121, "cr": 20261122, "mf_cdr": 20261123,
         "mf_cr": 20261124, "match_receiver": 5000000, "match_drift": 6000000,
         "cal_receiver": 7000000, "cal_drift": 8000000}
SCHEMES = ("cdr", "cd", "cr")
PRIORS = {"Q1": 0.90, "Q2": 0.75, "Q3": 0.30, "Q4": 0.35, "Q5": 0.95, "Q6": 0.50,
          "Q7": 0.60, "Q8": 0.25}


# ---------------------------------------------------------------- parsing
def clean(v):
    v = (v or "").strip()
    return None if v in NA else v


def is_num(v):
    v = clean(v)
    if v is None:
        return False
    try:
        x = float(v)
    except ValueError:
        return False
    return x == x and abs(x) != float("inf")


def num(v):
    return float(clean(v)) if is_num(v) else None


def is_one(v):
    return is_num(v) and float(clean(v)) == 1.0


def site(sound_file):
    """The reply flags Samburu files by a leading '20'; everything else is Amboseli."""
    return "samburu" if (sound_file or "").strip().startswith("20") else "amboseli"


SAMBURU_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})")


def samburu_ts(sound_file, start):
    """Seconds (UTC-naive epoch) of a Samburu call: YYYYMMDD_HHMM from the file name + start."""
    m = SAMBURU_RE.match((sound_file or "").strip())
    if not m or not is_num(start):
        return None
    y, mo, d, h, mi = (int(g) for g in m.groups())
    try:
        return calendar.timegm((y, mo, d, h, mi, 0, 0, 0, 0)) + float(clean(start))
    except (ValueError, OverflowError):
        return None


def _norm(k):
    return re.sub(r"[^a-z0-9]", "", (k or "").lower())


def amboseli_ts(row):
    """Seconds of an Amboseli call from the reply's timestamp file (Date is d/m/Y). If all three
    Call_start fields exist they give the time (seconds truncated); otherwise
    track start + tape-to-call - tape-to-track, as in the reply's code."""
    r = {_norm(k): v for k, v in row.items()}
    try:
        d, m, y = (int(x) for x in clean(r.get("date")).split("/"))
        day0 = calendar.timegm((y, m, d, 0, 0, 0, 0, 0, 0))
    except Exception:
        return None
    ch, cm, cs = num(r.get("callstarthour")), num(r.get("callstartminute")), num(r.get("callstartsecond"))
    if ch is not None and cm is not None and cs is not None:
        return day0 + ch * 3600 + cm * 60 + int(cs)
    th, tm = num(r.get("trackstarthour")), num(r.get("trackstartminute"))
    if th is None or tm is None:
        return None

    def hms(prefix):
        v = [num(r.get(prefix + s)) for s in ("hour", "minute", "second")]
        return None if any(x is None for x in v) else v[0] * 3600 + v[1] * 60 + v[2]

    clip, trk = hms("tapetoclipelapsed"), hms("tapetotrackelapsed")
    c2c = num(r.get("cliptocallelapsedsec"))
    if clip is None or trk is None or c2c is None:
        return None
    return day0 + th * 3600 + tm * 60 + clip + c2c - trk


def load(main_rows, amb_rows):
    amb = {}
    for r in amb_rows:
        k = clean(r.get("soundID"))
        if k:
            amb[k] = r
    calls, info = [], Counter()
    for r in main_rows:
        if not all(is_num(r.get(c)) for c in FEATURES):
            info["dropped_incomplete_features"] += 1
            continue
        caller, rec, date = clean(r.get("CallerID")), clean(r.get("Elicitor1_ID")), clean(r.get("date"))
        if not (caller and rec and date):
            info["dropped_missing_caller_receiver_date"] += 1
            continue
        s = site(r.get("sound.files"))
        if s == "samburu":
            ts = samburu_ts(r.get("sound.files"), r.get("start"))
        else:
            ar = amb.get(clean(r.get("soundID")) or "")
            ts = amboseli_ts(ar) if ar is not None else None
        info[("timed_" if ts is not None else "untimed_") + s] += 1
        calls.append({"id": clean(r.get("soundID")) or "", "caller": caller, "receiver": rec,
                      "date": date, "site": s, "ts": ts, "soc": clean(r.get("newSocCat")),
                      "ctx": clean(r.get("Context2")),
                      "cert": is_one(r.get("Cert_ID_Bin")) and is_one(r.get("Cert_Con_Bin")),
                      "x": [float(clean(r.get(c))) for c in FEATURES]})
    return calls, dict(sorted(info.items()))


def read_rows(blob):
    try:
        text = blob.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = blob.decode("latin-1")
    return list(csv.DictReader(io.StringIO(text)))


# ---------------------------------------------------------------- label structure
def time_groups(calls, gap=GROUP_GAP_SEC):
    """Chained 60-s groups within caller (receiver ignored; a gap of exactly 60 s stays in the
    group), as in the reply. Calls without a timestamp are their own group."""
    g = [None] * len(calls)
    by = defaultdict(list)
    for i, c in enumerate(calls):
        if c["ts"] is None:
            g[i] = "untimed__%d" % i
        else:
            by[c["caller"]].append(i)
    for caller, idx in by.items():
        idx.sort(key=lambda i: (calls[i]["ts"], calls[i]["id"]))
        n, prev = 0, None
        for i in idx:
            if prev is None or calls[i]["ts"] - prev > gap:
                n += 1
            g[i] = "%s__%d" % (caller, n)
            prev = calls[i]["ts"]
    return g


def eligible_pairs(calls, groups):
    """Same caller, same relationship category, both IDs and contexts certain, both contexts
    known, different temporal groups (the reply's model set)."""
    ia, ib = [], []
    by = defaultdict(list)
    for i, c in enumerate(calls):
        by[c["caller"]].append(i)
    for caller in sorted(by):
        idx = by[caller]
        for p in range(len(idx)):
            for q in range(p + 1, len(idx)):
                i, j = idx[p], idx[q]
                a, b = calls[i], calls[j]
                if not (a["cert"] and b["cert"]):
                    continue
                if a["soc"] is None or a["soc"] != b["soc"]:
                    continue
                if a["ctx"] is None or b["ctx"] is None:
                    continue
                if groups[i] == groups[j]:
                    continue
                ia.append(i)
                ib.append(j)
    return np.array(ia, dtype=np.int64), np.array(ib, dtype=np.int64)


def block_key(c, scheme):
    if scheme == "cdr":
        return (c["caller"], c["date"], c["soc"] or "NA")
    if scheme == "cd":
        return (c["caller"], c["date"])
    if scheme == "cr":
        return (c["caller"], c["soc"] or "NA")
    raise ValueError(scheme)


def blocks(calls, scheme):
    b = defaultdict(list)
    for i, c in enumerate(calls):
        b[block_key(c, scheme)].append(i)
    return [np.array(b[k], dtype=np.int64) for k in sorted(b)]


def permute(labels, blks, rng):
    out = labels.copy()
    for idx in blks:
        if len(idx) > 1:
            out[idx] = labels[idx[rng.permutation(len(idx))]]
    return out


def block_info(labels, blks, ia, ib):
    inf = [idx for idx in blks if len(set(labels[idx].tolist())) >= 2]
    lr = 0.0
    for idx in inf:
        cnt = list(Counter(labels[idx].tolist()).values())
        lr += (math.lgamma(sum(cnt) + 1) - sum(math.lgamma(m + 1) for m in cnt)) / math.log(10)
    mark = np.zeros(len(labels), dtype=bool)
    for idx in inf:
        mark[idx] = True
    return {"blocks": len(blks), "informative_blocks": len(inf),
            "calls_in_informative_blocks": int(sum(len(i) for i in inf)),
            "log10_relabelings": round(lr, 4),
            "pairs_with_a_movable_call": int(np.sum(mark[ia] | mark[ib])) if len(ia) else 0}


# ---------------------------------------------------------------- forest and statistic
class Forest:
    """Fixed in-bag sets (60% without replacement) and fixed tree seeds; only the labels and
    the features change between fits."""

    def __init__(self, n, n_trees=N_TREES, seed=FOREST_SEED, frac=SAMPLE_FRACTION):
        rng = np.random.default_rng(seed)
        k = int(frac * n)
        self.inbag = np.zeros((n_trees, n), dtype=bool)
        for t in range(n_trees):
            self.inbag[t, rng.choice(n, size=k, replace=False)] = True
        self.seeds = rng.integers(0, 2 ** 31 - 1, size=n_trees)
        self.n_trees = n_trees

    def leaves(self, X, y):
        from sklearn.tree import DecisionTreeClassifier
        L = np.empty((self.n_trees, X.shape[0]), dtype=np.int64)
        for t in range(self.n_trees):
            m = self.inbag[t]
            tree = DecisionTreeClassifier(max_features=MTRY, min_samples_leaf=1,
                                          random_state=int(self.seeds[t]))
            tree.fit(X[m], y[m])
            L[t] = tree.apply(X)
        return L


def gap_pair_prox(L, inbag, ia, ib):
    """Symmetrised RF-GAP proximity (Rhodes, Cutler & Moon 2023) for the listed pairs:
    p(i,j) = mean over trees where i is out of bag of [j in bag and in i's leaf] / (in-bag
    count of i's leaf); returned as (p(i,j) + p(j,i)) / 2."""
    T, n = L.shape
    oob = ~inbag
    w = np.zeros((T, n))
    for t in range(T):
        size = np.bincount(L[t][inbag[t]], minlength=int(L[t].max()) + 1)
        w[t] = 1.0 / np.maximum(size[L[t]], 1)
    s = oob.sum(axis=0).astype(float)
    same = L[:, ia] == L[:, ib]
    pab = ((same & oob[:, ia] & inbag[:, ib]) * w[:, ia]).sum(axis=0)
    pba = ((same & oob[:, ib] & inbag[:, ia]) * w[:, ib]).sum(axis=0)
    return 0.5 * (pab / np.maximum(s[ia], 1.0) + pba / np.maximum(s[ib], 1.0))


def avg_rank(x):
    """1-based ranks, ties get their average rank."""
    _, inv, cnt = np.unique(np.asarray(x), return_inverse=True, return_counts=True)
    cum = np.cumsum(cnt)
    return (cum - (cnt - 1) / 2.0)[np.asarray(inv).reshape(-1)]


def contrast(y, sr, sc, sd):
    """OLS coefficient of same-receiver in y ~ 1 + sameReceiver + sameContext + sameDate."""
    sr = np.asarray(sr, dtype=float)
    if len(sr) == 0 or sr.min() == sr.max():
        return float("nan")
    X = np.column_stack([np.ones(len(sr)), sr, sc, sd])
    return float(np.linalg.lstsq(X, np.asarray(y, dtype=float), rcond=None)[0][1])


def same_receiver(lab, ia, ib):
    return (lab[ia] == lab[ib]).astype(float)


def t_rf(forest, X, lab, ia, ib, sc, sd):
    L = forest.leaves(X, lab)
    prox = gap_pair_prox(L, forest.inbag, ia, ib)
    y = avg_rank(prox) / len(prox)
    return contrast(y, same_receiver(lab, ia, ib), sc, sd)


def mf_y(X, ia, ib):
    """Model-free similarity: rank of minus the Euclidean distance on z-scored features."""
    X = np.asarray(X, dtype=float)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Z = (X - X.mean(axis=0)) / sd
    d = np.sqrt(((Z[ia] - Z[ib]) ** 2).sum(axis=1))
    return avg_rank(-d) / len(d)


def pvalue(obs, null):
    null = np.asarray(null, dtype=float)
    if obs is None or not np.isfinite(obs):
        return float("nan")
    ge = int(np.sum(~np.isfinite(null) | (null >= obs - 1e-12)))
    return (1 + ge) / (len(null) + 1)


def reject_rate(ps):
    ps = [p for p in ps]
    return sum(1 for p in ps if p == p and p <= ALPHA) / len(ps) if ps else float("nan")


# ---------------------------------------------------------------- parallel refits
_W = {}


def _init(Xs, ia, ib, sc, sd, n_trees, seed):
    _W.clear()
    n = next(iter(Xs.values())).shape[0]
    _W.update(Xs=Xs, ia=ia, ib=ib, sc=sc, sd=sd, forest=Forest(n, n_trees, seed))


def _task(arg):
    key, lab = arg
    return t_rf(_W["forest"], _W["Xs"][key], lab, _W["ia"], _W["ib"], _W["sc"], _W["sd"])


def run_tasks(Xs, tasks, ia, ib, sc, sd, n_trees, seed, workers):
    if workers <= 1:
        _init(Xs, ia, ib, sc, sd, n_trees, seed)
        return [_task(t) for t in tasks]
    with ProcessPoolExecutor(max_workers=workers, initializer=_init,
                             initargs=(Xs, ia, ib, sc, sd, n_trees, seed)) as ex:
        return list(ex.map(_task, tasks, chunksize=4))


# ---------------------------------------------------------------- synthetic worlds
def keyed_effects(keys, rng, p):
    uniq = sorted(set(keys))
    E = rng.standard_normal((len(uniq), p))
    pos = {k: i for i, k in enumerate(uniq)}
    return E[np.array([pos[k] for k in keys], dtype=np.int64)]


def date_ordinal(c):
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%d/%m/%Y"):
        try:
            return datetime.datetime.strptime(c["date"], fmt).date().toordinal()
        except (ValueError, TypeError):
            pass
    if c["ts"] is not None:
        return int(c["ts"] // 86400) + 719163
    return None


def drift_effects(calls, rng, p, tau=TAU_DAYS):
    """One latent vector per (caller, date), an Ornstein-Uhlenbeck chain across each caller's
    dates (correlation exp(-gap/tau)); constant within a day, so labels stay exchangeable
    within (caller, date) blocks."""
    ordmap = {}
    for c in calls:
        ordmap.setdefault((c["caller"], c["date"]), date_ordinal(c))
    by = defaultdict(list)
    for k in sorted(ordmap):
        by[k[0]].append(k)
    E = {}
    for caller in sorted(by):
        ks = sorted(by[caller], key=lambda k: (ordmap[k] is None, ordmap[k] or 0, k[1]))
        prev, prev_o = None, None
        for k in ks:
            z = rng.standard_normal(p)
            o = ordmap[k]
            if prev is None or o is None or prev_o is None:
                v = z
            else:
                rho = math.exp(-abs(o - prev_o) / tau)
                v = rho * prev + math.sqrt(max(0.0, 1.0 - rho * rho)) * z
            E[k] = v
            prev, prev_o = v, o
    return np.array([E[(c["caller"], c["date"])] for c in calls])


def synth(calls, kind, size, seed, p=len(FEATURES)):
    """caller effect + noise (both unit variance) + size * (receiver-dyad effect | drift)."""
    rng = np.random.default_rng(seed)
    n = len(calls)
    X = keyed_effects([c["caller"] for c in calls], rng, p) + rng.standard_normal((n, p))
    if kind == "receiver":
        return X + size * keyed_effects([(c["caller"], c["receiver"]) for c in calls], rng, p)
    if kind == "drift":
        return X + size * drift_effects(calls, rng, p)
    raise ValueError(kind)


def match(kind, calls, lab, t_obs, run, grid, reps, seed0):
    Xs, tasks = {}, []
    for gi, g in enumerate(grid):
        for r in range(reps):
            key = "%s_%d_%d" % (kind, gi, r)
            Xs[key] = synth(calls, kind, g, seed0 + 1000 * gi + r)
            tasks.append((key, lab))
    ts = run(Xs, tasks)
    means = []
    for gi in range(len(grid)):
        v = np.array(ts[gi * reps:(gi + 1) * reps], dtype=float)
        means.append(float(np.mean(v[np.isfinite(v)])) if np.isfinite(v).any() else float("nan"))
    out = {"grid": list(grid), "mean_t": means}
    finite = [m for m in means if m == m]
    out["max_mean_t"] = max(finite) if finite else float("nan")
    if not (t_obs == t_obs) or t_obs <= 0:
        out.update(size=FALLBACK_SIZE, matched=False, reached=False)
        return out
    d = [abs(m - t_obs) if m == m else float("inf") for m in means]
    gi = int(np.argmin(d))
    out.update(size=grid[gi], matched=True, mean_t_at_size=means[gi],
               reached=bool(out["max_mean_t"] >= t_obs))
    return out


def calibrate(kind, size, schemes, calls, lab, blks, run, R, B, seed0):
    Xs, tasks = {}, []
    for r in range(R):
        key = "%s_cal_%d" % (kind, r)
        Xs[key] = synth(calls, kind, size, seed0 + r)
        tasks.append((key, lab))
        for si, s in enumerate(schemes):
            rng = np.random.default_rng(seed0 + 100000 * (si + 1) + r)
            for _ in range(B):
                tasks.append((key, permute(lab, blks[s], rng)))
    ts = run(Xs, tasks)
    obs, ps, k = [], {s: [] for s in schemes}, 0
    for r in range(R):
        t0 = ts[k]
        k += 1
        obs.append(t0)
        for s in schemes:
            ps[s].append(pvalue(t0, ts[k:k + B]))
            k += B
    return {"size": size, "R": R, "B": B, "t_obs": obs, "p": ps,
            "reject_rate": {s: reject_rate(ps[s]) for s in schemes}}


# ---------------------------------------------------------------- the analysis
def verdict_of(t_obs, p_cdr, power, size_rate):
    if not (size_rate == size_rate) or size_rate > SIZE_MAX:
        return "CALIBRATION_FAILED"
    if not (t_obs == t_obs) or t_obs <= 0:
        return "NO_POSITIVE_CONTRAST"
    if p_cdr <= ALPHA:
        return "RECEIVER_SIGNAL_SURVIVES_DAY_BLOCKING"
    if power >= POWER_MIN:
        return "NOT_DETECTED_WITH_POWER"
    return "UNDERPOWERED"


def predictions(r):
    out = {"Q1": r["t_obs"] > 0, "Q2": r["p"]["cr"] <= ALPHA, "Q3": r["p"]["cdr"] <= ALPHA,
           "Q4": r["power"] >= POWER_MIN, "Q5": r["drift_size_rate"] <= SIZE_MAX,
           "Q6": r["drift_fooled_rate_cr"] >= 0.5, "Q7": bool(r["match"]["drift"]["reached"]),
           "Q8": r["model_free"]["p"]["cdr"] <= ALPHA}
    out = {k: bool(v) for k, v in out.items()}
    brier = sum((PRIORS[k] - (1.0 if v else 0.0)) ** 2 for k, v in out.items()) / len(out)
    return out, brier


def analyse(calls, workers=1, n_trees=N_TREES, b_primary=B_PRIMARY, b_side=B_SIDE, b_mf=B_MF,
            grid=GRID, match_reps=MATCH_REPS, r_cal=R_CAL, b_cal=B_CAL, min_pairs=MIN_PAIRS,
            log=print):
    t0 = time.time()
    groups = time_groups(calls)
    ia, ib = eligible_pairs(calls, groups)
    recv = sorted(set(c["receiver"] for c in calls))
    code = {r: i for i, r in enumerate(recv)}
    lab = np.array([code[c["receiver"]] for c in calls], dtype=np.int64)
    sc = np.array([calls[i]["ctx"] == calls[j]["ctx"] for i, j in zip(ia, ib)], dtype=float)
    sd = np.array([calls[i]["date"] == calls[j]["date"] for i, j in zip(ia, ib)], dtype=float)
    blks = {s: blocks(calls, s) for s in SCHEMES}
    sr = same_receiver(lab, ia, ib)
    res = {"n_calls": len(calls), "n_callers": len(set(c["caller"] for c in calls)),
           "n_receivers": len(recv), "n_dates": len(set(c["date"] for c in calls)),
           "n_time_groups": len(set(groups)), "n_pairs": int(len(ia)),
           "n_pairs_same_receiver": int(sr.sum()), "n_pairs_same_date": int(sd.sum()),
           "n_pairs_same_context": int(sc.sum()),
           "blocks": {s: block_info(lab, blks[s], ia, ib) for s in SCHEMES},
           "settings": {"n_trees": n_trees, "mtry": MTRY, "sample_fraction": SAMPLE_FRACTION,
                        "forest_seed": FOREST_SEED, "b_primary": b_primary, "b_side": b_side,
                        "b_mf": b_mf, "grid": list(grid), "match_reps": match_reps,
                        "r_cal": r_cal, "b_cal": b_cal, "alpha": ALPHA, "seeds": SEEDS,
                        "tau_days": TAU_DAYS, "group_gap_sec": GROUP_GAP_SEC}}
    log(json.dumps({k: res[k] for k in res if k != "settings"}))
    if len(ia) < min_pairs or res["blocks"]["cdr"]["informative_blocks"] == 0 or sr.min() == sr.max():
        res["verdict"] = "NO_INFORMATIVE_BLOCKS"
        return res
    X = np.array([c["x"] for c in calls], dtype=float)

    def run(Xs, tasks):
        return run_tasks(Xs, tasks, ia, ib, sc, sd, n_trees, FOREST_SEED, workers)

    # 1. the RF-GAP contrast on the real data and its three nulls
    tasks = [("real", lab)]
    counts = {"cdr": b_primary, "cd": b_side, "cr": b_side}
    for s in SCHEMES:
        rng = np.random.default_rng(SEEDS[s])
        tasks += [("real", permute(lab, blks[s], rng)) for _ in range(counts[s])]
    ts = run({"real": X}, tasks)
    t_obs, k, nulls = ts[0], 1, {}
    for s in SCHEMES:
        nulls[s] = np.array(ts[k:k + counts[s]], dtype=float)
        k += counts[s]
    res["t_obs"] = t_obs
    res["p"] = {s: pvalue(t_obs, nulls[s]) for s in SCHEMES}
    res["null"] = {s: {"mean": float(np.nanmean(nulls[s])), "sd": float(np.nanstd(nulls[s])),
                       "q05": float(np.nanquantile(nulls[s], 0.05)),
                       "q95": float(np.nanquantile(nulls[s], 0.95)),
                       "n_nan": int(np.sum(~np.isfinite(nulls[s])))} for s in SCHEMES}
    log("real data done %.0fs: t_obs %.5f p %s" % (time.time() - t0, t_obs, res["p"]))

    # 2. model-free contrast (z-scored Euclidean distance; no forest)
    ymf = mf_y(X, ia, ib)
    tmf = contrast(ymf, sr, sc, sd)
    mf = {"t_obs": tmf, "p": {}}
    for s, seed in (("cdr", SEEDS["mf_cdr"]), ("cr", SEEDS["mf_cr"])):
        rng = np.random.default_rng(seed)
        null = [contrast(ymf, same_receiver(permute(lab, blks[s], rng), ia, ib), sc, sd)
                for _ in range(b_mf)]
        mf["p"][s] = pvalue(tmf, null)
    res["model_free"] = mf
    log("model-free %.5f p %s" % (tmf, mf["p"]))

    # 3. effect matching: synthetic worlds that reproduce the observed contrast
    res["match"] = {
        "receiver": match("receiver", calls, lab, t_obs, run, grid, match_reps,
                          SEEDS["match_receiver"]),
        "drift": match("drift", calls, lab, t_obs, run, grid, match_reps, SEEDS["match_drift"])}
    log("matching done %.0fs: receiver %s drift %s" % (
        time.time() - t0, res["match"]["receiver"]["size"], res["match"]["drift"]["size"]))

    # 4. calibration at the matched sizes
    rc = calibrate("receiver", res["match"]["receiver"]["size"], ("cdr",), calls, lab, blks, run,
                   r_cal, b_cal, SEEDS["cal_receiver"])
    dc = calibrate("drift", res["match"]["drift"]["size"], ("cdr", "cr"), calls, lab, blks, run,
                   r_cal, b_cal, SEEDS["cal_drift"])
    res["calibration"] = {"receiver": rc, "drift": dc}
    res["power"] = rc["reject_rate"]["cdr"]
    res["drift_size_rate"] = dc["reject_rate"]["cdr"]
    res["drift_fooled_rate_cr"] = dc["reject_rate"]["cr"]
    res["verdict"] = verdict_of(t_obs, res["p"]["cdr"], res["power"], res["drift_size_rate"])
    res["predictions"], res["brier"] = predictions(res)
    res["seconds"] = round(time.time() - t0, 1)
    log("verdict %s power %.2f size %.2f fooled %.2f (%.0fs)" % (
        res["verdict"], res["power"], res["drift_size_rate"], res["drift_fooled_rate_cr"],
        res["seconds"]))
    return res


# ---------------------------------------------------------------- data and main
def fetch(url, timeout=180):
    req = urllib.request.Request(url, headers={"User-Agent": "elephant-names-test/1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def get_file(data_dir, name):
    path = os.path.join(data_dir, name)
    for attempt in range(2):
        if attempt or not os.path.exists(path):
            data = fetch(URL % (RECORD, name))
            with open(path + ".part", "wb") as f:
                f.write(data)
            os.replace(path + ".part", path)
        with open(path, "rb") as f:
            blob = f.read()
        if hashlib.md5(blob).hexdigest() == MD5[name]:
            return blob, True
    return blob, False


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/data/elephant")
    ap.add_argument("--out-dir", default="/out")
    ap.add_argument("--workers", type=int, default=min(2, os.cpu_count() or 1))
    a = ap.parse_args(argv)
    os.makedirs(a.data_dir, exist_ok=True)
    os.makedirs(a.out_dir, exist_ok=True)
    out = {"version": VERSION, "python": sys.version.split()[0], "numpy": np.__version__,
           "record": RECORD, "utc_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "workers": a.workers}
    try:
        import sklearn
        out["sklearn"] = sklearn.__version__
    except ImportError:
        out["sklearn"] = None
    blobs, ok = {}, {}
    for name in (MAIN, AMB):
        blobs[name], ok[name] = get_file(a.data_dir, name)
    out["md5_ok"] = ok
    path = os.path.join(a.out_dir, "elephant_results.json")
    if not all(ok.values()):
        out["verdict"] = "DATA_MISMATCH"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
        print(json.dumps(out))
        return 0
    calls, info = load(read_rows(blobs[MAIN]), read_rows(blobs[AMB]))
    out["load"] = info
    out["n_usable"] = len(calls)
    out["provisional"] = len(calls) < 460
    out.update(analyse(calls, workers=a.workers, log=lambda m: print(m, flush=True)))
    out["utc_end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out.get(k) for k in ("verdict", "t_obs", "p", "power",
                                               "drift_size_rate", "drift_fooled_rate_cr",
                                               "brier")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
