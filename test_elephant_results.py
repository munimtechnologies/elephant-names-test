# SPDX-License-Identifier: Apache-2.0
# test_elephant_results.py: the grader for the pre-registered elephant "names" test
# (ELEPHANT-PREREG.md, job elephant-names-null-v1), written and published BEFORE the job has
# produced any result. With its own code (it does not import names_null.py) it re-derives, from
# the raw job output: the data and label-structure gates; the effect-matched sizes (grid value
# whose mean T is closest to T_obs, ties to the smaller; fallback 1.0 when T_obs <= 0) and the
# "reached" flags; power, size and the naive null's fooled rate from the stored calibration p
# values; the verdict (rules of ELEPHANT-PREREG.md section 7, in order); predictions Q1-Q8 and the
# Brier score. It checks every p value against its permutation lattice (1 + k) / (B + 1), the
# p values against the bounds their own stored null quantiles allow, the block counts for
# nesting ((caller, date, relationship) refines both side schemes), the settings against the
# registration, and everything against the job's own verdict, predictions and Brier score. It
# fixes now the exact lines RESULTS-ELEPHANT.md must print (render_md_lines). Until
# results/elephant/elephant_results.json exists the result tests are skipped; the toy tests
# always run. Stdlib only; reads files only if they exist and writes nothing.
# MV2, Munim, Inc. (mv2@munimtech.com). Made with AI agents. Apache-2.0.
import copy
import json
import math
import os
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results", "elephant", "elephant_results.json")
MD = os.path.join(HERE, "RESULTS-ELEPHANT.md")
HAVE = os.path.exists(RES)

VERSION = "elephant-names-null-1"
RECORD = "23284924"
MAIN = "20231225_dfall_obs_data_and_spectral_features_revision1_n469.csv"
AMB = "20260731_Amboseli_sound_files_used_in_names_paper_for_upload.csv"
ALPHA = 0.05
POWER_MIN = 0.80
SIZE_MAX = 0.20
FOOLED_MIN = 0.5
MIN_PAIRS = 100
N_USABLE_MIN = 460
FALLBACK_SIZE = 1.0
GRID = [0.25 * k for k in range(1, 17)]
R_CAL = 20
B_CAL = 99
SCHEMES = ("cdr", "cd", "cr")
B_NULL = {"cdr": 999, "cd": 199, "cr": 199}
B_MF = 999
SEEDS = {"cdr": 20261120, "cd": 20261121, "cr": 20261122, "mf_cdr": 20261123,
         "mf_cr": 20261124, "match_receiver": 5000000, "match_drift": 6000000,
         "cal_receiver": 7000000, "cal_drift": 8000000}
EXPECTED_SETTINGS = {"n_trees": 300, "mtry": 6, "sample_fraction": 0.6, "forest_seed": 12345,
                     "b_primary": 999, "b_side": 199, "b_mf": 999, "grid": GRID,
                     "match_reps": 4, "r_cal": 20, "b_cal": 99, "alpha": 0.05, "seeds": SEEDS,
                     "tau_days": 60.0, "group_gap_sec": 60.0}
CAL_SCHEMES = {"receiver": ("cdr",), "drift": ("cdr", "cr")}
TIMED_KEYS = ("timed_samburu", "timed_amboseli", "untimed_samburu", "untimed_amboseli")
INT_KEYS = ("n_calls", "n_callers", "n_receivers", "n_dates", "n_time_groups", "n_pairs",
            "n_pairs_same_receiver", "n_pairs_same_date", "n_pairs_same_context")
BLOCK_KEYS = ("blocks", "informative_blocks", "calls_in_informative_blocks",
              "pairs_with_a_movable_call")
ORDER = ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8")
PRIORS = {"Q1": 0.90, "Q2": 0.75, "Q3": 0.30, "Q4": 0.35, "Q5": 0.95, "Q6": 0.50,
          "Q7": 0.60, "Q8": 0.25}
PRED_TEXT = {"Q1": "the observed contrast is positive (T_obs > 0)",
             "Q2": "the naive within-caller null rejects (p_cr <= 0.05)",
             "Q3": "the primary day-blocked null rejects (p_cdr <= 0.05)",
             "Q4": "power at the matched receiver effect is at least 0.80",
             "Q5": "the size of the primary null in the drift world is at most 0.20",
             "Q6": "the naive null is fooled by drift in at least half of the drift-world sets",
             "Q7": "the drift world reaches the observed contrast on the grid",
             "Q8": "the model-free contrast rejects under the primary null (p <= 0.05)"}
SCHEME_TEXT = {"cdr": "caller, date, relationship (primary)", "cd": "caller, date",
               "cr": "caller, relationship (naive)"}
WORLD = {"receiver": "Receiver (names) world", "drift": "Drift world"}
DM = "DATA_MISMATCH"
NIB = "NO_INFORMATIVE_BLOCKS"


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def finite(x):
    return (isinstance(x, (int, float)) and not isinstance(x, bool) and x == x
            and abs(x) != float("inf"))


def is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def close(a, b, tol=1e-9):
    if a is None or b is None:
        return a is None and b is None
    if not (is_num(a) and is_num(b)):
        return False
    if a != a and b != b:
        return True
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def p_ok(p, n):
    """p = (1 + k) / (1 + n) for an integer 0 <= k <= n."""
    if not finite(p):
        return False
    k = p * (n + 1) - 1
    return abs(k - round(k)) < 1e-6 and 0 <= round(k) <= n


def p_valid(p, n, t_finite):
    """A p value from names_null.pvalue: on the lattice if the statistic is finite, else nan."""
    if t_finite:
        return p_ok(p, n)
    return is_num(p) and p != p


def argmin(v):
    """Index of the smallest value; the lowest index on ties (as numpy.argmin)."""
    best = 0
    for i in range(1, len(v)):
        if v[i] < v[best]:
            best = i
    return best


def le_alpha(p):
    return finite(p) and p <= ALPHA


def n_rej(ps):
    return sum(1 for p in ps if le_alpha(p))


def reject_rate(ps):
    return n_rej(ps) / len(ps) if ps else float("nan")


# --------------------------------------------------------------------------
# independent re-implementation of ELEPHANT-PREREG.md sections 3, 6 and 7
# --------------------------------------------------------------------------
def gate_verdict(r):
    md5 = r.get("md5_ok") or {}
    if not md5 or not all(md5.values()):
        return DM
    n, s = r.get("n_pairs"), r.get("n_pairs_same_receiver")
    inf = ((r.get("blocks") or {}).get("cdr") or {}).get("informative_blocks")
    if n < MIN_PAIRS or inf == 0 or s == 0 or s == n:
        return NIB
    return None


def match_expect(means, t_obs, grid):
    fin = [m for m in means if finite(m)]
    mx = max(fin) if fin else float("nan")
    if not finite(t_obs) or t_obs <= 0:
        return {"size": FALLBACK_SIZE, "matched": False, "reached": False, "max_mean_t": mx,
                "mean_t_at_size": None}
    d = [abs(m - t_obs) if finite(m) else float("inf") for m in means]
    gi = argmin(d)
    return {"size": grid[gi], "matched": True, "reached": bool(finite(mx) and mx >= t_obs),
            "max_mean_t": mx, "mean_t_at_size": means[gi]}


def verdict_of(t_obs, p_cdr, power, size):
    if not finite(size) or size > SIZE_MAX:
        return "CALIBRATION_FAILED"
    if not finite(t_obs) or t_obs <= 0:
        return "NO_POSITIVE_CONTRAST"
    if le_alpha(p_cdr):
        return "RECEIVER_SIGNAL_SURVIVES_DAY_BLOCKING"
    if finite(power) and power >= POWER_MIN:
        return "NOT_DETECTED_WITH_POWER"
    return "UNDERPOWERED"


def regrade(r):
    out = {"verdict": gate_verdict(r), "power": None, "drift_size_rate": None,
           "drift_fooled_rate_cr": None, "counts": None, "sizes": None, "match": None,
           "predictions": None, "brier": None, "n_true": None}
    if out["verdict"]:
        return out
    t = r["t_obs"]
    C = r["calibration"]
    rp, dp, fp = C["receiver"]["p"]["cdr"], C["drift"]["p"]["cdr"], C["drift"]["p"]["cr"]
    power, size, fooled = reject_rate(rp), reject_rate(dp), reject_rate(fp)
    me = {k: match_expect(r["match"][k]["mean_t"], t, r["match"][k]["grid"])
          for k in ("receiver", "drift")}
    pred = {"Q1": finite(t) and t > 0, "Q2": le_alpha(r["p"]["cr"]),
            "Q3": le_alpha(r["p"]["cdr"]), "Q4": power >= POWER_MIN, "Q5": size <= SIZE_MAX,
            "Q6": fooled >= FOOLED_MIN, "Q7": me["drift"]["reached"],
            "Q8": le_alpha(r["model_free"]["p"]["cdr"])}
    pred = {k: bool(pred[k]) for k in ORDER}
    brier = sum((PRIORS[k] - (1.0 if pred[k] else 0.0)) ** 2 for k in ORDER) / len(ORDER)
    out.update(verdict=verdict_of(t, r["p"]["cdr"], power, size), power=power,
               drift_size_rate=size, drift_fooled_rate_cr=fooled,
               counts={"power": n_rej(rp), "size": n_rej(dp), "fooled": n_rej(fp),
                       "R": len(rp)},
               sizes={k: me[k]["size"] for k in me}, match=me, predictions=pred, brier=brier,
               n_true=sum(1 for k in ORDER if pred[k]))
    return out


# --------------------------------------------------------------------------
# internal consistency of the raw output
# --------------------------------------------------------------------------
def check_structure(r):
    bad = [k for k in INT_KEYS if not (is_int(r.get(k)) and r[k] >= 0)]
    if bad:
        return ["count " + k for k in bad]
    n, npairs = r["n_calls"], r["n_pairs"]
    for k in ("n_pairs_same_receiver", "n_pairs_same_date", "n_pairs_same_context"):
        if r[k] > npairs:
            bad.append(k + " exceeds pairs")
    if not (1 <= r["n_callers"] <= r["n_time_groups"] <= n):
        bad.append("temporal groups")
    if not (r["n_receivers"] <= n and 1 <= r["n_dates"] <= n):
        bad.append("receivers or dates")
    b = r.get("blocks")
    if not isinstance(b, dict) or sorted(b) != sorted(SCHEMES):
        return bad + ["blocks"]
    for s in SCHEMES:
        x = b[s]
        if not (isinstance(x, dict) and all(is_int(x.get(k)) for k in BLOCK_KEYS)
                and finite(x.get("log10_relabelings"))):
            bad.append("blocks " + s)
            continue
        okb = (r["n_callers"] <= x["blocks"] <= n
               and 0 <= x["informative_blocks"] <= x["blocks"]
               and 2 * x["informative_blocks"] <= x["calls_in_informative_blocks"] <= n
               and 0 <= x["pairs_with_a_movable_call"] <= npairs
               and x["log10_relabelings"] >= 0
               and (x["informative_blocks"] > 0
                    or (x["calls_in_informative_blocks"] == 0
                        and x["log10_relabelings"] == 0)))
        if s == "cd" and x["blocks"] < r["n_dates"]:
            okb = False
        if not okb:
            bad.append("blocks " + s)
    if any(m.startswith("blocks") for m in bad):
        return bad
    c = b["cdr"]
    for parent in ("cd", "cr"):
        q = b[parent]
        if not (c["blocks"] >= q["blocks"]
                and c["calls_in_informative_blocks"] <= q["calls_in_informative_blocks"]
                and c["log10_relabelings"] <= q["log10_relabelings"] + 1e-3
                and c["pairs_with_a_movable_call"] <= q["pairs_with_a_movable_call"]):
            bad.append("blocks cdr not nested in " + parent)
    return bad


def null_ok(x, n):
    if not isinstance(x, dict):
        return False
    nn = x.get("n_nan")
    if not (is_int(nn) and 0 <= nn <= n):
        return False
    if nn == n:
        return True
    return (all(finite(x.get(k)) for k in ("mean", "sd", "q05", "q95")) and x["sd"] >= 0
            and x["q05"] <= x["q95"] + 1e-12)


def p_in_bounds(t, p, x, n):
    """numpy's linear quantiles bound how many finite null values can reach t:
    t > q95 -> at most nf - 1 - floor(0.95 (nf - 1)) of them; t < q05 -> at least
    nf - ceil(0.05 (nf - 1)). Undefined null values always count (names_null.pvalue)."""
    nn = x["n_nan"]
    nf = n - nn
    if nf == 0:
        return close(p, 1.0)
    if t > x["q95"] + 1e-12:
        most = nf - 1 - math.floor((nf - 1) * 0.95 + 1e-9) + nn
        if p > (1 + most) / (n + 1) + 1e-12:
            return False
    if t < x["q05"] - 1e-12:
        least = nf - math.ceil((nf - 1) * 0.05 - 1e-9) + nn
        if p < (1 + least) / (n + 1) - 1e-12:
            return False
    return True


def check_main(r):
    t = r.get("t_obs")
    if not is_num(t):
        return ["t_obs missing"]
    bad = []
    tf = finite(t)
    p = r.get("p") or {}
    nl = r.get("null") or {}
    for s in SCHEMES:
        ps = p.get(s)
        if not p_valid(ps, B_NULL[s], tf):
            bad.append("p " + s)
            continue
        x = nl.get(s)
        if not null_ok(x, B_NULL[s]):
            bad.append("null " + s)
            continue
        if tf and not p_in_bounds(t, ps, x, B_NULL[s]):
            bad.append("null %s: p outside its quantile bound" % s)
    mf = r.get("model_free") or {}
    tm = mf.get("t_obs")
    if not is_num(tm):
        bad.append("model-free t_obs")
    else:
        for s in ("cdr", "cr"):
            if not p_valid((mf.get("p") or {}).get(s), B_MF, finite(tm)):
                bad.append("model-free p " + s)
    return bad


def check_match_cal(r):
    bad = []
    t = r["t_obs"]
    M = r.get("match") or {}
    C = r.get("calibration") or {}
    for kind in ("receiver", "drift"):
        m = M.get(kind)
        tag = "match " + kind
        if (not isinstance(m, dict) or m.get("grid") != GRID
                or len(m.get("mean_t") or []) != len(GRID)
                or not all(is_num(v) for v in m["mean_t"])):
            bad.append(tag + ": grid")
            continue
        e = match_expect(m["mean_t"], t, m["grid"])
        for k in ("size", "matched", "reached"):
            if m.get(k) != e[k] or type(m.get(k)) is not type(e[k]):
                bad.append("%s: %s" % (tag, k))
        if not close(m.get("max_mean_t"), e["max_mean_t"]):
            bad.append(tag + ": max_mean_t")
        if e["matched"] and not close(m.get("mean_t_at_size"), e["mean_t_at_size"]):
            bad.append(tag + ": mean_t_at_size")
        c = C.get(kind)
        ctag = "calibration " + kind
        if not isinstance(c, dict):
            bad.append(ctag + ": missing")
            continue
        if c.get("size") != e["size"]:
            bad.append(ctag + ": size")
        if c.get("R") != R_CAL or c.get("B") != B_CAL:
            bad.append(ctag + ": R or B")
        to = c.get("t_obs") or []
        ps = c.get("p") or {}
        if len(to) != R_CAL or not all(is_num(v) for v in to):
            bad.append(ctag + ": t_obs")
            continue
        if sorted(ps) != sorted(CAL_SCHEMES[kind]):
            bad.append(ctag + ": schemes")
            continue
        for s in CAL_SCHEMES[kind]:
            lst = ps[s]
            if not isinstance(lst, list) or len(lst) != R_CAL:
                bad.append("%s: p %s length" % (ctag, s))
                continue
            for i, (x, ti) in enumerate(zip(lst, to)):
                if not p_valid(x, B_CAL, finite(ti)):
                    bad.append("%s: p %s %d" % (ctag, s, i))
            if not close((c.get("reject_rate") or {}).get(s), reject_rate(lst)):
                bad.append("%s: reject rate %s" % (ctag, s))
    return bad


def check_result(r):
    bad = []
    if r.get("version") != VERSION:
        bad.append("version")
    if r.get("record") != RECORD:
        bad.append("record")
    md5 = r.get("md5_ok")
    if not isinstance(md5, dict) or sorted(md5) != sorted((MAIN, AMB)):
        return bad + ["md5 flags"]
    if not all(md5.values()):
        if any(k in r for k in ("n_usable", "t_obs", "blocks")):
            bad.append("results despite a data mismatch")
        return bad
    n = r.get("n_usable")
    if not is_int(n) or r.get("n_calls") != n:
        return bad + ["usable calls"]
    if r.get("provisional") is not (n < N_USABLE_MIN):
        bad.append("provisional flag")
    ld = r.get("load") or {}
    if sum(ld.get(k, 0) for k in TIMED_KEYS) != n:
        bad.append("load counts")
    if r.get("settings") != EXPECTED_SETTINGS:
        bad.append("settings")
    st = check_structure(r)
    if st:
        return bad + st
    if gate_verdict(r) is not None:
        if any(k in r for k in ("t_obs", "match", "calibration")):
            bad.append("statistic despite no informative blocks")
        return bad
    return bad + check_main(r) + check_match_cal(r)


def compare_job(r, rg):
    """Mismatches between the job's own verdict and grading and the regrade."""
    bad = []
    if r.get("verdict") != rg["verdict"]:
        bad.append("verdict")
    if rg["predictions"] is None:
        if any(k in r for k in ("predictions", "brier", "power")):
            bad.append("grading without a test")
        return bad
    for k in ("power", "drift_size_rate", "drift_fooled_rate_cr"):
        if not close(r.get(k), rg[k]):
            bad.append(k)
    if r.get("predictions") != rg["predictions"]:
        bad.append("predictions")
    if not close(r.get("brier"), rg["brier"]):
        bad.append("brier")
    return bad


# --------------------------------------------------------------------------
# formats fixed before the result exists (RESULTS-ELEPHANT.md must contain these lines)
# --------------------------------------------------------------------------
def yes_no(v):
    return "not run" if v is None else ("yes" if v else "no")


def f5(x):
    return "{:+.5f}".format(x) if finite(x) else "n/a"


def fp(p):
    return "{:.4f}".format(p) if finite(p) else "n/a"


def md5_line(r):
    md5 = r.get("md5_ok") or {}
    return "Code version `{}`; data md5 match: main file {}, Amboseli timestamp file {}".format(
        r.get("version"), yes_no(md5.get(MAIN)), yes_no(md5.get(AMB)))


def data_line(r):
    ld = r.get("load") or {}
    return ("Usable calls: **{:,}** (provisional, fewer than 460: {}); dropped for incomplete "
            "features {:,}, for missing caller, receiver or date {:,}; with a timestamp: "
            "Samburu {:,}, Amboseli {:,}; without: Samburu {:,}, Amboseli {:,}").format(
        r["n_usable"], yes_no(r["provisional"]), ld.get("dropped_incomplete_features", 0),
        ld.get("dropped_missing_caller_receiver_date", 0), ld.get("timed_samburu", 0),
        ld.get("timed_amboseli", 0), ld.get("untimed_samburu", 0),
        ld.get("untimed_amboseli", 0))


def structure_line(r):
    return ("{:,} callers, {:,} receivers, {:,} dates, {:,} temporal groups; {:,} eligible "
            "pairs, of which {:,} same receiver, {:,} same date, {:,} same context").format(
        r["n_callers"], r["n_receivers"], r["n_dates"], r["n_time_groups"], r["n_pairs"],
        r["n_pairs_same_receiver"], r["n_pairs_same_date"], r["n_pairs_same_context"])


def block_row(r, s):
    """| blocks | n | informative | calls in informative | log10 relabelings | movable pairs |"""
    b = r["blocks"][s]
    return "| {} | {:,} | {:,} | {:,} | {:.1f} | {:,} |".format(
        SCHEME_TEXT[s], b["blocks"], b["informative_blocks"], b["calls_in_informative_blocks"],
        b["log10_relabelings"], b["pairs_with_a_movable_call"])


def tobs_line(r):
    return ("Observed contrast T_obs = **{}** (OLS coefficient of same receiver on "
            "rank(RF-GAP proximity)/n, with same context and same date)").format(f5(r["t_obs"]))


def null_row(r, s):
    """| null | B | null mean | null sd | null 95% quantile | p |"""
    x = r["null"][s]
    sd = "{:.5f}".format(x["sd"]) if finite(x["sd"]) else "n/a"
    return "| {} | {:,} | {} | {} | {} | **{}** |".format(
        SCHEME_TEXT[s], B_NULL[s], f5(x["mean"]), sd, f5(x["q95"]), fp(r["p"][s]))


def mf_line(r):
    m = r["model_free"]
    return ("Model-free contrast (z-scored Euclidean distance, no forest): T = {}; p within "
            "caller, date, relationship = {} (B = {:,}); p within caller, relationship = {} "
            "(B = {:,})").format(f5(m["t_obs"]), fp(m["p"]["cdr"]), B_MF, fp(m["p"]["cr"]),
                                 B_MF)


def match_line(rg, kind):
    e = rg["match"][kind]
    if not e["matched"]:
        return "{}: T_obs not positive, fallback size {:.2f}; largest mean T on the grid {}".format(
            WORLD[kind], e["size"], f5(e["max_mean_t"]))
    return ("{}: matched size **{:.2f}** (mean T there {}); largest mean T on the grid {}; "
            "reaches T_obs: {}").format(WORLD[kind], e["size"], f5(e["mean_t_at_size"]),
                                        f5(e["max_mean_t"]), yes_no(e["reached"]))


def grid_row(r, i):
    """| synthetic size | receiver-world mean T | drift-world mean T |"""
    return "| {:.2f} | {} | {} |".format(GRID[i], f5(r["match"]["receiver"]["mean_t"][i]),
                                         f5(r["match"]["drift"]["mean_t"][i]))


def cal_lines(rg):
    c, s = rg["counts"], rg["sizes"]
    return [
        "Power (receiver world at size {:.2f}, primary null, B = {}): **{} of {}** rejected; "
        "rate {:.2f}".format(s["receiver"], B_CAL, c["power"], c["R"], rg["power"]),
        "Size (drift world at size {:.2f}, primary null, B = {}): **{} of {}** rejected; "
        "rate {:.2f}".format(s["drift"], B_CAL, c["size"], c["R"], rg["drift_size_rate"]),
        "Fooled (drift world at size {:.2f}, naive within-caller null, B = {}): **{} of {}** "
        "rejected; rate {:.2f}".format(s["drift"], B_CAL, c["fooled"], c["R"],
                                       rg["drift_fooled_rate_cr"])]


def verdict_line(rg):
    return "Verdict: **`%s`**" % rg["verdict"]


def pred_end(k, v):
    return "| {:.2f} | **{}** |".format(
        PRIORS[k], "not graded" if v is None else ("true" if v else "false"))


def required_strings(r, rg):
    need = [md5_line(r)]
    if rg["verdict"] != DM:
        need += [data_line(r), structure_line(r)] + [block_row(r, s) for s in SCHEMES]
    if rg["predictions"] is not None:
        need += [tobs_line(r)] + [null_row(r, s) for s in SCHEMES] + [mf_line(r)]
        need += [match_line(rg, k) for k in ("receiver", "drift")]
        need += [grid_row(r, i) for i in range(len(GRID))]
        need += cal_lines(rg)
    need.append(verdict_line(rg))
    if rg["predictions"] is None:
        need.append("0 of 0 graded predictions true")
    else:
        need.append("%d of %d graded predictions true" % (rg["n_true"], len(ORDER)))
        need.append("Brier score **%.3f**" % rg["brier"])
    return need


def render_md_lines(r, rg):
    """Every fixed line, in one list (prediction rows included), for writing RESULTS-ELEPHANT.md."""
    pr = rg["predictions"] or {}
    return required_strings(r, rg) + [
        "| %s | %s %s" % (k, PRED_TEXT[k], pred_end(k, pr.get(k))) for k in ORDER]


def check_md(md, r, rg):
    """List of required lines missing from md (empty list = all present)."""
    missing = [s for s in required_strings(r, rg) if s not in md]
    pr = rg["predictions"] or {}
    lines = md.splitlines()
    for k in ORDER:
        hits = [ln for ln in lines if ln.startswith("| %s |" % k)]
        if len(hits) != 1 or not hits[0].rstrip().endswith(pred_end(k, pr.get(k))):
            missing.append("prediction row " + k)
    return missing


# --------------------------------------------------------------------------
# result tests (skipped until the job's output is published)
# --------------------------------------------------------------------------
@unittest.skipUnless(HAVE, "results/elephant/elephant_results.json not published yet")
class TestElephantResults(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(RES, encoding="utf-8") as f:
            cls.r = json.load(f)
        cls.rg = regrade(cls.r)
        cls.md = open(MD, encoding="utf-8").read() if os.path.exists(MD) else ""

    def test_consistency(self):
        self.assertEqual(check_result(self.r), [])

    def test_job_matches_regrade(self):
        self.assertEqual(compare_job(self.r, self.rg), [])

    @unittest.skipUnless(os.path.exists(MD), "RESULTS-ELEPHANT.md not written yet")
    def test_md_lines(self):
        self.assertEqual(check_md(self.md, self.r, self.rg), [])


# --------------------------------------------------------------------------
# toy tests (always run; every expected value worked out by hand)
# --------------------------------------------------------------------------
REC_MEANS = [0.01 * (i + 1) for i in range(16)]      # 0.01 .. 0.16
DRIFT_MEANS = [0.002 * (i + 1) for i in range(16)]   # 0.002 .. 0.032


def toy_result(t_obs=0.05, p=None, mf_p=None, rec_p=None, drift_cdr=None, drift_cr=None):
    """Default: verdict NOT_DETECTED_WITH_POWER, power 17/20, size 1/20, fooled 12/20."""
    p = dict(p or {"cdr": 0.2, "cd": 0.1, "cr": 0.005})
    mf_p = dict(mf_p or {"cdr": 0.01, "cr": 0.001})
    rec_p = list(rec_p or [0.01] * 17 + [0.5] * 3)
    drift_cdr = list(drift_cdr or [0.03] + [0.6] * 19)
    drift_cr = list(drift_cr or [0.02] * 12 + [0.3] * 8)
    pos = t_obs > 0

    def mt(means, size, gi, reached):
        m = {"grid": list(GRID), "mean_t": list(means), "max_mean_t": max(means)}
        if pos:
            m.update(size=size, matched=True, mean_t_at_size=means[gi], reached=reached)
        else:
            m.update(size=FALLBACK_SIZE, matched=False, reached=False)
        return m

    def rate(ps):
        return sum(1 for x in ps if x <= 0.05) / len(ps)

    mr = mt(REC_MEANS, 1.25, 4, True)
    md = mt(DRIFT_MEANS, 4.0, 15, False)
    rc = {"size": mr["size"], "R": 20, "B": 99, "t_obs": [t_obs] * 20, "p": {"cdr": rec_p},
          "reject_rate": {"cdr": rate(rec_p)}}
    dc = {"size": md["size"], "R": 20, "B": 99, "t_obs": [t_obs] * 20,
          "p": {"cdr": drift_cdr, "cr": drift_cr},
          "reject_rate": {"cdr": rate(drift_cdr), "cr": rate(drift_cr)}}
    return {"version": VERSION, "python": "3.12.8", "numpy": "2.1.3", "sklearn": "1.5.2",
            "record": RECORD, "utc_start": "2026-11-20T10:00:00Z", "workers": 2,
            "md5_ok": {MAIN: True, AMB: True},
            "load": {"timed_samburu": 371, "timed_amboseli": 11, "untimed_amboseli": 87},
            "n_usable": 469, "provisional": False, "n_calls": 469, "n_callers": 20,
            "n_receivers": 60, "n_dates": 80, "n_time_groups": 300, "n_pairs": 2000,
            "n_pairs_same_receiver": 400, "n_pairs_same_date": 600,
            "n_pairs_same_context": 1500,
            "blocks": {
                "cdr": {"blocks": 150, "informative_blocks": 40,
                        "calls_in_informative_blocks": 160, "log10_relabelings": 30.0,
                        "pairs_with_a_movable_call": 900},
                "cd": {"blocks": 120, "informative_blocks": 45,
                       "calls_in_informative_blocks": 200, "log10_relabelings": 40.0,
                       "pairs_with_a_movable_call": 1000},
                "cr": {"blocks": 30, "informative_blocks": 25,
                       "calls_in_informative_blocks": 420, "log10_relabelings": 300.0,
                       "pairs_with_a_movable_call": 1990}},
            "settings": copy.deepcopy(EXPECTED_SETTINGS),
            "t_obs": t_obs, "p": p,
            "null": {"cdr": {"mean": 0.03, "sd": 0.03, "q05": -0.02, "q95": 0.08, "n_nan": 0},
                     "cd": {"mean": 0.02, "sd": 0.02, "q05": -0.01, "q95": 0.06, "n_nan": 0},
                     "cr": {"mean": 0.0, "sd": 0.01, "q05": -0.016, "q95": 0.016,
                            "n_nan": 0}},
            "model_free": {"t_obs": 0.03 if pos else -0.01, "p": mf_p},
            "match": {"receiver": mr, "drift": md},
            "calibration": {"receiver": rc, "drift": dc},
            "power": rc["reject_rate"]["cdr"], "drift_size_rate": dc["reject_rate"]["cdr"],
            "drift_fooled_rate_cr": dc["reject_rate"]["cr"]}


def attach(r):
    rg = regrade(r)
    r["verdict"] = rg["verdict"]
    if rg["predictions"] is not None:
        r["predictions"] = dict(rg["predictions"])
        r["brier"] = rg["brier"]
    return r


STAT_KEYS = ("t_obs", "p", "null", "model_free", "match", "calibration", "power",
             "drift_size_rate", "drift_fooled_rate_cr")


def toy_noinf(n_pairs=90, same=30):
    r = toy_result()
    for k in STAT_KEYS:
        del r[k]
    r.update(n_pairs=n_pairs, n_pairs_same_receiver=same, n_pairs_same_date=30,
             n_pairs_same_context=60)
    r["blocks"]["cdr"]["pairs_with_a_movable_call"] = 50
    r["blocks"]["cd"]["pairs_with_a_movable_call"] = 60
    r["blocks"]["cr"]["pairs_with_a_movable_call"] = 90
    return r


def toy_mismatch():
    return {"version": VERSION, "python": "3.12.8", "numpy": "2.1.3", "sklearn": "1.5.2",
            "record": RECORD, "utc_start": "2026-11-20T10:00:00Z", "workers": 2,
            "md5_ok": {MAIN: True, AMB: False}, "verdict": DM}


NEG = {"t_obs": -0.03, "p": {"cdr": 1.0, "cd": 1.0, "cr": 1.0},
       "mf_p": {"cdr": 0.9, "cr": 0.9}}


class TestToy(unittest.TestCase):
    def test_default(self):
        r = toy_result()
        rg = regrade(r)
        self.assertEqual(rg["verdict"], "NOT_DETECTED_WITH_POWER")
        self.assertEqual((rg["power"], rg["drift_size_rate"], rg["drift_fooled_rate_cr"]),
                         (0.85, 0.05, 0.6))
        self.assertEqual(rg["counts"], {"power": 17, "size": 1, "fooled": 12, "R": 20})
        self.assertEqual(rg["sizes"], {"receiver": 1.25, "drift": 4.0})
        self.assertEqual([rg["predictions"][k] for k in ORDER],
                         [True, True, False, True, True, True, False, True])
        # .01 + .0625 + .09 + .4225 + .0025 + .25 + .36 + .5625 = 1.76
        self.assertAlmostEqual(rg["brier"], 0.22, places=12)
        self.assertEqual(rg["n_true"], 6)
        self.assertEqual(check_result(r), [])
        self.assertEqual(compare_job(r, rg), ["verdict", "predictions", "brier"])
        attach(r)
        self.assertEqual(compare_job(r, rg), [])
        r2 = json.loads(json.dumps(r))
        self.assertEqual(check_result(r2), [])
        self.assertEqual(compare_job(r2, regrade(r2)), [])
        r2["power"] = 0.8
        r2["brier"] = 0.2
        self.assertEqual(compare_job(r2, rg), ["power", "brier"])

    def test_verdicts(self):
        rg = regrade(toy_result(p={"cdr": 0.001, "cd": 0.1, "cr": 0.005}))
        self.assertEqual(rg["verdict"], "RECEIVER_SIGNAL_SURVIVES_DAY_BLOCKING")
        self.assertTrue(rg["predictions"]["Q3"])
        self.assertAlmostEqual(rg["brier"], 0.27, places=12)
        r = toy_result(rec_p=[0.01] * 15 + [0.5] * 5)
        rg = regrade(r)
        self.assertEqual(rg["verdict"], "UNDERPOWERED")
        self.assertEqual(rg["power"], 0.75)
        self.assertFalse(rg["predictions"]["Q4"])
        self.assertAlmostEqual(rg["brier"], 0.1825, places=12)
        self.assertEqual(check_result(r), [])
        # p = 0.05 rejects; 16 of 20 is exactly the 0.80 needed
        rg = regrade(toy_result(rec_p=[0.05] * 16 + [0.06] * 4))
        self.assertEqual(rg["power"], 0.8)
        self.assertEqual(rg["verdict"], "NOT_DETECTED_WITH_POWER")
        # 5 of 20 drift rejections fail calibration, 4 of 20 do not
        rg = regrade(toy_result(drift_cdr=[0.03] * 5 + [0.6] * 15))
        self.assertEqual(rg["verdict"], "CALIBRATION_FAILED")
        self.assertFalse(rg["predictions"]["Q5"])
        rg = regrade(toy_result(drift_cdr=[0.03] * 4 + [0.6] * 16))
        self.assertEqual(rg["drift_size_rate"], 0.2)
        self.assertEqual(rg["verdict"], "NOT_DETECTED_WITH_POWER")
        self.assertTrue(rg["predictions"]["Q5"])
        rg = regrade(toy_result(drift_cr=[0.02] * 9 + [0.3] * 11))
        self.assertEqual(rg["drift_fooled_rate_cr"], 0.45)
        self.assertFalse(rg["predictions"]["Q6"])
        # negative contrast: fallback size 1.0 for both worlds
        r = toy_result(**NEG)
        rg = regrade(r)
        self.assertEqual(rg["verdict"], "NO_POSITIVE_CONTRAST")
        self.assertEqual(rg["sizes"], {"receiver": 1.0, "drift": 1.0})
        self.assertEqual([rg["predictions"][k] for k in ORDER],
                         [False, False, False, True, True, True, False, False])
        # .81 + .5625 + .09 + .4225 + .0025 + .25 + .36 + .0625 = 2.56
        self.assertAlmostEqual(rg["brier"], 0.32, places=12)
        self.assertEqual(check_result(r), [])
        rg = regrade(toy_result(drift_cdr=[0.03] * 5 + [0.6] * 15, **NEG))
        self.assertEqual(rg["verdict"], "CALIBRATION_FAILED")
        nan = float("nan")
        self.assertEqual(verdict_of(nan, nan, 0.9, 0.05), "NO_POSITIVE_CONTRAST")
        self.assertEqual(verdict_of(0.1, 0.05, 0.1, 0.2),
                         "RECEIVER_SIGNAL_SURVIVES_DAY_BLOCKING")
        self.assertEqual(verdict_of(0.1, nan, 0.9, 0.0), "NOT_DETECTED_WITH_POWER")
        self.assertEqual(verdict_of(0.1, 0.01, 0.9, nan), "CALIBRATION_FAILED")
        self.assertEqual(verdict_of(0.0, 0.01, 0.9, 0.0), "NO_POSITIVE_CONTRAST")

    def test_match_helper(self):
        e = match_expect([0.25, 0.75], 0.5, [0.25, 0.5])
        self.assertEqual((e["size"], e["matched"], e["reached"], e["max_mean_t"]),
                         (0.25, True, True, 0.75))
        e = match_expect([float("nan"), 0.4], 0.1, [0.25, 0.5])
        self.assertEqual((e["size"], e["mean_t_at_size"], e["reached"]), (0.5, 0.4, True))
        e = match_expect([0.01, 0.02], 0.5, [0.25, 0.5])
        self.assertEqual((e["size"], e["reached"]), (0.5, False))
        e = match_expect([0.01, 0.02], 0.0, [0.25, 0.5])
        self.assertEqual((e["size"], e["matched"], e["reached"], e["max_mean_t"]),
                         (1.0, False, False, 0.02))

    def test_gates(self):
        r = toy_mismatch()
        rg = regrade(r)
        self.assertEqual(rg["verdict"], DM)
        self.assertIsNone(rg["predictions"])
        self.assertEqual(check_result(r), [])
        self.assertEqual(compare_job(r, rg), [])
        r["n_usable"] = 469
        self.assertEqual(check_result(r), ["results despite a data mismatch"])
        r = attach(toy_noinf())
        rg = regrade(r)
        self.assertEqual(rg["verdict"], NIB)
        self.assertEqual(check_result(r), [])
        self.assertEqual(compare_job(r, rg), [])
        r["t_obs"] = 0.01
        self.assertEqual(check_result(r), ["statistic despite no informative blocks"])
        self.assertEqual(regrade(toy_noinf(n_pairs=2000, same=2000))["verdict"], NIB)
        self.assertEqual(regrade(toy_noinf(n_pairs=2000, same=0))["verdict"], NIB)
        r = toy_noinf(n_pairs=100, same=30)
        r["blocks"]["cdr"]["informative_blocks"] = 0
        r["blocks"]["cdr"]["calls_in_informative_blocks"] = 0
        r["blocks"]["cdr"]["log10_relabelings"] = 0.0
        self.assertEqual(regrade(r)["verdict"], NIB)
        self.assertEqual(check_result(r), [])

    def test_consistency_catches_errors(self):
        r = toy_result()
        r["calibration"]["receiver"]["reject_rate"]["cdr"] = 0.8
        r["p"]["cd"] = 0.0123
        r["p"]["cr"] = 0.1
        r["match"]["receiver"]["size"] = 1.5
        r["match"]["drift"]["reached"] = True
        r["blocks"]["cdr"]["calls_in_informative_blocks"] = 250
        self.assertEqual(sorted(check_result(r)), ["blocks cdr not nested in cd"])
        r["blocks"]["cdr"]["calls_in_informative_blocks"] = 160
        self.assertEqual(sorted(check_result(r)), sorted([
            "calibration receiver: reject rate cdr", "p cd",
            "null cr: p outside its quantile bound", "match receiver: size",
            "match drift: reached"]))
        r = toy_result()
        r["calibration"]["drift"]["p"]["cr"][0] = 0.555     # off the lattice, not a rejection
        r["calibration"]["drift"]["size"] = 3.75
        r["model_free"]["p"]["cr"] = 0.5005
        r["settings"]["b_cal"] = 199
        r["n_usable"] = r["n_calls"] = 450
        r["load"]["timed_samburu"] = 352
        self.assertEqual(sorted(check_result(r)), sorted([
            "calibration drift: p cr 0", "calibration drift: size", "model-free p cr",
            "settings", "provisional flag", "calibration drift: reject rate cr"]))
        # negative contrast: p must be at least 0.95 once T_obs sits below the null's 5% point
        r = toy_result(**NEG)
        r["p"]["cdr"] = 0.9
        self.assertEqual(check_result(r), ["null cdr: p outside its quantile bound"])
        r = toy_result()
        r["n_pairs_same_date"] = 2001
        r["blocks"]["cd"]["blocks"] = 70
        self.assertEqual(check_result(r), ["n_pairs_same_date exceeds pairs", "blocks cd"])

    def test_helpers(self):
        self.assertTrue(p_ok(1 / 1000, 999))
        self.assertTrue(p_ok(0.03, 99))
        self.assertTrue(p_ok(1.0, 199))
        self.assertFalse(p_ok(0.0123, 199))
        self.assertFalse(p_ok(0.001, 99))
        self.assertTrue(p_valid(float("nan"), 99, False))
        self.assertFalse(p_valid(0.5, 99, False))
        self.assertFalse(p_valid(float("nan"), 99, True))
        self.assertEqual(reject_rate([0.05, 0.06, float("nan"), 0.01]), 0.5)
        x = {"mean": 0.0, "sd": 0.01, "q05": -0.016, "q95": 0.016, "n_nan": 0}
        self.assertTrue(p_in_bounds(0.05, 11 / 200, x, 199))
        self.assertFalse(p_in_bounds(0.05, 12 / 200, x, 199))
        self.assertTrue(p_in_bounds(-0.05, 190 / 200, x, 199))
        self.assertFalse(p_in_bounds(-0.05, 189 / 200, x, 199))
        x["n_nan"] = 199
        self.assertTrue(p_in_bounds(0.05, 1.0, x, 199))
        self.assertFalse(close(None, 0.1))
        self.assertTrue(close(float("nan"), float("nan")))
        self.assertEqual(argmin([0.3, 0.1, 0.1]), 1)

    def test_formats(self):
        r = toy_result()
        rg = regrade(r)
        self.assertEqual(md5_line(r), "Code version `elephant-names-null-1`; data md5 match: "
                                      "main file yes, Amboseli timestamp file yes")
        self.assertEqual(data_line(r),
                         "Usable calls: **469** (provisional, fewer than 460: no); dropped for "
                         "incomplete features 0, for missing caller, receiver or date 0; with a "
                         "timestamp: Samburu 371, Amboseli 11; without: Samburu 0, Amboseli 87")
        self.assertEqual(structure_line(r),
                         "20 callers, 60 receivers, 80 dates, 300 temporal groups; 2,000 "
                         "eligible pairs, of which 400 same receiver, 600 same date, 1,500 "
                         "same context")
        self.assertEqual(block_row(r, "cdr"),
                         "| caller, date, relationship (primary) | 150 | 40 | 160 | 30.0 | 900 |")
        self.assertEqual(block_row(r, "cr"),
                         "| caller, relationship (naive) | 30 | 25 | 420 | 300.0 | 1,990 |")
        self.assertEqual(tobs_line(r),
                         "Observed contrast T_obs = **+0.05000** (OLS coefficient of same "
                         "receiver on rank(RF-GAP proximity)/n, with same context and same date)")
        self.assertEqual(null_row(r, "cdr"),
                         "| caller, date, relationship (primary) | 999 | +0.03000 | 0.03000 | "
                         "+0.08000 | **0.2000** |")
        self.assertEqual(null_row(r, "cr"),
                         "| caller, relationship (naive) | 199 | +0.00000 | 0.01000 | +0.01600 | "
                         "**0.0050** |")
        self.assertEqual(mf_line(r),
                         "Model-free contrast (z-scored Euclidean distance, no forest): T = "
                         "+0.03000; p within caller, date, relationship = 0.0100 (B = 999); p "
                         "within caller, relationship = 0.0010 (B = 999)")
        self.assertEqual(match_line(rg, "receiver"),
                         "Receiver (names) world: matched size **1.25** (mean T there +0.05000); "
                         "largest mean T on the grid +0.16000; reaches T_obs: yes")
        self.assertEqual(match_line(rg, "drift"),
                         "Drift world: matched size **4.00** (mean T there +0.03200); largest "
                         "mean T on the grid +0.03200; reaches T_obs: no")
        self.assertEqual(grid_row(r, 0), "| 0.25 | +0.01000 | +0.00200 |")
        self.assertEqual(grid_row(r, 15), "| 4.00 | +0.16000 | +0.03200 |")
        self.assertEqual(cal_lines(rg), [
            "Power (receiver world at size 1.25, primary null, B = 99): **17 of 20** rejected; "
            "rate 0.85",
            "Size (drift world at size 4.00, primary null, B = 99): **1 of 20** rejected; "
            "rate 0.05",
            "Fooled (drift world at size 4.00, naive within-caller null, B = 99): **12 of 20** "
            "rejected; rate 0.60"])
        self.assertEqual(verdict_line(rg), "Verdict: **`NOT_DETECTED_WITH_POWER`**")
        rn = regrade(toy_result(**NEG))
        self.assertEqual(match_line(rn, "receiver"),
                         "Receiver (names) world: T_obs not positive, fallback size 1.00; "
                         "largest mean T on the grid +0.16000")
        self.assertEqual(pred_end("Q4", True), "| 0.35 | **true** |")
        self.assertEqual(pred_end("Q8", None), "| 0.25 | **not graded** |")

    def test_md_check(self):
        r = toy_result()
        rg = regrade(r)
        lines = render_md_lines(r, rg)
        # md5 + data + structure + 3 blocks + T_obs + 3 nulls + model-free + 2 matches
        # + 16 grid rows + 3 calibration + verdict + 2 score lines + 8 predictions
        self.assertEqual(len(lines), 43)
        self.assertIn("6 of 8 graded predictions true", lines)
        self.assertIn("Brier score **0.220**", lines)
        self.assertIn("| Q3 | the primary day-blocked null rejects (p_cdr <= 0.05) | 0.30 | "
                      "**false** |", lines)
        md = "\n".join(lines)
        self.assertEqual(check_md(md, r, rg), [])
        v = verdict_line(rg)
        self.assertEqual(check_md("\n".join(ln for ln in lines if ln != v), r, rg), [v])
        no_q3 = "\n".join(ln for ln in lines if not ln.startswith("| Q3 |"))
        self.assertEqual(check_md(no_q3, r, rg), ["prediction row Q3"])
        r = toy_noinf()
        lines = render_md_lines(r, regrade(r))
        self.assertEqual(len(lines), 16)
        self.assertIn("Verdict: **`NO_INFORMATIVE_BLOCKS`**", lines)
        self.assertIn("0 of 0 graded predictions true", lines)
        r = toy_mismatch()
        lines = render_md_lines(r, regrade(r))
        self.assertEqual(len(lines), 11)
        self.assertEqual(lines[0], "Code version `elephant-names-null-1`; data md5 match: main "
                                   "file yes, Amboseli timestamp file no")
        self.assertEqual(lines[-1], "| Q8 | the model-free contrast rejects under the primary "
                                    "null (p <= 0.05) | 0.25 | **not graded** |")


if __name__ == "__main__":
    unittest.main()
