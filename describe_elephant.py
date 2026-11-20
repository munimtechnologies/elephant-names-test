# SPDX-License-Identifier: Apache-2.0
# describe_elephant.py: label structure of the public elephant "names" data (Pardo et al.
# 2024, Nature Ecology & Evolution; data and reply code: Zenodo record 23284924, CC BY 4.0),
# computed BEFORE any test is registered. It reads ONLY labels (caller, receiver, date, site,
# context, certainty) and whether each call's acoustic features are complete; it computes no
# acoustic statistic, distance, model or similarity, so the coming pre-registration stays blind
# to every outcome. What it measures is the power of a within-(caller, day) test: how many
# same-caller call pairs share a day, and how many caller-day blocks hold calls to two or more
# receivers.
# MV2, Munim, Inc. (mv2@munimtech.com). Made with AI agents. Apache-2.0.
import argparse
import csv
import hashlib
import io
import json
import math
import os
import sys
import time
import urllib.request
from collections import Counter, defaultdict

VERSION = "elephant-describe-1"
RECORD = "23284924"
API = "https://zenodo.org/api/records/" + RECORD
URL = "https://zenodo.org/records/%s/files/%s?download=1"
MAIN = "20231225_dfall_obs_data_and_spectral_features_revision1_n469.csv"
AMB = "20260731_Amboseli_sound_files_used_in_names_paper_for_upload.csv"
FILES = (MAIN, AMB)
NA = {"", "NA", "na", "N/A", "NaN", "nan"}
FEATURES = (["sprsMed", "sprsMbw", "sprsEqbw", "sprsMc"]
            + ["V%d" % i for i in range(1, 75)]
            + ["%s%d" % (p, i) for p in ("Fw", "Mw", "F", "M") for i in range(1, 5)])


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


def site(sound_file):
    """The reply code flags Samburu files by a leading '20' (digital recordings named by date);
    everything else is the Amboseli archive."""
    return "samburu" if (sound_file or "").strip().startswith("20") else "amboseli"


def log10_relabelings(counts):
    """log10 of the number of distinct receiver relabelings of one block: n! / prod m_k!."""
    n = sum(counts)
    return (math.lgamma(n + 1) - sum(math.lgamma(m + 1) for m in counts)) / math.log(10)


def pair_table(calls):
    """calls: list of (caller, receiver, date). Same-caller unordered pairs by (same receiver,
    same date)."""
    by_caller = defaultdict(list)
    for c, r, d in calls:
        by_caller[c].append((r, d))
    t = Counter()
    for lst in by_caller.values():
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                sr = lst[i][0] == lst[j][0]
                sd = lst[i][1] == lst[j][1]
                t[("same" if sr else "diff") + "_receiver_" + ("same" if sd else "diff")
                  + "_date"] += 1
    out = {k: t.get(k, 0) for k in ("same_receiver_same_date", "same_receiver_diff_date",
                                     "diff_receiver_same_date", "diff_receiver_diff_date")}
    out["callers"] = len(by_caller)
    out["callers_with_2plus_calls"] = sum(1 for v in by_caller.values() if len(v) >= 2)
    out["callers_with_2plus_receivers"] = sum(1 for v in by_caller.values()
                                              if len(set(r for r, _ in v)) >= 2)
    return out


def block_table(calls):
    """Caller-day blocks: the only blocks in which a within-(caller, day) permutation of
    receiver labels can move anything are those with calls to 2 or more receivers."""
    blocks = defaultdict(Counter)
    for c, r, d in calls:
        blocks[(c, d)][r] += 1
    inf = [b for b in blocks.values() if len(b) >= 2]
    same = diff = 0
    for b in inf:
        n = sum(b.values())
        s = sum(m * (m - 1) // 2 for m in b.values())
        same += s
        diff += n * (n - 1) // 2 - s
    return {"blocks": len(blocks),
            "blocks_2plus_calls": sum(1 for b in blocks.values() if sum(b.values()) >= 2),
            "blocks_2plus_receivers": len(inf),
            "callers_with_informative_block": len(set(c for (c, _), b in blocks.items()
                                                      if len(b) >= 2)),
            "calls_in_informative_blocks": sum(sum(b.values()) for b in inf),
            "pairs_in_informative_blocks_same_receiver": same,
            "pairs_in_informative_blocks_diff_receiver": diff,
            "log10_relabelings": sum(log10_relabelings(list(b.values())) for b in inf),
            "informative_block_sizes": sorted((sum(b.values()) for b in inf), reverse=True)}


def describe(rows):
    out = {"n_rows": len(rows)}
    have = set(rows[0].keys()) if rows else set()
    out["feature_columns_present"] = sum(1 for c in FEATURES if c in have)
    out["feature_columns_expected"] = len(FEATURES)
    comp = [r for r in rows if all(is_num(r.get(c)) for c in FEATURES if c in have)]
    out["n_complete_features"] = len(comp)
    use = [r for r in comp if clean(r.get("CallerID")) and clean(r.get("Elicitor1_ID"))
           and clean(r.get("date"))]
    out["n_with_caller_receiver_date"] = len(use)
    out["call_type"] = dict(Counter(clean(r.get("Call_Type")) or "NA" for r in rows))
    out["site"] = dict(Counter(site(r.get("sound.files")) for r in use))
    out["context2"] = dict(Counter(clean(r.get("Context2")) or "NA" for r in use))
    out["newSocCat"] = dict(Counter(clean(r.get("newSocCat")) or "NA" for r in use))
    calls = [(clean(r["CallerID"]), clean(r["Elicitor1_ID"]), clean(r["date"])) for r in use]
    out["n_callers"] = len(set(c for c, _, _ in calls))
    out["n_receivers"] = len(set(r for _, r, _ in calls))
    out["n_dates"] = len(set(d for _, _, d in calls))
    out["n_dyads"] = len(set((c, r) for c, r, _ in calls))
    out["calls_per_caller"] = sorted(Counter(c for c, _, _ in calls).values(), reverse=True)
    out["calls_per_dyad"] = sorted(Counter((c, r) for c, r, _ in calls).values(), reverse=True)
    cert = [r for r in use if clean(r.get("Cert_ID_Bin")) == "1"
            and clean(r.get("Cert_ID_Elic1_Bin")) == "1"]
    out["n_caller_and_receiver_certain"] = len(cert)
    out["pairs"] = pair_table(calls)
    out["blocks"] = block_table(calls)
    certc = [(clean(r["CallerID"]), clean(r["Elicitor1_ID"]), clean(r["date"])) for r in cert]
    out["pairs_certain"] = pair_table(certc)
    out["blocks_certain"] = block_table(certc)
    for s in ("samburu", "amboseli"):
        sc = [(clean(r["CallerID"]), clean(r["Elicitor1_ID"]), clean(r["date"])) for r in use
              if site(r.get("sound.files")) == s]
        out["blocks_" + s] = block_table(sc)
    return out


def read_rows(blob):
    try:
        text = blob.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = blob.decode("latin-1")
    return list(csv.DictReader(io.StringIO(text)))


def fetch(url, timeout=120):
    req = urllib.request.Request(url, headers={"User-Agent": "song-zipf-test-describe/1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/data/elephant")
    ap.add_argument("--out-dir", default="/out")
    a = ap.parse_args(argv)
    os.makedirs(a.data_dir, exist_ok=True)
    os.makedirs(a.out_dir, exist_ok=True)
    out = {"version": VERSION, "python": sys.version.split()[0], "record": RECORD,
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "files": {}}
    try:
        meta = json.loads(fetch(API).decode("utf-8"))
        md5 = {f.get("key"): (f.get("checksum") or "").replace("md5:", "")
               for f in meta.get("files") or []}
    except Exception as e:
        md5 = {}
        out["api_error"] = repr(e)[:200]
    blobs = {}
    for name in FILES:
        path = os.path.join(a.data_dir, name)
        if not os.path.exists(path):
            data = fetch(URL % (RECORD, name))
            with open(path, "wb") as f:
                f.write(data)
        with open(path, "rb") as f:
            blobs[name] = f.read()
        got = hashlib.md5(blobs[name]).hexdigest()
        out["files"][name] = {"bytes": len(blobs[name]), "md5": got,
                              "sha256": hashlib.sha256(blobs[name]).hexdigest(),
                              "md5_matches_zenodo": (got == md5[name]) if name in md5 else None}
    rows = read_rows(blobs[MAIN])
    out["columns"] = list(rows[0].keys()) if rows else []
    out["describe"] = describe(rows)
    amb = read_rows(blobs[AMB])
    out["amboseli_timestamp_file"] = {"rows": len(amb),
                                      "columns": list(amb[0].keys()) if amb else []}
    with open(os.path.join(a.out_dir, "elephant_describe.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    d = out["describe"]
    print(json.dumps({k: d[k] for k in ("n_rows", "n_complete_features",
                                        "n_with_caller_receiver_date", "n_callers",
                                        "n_receivers", "n_dates", "pairs", "blocks")},
                     indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
