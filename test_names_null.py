# SPDX-License-Identifier: Apache-2.0
# Tests for names_null.py (expected values worked out by hand; see comments).
# MV2, Munim, Inc. (mv2@munimtech.com). Made with AI agents. Apache-2.0.
import calendar
import math
import unittest

import numpy as np

import names_null as N


def toy_calls(seed=1, n_callers=3, receivers=2, dates=2, per=3, delta=3.0):
    """3 callers x 2 receivers x 2 dates x 3 calls = 36 calls; every (caller, date) block holds
    two receivers; receiver-dyad effect of size delta on 94 features."""
    rng = np.random.default_rng(seed)
    p = len(N.FEATURES)
    calls = []
    for ci in range(n_callers):
        a = rng.standard_normal(p)
        for ri in range(receivers):
            u = rng.standard_normal(p)
            for di in range(dates):
                for k in range(per):
                    x = a + delta * u + rng.standard_normal(p)
                    calls.append({"id": "c%d_r%d_d%d_%d" % (ci, ri, di, k), "caller": "C%d" % ci,
                                  "receiver": "R%d_%d" % (ci, ri),
                                  "date": "11/%d/2019" % (10 + di), "site": "samburu",
                                  "ts": None, "soc": "A", "ctx": "greeting", "cert": True,
                                  "x": x.tolist()})
    return calls


def call(caller, ts, soc="S", ctx="g", cert=True, i=0):
    return {"id": "x%d" % i, "caller": caller, "receiver": "r", "date": "1/1/2020",
            "site": "samburu", "ts": ts, "soc": soc, "ctx": ctx, "cert": cert, "x": []}


class TestParsing(unittest.TestCase):
    def test_helpers(self):
        self.assertEqual(len(N.FEATURES), 94)
        self.assertTrue(N.is_one("1"))
        self.assertTrue(N.is_one("1.0"))
        self.assertFalse(N.is_one("0"))
        self.assertFalse(N.is_one("NA"))
        self.assertIsNone(N.num("NA"))
        self.assertEqual(N.num("5.00E-04"), 0.0005)

    def test_samburu_ts(self):
        want = calendar.timegm((2019, 11, 22, 10, 12, 0, 0, 0, 0)) + 116.5
        self.assertAlmostEqual(N.samburu_ts("20191122_1012.WAV", "116.5"), want, places=6)
        self.assertAlmostEqual(N.samburu_ts("20191122_1012_2.wav", "0"), want - 116.5, places=6)
        self.assertIsNone(N.samburu_ts("B1600222.wav", "1"))
        self.assertIsNone(N.samburu_ts("20191122_1012.WAV", "NA"))

    def test_amboseli_ts(self):
        row = {"soundID": "B1600222_sel41.wav", "Date": "3/4/1999", "Call_start_hour": "",
               "Call_start_minute": "", "Call_start_second": "", "Track_start_hour": "17",
               "Track_start_minute": "24", "Track_start_second": "",
               "Tape_to_track_elapsed_hour": "0", "Tape_to_track_elapsed_minute": "0",
               "Tape_to_track_elapsed_second": "0", "Tape_to_clip_elapsed_hour": "0",
               "Tape_to_clip_elapsed_minute": "2", "Tape_to_clip_elapsed_second": "22",
               "Clip_to_call_elapsed_sec": "0.0215"}
        day0 = calendar.timegm((1999, 4, 3, 0, 0, 0, 0, 0, 0))     # d/m/Y: 3 April 1999
        self.assertAlmostEqual(N.amboseli_ts(row), day0 + 17 * 3600 + 24 * 60 + 142.0215,
                               places=6)
        row2 = dict(row, Call_start_hour="8", Call_start_minute="5", Call_start_second="7.9",
                    Track_start_hour="", Track_start_minute="")
        self.assertEqual(N.amboseli_ts(row2), day0 + 8 * 3600 + 5 * 60 + 7)
        row3 = dict(row, Track_start_hour="")
        self.assertIsNone(N.amboseli_ts(row3))
        self.assertIsNone(N.amboseli_ts(dict(row, Date="NA")))

    def test_load(self):
        def r(sound, sid, caller="A", rec="R1", feat="0.5", cert="1"):
            x = {k: feat for k in N.FEATURES}
            x.update({"sound.files": sound, "soundID": sid, "start": "10", "CallerID": caller,
                      "Elicitor1_ID": rec, "date": "11/22/2019", "newSocCat": "A2J",
                      "Context2": "care", "Cert_ID_Bin": cert, "Cert_Con_Bin": "1"})
            return x
        rows = [r("20191122_1012.WAV", "s1"), r("B16.wav", "B16_sel1.wav"),
                r("B17.wav", "B17_sel1.wav", cert="0"), r("20191122_1012.WAV", "s2", rec="NA"),
                r("20191122_1012.WAV", "s3", feat="NA")]
        amb = [{"soundID": "B16_sel1.wav", "Date": "3/4/1999", "Call_start_hour": "8",
                "Call_start_minute": "0", "Call_start_second": "0"}]
        calls, info = N.load(rows, amb)
        self.assertEqual(len(calls), 3)
        self.assertEqual(info, {"dropped_incomplete_features": 1,
                                "dropped_missing_caller_receiver_date": 1,
                                "timed_amboseli": 1, "timed_samburu": 1,
                                "untimed_amboseli": 1})
        self.assertEqual([c["cert"] for c in calls], [True, True, False])
        self.assertEqual(len(calls[0]["x"]), 94)


class TestStructure(unittest.TestCase):
    def test_time_groups(self):
        calls = [call("A", 0, i=0), call("A", 60, i=1), call("A", 121, i=2),
                 call("A", None, i=3), call("B", 10, i=4)]
        # 60 s gap stays in the group (reply: new group only if gap > 60); 61 s starts one
        self.assertEqual(N.time_groups(calls), ["A__1", "A__1", "A__2", "untimed__3", "B__1"])

    def test_eligible_pairs(self):
        calls = [call("A", 0, i=0), call("A", 30, i=1), call("A", 500, i=2),
                 call("A", 1000, soc="S2", i=3), call("A", 2000, cert=False, i=4),
                 call("B", 0, i=5), call("B", 100, ctx=None, i=6)]
        ia, ib = N.eligible_pairs(calls, N.time_groups(calls))
        # (0,1) same 60-s group; 3 other category; 4 uncertain; 6 no context
        self.assertEqual(ia.tolist(), [0, 1])
        self.assertEqual(ib.tolist(), [2, 2])

    def test_blocks_and_permute(self):
        calls = [{"caller": c, "date": d, "soc": "S"} for c, d in
                 [("A", "d1"), ("A", "d1"), ("A", "d1"), ("A", "d2"), ("B", "d1"), ("B", "d1")]]
        lab = np.array([0, 0, 1, 1, 2, 2])
        blks = N.blocks(calls, "cd")
        self.assertEqual([b.tolist() for b in blks], [[0, 1, 2], [3], [4, 5]])
        for seed in range(5):
            out = N.permute(lab, blks, np.random.default_rng(seed))
            self.assertEqual(sorted(out[:3].tolist()), [0, 0, 1])
            self.assertEqual(out[3:].tolist(), [1, 2, 2])
        info = N.block_info(lab, blks, np.array([0, 3]), np.array([3, 4]))
        self.assertEqual(info["informative_blocks"], 1)
        self.assertEqual(info["calls_in_informative_blocks"], 3)
        self.assertAlmostEqual(info["log10_relabelings"], round(math.log10(3), 4), places=9)
        self.assertEqual(info["pairs_with_a_movable_call"], 1)
        self.assertEqual(len(N.blocks(calls, "cdr")), 3)
        self.assertEqual(len(N.blocks(calls, "cr")), 2)


class TestStatistic(unittest.TestCase):
    def test_avg_rank(self):
        self.assertEqual(N.avg_rank([3, 1, 2, 2]).tolist(), [4.0, 1.0, 2.5, 2.5])

    def test_contrast(self):
        sr = np.array([0, 1, 0, 1, 0, 1.0])
        sc = np.array([0, 0, 1, 1, 0, 1.0])
        sd = np.array([0, 0, 0, 0, 1, 1.0])
        y = 0.1 + 0.3 * sr + 0.2 * sc - 0.1 * sd
        self.assertAlmostEqual(N.contrast(y, sr, sc, sd), 0.3, places=9)
        self.assertTrue(math.isnan(N.contrast(y, np.zeros(6), sc, sd)))

    def test_pvalue(self):
        self.assertAlmostEqual(N.pvalue(1.0, [0, 0.5, 1.0, 2.0]), 0.6)
        self.assertAlmostEqual(N.pvalue(1.0, [float("nan"), 0.0]), 2 / 3)
        self.assertTrue(math.isnan(N.pvalue(float("nan"), [0.0])))
        self.assertAlmostEqual(N.reject_rate([0.01, 0.05, 0.2, float("nan")]), 0.5)

    def test_gap_prox(self):
        # tree 0: in bag {0,1}, leaves [1,1,1,2]; tree 1: in bag {2,3}, leaves [5,5,5,6]
        L = np.array([[1, 1, 1, 2], [5, 5, 5, 6]])
        inbag = np.array([[True, True, False, False], [False, False, True, True]])
        ia = np.array([0, 0, 2, 1, 1])
        ib = np.array([1, 2, 3, 3, 2])
        # (0,2): p(0,2)=1 (tree 1, leaf with 1 in-bag call), p(2,0)=1/2 (tree 0, 2 in-bag) -> 0.75
        self.assertEqual(N.gap_pair_prox(L, inbag, ia, ib).tolist(), [0.0, 0.75, 0.0, 0.0, 0.75])

    def test_forest_fixed(self):
        f1, f2 = N.Forest(36, n_trees=5), N.Forest(36, n_trees=5)
        self.assertTrue((f1.inbag == f2.inbag).all())
        self.assertEqual(f1.inbag.sum(axis=1).tolist(), [21] * 5)     # int(0.6 * 36)

    def test_t_rf_detects_strong_receiver_effect(self):
        calls = toy_calls(delta=3.0)
        groups = N.time_groups(calls)
        ia, ib = N.eligible_pairs(calls, groups)
        self.assertEqual(len(ia), 3 * 66)
        code = {r: i for i, r in enumerate(sorted(set(c["receiver"] for c in calls)))}
        lab = np.array([code[c["receiver"]] for c in calls])
        sc = np.ones(len(ia))
        sd = np.array([calls[i]["date"] == calls[j]["date"] for i, j in zip(ia, ib)], float)
        X = np.array([c["x"] for c in calls])
        f = N.Forest(len(calls), n_trees=40)
        t1 = N.t_rf(f, X, lab, ia, ib, sc, sd)
        t2 = N.t_rf(f, X, lab, ia, ib, sc, sd)
        self.assertEqual(t1, t2)
        self.assertGreater(t1, 0.1)
        self.assertGreater(N.contrast(N.mf_y(X, ia, ib), N.same_receiver(lab, ia, ib), sc, sd),
                           0.1)

    def test_parallel_equals_serial(self):
        calls = toy_calls(delta=1.0)
        ia, ib = N.eligible_pairs(calls, N.time_groups(calls))
        lab = np.array([int(c["receiver"][-1]) + 2 * int(c["caller"][1]) for c in calls])
        sc = np.ones(len(ia))
        sd = np.zeros(len(ia))
        X = np.array([c["x"] for c in calls])
        blks = N.blocks(calls, "cdr")
        rng = np.random.default_rng(3)
        tasks = [("real", lab)] + [("real", N.permute(lab, blks, rng)) for _ in range(3)]
        a = N.run_tasks({"real": X}, tasks, ia, ib, sc, sd, 10, 7, 1)
        b = N.run_tasks({"real": X}, tasks, ia, ib, sc, sd, 10, 7, 2)
        self.assertEqual(a, b)


class TestWorlds(unittest.TestCase):
    def test_synth(self):
        calls = toy_calls()
        a = N.synth(calls, "receiver", 0.5, 11)
        b = N.synth(calls, "receiver", 0.5, 11)
        self.assertEqual(a.shape, (36, 94))
        self.assertTrue((a == b).all())
        d = N.synth(calls, "drift", 1.0, 11)
        self.assertEqual(d.shape, (36, 94))
        with self.assertRaises(ValueError):
            N.synth(calls, "other", 1.0, 1)

    def test_drift_effects(self):
        calls = toy_calls()
        rng = np.random.default_rng(0)
        E = N.drift_effects(calls, rng, 94, tau=1e12)
        # rows 0..2 are (C0, R0_0, 11/10), rows 3..5 (C0, R0_0, 11/11): one vector per day,
        # and with an enormous time scale the next day's vector is (almost) the same
        self.assertTrue(np.allclose(E[0], E[2]))
        self.assertTrue(np.allclose(E[0], E[3], atol=1e-4))
        E2 = N.drift_effects(calls, np.random.default_rng(0), 94, tau=1e-9)
        self.assertFalse(np.allclose(E2[0], E2[3], atol=0.1))
        self.assertEqual(N.date_ordinal({"date": "11/22/2019", "ts": None}),
                         N.date_ordinal({"date": "2019-11-22", "ts": None}))
        self.assertEqual(N.date_ordinal({"date": "bad", "ts": 86400.0 * 2}), 719165)


class TestVerdict(unittest.TestCase):
    def test_verdict(self):
        self.assertEqual(N.verdict_of(0.1, 0.01, 0.1, 0.05), "RECEIVER_SIGNAL_SURVIVES_DAY_BLOCKING")
        self.assertEqual(N.verdict_of(0.1, 0.2, 0.9, 0.05), "NOT_DETECTED_WITH_POWER")
        self.assertEqual(N.verdict_of(0.1, 0.2, 0.5, 0.05), "UNDERPOWERED")
        self.assertEqual(N.verdict_of(-0.1, 0.01, 1.0, 0.0), "NO_POSITIVE_CONTRAST")
        self.assertEqual(N.verdict_of(0.1, 0.01, 1.0, 0.25), "CALIBRATION_FAILED")
        self.assertEqual(N.verdict_of(0.1, 0.01, 1.0, float("nan")), "CALIBRATION_FAILED")

    def test_predictions(self):
        r = {"t_obs": 0.1, "p": {"cr": 0.01, "cdr": 0.5}, "power": 0.5,
             "drift_size_rate": 0.05, "drift_fooled_rate_cr": 0.6,
             "match": {"drift": {"reached": True}}, "model_free": {"p": {"cdr": 0.5}}}
        out, brier = N.predictions(r)
        self.assertEqual(out, {"Q1": True, "Q2": True, "Q3": False, "Q4": False, "Q5": True,
                               "Q6": True, "Q7": True, "Q8": False})
        # (.01 + .0625 + .09 + .1225 + .0025 + .25 + .16 + .0625) / 8
        self.assertAlmostEqual(brier, 0.095, places=9)

    def test_analyse_smoke(self):
        calls = toy_calls(delta=1.0)
        res = N.analyse(calls, workers=1, n_trees=10, b_primary=9, b_side=9, b_mf=9,
                        grid=[0.5, 1.0], match_reps=1, r_cal=2, b_cal=4, min_pairs=10,
                        log=lambda m: None)
        self.assertIn(res["verdict"], {"RECEIVER_SIGNAL_SURVIVES_DAY_BLOCKING",
                                       "NOT_DETECTED_WITH_POWER", "UNDERPOWERED",
                                       "NO_POSITIVE_CONTRAST", "CALIBRATION_FAILED"})
        self.assertEqual(res["n_pairs"], 198)
        self.assertEqual(res["blocks"]["cdr"]["informative_blocks"], 6)
        self.assertEqual(len(res["calibration"]["drift"]["p"]["cr"]), 2)
        self.assertIn("brier", res)


if __name__ == "__main__":
    unittest.main()
