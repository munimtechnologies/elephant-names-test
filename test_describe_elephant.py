# SPDX-License-Identifier: Apache-2.0
# Toy tests for describe_elephant.py (every expected value worked out by hand).
# MV2, Munim, Inc. (mv2@munimtech.com). Made with AI agents. Apache-2.0.
import math
import unittest

import describe_elephant as D

# caller A: a1 (r1, d1), a2 (r1, d1), a3 (r2, d1), a4 (r2, d2); caller B: b1, b2 (r3, d1)
CALLS = [("A", "r1", "d1"), ("A", "r1", "d1"), ("A", "r2", "d1"), ("A", "r2", "d2"),
         ("B", "r3", "d1"), ("B", "r3", "d1")]


def row(c, r, d, sound="20191122_1012.WAV", feat="0.5", cert="1"):
    x = {k: feat for k in D.FEATURES}
    x.update({"CallerID": c, "Elicitor1_ID": r, "date": d, "sound.files": sound,
              "Call_Type": "RUM", "Context2": "greeting", "newSocCat": "A2A",
              "Cert_ID_Bin": cert, "Cert_ID_Elic1_Bin": "1"})
    return x


class TestDescribe(unittest.TestCase):
    def test_pairs(self):
        # A: a1a2 SR-SD, a1a3 DR-SD, a1a4 DR-DD, a2a3 DR-SD, a2a4 DR-DD, a3a4 SR-DD; B: SR-SD
        self.assertEqual(D.pair_table(CALLS), {
            "same_receiver_same_date": 2, "same_receiver_diff_date": 1,
            "diff_receiver_same_date": 2, "diff_receiver_diff_date": 2, "callers": 2,
            "callers_with_2plus_calls": 2, "callers_with_2plus_receivers": 1})

    def test_blocks(self):
        b = D.block_table(CALLS)
        # (A, d1) = {r1: 2, r2: 1} is the only block with two receivers: 3 pairs, 1 same
        self.assertEqual({k: v for k, v in b.items() if k != "log10_relabelings"}, {
            "blocks": 3, "blocks_2plus_calls": 2, "blocks_2plus_receivers": 1,
            "callers_with_informative_block": 1, "calls_in_informative_blocks": 3,
            "pairs_in_informative_blocks_same_receiver": 1,
            "pairs_in_informative_blocks_diff_receiver": 2, "informative_block_sizes": [3]})
        # 3! / (2! 1!) = 3 relabelings
        self.assertAlmostEqual(b["log10_relabelings"], math.log10(3), places=9)
        self.assertAlmostEqual(D.log10_relabelings([2, 2]), math.log10(6), places=9)

    def test_helpers(self):
        self.assertIsNone(D.clean(" NA "))
        self.assertEqual(D.clean(" R7 "), "R7")
        self.assertTrue(D.is_num("-1.5e-3"))
        self.assertFalse(D.is_num("NA"))
        self.assertFalse(D.is_num("nan"))
        self.assertFalse(D.is_num("abc"))
        self.assertEqual(D.site("20191122_1012.WAV"), "samburu")
        self.assertEqual(D.site("AMB_tape_12.wav"), "amboseli")
        self.assertEqual(len(D.FEATURES), 94)

    def test_describe(self):
        rows = [row(c, r, d) for c, r, d in CALLS]
        rows.append(row("C", "NA", "d1"))                         # no receiver
        rows.append(row("C", "r1", "d1", feat="NA"))              # incomplete features
        rows.append(row("A", "r1", "d2", sound="AMB1.wav", cert="0"))
        d = D.describe(rows)
        self.assertEqual(d["n_rows"], 9)
        self.assertEqual(d["n_complete_features"], 8)
        self.assertEqual(d["n_with_caller_receiver_date"], 7)
        self.assertEqual(d["site"], {"samburu": 6, "amboseli": 1})
        self.assertEqual(d["n_callers"], 2)
        self.assertEqual(d["n_receivers"], 3)
        self.assertEqual(d["n_dates"], 2)
        self.assertEqual(d["n_dyads"], 3)                          # A-r1, A-r2, B-r3
        self.assertEqual(d["calls_per_caller"], [5, 2])
        self.assertEqual(d["n_caller_and_receiver_certain"], 6)
        self.assertEqual(d["pairs_certain"], D.pair_table(CALLS))
        # the extra Amboseli call A (r1, d2) joins block (A, d2) = {r2: 1, r1: 1}
        self.assertEqual(d["blocks"]["blocks_2plus_receivers"], 2)
        self.assertEqual(d["blocks_amboseli"]["blocks"], 1)
        self.assertEqual(d["blocks_samburu"]["blocks_2plus_receivers"], 1)

    def test_read_rows(self):
        blob = b'\xef\xbb\xbfCallerID,Comments\nR7,"a, b"\n'
        self.assertEqual(D.read_rows(blob), [{"CallerID": "R7", "Comments": "a, b"}])


if __name__ == "__main__":
    unittest.main()
