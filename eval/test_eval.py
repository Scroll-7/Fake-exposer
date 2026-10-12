"""Self-tests for the evaluation harness. No dataset or GPU required.

Run from the project root:
    python -m unittest eval.test_eval -v
"""

from __future__ import annotations

import unittest

import numpy as np

from eval.dedupe import cluster_stats, cluster_texts, content_hash, normalize
from eval.metrics import (
    best_threshold,
    classification_report_dict,
    expected_calibration_error,
)
from eval.split import assert_no_cluster_leak, group_stratified_split


class TestDedupe(unittest.TestCase):
    def test_normalize_strips_urls_and_punctuation(self):
        self.assertEqual(
            normalize("Hello, WORLD! https://x.com/a"), "hello world"
        )

    def test_content_hash_ignores_formatting(self):
        self.assertEqual(
            content_hash("Breaking!!!  HTTPS://a.b"),
            content_hash("breaking"),
        )

    def test_exact_duplicates_share_cluster(self):
        texts = ["the cat sat", "the cat sat", "a totally different sentence here"]
        labels, _ = cluster_texts(texts, threshold=0.9)
        self.assertEqual(labels[0], labels[1])
        self.assertNotEqual(labels[0], labels[2])

    def test_near_duplicates_share_cluster(self):
        # One changed word in a 60-word doc => Jaccard ~0.84 (above 0.8).
        base = " ".join(f"token{i}" for i in range(60))
        near = base.replace("token30", "tokenXX")
        labels, ran = cluster_texts(
            [base, near, "unrelated sentence about weather and football"],
            threshold=0.8,
        )
        if not ran:
            self.skipTest("datasketch not installed")
        self.assertEqual(labels[0], labels[1])
        self.assertNotEqual(labels[0], labels[2])

    def test_cluster_stats(self):
        texts = ["a", "a", "b", "c", "c", "c"]
        labels, _ = cluster_texts(texts)
        stats = cluster_stats(labels)
        self.assertEqual(stats["rows"], 6)
        self.assertEqual(stats["clusters"], 3)
        self.assertEqual(stats["duplicate_rows_removed"], 3)
        self.assertEqual(stats["largest_cluster"], 3)


class TestSplit(unittest.TestCase):
    def test_no_cluster_spans_the_split(self):
        cluster_ids = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
        labels = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0]
        tr, te = group_stratified_split(cluster_ids, labels, 0.3, seed=1)
        assert_no_cluster_leak(cluster_ids, tr, te)
        self.assertEqual(len(set(tr) & set(te)), 0)
        self.assertEqual(len(tr) + len(te), len(cluster_ids))

    def test_both_splits_nonempty(self):
        cluster_ids = list(range(50))
        labels = [i % 2 for i in range(50)]
        tr, te = group_stratified_split(cluster_ids, labels, 0.2, seed=7)
        self.assertGreater(len(tr), 0)
        self.assertGreater(len(te), 0)


class TestMetrics(unittest.TestCase):
    def test_perfect_predictions(self):
        y = [0, 1, 0, 1, 1]
        rep = classification_report_dict(y, y, [0.0, 1.0, 0.0, 1.0, 1.0])
        self.assertAlmostEqual(rep["accuracy"], 1.0)
        self.assertAlmostEqual(rep["f1_macro"], 1.0)
        self.assertAlmostEqual(rep["ece"], 0.0)

    def test_ece_high_when_overconfident_and_wrong(self):
        y = [0, 0, 0, 0]
        prob = [0.99, 0.99, 0.99, 0.99]
        self.assertGreater(expected_calibration_error(y, prob), 0.9)

    def test_confusion_matrix_shape(self):
        y = [0, 1, 0, 1]
        pred = [0, 0, 1, 1]
        rep = classification_report_dict(y, pred)
        self.assertEqual(np.array(rep["confusion_matrix"]).shape, (2, 2))

    def test_best_threshold_finds_separable_split(self):
        # Real ~0.1, fake ~0.9 => any threshold in between is perfect.
        y = [0, 0, 0, 1, 1, 1]
        prob = [0.1, 0.15, 0.2, 0.8, 0.85, 0.9]
        t, f1 = best_threshold(y, prob)
        self.assertGreater(f1, 0.99)
        self.assertTrue(0.2 < t < 0.8)


class TestManifest(unittest.TestCase):
    def _write(self, text):
        import tempfile

        fd, path = tempfile.mkstemp(suffix=".jsonl")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return path

    def test_valid_manifest_parses(self):
        from eval.pipeline_eval import load_manifest

        path = self._write(
            '{"id":"a","type":"text","input":"hi","label":1}\n'
            "\n"
            "# a comment line\n"
            '{"id":"b","type":"image","input":"x.jpg","label":0}\n'
        )
        items = load_manifest(path)
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]["label"], 1)

    def test_invalid_type_rejected(self):
        from eval.pipeline_eval import load_manifest

        path = self._write('{"id":"a","type":"audio","input":"x","label":1}\n')
        with self.assertRaises(SystemExit):
            load_manifest(path)

    def test_missing_label_rejected(self):
        from eval.pipeline_eval import load_manifest

        path = self._write('{"id":"a","type":"text","input":"x"}\n')
        with self.assertRaises(SystemExit):
            load_manifest(path)


if __name__ == "__main__":
    unittest.main()
