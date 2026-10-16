"""Self-tests for the evaluation harness. No dataset or GPU required.

Run from the project root:
    python -m unittest eval.test_eval -v
"""

from __future__ import annotations

import contextlib
import io
import os
import tempfile
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


class TestImageEval(unittest.TestCase):
    def test_collect_images_walks_recursively_and_filters(self):
        from eval.image_eval import collect_images

        d = tempfile.mkdtemp()
        os.makedirs(os.path.join(d, "sub"))
        for name in ["a.jpg", "b.PNG", "c.jpeg", "d.webp", "e.bmp", "note.txt", "x.gif"]:
            open(os.path.join(d, name), "w").close()
        open(os.path.join(d, "sub", "nested.png"), "w").close()

        got = collect_images(d)
        names = sorted(os.path.basename(p) for p in got)
        self.assertEqual(
            names, ["a.jpg", "b.PNG", "c.jpeg", "d.webp", "e.bmp", "nested.png"]
        )
        self.assertEqual(got, sorted(got))  # deterministic ordering

    def test_collect_images_empty_when_dir_missing(self):
        from eval.image_eval import collect_images

        self.assertEqual(collect_images(os.path.join(tempfile.mkdtemp(), "nope")), [])

    def test_preprocess_shape_and_range(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow not installed")
        from eval.image_eval import preprocess

        buf = io.BytesIO()
        Image.new("RGB", (300, 200), (128, 64, 32)).save(buf, format="PNG")
        arr = preprocess(buf.getvalue())
        self.assertEqual(arr.shape, (1, 3, 224, 224))
        self.assertEqual(arr.dtype, np.float32)
        self.assertLess(float(np.abs(arr).max()), 10.0)

    def test_score_images_softmax_reads_ai_index(self):
        import eval.image_eval as ie

        class FakeSession:
            def get_inputs(self):
                return [type("I", (), {"name": "input"})()]

            def run(self, _outputs, _feed):
                return [np.array([[0.0, 5.0]], dtype=np.float32)]  # favours AI

        path = os.path.join(tempfile.mkdtemp(), "fake.png")
        open(path, "wb").close()

        orig = ie.preprocess
        ie.preprocess = lambda _b: np.zeros((1, 3, 224, 224), dtype=np.float32)
        try:
            probs = ie.score_images(FakeSession(), [path, path])
        finally:
            ie.preprocess = orig

        expected = float(np.exp(5.0) / (np.exp(0.0) + np.exp(5.0)))
        self.assertEqual(probs.shape, (2,))
        np.testing.assert_allclose(probs[0], expected, rtol=1e-6)
        self.assertGreater(probs[0], 0.99)

    def test_score_images_symmetric_logits_give_half(self):
        import eval.image_eval as ie

        class FlatSession:
            def get_inputs(self):
                return [type("I", (), {"name": "input"})()]

            def run(self, _outputs, _feed):
                return [np.array([[1.0, 1.0]], dtype=np.float32)]

        path = os.path.join(tempfile.mkdtemp(), "fake.png")
        open(path, "wb").close()

        orig = ie.preprocess
        ie.preprocess = lambda _b: np.zeros((1, 3, 224, 224), dtype=np.float32)
        try:
            probs = ie.score_images(FlatSession(), [path])
        finally:
            ie.preprocess = orig
        np.testing.assert_allclose(probs[0], 0.5, rtol=1e-9)


class TestImageEvalMain(unittest.TestCase):
    def _run(self, argv):
        from eval.image_eval import main

        with contextlib.redirect_stdout(io.StringIO()):
            return main(argv)

    def test_missing_model_returns_1(self):
        rc = self._run(["--real", "x", "--ai", "y", "--model", "definitely-absent.onnx"])
        self.assertEqual(rc, 1)

    def test_missing_folders_returns_1(self):
        d = tempfile.mkdtemp()
        rc = self._run(
            [
                "--real",
                os.path.join(d, "no_real"),
                "--ai",
                os.path.join(d, "no_ai"),
                "--model",
                __file__,
            ]
        )
        self.assertEqual(rc, 1)

    def test_empty_folders_returns_1(self):
        d = tempfile.mkdtemp()
        real = os.path.join(d, "real")
        ai = os.path.join(d, "ai")
        os.makedirs(real)
        os.makedirs(ai)
        rc = self._run(["--real", real, "--ai", ai, "--model", __file__])
        self.assertEqual(rc, 1)


class TestFetchCifake(unittest.TestCase):
    def test_target_dir_mapping(self):
        from eval.fetch_cifake import LABEL_AI, LABEL_REAL, target_dir

        self.assertEqual(target_dir(LABEL_REAL), "real")
        self.assertEqual(target_dir(LABEL_AI), "ai")
        self.assertEqual(target_dir(0), "real")
        self.assertEqual(target_dir(1), "ai")

    def test_write_images_writes_jpgs(self):
        from eval.fetch_cifake import write_images

        d = tempfile.mkdtemp()
        raw = b"\xff\xd8\xff\xd9"  # minimal dummy JPEG bytes
        n = write_images([raw, raw], d)
        self.assertEqual(n, 2)
        self.assertEqual(len(os.listdir(d)), 2)
        # check files are readable
        for name in sorted(os.listdir(d)):
            with open(os.path.join(d, name), "rb") as f:
                self.assertEqual(f.read(), raw)


class TestMakeSplit(unittest.TestCase):
    def test_label_to_dir_mapping(self):
        from eval.make_split import label_to_dir

        self.assertEqual(label_to_dir("0", 0), "real")
        self.assertEqual(label_to_dir("1", 0), "ai")
        self.assertEqual(label_to_dir("1", "1"), "real")
        self.assertEqual(label_to_dir("0", "1"), "ai")

    def test_is_image_filters_by_extension(self):
        from eval.make_split import is_image

        self.assertTrue(is_image("a/b.JPG"))
        self.assertTrue(is_image("x.webp"))
        self.assertFalse(is_image("note.txt"))
        self.assertFalse(is_image("noext"))

    def test_builds_balanced_split_from_csv(self):
        from eval.make_split import main

        root = tempfile.mkdtemp()
        for name in ["0.jpg", "1.jpg", "2.jpg", "3.jpg"]:
            open(os.path.join(root, name), "wb").close()
        csv_path = os.path.join(root, "labels.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("file_name,label\n0.jpg,0\n1.jpg,1\n2.jpg,0\n3.jpg,1\n")
        out = os.path.join(root, "out")

        with contextlib.redirect_stdout(io.StringIO()):
            rc = main(["--csv", csv_path, "--root", root, "--out", out, "--limit", "1"])

        self.assertEqual(rc, 0)
        self.assertEqual(len(os.listdir(os.path.join(out, "real"))), 1)
        self.assertEqual(len(os.listdir(os.path.join(out, "ai"))), 1)


class TestResolveModel(unittest.TestCase):
    def test_default_when_unset(self):
        from nonescape_model import DEFAULT_MODEL, resolve_model_path

        self.assertEqual(resolve_model_path({}), DEFAULT_MODEL)
        self.assertEqual(resolve_model_path({"OTHER": "x"}), DEFAULT_MODEL)

    def test_env_override_wins(self):
        from nonescape_model import resolve_model_path

        self.assertEqual(resolve_model_path({"NONESCAPE_MODEL": "v1.onnx"}), "v1.onnx")


def _load_probe():
    import importlib.util

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "bin", "probe-nonescape.py")
    spec = importlib.util.spec_from_file_location("probe_nonescape", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestProbeNonescape(unittest.TestCase):
    def test_softmax_ai_is_sigmoid_of_logit_gap(self):
        probe = _load_probe()
        p = probe.softmax_ai(np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32))
        self.assertAlmostEqual(float(p[0]), 0.5, places=5)
        self.assertAlmostEqual(float(p[1]), 1.0 / (1.0 + np.exp(2.0)), places=5)

    def test_ece_flags_miscalibration(self):
        probe = _load_probe()
        good = probe.ece_score([1] * 90 + [0] * 10, [0.9] * 100)
        bad = probe.ece_score([1] * 50 + [0] * 50, [0.9] * 100)
        self.assertLess(good, 0.05)
        self.assertGreater(bad, good)

    def test_read_labels_and_sampling(self):
        probe = _load_probe()
        d = tempfile.mkdtemp()
        names = ["r1.jpg", "r2.jpg", "a1.jpg", "a2.jpg", "a3.jpg"]
        for n in names:
            open(os.path.join(d, n), "wb").close()
        csv_path = os.path.join(d, "train.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("file_name,label\nr1.jpg,0\nr2.jpg,0\na1.jpg,1\na2.jpg,1\na3.jpg,1\n")
        rows = probe.read_labels(csv_path, d)
        self.assertEqual(len(rows), 5)
        self.assertEqual(sum(y for _, y in rows), 3)  # three AI (label 1)
        train, val = probe.sample_balanced(rows, per_class=1, val_per_class=1, exclude_dirs=[])
        self.assertEqual(len(train), 2)  # 1 real + 1 ai
        self.assertEqual(len(val), 2)
        self.assertEqual(sum(y for _, y in train), 1)


if __name__ == "__main__":
    unittest.main()
