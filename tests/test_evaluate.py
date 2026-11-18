import contextlib
import csv
import io
import os
import tempfile
import unittest

import numpy as np

from openpronounce import evaluate


class TestEvaluate(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.manifest = os.path.join(self.tmp.name, "manifest.csv")
        self.out = os.path.join(self.tmp.name, "results.csv")

    def _write(self, rows, columns=("audio", "text", "score", "l1")):
        with open(self.manifest, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            writer.writerows(rows)

    def _rows(self, n=40):
        rng = np.random.default_rng(0)
        human = rng.uniform(1, 5, n)
        self._write([(f"clips/{i}.wav", "hello" if i % 2 else "", f"{h:.2f}".replace(".", ","), "fr" if i < n // 2 else "es")
                     for i, h in enumerate(human)])
        return human

    @staticmethod
    def _assess(human, noise=0.0):
        rng = np.random.default_rng(1)

        def assess(path, text, lang):
            index = int(os.path.basename(path)[:-4])
            return {"mode": "read" if text else "free", "score": round(20 * human[index] + noise * rng.standard_normal(), 2),
                    "accuracy": 70.0, "fluency": 70.0, "prosodic": 70.0, "reliable": index != 3, "issues": "noisy" if index == 5 else ""}
        return assess

    def test_manifest_paths_scores_and_modes(self):
        self._rows()
        rows = evaluate.read_manifest(self.manifest)
        self.assertEqual(rows[0]["audio"], os.path.join(self.tmp.name, "clips", "0.wav"))
        self.assertEqual(rows[0]["text"], "")
        self.assertEqual(rows[1]["text"], "hello")
        self.assertIsInstance(rows[0]["score"], float)  # decimal comma accepted
        self.assertEqual(rows[0]["l1"], "fr")

    def test_missing_column(self):
        self._write([("a.wav", "hello")], columns=("audio", "text"))
        with self.assertRaisesRegex(ValueError, "missing column"):
            evaluate.read_manifest(self.manifest)

    def test_perfect_agreement_and_scale_mapping(self):
        human = self._rows()
        rows = evaluate.read_manifest(self.manifest)
        results = evaluate.run(rows, self.out, assess=self._assess(human))
        lines, figures = evaluate.report(rows, results, group_by=["l1"])
        self.assertAlmostEqual(figures["all"]["pearson"], 1.0, places=3)
        self.assertAlmostEqual(figures["all"]["slope"], 0.05, places=3)   # ours is 20 x theirs
        self.assertAlmostEqual(figures["all"]["mae"], 0.0, places=2)
        self.assertEqual(figures["groups"][("l1", "fr")]["n"], 20)
        self.assertIn("flagged unreliable: 1, poor recording quality: 1", "\n".join(lines))
        self.assertIn("fewer than 100 rated recordings", lines[-1])

    def test_noise_lowers_the_agreement_and_gives_an_interval(self):
        human = self._rows(60)
        rows = evaluate.read_manifest(self.manifest)
        results = evaluate.run(rows, self.out, assess=self._assess(human, noise=15.0))
        _, figures = evaluate.report(rows, results)
        a = figures["all"]
        self.assertLess(a["pearson"], 0.95)
        self.assertLess(a["low"], a["pearson"])
        self.assertGreater(a["high"], a["pearson"])
        self.assertGreater(a["mae"], 0.2)

    def test_run_resumes_and_survives_a_broken_file(self):
        human = self._rows(10)
        rows = evaluate.read_manifest(self.manifest)
        calls = []

        def flaky(path, text, lang):
            calls.append(path)
            if path.endswith("4.wav"):
                raise RuntimeError("cannot decode")
            return self._assess(human)(path, text, lang)

        evaluate.run(rows[:6], self.out, assess=flaky)
        results = evaluate.run(rows, self.out, assess=flaky)
        self.assertEqual(len(calls), 10)   # nothing scored twice
        self.assertEqual(len(results), 10)
        self.assertIn("cannot decode", results[rows[4]["audio"]]["error"])
        lines, figures = evaluate.report(rows, results)
        self.assertEqual(figures["all"]["n"], 9)
        self.assertIn("1 could not be scored", lines[0])

    def test_too_few_recordings(self):
        human = self._rows(2)
        rows = evaluate.read_manifest(self.manifest)
        lines, figures = evaluate.report(rows, evaluate.run(rows, self.out, assess=self._assess(human)))
        self.assertIsNone(figures["all"])
        self.assertIn("not enough rated recordings", lines[-1])

    def test_cli_report_only(self):
        human = self._rows()
        rows = evaluate.read_manifest(self.manifest)
        evaluate.run(rows, self.out, assess=self._assess(human))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(evaluate.main([self.manifest, "--out", self.out, "--report-only", "--group-by", "l1"]), 0)
        self.assertIn("l1=es", out.getvalue())
        self.assertIn("yours = 0.0500 x ours", out.getvalue())
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            evaluate.main([self.manifest, "--report-only", "--group-by", "nope"])


if __name__ == "__main__":
    unittest.main()
