import json
import os
import tempfile
import unittest

import numpy as np

from openpronounce import gop, scorer


def _tone(seconds=1.0, f0=150.0, sr=16000):
    t = np.arange(int(seconds * sr)) / sr
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 2 * t)  # 2 Hz amplitude modulation: energy dips
    return (0.3 * envelope * np.sin(2 * np.pi * f0 * t)).astype(np.float32)


class TestProsodyFeatures(unittest.TestCase):

    def test_names_and_pitch(self):
        features = scorer.prosody_features(_tone(f0=150.0))
        self.assertEqual(sorted(features), sorted(scorer.PROSODY_FEATURES))
        self.assertAlmostEqual(features["f0_mean"], 150.0, delta=10.0)
        self.assertAlmostEqual(features["duration"], 1.0, places=3)
        self.assertTrue(all(np.isfinite(v) for v in features.values()))

    def test_very_short_waveform(self):
        features = scorer.prosody_features(np.zeros(100, dtype=np.float32))
        self.assertTrue(all(np.isfinite(v) for v in features.values()))


class TestScorer(unittest.TestCase):

    def test_predict_applies_calibration_and_clips(self):
        tree = {"type": "trees", "features": ["a", "b"], "baseline": 5.0, "link": "identity",
                "trees": [[[0, 0.5, 1, 2, 0.0], [-1, 0.0, 0, 0, -1.0], [-1, 0.0, 0, 0, 4.0]]]}
        model = scorer.Scorer({"features": ["a", "b"], "models": {"total": tree}, "calibration": {"total": [2.0, 0.0]}})
        self.assertEqual(model.predict({"a": 0.0, "b": 0.0})["total"], 8.0)   # (5 - 1) * 2
        self.assertEqual(model.predict({"a": 1.0, "b": 0.0})["total"], 10.0)  # (5 + 4) * 2, clipped

    def test_shipped_english_model(self):
        model = scorer.load_scorer("en")
        self.assertIsNotNone(model)
        self.assertEqual(model.features, scorer.FEATURES)
        self.assertEqual(sorted(model.models), sorted(scorer.TARGETS))
        row = {name: 0.0 for name in scorer.FEATURES}
        scores = model.predict(row)
        for target in scorer.TARGETS:
            self.assertTrue(0.0 <= scores[target] <= 10.0)
        if model.sentence_match is not None:
            p = model.match_probability(np.array([[row[n] for n in model.features]]))
            self.assertTrue(0.0 <= p[0] <= 1.0)

    def test_score_is_on_a_hundred(self):
        row = {name: 0.0 for name in scorer.FEATURES}
        scores = scorer.score(row, "en")
        self.assertIsNotNone(scores)
        for target in scorer.TARGETS:
            self.assertTrue(0.0 <= scores[target] <= 100.0)
        self.assertIsNone(scorer.score(row, "xx"))

    def test_model_json_roundtrip(self):
        model = scorer.load_scorer("en")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.json")
            model.save(path)
            again = scorer.Scorer.load(path)
        row = np.zeros((1, len(model.features)))
        for target in scorer.TARGETS:
            self.assertEqual(model.predict_matrix(row)[target][0], again.predict_matrix(row)[target][0])
