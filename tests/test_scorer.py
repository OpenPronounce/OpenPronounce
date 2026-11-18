import json
import os
import tempfile
import unittest
from unittest import mock

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
        # A language without its own scorer gets the generic one, for read and for free speech
        self.assertIsNotNone(scorer.score(row, "xx"))
        self.assertEqual(scorer.load_scorer("xx", "free_scorer").data["lang"], "generic")
        self.assertEqual(scorer.load_scorer("en", "free_scorer").data["lang"], "en")

    def test_model_json_roundtrip(self):
        model = scorer.load_scorer("en")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.json")
            model.save(path)
            again = scorer.Scorer.load(path)
        row = np.zeros((1, len(model.features)))
        for target in scorer.TARGETS:
            self.assertEqual(model.predict_matrix(row)[target][0], again.predict_matrix(row)[target][0])



class TestEmbeddingHead(unittest.TestCase):
    """A scorer may blend its trees with a linear head on the recognizer's pooled hidden states."""

    TREE = {"type": "trees", "features": ["a"], "baseline": 4.0, "link": "identity", "trees": []}

    def _scorer(self, share=0.5):
        head = {"layer": 20, "share": share, "center": [1.0, 0.0], "scale": [2.0, 1.0], "weights": {"total": [3.0, 1.0]},
                "intercept": {"total": 5.0}, "calibration": {"total": [1.0, 0.5]}}
        return scorer.Scorer({"features": ["a"], "models": {"total": self.TREE}, "calibration": {"total": [1.0, 0.0]},
                              "embedding": head})

    def test_pooling(self):
        pooled = scorer.pooled_embedding(np.array([[0.0, 2.0], [2.0, 2.0]]))
        np.testing.assert_allclose(pooled, [1.0, 2.0, 1.0, 0.0])

    def test_blend_and_its_calibration(self):
        model = self._scorer()
        # head: ((3 - 1) / 2) * 3 + 0 * 1 + 5 = 8; blend (4 + 8) / 2 = 6; calibration + 0.5
        self.assertAlmostEqual(model.predict({"a": 0.0, "embedding": [3.0, 0.0]})["total"], 6.5)
        self.assertAlmostEqual(self._scorer(share=1.0).predict({"a": 0.0, "embedding": [3.0, 0.0]})["total"], 8.5)

    def test_trees_answer_alone_without_an_embedding(self):
        self.assertAlmostEqual(self._scorer().predict({"a": 0.0})["total"], 4.0)
        np.testing.assert_allclose(self._scorer().predict_matrix(np.zeros((2, 1)))["total"], [4.0, 4.0])

    def test_scorer_without_a_head_ignores_the_embedding(self):
        model = scorer.Scorer({"features": ["a"], "models": {"total": self.TREE}})
        self.assertAlmostEqual(model.predict({"a": 0.0, "embedding": [3.0, 0.0]})["total"], 4.0)

    def test_another_layer_is_not_fed_to_the_head(self):
        row = {"a": 0.0, "embedding": [3.0, 0.0]}
        with mock.patch.object(scorer, "load_scorer", return_value=self._scorer()):
            with mock.patch.object(gop, "hidden_layer", return_value=20):
                self.assertEqual(scorer.score(row, "en")["total"], 65.0)
            with mock.patch.object(gop, "hidden_layer", return_value=12):
                self.assertEqual(scorer.score(row, "en")["total"], 40.0)


class TestNativeShift(unittest.TestCase):
    """A generic scorer reads the features of a language after its native shift."""

    def setUp(self):
        self.row = {name: 0.0 for name in scorer.FREE_FEATURES}

    def _predicted_with(self, lang, shift, kind="scorer"):
        seen = {}

        def predict(model, features):
            seen.update(features)
            return {target: 5.0 for target in scorer.TARGETS}

        with mock.patch.object(scorer, "native_shift", return_value=shift), \
                mock.patch.object(scorer.Scorer, "predict", predict), \
                mock.patch.object(scorer.Scorer, "match_probability", return_value=np.array([1.0])):
            scorer.score(self.row, lang, kind)
        return seen

    def test_generic_scorer_shifts_the_features(self):
        seen = self._predicted_with("xx", {"gm_mean": 0.5, "not_a_feature": 1.0})
        self.assertEqual(seen["gm_mean"], 0.5)
        self.assertEqual(seen["gm_std"], 0.0)
        self.assertNotIn("not_a_feature", seen)
        self.assertEqual(self.row["gm_mean"], 0.0)  # the caller's features are left alone

    def test_scorer_of_the_language_does_not(self):
        self.assertEqual(self._predicted_with("en", {"gm_mean": 0.5})["gm_mean"], 0.0)
        self.assertEqual(self._predicted_with("en", {"gm_mean": 0.5}, "free_scorer")["gm_mean"], 0.0)

    def test_shipped_shifts(self):
        self.assertEqual(scorer.native_shift("xx"), {})
        for kind in ("scorer", "free_scorer"):
            shift = scorer.native_shift("nl", kind)
            self.assertEqual(sorted(shift), sorted(scorer.SHIFTED_FEATURES[kind]))
            self.assertEqual(sorted(scorer.load_scorer("generic", kind).data["native_reference"]), sorted(scorer.SHIFTED_FEATURES[kind]))

    def test_offsets_bring_the_natives_onto_the_reference(self):
        reference = scorer.load_scorer("generic").data["native_reference"]
        rows = [{name: reference[name] - 2.0 for name in scorer.ALIGNMENT_FEATURES},
                {name: reference[name] - 4.0 for name in scorer.ALIGNMENT_FEATURES}]
        offsets = scorer.native_offsets(rows)
        self.assertEqual(sorted(offsets), sorted(scorer.ALIGNMENT_FEATURES))
        for value in offsets.values():
            self.assertAlmostEqual(value, 3.0, places=4)
