import sys
import unittest

import numpy as np

from openpronounce import gop, phones

# A tiny recognizer vocabulary: blank, specials, then a few English phones (one of them twice,
# "ɑː" and "ɑ" both normalize to ɑ).
VOCAB = ("<pad>", "<s>", "</s>", "<unk>", "h", "ə", "l", "oʊ", "ɑ", "ɑː", "w", "t", "ɾ")


def _posteriors(frames, n_classes=len(VOCAB), peak=0.97):
    """Frame posteriors following ``frames`` (token id per frame, 0 = blank), as log probabilities."""
    probs = np.full((len(frames), n_classes), (1 - peak) / (n_classes - 1))
    for t, token in enumerate(frames):
        probs[t, token] = peak
    return np.log(probs)


def _say(*ids, blanks=2):
    frames = [0] * blanks
    for i in ids:
        frames += [i] * 2 + [0] * blanks
    return frames


class TestLabels(unittest.TestCase):

    def test_recognizer_spelling_maps_to_its_own_token(self):
        self.assertEqual(gop._label_ids("ɑ", VOCAB, "en"), ((8,),))
        self.assertEqual(gop._label_ids("ɑː", VOCAB, "en"), ((9,),))
        self.assertEqual(gop._label_ids("ɾ", VOCAB, "en"), ((12,),))  # flapped t stays distinct from t

    def test_normalized_phone_without_its_own_token_takes_the_group(self):
        self.assertEqual(gop._label_ids("ɜ", ("<pad>", "ɚ", "ɜː"), "en"), ((1, 2),))  # via its normalized form ɚ
        self.assertEqual(gop._label_ids("ɚ", ("<pad>", "ɜː", "ɚ"), "en"), ((2,),))
        self.assertEqual(gop._label_ids("oɹ", ("<pad>", "ɔːɹ", "oːɹ"), "en"), ((1, 2),))

    def test_unknown_phone_is_split_into_known_tokens(self):
        self.assertEqual(gop._label_ids("hə", VOCAB, "en"), ((4,), (5,)))
        self.assertEqual(gop._label_ids("x", VOCAB, "en"), ())


class TestAnalyze(unittest.TestCase):

    def test_too_short(self):
        lp = _posteriors([0, 4, 0])
        self.assertIsNone(gop.analyze(lp, VOCAB, [["h", "ə", "l", "oʊ"]], ["hello"]))

    def test_correct_reading_has_high_gop_everywhere(self):
        lp = _posteriors(_say(4, 5, 6, 7))
        analysis = gop.analyze(lp, VOCAB, [["h", "ə", "l", "oʊ"]], ["hello"])
        self.assertEqual([p["phone"] for p in analysis["phones"]], ["h", "ə", "l", "oʊ"])
        for p in analysis["phones"]:
            self.assertGreater(p["gop_margin"], -0.1)
            self.assertGreater(p["gop_del"], 1.0)
            self.assertIsNone(p["heard"])
        self.assertGreater(analysis["utterance"]["plausible_phone_ratio"], 0.99)
        starts = [p["start"] for p in analysis["phones"]]
        self.assertEqual(starts, sorted(starts))

    def test_substituted_phone_gets_low_gop_and_the_heard_token(self):
        # "hɑloʊ" for "həloʊ"
        lp = _posteriors(_say(4, 8, 6, 7))
        analysis = gop.analyze(lp, VOCAB, [["h", "ə", "l", "oʊ"]], ["hello"])
        schwa = analysis["phones"][1]
        others = [p for p in analysis["phones"] if p is not schwa]
        self.assertLess(schwa["gop_margin"], min(p["gop_margin"] for p in others) - 1)
        self.assertEqual(VOCAB[schwa["heard"]], "ɑ")
        self.assertLess(schwa["gop_del"], min(p["gop_del"] for p in others))

    def test_espeak_spelling_is_aligned_on_its_own_token(self):
        # espeak writes ɑː, the recognizer emits ɑː: a perfect match, while a plain ɑ is a different token
        lp = _posteriors(_say(10, 9, 11))
        analysis = gop.analyze(lp, VOCAB, [["w", "ɑː", "t"]], ["what"])
        self.assertGreater(analysis["phones"][1]["gop_margin"], -0.1)
        analysis = gop.analyze(lp, VOCAB, [["w", "ɑ", "t"]], ["what"])
        self.assertLess(analysis["phones"][1]["gop_margin"], -1.0)

    def test_utterance_features_are_complete_and_finite(self):
        lp = _posteriors(_say(4, 5, 6, 7) + [0] * 20 + _say(10, 8, 11))
        analysis = gop.analyze(lp, VOCAB, [["h", "ə", "l", "oʊ"], ["w", "ɑ", "t"]], ["hello", "what"])
        features = gop.utterance_features(analysis)
        self.assertEqual(list(features), gop.UTTERANCE_FEATURES)
        self.assertTrue(all(np.isfinite(v) for v in features.values()))
        self.assertEqual(features["n_words"], 2)
        self.assertEqual(features["n_pauses"], 1)


class TestWordModel(unittest.TestCase):

    def test_logistic_model_predicts_probabilities(self):
        model = gop.LogisticModel({"features": ["a", "b"], "mean": [0, 0], "scale": [1, 1], "coef": [2.0, 0.0],
                                   "intercept": 0.0, "phone_bias": {"h": 1.0}})
        p = model.predict(np.array([[0.0, 5.0], [1.0, 0.0]]), ["x", "h"])
        self.assertAlmostEqual(p[0], 0.5)
        self.assertAlmostEqual(p[1], 1 / (1 + np.exp(-3.0)))

    def test_shipped_english_model_loads(self):
        model = gop.load_model("en")
        self.assertIsNotNone(model)
        self.assertEqual(model.phone.features, gop.PHONE_FEATURES)
        self.assertEqual(model.word.features, gop.WORD_FEATURES)
        self.assertTrue(0 < model.threshold < 1)


class TestComparePhonesWithPosteriors(unittest.TestCase):

    def test_reports_words_with_probabilities_and_reliability(self):
        if gop.load_model("en") is None:
            self.skipTest("no English word model")
        lp = _posteriors(_say(4, 5, 6, 7) + [0] * 10 + _say(10, 8, 11))
        recognition = phones.decode_ctc(lp, VOCAB)
        result = phones.compare_phones(recognition, "hello what")
        self.assertIn("words", result)
        self.assertIn("alignment", result)
        self.assertEqual([w["word"] for w in result["words"]], ["hello", "what"])
        self.assertTrue(result["alignment"]["reliable"])
        for word in result["words"]:
            self.assertTrue(0 <= word["confidence"] <= 1)
            self.assertGreater(len(word["phones"]), 0)
            for phone in word["phones"]:
                self.assertTrue(0 <= phone["confidence"] <= 1)
                self.assertLessEqual(phone["start"], phone["end"])
        self.assertEqual([e["word"] for e in result["errors"]], [w["word"] for w in result["words"] if w["mispronounced"]])

    def test_another_sentence_is_unreliable(self):
        if gop.load_model("en") is None:
            self.skipTest("no English word model")
        lp = _posteriors(_say(4, 5, 6, 7) + _say(4, 5, 6, 7))  # "hello hello" said for "what what what"
        result = phones.compare_phones(phones.decode_ctc(lp, VOCAB), "what what what")
        self.assertFalse(result["alignment"]["reliable"])
        self.assertLess(result["alignment"]["plausible_phone_ratio"], 0.5)

    def test_rule_path_is_still_available(self):
        lp = _posteriors(_say(4, 5, 6, 7))
        result = phones.compare_phones(phones.decode_ctc(lp, VOCAB), "hello", use_model=False)
        self.assertNotIn("words", result)
        self.assertEqual(result["errors"], [])


class TestExportedModels(unittest.TestCase):
    """The JSON model classes reproduce scikit-learn's predictions."""

    def setUp(self):
        rng = np.random.default_rng(0)
        self.X = rng.normal(size=(400, 5))
        self.y = (self.X[:, 0] + self.X[:, 1] ** 2 - self.X[:, 2] * self.X[:, 3] > 0.5).astype(int)

    def test_mlp_matches_sklearn(self):
        from sklearn.neural_network import MLPClassifier

        sys.path.insert(0, "benchmarks")
        from word_model import export_mlp

        mean, scale = self.X.mean(axis=0), self.X.std(axis=0)
        clf = MLPClassifier((16, 8), max_iter=500, random_state=0).fit((self.X - mean) / scale, self.y)
        model = gop.MLPModel(export_mlp(clf, ["a", "b", "c", "d", "e"], mean, scale, []))
        np.testing.assert_allclose(model.predict(self.X), clf.predict_proba((self.X - mean) / scale)[:, 1], atol=1e-6)

    def test_trees_match_sklearn(self):
        from sklearn.ensemble import HistGradientBoostingClassifier

        sys.path.insert(0, "benchmarks")
        from word_model import export_trees

        clf = HistGradientBoostingClassifier(max_iter=30, max_leaf_nodes=8, random_state=0).fit(self.X, self.y)
        model = gop.TreeEnsemble(export_trees(clf, ["a", "b", "c", "d", "e"]))
        np.testing.assert_allclose(model.predict(self.X), clf.predict_proba(self.X)[:, 1], atol=1e-6)

    def test_regression_trees_match_sklearn(self):
        from sklearn.ensemble import HistGradientBoostingRegressor

        sys.path.insert(0, "benchmarks")
        from word_model import export_trees

        target = self.X[:, 0] * 2 + np.sin(self.X[:, 1])
        reg = HistGradientBoostingRegressor(max_iter=30, max_leaf_nodes=8, random_state=0).fit(self.X, target)
        model = gop.TreeEnsemble(export_trees(reg, ["a", "b", "c", "d", "e"]))
        np.testing.assert_allclose(model.predict(self.X), reg.predict(self.X), atol=1e-6)
