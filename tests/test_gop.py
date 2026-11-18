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


def _recognition(lp, lang="en"):
    """Decode ``lp`` with the hidden states the word model of ``lang`` expects (zeros when it reads some)."""
    model = gop.load_model(lang)
    hidden = None
    if model is not None and model.hidden_layer is not None:
        hidden = np.zeros((len(lp), model.phone.projection.shape[0]), dtype=np.float32)
    return phones.decode_ctc(lp, VOCAB, hidden=hidden)


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


class TestLanguageVariants(unittest.TestCase):

    VOCAB = ("<pad>", "ʋ", "v", "ɪ", "t", "tː", "iː", "i")

    def test_english_accepts_nothing_else(self):
        self.assertEqual(gop._label_ids("v", self.VOCAB, "en"), ((2,),))
        self.assertEqual(gop._label_ids("iː", self.VOCAB, "en"), ((6,),))

    def test_dutch_w_is_also_heard_as_v(self):
        self.assertEqual(gop._label_ids("ʋ", self.VOCAB, "nl"), ((1, 2),))
        self.assertEqual(gop._label_ids("ʋ", self.VOCAB, "fr"), ((1,),))

    def test_long_consonant_accepts_the_short_one_but_a_long_vowel_does_not(self):
        self.assertEqual(gop._label_ids("tː", self.VOCAB, "it"), ((5, 4),))
        self.assertEqual(gop._label_ids("iː", self.VOCAB, "it"), ((6,),))


class TestLanguageThreshold(unittest.TestCase):

    def test_english_keeps_the_threshold_of_its_own_model(self):
        from openpronounce.languages import LANGUAGES
        self.assertIsNone(LANGUAGES["en"].word_threshold)
        self.assertFalse(gop.load_model("en").generic)

    def test_other_languages_report_from_a_lower_probability_than_the_generic_model(self):
        from openpronounce.languages import LANGUAGES
        model = gop.load_model("fr")
        self.assertTrue(model.generic)
        for code, language in LANGUAGES.items():
            if code != "en":
                self.assertGreater(language.word_threshold, 0.0)
                self.assertLess(language.word_threshold, model.threshold)


class TestRescale(unittest.TestCase):

    def test_the_language_threshold_lands_on_the_model_threshold(self):
        from openpronounce import phones
        self.assertAlmostEqual(phones._rescale(0.09, 0.09, 0.31), 0.31)
        self.assertAlmostEqual(phones._rescale(0.0, 0.09, 0.31), 0.0)
        self.assertAlmostEqual(phones._rescale(1.0, 0.09, 0.31), 1.0)
        self.assertLess(phones._rescale(0.05, 0.09, 0.31), 0.31)
        self.assertGreater(phones._rescale(0.2, 0.09, 0.31), 0.31)


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

    def test_hidden_states_give_each_phone_an_embedding(self):
        lp = _posteriors(_say(4, 5, 6, 7))
        hidden = np.arange(len(lp), dtype=np.float32)[:, None] * np.ones((1, 3))
        analysis = gop.analyze(lp, VOCAB, [["h", "ə", "l", "oʊ"]], ["hello"], hidden=hidden)
        for p in analysis["phones"]:
            self.assertEqual(p["embedding"].shape, (3,))
            frames = hidden[max(0, p["start"] - gop.EMBEDDING_MARGIN):p["end"] + gop.EMBEDDING_MARGIN]
            np.testing.assert_allclose(p["embedding"], frames.mean(axis=0))
        self.assertEqual(gop.embedding_matrix(analysis["phones"]).shape, (4, 3))

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
        result = phones.compare_phones(_recognition(lp), "hello what")
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
        result = phones.compare_phones(_recognition(lp), "what what what")
        self.assertFalse(result["alignment"]["reliable"])
        self.assertLess(result["alignment"]["plausible_phone_ratio"], 0.5)

    def test_without_the_hidden_states_the_model_needs_falls_back_to_the_rule(self):
        model = gop.load_model("en")
        if model is None or model.hidden_layer is None:
            self.skipTest("the English word model reads posteriors only")
        lp = _posteriors(_say(4, 5, 6, 7))
        result = phones.compare_phones(phones.decode_ctc(lp, VOCAB), "hello")
        self.assertNotIn("words", result)

    def test_rule_path_is_still_available(self):
        lp = _posteriors(_say(4, 5, 6, 7))
        result = phones.compare_phones(phones.decode_ctc(lp, VOCAB), "hello", use_model=False)
        self.assertNotIn("words", result)
        self.assertEqual(result["errors"], [])


class TestMissingAndExtra(unittest.TestCase):

    GROUPS, WORDS = [["h", "ə", "l", "oʊ"], ["w", "ɑ", "t"]], ["hello", "what"]

    def _extra(self, frames):
        lp = _posteriors(frames)
        recognition = phones.decode_ctc(lp, VOCAB)
        analysis = gop.analyze(lp, VOCAB, self.GROUPS, self.WORDS)
        return recognition, gop.extra_speech(analysis, recognition.spans, recognition.confidences)

    def test_exact_reading_has_no_extra_speech(self):
        _, events = self._extra(_say(4, 5, 6, 7) + [0] * 6 + _say(10, 8, 11))
        self.assertEqual(events, [])

    def test_speech_between_two_words_is_located(self):
        recognition, events = self._extra(_say(4, 5, 6, 7) + [0] * 6 + _say(11, 4, 11) + [0] * 6 + _say(10, 8, 11))
        self.assertEqual(len(events), 1)
        self.assertEqual((events[0]["after"], events[0]["before"]), (0, 1))
        self.assertEqual([recognition.phones[i] for i in events[0]["heard"]], ["t", "h", "t"])
        self.assertLess(events[0]["start"], events[0]["end"])

    def test_speech_after_the_sentence_is_located(self):
        recognition, events = self._extra(_say(4, 5, 6, 7) + [0] * 6 + _say(10, 8, 11) + [0] * 6 + _say(4, 5, 6, 7))
        self.assertEqual([(e["after"], e["before"]) for e in events], [(1, None)])
        self.assertEqual(len(events[0]["heard"]), 4)

    def test_missing_features_tell_a_word_that_was_not_said(self):
        lp = _posteriors(_say(4, 5, 6, 7) + [0] * 12)
        recognition = phones.decode_ctc(lp, VOCAB)
        analysis = gop.analyze(lp, VOCAB, self.GROUPS, self.WORDS)
        indices, matrix = gop.missing_matrix(analysis, recognition.spans, recognition.confidences)
        self.assertEqual(indices, [0, 1])
        self.assertEqual(matrix.shape, (2, len(gop.MISSING_FEATURES)))
        said, absent = (dict(zip(gop.MISSING_FEATURES, row)) for row in matrix)
        self.assertGreater(said["del_mean"], absent["del_mean"])
        self.assertGreater(said["heard_per_phone"], absent["heard_per_phone"])
        self.assertEqual(absent["heard_per_phone"], 0)

    def test_word_that_was_not_said_gets_a_higher_missing_confidence(self):
        if gop.load_model("en") is None or gop.load_missing_model("en") is None:
            self.skipTest("no English models")
        lp = _posteriors(_say(4, 5, 6, 7) + [0] * 12)
        result = phones.compare_phones(_recognition(lp), "hello what")
        hello, what = result["words"]
        self.assertGreater(what["missing_confidence"], hello["missing_confidence"])
        self.assertFalse(hello.get("missing", False))
        self.assertEqual(result["extra"], [])
        for error in result["errors"]:
            self.assertEqual(error.get("missing", False), bool(result["words"][error["position"]].get("missing")))


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

    def test_mlp_with_embeddings_matches_sklearn(self):
        from sklearn.decomposition import PCA
        from sklearn.neural_network import MLPClassifier

        sys.path.insert(0, "benchmarks")
        from word_model import export_mlp

        rng = np.random.default_rng(1)
        E = rng.normal(size=(400, 12)) + self.y[:, None]
        pca = PCA(4, random_state=0).fit(E)
        Z = pca.transform(E)
        scale = Z.std(axis=0) + 1e-9
        mean, sd = self.X.mean(axis=0), self.X.std(axis=0)
        inputs = np.column_stack([(self.X - mean) / sd, Z / scale])
        clf = MLPClassifier((16, 8), max_iter=500, random_state=0).fit(inputs, self.y)
        block = {"layer": 3, "mean": pca.mean_.tolist(), "projection": pca.components_.T.tolist(), "scale": scale.tolist()}
        model = gop.MLPModel(export_mlp(clf, ["a", "b", "c", "d", "e"], mean, sd, [], block))
        self.assertEqual(model.hidden_layer, 3)
        np.testing.assert_allclose(model.predict(self.X, embeddings=E), clf.predict_proba(inputs)[:, 1], atol=1e-6)
        with self.assertRaises(ValueError):
            model.predict(self.X)

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
