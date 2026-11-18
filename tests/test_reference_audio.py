import unittest
from unittest.mock import patch

import numpy as np

from openpronounce import speech


@patch("openpronounce.speech.phones.is_enabled", return_value=False)
@patch("openpronounce.speech.transcribe", return_value="hello world")
@patch("openpronounce.speech.audio.text2speech")
class TestReferenceAudio(unittest.TestCase):

    def test_reference_audio_replaces_tts(self, mock_tts, _transcribe, _enabled):
        learner = np.zeros(16000, dtype=np.float32)
        reference = np.ones(16000, dtype=np.float32)
        with patch("openpronounce.speech.extract_embeddings", side_effect=lambda a, sr: np.full((5, 2), float(a[0]))) as emb:
            result = speech.compare_audio_with_text(learner, "hello world", reference_audio=reference, prosody=False)
        mock_tts.assert_not_called()
        self.assertEqual(emb.call_count, 2)
        self.assertNotIn("prosody", result)
        self.assertAlmostEqual(result["acoustic_distance"], np.sqrt(2), places=3)

    def test_reference_embeddings_skip_encoding(self, mock_tts, _transcribe, _enabled):
        learner = np.zeros(16000, dtype=np.float32)
        with patch("openpronounce.speech.extract_embeddings", return_value=np.zeros((5, 2))) as emb:
            result = speech.compare_audio_with_text(learner, "hello world", reference_embeddings=np.zeros((5, 2)),
                                                    prosody=False)
        mock_tts.assert_not_called()
        self.assertEqual(emb.call_count, 1)
        self.assertEqual(result["acoustic_distance"], 0.0)

    @patch("openpronounce.speech.extract_f0", return_value=np.array([100.0]))
    @patch("openpronounce.speech.extract_energy", return_value=np.array([1.0]))
    def test_default_still_uses_tts_and_prosody(self, _energy, _f0, mock_tts, _transcribe, _enabled):
        mock_tts.return_value = "ref.wav"
        with patch("openpronounce.speech.audio.load", return_value=np.zeros(16000, dtype=np.float32)), \
             patch("openpronounce.speech.extract_embeddings", return_value=np.zeros((5, 2))):
            result = speech.compare_audio_with_text(np.zeros(16000, dtype=np.float32), "hello world")
        mock_tts.assert_called_once()
        self.assertIn("prosody", result)


class TestPhoneModelOnly(unittest.TestCase):

    PHONE_RESULT = {"errors": [], "words_with_errors": [], "phone_error_rate": 0.1, "expected_phones": [["h", "aɪ"]],
                    "heard_phones": ["h", "aɪ"], "heard_phones_confidence": [0.9, 0.9], "words": [], "alignment": {}}
    ANALYSIS = {"analysis": {}, "phone_probabilities": [], "word_probabilities": {}, "threshold": 0.3}

    def test_transcription_false_never_touches_the_word_model(self):
        scores = {"total": 80.0, "accuracy": 81.0, "fluency": 79.0, "prosodic": 78.0, "sentence_probability": 0.9}
        with patch("openpronounce.speech._load_models", side_effect=AssertionError("word model loaded")), \
                patch("openpronounce.speech.audio.text2speech", side_effect=AssertionError("reference synthesized")), \
                patch("openpronounce.speech.phones.is_enabled", return_value=True), \
                patch("openpronounce.speech.phones.recognize_phones", return_value="recognition"), \
                patch("openpronounce.speech.phones.compare_phones", return_value=(dict(self.PHONE_RESULT, alignment={}), self.ANALYSIS)), \
                patch("openpronounce.speech.scorer.load_scorer", return_value=object()), \
                patch("openpronounce.speech.scorer.features", return_value={}), \
                patch("openpronounce.speech.scorer.score", return_value=dict(scores)):
            result = speech.compare_audio_with_text(np.zeros(16000, dtype=np.float32), "hi", acoustic=False,
                                                    transcription=False, prosody=False)
        self.assertEqual(result["score"], 80.0)
        self.assertIsNone(result["transcribe"])
        self.assertIsNone(result["formula_score"])
        self.assertTrue(result["differences"]["alignment"]["reliable"])
        self.assertEqual(result["differences"]["phoneme_error_rate"], 0.1)
        self.assertIn("silent", result["audio_quality"]["issues"])

    def test_transcription_false_needs_a_trained_scorer(self):
        with patch("openpronounce.speech.phones.is_enabled", return_value=True), \
                patch("openpronounce.speech.scorer.load_scorer", return_value=None):
            with self.assertRaisesRegex(ValueError, "trained scorer"):
                speech.compare_audio_with_text(np.zeros(16000, dtype=np.float32), "bonjour", lang="fr", acoustic=False,
                                               transcription=False)


class TestReliability(unittest.TestCase):

    @staticmethod
    def _voice(seconds=3.0):
        t = np.arange(int(seconds * 16000)) / 16000
        return (0.3 * (np.sin(2 * np.pi * 1.5 * t) > 0) * np.sin(2 * np.pi * 200 * t)).astype(np.float32)

    def _assess(self, sound, sentence_probability):
        scores = {"total": 45.0, "accuracy": 45.0, "fluency": 45.0, "prosodic": 45.0, "sentence_probability": sentence_probability}
        with patch("openpronounce.speech.phones.is_enabled", return_value=True), \
                patch("openpronounce.speech.phones.recognize_phones", return_value="recognition"), \
                patch("openpronounce.speech.phones.compare_phones",
                      return_value=(dict(TestPhoneModelOnly.PHONE_RESULT, alignment={}), TestPhoneModelOnly.ANALYSIS)), \
                patch("openpronounce.speech.scorer.load_scorer", return_value=object()), \
                patch("openpronounce.speech.scorer.features", return_value={}), \
                patch("openpronounce.speech.scorer.score", return_value=scores):
            return speech.compare_audio_with_text(sound, "hi", acoustic=False, transcription=False, prosody=False)

    def test_a_good_recording_of_the_sentence_is_reliable(self):
        result = self._assess(self._voice(), 0.97)
        self.assertTrue(result["reliable"])
        self.assertEqual(result["reasons"], [])

    def test_another_sentence_is_not(self):
        result = self._assess(self._voice(), 0.02)
        self.assertFalse(result["reliable"])
        self.assertEqual(result["reasons"], ["the recording does not match the sentence"])
        self.assertEqual(result["score"], 45.0)

    def test_a_poor_recording_is_not(self):
        result = self._assess(np.clip(10 * self._voice(), -1, 1), 0.97)
        self.assertEqual(result["reasons"], ["recording clipped"])


class TestCliSummary(unittest.TestCase):

    def _run(self, result):
        import contextlib
        import io

        from openpronounce import cli
        out = io.StringIO()
        with patch("openpronounce.audio.load", return_value=np.zeros(10)), \
                patch("openpronounce.speech.compare_audio_with_text", return_value=result), contextlib.redirect_stdout(out):
            cli.main(["recording.wav", "hi", "--fast"])
        return out.getvalue()

    def test_unreliable_result_says_why_and_claims_nothing_about_the_words(self):
        text = self._run({"score": 45.0, "scores": None, "reliable": False, "transcribe": None,
                          "reasons": ["the recording does not match the sentence"], "differences": {"errors": []}})
        self.assertIn("Warning      : the recording does not match the sentence", text)
        self.assertNotIn("No mispronounced word", text)
        self.assertNotIn("Transcription", text)

    def test_reliable_result(self):
        text = self._run({"score": 82.0, "scores": None, "reliable": True, "transcribe": "HI", "reasons": [],
                          "differences": {"errors": []}})
        self.assertIn("No mispronounced word detected.", text)
        self.assertNotIn("Warning", text)
