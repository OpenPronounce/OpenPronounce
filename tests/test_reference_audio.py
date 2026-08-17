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
