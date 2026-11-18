import contextlib
import io
import unittest
from unittest import mock

import numpy as np

import openpronounce
from openpronounce import cli, free, gop, phones, scorer


def _recognition(expected_phones, frames_per_phone=6):
    """Posteriors that say ``expected_phones`` clearly, one blank frame between phones."""
    vocab = ("<pad>", "<s>", "</s>", "<unk>") + tuple(sorted(set(expected_phones)))
    rows = []
    for phone in expected_phones:
        for _ in range(frames_per_phone):
            row = np.full(len(vocab), 0.01)
            row[vocab.index(phone)] = 1.0
            rows.append(row)
        blank = np.full(len(vocab), 0.01)
        blank[0] = 1.0
        rows.append(blank)
    posteriors = np.array(rows)
    posteriors /= posteriors.sum(axis=1, keepdims=True)
    return phones.decode_ctc(np.log(posteriors), vocab, 0, "en", True, None)


def _speech(seconds, sr=16000):
    t = np.arange(int(seconds * sr)) / sr
    return (0.3 * np.sin(2 * np.pi * 180 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)


class TestRecognitionFeatures(unittest.TestCase):

    def test_names_and_values(self):
        recognition = _recognition(["h", "ə", "l", "oʊ"])
        features = scorer.recognition_features(recognition.log_posteriors, recognition.phones, recognition.confidences,
                                               ["h", "ə", "l", "oʊ"])
        self.assertEqual(sorted(features), sorted(scorer.RECOGNITION_FEATURES))
        self.assertEqual(features["transcript_phone_error_rate"], 0.0)
        self.assertGreater(features["post_top_mean"], 0.9)
        self.assertAlmostEqual(features["blank_ratio"], 4 / 28, places=3)

    def test_disagreement_with_the_transcript(self):
        recognition = _recognition(["h", "ə", "l", "oʊ"])
        features = scorer.recognition_features(recognition.log_posteriors, recognition.phones, recognition.confidences,
                                               ["b", "æ", "d"])
        self.assertGreater(features["transcript_phone_error_rate"], 0.9)

    def test_free_features_extend_the_read_speech_ones(self):
        self.assertEqual(scorer.FREE_FEATURES[:len(scorer.FEATURES)], scorer.FEATURES)
        self.assertEqual(len(set(scorer.FREE_FEATURES)), len(scorer.FREE_FEATURES))


class TestAssess(unittest.TestCase):

    def setUp(self):
        self.hello = _recognition(["h", "ə", "l", "oʊ", "w", "ɜː", "l", "d"])
        for target, value in (("openpronounce.free.speech.transcribe", mock.Mock(return_value="HELLO WORLD")),
                              ("openpronounce.free.phones.recognize_phones", mock.Mock(return_value=self.hello))):
            patcher = mock.patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        # the shipped word model reads hidden states, absent from the synthetic recognition
        patcher = mock.patch.object(gop, "load_model", return_value=None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_silence_gives_no_score(self):
        result = free.assess(np.zeros(32000, dtype=np.float32))
        self.assertEqual(result["mode"], "free")
        self.assertIsNone(result["score"])
        self.assertFalse(result["reliable"])
        self.assertEqual(result["segments"], [])
        self.assertEqual(result["transcribe"], "")

    def test_long_recording_is_assessed_segment_by_segment(self):
        sound = np.concatenate([_speech(12), np.zeros(16000, dtype=np.float32), _speech(12)])
        with mock.patch.object(free, "segment_features", return_value=None) as segment_features:
            result = free.assess(sound, max_segment_seconds=15)
        self.assertEqual(len(result["segments"]), 2)
        self.assertEqual(segment_features.call_count, 2)
        self.assertEqual(result["transcribe"], "hello world hello world")
        self.assertGreater(result["segments"][1]["start"], 12.0)
        self.assertAlmostEqual(result["duration"], 25.0, places=1)

    def test_scores_are_weighted_by_duration_and_words_are_offset(self):
        sound = np.concatenate([_speech(4), np.zeros(16000, dtype=np.float32), _speech(12)])
        report = {"words": [{"word": "hello", "start": 0.5, "end": 0.9, "confidence": 0.8, "mispronounced": True,
                             "expected": ["h", "ə", "l", "oʊ"], "actual": ["h", "ɛ", "l"]}]}
        scores = iter([{t: 40.0 for t in free.TARGETS}, {t: 80.0 for t in free.TARGETS}])
        with mock.patch.object(free, "segment_features", return_value=({}, report)), \
                mock.patch.object(scorer, "load_scorer", return_value=object()), \
                mock.patch.object(scorer, "score", side_effect=lambda *a, **k: next(scores)):
            result = free.assess(sound, max_segment_seconds=15)
        self.assertEqual(len(result["segments"]), 2)
        self.assertAlmostEqual(result["score"], (40 * 4.3 + 80 * 12.3) / 16.6, delta=1.0)
        self.assertEqual(result["unclear_words"], ["hello", "hello"])
        self.assertGreater(result["words"][1]["start"], 5.0)
        self.assertEqual(result["words"][0]["heard"], "hɛl")

    def test_too_little_speech_is_flagged(self):
        report = {"words": [{"word": "hi", "start": 0.1, "end": 0.3, "confidence": 0.1, "mispronounced": False,
                             "expected": ["h", "aɪ"], "actual": ["h", "aɪ"]}]}
        with mock.patch.object(free, "segment_features", return_value=({}, report)), \
                mock.patch.object(scorer, "load_scorer", return_value=object()), \
                mock.patch.object(scorer, "score", return_value={t: 70.0 for t in free.TARGETS}):
            result = free.assess(_speech(1.0))
        self.assertFalse(result["reliable"])
        self.assertEqual(len(result["reasons"]), 2)
        self.assertEqual(result["score"], 70.0)


class TestEntryPoints(unittest.TestCase):

    def test_assess_dispatches_on_the_text(self):
        with mock.patch.object(openpronounce, "compare_audio_with_text", return_value="read") as read, \
                mock.patch.object(openpronounce, "assess_free", return_value="free") as spoken:
            self.assertEqual(openpronounce.assess([0.0], "hello"), "read")
            self.assertEqual(openpronounce.assess([0.0]), "free")
            self.assertEqual(openpronounce.assess([0.0], "  "), "free")
        read.assert_called_once()
        self.assertEqual(spoken.call_count, 2)

    def test_cli_without_a_text(self):
        result = {"mode": "free", "score": 71.5, "scores": {"total": 71.5, "accuracy": 70.0, "fluency": 75.0, "prosodic": 69.0},
                  "transcribe": "hello world", "speech_duration": 2.0, "segments": [{}], "unclear_words": ["world"],
                  "reliable": False, "reasons": ["less than 3 s of speech"]}
        out = io.StringIO()
        with mock.patch("openpronounce.audio.load", return_value=np.zeros(10)), \
                mock.patch("openpronounce.free.assess", return_value=result), contextlib.redirect_stdout(out):
            self.assertEqual(cli.main(["recording.wav"]), 0)
        self.assertIn("Score        : 71.5/100", out.getvalue())
        self.assertIn("less than 3 s of speech", out.getvalue())
        self.assertIn("Unclear words: world", out.getvalue())


if __name__ == "__main__":
    unittest.main()
