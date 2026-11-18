import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import soundfile as sf

from openpronounce import audio, tts


def sine(sr, seconds=0.5, freq=440.0):
    t = np.arange(int(sr * seconds)) / sr
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class TestLoad(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "tone.wav")
        sf.write(self.path, np.stack([sine(44100), sine(44100)], axis=1), 44100)

    def tearDown(self):
        self.tmp.cleanup()

    def test_load_returns_mono_16k_float32(self):
        waveform = audio.load(self.path)
        self.assertEqual(waveform.ndim, 1)
        self.assertEqual(waveform.dtype, np.float32)
        self.assertAlmostEqual(len(waveform) / audio.TARGET_SR, 0.5, places=2)

    def test_webm2wav_writes_16k_wav_next_to_input(self):
        out = audio.webm2wav(self.path)
        self.assertEqual(out, os.path.join(self.tmp.name, "tone.16k.wav"))
        waveform, sr = sf.read(out)
        self.assertEqual(sr, audio.TARGET_SR)
        self.assertEqual(waveform.ndim, 1)

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg not installed")
    def test_load_browser_webm_opus_through_ffmpeg(self):
        webm = os.path.join(self.tmp.name, "rec.webm")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", self.path, "-c:a", "libopus", webm], check=True)
        waveform = audio.load(webm)
        self.assertEqual(waveform.dtype, np.float32)
        self.assertAlmostEqual(len(waveform) / audio.TARGET_SR, 0.5, places=1)
        out = audio.webm2wav(webm)
        self.assertTrue(out.endswith("rec.16k.wav"))

    def test_webm2wav_unreadable_file(self):
        bad = os.path.join(self.tmp.name, "bad.webm")
        with open(bad, "wb") as f:
            f.write(b"not audio")
        with self.assertRaises(RuntimeError):
            audio.webm2wav(bad)


class TestBackendSelection(unittest.TestCase):

    def test_default_is_gtts(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(tts.resolve("en"), ("gtts", "com"))

    def test_env_var(self):
        with patch.dict(os.environ, {"OPENPRONOUNCE_TTS": "piper"}):
            self.assertEqual(tts.resolve("en"), ("piper", "en_US-lessac-medium"))
        with patch.dict(os.environ, {"OPENPRONOUNCE_TTS": "kokoro", "OPENPRONOUNCE_TTS_VOICE": "af_bella"}):
            self.assertEqual(tts.resolve("en"), ("kokoro", "af_bella"))

    def test_kwargs_override_env(self):
        with patch.dict(os.environ, {"OPENPRONOUNCE_TTS": "piper", "OPENPRONOUNCE_TTS_VOICE": "x"}):
            self.assertEqual(tts.resolve("fr", backend="kokoro", voice="ff_siwis"), ("kokoro", "ff_siwis"))

    def test_default_voice_per_language(self):
        self.assertEqual(tts.resolve("fr", backend="piper"), ("piper", "fr_FR-siwis-medium"))
        self.assertEqual(tts.resolve("fr", backend="kokoro"), ("kokoro", "ff_siwis"))

    def test_unknown_backend(self):
        with self.assertRaises(ValueError):
            tts.resolve("en", backend="espeak")
        with patch.dict(os.environ, {"OPENPRONOUNCE_TTS": "nope"}):
            with self.assertRaises(ValueError):
                audio.text2speech("hello")

    def test_language_without_default_voice(self):
        with self.assertRaisesRegex(ValueError, "OPENPRONOUNCE_TTS_VOICE"):
            tts.resolve("xx", backend="piper")
        with self.assertRaisesRegex(ValueError, "Kokoro does not support"):
            tts.resolve("de", backend="kokoro")

    def test_missing_optional_dependency(self):
        with patch.dict(sys.modules, {"piper": None}):
            with self.assertRaisesRegex(ImportError, r"openpronounce\[tts-piper\]"):
                tts.synthesize_piper("hello", "en", "en_US-lessac-medium")
        with patch.dict(sys.modules, {"kokoro": None}):
            with self.assertRaisesRegex(ImportError, r"openpronounce\[tts-kokoro\]"):
                tts.synthesize_kokoro("hello", "en", "af_heart")


class TestText2Speech(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = patch.object(audio, "CACHE_DIR", self.tmp.name)
        self.cache.start()
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.cache.stop()
        self.tmp.cleanup()

    def assert_16k_mono_wav(self, path):
        self.assertTrue(path.startswith(self.tmp.name))
        waveform, sr = sf.read(path)
        self.assertEqual(sr, audio.TARGET_SR)
        self.assertEqual(waveform.ndim, 1)
        self.assertGreater(len(waveform), 0)
        return waveform

    def test_gtts_backend(self):
        def fake_save(self, path):
            sf.write(path, sine(24000), 24000, format="MP3")

        with patch("gtts.gTTS.save", fake_save):
            path = audio.text2speech("hello")
        self.assert_16k_mono_wav(path)

    def test_piper_backend(self):
        synth = MagicMock(return_value=(sine(22050), 22050))
        with patch.dict(tts.BACKENDS, {"piper": (synth, tts.piper_default_voice)}):
            path = audio.text2speech("hello", backend="piper")
        synth.assert_called_once_with("hello", "en", "en_US-lessac-medium")
        waveform = self.assert_16k_mono_wav(path)
        self.assertAlmostEqual(len(waveform) / audio.TARGET_SR, 0.5, places=2)

    def test_kokoro_backend(self):
        synth = MagicMock(return_value=(sine(24000), 24000))
        with patch.dict(tts.BACKENDS, {"kokoro": (synth, tts.kokoro_default_voice)}):
            with patch.dict(os.environ, {"OPENPRONOUNCE_TTS": "kokoro"}):
                path = audio.text2speech("bonjour", lang="fr")
        synth.assert_called_once_with("bonjour", "fr", "ff_siwis")
        self.assert_16k_mono_wav(path)

    def test_cache_key_includes_backend_and_voice(self):
        with patch.object(tts, "synthesize", return_value=(sine(16000), 16000)) as synth:
            a = audio.text2speech("hello", backend="piper")
            b = audio.text2speech("hello", backend="kokoro")
            c = audio.text2speech("hello", backend="piper", voice="en_US-amy-medium")
            again = audio.text2speech("hello", backend="piper")
        self.assertEqual(len({a, b, c}), 3)
        self.assertEqual(again, a)
        self.assertEqual(synth.call_count, 3)

    def test_explicit_filename(self):
        target = os.path.join(self.tmp.name, "ref.wav")
        with patch.object(tts, "synthesize", return_value=(sine(16000), 16000)):
            self.assertEqual(audio.text2speech("hello", filename=target), target)
        self.assert_16k_mono_wav(target)


if __name__ == "__main__":
    unittest.main()


class TestSplitSpeech(unittest.TestCase):

    @staticmethod
    def _burst(seconds, sr=16000):
        t = np.arange(int(seconds * sr)) / sr
        return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    def _recording(self, pattern, sr=16000):
        """``pattern``: seconds of speech (positive) and silence (negative)."""
        return np.concatenate([self._burst(s, sr) if s > 0 else np.zeros(int(-s * sr), dtype=np.float32) for s in pattern])

    def test_short_recording_is_one_span_without_the_silence(self):
        spans = audio.split_speech(self._recording([-1, 2, -1]))
        self.assertEqual(len(spans), 1)
        start, end = spans[0]
        self.assertAlmostEqual(start / 16000, 0.85, delta=0.1)
        self.assertAlmostEqual(end / 16000, 3.15, delta=0.1)

    def test_recording_is_cut_at_every_long_pause(self):
        spans = audio.split_speech(self._recording([3, -1, 4, -0.2, 2, -1, 3]))
        self.assertEqual(len(spans), 3)
        self.assertAlmostEqual((spans[1][1] - spans[1][0]) / 16000, 6.5, delta=0.2)
        for (_, end), (start, _) in zip(spans, spans[1:]):
            self.assertLessEqual(end, start)

    def test_long_span_is_cut_at_its_longest_pause(self):
        spans = audio.split_speech(self._recording([6, -0.2, 5, -0.4, 6]), max_seconds=12)
        self.assertEqual(len(spans), 2)
        self.assertAlmostEqual(spans[0][1] / 16000, 11.4, delta=0.3)
        self.assertTrue(all((end - start) / 16000 <= 12.4 for start, end in spans))

    def test_short_span_joins_its_nearest_neighbour(self):
        spans = audio.split_speech(self._recording([3, -2, 0.5, -0.6, 3]))
        self.assertEqual(len(spans), 2)
        self.assertAlmostEqual(spans[1][0] / 16000, 4.85, delta=0.1)

    def test_speech_without_a_pause_is_cut_in_equal_parts(self):
        spans = audio.split_speech(self._recording([50]), max_seconds=20)
        self.assertEqual(len(spans), 3)
        self.assertTrue(all((end - start) / 16000 <= 20.4 for start, end in spans))

    def test_silence_and_empty(self):
        self.assertEqual(audio.split_speech(np.zeros(16000, dtype=np.float32)), [])
        self.assertEqual(audio.split_speech(np.zeros(0, dtype=np.float32)), [])


class TestQuality(unittest.TestCase):

    @staticmethod
    def _voice(seconds=3.0, amplitude=0.3, noise=0.0, sr=16000):
        t = np.arange(int(seconds * sr)) / sr
        envelope = (np.sin(2 * np.pi * 1.5 * t) > 0).astype(np.float32)  # speech bursts and pauses
        rng = np.random.default_rng(0)
        return (amplitude * envelope * np.sin(2 * np.pi * 200 * t) + noise * rng.standard_normal(len(t))).astype(np.float32)

    def test_clean_recording(self):
        report = audio.quality(self._voice(noise=0.001))
        self.assertEqual(report["issues"], [])
        self.assertAlmostEqual(report["duration"], 3.0, places=2)
        self.assertGreater(report["snr_db"], 30)

    def test_silent_and_too_short(self):
        self.assertEqual(audio.quality(np.zeros(1600, dtype=np.float32))["issues"], ["too_short", "silent"])
        self.assertEqual(audio.quality(np.zeros(0, dtype=np.float32))["issues"], ["too_short", "silent"])

    def test_too_quiet(self):
        self.assertIn("too_quiet", audio.quality(self._voice(amplitude=0.0002))["issues"])

    def test_clipped(self):
        report = audio.quality(np.clip(self._voice(amplitude=3.0), -1, 1))
        self.assertIn("clipped", report["issues"])
        self.assertGreater(report["clipping_ratio"], 0.01)

    def test_noisy(self):
        self.assertIn("noisy", audio.quality(self._voice(amplitude=0.1, noise=0.08))["issues"])
