"""Utterance-level scores (total, accuracy, fluency, prosody) learned from human ratings.

The features come from the forced alignment of the expected phones on the phone
model's posteriors (:func:`openpronounce.gop.utterance_features`: GOP aggregates, speech
rate, pauses, rhythm), from the word model's probabilities and from a cheap prosody
pass on the waveform (YIN pitch and RMS energy). Gradient-boosted trees trained on
speechocean762 (``models/scorer_<lang>.json``) map them to the 0-10 scales of the
raters; :func:`score` returns them on 0-100.

On the speechocean762 test split (2500 utterances, speakers disjoint from the training
ones) the English model reaches Pearson 0.77 with the human total, 0.73 with accuracy,
0.79 with fluency and 0.77 with prosody.
"""

import json
import os
from functools import lru_cache

import numpy as np

from . import gop
from .languages import DEFAULT_LANGUAGE

MODELS_DIR = gop.MODELS_DIR
TARGETS = ("total", "accuracy", "fluency", "prosodic")
HOP = 320  # 20 ms at 16 kHz, like the phone model's frames
FRAME = 1024

PROSODY_FEATURES = ["f0_mean", "f0_std_semitones", "f0_range_semitones", "f0_slope", "f0_delta_mean", "rms_mean",
                    "rms_std_norm", "rms_dyn_db", "voiced_ratio", "n_energy_dips", "duration"]
MODEL_FEATURES = ["p_phone_mean", "p_phone_max", "frac_phone_bad", "p_word_mean", "p_word_max", "frac_word_bad",
                  "n_word_bad"]
FEATURES = gop.UTTERANCE_FEATURES + MODEL_FEATURES + PROSODY_FEATURES


def prosody_features(waveform, sampling_rate=16000):
    """Pitch and energy descriptors of a waveform (YIN, no voicing decision: frames under 10 % of the peak RMS are dropped)."""
    import librosa

    waveform = np.asarray(waveform, dtype=np.float32)
    if len(waveform) < FRAME:
        waveform = np.pad(waveform, (0, FRAME - len(waveform)))
    f0 = librosa.yin(waveform, fmin=60, fmax=400, sr=sampling_rate, frame_length=FRAME, hop_length=HOP)
    rms = librosa.feature.rms(y=waveform, frame_length=FRAME, hop_length=HOP)[0]
    voiced = rms > max(rms.max() * 0.1, 1e-4)
    f0v = f0[voiced] if voiced.sum() > 3 else f0
    lf = np.log(np.maximum(f0v, 1e-3))
    semitone = 12 / np.log(2)
    return {
        "f0_mean": float(f0v.mean()),
        "f0_std_semitones": float(lf.std() * semitone),
        "f0_range_semitones": float((np.quantile(lf, 0.95) - np.quantile(lf, 0.05)) * semitone),
        "f0_slope": float(np.polyfit(np.arange(len(lf)), lf, 1)[0] * 100) if len(lf) > 2 else 0.0,
        "f0_delta_mean": float(np.abs(np.diff(lf)).mean()) if len(lf) > 1 else 0.0,
        "rms_mean": float(rms.mean()),
        "rms_std_norm": float(rms.std() / (rms.mean() + 1e-9)),
        "rms_dyn_db": float(20 * np.log10((np.quantile(rms, 0.95) + 1e-9) / (np.quantile(rms, 0.2) + 1e-9))),
        "voiced_ratio": float(voiced.mean()),
        "n_energy_dips": int((voiced[1:] != voiced[:-1]).sum() // 2),
        "duration": len(waveform) / sampling_rate,
    }


def model_features(phone_probabilities, word_probabilities, threshold):
    """Aggregates of the word model's output (see :meth:`openpronounce.gop.WordModel.predict`)."""
    phone_probabilities = np.asarray(phone_probabilities, dtype=float)
    words = np.array(list(word_probabilities.values()), dtype=float)
    return {
        "p_phone_mean": float(phone_probabilities.mean()), "p_phone_max": float(phone_probabilities.max()),
        "frac_phone_bad": float(np.mean(phone_probabilities >= 0.5)),
        "p_word_mean": float(words.mean()), "p_word_max": float(words.max()),
        "frac_word_bad": float(np.mean(words >= threshold)), "n_word_bad": int(np.sum(words >= threshold)),
    }


def features(analysis, phone_probabilities, word_probabilities, threshold, waveform, sampling_rate=16000):
    """Every scorer feature of an utterance, as a dict keyed like :data:`FEATURES`."""
    return {**gop.utterance_features(analysis), **model_features(phone_probabilities, word_probabilities, threshold),
            **prosody_features(waveform, sampling_rate)}


class Scorer:
    """One regression model per target, exported as JSON (see ``benchmarks/scorer.py --fit``)."""

    def __init__(self, data):
        self.data = data
        self.features = data["features"]
        self.models = {target: gop.MODEL_TYPES[model["type"]](model) for target, model in data["models"].items()}
        self.calibration = data.get("calibration", {})
        self.sentence_match = gop.LogisticModel(data["sentence_match"]) if "sentence_match" in data else None

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False)

    def predict(self, feature_dict):
        """Scores on the raters' 0-10 scale, one per target."""
        row = np.array([[feature_dict[name] for name in self.features]], dtype=np.float64)
        return {target: float(values[0]) for target, values in self.predict_matrix(row).items()}

    def predict_matrix(self, matrix):
        """Same for a ``(n, len(features))`` matrix: one clipped array per target."""
        out = {}
        for target, model in self.models.items():
            raw = model.predict(matrix)
            slope, intercept = self.calibration.get(target, (1.0, 0.0))
            out[target] = np.clip(slope * raw + intercept, 0.0, 10.0)
        return out

    def match_probability(self, matrix):
        """Probability that the recording is a reading of the expected sentence (``None`` without a match model)."""
        if self.sentence_match is None:
            return None
        columns = [self.features.index(name) for name in self.sentence_match.features]
        return self.sentence_match.predict(np.asarray(matrix, dtype=np.float64)[:, columns])


@lru_cache(maxsize=8)
def load_scorer(lang=DEFAULT_LANGUAGE):
    """The :class:`Scorer` of ``lang`` or ``None``."""
    path = os.path.join(MODELS_DIR, f"scorer_{lang}.json")
    return Scorer.load(path) if os.path.exists(path) else None


def score(feature_dict, lang=DEFAULT_LANGUAGE):
    """0-100 scores (``total``, ``accuracy``, ``fluency``, ``prosodic``) of an utterance, or ``None`` without a model.

    The raters' 0-10 scale is multiplied by 10. ``sentence_probability`` (0-1) says how
    likely the recording is a reading of the expected sentence at all, when the model has it.
    """
    scorer = load_scorer(lang)
    if scorer is None:
        return None
    scores = {target: round(10.0 * value, 1) for target, value in scorer.predict(feature_dict).items()}
    if scorer.sentence_match is not None:
        row = np.array([[feature_dict[name] for name in scorer.features]], dtype=np.float64)
        scores["sentence_probability"] = round(float(scorer.match_probability(row)[0]), 3)
    return scores
