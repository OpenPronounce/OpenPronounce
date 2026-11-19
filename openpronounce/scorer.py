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
# Free speech (no expected text): what the phone recognizer says on its own, and how far it
# is from the phones of the automatic transcript.
RECOGNITION_FEATURES = ["post_top_mean", "post_top_q10", "post_entropy_mean", "blank_ratio", "phone_confidence_mean",
                        "phone_confidence_q10", "transcript_phone_error_rate", "heard_phones_per_second"]
FREE_FEATURES = FEATURES + RECOGNITION_FEATURES
# What depends on how well the recognizer and espeak agree in a language, more than on the speaker:
# the features a generic scorer reads after :func:`native_shift`. Speech rate, pauses and pitch stay as they are.
ALIGNMENT_FEATURES = [name for name in gop.UTTERANCE_FEATURES
                      if name.split("_")[0] in ("gm", "del", "af", "lp", "frac", "ll", "heard", "greedy", "plausible")]
#: Key of the pooled hidden states in a feature dict. Not a number: a vector, read by the
#: scorers that have an embedding head and ignored by the others.
EMBEDDING = "embedding"
SHIFTED_FEATURES = {"scorer": ALIGNMENT_FEATURES,
                    "free_scorer": ALIGNMENT_FEATURES + [name for name in RECOGNITION_FEATURES if name != "heard_phones_per_second"]}


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


def recognition_features(log_posteriors, heard_phones, heard_confidences, expected_phones, blank_id=0):
    """Text-independent descriptors of a recording, plus its distance to the transcript's phones.

    ``log_posteriors`` are the phone recognizer's frames, ``heard_phones`` / ``heard_confidences``
    its greedy decoding, ``expected_phones`` the flat phones of the automatic transcript.
    """
    from rapidfuzz.distance import Levenshtein

    posteriors = np.exp(np.asarray(log_posteriors, dtype=np.float64))
    best = posteriors.argmax(axis=1)
    speech = best != blank_id
    frames = posteriors[speech] if speech.any() else posteriors
    top = frames.max(axis=1)
    entropy = -(frames * np.log(np.maximum(frames, 1e-12))).sum(axis=1)
    confidences = np.asarray(heard_confidences, dtype=float) if len(heard_confidences) else np.zeros(1)
    expected_phones, heard_phones = list(expected_phones), list(heard_phones)
    return {
        "post_top_mean": float(top.mean()), "post_top_q10": float(np.quantile(top, 0.1)),
        "post_entropy_mean": float(entropy.mean()), "blank_ratio": float(1.0 - speech.mean()),
        "phone_confidence_mean": float(confidences.mean()), "phone_confidence_q10": float(np.quantile(confidences, 0.1)),
        "transcript_phone_error_rate": Levenshtein.distance(expected_phones, heard_phones) / max(1, len(expected_phones)),
        "heard_phones_per_second": len(heard_phones) / max(1e-3, len(posteriors) * gop.FRAME_SECONDS),
    }


def pooled_embedding(hidden):
    """One vector for an utterance: mean and standard deviation over its frames of the recognizer's hidden states."""
    hidden = np.asarray(hidden, dtype=np.float32)
    return np.concatenate([hidden.mean(axis=0), hidden.std(axis=0)])


class EmbeddingHead:
    """Linear model on :func:`pooled_embedding`, blended with the trees of a :class:`Scorer`.

    The trees read some sixty measures of the alignment, the rhythm and the pitch. The hidden
    states of the recognizer keep what those measures drop. ``share`` is the weight of this head
    in the blend, ``calibration`` what maps the blend back onto the rating scale.
    """

    def __init__(self, data):
        self.layer = data["layer"]
        self.share = float(data["share"])
        self.center = np.asarray(data["center"], dtype=np.float64)
        self.scale = np.asarray(data["scale"], dtype=np.float64)
        self.weights = {target: np.asarray(w, dtype=np.float64) for target, w in data["weights"].items()}
        self.intercept = data["intercept"]
        self.calibration = data.get("calibration", {})

    def predict(self, embeddings, target):
        return ((np.asarray(embeddings, dtype=np.float64) - self.center) / self.scale) @ self.weights[target] + self.intercept[target]


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
        self.embedding = EmbeddingHead(data["embedding"]) if "embedding" in data else None
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
        embedding = feature_dict.get(EMBEDDING) if self.embedding is not None else None
        embeddings = None if embedding is None else np.asarray(embedding, dtype=np.float64)[None, :]
        return {target: float(values[0]) for target, values in self.predict_matrix(row, embeddings).items()}

    def predict_matrix(self, matrix, embeddings=None):
        """Same for a ``(n, len(features))`` matrix: one clipped array per target.

        ``embeddings`` (``(n, dim)``, see :func:`pooled_embedding`) are blended in when the
        scorer has an embedding head; without them the trees answer alone.
        """
        head = self.embedding if embeddings is not None else None
        out = {}
        for target, model in self.models.items():
            raw = model.predict(matrix)
            slope, intercept = self.calibration.get(target, (1.0, 0.0))
            if head is not None:
                raw = (1.0 - head.share) * raw + head.share * head.predict(embeddings, target)
                slope, intercept = head.calibration.get(target, (1.0, 0.0))
            out[target] = np.clip(slope * raw + intercept, 0.0, 10.0)
        return out

    def match_probability(self, matrix):
        """Probability that the recording is a reading of the expected sentence (``None`` without a match model)."""
        if self.sentence_match is None:
            return None
        columns = [self.features.index(name) for name in self.sentence_match.features]
        return self.sentence_match.predict(np.asarray(matrix, dtype=np.float64)[:, columns])


@lru_cache(maxsize=8)
def load_scorer(lang=DEFAULT_LANGUAGE, kind="scorer"):
    """The :class:`Scorer` of ``lang`` or ``None``. ``kind="free_scorer"`` is the one for speech without an expected text.

    A language without its own scorer gets the generic one of that kind, trained on English
    learners with the features of the generic word model. A phone recognizer that declares a
    suffix gets the scorer trained on its features first (:func:`openpronounce.gop.model_names`).
    """
    path = gop.find_model(kind, lang)
    return Scorer.load(path) if path is not None else None


@lru_cache(maxsize=16)
def native_shift(lang, kind="scorer"):
    """Offsets added to the features of ``lang`` before the generic scorer of ``kind`` reads them.

    The generic scorers learned on learners of English what a good speaker looks like to the
    phone recognizer. The recognizer is less at ease with a native of Dutch than with a
    native of English, and the phones espeak writes do not sit as well on what it hears:
    ``models/native_<lang>.json`` holds, feature by feature, the distance between the
    native speakers of ``lang`` and those of English (``benchmarks/native_calibration.py``).
    """
    path = gop.model_path(f"native_{lang}.json")
    if path is None:
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("shift", {}).get(kind, {})


def native_offsets(rows, kind="scorer"):
    """The shift of a language from the features of its native speakers: ``{feature: offset}``.

    ``rows`` are feature dicts (:func:`features`, or the segments of free speech) of native
    recordings scored with the generic models; the reference is the mean the generic scorer
    of ``kind`` keeps for the native speakers of English (``native_reference``).
    """
    reference = load_scorer("generic", kind).data.get("native_reference")
    if reference is None:
        raise ValueError(f"the generic {kind} has no native reference: run benchmarks/native_calibration.py --reference")
    return {name: round(float(reference[name] - np.mean([row[name] for row in rows])), 5) for name in SHIFTED_FEATURES[kind]}


def score(feature_dict, lang=DEFAULT_LANGUAGE, kind="scorer"):
    """0-100 scores (``total``, ``accuracy``, ``fluency``, ``prosodic``) of an utterance, or ``None`` without a model.

    The raters' 0-10 scale is multiplied by 10. ``sentence_probability`` (0-1) says how
    likely the recording is a reading of the expected sentence at all, when the model has it.
    With a generic scorer, the features are first moved by :func:`native_shift`. A scorer with an
    embedding head also reads ``feature_dict["embedding"]`` (:func:`pooled_embedding`) when it is there.
    """
    scorer = load_scorer(lang, kind)
    if scorer is None:
        return None
    if scorer.data.get("lang") == "generic":
        shift = native_shift(lang, kind)
        if shift:
            feature_dict = {**feature_dict, **{name: feature_dict[name] + delta for name, delta in shift.items() if name in feature_dict}}
    if scorer.embedding is not None and EMBEDDING in feature_dict and scorer.embedding.layer != gop.hidden_layer(lang):
        # hidden states of another layer than the one the head learned from: the trees answer alone
        feature_dict = {name: value for name, value in feature_dict.items() if name != EMBEDDING}
    scores = {target: round(10.0 * value, 1) for target, value in scorer.predict(feature_dict).items()}
    if scorer.sentence_match is not None:
        row = np.array([[feature_dict[name] for name in scorer.features]], dtype=np.float64)
        scores["sentence_probability"] = round(float(scorer.match_probability(row)[0]), 3)
    return scores
