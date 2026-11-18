"""Goodness of pronunciation (GOP) of every expected phone, from forced alignment.

Instead of decoding what the recognizer heard and comparing strings, the expected
phones of the sentence are force-aligned on the frame posteriors of the phone model
(:mod:`openpronounce.alignment`). Every expected phone then gets a frame span and a
set of continuous scores: how far its posterior is from the best token on its frames
(the classic GOP margin), how much the whole sentence likelihood drops when the phone
is deleted or replaced by a close one (alignment-free GOP), its duration, its context.

A small logistic model trained on speechocean762 (``models/word_mdd_<lang>.json``)
turns these scores into the probability that a phone, then a word, is mispronounced.
Without a model for the language, the probabilities fall back on the generic model.
"""

import json
import logging
import math
import os
from functools import lru_cache

import numpy as np

from . import alignment
from .languages import DEFAULT_LANGUAGE, get_language

logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODELS_ENV = "OPENPRONOUNCE_MODELS_DIR"
PACKAGE_LICENSE = "AGPL-3.0-or-later"
MODEL_KINDS = {"free_scorer": "free speech scorer", "scorer": "utterance scorer", "word_mdd": "word model",
               "word_missing": "missing word model", "native": "native speakers of the language against those of English"}


def models_dirs():
    """Directories searched for model files, in order.

    ``$OPENPRONOUNCE_MODELS_DIR`` (one directory, or several separated by ``os.pathsep``)
    comes before the models shipped in the package, so a directory holding
    ``scorer_<lang>.json`` / ``word_mdd_<lang>.json`` adds a language or replaces a
    shipped model without touching the installation.
    """
    extra = [d for d in os.environ.get(MODELS_ENV, "").split(os.pathsep) if d]
    return extra + [MODELS_DIR]


def model_path(name):
    """Path of the first ``name`` found in :func:`models_dirs`, or ``None``."""
    for directory in models_dirs():
        path = os.path.join(directory, name)
        if os.path.exists(path):
            return path
    return None


def installed_models():
    """One dict per model file that would be loaded: ``kind``, ``lang``, ``path``, ``bundled``, ``license``, ``trained_on``, ``metrics``.

    A file shadowed by an earlier directory is not listed. Shipped models follow the licence of the package; a
    model from another directory says so in its ``license`` key, or is ``unspecified``.
    """
    found = {}
    for directory in models_dirs():
        if not os.path.isdir(directory):
            continue
        for name in sorted(os.listdir(directory)):
            kind = next((k for k in MODEL_KINDS if name.startswith(k + "_") and name.endswith(".json")), None)
            if kind is None or name in found:
                continue
            path = os.path.join(directory, name)
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            bundled = os.path.abspath(directory) == os.path.abspath(MODELS_DIR)
            found[name] = {
                "kind": kind, "lang": name[len(kind) + 1:-len(".json")], "path": path, "bundled": bundled,
                "license": data.get("license", PACKAGE_LICENSE if bundled else "unspecified"),
                "trained_on": data.get("trained_on"), "metrics": data.get("metrics", {}),
            }
    return sorted(found.values(), key=lambda m: (m["lang"], m["kind"]))
# Seconds per frame of the wav2vec2 phone model (320 samples at 16 kHz).
FRAME_SECONDS = 0.02
# An expected phone whose log posterior stays below this on all its frames is implausible
# (used by the reliability estimate, not by the classifier).
PLAUSIBLE_LOG_POSTERIOR = -1.0
# A token other than the expected phone is "heard" on the phone's frames when its
# posterior reaches this.
HEARD_LOG_POSTERIOR = math.log(0.1)
# The deletion / substitution likelihoods of a phone are computed on this many phones on
# each side and this many frames beyond their spans.
PERTURBATION_CONTEXT = 2
PERTURBATION_MARGIN = 5
# A phone's embedding is the mean hidden state over its aligned span widened by this many frames.
EMBEDDING_MARGIN = 1

PHONE_FEATURES = [
    "gop_margin", "gop_margin_min", "gop_dnn", "lp_target", "ctx_lp_target", "gop_del", "gop_af", "dur",
    "n_tokens", "pos_in_word", "is_first", "is_last", "n_phones", "function_word", "prev_gm", "next_gm",
    "utt_gm_mean", "utt_gm_std", "ll_full_per_phone", "heard_ratio", "greedy_ll_per_frame",
]
_AGGREGATED = ("p", "gm", "del", "af", "lp", "dur")
WORD_FEATURES = sorted(
    [f"{name}_{stat}" for name in _AGGREGATED for stat in ("min", "max", "mean")] + ["p_any"]
) + ["n_phones", "function_word", "utt_gm_mean", "utt_gm_std", "heard_ratio", "ll_full_per_phone"]

FUNCTION_WORDS = {
    "en": {"a", "an", "the", "of", "to", "in", "on", "at", "for", "and", "or", "is", "are", "was", "it", "i",
           "you", "he", "she", "we", "they", "that", "this", "my", "your", "his", "her", "its", "our", "their",
           "do", "does", "did", "have", "has", "had", "be", "am", "not", "no", "so", "if", "as", "by", "with",
           "from", "but", "me", "him", "us", "them"},
}


# ---------------------------------------------------------------------------
# Expected phones -> recognizer tokens
# ---------------------------------------------------------------------------

def _token_groups(vocab, lang):
    from .phones import _token_ids_by_phone

    return _token_ids_by_phone(vocab, lang)


_VOWEL_LETTERS = set("aeiouyæɑɒɔəɛɜɪʊʌøœɐɨɵʉɯɤ")


def _is_vowel_token(token):
    return token[:1] in _VOWEL_LETTERS


@lru_cache(maxsize=4096)
def _label_ids(phone, vocab, lang):
    """Token ids standing for ``phone``: one group per label, usually a single group.

    A phone spelled like a recognizer token (with or without its length mark) maps to
    that token alone, so that the GOP keeps telling apart phones that
    :func:`openpronounce.phones.normalize_phone` merges for display (ɔ/ɑ, ɾ/t). A
    normalized phone maps to every token that normalizes to it; anything else (a merged
    diphthong, a cluster) is split greedily into the longest known tokens, each becoming
    its own label.
    """
    index = {token: i for i, token in enumerate(vocab)}
    # What native voices are heard to say for this phone of espeak, and the short
    # consonant for a long one (geminates are rarely recognized as such).
    accepted = [index[token] for token in get_language(lang).accepted(phone) if token in index]
    short = phone.replace("ː", "")
    if phone.endswith("ː") and short in index and not _is_vowel_token(short):
        accepted.append(index[short])
    for spelling in (phone, short):
        if spelling in index and not spelling.startswith("<"):
            return (tuple(dict.fromkeys([index[spelling]] + accepted)),)
    if accepted:
        return (tuple(dict.fromkeys(accepted)),)
    groups = _token_groups(vocab, lang)
    from .phones import normalize_phone

    normalized = normalize_phone(phone, lang)
    if normalized in groups:
        return (tuple(groups[normalized]),)

    longest = max(len(token) for token in vocab)
    labels, i = [], 0
    while i < len(phone):
        for length in range(min(longest, len(phone) - i), 0, -1):
            piece = normalize_phone(phone[i:i + length], lang)
            if piece in groups:
                labels.append(tuple(groups[piece]))
                i += length
                break
        else:
            i += 1
    return tuple(labels)


def _reduce(log_posteriors, columns):
    """Log posteriors of the label columns: blank first, then log-sum-exp over each group of token ids."""
    reduced = np.empty((log_posteriors.shape[0], len(columns) + 1), dtype=np.float32)
    reduced[:, 0] = log_posteriors[:, 0]
    for j, ids in enumerate(columns, 1):
        if len(ids) == 1:
            reduced[:, j] = log_posteriors[:, ids[0]]
        else:
            block = log_posteriors[:, list(ids)]
            m = block.max(axis=1)
            reduced[:, j] = m + np.log(np.exp(block - m[:, None]).sum(axis=1))
    return reduced


class _Labels:
    """Expected phones of an utterance as CTC labels over a reduced posterior matrix."""

    def __init__(self, log_posteriors, vocab, groups, lang):
        self.lang = lang
        self.vocab = vocab
        self.phones = []  # (word_index, position, phone, label_indices)
        columns = {}
        sequence = []
        for word_index, group in enumerate(groups):
            for position, phone in enumerate(group):
                indices = []
                for ids in _label_ids(phone, vocab, lang):
                    column = columns.setdefault(ids, len(columns) + 1)
                    indices.append(len(sequence))
                    sequence.append(column)
                if indices:
                    self.phones.append((word_index, position, phone, indices))
        self.columns = sorted(columns, key=columns.get)
        self.sequence = sequence
        self.reduced = _reduce(log_posteriors, self.columns)
        self.token_ids = [set(ids) for ids in self.columns]

    def column_of(self, phone):
        ids = _label_ids(phone, self.vocab, self.lang)
        return [self.columns.index(group) + 1 if group in self.columns else None for group in ids]


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def _near_phones(lang):
    from .phones import _NEAR_PHONES

    near = {}
    for group in _NEAR_PHONES.get(lang, []):
        for phone in group:
            near.setdefault(phone, set()).update(group - {phone})
    return near


def analyze(log_posteriors, vocab, groups, words=None, lang=DEFAULT_LANGUAGE, hidden=None):
    """Force-align the expected phones (``groups``, normalized, one list per word) and score each of them.

    Returns ``None`` when the recording is too short for the phones. Otherwise a dict with
    ``phones`` (one dict per expected phone: ``word_index``, ``position``, ``phone``,
    ``start``, ``end`` (frames), ``heard`` (the token id heard instead, or ``None``) and
    every :data:`PHONE_FEATURES` entry) and ``utterance`` (``log_likelihood_per_phone``,
    ``plausible_phone_ratio``, ``heard_ratio``, ``n_frames``). With ``hidden`` (the
    ``(frames, dim)`` hidden states of the recognizer) every phone also gets ``embedding``,
    the mean hidden state over its span (see :data:`EMBEDDING_MARGIN`).
    """
    log_posteriors = np.asarray(log_posteriors, dtype=np.float32)
    labels = _Labels(log_posteriors, vocab, groups, lang)
    if not labels.phones or not alignment.fits(log_posteriors.shape[0], labels.sequence):
        return None
    reduced, sequence = labels.reduced, labels.sequence
    n_frames = log_posteriors.shape[0]
    spans, _ = alignment.forced_align(reduced, sequence)
    ll_full = alignment.log_likelihood(reduced, sequence)
    frame_max = log_posteriors.max(axis=1)
    greedy = log_posteriors.argmax(axis=1)
    n_heard = int(((greedy[1:] != greedy[:-1]) & (greedy[1:] != 0)).sum() + (greedy[0] != 0))
    heard_ratio = n_heard / len(labels.phones)
    near = _near_phones(lang)
    function_words = FUNCTION_WORDS.get(lang, set())
    n_phones = [len(g) for g in groups]

    scored = []
    for j, (word_index, position, phone, indices) in enumerate(labels.phones):
        own = [spans[i] for i in indices]
        start = min(s for s, _ in own)
        end = max(e for _, e in own)
        if end <= start:
            start = max(0, min(start, n_frames - 1))
            end = start + 1
        margins, targets, dnns, contexts = [], [], [], []
        for i, (s, e) in zip(indices, own):
            if e <= s:
                s = max(0, min(s, n_frames - 1))
                e = s + 1
            column = reduced[s:e, sequence[i]]
            margins.append(float((column - frame_max[s:e]).mean()))
            targets.append(float(column.mean()))
            dnns.append(float(-np.log(np.exp(column).mean() + 1e-12)))
            contexts.append(float(reduced[max(0, s - 2):min(n_frames, e + 2), sequence[i]].max()))
        # Likelihood drop when the phone is deleted or replaced by a close one, computed on
        # a window of neighbouring phones and frames: far frames align the same way in
        # every variant and would only add cost.
        first = labels.phones[max(0, j - PERTURBATION_CONTEXT)][3][0]
        last = labels.phones[min(len(labels.phones) - 1, j + PERTURBATION_CONTEXT)][3][-1] + 1
        frame_start = max(0, min(spans[first][0], start) - PERTURBATION_MARGIN)
        frame_end = min(n_frames, max(spans[last - 1][1], end) + PERTURBATION_MARGIN)
        window = reduced[frame_start:frame_end]
        before, after = sequence[first:indices[0]], sequence[indices[-1] + 1:last]
        candidates = [before + after]
        for alternative in near.get(phone, ()):
            columns = labels.column_of(alternative)
            if columns and None not in columns:
                candidates.append(before + columns + after)
        reference = alignment.log_likelihood(window, sequence[first:last])
        likelihoods = [alignment.log_likelihood(window, c) - reference for c in candidates]
        own_ids = set().union(*(labels.token_ids[sequence[i] - 1] for i in indices))
        competitors = log_posteriors[start:end].copy()
        competitors[:, list(own_ids)] = NEG_INF
        competitors[:, 0] = NEG_INF
        other = int(competitors.max(axis=0).argmax())
        heard = other if competitors[:, other].max() >= HEARD_LOG_POSTERIOR else None
        scored.append({
            "word_index": word_index, "position": position, "phone": phone, "start": start, "end": end,
            "heard": heard,
            "gop_margin": float(np.mean(margins)), "gop_margin_min": float(np.min(margins)),
            "gop_dnn": float(np.mean(dnns)), "lp_target": float(np.mean(targets)),
            "ctx_lp_target": float(np.mean(contexts)), "gop_del": -likelihoods[0],
            "gop_af": -max(likelihoods), "dur": end - start, "n_tokens": len(indices),
            "pos_in_word": position, "is_first": int(position == 0),
            "is_last": int(position == n_phones[word_index] - 1), "n_phones": n_phones[word_index],
            "function_word": int(words is not None and words[word_index] in function_words),
        })
        if hidden is not None:
            scored[-1]["embedding"] = np.asarray(
                hidden[max(0, start - EMBEDDING_MARGIN):min(n_frames, end + EMBEDDING_MARGIN)], dtype=np.float32
            ).mean(axis=0)
    margins = np.array([p["gop_margin"] for p in scored])
    utterance = {
        "n_frames": n_frames,
        "ll_full_per_phone": ll_full / len(sequence),
        "heard_ratio": heard_ratio,
        "greedy_ll_per_frame": float(frame_max.mean()),
        "plausible_phone_ratio": float(np.mean([p["ctx_lp_target"] >= PLAUSIBLE_LOG_POSTERIOR for p in scored])),
    }
    for j, p in enumerate(scored):
        p.update({
            "prev_gm": scored[j - 1]["gop_margin"] if j else 0.0,
            "next_gm": scored[j + 1]["gop_margin"] if j + 1 < len(scored) else 0.0,
            "utt_gm_mean": float(margins.mean()), "utt_gm_std": float(margins.std()),
            "ll_full_per_phone": utterance["ll_full_per_phone"], "heard_ratio": heard_ratio,
            "greedy_ll_per_frame": utterance["greedy_ll_per_frame"],
        })
    return {"phones": scored, "utterance": utterance}


NEG_INF = -1e9


def phone_matrix(scored):
    """``(n_phones, len(PHONE_FEATURES))`` feature matrix of the phones returned by :func:`analyze`."""
    return np.array([[p[name] for name in PHONE_FEATURES] for p in scored], dtype=np.float64)


def embedding_matrix(scored):
    """``(n_phones, dim)`` matrix of the phone embeddings computed by :func:`analyze` with hidden states."""
    return np.stack([p["embedding"] for p in scored]).astype(np.float64)


def word_matrix(scored, probabilities):
    """Aggregate the phones of each word: returns ``(word_indices, matrix)`` with :data:`WORD_FEATURES` columns."""
    by_word = {}
    for p, probability in zip(scored, probabilities):
        by_word.setdefault(p["word_index"], []).append((p, probability))
    rows, indices = [], []
    for word_index, items in by_word.items():
        values = {"p": [q for _, q in items]}
        for name, key in (("gm", "gop_margin"), ("del", "gop_del"), ("af", "gop_af"), ("lp", "lp_target"),
                          ("dur", "dur")):
            values[name] = [p[key] for p, _ in items]
        aggregated = {}
        for name, series in values.items():
            aggregated[f"{name}_min"], aggregated[f"{name}_max"] = min(series), max(series)
            aggregated[f"{name}_mean"] = sum(series) / len(series)
        aggregated["p_any"] = 1.0 - float(np.prod([1.0 - q for q in values["p"]]))
        first = items[0][0]
        for name in ("n_phones", "function_word", "utt_gm_mean", "utt_gm_std", "heard_ratio", "ll_full_per_phone"):
            aggregated[name] = first[name]
        rows.append([aggregated[name] for name in WORD_FEATURES])
        indices.append(word_index)
    return indices, np.array(rows, dtype=np.float64)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class LogisticModel:
    """Standardized logistic regression with an optional per-phone bias, stored as plain JSON."""

    def __init__(self, data):
        self.features = data["features"]
        self.mean = np.array(data["mean"])
        self.scale = np.array(data["scale"])
        self.coef = np.array(data["coef"])
        self.intercept = float(data["intercept"])
        self.phone_bias = data.get("phone_bias", {})

    def predict(self, matrix, phones=None):
        z = ((matrix - self.mean) / self.scale) @ self.coef + self.intercept
        if phones is not None and self.phone_bias:
            z = z + np.array([self.phone_bias.get(p, 0.0) for p in phones])
        return 1.0 / (1.0 + np.exp(-z))

    def to_dict(self):
        return {"type": "logistic", "features": self.features, "mean": self.mean.tolist(),
                "scale": self.scale.tolist(), "coef": self.coef.tolist(), "intercept": self.intercept,
                "phone_bias": self.phone_bias}


class MLPModel:
    """Small ReLU network with a sigmoid output on standardized features plus one-hot phone identity (JSON weights).

    An optional ``embedding`` block (``layer``, ``mean``, ``projection``, ``scale``) appends the
    projected and standardized phone embeddings of that recognizer layer to the inputs.
    """

    def __init__(self, data):
        self.data = data
        self.features = data["features"]
        self.mean = np.array(data["mean"])
        self.scale = np.array(data["scale"])
        self.phones = data.get("phones", [])
        self.index = {p: i for i, p in enumerate(self.phones)}
        self.weights = [np.array(w) for w in data["weights"]]
        self.biases = [np.array(b) for b in data["biases"]]
        self.embedding = data.get("embedding")
        self.hidden_layer = self.embedding["layer"] if self.embedding else None
        if self.embedding:
            self.embedding_mean = np.array(self.embedding["mean"])
            self.projection = np.array(self.embedding["projection"])
            self.embedding_scale = np.array(self.embedding["scale"])

    def predict(self, matrix, phones=None, embeddings=None):
        x = (matrix - self.mean) / self.scale
        if self.phones:
            onehot = np.zeros((len(x), len(self.phones)))
            for row, phone in enumerate(phones):
                if phone in self.index:
                    onehot[row, self.index[phone]] = 1.0
            x = np.column_stack([x, onehot])
        if self.embedding:
            if embeddings is None:
                raise ValueError("this model needs the phone embeddings (hidden states of the recognizer)")
            x = np.column_stack([x, ((embeddings - self.embedding_mean) @ self.projection) / self.embedding_scale])
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            x = np.maximum(x @ w + b, 0.0)
        z = (x @ self.weights[-1] + self.biases[-1]).ravel()
        return 1.0 / (1.0 + np.exp(-z))

    def to_dict(self):
        return self.data


class TreeEnsemble:
    """Gradient-boosted decision trees (numeric splits only) exported to JSON.

    ``trees`` is a list of node tables; each node is ``[feature, threshold, left, right, value]``
    with ``feature = -1`` for a leaf. The raw score is the baseline plus the leaf values;
    ``link`` is ``"sigmoid"`` (a probability, the default) or ``"identity"`` (a regression).
    """

    def __init__(self, data):
        self.data = data
        self.features = data["features"]
        self.baseline = float(data["baseline"])
        self.link = data.get("link", "sigmoid")
        self.trees = [np.array(tree, dtype=np.float64) for tree in data["trees"]]

    def predict(self, matrix, phones=None):
        matrix = np.asarray(matrix, dtype=np.float64)
        raw = np.full(len(matrix), self.baseline)
        for tree in self.trees:
            node = np.zeros(len(matrix), dtype=np.int64)
            feature = tree[node, 0]
            while (feature >= 0).any():
                active = feature >= 0
                go_left = matrix[np.arange(len(matrix))[active], feature[active].astype(int)] <= tree[node[active], 1]
                node[active] = np.where(go_left, tree[node[active], 2], tree[node[active], 3]).astype(int)
                feature = tree[node, 0]
            raw += tree[node, 4]
        if self.link == "identity":
            return raw
        return 1.0 / (1.0 + np.exp(-raw))

    def to_dict(self):
        return self.data


MODEL_TYPES = {"logistic": LogisticModel, "mlp": MLPModel, "trees": TreeEnsemble}


def _stage(data):
    return MODEL_TYPES[data.get("type", "logistic")](data)


class WordModel:
    """Phone model + word model + decision threshold, as shipped in ``models/word_mdd_<lang>.json``."""

    def __init__(self, data):
        self.data = data
        self.phone = _stage(data["phone"])
        self.word = _stage(data["word"])
        self.threshold = float(data["threshold"])
        #: Trained without the identity of the phones: the model every language without its own falls back on.
        self.generic = data.get("lang") == "generic"
        # Recognizer layer whose hidden states the phone stage reads (None: posteriors only).
        self.hidden_layer = getattr(self.phone, "hidden_layer", None)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=1)

    def predict(self, analysis):
        """``(phone_probabilities, {word_index: probability})`` for the output of :func:`analyze`."""
        scored = analysis["phones"]
        if self.hidden_layer is None:
            phone_probabilities = self.phone.predict(phone_matrix(scored), [p["phone"] for p in scored])
        else:
            phone_probabilities = self.phone.predict(phone_matrix(scored), [p["phone"] for p in scored],
                                                     embedding_matrix(scored))
        indices, matrix = word_matrix(scored, phone_probabilities)
        return phone_probabilities, dict(zip(indices, self.word.predict(matrix)))


@lru_cache(maxsize=8)
def load_model(lang=DEFAULT_LANGUAGE):
    """The :class:`WordModel` of ``lang``, or the generic (no phone identity) model, or ``None``."""
    for name in (f"word_mdd_{lang}.json", "word_mdd_generic.json"):
        path = model_path(name)
        if path is not None:
            return WordModel.load(path)
    return None


def hidden_layer(lang=DEFAULT_LANGUAGE):
    """The recognizer layer the word model of ``lang`` reads, or ``None`` (no model, or posteriors only)."""
    model = load_model(lang)
    return model.hidden_layer if model is not None else None


# ---------------------------------------------------------------------------
# Words that were not said, speech that was not expected
# ---------------------------------------------------------------------------

#: A recognized phone counts as heard from this confidence on.
HEARD_CONFIDENCE = 0.5
MISSING_FEATURES = ["del_max", "del_mean", "ctx_max", "ctx_mean", "lp_mean", "gm_mean", "frames_per_phone",
                    "heard_per_phone", "n_phones"]
#: Heard phones nobody expected, counted around a word boundary: the ones between the two
#: words, plus half of what each neighbour holds beyond its own phones.
EXTRA_SPEECH_SCORE = 2.0
EXTRA_NEIGHBOUR_WEIGHT = 0.5


def _word_spans(analysis):
    spans = {}
    for p in analysis["phones"]:
        start, end = spans.get(p["word_index"], (p["start"], p["end"]))
        spans[p["word_index"]] = (min(start, p["start"]), max(end, p["end"]))
    return spans


def _heard_spans(spans, confidences):
    return [span for span, confidence in zip(spans, confidences) if confidence >= HEARD_CONFIDENCE]


def missing_matrix(analysis, spans, confidences):
    """``(word_indices, matrix)`` of :data:`MISSING_FEATURES`, one row per aligned word.

    A word that was not said leaves traces the mispronounced ones do not: the alignment
    loses nothing when its phones are removed (``del_*``), none of them is plausible
    anywhere near (``ctx_*``), and it is squeezed into a few frames where no phone was heard.
    ``spans`` and ``confidences`` are the frame spans and confidences of the recognized phones.
    """
    heard = _heard_spans(spans, confidences)
    by_word = {}
    for p in analysis["phones"]:
        by_word.setdefault(p["word_index"], []).append(p)
    indices, rows = sorted(by_word), []
    for index in indices:
        phones = by_word[index]
        start, end = min(p["start"] for p in phones), max(p["end"] for p in phones)
        deletions = [p["gop_del"] for p in phones]
        contexts = [p["ctx_lp_target"] for p in phones]
        rows.append([
            max(deletions), float(np.mean(deletions)), max(contexts), float(np.mean(contexts)),
            float(np.mean([p["lp_target"] for p in phones])), float(np.mean([p["gop_margin"] for p in phones])),
            (end - start) / len(phones), sum(1 for s, e in heard if s < end and e > start) / len(phones), len(phones),
        ])
    return indices, np.array(rows, dtype=float)


class MissingModel:
    """Logistic model and threshold of ``models/word_missing_<lang>.json``."""

    def __init__(self, data):
        self.data = data
        self.model = LogisticModel(data["model"])
        self.threshold = float(data["threshold"])

    def predict(self, analysis, spans, confidences):
        """``{word_index: probability that the word was not said}``."""
        indices, matrix = missing_matrix(analysis, spans, confidences)
        return dict(zip(indices, self.model.predict(matrix).tolist()))


@lru_cache(maxsize=8)
def load_missing_model(lang=DEFAULT_LANGUAGE):
    """The :class:`MissingModel` of ``lang``, or the generic one, or ``None``."""
    for name in (f"word_missing_{lang}.json", "word_missing_generic.json"):
        path = model_path(name)
        if path is not None:
            with open(path, encoding="utf-8") as f:
                return MissingModel(json.load(f))
    return None


def extra_speech(analysis, spans, confidences):
    """Stretches where more was said than expected: ``[{after, before, start, end, heard}]``.

    ``after`` and ``before`` are the indices of the words around the stretch (``None`` at the
    beginning and at the end of the sentence), ``start`` / ``end`` frames, ``heard`` the
    indices of the recognized phones nobody expected there. Around every word boundary the
    heard phones lying between the two words are counted, plus :data:`EXTRA_NEIGHBOUR_WEIGHT`
    of what each neighbour holds beyond its own phones (the alignment stretches a word over
    speech it has no other place for); :data:`EXTRA_SPEECH_SCORE` makes a stretch.
    Boundaries that share a stretched word are reported once.
    """
    heard = [(i, span) for i, (span, confidence) in enumerate(zip(spans, confidences)) if confidence >= HEARD_CONFIDENCE]
    word_spans = _word_spans(analysis)
    indices = sorted(word_spans)
    if not indices:
        return []
    n_phones = {index: 0 for index in indices}
    for p in analysis["phones"]:
        n_phones[p["word_index"]] += 1
    surplus = {}
    for index in indices:
        start, end = word_spans[index]
        surplus[index] = max(0, sum(1 for _, (s, e) in heard if s < end and e > start) - n_phones[index])
    n_frames = analysis["utterance"]["n_frames"]
    fired = []
    for b in range(len(indices) + 1):
        previous = indices[b - 1] if b else None
        following = indices[b] if b < len(indices) else None
        gap_start = word_spans[previous][1] if previous is not None else 0
        gap_end = word_spans[following][0] if following is not None else n_frames
        between = [i for i, (s, e) in heard if s >= gap_start and e <= gap_end]
        score = len(between) + EXTRA_NEIGHBOUR_WEIGHT * (surplus.get(previous, 0) + surplus.get(following, 0))
        if score >= EXTRA_SPEECH_SCORE:
            fired.append((b, previous, following, between))
    # A stretched word fires the boundary on each of its sides. When the phones lie on one
    # side only, that side is the stretch; when they are all inside the word, the two
    # boundaries make one stretch around it.
    with_phones = {b for b, _, _, between in fired if between}
    events = []
    for b, previous, following, between in fired:
        if not between:
            if (b - 1 in with_phones and surplus.get(previous, 0)) or (b + 1 in with_phones and surplus.get(following, 0)):
                continue
            if events and events[-1]["_boundary"] == b - 1 and not events[-1]["heard"] and surplus.get(previous, 0):
                events[-1].update(before=following, _boundary=b)
                continue
        events.append({"after": previous, "before": following, "heard": list(between), "_boundary": b})
    for event in events:
        del event["_boundary"]
        if not event["heard"]:
            # Everything sits inside the neighbours: show what they hold.
            inside = [word_spans[i] for i in (event["after"], event["before"]) if i is not None and surplus[i]]
            event["heard"] = [i for i, (s, e) in heard if any(s < end and e > start for start, end in inside)]
        frames = [spans[i] for i in event["heard"]]
        event["start"] = min(s for s, _ in frames) if frames else 0
        event["end"] = max(e for _, e in frames) if frames else 0
    return events


# ---------------------------------------------------------------------------
# Utterance-level features (pronunciation aggregates + fluency from the alignment)
# ---------------------------------------------------------------------------

# A gap between two consecutive words longer than this (seconds) is a pause.
PAUSE_SECONDS = 0.25

UTTERANCE_FEATURES = [
    "gm_mean", "gm_std", "gm_min", "gm_q10", "del_mean", "del_min", "del_q10", "af_mean", "af_min", "af_q10",
    "lp_mean", "lp_q10", "frac_gm_below_1", "frac_gm_below_3", "frac_heard_other", "ll_full_per_phone",
    "heard_ratio", "greedy_ll_per_frame", "plausible_phone_ratio",
    "n_phones", "n_words", "speech_seconds", "phones_per_second", "phone_dur_mean", "phone_dur_std", "phone_dur_cv",
    "gap_mean", "gap_max", "gap_std", "n_pauses", "pause_ratio", "leading_silence", "trailing_silence",
    "word_dur_cv", "articulation_rate",
]


def utterance_features(analysis):
    """Utterance-level descriptors of the alignment: GOP aggregates and fluency (rate, pauses, rhythm).

    Durations are in seconds (:data:`FRAME_SECONDS` per frame). ``n_pauses`` counts the
    inter-word gaps longer than :data:`PAUSE_SECONDS`; ``pause_ratio`` is the share of the
    speech span spent in those gaps; ``articulation_rate`` is phones per second of speech
    excluding pauses.
    """
    scored = analysis["phones"]
    utterance = analysis["utterance"]
    margins = np.array([p["gop_margin"] for p in scored])
    deletions = np.array([p["gop_del"] for p in scored])
    free = np.array([p["gop_af"] for p in scored])
    targets = np.array([p["lp_target"] for p in scored])
    durations = np.array([p["end"] - p["start"] for p in scored], dtype=float) * FRAME_SECONDS
    words = {}
    for p in scored:
        start, end = words.get(p["word_index"], (p["start"], p["end"]))
        words[p["word_index"]] = (min(start, p["start"]), max(end, p["end"]))
    spans = [words[k] for k in sorted(words)]
    gaps = np.array([max(0, spans[i + 1][0] - spans[i][1]) for i in range(len(spans) - 1)], dtype=float) * FRAME_SECONDS
    word_durations = np.array([e - s for s, e in spans], dtype=float) * FRAME_SECONDS
    first, last = spans[0][0], spans[-1][1]
    speech = max(FRAME_SECONDS, (last - first) * FRAME_SECONDS)
    pauses = gaps[gaps > PAUSE_SECONDS] if len(gaps) else np.array([])
    pause_time = float(pauses.sum())
    features = {
        "gm_mean": margins.mean(), "gm_std": margins.std(), "gm_min": margins.min(),
        "gm_q10": np.quantile(margins, 0.1),
        "del_mean": deletions.mean(), "del_min": deletions.min(), "del_q10": np.quantile(deletions, 0.1),
        "af_mean": free.mean(), "af_min": free.min(), "af_q10": np.quantile(free, 0.1),
        "lp_mean": targets.mean(), "lp_q10": np.quantile(targets, 0.1),
        "frac_gm_below_1": np.mean(margins < -1.0), "frac_gm_below_3": np.mean(margins < -3.0),
        "frac_heard_other": np.mean([p["heard"] is not None for p in scored]),
        "ll_full_per_phone": utterance["ll_full_per_phone"], "heard_ratio": utterance["heard_ratio"],
        "greedy_ll_per_frame": utterance["greedy_ll_per_frame"],
        "plausible_phone_ratio": utterance["plausible_phone_ratio"],
        "n_phones": len(scored), "n_words": len(spans), "speech_seconds": speech,
        "phones_per_second": len(scored) / speech,
        "phone_dur_mean": durations.mean(), "phone_dur_std": durations.std(),
        "phone_dur_cv": durations.std() / max(durations.mean(), 1e-6),
        "gap_mean": gaps.mean() if len(gaps) else 0.0, "gap_max": gaps.max() if len(gaps) else 0.0,
        "gap_std": gaps.std() if len(gaps) else 0.0,
        "n_pauses": len(pauses), "pause_ratio": pause_time / speech,
        "leading_silence": first * FRAME_SECONDS,
        "trailing_silence": (utterance["n_frames"] - last) * FRAME_SECONDS,
        "word_dur_cv": word_durations.std() / max(word_durations.mean(), 1e-6),
        "articulation_rate": len(scored) / max(FRAME_SECONDS, speech - pause_time),
    }
    return {name: float(features[name]) for name in UTTERANCE_FEATURES}
