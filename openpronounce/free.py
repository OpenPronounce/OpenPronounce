"""Assessment of free speech: no expected text, any length.

The recording is cut at its pauses (:func:`openpronounce.audio.split_speech`), every
part is transcribed, and the transcript plays the role of the expected text: its phones
are aligned on the phone recognizer's posteriors like a read sentence would be. The
scores come from a model trained for that situation (``models/free_scorer_<lang>.json``).

What this can and cannot say. A word the transcription got wrong is compared with the
wrong phones, so word-level remarks are weaker than with a known text: they are
reported as ``unclear`` words, not as mispronunciations, and the scores lean on
fluency, prosody and on how confidently the two recognizers agree. Give the text
whenever you know it (:func:`openpronounce.speech.compare_audio_with_text`).
"""

import logging

import numpy as np

from . import audio, g2p, phones, scorer, speech
from .languages import DEFAULT_LANGUAGE, get_language

logger = logging.getLogger(__name__)

TARGETS = ("total", "accuracy", "fluency", "prosodic")
#: Below this much speech the scores are returned but flagged unreliable.
MIN_RELIABLE_SECONDS = 3.0
MIN_RELIABLE_WORDS = 4


def segment_features(waveform, transcript, recognition, lang=DEFAULT_LANGUAGE, sampling_rate=16000):
    """``(features, report)`` of one segment, or ``None`` when nothing can be aligned.

    ``features`` holds every :data:`openpronounce.scorer.FREE_FEATURES` entry, plus the pooled
    hidden states of the recognizer under ``"embedding"`` when it kept them; ``report`` is the
    output of :func:`openpronounce.phones.compare_phones` against the transcript.
    """
    words, groups = phones.get_expected_phones(transcript, lang)
    if not words:
        return None
    report, analysis = phones.compare_phones(recognition, transcript, lang, return_analysis=True)
    if analysis is None:
        return None
    features = scorer.features(analysis["analysis"], analysis["phone_probabilities"], analysis["word_probabilities"],
                               analysis["threshold"], waveform, sampling_rate)
    features.update(scorer.recognition_features(
        recognition.log_posteriors, recognition.phones, recognition.confidences, [p for g in groups for p in g]))
    if recognition.hidden is not None:
        features[scorer.EMBEDDING] = scorer.pooled_embedding(recognition.hidden)
    return features, report


def segment_results(waveform, sampling_rate=16000, lang=DEFAULT_LANGUAGE, max_segment_seconds=12.0):
    """Cut ``waveform`` at its pauses and yield ``(start, end, transcript, result)`` per part.

    ``start`` and ``end`` are samples, ``result`` the output of :func:`segment_features`
    or ``None`` when nothing was transcribed or aligned.
    """
    waveform = np.asarray(waveform, dtype=np.float32)
    for start, end in audio.split_speech(waveform, sampling_rate, max_seconds=max_segment_seconds):
        piece = waveform[start:end]
        transcript = " ".join(g2p.words_of(speech.transcribe(piece, lang)))
        result = None
        if transcript.strip():
            recognition = phones.recognize_phones(piece, sampling_rate, lang=lang)
            result = segment_features(piece, transcript, recognition, lang, sampling_rate)
        yield start, end, transcript, result


def assess(waveform, sampling_rate=16000, lang=DEFAULT_LANGUAGE, max_segment_seconds=12.0):
    """Assess a recording without knowing what the speaker meant to say.

    Returns a JSON-serializable dict: ``mode`` (``"free"``), ``score`` and ``scores``
    (``total``, ``accuracy``, ``fluency``, ``prosodic`` on 0-100, duration-weighted over the
    segments; ``None`` when the language has no free-speech model), ``transcribe``,
    ``duration`` and ``speech_duration`` in seconds, ``words`` (every word with ``start`` /
    ``end`` in seconds of the whole recording, ``confidence`` that it was unclear and
    ``unclear``), ``unclear_words``, ``segments`` and ``reliable``: ``False`` when there is
    too little speech to say anything (see ``reasons``).
    """
    lang = get_language(lang).code
    waveform = np.asarray(waveform, dtype=np.float32)
    duration = len(waveform) / sampling_rate
    has_model = scorer.load_scorer(lang, "free_scorer") is not None
    segments, words, weights, totals = [], [], [], {target: [] for target in TARGETS}
    for start, end, transcript, result in segment_results(waveform, sampling_rate, lang, max_segment_seconds):
        offset = start / sampling_rate
        segment = {"start": round(offset, 3), "end": round(end / sampling_rate, 3), "transcribe": transcript, "scores": None}
        segments.append(segment)
        if result is None:
            continue
        features, report = result
        for word in report.get("words", []):
            words.append({
                "word": word["word"], "start": round(word["start"] + offset, 3), "end": round(word["end"] + offset, 3),
                "confidence": word["confidence"], "unclear": bool(word["mispronounced"]),
                "expected": "".join(word["expected"]), "heard": "".join(word["actual"]),
            })
        if has_model:
            segment["scores"] = scorer.score(features, lang, "free_scorer")
            weights.append((end - start) / sampling_rate)
            for target in TARGETS:
                totals[target].append(segment["scores"][target])

    scores = None
    if weights:
        scores = {target: round(float(np.average(totals[target], weights=weights)), 1) for target in TARGETS}
    speech_duration = sum(s["end"] - s["start"] for s in segments)
    quality = audio.quality(waveform, sampling_rate)
    reasons = [f"recording {issue.replace('_', ' ')}" for issue in quality["issues"]]
    if not has_model:
        reasons.append(f"no free-speech model for {lang!r}")
    if speech_duration < MIN_RELIABLE_SECONDS and "too_short" not in quality["issues"]:
        reasons.append(f"less than {MIN_RELIABLE_SECONDS:g} s of speech")
    if len(words) < MIN_RELIABLE_WORDS:
        reasons.append(f"fewer than {MIN_RELIABLE_WORDS} words recognized")
    return {
        "mode": "free",
        "score": scores["total"] if scores else None,
        "scores": scores,
        "transcribe": " ".join(s["transcribe"] for s in segments if s["transcribe"]),
        "language": lang,
        "duration": round(duration, 3),
        "speech_duration": round(speech_duration, 3),
        "words": words,
        "unclear_words": [w["word"] for w in words if w["unclear"]],
        "segments": segments,
        "audio_quality": quality,
        "reliable": not reasons,
        "reasons": reasons,
    }
