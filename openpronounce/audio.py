"""Audio loading, conversion and reference-speech generation."""

import hashlib
import logging
import os
import shutil
import subprocess
import tempfile

import librosa
import numpy as np
import soundfile as sf

from openpronounce import tts

logger = logging.getLogger(__name__)

TARGET_SR = 16000

CACHE_DIR = os.environ.get(
    "OPENPRONOUNCE_CACHE_DIR",
    os.path.join(tempfile.gettempdir(), "openpronounce"),
)


def _decode_with_ffmpeg(file_path, sr):
    """Decode any container/codec ffmpeg knows (webm/opus, m4a, ...) to a mono float32 waveform."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is not installed")
    result = subprocess.run(
        [ffmpeg, "-v", "error", "-i", file_path, "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(sr), "-"],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", "replace").strip() or f"ffmpeg failed on {file_path!r}")
    return np.frombuffer(result.stdout, dtype=np.float32).copy()


#: Thresholds of :func:`quality`. A recording that trips one is still scored; the issue says why the score may be off.
#: The models normalize the level, so a quiet recording is fine as long as its 16 bits hold the
#: signal: the score of a native reading starts to drop below -60 dBFS (1.5 points there, 9 at -70).
MIN_SECONDS = 0.5
MAX_CLIPPING_RATIO = 0.01
MIN_LEVEL_DB = -60.0
MIN_SNR_DB = 10.0


def quality(waveform, sr=TARGET_SR):
    """Is this recording fit to be scored: ``duration``, ``level_db``, ``clipping_ratio``, ``snr_db`` and ``issues``.

    ``level_db`` is the RMS of the loudest tenth of the frames (dBFS), ``clipping_ratio``
    the share of samples at full scale, ``snr_db`` the gap between that loud tenth and the
    quietest tenth: a rough figure, low when there is no pause to hear the noise floor in.
    ``issues`` lists what is wrong among ``too_short``, ``silent``, ``too_quiet``,
    ``clipped`` and ``noisy``; empty when the recording is fine.
    """
    waveform = np.asarray(waveform, dtype=np.float32)
    duration = len(waveform) / sr
    report = {"duration": round(duration, 3), "level_db": None, "clipping_ratio": 0.0, "snr_db": None, "issues": []}
    if duration < MIN_SECONDS:
        report["issues"].append("too_short")
    if not len(waveform) or not np.any(np.abs(waveform) > 1e-4):
        report["issues"].append("silent")
        return report
    frame = min(len(waveform), 400)
    rms = np.sqrt(np.mean(waveform[: len(waveform) // frame * frame].reshape(-1, frame) ** 2, axis=1))
    loud, quiet = np.quantile(rms, 0.9), np.quantile(rms, 0.1)
    report["level_db"] = round(float(20 * np.log10(max(loud, 1e-6))), 1)
    report["clipping_ratio"] = round(float(np.mean(np.abs(waveform) >= 0.999)), 4)
    report["snr_db"] = round(float(20 * np.log10(max(loud, 1e-6) / max(quiet, 1e-6))), 1)
    if report["level_db"] < MIN_LEVEL_DB:
        report["issues"].append("too_quiet")
    if report["clipping_ratio"] > MAX_CLIPPING_RATIO:
        report["issues"].append("clipped")
    if report["snr_db"] < MIN_SNR_DB and duration >= 2.0:
        report["issues"].append("noisy")
    return report


def split_speech(waveform, sr=TARGET_SR, max_seconds=12.0, min_pause=0.5, min_unit=1.5, top_db=35.0, padding=0.15,
                 min_seconds=0.3):
    """Cut a recording into sentence-like spans: ``[(start, end)]`` in samples, each at most ``max_seconds`` long.

    The speech is cut at every pause of ``min_pause`` seconds or more (silence is ``top_db``
    below the peak). A span longer than ``max_seconds`` is cut again at its longest shorter
    pause, or in equal parts when it has none; a span shorter than ``min_unit`` joins its
    nearest neighbour when the two fit in ``max_seconds``. Spans are widened by ``padding``
    seconds without ever overlapping. An empty or silent recording gives ``[]``.
    """
    waveform = np.asarray(waveform)
    limit, pad = int(max_seconds * sr), int(padding * sr)
    if len(waveform) < int(min_seconds * sr) or not np.any(np.abs(waveform) > 1e-4):
        return []
    stretches = [(int(a), int(b)) for a, b in librosa.effects.split(waveform, top_db=top_db, frame_length=1024, hop_length=256)]

    def cut(group):
        """Split a run of stretches at its longest gap until every part fits in ``limit``."""
        if group[-1][1] - group[0][0] <= limit:
            return [group]
        if len(group) == 1:
            start, end = group[0]
            edges = np.linspace(start, end, -(-(end - start) // limit) + 1).astype(int)
            return [[(int(a), int(b))] for a, b in zip(edges[:-1], edges[1:])]
        widest = max(range(1, len(group)), key=lambda i: group[i][0] - group[i - 1][1])
        return cut(group[:widest]) + cut(group[widest:])

    groups = []
    for stretch in stretches:
        if groups and stretch[0] - groups[-1][-1][1] < int(min_pause * sr):
            groups[-1].append(stretch)
        else:
            groups.append([stretch])
    spans = [[part[0][0], part[-1][1]] for group in groups for part in cut(group)]

    short = int(min_unit * sr)
    while len(spans) > 1:
        candidates = []
        for i, (start, end) in enumerate(spans):
            if end - start >= short:
                continue
            for j in (i - 1, i + 1):
                if 0 <= j < len(spans) and max(end, spans[j][1]) - min(start, spans[j][0]) <= limit:
                    gap = spans[max(i, j)][0] - spans[min(i, j)][1]
                    candidates.append((gap, min(i, j)))
        if not candidates:
            break
        _, i = min(candidates)
        spans[i:i + 2] = [[spans[i][0], spans[i + 1][1]]]

    out = []
    for i, (start, end) in enumerate(spans):
        before = (start - spans[i - 1][1]) // 2 if i else start
        after = (spans[i + 1][0] - end) // 2 if i + 1 < len(spans) else len(waveform) - end
        start, end = start - min(pad, before), end + min(pad, after)
        if end - start >= int(min_seconds * sr):
            out.append((int(start), int(end)))
    return out


def load(file_path, sr=TARGET_SR):
    """Load any audio file (wav, mp3, flac, ogg, webm, m4a...) as a mono float32 waveform at ``sr`` Hz.

    libsndfile (through librosa) handles wav/flac/ogg/mp3; anything it cannot open
    (browser webm/opus recordings, m4a...) is decoded with ffmpeg.
    """
    try:
        waveform, _ = librosa.load(file_path, sr=sr, mono=True)
        return waveform
    except Exception as e:  # noqa: BLE001 - libsndfile cannot read this format, try ffmpeg
        libsndfile_error = e
    try:
        return _decode_with_ffmpeg(file_path, sr)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"Unable to decode {file_path!r} (libsndfile: {libsndfile_error}; ffmpeg: {e}). "
            "Make sure ffmpeg is installed."
        ) from e


def webm2wav(file_path):
    """Convert a browser-recorded file (webm/ogg/wav/...) to a 16 kHz mono ``*.16k.wav`` file next to it."""
    output_path = os.path.splitext(file_path)[0] + ".16k.wav"
    waveform = load(file_path)
    sf.write(output_path, waveform, TARGET_SR)
    return output_path


# Backward-compatible alias (the original name was a typo)
webp2wav = webm2wav


def text2speech(text, lang="en", filename=None, target_sr=TARGET_SR, *, backend=None, voice=None):
    """Generate a reference pronunciation of ``text`` as a 16 kHz mono wav file and return its path.

    The synthesizer is chosen with ``backend`` (``gtts``, ``piper`` or ``kokoro``), falling
    back to the ``OPENPRONOUNCE_TTS`` environment variable, then to gTTS; the voice with
    ``voice`` / ``OPENPRONOUNCE_TTS_VOICE``, then to a per-language default. See
    :mod:`openpronounce.tts`. Results are cached in ``CACHE_DIR`` keyed by
    ``(backend, voice, lang, text)`` so that repeated comparisons against the same
    sentence are free.
    """
    backend, voice = tts.resolve(lang, backend=backend, voice=voice)
    if filename is None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        key = hashlib.sha1(
            f"{backend}\x00{voice}\x00{lang}\x00{target_sr}\x00{text}".encode("utf-8")
        ).hexdigest()
        filename = os.path.join(CACHE_DIR, f"tts-{key}.wav")
        if os.path.exists(filename):
            return filename

    logger.info("Synthesizing reference with %s (voice %s, lang %s)", backend, voice, lang)
    waveform, sr = tts.synthesize(text, lang, backend, voice)
    if sr != target_sr:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)
    sf.write(filename, waveform, target_sr)
    return filename
