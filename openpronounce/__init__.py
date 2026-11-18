"""OpenPronounce: open-source, phoneme-level pronunciation assessment (English by default).

Typical usage::

    from openpronounce import load_audio, compare_audio_with_text

    sound = load_audio("recording.wav")
    result = compare_audio_with_text(sound, "Hello, I am a developer")
    print(result["score"], result["differences"]["errors"])

    result = compare_audio_with_text(load_audio("bonjour.wav"), "Bonjour le monde", lang="fr")

    result = assess(sound)                    # no expected text: free speech, any length
    print(result["score"], result["transcribe"], result["unclear_words"])
"""

from .audio import load as load_audio, text2speech
from .free import assess as assess_free
from .languages import LANGUAGES, get_language
from .phones import compare_phones, recognize_phones, transcribe_phones
from .speech import (
    compare_audio_with_text,
    compare_transcriptions,
    get_phonemes,
    get_phonemes_with_word_mapping,
    transcribe,
)

__version__ = "0.3.0"


def assess(waveform, text=None, lang="en", **options):
    """Assess a 16 kHz mono waveform: against ``text`` when given, as free speech otherwise.

    With a text this is :func:`compare_audio_with_text`; without one,
    :func:`openpronounce.free.assess`. The two results do not have the same shape: read
    their ``mode`` key (``"free"``) or the documentation of each function.
    """
    if text is not None and text.strip():
        return compare_audio_with_text(waveform, text, lang=lang, **options)
    return assess_free(waveform, lang=lang, **options)


__all__ = [
    "__version__",
    "assess",
    "assess_free",
    "load_audio",
    "text2speech",
    "LANGUAGES",
    "get_language",
    "compare_audio_with_text",
    "compare_transcriptions",
    "compare_phones",
    "recognize_phones",
    "transcribe_phones",
    "get_phonemes",
    "get_phonemes_with_word_mapping",
    "transcribe",
]
