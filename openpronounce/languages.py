"""Registry of the supported languages.

Each language binds a 2-letter code (also the gTTS code) to the espeak voice used
for phonemization and to the Wav2Vec2 CTC checkpoint used for word transcription.
The phone recognizer (see :mod:`openpronounce.phones`) is multilingual and shared.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Language:
    code: str
    espeak: str
    asr_model: str
    name: str
    #: Mean per-frame DTW distance between the Wav2Vec2 embeddings of two good native
    #: voices of this language (gTTS vs Piper on three sentences, English checkpoint).
    #: The acoustic term of the score maps this value to 100 and ``acoustic_good + 9`` to 0.
    #: English is calibrated on human readings and speechocean762 (see benchmarks/);
    #: for the other languages the English embeddings put even two native voices
    #: further apart, hence the offset.
    acoustic_good: float = 6.0
    #: Recognizer tokens also accepted for a phone espeak writes: what native speakers are
    #: heard to say where espeak expects that phone (``benchmarks/native_calibration.py`` on
    #: the dev split of FLEURS, see benchmarks/README.md). Without them a native reading is
    #: blamed for the gap between the two inventories, or for a regional pronunciation.
    variants: tuple = ()
    #: Probability of the generic word model that counts as its threshold in this language.
    #: That threshold is tuned on learners of English, where a wrong word gets a higher
    #: probability than elsewhere: this value finds a word replaced by another as often as in
    #: English (four times in five), measured on native speakers (FLEURS dev). The reports
    #: show the probability moved onto the model's scale. ``None`` keeps the model's own.
    word_threshold: float | None = None

    def accepted(self, phone):
        """Tokens accepted for ``phone`` besides itself."""
        return dict(self.variants).get(phone, ())


DEFAULT_LANGUAGE = "en"

LANGUAGES = {
    "en": Language("en", "en-us", "facebook/wav2vec2-large-960h", "English", acoustic_good=6.0),
    "fr": Language("fr", "fr-fr", "jonatasgrosman/wav2vec2-large-xlsr-53-french", "French", acoustic_good=9.0,
                   word_threshold=0.09),
    "es": Language("es", "es", "jonatasgrosman/wav2vec2-large-xlsr-53-spanish", "Spanish", acoustic_good=10.0,
                   word_threshold=0.07),
    "de": Language("de", "de", "jonatasgrosman/wav2vec2-large-xlsr-53-german", "German", acoustic_good=10.0,
                   word_threshold=0.06),
    "it": Language("it", "it", "jonatasgrosman/wav2vec2-large-xlsr-53-italian", "Italian", acoustic_good=9.0,
                   word_threshold=0.22, variants=(("ɪ", ("i",)), ("ʊ", ("u",)))),
    "pt": Language("pt", "pt-br", "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese", "Portuguese", acoustic_good=11.0,
                   word_threshold=0.15, variants=(("æ", ("ɐ",)), ("y", ("ɨ", "i")), ("r", ("ʁ",)), ("x", ("ʁ",)))),
    "nl": Language("nl", "nl", "jonatasgrosman/wav2vec2-large-xlsr-53-dutch", "Dutch", acoustic_good=13.0,
                   word_threshold=0.08,
                   variants=(("ʋ", ("v",)), ("ɣ", ("x",)), ("ɛɪ", ("aɪ",)), ("œy", ("aʊ",)), ("ʌʊ", ("aʊ",)),
                             ("ɑ", ("ɔ", "a")), ("ɪː", ("eː",)), ("ɔː", ("oː",)), ("ɵ", ("ə", "y", "ʊ")))),
}


def get_language(code):
    """Return the :class:`Language` for ``code`` or raise ``ValueError`` listing the supported codes."""
    try:
        return LANGUAGES[code]
    except KeyError:
        raise ValueError(f"unsupported language {code!r}, expected one of: {', '.join(LANGUAGES)}") from None
