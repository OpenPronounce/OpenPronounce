"""Grapheme to phoneme: the phones a text is expected to be pronounced with.

Two backends, chosen per language or forced with ``$OPENPRONOUNCE_G2P``:

``lexicon``
    A pronunciation dictionary shipped with the package. English uses CMUdict
    (``data/cmudict.dict.gz``, BSD-style licence in ``data/cmudict.LICENSE``), mapped
    to the phone recognizer's own symbols. A word missing from the dictionary is
    derived from a known word (plural, past tense, possessive, compound) or, failing
    that, spelled out with letter-to-sound rules: that last pronunciation is a guess,
    :func:`is_guessed` says so and the assessment never reports an error on it.

``espeak``
    The espeak-ng program, run as a separate process and read through its standard
    output. espeak-ng is GPL-3; nothing of it is imported or linked, it only has to be
    installed on the system. Every language other than English uses it.

On the speechocean762 test split the two backends give the same results with the
shipped English models (word AUC 0.928 against 0.924, total Pearson 0.791 against 0.790).
"""

import gzip
import logging
import os
import re
import shutil
import subprocess
from functools import lru_cache

from .languages import DEFAULT_LANGUAGE, get_language

logger = logging.getLogger(__name__)

BACKEND_ENV = "OPENPRONOUNCE_G2P"
BACKENDS = ("lexicon", "espeak")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
LEXICONS = {"en": "cmudict.dict.gz"}

_WORD_RE = re.compile(r"\b[\w']+\b")

# ARPAbet (CMUdict) to the recognizer's espeak-style symbols.
_VOWELS = {"AA": "ɑː", "AE": "æ", "AO": "ɔː", "AW": "aʊ", "AY": "aɪ", "EH": "ɛ", "EY": "eɪ", "IH": "ɪ",
           "IY": "iː", "OW": "oʊ", "OY": "ɔɪ", "UH": "ʊ", "UW": "uː"}
_CONSONANTS = {"B": "b", "CH": "tʃ", "D": "d", "DH": "ð", "F": "f", "G": "ɡ", "HH": "h", "JH": "dʒ", "K": "k",
               "L": "l", "M": "m", "N": "n", "NG": "ŋ", "P": "p", "R": "ɹ", "S": "s", "SH": "ʃ", "T": "t",
               "TH": "θ", "V": "v", "W": "w", "Y": "j", "Z": "z", "ZH": "ʒ"}
_R_COLOURED = {"AA": "ɑːɹ", "AO": "ɔːɹ", "EH": "ɛɹ", "IH": "ɪɹ", "IY": "ɪɹ", "UH": "ʊɹ"}
_VOICELESS = {"P", "T", "K", "F", "TH"}
_SIBILANTS = {"S", "Z", "SH", "ZH", "CH", "JH"}


def backend(lang=DEFAULT_LANGUAGE):
    """Backend used for ``lang``: ``$OPENPRONOUNCE_G2P`` when set, else the lexicon when the language has one."""
    forced = os.environ.get(BACKEND_ENV, "").strip().lower()
    if forced:
        if forced not in BACKENDS:
            raise ValueError(f"{BACKEND_ENV}={forced!r}, expected one of: {', '.join(BACKENDS)}")
        if forced == "lexicon" and lang not in LEXICONS:
            raise ValueError(f"no pronunciation lexicon for {lang!r}: unset {BACKEND_ENV} or use espeak")
        return forced
    return "lexicon" if lang in LEXICONS else "espeak"


# ---------------------------------------------------------------------------
# Lexicon backend
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def _lexicon(lang):
    """``{word: ARPAbet symbols}``, first pronunciation of each entry."""
    entries = {}
    with gzip.open(os.path.join(DATA_DIR, LEXICONS[lang]), "rt", encoding="utf-8") as f:
        for line in f:
            line = line.split("#")[0].strip()
            if not line:
                continue
            word, *symbols = line.split()
            if "(" not in word:  # "read(2)": alternate pronunciation
                entries[word] = tuple(symbols)
    return entries


def _is_vowel(symbol):
    return symbol is not None and symbol[-1].isdigit()


def arpabet_to_phones(symbols):
    """ARPAbet symbols (with stress digits) as the recognizer's phones: r-coloured vowels, flaps and syllabic l merged like espeak does."""
    out, i = [], 0
    while i < len(symbols):
        symbol = symbols[i]
        base, stress = (symbol[:-1], symbol[-1]) if symbol[-1].isdigit() else (symbol, None)
        following = symbols[i + 1] if i + 1 < len(symbols) else None
        after = symbols[i + 2] if i + 2 < len(symbols) else None
        if base in _R_COLOURED and following == "R" and not _is_vowel(after):
            out.append(_R_COLOURED[base])
            i += 2
            continue
        if base == "AH" and stress == "0" and following == "L" and not _is_vowel(after):
            out.append("əl")
            i += 2
            continue
        if base == "ER":
            out.append("ɜː" if stress == "1" else "ɚ")
        elif base == "AH":
            out.append("ə" if stress == "0" else "ʌ")
        elif base == "IY" and stress == "0" and following is None:
            out.append("i")
        elif base in _VOWELS:
            out.append(_VOWELS[base])
        elif base == "T" and i and _is_vowel(symbols[i - 1]) and following is not None and following[-1] == "0":
            out.append("ɾ")
        else:
            out.append(_CONSONANTS[base])
        i += 1
    return tuple(out)


def _suffixed(word, lexicon, depth=0):
    """ARPAbet of ``word`` as a known word plus an ending (plural, possessive, past tense, -ing, -ly), or ``None``."""
    if word in lexicon:
        return lexicon[word]
    if depth >= 2:
        return None

    def base_of(stem):
        candidates = [stem, stem + "e"]
        if len(stem) > 2 and stem[-1] == stem[-2]:
            candidates.append(stem[:-1])  # "stopped", and the learner's "jumpped"
        if stem.endswith("i"):
            candidates.append(stem[:-1] + "y")  # "carried"
        for candidate in candidates:
            found = _suffixed(candidate, lexicon, depth + 1)
            if found:
                return found
        return None

    for ending in ("'s", "s'", "es", "s"):
        if word.endswith(ending) and len(word) > len(ending) + 1:
            base = base_of(word[:-len(ending)])
            if base:
                last = base[-1].rstrip("012")
                return base + (("IH0", "Z") if last in _SIBILANTS else ("S",) if last in _VOICELESS else ("Z",))
    for ending in ("ed", "d"):
        if word.endswith(ending) and len(word) > len(ending) + 2:
            base = base_of(word[:-len(ending)])
            if base:
                last = base[-1].rstrip("012")
                return base + (("IH0", "D") if last in ("T", "D") else ("T",) if last in _VOICELESS | {"S", "SH", "CH"}
                               else ("D",))
    for ending, symbols in (("ing", ("IH0", "NG")), ("ly", ("L", "IY0"))):
        if word.endswith(ending) and len(word) > len(ending) + 2:
            base = base_of(word[:-len(ending)])
            if base:
                return base + symbols
    return None


def _compound(word, lexicon):
    """ARPAbet of ``word`` as two known words of three letters or more ("birdbath"), or ``None``."""
    best = None
    for cut in range(3, len(word) - 2):
        head, tail = word[:cut], word[cut:]
        if head in lexicon and tail in lexicon and (best is None or min(cut, len(word) - cut) > best[0]):
            best = (min(cut, len(word) - cut), lexicon[head] + lexicon[tail])
    return best[1] if best else None


_GRAPHEMES = [
    ("tch", ("CH",)), ("igh", ("AY1",)), ("sch", ("SH",)),
    ("ch", ("CH",)), ("sh", ("SH",)), ("th", ("TH",)), ("ph", ("F",)), ("ck", ("K",)), ("ng", ("NG",)),
    ("qu", ("K", "W")), ("wh", ("W",)), ("ee", ("IY1",)), ("ea", ("IY1",)), ("oo", ("UW1",)), ("ou", ("AW1",)),
    ("ow", ("OW1",)), ("ai", ("EY1",)), ("ay", ("EY1",)), ("oa", ("OW1",)), ("oi", ("OY1",)), ("oy", ("OY1",)),
    ("au", ("AO1",)), ("aw", ("AO1",)), ("ie", ("IY1",)), ("ar", ("AA1", "R")), ("er", ("ER1",)),
    ("ir", ("ER1",)), ("ur", ("ER1",)), ("or", ("AO1", "R")),
    ("a", ("AE1",)), ("b", ("B",)), ("d", ("D",)), ("e", ("EH1",)), ("f", ("F",)), ("g", ("G",)), ("h", ("HH",)),
    ("i", ("IH1",)), ("j", ("JH",)), ("k", ("K",)), ("l", ("L",)), ("m", ("M",)), ("n", ("N",)), ("o", ("AA1",)),
    ("p", ("P",)), ("q", ("K",)), ("r", ("R",)), ("s", ("S",)), ("t", ("T",)), ("u", ("AH1",)), ("v", ("V",)),
    ("w", ("W",)), ("x", ("K", "S")), ("z", ("Z",)),
]


def _spelled(word):
    """Rough letter-to-sound rules, for words no dictionary entry explains. Always a guess."""
    letters = re.sub(r"[^a-z]", "", word)
    letters = re.sub(r"([b-df-hj-np-tv-z])\1", r"\1", letters)  # doubled consonant
    if len(letters) > 3 and letters.endswith("e") and letters[-2] not in "aeiou":
        letters = letters[:-1]  # silent final e
    out, i = [], 0
    while i < len(letters):
        if letters[i] == "c":
            out.append("S" if letters[i + 1:i + 2] in ("e", "i", "y") else "K")
            i += 1
            continue
        if letters[i] == "y":
            out.append("Y" if i == 0 else "IY0" if i == len(letters) - 1 else "IH1")
            i += 1
            continue
        if letters[i] == "i" and i == len(letters) - 1 and i:
            out.append("IY0")
            i += 1
            continue
        for grapheme, symbols in _GRAPHEMES:
            if letters.startswith(grapheme, i):
                out.extend(symbols)
                i += len(grapheme)
                break
        else:
            i += 1
    return tuple(out)


_ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
         "seventeen eighteen nineteen").split()
_TENS = "twenty thirty forty fifty sixty seventy eighty ninety".split()


def number_words(n):
    """English words of a non-negative integer below one billion; longer digit strings are read digit by digit."""
    if n < 20:
        return [_ONES[n]]
    if n < 100:
        return [_TENS[n // 10 - 2]] + ([_ONES[n % 10]] if n % 10 else [])
    if n < 1000:
        return [_ONES[n // 100], "hundred"] + (number_words(n % 100) if n % 100 else [])
    for size, name in ((10 ** 6, "million"), (10 ** 3, "thousand")):
        if n >= size:
            return number_words(n // size) + [name] + (number_words(n % size) if n % size else [])
    raise ValueError(n)


@lru_cache(maxsize=8192)
def _lexicon_word(word, lang):
    """``(phones, guessed)`` of one lower-case word."""
    lexicon = _lexicon(lang)
    if word.isdigit():
        spoken = number_words(int(word)) if len(word) <= 9 and not (len(word) > 1 and word[0] == "0") \
            else [_ONES[int(d)] for d in word]
        return tuple(p for w in spoken for p in arpabet_to_phones(lexicon[w])), False
    symbols = _suffixed(word, lexicon) or _suffixed(word.replace("'", ""), lexicon) or _compound(word, lexicon)
    if symbols:
        return arpabet_to_phones(symbols), False
    return arpabet_to_phones(_spelled(word)), True


# ---------------------------------------------------------------------------
# espeak backend: the espeak-ng program, run as a separate process
# ---------------------------------------------------------------------------

ESPEAK_ENV = "OPENPRONOUNCE_ESPEAK"
_ESPEAK_NOISE = re.compile(r"\([a-z-]+\)|[ˈˌ\-]")


def espeak_binary():
    """Path of the espeak-ng program (``$OPENPRONOUNCE_ESPEAK``, else ``espeak-ng`` or ``espeak`` on the PATH)."""
    binary = os.environ.get(ESPEAK_ENV) or shutil.which("espeak-ng") or shutil.which("espeak")
    if not binary:
        raise RuntimeError(
            "this language needs the espeak-ng program: apt install espeak-ng, brew install espeak-ng, "
            f"or point ${ESPEAK_ENV} at it"
        )
    return binary


def _espeak(text, voice):
    """Phones of ``text`` as espeak-ng prints them: one tuple of phones per space-separated chunk of its output."""
    out = subprocess.run([espeak_binary(), "-q", "--ipa", "--sep=_", "-v", voice, text],
                         capture_output=True, text=True, encoding="utf-8", check=True, timeout=30).stdout
    groups = []
    for chunk in out.split():
        # Stress marks, liaison hyphens and language switches such as "(en)" are not phones
        phones = [p for p in _ESPEAK_NOISE.sub("", chunk).split("_") if p]
        if phones:
            groups.append(tuple(phones))
    return groups


def _espeak_words(words, lang):
    voice = get_language(lang).espeak
    groups = _espeak(" ".join(words), voice)
    if len(groups) != len(words):
        # espeak merges some word sequences ("would have to"): one call per word
        logger.debug("espeak returned %d groups for %d words, falling back to per-word", len(groups), len(words))
        groups = []
        for word in words:
            parts = _espeak(word, voice)
            groups.append(tuple(p for part in parts for p in part))
    return tuple(groups)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def words_of(text):
    """Lower-case words of ``text``, punctuation dropped."""
    return tuple(w.lower() for w in _WORD_RE.findall(text))


@lru_cache(maxsize=1024)
def phones_by_word(text, lang=DEFAULT_LANGUAGE):
    """``(words, phones_per_word)`` of ``text``, phones as the backend writes them (not normalized)."""
    words = words_of(text)
    if not words:
        return (), ()
    if backend(lang) == "lexicon":
        return words, tuple(_lexicon_word(word, lang)[0] for word in words)
    return words, _espeak_words(words, lang)


def is_guessed(word, lang=DEFAULT_LANGUAGE):
    """True when the expected pronunciation of ``word`` is a letter-to-sound guess, not a dictionary entry."""
    if backend(lang) != "lexicon":
        return False
    return _lexicon_word(word.lower(), lang)[1]


def clear_cache():
    """Forget cached pronunciations (after changing ``$OPENPRONOUNCE_G2P``)."""
    phones_by_word.cache_clear()
    _lexicon_word.cache_clear()
