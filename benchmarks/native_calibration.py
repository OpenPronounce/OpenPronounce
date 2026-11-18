"""Check a language on native recordings: scores, words reported, notation gaps, reporting threshold, native shift.

    python benchmarks/native_calibration.py manifest.csv --lang nl
    python benchmarks/native_calibration.py manifest.csv --lang nl --free --shift openpronounce/models/native_nl.json

``manifest.csv`` has the columns ``audio`` (path, relative to the manifest) and ``text``:
native speakers reading sentences, a hundred recordings or more. No rating is needed.

``--shift`` writes what the generic scorers subtract for this language: the distance, feature
by feature, between these native speakers and the native speakers of English the scorers
keep as their reference (sentences of about twenty words; use recordings of that kind).
``--free`` adds the part for speech without a text, which transcribes every recording.

    python benchmarks/native_calibration.py english.csv --lang en --free --reference

``--reference`` is the other side of that distance: native speakers of English, read through
the generic models, whose means are written into the generic scorers. Only needed after a
generic scorer was trained again; every ``native_<lang>.json`` has to be written again after it.

Three more things come out. The scores natives get, to compare with the 74 English natives get
on the same kind of sentences. The phones espeak writes one way and the recognizer hears
another: a phone heard as the same other token most of the time is a difference of
notation or a regional pronunciation, to add to ``Language.variants`` in
``openpronounce/languages.py``. And the probability from which a word should be reported
(``Language.word_threshold``): every sentence is also scored with one word replaced by
another, and the threshold is the one that finds the replaced word as often as in English.
"""

import argparse
import collections
import csv
import os
import random
import sys

import numpy as np

#: A phone is "not recognized" below this GOP margin (log posterior of the phone minus the best token's).
MARGIN = -2.0
MIN_OCCURRENCES = 20
MIN_SHARE = 0.5
#: Share of the replaced words the generic model reports on English native speakers (FLEURS, dev split).
ENGLISH_FOUND = 0.79
REPLACEMENTS = 3


def read_manifest(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows or "audio" not in rows[0] or "text" not in rows[0]:
        raise SystemExit(f"{path}: needs the columns audio and text")
    folder = os.path.dirname(os.path.abspath(path))
    return [(os.path.join(folder, row["audio"]), row["text"]) for row in rows if row["text"].strip()]


def calibrate(recordings, lang, seed=0, free_speech=False):
    from openpronounce import audio, free, g2p, gop, phones, scorer

    features = {"scorer": [], "free_scorer": []}
    vocab = phones.phone_vocab()
    rng = random.Random(seed)
    pool = [w for _, text in recordings for w in g2p.words_of(text) if len(w) >= 4]
    margins, heard = collections.defaultdict(list), collections.defaultdict(collections.Counter)
    scores, own, replaced, flagged, unreliable = [], [], [], 0, 0
    for k, (path, text) in enumerate(recordings, 1):
        sound = audio.load(path)
        recognition = phones.recognize_phones(sound, lang=lang)
        result, bundle = phones.compare_phones(recognition, text, lang, return_analysis=True)
        if bundle is None:
            continue
        features["scorer"].append(scorer.features(bundle["analysis"], bundle["phone_probabilities"], bundle["word_probabilities"],
                                                  bundle["threshold"], sound))
        marks = scorer.score(features["scorer"][-1], lang)
        if free_speech:
            features["free_scorer"] += [found[0] for *_, found in free.segment_results(sound, lang=lang) if found is not None]
        scores.append(marks["total"])
        unreliable += marks.get("sentence_probability", 1.0) < 0.5
        flagged += len(result["errors"])
        # the model's own probabilities, not the ones the report shows on the scale of English
        own += [(bundle["word_probabilities"][w["position"]], bool(w.get("missing"))) for w in result["words"] if not w.get("guessed")]
        words = list(g2p.words_of(text))
        for _ in range(REPLACEMENTS):
            position = rng.randrange(len(words))
            others = [w for w in pool if w != words[position] and w[:2] != words[position][:2]]
            changed, raw = phones.compare_phones(recognition, " ".join(words[:position] + [rng.choice(others)] + words[position + 1:]),
                                                 lang, return_analysis=True)
            replaced += [(raw["word_probabilities"][position], bool(w.get("missing"))) for w in changed.get("words", [])
                         if w["position"] == position]
        for phone in bundle["analysis"]["phones"]:
            margins[phone["phone"]].append(phone["gop_margin"])
            if phone["gop_margin"] < MARGIN and phone["heard"] is not None:
                heard[phone["phone"]][vocab[phone["heard"]]] += 1
        print(f"\r{k}/{len(recordings)}", end="", file=sys.stderr)
    print(file=sys.stderr)
    summary = {"score": float(np.mean(scores)), "low": float(np.percentile(scores, 10)), "flagged": flagged,
               "words": len(own), "unreliable": int(unreliable), "recordings": len(scores)}
    return margins, heard, summary, threshold_for(own, replaced, gop.load_model(lang)), features


def write_shift(path, lang, features, source):
    """Write ``native_<lang>.json``: the offsets of :func:`openpronounce.scorer.native_offsets` for each kind measured."""
    import json

    from openpronounce import scorer

    data = {"lang": lang, "trained_on": f"{source}, {len(features['scorer'])} native recordings, generic models",
            "shift": {kind: scorer.native_offsets(rows, kind) for kind, rows in features.items() if rows}}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    return data


def threshold_for(own, replaced, model):
    """Reports at the current threshold, and the threshold that finds a replaced word as often as in English."""
    own_p, own_missing = np.array([p for p, _ in own]), np.array([m for _, m in own])
    new_p, new_missing = np.array([p for p, _ in replaced]), np.array([m for _, m in replaced])

    def rates(t):
        return float(np.mean((new_p >= t) | new_missing)), float(np.mean((own_p >= t) | own_missing))

    candidates = [t for t in np.arange(round(model.threshold, 2), 0.0, -0.01) if rates(t)[0] >= ENGLISH_FOUND]
    suggested = round(float(candidates[0]), 2) if candidates else None
    return {"model": model.threshold, "at_model": rates(model.threshold), "suggested": suggested,
            "at_suggested": rates(suggested) if suggested else None}


def write_reference(features, source):
    """Write the means of English natives into the generic scorers: ``{kind: path}`` of the files changed."""
    from openpronounce import gop, scorer

    written = {}
    for kind, rows in features.items():
        path = gop.model_path(f"{kind}_generic.json")
        if not rows or path is None:
            continue
        model = scorer.Scorer.load(path)
        model.data["native_reference"] = {name: round(float(np.mean([row[name] for row in rows])), 5)
                                          for name in scorer.SHIFTED_FEATURES[kind]}
        model.data["native_reference_source"] = (f"{source}, {len(rows)} {'segments' if kind == 'free_scorer' else 'recordings'}"
                                                 " of native speakers, generic word model")
        model.save(path)
        written[kind] = path
    scorer.load_scorer.cache_clear()
    return written


def scorer_is_own(lang):
    from openpronounce import scorer

    return scorer.load_scorer(lang).data.get("lang") != "generic"


def shift_installed(lang):
    from openpronounce import scorer

    return bool(scorer.native_shift(lang))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest")
    parser.add_argument("--lang", required=True)
    parser.add_argument("--free", action="store_true", help="also measure speech without a text (transcribes every recording)")
    parser.add_argument("--shift", metavar="OUT", help="write the native shift of the language, e.g. models/native_nl.json")
    parser.add_argument("--reference", action="store_true", help="with --lang en: write the native reference of the generic scorers")
    args = parser.parse_args(argv)
    if args.reference:
        if args.lang != "en" or args.shift:
            raise SystemExit("--reference takes native speakers of English (--lang en) and goes without --shift")
        from word_model import english_on_generic_models

        english_on_generic_models()
    margins, heard, summary, threshold, features = calibrate(read_manifest(args.manifest), args.lang, free_speech=args.free)
    if args.reference:
        for kind, path in write_reference(features, os.path.basename(args.manifest)).items():
            print(f"native reference of the {kind.replace('_', ' ')} written to {path}")
    if args.shift:
        if scorer_is_own(args.lang):
            raise SystemExit(f"{args.lang!r} has a scorer of its own: the native shift only serves the generic scorers")
        data = write_shift(args.shift, args.lang, features, os.path.basename(args.manifest))
        print(f"native shift written to {args.shift} ({', '.join(data['shift'])}); scores below were computed "
              f"{'with the shift already installed' if shift_installed(args.lang) else 'without it'}")
    print(f"{summary['recordings']} recordings: mean score {summary['score']:.1f} (one in ten under {summary['low']:.0f}), "
          f"{summary['flagged']} of {summary['words']} words reported, {summary['unreliable']} recordings unreliable")
    found, false_alarms = threshold["at_model"]
    print(f"a word replaced by another is reported {found:.0%} of the time at the model's threshold ({threshold['model']:.2f}), "
          f"with {false_alarms:.2%} of the words really read")
    if threshold["suggested"]:
        found, false_alarms = threshold["at_suggested"]
        print(f"word_threshold={threshold['suggested']:.2f} reports it {found:.0%} of the time, as in English, "
              f"with {false_alarms:.2%} of the words really read")
    print(f"{'phone':8s} {'n':>4s} {'margin':>7s}  heard instead (share of the occurrences)")
    suggestions = []
    for phone, values in sorted(margins.items(), key=lambda item: np.mean(item[1])):
        if np.mean(values) > MARGIN / 2:
            break
        others = [(token, count / len(values)) for token, count in heard[phone].most_common(3)]
        print(f"{phone:8s} {len(values):4d} {np.mean(values):7.2f}  " + ", ".join(f"{t} {s:.0%}" for t, s in others))
        accepted = tuple(token for token, share in others if share >= MIN_SHARE / 2)
        if len(values) >= MIN_OCCURRENCES and sum(share for token, share in others if token in accepted) >= MIN_SHARE:
            suggestions.append((phone, accepted))
    if suggestions:
        print("\nvariants worth trying for this language (keep one only if the figures above improve with it):")
        print("    variants=(" + ", ".join(f'("{phone}", {accepted!r})' for phone, accepted in suggestions) + ")")


if __name__ == "__main__":
    main()
