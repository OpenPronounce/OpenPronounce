"""Words that were not said, speech that was not expected: synthetic cases on speechocean762.

    python benchmarks/missing_extra.py --fit --out openpronounce/models/word_missing_generic.json
    python benchmarks/missing_extra.py --report

Learners of the corpus read what they are given, so the cases are built from the text:
a word added to the sentence is a word the speaker did not say, a word removed from it
is a word said in excess. Needs the cached posteriors (``word_model.py --extract``).
"""

import argparse
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from speechocean762 import HUMAN_BAD_WORD, load_rows  # noqa: E402
from word_model import load_cache, load_posteriors  # noqa: E402

#: Share of the real words of the train split the missing-word threshold lets through.
FALSE_ALARM_RATE = 0.005


def _analyze(recognition, text, lang):
    from openpronounce import phones

    result, bundle = phones.compare_phones(recognition, text, lang, return_analysis=True)
    return (result, bundle) if bundle is not None else (None, None)


def cases(split, lang="en", draws=1, seed=0):
    """Missing-word rows (features, label), word flags of the untouched sentences and extra-speech outcomes."""
    from openpronounce import gop, phones

    rng = random.Random(seed)
    labels, vocab = load_cache(split)
    rows = {r["utt"]: r for r in load_rows(split)}
    pool = [w["text"] for r in rows.values() for w in r["words"] if len(w["phones"]) >= 2]
    X, y, words, extra = [], [], [], {"hit": [], "phones": [], "false_boundaries": 0, "boundaries": 0, "sentences": 0,
                                       "false_sentences": 0}
    for lab in labels:
        log_posteriors, hidden = load_posteriors(split, lab["utt"])
        recognition = phones.decode_ctc(log_posteriors, vocab, 0, lang, True, hidden)
        text, human = lab["text"].split(), rows[lab["utt"]]["words"]
        result, bundle = _analyze(recognition, lab["text"], lang)
        if result is None or len(result["words"]) != len(human):
            continue
        _, matrix = gop.missing_matrix(bundle["analysis"], recognition.spans, recognition.confidences)
        X.extend(matrix)
        y.extend([0] * len(matrix))
        words.extend((w["confidence"] >= bundle["threshold"] and not w.get("guessed"), bool(w.get("missing")),
                      g["accuracy"] < HUMAN_BAD_WORD) for w, g in zip(result["words"], human))
        extra["sentences"] += 1
        extra["boundaries"] += len(text) + 1
        extra["false_boundaries"] += len(result["extra"])
        extra["false_sentences"] += bool(result["extra"])
        for _ in range(draws):
            k = rng.randrange(len(text) + 1)
            _, changed = _analyze(recognition, " ".join(text[:k] + [rng.choice(pool)] + text[k:]), lang)
            if changed is not None:
                indices, matrix = gop.missing_matrix(changed["analysis"], recognition.spans, recognition.confidences)
                if k in indices:
                    X.append(matrix[indices.index(k)])
                    y.append(1)
        candidates = [i for i, g in enumerate(human) if len(g["phones"]) >= 2 and g["accuracy"] >= HUMAN_BAD_WORD]
        if len(text) >= 2 and candidates:
            k = rng.choice(candidates)
            shorter, _ = _analyze(recognition, " ".join(text[:k] + text[k + 1:]), lang)
            if shorter is not None and len(shorter["words"]) == len(text) - 1:
                before, after = (k if k < len(text) - 1 else None), (k - 1 if k else None)
                extra["hit"].append(any(e["after"] == after or e["before"] == before for e in shorter["extra"]))
                extra["phones"].append(len(human[k]["phones"]))
    return np.array(X), np.array(y), np.array(words, dtype=bool), extra


def fit(args):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    from openpronounce import gop

    X, y, _, _ = cases("train", args.lang, draws=2)
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(max_iter=5000).fit(scaler.transform(X), y)
    probabilities = model.predict_proba(scaler.transform(X))[:, 1]
    threshold = float(np.quantile(probabilities[y == 0], 1 - FALSE_ALARM_RATE))
    data = {
        "lang": "generic", "threshold": round(threshold, 4),
        "model": {"type": "logistic", "features": gop.MISSING_FEATURES, "mean": scaler.mean_.tolist(),
                  "scale": scaler.scale_.tolist(), "coef": model.coef_[0].tolist(), "intercept": float(model.intercept_[0])},
        "trained_on": f"speechocean762 train split, {int((y == 0).sum())} read words and {int(y.sum())} words added to the text",
        "metrics": {},
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    gop.load_missing_model.cache_clear()
    data["metrics"] = report(args)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    print(f"model written to {args.out}")


def report(args):
    from openpronounce import gop

    model = gop.load_missing_model(args.lang)
    if model is None:
        raise SystemExit("no missing-word model: run --fit --out first")
    X, y, words, extra = cases("test", args.lang)
    probabilities = model.model.predict(X)
    recall = float((probabilities[y == 1] >= model.threshold).mean())
    false_alarms = float((probabilities[y == 0] >= model.threshold).mean())
    flagged, missing, bad = words[:, 0], words[:, 1], words[:, 2]
    hit, n_phones = np.array(extra["hit"]), np.array(extra["phones"])

    def precision_recall(mask):
        return float((mask & bad).sum() / max(1, mask.sum())), float((mask & bad).sum() / max(1, bad.sum()))

    before, after = precision_recall(flagged), precision_recall(flagged | missing)
    print(f"missing words: {int(y.sum())} words added to the text, {recall:.3f} found; "
          f"{false_alarms * 100:.2f} % of the {int((y == 0).sum())} words really read are called missing")
    print(f"  of those {int(missing.sum())} words, {int((missing & bad).sum())} were rated below {HUMAN_BAD_WORD} by the raters")
    print(f"  words reported, without / with the missing ones: precision {before[0]:.3f} / {after[0]:.3f}, "
          f"recall {before[1]:.3f} / {after[1]:.3f}")
    print(f"extra speech: {len(hit)} words removed from the text, {hit.mean():.3f} found "
          f"({hit[n_phones >= 3].mean():.3f} of the {int((n_phones >= 3).sum())} with 3 phones or more)")
    print(f"  on the untouched sentences: {extra['false_boundaries']} stretches reported in {extra['false_sentences']} "
          f"of {extra['sentences']} sentences ({extra['false_sentences'] / extra['sentences'] * 100:.1f} %)")
    return {"missing_recall": round(recall, 4), "missing_false_alarm_rate": round(false_alarms, 4),
            "word_precision_recall": [round(v, 4) for v in after],
            "extra_recall": round(float(hit.mean()), 4), "extra_recall_3_phones": round(float(hit[n_phones >= 3].mean()), 4),
            "extra_false_sentence_rate": round(extra["false_sentences"] / extra["sentences"], 4)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fit", action="store_true")
    mode.add_argument("--report", action="store_true")
    parser.add_argument("--out", help="where --fit writes the model")
    parser.add_argument("--lang", default="en")
    args = parser.parse_args(argv)
    if args.fit:
        if not args.out:
            parser.error("--fit needs --out")
        fit(args)
    else:
        report(args)


if __name__ == "__main__":
    main()
