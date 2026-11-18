"""Free speech on speechocean762: the prompt is hidden, the automatic transcript replaces it.

    python benchmarks/free.py --transcribe train   # word transcripts without the prompt (asr_<split>.json)
    python benchmarks/free.py --transcribe test
    python benchmarks/free.py --features train     # needs logits_<split>, see word_model.py --extract
    python benchmarks/free.py --features test
    python benchmarks/free.py --fit --out openpronounce/models/free_scorer_en.json
    python benchmarks/free.py --report             # shipped free-speech scorer on the test split

The English scorer blends its trees with a linear head on the hidden states of the phone
recognizer, pooled over the utterance (``scorer.EmbeddingHead``); the weight of the head is
chosen on out-of-fold predictions of the train split, speakers kept apart.

``--generic`` on the last four builds the scorer of the other languages
(``free_scorer_generic.json``): English is read through the generic word model, and the
tables go to ``free_generic_<split>.npz``. Run ``native_calibration.py --reference``
afterwards, a new file has no native reference.

speechocean762 is read speech in short sentences: this measures what is lost when the
text is unknown, not how the scorer behaves on long spontaneous speech, for which no
rated corpus can be used here.
"""

import argparse
import io
import json
import logging
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from groups import AGE_GROUPS, speaker_metadata  # noqa: E402
from speechocean762 import DATASET_DIR, load_rows  # noqa: E402
from word_model import english_on_generic_models, export_trees, load_cache, load_posteriors  # noqa: E402

logger = logging.getLogger("free")
TARGETS = ("total", "accuracy", "fluency", "prosodic")
FRAME = 320  # samples per frame of the recognizer (20 ms at 16 kHz)
GENERIC = False  # set by --generic


def transcripts_path(split):
    return os.path.join(DATASET_DIR, f"asr_{split}.json")


def table_path(split):
    return os.path.join(DATASET_DIR, f"free_{'generic_' if GENERIC else ''}{split}.npz")


def transcribe(split, threads):
    """Transcribe every utterance of ``split`` with the word model of the library. Resumable."""
    import torch

    from openpronounce import audio, speech

    torch.set_num_threads(threads)
    path = transcripts_path(split)
    done = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            done = json.load(f)

    def save():
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            json.dump(done, f)
        os.replace(path + ".tmp", path)

    rows = sorted(load_rows(split), key=lambda r: r["utt"])
    t0 = time.time()
    for k, row in enumerate(rows, 1):
        if row["utt"] in done:
            continue
        done[row["utt"]] = speech.transcribe(audio.load(io.BytesIO(row["audio"]["bytes"])), "en")
        if k % 100 == 0:
            save()
            logger.info("[%d/%d] %.0fs", k, len(rows), time.time() - t0)
    save()


def features(split, lang="en"):
    """One row per utterance the free pipeline can score: ``scorer.FREE_FEATURES``, human scores, WER against the prompt."""
    from rapidfuzz.distance import Levenshtein

    from openpronounce import audio, free, g2p, phones, scorer

    labels, vocab = load_cache(split)
    with open(transcripts_path(split), encoding="utf-8") as f:
        transcripts = json.load(f)
    audio_bytes = {r["utt"]: r["audio"]["bytes"] for r in load_rows(split)}
    rows, skipped = [], []
    t0 = time.time()
    for k, lab in enumerate(labels, 1):
        transcript = " ".join(g2p.words_of(transcripts[lab["utt"]]))
        result = None
        if transcript:
            log_posteriors, hidden = load_posteriors(split, lab["utt"])
            sound = audio.load(io.BytesIO(audio_bytes[lab["utt"]]))
            # Same silence trimming as free.assess, which only ever sees the spans of split_speech
            spans = audio.split_speech(sound)
            if spans:
                first, last = spans[0][0] // FRAME, -(-spans[-1][1] // FRAME)
                sound = sound[first * FRAME:last * FRAME]
                log_posteriors, hidden = log_posteriors[first:last], None if hidden is None else hidden[first:last]
            recognition = phones.decode_ctc(log_posteriors, vocab, 0, lang, True, hidden)
            result = free.segment_features(sound, transcript, recognition, lang)
        if result is None:
            skipped.append(lab["utt"])
            continue
        prompt = list(g2p.words_of(lab["text"]))
        rows.append({"utt": lab["utt"], "speaker": lab["speaker"], **{t: lab[t] for t in TARGETS},
                     "wer": Levenshtein.distance(prompt, transcript.split()) / max(1, len(prompt)),
                     "n_prompt_words": len(prompt), **{name: result[0][name] for name in scorer.FREE_FEATURES},
                     **({"embedding": result[0][scorer.EMBEDDING]} if scorer.EMBEDDING in result[0] else {})})
        if k % 500 == 0:
            logger.info("[%d/%d] %.0fs", k, len(labels), time.time() - t0)
    table = {key: np.array([r[key] for r in rows]) for key in rows[0]}
    table["skipped"] = np.array(skipped)
    np.savez(table_path(split), **table)
    logger.info("%d utterances, %d without a usable transcript -> %s", len(rows), len(skipped), table_path(split))


def _load(split):
    return dict(np.load(table_path(split), allow_pickle=True))


def _matrix(table, names):
    return np.column_stack([table[name].astype(float) for name in names])


def _gbm(X, y):
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=20,
                                         l2_regularization=1.0, random_state=0).fit(X, y)


def _recalibration(X, y, folds=5, seed=0):
    order = np.random.RandomState(seed).permutation(len(X))
    oof = np.zeros(len(X))
    for part in np.array_split(order, folds):
        mask = np.ones(len(X), dtype=bool)
        mask[part] = False
        oof[part] = _gbm(X[mask], y[mask]).predict(X[part])
    slope, intercept = np.polyfit(oof, y, 1)
    return float(slope), float(intercept)


SHARES = (0.0, 0.25, 0.5, 0.75, 1.0)
ALPHAS = np.logspace(1, 5, 9)


def _ridge(E, y):
    """``(center, scale, weights, intercept)`` of a ridge regression on standardized embeddings."""
    from sklearn.linear_model import RidgeCV

    center, scale = E.mean(axis=0), E.std(axis=0) + 1e-6
    model = RidgeCV(alphas=ALPHAS).fit((E - center) / scale, y)
    return center, scale, model.coef_, float(model.intercept_)


def _ridge_predict(fitted, E):
    center, scale, weights, intercept = fitted
    return ((E - center) / scale) @ weights + intercept


def _out_of_fold(X, E, y, speakers, folds=5, seed=0):
    """Out-of-fold predictions of the trees and of the embedding head, no speaker on both sides of a fold."""
    names = np.unique(speakers)
    np.random.RandomState(seed).shuffle(names)
    trees, head = np.zeros(len(y)), np.zeros(len(y))
    for part in np.array_split(names, folds):
        held = np.isin(speakers, part)
        trees[held] = _gbm(X[~held], y[~held]).predict(X[held])
        head[held] = _ridge_predict(_ridge(E[~held], y[~held]), E[held])
    return trees, head


def _embedding_head(X, E, targets, speakers, layer):
    """The ``embedding`` entry of a scorer file: one ridge per target, the share of the blend and its calibration."""
    oof = {target: _out_of_fold(X, E, y, speakers) for target, y in targets.items()}
    agreement = {share: np.mean([_pearson(y, (1 - share) * oof[t][0] + share * oof[t][1]) for t, y in targets.items()])
                 for share in SHARES}
    share = max(agreement, key=agreement.get)
    print("out-of-fold Pearson on train, mean of the four scores, by share of the embedding head: "
          + ", ".join(f"{s:g}: {a:.3f}" for s, a in agreement.items()) + f" -> {share:g}")
    fitted = {target: _ridge(E, y) for target, y in targets.items()}
    center, scale = fitted["total"][0], fitted["total"][1]
    calibration = {}
    for target, y in targets.items():
        slope, intercept = np.polyfit((1 - share) * oof[target][0] + share * oof[target][1], y, 1)
        calibration[target] = (float(slope), float(intercept))
    return {"layer": layer, "pooling": "mean and standard deviation over the frames", "share": share,
            "center": [round(float(v), 5) for v in center], "scale": [round(float(v), 5) for v in scale],
            "weights": {t: [round(float(v), 7) for v in fitted[t][2]] for t in targets},
            "intercept": {t: round(fitted[t][3], 5) for t in targets}, "calibration": calibration}


def _pearson(y, p):
    return float(np.corrcoef(y, p)[0, 1]) if len(y) > 2 and np.std(y) > 0 and np.std(p) > 0 else float("nan")


def _breakdown(test, predictions):
    """Pearson with the human total for everyone, by age group and by sentence length."""
    metadata = speaker_metadata()
    ages = np.array([metadata[str(s)][1] for s in test["speaker"].tolist()])
    y, p, n = test["total"].astype(float), predictions["total"], test["n_prompt_words"]
    masks = [("all", np.ones(len(y), dtype=bool))]
    masks += [(name, (ages >= low) & (ages <= high)) for name, low, high in AGE_GROUPS]
    masks += [("1 to 4 words", n <= 4), ("5 to 8 words", (n >= 5) & (n <= 8)), ("9 words and more", n >= 9)]
    rows = {}
    for name, mask in masks:
        rows[name] = {"utterances": int(mask.sum()), "pearson": round(_pearson(y[mask], p[mask]), 4),
                      "mae": round(float(np.abs(y[mask] - p[mask]).mean()), 3), "wer": round(float(test["wer"][mask].mean()), 3)}
        print(f"  {name:26s} n={rows[name]['utterances']:5d}  total r={rows[name]['pearson']:.3f}  MAE={rows[name]['mae']:.2f}"
              f"  WER against the prompt {rows[name]['wer']:.2f}")
    return rows


def fit(args):
    from openpronounce import gop, scorer

    train, test = _load("train"), _load("test")
    # the generic word model keeps no hidden states at run time, so its scorer has no head to feed
    with_head = not GENERIC and "embedding" in train and gop.hidden_layer(args.lang) is not None
    print(f"train {len(train['utt'])} utterances ({len(train['skipped'])} skipped), "
          f"test {len(test['utt'])} ({len(test['skipped'])} skipped)")
    text_free = scorer.PROSODY_FEATURES + [n for n in scorer.RECOGNITION_FEATURES if n != "transcript_phone_error_rate"]
    systems = [("alignment on the transcript (as for read speech)", list(scorer.FEATURES)),
               ("text-free features only", text_free),
               ("all free-speech features (shipped)", list(scorer.FREE_FEATURES))]
    print(f"{'system':50s} " + " ".join(f"{t:>10s}" for t in TARGETS))
    constrained = scorer.load_scorer(args.lang)
    if constrained is not None:
        zero_shot = constrained.predict_matrix(_matrix(test, constrained.features))
        print(f"{'read-speech scorer applied as is':50s} "
              + " ".join(f"{_pearson(test[t].astype(float), zero_shot[t]):10.3f}" for t in TARGETS))
    metrics = {}
    for name, names in systems:
        cells = []
        for target in TARGETS:
            pred = _gbm(_matrix(train, names), train[target].astype(float)).predict(_matrix(test, names))
            cells.append(_pearson(test[target].astype(float), pred))
            if name.endswith("(shipped)"):
                metrics[target] = {"pearson": round(cells[-1], 4)}
        print(f"{name:50s} " + " ".join(f"{c:10.3f}" for c in cells))
    if args.out:
        names = list(scorer.FREE_FEATURES)
        X = _matrix(train, names)
        data = {"lang": "generic" if GENERIC else args.lang, "features": names,
                "models": {t: export_trees(_gbm(X, train[t].astype(float)), names) for t in TARGETS},
                "calibration": {t: _recalibration(X, train[t].astype(float)) for t in TARGETS},
                "trained_on": f"speechocean762 train split, {len(X)} utterances, prompt hidden (wav2vec2-large-960h transcripts)",
                "metrics": {"test_split": f"speechocean762 test, {len(test['utt'])} utterances, prompt hidden", **metrics}}
        embeddings = None
        if with_head:
            data["metrics"]["trees_alone"] = metrics
            data["embedding"] = _embedding_head(X, np.stack(train["embedding"]).astype(np.float64),
                                                {t: train[t].astype(float) for t in TARGETS}, train["speaker"].astype(str),
                                                gop.hidden_layer(args.lang))
            embeddings = np.stack(test["embedding"]).astype(np.float64)
        model = scorer.Scorer(data)
        predictions = model.predict_matrix(_matrix(test, names), embeddings)
        if with_head:
            for target in TARGETS:
                data["metrics"][target] = {"pearson": round(_pearson(test[target].astype(float), predictions[target]), 4)}
            print(f"{'trees + hidden states of the recognizer (shipped)':50s} "
                  + " ".join(f"{data['metrics'][t]['pearson']:10.3f}" for t in TARGETS))
        data["metrics"]["breakdown"] = _breakdown(test, predictions)
        model.save(args.out)
        print(f"model written to {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)")


def report(args):
    from openpronounce import scorer

    model = scorer.load_scorer(args.lang, "free_scorer")
    if model is None:
        raise SystemExit(f"no free-speech scorer for {args.lang!r}: run --fit --out first")
    test = _load("test")
    embeddings = np.stack(test["embedding"]).astype(np.float64) if model.embedding is not None and "embedding" in test else None
    predictions = model.predict_matrix(_matrix(test, model.features), embeddings)
    print(f"{len(test['utt'])} utterances scored, {len(test['skipped'])} without a usable transcript")
    for target in TARGETS:
        y = test[target].astype(float)
        print(f"{target:10s} Pearson {_pearson(y, predictions[target]):.3f}  MAE {np.abs(y - predictions[target]).mean():.3f}")
    _breakdown(test, predictions)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--transcribe", metavar="SPLIT")
    mode.add_argument("--features", metavar="SPLIT")
    mode.add_argument("--fit", action="store_true")
    mode.add_argument("--report", action="store_true")
    parser.add_argument("--out", help="where --fit writes the scorer (default: not written)")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--generic", action="store_true", help="the scorer of the other languages: English through the generic word model")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    if args.generic:
        global GENERIC
        GENERIC = True
        english_on_generic_models()
    if args.transcribe:
        transcribe(args.transcribe, args.threads)
    elif args.features:
        features(args.features, args.lang)
    elif args.fit:
        fit(args)
    else:
        report(args)


if __name__ == "__main__":
    main()
