"""Utterance-level scoring experiments on speechocean762 (human total / accuracy / fluency / prosodic).

    python benchmarks/scorer.py --features train      # utterance features (needs logits_<split>, see word_model.py)
    python benchmarks/scorer.py --features test
    python benchmarks/scorer.py --fit --out openpronounce/models/scorer_en.json
    python benchmarks/scorer.py --report              # shipped scorer on the test split

``--generic`` on every step builds the scorer of the other languages (``scorer_generic.json``):
English is read through the generic word model, and the tables go to ``utt_generic_<split>.npz``.
Run ``native_calibration.py --reference`` afterwards, a new file has no native reference.

``--wavlm <split>`` caches WavLM-base-plus pooled layers and ``--fit --wavlm-layers`` adds
them to the comparison (they are not used by the shipped scorer).
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
from speechocean762 import DATASET_DIR, load_rows  # noqa: E402
from word_model import english_on_generic_models, load_cache, load_posteriors, logits_dir  # noqa: E402

logger = logging.getLogger("scorer")
TARGETS = ("total", "accuracy", "fluency", "prosodic")
GENERIC = False  # set by --generic


def table_path(split):
    return os.path.join(DATASET_DIR, f"utt_{'generic_' if GENERIC else ''}{split}.npz")


def wavlm_dir(split):
    return os.path.join(DATASET_DIR, f"wavlm_{split}")


def extract_wavlm(split, threads, layers=(6, 9, 12)):
    """Cache WavLM-base-plus mean/std pooled hidden states of every layer (and frames of ``layers``)."""
    import torch
    from transformers import Wav2Vec2FeatureExtractor, WavLMModel

    from openpronounce import audio

    torch.set_num_threads(threads)
    out = wavlm_dir(split)
    os.makedirs(out, exist_ok=True)
    extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
    model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").eval()
    rows = sorted(load_rows(split), key=lambda r: r["utt"])
    pooled = {}
    t0 = time.time()
    for k, row in enumerate(rows, 1):
        path = os.path.join(out, row["utt"] + ".npz")
        if os.path.exists(path):
            pooled[row["utt"]] = np.load(path)["pooled"]
            continue
        sound = audio.load(io.BytesIO(row["audio"]["bytes"]))
        inputs = extractor(sound, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            hidden = model(inputs.input_values, output_hidden_states=True).hidden_states
        hidden = torch.stack([h[0] for h in hidden]).numpy()
        pool = np.concatenate([hidden.mean(axis=1), hidden.std(axis=1)], axis=1).astype(np.float32)
        np.savez(path, pooled=pool, **{f"l{l}": hidden[l].astype(np.float16) for l in layers})
        pooled[row["utt"]] = pool
        if k % 100 == 0:
            logger.info("[%d/%d] %.0fs", k, len(rows), time.time() - t0)
    utts = sorted(pooled)
    np.savez(os.path.join(out, "pooled_all.npz"), utts=np.array(utts), pooled=np.stack([pooled[u] for u in utts]))


def _negatives(labels, every=5, truncate_every=10, seed=0):
    """Synthetic negatives: ``(utt, kind, text, frame_fraction, targets)`` for a subset of ``labels``.

    ``wrong``: the posteriors of one utterance against the text of another one (all ratings
    1). ``truncated``: the first 60 % of the frames against the full text (ratings 0.4x).
    speechocean762 has neither case, and tree models cannot extrapolate to them.
    """
    import random

    rng = random.Random(seed)
    out = []
    for k, lab in enumerate(labels):
        if k % every == 0:
            other = rng.choice(labels)
            while other["utt"] == lab["utt"] or other["text"].lower() == lab["text"].lower():
                other = rng.choice(labels)
            out.append((lab["utt"], "wrong", other["text"], 1.0, {t: 1.0 for t in TARGETS}))
        if k % truncate_every == 0 and len(lab["words"]) >= 4:
            out.append((lab["utt"], "truncated", lab["text"], 0.6, {t: 0.4 * lab[t] for t in TARGETS}))
    return out


def features(split, lang="en"):
    """One row per utterance: every ``openpronounce.scorer.FEATURES`` entry plus the human scores.

    Writes ``utt_<split>.npz`` (real readings) and ``utt_<split>_neg.npz`` (synthetic negatives, see
    :func:`_negatives`, column ``kind``).
    """
    from openpronounce import audio, gop, phones, scorer

    labels, vocab = load_cache(split)
    audio_bytes = {r["utt"]: r["audio"]["bytes"] for r in load_rows(split)}
    model = gop.load_model(lang)
    sounds = {}

    def sound_of(utt):
        if utt not in sounds:
            sounds[utt] = audio.load(io.BytesIO(audio_bytes[utt]))
        return sounds[utt]

    def row_for(utt, text, fraction):
        log_posteriors, hidden = load_posteriors(split, utt)
        sound = sound_of(utt)
        if fraction < 1.0:
            log_posteriors = log_posteriors[: int(len(log_posteriors) * fraction)]
            hidden = hidden[: len(log_posteriors)] if hidden is not None else None
            sound = sound[: int(len(sound) * fraction)]
        words, groups = phones.get_expected_phones(text, lang, normalize=False)
        analysis = gop.analyze(log_posteriors, vocab, groups, words, lang, hidden=hidden)
        if analysis is None:
            return None
        phone_p, word_p = model.predict(analysis)
        return scorer.features(analysis, phone_p, word_p, model.threshold, sound)

    rows = []
    t0 = time.time()
    for k, lab in enumerate(labels, 1):
        feats = row_for(lab["utt"], lab["text"], 1.0)
        if feats is None:
            continue
        rows.append({"utt": lab["utt"], "speaker": lab["speaker"], **{t: lab[t] for t in TARGETS},
                     "completeness": lab.get("completeness", -1), **feats})
        if k % 500 == 0:
            logger.info("[%d/%d] %.0fs", k, len(labels), time.time() - t0)
    np.savez(table_path(split), **{key: np.array([r[key] for r in rows]) for key in rows[0]})
    logger.info("%d utterances -> %s", len(rows), table_path(split))

    negatives = []
    for utt, kind, text, fraction, targets in _negatives(labels):
        feats = row_for(utt, text, fraction)
        if feats is not None:
            negatives.append({"utt": utt, "kind": kind, **targets, **feats})
    path = table_path(split).replace(".npz", "_neg.npz")
    np.savez(path, **{key: np.array([r[key] for r in negatives]) for key in negatives[0]})
    logger.info("%d synthetic negatives -> %s", len(negatives), path)


def _load(split):
    table = dict(np.load(table_path(split), allow_pickle=True))
    path = table_path(split).replace(".npz", "_neg.npz")
    table["_negatives"] = dict(np.load(path, allow_pickle=True)) if os.path.exists(path) else None
    wavlm = None
    path = os.path.join(wavlm_dir(split), "pooled_all.npz")
    if os.path.exists(path):
        d = np.load(path)
        index = {u: i for i, u in enumerate(d["utts"].tolist())}
        wavlm = d["pooled"][[index[u] for u in table["utt"].tolist()]]
    return table, wavlm


def _numeric(table):
    from openpronounce import scorer

    names = list(scorer.FEATURES)
    return names, np.column_stack([table[k].astype(float) for k in names])


def _ridge(X_train, y_train, X_test):
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 16)))
    model.fit(X_train, y_train)
    return model.predict(X_test)


def _gbm_model(X_train, y_train):
    from sklearn.ensemble import HistGradientBoostingRegressor

    model = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=20,
                                          l2_regularization=1.0, random_state=0)
    return model.fit(X_train, y_train)


def _gbm(X_train, y_train, X_test):
    return _gbm_model(X_train, y_train).predict(X_test)


def _corr(y, pred):
    from scipy.stats import pearsonr, spearmanr

    return pearsonr(y, pred)[0], spearmanr(y, pred)[0]


def _stack(table, negatives, names):
    X = np.column_stack([table[k].astype(float) for k in names])
    if negatives is None:
        return X, {t: table[t].astype(float) for t in TARGETS}, np.ones(len(X), dtype=bool)
    Xn = np.column_stack([negatives[k].astype(float) for k in names])
    y = {t: np.concatenate([table[t].astype(float), negatives[t].astype(float)]) for t in TARGETS}
    return np.vstack([X, Xn]), y, np.concatenate([np.ones(len(X), dtype=bool), np.zeros(len(Xn), dtype=bool)])


def _recalibration(X, y, folds=5, seed=0):
    """``(slope, intercept)`` mapping out-of-fold GBM predictions back onto the rating scale (trees shrink the extremes)."""
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(X))
    oof = np.zeros(len(X))
    for part in np.array_split(order, folds):
        mask = np.ones(len(X), dtype=bool)
        mask[part] = False
        oof[part] = _gbm_model(X[mask], y[mask]).predict(X[part])
    slope, intercept = np.polyfit(oof, y, 1)
    return float(slope), float(intercept)


def fit(args):
    from sklearn.metrics import roc_auc_score

    from openpronounce import gop, scorer
    from word_model import _fit_logistic, export_trees

    train, w_train = _load("train")
    test, w_test = _load("test")
    names, X_test = _numeric(test)
    X_train_pos = _numeric(train)[1]
    X_train, y_train, real_train = _stack(train, train["_negatives"], names)
    print(f"train {real_train.sum()} utterances + {(~real_train).sum()} synthetic negatives, test {len(X_test)}, "
          f"{len(names)} features")
    alignment_only = [i for i, n in enumerate(names) if n in gop.UTTERANCE_FEATURES]
    y_pos = {t: train[t].astype(float) for t in TARGETS}
    systems = [("ridge alignment features", _ridge, X_train_pos[:, alignment_only], y_pos, X_test[:, alignment_only]),
               ("GBM alignment features", _gbm, X_train_pos[:, alignment_only], y_pos, X_test[:, alignment_only]),
               ("ridge all features", _ridge, X_train_pos, y_pos, X_test),
               ("GBM all features, real readings only", _gbm, X_train_pos, y_pos, X_test),
               ("GBM all features + negatives (shipped)", _gbm, X_train, y_train, X_test)]
    if w_train is not None and args.wavlm_layers:
        all_layers_train = w_train.reshape(len(w_train), -1)
        all_layers_test = w_test.reshape(len(w_test), -1)
        systems += [
            ("ridge wavlm all layers", _ridge, all_layers_train, y_pos, all_layers_test),
            ("ridge all features + wavlm all layers", _ridge, np.column_stack([X_train_pos, all_layers_train]), y_pos,
             np.column_stack([X_test, all_layers_test])),
        ]
    print(f"{'system':44s} " + " ".join(f"{t:>16s}" for t in TARGETS))
    metrics = {}
    for name, method, A, y, B in systems:
        cells = []
        for target in TARGETS:
            pred = method(A, y[target], B)
            p, sp = _corr(test[target].astype(float), pred)
            cells.append(f"{p:.3f} / {sp:.3f}")
            if name.endswith("(shipped)"):
                metrics[target] = {"pearson": round(float(p), 4), "spearman": round(float(sp), 4)}
        print(f"{name:44s} " + " ".join(f"{c:>16s}" for c in cells))

    # sentence match: real reading (1) vs wrong sentence / truncated reading (0), alignment features only
    match = None
    if train["_negatives"] is not None:
        match = _fit_logistic(X_train[:, alignment_only], real_train, [names[i] for i in alignment_only], C=1.0)
        if test["_negatives"] is not None:
            Xt, _, real_test = _stack(test, test["_negatives"], names)
            p = match.predict(Xt[:, alignment_only])
            kinds = test["_negatives"]["kind"]
            print(f"sentence match: AUC={roc_auc_score(real_test, p):.3f}, real readings >= 0.5: "
                  f"{np.mean(p[real_test] >= 0.5):.3f}, wrong sentence < 0.5: "
                  f"{np.mean(p[~real_test][kinds == 'wrong'] < 0.5):.3f}, truncated < 0.5: "
                  f"{np.mean(p[~real_test][kinds == 'truncated'] < 0.5):.3f}")
            metrics["sentence_match_auc"] = round(float(roc_auc_score(real_test, p)), 4)

    if args.out:
        models, calibration = {}, {}
        for target in TARGETS:
            models[target] = export_trees(_gbm_model(X_train, y_train[target]), names)
            calibration[target] = _recalibration(X_train, y_train[target])
        data = {"lang": "generic" if GENERIC else "en", "features": names, "models": models, "calibration": calibration,
                "trained_on": f"speechocean762 train split, {int(real_train.sum())} utterances"
                              f" + {int((~real_train).sum())} synthetic negatives"
                              + (", features from the generic word model" if GENERIC else ""),
                "metrics": {"test_split": "speechocean762 test, 2500 utterances", **metrics}}
        if match is not None:
            data["sentence_match"] = match.to_dict()
        model = scorer.Scorer(data)
        model.save(args.out)
        print(f"model written to {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)")
        report(args, model)


def report(args, model=None):
    from openpronounce import scorer

    model = model or scorer.load_scorer(args.lang)
    test, _ = _load("test")
    predictions = model.predict_matrix(np.column_stack([test[name].astype(float) for name in model.features]))
    print(f"{'target':10s} {'Pearson':>8s} {'Spearman':>9s} {'MAE':>6s}  (shipped scorer, after calibration)")
    for target in TARGETS:
        pred = predictions[target]
        p, sp = _corr(test[target].astype(float), pred)
        print(f"{target:10s} {p:8.3f} {sp:9.3f} {np.abs(pred - test[target].astype(float)).mean():6.3f}")
    print("predicted total quantiles 5/25/50/75/95 %:", np.quantile(predictions["total"], [.05, .25, .5, .75, .95]).round(2),
          " human:", np.quantile(test["total"].astype(float), [.05, .25, .5, .75, .95]))
    negatives = test["_negatives"]
    if negatives is not None:
        Xn = np.column_stack([negatives[name].astype(float) for name in model.features])
        pred = model.predict_matrix(Xn)["total"]
        for kind in ("wrong", "truncated"):
            sel = negatives["kind"] == kind
            print(f"synthetic {kind:9s} ({sel.sum()}): predicted total mean {pred[sel].mean():.2f}, "
                  f"90th percentile {np.quantile(pred[sel], 0.9):.2f}")
        if model.sentence_match is not None:
            p = model.match_probability(Xn)
            print(f"sentence match on negatives < 0.5: {np.mean(p < 0.5):.3f}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--features", metavar="SPLIT")
    mode.add_argument("--wavlm", metavar="SPLIT", help="cache WavLM pooled features of a split")
    mode.add_argument("--fit", action="store_true")
    mode.add_argument("--report", action="store_true", help="shipped scorer on the test split")
    parser.add_argument("--out", help="where --fit writes the scorer (default: not written)")
    parser.add_argument("--wavlm-layers", action="store_true", help="also try WavLM pooled layers (needs --wavlm caches)")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--generic", action="store_true", help="the scorer of the other languages: English through the generic word model")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    logging.getLogger("phonemizer").setLevel(logging.ERROR)
    if args.generic:
        global GENERIC
        GENERIC = True
        english_on_generic_models()
    if args.features:
        features(args.features, args.lang)
    elif args.wavlm:
        extract_wavlm(args.wavlm, args.threads)
    elif args.report:
        report(args)
    else:
        fit(args)


if __name__ == "__main__":
    main()
