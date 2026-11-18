"""Train and evaluate the word mispronunciation model (``openpronounce/models/word_mdd_<lang>.json``).

    # 1. cache the phone posteriors and the hidden states of one recognizer layer for both
    #    speechocean762 splits (~30 min and 1.4 GB each on CPU)
    python benchmarks/word_model.py --extract train
    python benchmarks/word_model.py --extract test

    # 2. force-align the expected phones and compute the GOP features (~4 min per split)
    python benchmarks/word_model.py --features train
    python benchmarks/word_model.py --features test

    # 3. fit on the train split, report on the test split, write the model
    python benchmarks/word_model.py --fit --out openpronounce/models/word_mdd_en.json

    # 4. report the shipped model on the test split
    python benchmarks/word_model.py --report

Everything is cached under ``~/.cache/openpronounce/speechocean762/``: ``logits_<split>``
(float16 posteriors ``<utt>.npy``, hidden states ``<utt>.hidden.npy`` with their layer in
``hidden.json``, ``labels.json``, ``vocab.json``) and ``gop_<split>.npz`` (one row per expected
phone, including its pooled ``embedding``). Labels: a phone is wrong when its human accuracy is below 2 (0-2
scale), a word when its accuracy is below 5 (strict) or 7 (lenient), 0-10 scale.
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

logger = logging.getLogger("word_model")
WORD_LABELS = (5, 7)
# Recognizer layer whose hidden states feed the phone stage (0: input of the first transformer layer).
HIDDEN_LAYER = 20
# Dimensions kept of the phone embeddings (PCA fitted on the train split).
EMBEDDING_COMPONENTS = 16


def logits_dir(split):
    return os.path.join(DATASET_DIR, f"logits_{split}")


def features_path(split):
    return os.path.join(DATASET_DIR, f"gop_{split}.npz")


# ---------------------------------------------------------------------------
# Caches
# ---------------------------------------------------------------------------

def extract(split, threads, hidden_layer=HIDDEN_LAYER):
    import torch

    torch.set_num_threads(threads)
    from openpronounce import audio, phones

    rows = sorted(load_rows(split), key=lambda r: r["utt"])
    out = logits_dir(split)
    os.makedirs(out, exist_ok=True)
    labels = [{
        "utt": r["utt"], "speaker": r["speaker"], "text": r["text"], "total": r["total"], "accuracy": r["accuracy"],
        "fluency": r["fluency"], "prosodic": r["prosodic"], "completeness": r["completeness"],
        "words": [{"text": w["text"], "accuracy": w["accuracy"], "phones": w["phones"],
                   "phones_accuracy": w["phones-accuracy"]} for w in r["words"]],
    } for r in rows]
    with open(os.path.join(out, "labels.json"), "w") as f:
        json.dump(labels, f)
    with open(os.path.join(out, "vocab.json"), "w") as f:
        json.dump(list(phones.phone_vocab()), f)
    with open(os.path.join(out, "hidden.json"), "w") as f:
        json.dump({"layer": hidden_layer}, f)
    t0 = time.time()
    for k, row in enumerate(rows, 1):
        path = os.path.join(out, row["utt"] + ".npy")
        hidden_path = os.path.join(out, row["utt"] + ".hidden.npy")
        if os.path.exists(path) and os.path.exists(hidden_path):
            continue
        sound = audio.load(io.BytesIO(row["audio"]["bytes"]))
        log_posteriors, hidden = phones.phone_log_posteriors(sound, hidden_layer=hidden_layer)
        np.save(path, log_posteriors.astype(np.float16))
        np.save(hidden_path, hidden.astype(np.float16))
        if k % 100 == 0:
            logger.info("[%d/%d] %.0fs", k, len(rows), time.time() - t0)


GENERIC_KINDS = ("word_mdd", "word_missing", "scorer", "free_scorer")


def english_on_generic_models():
    """Make English run on the generic models, as every other language does.

    The generic scorers have to learn from the features the other languages will give them,
    and their native reference has to be measured the same way. This puts first in
    ``$OPENPRONOUNCE_MODELS_DIR`` a directory where the generic models carry the name of
    English. Call it before openpronounce loads a model. Returns the directory.
    """
    import tempfile

    from openpronounce import gop

    directory = tempfile.mkdtemp(prefix="openpronounce-generic-")
    for kind in GENERIC_KINDS:
        source = gop.model_path(f"{kind}_generic.json")
        if source is not None:
            os.symlink(source, os.path.join(directory, f"{kind}_en.json"))
    os.environ[gop.MODELS_ENV] = os.pathsep.join([directory] + [d for d in os.environ.get(gop.MODELS_ENV, "").split(os.pathsep) if d])
    return directory

def load_cache(split):
    out = logits_dir(split)
    with open(os.path.join(out, "labels.json")) as f:
        labels = json.load(f)
    with open(os.path.join(out, "vocab.json")) as f:
        vocab = tuple(json.load(f))
    return labels, vocab


def cached_hidden_layer(split):
    """Layer of the cached hidden states of ``split``, or ``None`` when only posteriors were cached."""
    path = os.path.join(logits_dir(split), "hidden.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)["layer"]


def load_posteriors(split, utt):
    """``(log_posteriors, hidden)`` of a cached utterance, ``hidden`` being ``None`` when not cached."""
    log_posteriors = np.load(os.path.join(logits_dir(split), utt + ".npy")).astype(np.float32)
    path = os.path.join(logits_dir(split), utt + ".hidden.npy")
    hidden = np.load(path).astype(np.float32) if os.path.exists(path) else None
    return log_posteriors, hidden


def features(split, lang="en", normalize=False):
    """One row per expected phone of the split: GOP features, phone and word labels, utterance scores.

    The expected phones are espeak's own (``normalize=False``), as in the library.
    """
    from openpronounce import gop, phones

    labels, vocab = load_cache(split)
    rows, embeddings, skipped = [], [], 0
    t0 = time.time()
    for k, lab in enumerate(labels, 1):
        log_posteriors, hidden = load_posteriors(split, lab["utt"])
        words, groups = phones.get_expected_phones(lab["text"], lang, normalize=normalize)
        if words != [w["text"].lower() for w in lab["words"]]:
            skipped += 1
            continue
        analysis = gop.analyze(log_posteriors, vocab, groups, words, lang, hidden=hidden)
        if analysis is None:
            skipped += 1
            continue
        for p in analysis["phones"]:
            word = lab["words"][p["word_index"]]
            matched = len(word["phones"]) == len(groups[p["word_index"]])
            if hidden is not None:
                embeddings.append(p["embedding"].astype(np.float16))
            rows.append({
                "utt": lab["utt"], "speaker": lab["speaker"], "word_index": p["word_index"], "phone": p["phone"],
                "word_accuracy": word["accuracy"],
                "phone_accuracy": word["phones_accuracy"][p["position"]] if matched else -1.0,
                "utt_total": lab["total"], "utt_accuracy": lab["accuracy"], "utt_fluency": lab["fluency"],
                "utt_prosodic": lab["prosodic"], "plausible_phone_ratio": analysis["utterance"]["plausible_phone_ratio"],
                **{name: p[name] for name in gop.PHONE_FEATURES},
            })
        if k % 500 == 0:
            logger.info("[%d/%d] %.0fs", k, len(labels), time.time() - t0)
    logger.info("%d phones, %d utterances skipped", len(rows), skipped)
    table = {key: np.array([r[key] for r in rows]) for key in rows[0]}
    if embeddings:
        table["embedding"] = np.stack(embeddings)
        table["embedding_layer"] = np.array(cached_hidden_layer(split))
    np.savez(features_path(split), **table)


# ---------------------------------------------------------------------------
# Fit / report
# ---------------------------------------------------------------------------

def _load(split):
    return dict(np.load(features_path(split), allow_pickle=True))


def _grouped(table, utts):
    """Split ``table`` (dict of arrays) on utterance membership: ``(rows in utts, rows not in utts)``."""
    mask = np.isin(table["utt"], utts)
    return {k: v[mask] for k, v in table.items()}, {k: v[~mask] for k, v in table.items()}


def _phone_matrix(table):
    from openpronounce import gop

    return np.column_stack([table[name].astype(float) for name in gop.PHONE_FEATURES])


def _fit_logistic(matrix, y, features, phones=None, C=0.3):
    """Standardize, fit a logistic regression (one-hot phone identity appended), return a ``LogisticModel``."""
    from sklearn.linear_model import LogisticRegression

    from openpronounce.gop import LogisticModel

    mean, scale = matrix.mean(axis=0), matrix.std(axis=0) + 1e-9
    X = (matrix - mean) / scale
    vocabulary = []
    if phones is not None:
        vocabulary = sorted(set(phones.tolist()))
        onehot = np.zeros((len(phones), len(vocabulary)))
        onehot[np.arange(len(phones)), [vocabulary.index(p) for p in phones]] = 1.0
        X = np.column_stack([X, onehot])
    clf = LogisticRegression(C=C, max_iter=5000).fit(X, y)
    coef = clf.coef_[0]
    n = len(features)
    return LogisticModel({
        "features": list(features), "mean": mean.tolist(), "scale": scale.tolist(), "coef": coef[:n].tolist(),
        "intercept": float(clf.intercept_[0]),
        "phone_bias": {p: float(w) for p, w in zip(vocabulary, coef[n:])},
    })


def export_mlp(clf, features, mean, scale, phones, embedding=None):
    """JSON dict of a fitted ``MLPClassifier`` (ReLU, logistic output) for :class:`gop.MLPModel`.

    ``embedding`` is the optional ``{layer, mean, projection, scale}`` block of the phone embeddings.
    """
    assert clf.activation == "relu" and clf.out_activation_ == "logistic"
    data = {"type": "mlp", "features": list(features), "mean": np.asarray(mean).tolist(),
            "scale": np.asarray(scale).tolist(), "phones": list(phones),
            "weights": [w.tolist() for w in clf.coefs_], "biases": [b.tolist() for b in clf.intercepts_]}
    if embedding is not None:
        data["embedding"] = embedding
    return data


def export_trees(model, features):
    """JSON dict of a fitted ``HistGradientBoostingClassifier`` (binary) or ``Regressor`` for :class:`gop.TreeEnsemble`."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    trees = []
    for (predictor,) in model._predictors:
        nodes = predictor.nodes
        trees.append([[-1 if n["is_leaf"] else int(n["feature_idx"]), float(n["num_threshold"]), int(n["left"]),
                       int(n["right"]), float(n["value"])] for n in nodes])
    return {"type": "trees", "features": list(features), "baseline": float(np.ravel(model._baseline_prediction)[0]),
            "link": "identity" if isinstance(model, HistGradientBoostingRegressor) else "sigmoid", "trees": trees}


def _fit_mlp(matrix, y, features, phones=None, embeddings=None, layer=None, hidden=(64, 32), alpha=1e-3,
             components=EMBEDDING_COMPONENTS):
    """Standardize, one-hot the phone identity, project the embeddings (PCA), fit an MLP; returns the exported dict."""
    from sklearn.decomposition import PCA
    from sklearn.neural_network import MLPClassifier

    mean, scale = matrix.mean(axis=0), matrix.std(axis=0) + 1e-9
    X = (matrix - mean) / scale
    vocabulary = []
    if phones is not None:
        vocabulary = sorted(set(phones.tolist()))
        onehot = np.zeros((len(phones), len(vocabulary)))
        onehot[np.arange(len(phones)), [vocabulary.index(p) for p in phones]] = 1.0
        X = np.column_stack([X, onehot])
    block = None
    if embeddings is not None:
        embeddings = np.asarray(embeddings, dtype=np.float64)
        pca = PCA(components, random_state=0).fit(embeddings)
        projected = pca.transform(embeddings)
        block = {"layer": int(layer), "mean": pca.mean_.tolist(), "projection": pca.components_.T.tolist(),
                 "scale": (projected.std(axis=0) + 1e-9).tolist()}
        X = np.column_stack([X, projected / np.array(block["scale"])])
    clf = MLPClassifier(hidden, alpha=alpha, early_stopping=True, max_iter=300, random_state=0).fit(X, y)
    return export_mlp(clf, features, mean, scale, vocabulary, block)


def _fit_trees(matrix, y, features):
    from sklearn.ensemble import HistGradientBoostingClassifier

    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=30,
                                         l2_regularization=1.0, random_state=0).fit(matrix, y)
    return export_trees(clf, features)


def _word_rows(table, phone_probabilities):
    """Aggregate per word with :func:`gop.word_matrix`; returns ``(matrix, word_accuracy, utt)``."""
    from openpronounce import gop

    keys = np.array([f"{u}|{w}" for u, w in zip(table["utt"], table["word_index"])])
    order = np.argsort(keys, kind="stable")
    matrices, accuracies, utts = [], [], []
    start = 0
    sorted_keys = keys[order]
    while start < len(order):
        end = start
        while end < len(order) and sorted_keys[end] == sorted_keys[start]:
            end += 1
        idx = order[start:end]
        scored = [{name: table[name][i] for name in gop.PHONE_FEATURES + ["word_index"]} for i in idx]
        _, matrix = gop.word_matrix(scored, phone_probabilities[idx])
        matrices.append(matrix[0])
        accuracies.append(table["word_accuracy"][idx[0]])
        utts.append(table["utt"][idx[0]])
        start = end
    return np.array(matrices), np.array(accuracies, dtype=float), np.array(utts)


def _report(name, y, score):
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

    precision, recall, _ = precision_recall_curve(y, score)

    def at(target):
        return max(precision[recall >= target]) if (recall >= target).any() else 0.0

    print(f"  {name:36s} AUC={roc_auc_score(y, score):.3f} AP={average_precision_score(y, score):.3f} "
          f"P@R.5={at(0.5):.3f} P@R.7={at(0.7):.3f} P@R.9={at(0.9):.3f}")


def _best_f1_threshold(y, score):
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(y, score)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-9)
    return float(thresholds[f1.argmax()])


def _embeddings(table, use_embeddings):
    """Phone embeddings of a feature table (float64), or ``None`` when absent or not wanted."""
    if not use_embeddings or "embedding" not in table:
        return None
    return table["embedding"].astype(np.float64)


def fit(train, test, out, phone_stage="mlp", word_stage="trees", with_phone_identity=True, folds=5,
        use_embeddings=True, components=EMBEDDING_COMPONENTS):
    """Fit the phone stage, then the word stage on out-of-fold phone probabilities (``folds=0``: in-sample).

    The phone stage reads the pooled hidden states (``embedding`` column of the tables) when
    present and ``use_embeddings`` is true; only the MLP stage supports them.
    """
    from scipy.stats import pearsonr
    from sklearn.metrics import average_precision_score, roc_auc_score

    from openpronounce import gop

    E_train, E_test = _embeddings(train, use_embeddings), _embeddings(test, use_embeddings)
    if E_train is not None and phone_stage != "mlp":
        raise SystemExit("embeddings need --phone-stage mlp")
    layer = int(train["embedding_layer"]) if E_train is not None else None
    if E_train is not None and (E_test is None or int(test["embedding_layer"]) != layer):
        raise SystemExit("train and test embeddings come from different layers")

    def fit_phone(rows, phones):
        y = train["phone_accuracy"][rows] < 2
        if phone_stage == "mlp":
            embeddings = E_train[rows] if E_train is not None else None
            return gop.MLPModel(_fit_mlp(X_train[rows], y, gop.PHONE_FEATURES, phones, embeddings, layer,
                                         components=components))
        return _fit_logistic(X_train[rows], y, gop.PHONE_FEATURES, phones)

    def predict_phone(model, matrix, phones, embeddings):
        if getattr(model, "hidden_layer", None) is not None:
            return model.predict(matrix, phones, embeddings)
        return model.predict(matrix, phones)

    def fit_word(matrix, y):
        if word_stage == "trees":
            return gop.TreeEnsemble(_fit_trees(matrix, y, gop.WORD_FEATURES))
        return _fit_logistic(matrix, y, gop.WORD_FEATURES)

    def ids(table, rows=slice(None)):
        return table["phone"][rows] if with_phone_identity else None

    matched = train["phone_accuracy"] >= 0
    X_train = _phone_matrix(train)
    phone_model = fit_phone(matched, ids(train, matched))
    # out-of-fold phone probabilities on the train split, so the word model does not learn on overfit inputs
    oof = predict_phone(phone_model, X_train, ids(train), E_train)
    utts = np.unique(train["utt"])
    rng = np.random.RandomState(0)
    rng.shuffle(utts)
    for fold in np.array_split(utts, folds) if folds else []:
        held = np.isin(train["utt"], fold)
        fit_rows = ~held & matched
        model = fit_phone(fit_rows, ids(train, fit_rows))
        oof[held] = predict_phone(model, X_train[held], ids(train, held), E_train[held] if E_train is not None else None)
    Xw_train, acc_train, _ = _word_rows(train, oof)
    word_model = fit_word(Xw_train, acc_train < 5)
    threshold = _best_f1_threshold(acc_train < 5, word_model.predict(Xw_train))

    metrics = {}
    test_matched = test["phone_accuracy"] >= 0
    p_test = predict_phone(phone_model, _phone_matrix(test), ids(test), E_test)
    print(f"phone level (test, {int(test_matched.sum())} phones with a human score):")
    y = test["phone_accuracy"][test_matched] < 2
    _report(f"{phone_stage} p(phone wrong)", y, p_test[test_matched])
    _report("raw gop_margin", y, -test["gop_margin"][test_matched])
    metrics["phone_auc"] = float(roc_auc_score(y, p_test[test_matched]))
    metrics["phone_pearson"] = float(pearsonr(-p_test[test_matched], test["phone_accuracy"][test_matched])[0])
    print(f"  Pearson(-p, phone accuracy)={metrics['phone_pearson']:.3f} "
          f"(raw gop_margin {pearsonr(test['gop_margin'][test_matched], test['phone_accuracy'][test_matched])[0]:.3f})")
    Xw_test, acc_test, _ = _word_rows(test, p_test)
    score = word_model.predict(Xw_test)
    for label in WORD_LABELS:
        y = acc_test < label
        print(f"word level, human accuracy < {label} ({len(y)} words, prevalence {y.mean():.3f}):")
        _report(f"{word_stage} word model", y, score)
        _report("raw min gop_margin", y, -Xw_test[:, gop.WORD_FEATURES.index("gm_min")])
        flagged = score >= threshold
        tp = (flagged & y).sum()
        precision, recall = tp / max(1, flagged.sum()), tp / max(1, y.sum())
        print(f"  at threshold {threshold:.3f}: P={precision:.3f} R={recall:.3f} flagged {flagged.mean():.1%}")
        metrics[f"word_auc_lt{label}"] = float(roc_auc_score(y, score))
        metrics[f"word_ap_lt{label}"] = float(average_precision_score(y, score))
        metrics[f"word_precision_lt{label}"], metrics[f"word_recall_lt{label}"] = float(precision), float(recall)
    model = gop.WordModel({
        "lang": "en" if with_phone_identity else None,
        "phone": phone_model.to_dict(), "word": word_model.to_dict(), "threshold": threshold,
        "trained_on": f"speechocean762 train split, {len(np.unique(train['utt']))} utterances",
        "metrics": {"test_split": "speechocean762 test, 2500 utterances", **metrics},
    })
    if out:
        model.save(out)
        print(f"model written to {out} ({os.path.getsize(out) / 1e6:.1f} MB)")


def report(test, lang):
    from openpronounce import gop

    model = gop.load_model(lang)
    phones = test["phone"] if getattr(model.phone, "phones", None) or getattr(model.phone, "phone_bias", None) else None
    if model.hidden_layer is None:
        p = model.phone.predict(_phone_matrix(test), phones)
    else:
        if "embedding" not in test or int(test["embedding_layer"]) != model.hidden_layer:
            raise SystemExit(f"the shipped model reads layer {model.hidden_layer}: re-run --extract and --features")
        p = model.phone.predict(_phone_matrix(test), phones, test["embedding"].astype(np.float64))
    Xw, acc, _ = _word_rows(test, p)
    score = model.word.predict(Xw)
    for label in WORD_LABELS:
        y = acc < label
        print(f"word level, human accuracy < {label} ({len(y)} words, prevalence {y.mean():.3f}):")
        _report("shipped word model", y, score)
        flagged = score >= model.threshold
        tp = (flagged & y).sum()
        print(f"  at threshold {model.threshold:.3f}: P={tp / max(1, flagged.sum()):.3f} R={tp / max(1, y.sum()):.3f} "
              f"flagged {flagged.mean():.1%}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--extract", metavar="SPLIT", help="cache the phone posteriors of a split")
    mode.add_argument("--features", metavar="SPLIT", help="GOP feature table of a split")
    mode.add_argument("--fit", action="store_true", help="fit on train, report on test")
    mode.add_argument("--report", action="store_true", help="report the shipped model on test")
    parser.add_argument("--out", help="where --fit writes the model (default: not written)")
    parser.add_argument("--generic", action="store_true", help="--fit without phone identity (other languages)")
    parser.add_argument("--phone-stage", default="mlp", choices=["mlp", "logistic"])
    parser.add_argument("--word-stage", default="trees", choices=["trees", "logistic"])
    parser.add_argument("--folds", type=int, default=5, help="out-of-fold phone probabilities for the word stage (0: in-sample)")
    parser.add_argument("--no-embeddings", action="store_true", help="--fit on the GOP features only")
    parser.add_argument("--components", type=int, default=EMBEDDING_COMPONENTS, help="PCA dimensions of the embeddings")
    parser.add_argument("--hidden-layer", type=int, default=HIDDEN_LAYER, help="recognizer layer cached by --extract")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    logging.getLogger("phonemizer").setLevel(logging.ERROR)
    if args.extract:
        extract(args.extract, args.threads, args.hidden_layer)
    elif args.features:
        features(args.features, args.lang)
    elif args.fit:
        fit(_load("train"), _load("test"), args.out, args.phone_stage, args.word_stage, with_phone_identity=not args.generic,
            folds=args.folds, use_embeddings=not args.no_embeddings, components=args.components)
    else:
        report(_load("test"), args.lang)


if __name__ == "__main__":
    main()
