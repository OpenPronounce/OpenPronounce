"""Word model and utterance scorer on the speechocean762 caches of one variant.

    .venv-exp/bin/python experiments/browser/eval_variant.py --variant int8            # shipped models, no refit
    .venv-exp/bin/python experiments/browser/eval_variant.py --variant int8 --refit    # refit on <variant> train features

``--variant torch`` reads the original posteriors (~/.cache/openpronounce/speechocean762, read-only)
and writes its tables under /mnt/t7/openpronounce-browser/so762/torch-current (or torch-refit).
The refit writes word_mdd_en.json / scorer_en.json under so762/<variant>/models and points
$OPENPRONOUNCE_MODELS_DIR at it before computing the scorer features.
"""

import argparse
import logging
import os
import types

import common


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True)
    parser.add_argument("--refit", action="store_true")
    parser.add_argument("--skip-gop", action="store_true", help="reuse gop_<split>.npz already computed")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("phonemizer").setLevel(logging.ERROR)

    if args.variant == "torch":
        # read the original posteriors, never write into the original cache: tables go to so762/torch-<mode>
        name = "torch-refit" if args.refit else "torch-current"
        common.use_variant(name)
        import scorer as bench_scorer
        import word_model
        word_model.logits_dir = lambda split: os.path.join(common.ORIGINAL, f"logits_{split}")
        bench_scorer.logits_dir = word_model.logits_dir
        d = common.variant_dir(name)
    else:
        d = common.use_variant(args.variant)

    import scorer as bench_scorer
    import word_model

    splits = ("train", "test") if args.refit else ("test",)
    if not args.skip_gop:
        for split in splits:
            word_model.features(split, "en")

    if not args.refit:
        print("=== word model (shipped), test ===", flush=True)
        word_model.report(word_model._load("test"), "en")
        bench_scorer.features("test", "en")
        print("=== utterance scorer (shipped), test ===", flush=True)
        bench_scorer.report(types.SimpleNamespace(lang="en"))
        return

    models = os.path.join(d, "models")
    os.makedirs(models, exist_ok=True)
    print("=== word model refit on train, test ===", flush=True)
    word_model.fit(word_model._load("train"), word_model._load("test"), os.path.join(models, "word_mdd_en.json"))
    from openpronounce import gop, scorer
    os.environ[gop.MODELS_ENV] = models
    gop.load_model.cache_clear()
    scorer.load_scorer.cache_clear()
    for split in ("train", "test"):
        bench_scorer.features(split, "en")
    print("=== utterance scorer refit on train, test ===", flush=True)
    bench_scorer.fit(types.SimpleNamespace(lang="en", out=os.path.join(models, "scorer_en.json"), wavlm_layers=False))


if __name__ == "__main__":
    main()
