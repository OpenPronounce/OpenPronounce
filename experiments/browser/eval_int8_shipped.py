"""Student int8 ONNX caches scored with the student models shipped in openpronounce/models (no refit).

    .venv-exp/bin/python experiments/browser/eval_int8_shipped.py <dir holding copies named word_mdd_en.json / scorer_en.json>
"""
import logging
import os
import sys
import types

import common

ROOT = common.ROOT
logging.basicConfig(level=logging.WARNING)
os.environ["OPENPRONOUNCE_MODELS_DIR"] = sys.argv[1]
common.use_variant("student-int8-shipped")
import scorer as bench_scorer  # noqa: E402
import word_model  # noqa: E402

source = os.path.join(ROOT, "so762", "student-int8")
word_model.logits_dir = lambda split: os.path.join(source, f"logits_{split}")
bench_scorer.logits_dir = word_model.logits_dir
word_model.features_path = lambda split: os.path.join(source, f"gop_{split}.npz")
print("=== word model (student, shipped), int8 test ===", flush=True)
word_model.report(word_model._load("test"), "en")
bench_scorer.features("test", "en")
print("=== scorer (student, shipped), int8 test ===", flush=True)
bench_scorer.report(types.SimpleNamespace(lang="en"))
