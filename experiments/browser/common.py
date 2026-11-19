"""Shared helpers of the browser experiments (paths, ONNX sessions, speechocean762 caches)."""

import io
import json
import os
import shutil
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "benchmarks"))
sys.path.insert(0, REPO)

ROOT = "/mnt/t7/openpronounce-browser"
ONNX_DIR = os.path.join(ROOT, "onnx")
ORIGINAL = os.path.join(os.path.expanduser("~"), ".cache", "openpronounce", "speechocean762")


def variant_dir(variant):
    """A DATASET_DIR look-alike for ``variant``: own logits_<split>, the parquet files shared."""
    if variant == "torch":
        return ORIGINAL
    d = os.path.join(ROOT, "so762", variant)  # exFAT: no symlinks, the parquet path is patched instead
    os.makedirs(d, exist_ok=True)
    return d


def _original_parquet(split="test"):
    import speechocean762

    return os.path.join(ORIGINAL, speechocean762.PARQUET[split])


def use_variant(variant):
    """Point the benchmark modules at the caches of ``variant`` (call before using them)."""
    import scorer as bench_scorer
    import speechocean762
    import word_model

    d = variant_dir(variant)
    speechocean762.download = _original_parquet
    for module in (speechocean762, word_model, bench_scorer):
        module.DATASET_DIR = d
    return d


def normalize(wave):
    """What Wav2Vec2FeatureExtractor(do_normalize=True) does to one waveform."""
    wave = np.asarray(wave, dtype=np.float32)
    return ((wave - wave.mean()) / np.sqrt(wave.var() + 1e-7)).astype(np.float32)


def session(path, threads=8):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])


def run(sess, wave):
    outputs = sess.run(None, {"input_values": normalize(wave)[None]})
    return [o[0] for o in outputs]


def log_softmax(logits):
    logits = logits.astype(np.float32)
    m = logits.max(axis=-1, keepdims=True)
    return logits - m - np.log(np.exp(logits - m).sum(axis=-1, keepdims=True))


def rows(split):
    import speechocean762

    speechocean762.download = _original_parquet
    return sorted(speechocean762.load_rows(split), key=lambda r: r["utt"])


def sound(row):
    from openpronounce import audio

    return audio.load(io.BytesIO(row["audio"]["bytes"]))


def prepare_logits_dir(variant, split):
    """``logits_<split>`` of ``variant`` with the labels, vocab and layer files of the original cache."""
    out = os.path.join(variant_dir(variant), f"logits_{split}")
    os.makedirs(out, exist_ok=True)
    for name in ("labels.json", "vocab.json", "hidden.json"):
        shutil.copy(os.path.join(ORIGINAL, f"logits_{split}", name), out)
    return out


def sample_utts(n=200, seed=0, split="test"):
    with open(os.path.join(ORIGINAL, f"logits_{split}", "labels.json")) as f:
        utts = sorted(lab["utt"] for lab in json.load(f))
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(utts, size=min(n, len(utts)), replace=False).tolist())
