"""Python reference for the parity harness: the library's read-speech path on the ONNX student.

For every speechocean762 test utterance, writes the original wav bytes and one JSON line with
the result of ``openpronounce.assess(wave, text, acoustic=False, transcription=False)``, the
phone posteriors and hidden states coming from the ONNX student (onnxruntime CPU) instead of
the torch phone model. With ``$OPENPRONOUNCE_PHONEME_MODEL`` pointing at the student checkpoint,
the library picks the scorers shipped for it (``word_mdd_en_student.json``,
``scorer_en_student.json``); ``--models`` replaces them with another directory. Run from the
repository root of this worktree:

    OPENPRONOUNCE_PHONEME_MODEL=/mnt/t7/openpronounce-browser/hub/wav2vec2-base-phones-en \
    PYTHONPATH=. .venv-exp/bin/python js/parity/reference.py \
        --onnx /mnt/t7/openpronounce-browser/hub/wav2vec2-base-phones-en/onnx/model_int8.onnx \
        --out /mnt/t7/openpronounce-browser/js/so762-shipped
"""

import argparse
import io
import json
import logging
import os
import sys
import time

import numpy as np

SO762 = os.path.join(os.path.expanduser("~"), ".cache", "openpronounce", "speechocean762")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--models", help="directory searched before the shipped models (default: the shipped ones only)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0, help="first N utterances (sorted by id), 0 for all")
    parser.add_argument("--dump-tensors", type=int, default=0, help="also write float32 posteriors and hidden states of the first N")
    parser.add_argument("--prosody", type=int, default=0, help="run with prosody=True (pYIN contour) on the first N")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--opt", default="ORT_ENABLE_ALL", help="onnxruntime graph optimization level (noise floor runs)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.models:
        os.environ["OPENPRONOUNCE_MODELS_DIR"] = args.models
    os.environ["OPENPRONOUNCE_G2P"] = "lexicon"

    import onnxruntime as ort
    import pyarrow.parquet as pq

    import openpronounce
    from openpronounce import audio, phones

    from openpronounce import gop

    print("openpronounce from", os.path.dirname(openpronounce.__file__), file=sys.stderr)
    print("word model", gop.find_model("word_mdd"), "scorer", gop.find_model("scorer"), file=sys.stderr)
    with open(os.path.join(SO762, "logits_test", "vocab.json"), encoding="utf-8") as f:
        vocab = json.load(f)
    ids = {token: i for i, token in enumerate(vocab)}
    phones._vocab_ids = lambda: ids
    phones.phone_vocab.cache_clear()

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = args.threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = getattr(ort.GraphOptimizationLevel, args.opt)
    sess = ort.InferenceSession(args.onnx, opts, providers=["CPUExecutionProvider"])
    current = {}

    def phone_log_posteriors(wave, sampling_rate=16000, hidden_layer=None):
        wave = np.asarray(wave, dtype=np.float32)
        x = ((wave - wave.mean()) / np.sqrt(wave.var() + 1e-7)).astype(np.float32)
        logits, hidden = sess.run(["logits", "hidden20"], {"input_values": x[None]})
        logits, hidden = logits[0].astype(np.float32), hidden[0].astype(np.float32)
        m = logits.max(axis=-1, keepdims=True)
        logp = logits - m - np.log(np.exp(logits - m).sum(axis=-1, keepdims=True))
        current["logp"], current["hidden"] = logp, hidden
        return logp if hidden_layer is None else (logp, hidden)

    phones.phone_log_posteriors = phone_log_posteriors

    table = pq.read_table(os.path.join(SO762, "data", "test-00000-of-00001.parquet")).to_pylist()
    for row in table:
        row["utt"] = os.path.splitext(row["audio"]["path"])[0]
    table.sort(key=lambda r: r["utt"])
    if args.limit:
        table = table[: args.limit]

    wav_dir = os.path.join(args.out, "wav")
    tensor_dir = os.path.join(args.out, "tensors")
    os.makedirs(wav_dir, exist_ok=True)
    os.makedirs(tensor_dir, exist_ok=True)
    done = set()
    results_path = os.path.join(args.out, "reference.jsonl")
    if os.path.exists(results_path):
        with open(results_path, encoding="utf-8") as f:
            done = {json.loads(line)["utt"] for line in f if line.strip()}
    t0 = time.time()
    with open(results_path, "a", encoding="utf-8") as out:
        for k, row in enumerate(table):
            utt = row["utt"]
            if utt in done:
                continue
            wav_path = os.path.join(wav_dir, utt + ".wav")
            if not os.path.exists(wav_path):
                with open(wav_path, "wb") as f:
                    f.write(row["audio"]["bytes"])
            wave = audio.load(io.BytesIO(row["audio"]["bytes"]))
            t = time.time()
            result = openpronounce.assess(wave, row["text"], acoustic=False, transcription=False,
                                          prosody=k < args.prosody)
            elapsed = time.time() - t
            if k < args.dump_tensors:
                current["logp"].astype(np.float32).tofile(os.path.join(tensor_dir, utt + ".logp.f32"))
                current["hidden"].astype(np.float32).tofile(os.path.join(tensor_dir, utt + ".hidden.f32"))
            human = {key: row.get(key) for key in ("total", "accuracy", "fluency", "prosodic")}
            out.write(json.dumps({"utt": utt, "text": row["text"], "human": human, "seconds": round(elapsed, 3),
                                  "frames": int(current["logp"].shape[0]), "result": result}, ensure_ascii=False) + "\n")
            out.flush()
            if (k + 1) % 100 == 0:
                logging.info("[%d/%d] %.0fs", k + 1, len(table), time.time() - t0)


if __name__ == "__main__":
    main()
