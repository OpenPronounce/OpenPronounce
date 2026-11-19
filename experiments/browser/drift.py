"""Output drift of quantized ONNX models against the fp32 ONNX export, on a speechocean762 test sample.

    .venv-exp/bin/python experiments/browser/drift.py --n 200 [--variants int8 fp16] [--student]

Phone model: frame argmax agreement, mean frame KL(fp32 || variant), layer-20 cosine, and the
edit distance between the greedy CTC phone strings. ASR model: frame argmax agreement, KL,
last-hidden cosine, WER of the greedy transcript against the fp32 one and against the prompt.
"""

import argparse
import json
import time

import numpy as np

import common


def greedy(ids, blank=0):
    out, prev = [], None
    for i in ids:
        if i != prev and i != blank:
            out.append(int(i))
        prev = i
    return out


def edit(a, b):
    from rapidfuzz.distance import Levenshtein

    return Levenshtein.distance(a, b) / max(1, len(b))


def chars_to_words(ids, vocab):
    text = "".join(vocab[i] for i in ids).replace("|", " ")
    return text.split()


def cos(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    return float(np.mean((a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-9)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--variants", nargs="+", default=["int8", "fp16"])
    parser.add_argument("--models", nargs="+", default=["phone", "asr"])
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--out", default="/mnt/t7/openpronounce-browser/drift.json")
    args = parser.parse_args()
    utts = set(common.sample_utts(args.n))
    rows = [r for r in common.rows("test") if r["utt"] in utts]
    from transformers import Wav2Vec2Processor
    import os
    os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    asr_vocab = {v: k for k, v in Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h").tokenizer.get_vocab().items()}
    sounds = [common.sound(r) for r in rows]
    results = {}
    for kind in args.models:
        ref_sess = common.session(f"{common.ONNX_DIR}/{kind}.fp32.onnx", args.threads)
        refs = [common.run(ref_sess, w)[:2] for w in sounds]
        del ref_sess
        for variant in args.variants:
            sess = common.session(f"{common.ONNX_DIR}/{kind}.{variant}.onnx", args.threads)
            agree, kls, coss, eds, wer_ref, wer_text, wer_text_fp32 = [], [], [], [], [], [], []
            t0 = time.time()
            for row, w, (l_ref, h_ref) in zip(rows, sounds, refs):
                l, h = common.run(sess, w)[:2]
                lp_ref, lp = common.log_softmax(l_ref), common.log_softmax(l)
                agree.append(float((lp.argmax(-1) == lp_ref.argmax(-1)).mean()))
                kls.append(float((np.exp(lp_ref) * (lp_ref - lp)).sum(-1).mean()))
                coss.append(cos(h, h_ref))
                g, g_ref = greedy(lp.argmax(-1)), greedy(lp_ref.argmax(-1))
                if kind == "phone":
                    eds.append(edit(g, g_ref))
                else:
                    words, words_ref = chars_to_words(g, asr_vocab), chars_to_words(g_ref, asr_vocab)
                    prompt = row["text"].upper().split()
                    wer_ref.append(edit(words, words_ref))
                    wer_text.append(edit(words, prompt))
                    wer_text_fp32.append(edit(words_ref, prompt))
            res = {"n": len(rows), "argmax_agree": np.mean(agree), "kl_mean": np.mean(kls), "kl_p99": float(np.quantile(kls, .99)),
                   "hidden_cos_mean": np.mean(coss), "hidden_cos_min": float(np.min(coss)), "seconds": time.time() - t0}
            if kind == "phone":
                res["greedy_phone_edit_vs_fp32"] = np.mean(eds)
            else:
                res.update({"wer_vs_fp32": np.mean(wer_ref), "wer_vs_prompt": np.mean(wer_text),
                            "wer_vs_prompt_fp32": np.mean(wer_text_fp32)})
            results[f"{kind}.{variant}"] = {k: float(v) for k, v in res.items()}
            print(kind, variant, json.dumps(results[f"{kind}.{variant}"]), flush=True)
            del sess
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
