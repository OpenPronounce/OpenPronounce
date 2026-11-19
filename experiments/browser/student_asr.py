"""Greedy char transcripts of the student's ASR head against wav2vec2-large-960h (fp32 ONNX), same 200 test utterances as drift.py.

    HF_HOME=~/.cache/huggingface .venv-exp/bin/python experiments/browser/student_asr.py student.int8.onnx
"""

import json
import os
import sys

import numpy as np

import common
from drift import chars_to_words, edit, greedy

os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
from transformers import Wav2Vec2Processor  # noqa: E402

vocab = {v: k for k, v in Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h").tokenizer.get_vocab().items()}
utts = set(common.sample_utts(200))
rows = [r for r in common.rows("test") if r["utt"] in utts]
teacher = common.session(f"{common.ONNX_DIR}/asr.fp32.onnx")
for name in sys.argv[1:]:
    student = common.session(f"{common.ONNX_DIR}/{name}")
    agree, wer_t, wer_p, wer_p_teacher = [], [], [], []
    for row in rows:
        wave = common.sound(row)
        t = teacher.run(None, {"input_values": common.normalize(wave)[None]})[0][0]
        s = student.run(None, {"input_values": common.normalize(wave)[None]})[2][0]
        n = min(len(t), len(s))
        agree.append(float((t[:n].argmax(-1) == s[:n].argmax(-1)).mean()))
        wt, ws = chars_to_words(greedy(t.argmax(-1)), vocab), chars_to_words(greedy(s.argmax(-1)), vocab)
        prompt = row["text"].upper().split()
        wer_t.append(edit(ws, wt))
        wer_p.append(edit(ws, prompt))
        wer_p_teacher.append(edit(wt, prompt))
    print(json.dumps({"model": name, "n": len(rows), "argmax_agree_vs_960h": np.mean(agree), "wer_vs_960h": np.mean(wer_t),
                      "wer_vs_prompt": np.mean(wer_p), "wer_vs_prompt_960h_fp32": np.mean(wer_p_teacher)}))
