# Browser feasibility (ONNX Runtime Web)

Experiment code, not part of the package. Large files live on `/mnt/t7/openpronounce-browser/`
(ONNX models, speechocean762 caches per variant, distillation audio and checkpoints, logs).

Environments: `.venv-exp` (CPU) with `PYTHONPATH=/mnt/t7/openpronounce-browser/pylib` for the
`onnx` package (installed there with `uv pip install --target`, not in the venv), and
`~/.venvs/openpronounce-gpu` for the distillation. `HF_HOME=~/.cache/huggingface` for anything
that loads `wav2vec2-large-960h`.

```bash
cd experiments/browser
export PYTHONPATH=/mnt/t7/openpronounce-browser/pylib HF_HOME=~/.cache/huggingface
PY=../../.venv-exp/bin/python

# phase 0: export, int8, fp16; drift; downstream; speed
$PY export_onnx.py                                   # phone.* and asr.* {fp32,int8,fp16}.onnx
$PY drift.py --n 200                                 # vs fp32 ONNX, 200 test utterances (seed 0)
$PY extract_onnx.py --model phone.int8.onnx --variant int8 --split test   # and --split train
$PY eval_variant.py --variant int8                   # shipped word model + scorer on int8 features
$PY eval_variant.py --variant torch                  # same code on the original torch caches
$PY eval_variant.py --variant torch --refit          # refit reference for the student
$PY bench_cpu.py phone.int8.onnx phone.fp32.onnx asr.int8.onnx
(cd web && npm install && node bench_wasm.mjs /mnt/t7/openpronounce-browser/onnx/phone.int8.onnx 4)

# phase 1: distillation (GPU), export, evaluation
$PY predecode.py                                     # CV27 en subset + so762 train -> int16 file
~/.venvs/openpronounce-gpu/bin/python distill.py --hours 2.5
$PY export_student.py --ckpt /mnt/t7/openpronounce-browser/student/final.pt
bash run_student_eval.sh student.int8.onnx student-int8   # extract both splits, shipped + refit eval
```

The ONNX graphs take the normalized waveform (zero mean, unit variance, `(1, samples)`
float32); the browser side has to do that normalization itself. `phone.onnx` returns the
phone logits and `hidden_states[20]`, `asr.onnx` the char logits and the last hidden state,
the student returns phone logits, its stand-in for layer 20 and char logits.

## Results (2026-11-19)

speechocean762 full test split (2500 utterances) for every downstream number. "current
code" is the same benchmark code run on the original torch caches; the shipped figures
(0.924 / 0.790) come from older tables and are not the like-for-like reference.
CPU: Ryzen 7 PRO 7840U, input of noise, median of 3 runs after a warm-up.

| variant | file (raw / gzip) | word AUC (<5) | total | accuracy | fluency | prosodic | CPU 5 s, 1 / 4 thr | WASM 5 s, 1 / 4 thr |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| large fp32 torch, shipped models (current code) | 1264 MB | 0.928 | 0.790 | 0.743 | 0.827 | 0.808 | 7.25 s* / 0.57 s | |
| large fp32 torch, refit | | 0.926 | 0.785 | 0.742 | 0.818 | 0.805 | | |
| large int8, shipped models | 355 / 233 MB | 0.928 | 0.791 | 0.746 | 0.830 | 0.812 | 0.75 / 0.26 s | 3.69 / 1.28 s |
| student fp32, shipped models | 382 MB | 0.937 | 0.789 | 0.749 | 0.831 | 0.811 | 0.62 / 0.24 s | |
| student fp32, refit | | 0.938 | 0.800 | 0.762 | 0.839 | 0.821 | | |
| student int8, shipped models | 123 / 87 MB | 0.935 | 0.785 | 0.748 | 0.827 | 0.807 | 0.38 / 0.14 s | 1.61 / 0.59 s |
| student int8, refit | | 0.937 | 0.791 | 0.758 | 0.836 | 0.817 | | |

18 s clip: large int8 3.45 / 1.27 s CPU, 16.1 / 5.4 s WASM; student int8 1.69 / 0.72 s CPU,
6.8 / 2.4 s WASM. (*) large fp32 at 1 thread takes longer on 5 s than on 18 s (7.25 vs
6.74 s, reproduced twice), not explained.

The sizes above are the phone model only (or the student, which carries both heads). Free
speech also needs `asr.int8.onnx` (355 / 254 MB). The student's char head is not good
enough: greedy WER against the prompt 0.569 (int8) and 0.536 (fp32) against 0.478 for
wav2vec2-large-960h fp32 (200 test utterances), so it does not replace the 960h model yet.
Drift on 200 test utterances (`drift.json`): phone int8 frame argmax agreement 99.0 %, KL
0.017, layer-20 cosine 0.995; asr int8 96.2 %, WER vs fp32 transcript 14.6 %, WER vs prompt
0.492 vs 0.478; fp16 is lossless for both (99.98 % / 99.95 %).
