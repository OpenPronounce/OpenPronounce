# Parity results (2026-11-19)

## Shipped student models

Python ran the library of `feat/student-phone-model` with
`OPENPRONOUNCE_PHONEME_MODEL=/mnt/t7/openpronounce-browser/hub/wav2vec2-base-phones-en`, so it
picked the shipped `word_mdd_en_student.json` and `scorer_en_student.json` (fitted on fp32 student
features), and the hub's `onnx/model_int8.onnx` through onnxruntime 1.31 CPU. JS used the same
files, copied by `tools/copy-models.mjs`, and onnxruntime-web 1.22 (WASM) under Node 24. First
300 test utterances by id; outputs in `/mnt/t7/openpronounce-browser/js/so762-shipped/`.

| comparison | utterances | total, mean abs diff | within 0.5 pt | max | Pearson | word flags agree |
|---|--:|--:|--:|--:|--:|--:|
| JS reads Python's posteriors (port only) | 300 | 0 | 100 % | 0 | 1 | 100 % (same JSON on 300/300) |
| int8 student, each side runs its own model | 300 | 0.68 | 62 % | 8.5 | 0.989 | 100 % (21/21 flags) |

`reliable` agrees on 300/300. On these 300 utterances the Pearson with the human total is 0.472
(JS) and 0.459 (Python); the first 300 ids are a hard subset (0.488 with the earlier int8 refit
on the same utterances), so this is no measure of the scorer. The pYIN contour matches librosa
within one cent on 99.1 % of frames (50 utterances).

## Earlier run: int8 refit of the distillation study

speechocean762 test split. Python ran the library from this worktree with the ONNX student
in place of the torch phone model (`parity/reference.py`, onnxruntime 1.31 CPU), scored by
the int8 refit (`so762/student-int8/models`). JS ran onnxruntime-web 1.22 (WASM) under
Node 24. Outputs are in `/mnt/t7/openpronounce-browser/js/`.

| comparison | utterances | total, mean abs diff | within 0.5 pt | Pearson | word flags agree |
|---|--:|--:|--:|--:|--:|
| JS reads Python's posteriors (port only) | 300 | 0 | 100 % | 1 | 100 % (same JSON on 300/300) |
| fp32 student, each side runs its own model | 300 | 0 | 100 % | 1 | 100 % (same JSON on 297/300) |
| int8 student, each side runs its own model | 2500 | 0.76 | 64 % | 0.991 | 99.3 % |
| Python against Python: int8, ORT_ENABLE_ALL vs ORT_DISABLE_ALL | 300 | 0.78 | 57 % | 0.984 | 99.5 % |
| Python against Python: int8 vs fp32 | 300 | 1.07 | 46 % | 0.978 | 99.4 % |

On the 2500 int8 utterances: p95 of the total gap is 2.7 points, p99 7.7, max 35.6 (47 above
5 points). `reliable` agrees on 99.9 %. Pearson with the human total is 0.796 (JS) and 0.791
(Python).

So the port has no measurable error. The int8 gap comes from the kernels: two onnxruntime
builds, or two optimization levels of the same one, do not round int8 matrix products the same
way (hidden-state cosine 0.9994 between them), and the score trees amplify that noise. The
target of 0.5 point on 95 % of utterances cannot be met with an int8 model run by two
different runtimes. It is met in fp32.

The pYIN contour (`prosody` key, display only) matches librosa within one cent on 99.6 % of
frames (100 utterances, median Pearson 1, worst 0.58).

Speed in Chrome (Ryzen 7 PRO 7840U, machine under load, load average about 25 on 16 threads):
4.9 s clip, WASM 4 threads: model 2.1 to 2.4 s, scoring 60 to 120 ms; WebGPU (with WASM fallback):
model 1.7 to 2.0 s. Page ready 4.4 to 5 s after load from a local server. The unloaded figure
from the distillation study is 0.59 s for 5 s (WASM, 4 threads).
