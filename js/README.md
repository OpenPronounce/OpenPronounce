# openpronounce (browser)

A JavaScript port of the read-speech path of the Python library: you give it a 16 kHz
recording and the sentence that was read, it gives back the same JSON as
`openpronounce.assess(waveform, sentence, acoustic=False, transcription=False)`. The phone
model is the distilled wav2vec2 student exported to ONNX and run by onnxruntime-web, so the
audio never leaves the page. English only for now, and no free speech.

The package is not published. It reads five data files and the ONNX model:

```bash
npm install
npm run build:data   # models/cmudict.txt.gz from the Python package, JSON models copied into models/
```

`tools/copy-models.mjs` copies the word model and the scorer the Python package ships for the
student (`openpronounce/models/word_mdd_en_student.json`, `scorer_en_student.json`), renamed
`word_mdd_en.json` and `scorer_en.json`. They were fitted on fp32 student features; on the int8
ONNX features they give a word AUC of 0.939 and a Pearson of 0.798 with the human total. A
directory given as first argument replaces them with its own `word_mdd_en.json` and
`scorer_en.json`, which must be fitted on the features of the same student.

## Demo

```bash
OPENPRONOUNCE_ONNX=/mnt/t7/openpronounce-browser/hub/wav2vec2-base-phones-en/onnx/model_int8.onnx node tools/serve.mjs
```

Then open http://127.0.0.1:8765/demo/, type a sentence and record yourself, or pick an audio
file. Add `?ep=webgpu` to try WebGPU before WASM. For testing without a microphone, set
`OPENPRONOUNCE_CLIPS` to a directory of WAV files and open
`/demo/?clip=/clips/<name>.wav&text=<sentence>`: the page scores the clip as soon as the model
is ready.

## Using it in a page

```js
import * as ort from "onnxruntime-web";
import { createAssessor, decodeToMono16k } from "openpronounce";

const assessor = await createAssessor({ ort, modelUrl: "/models/student.int8.onnx", dataUrl: "/models/" });
const samples = await decodeToMono16k(await file.arrayBuffer());
const result = await assessor.assess(samples, "The quick brown fox");
```

Multi-threaded WASM needs the page to be cross-origin isolated (`Cross-Origin-Opener-Policy:
same-origin` and `Cross-Origin-Embedder-Policy: require-corp`). Without it, onnxruntime
falls back to a single thread.

## Tests and parity

`npm test` checks the pure functions (alignment, Levenshtein, models, G2P, numpy-style
reductions) against fixtures written by the Python library
(`test/fixtures/make_fixtures.py`).

The parity harness compares the two implementations on speechocean762. The Python side
writes the reference, the JS side reads it:

```bash
# from the repository root; the student checkpoint makes the library pick its _student models
HUB=/mnt/t7/openpronounce-browser/hub/wav2vec2-base-phones-en
OPENPRONOUNCE_PHONEME_MODEL=$HUB PYTHONPATH=. .venv-exp/bin/python js/parity/reference.py \
    --onnx $HUB/onnx/model_int8.onnx --out /mnt/t7/openpronounce-browser/js/so762-shipped \
    --limit 300 --dump-tensors 300 --prosody 50
cd js
REF=/mnt/t7/openpronounce-browser/js/so762-shipped
node parity/parity.mjs --ref $REF --mode tensors   # JS reads the posteriors Python computed: tests the port alone
node parity/parity.mjs --ref $REF --model $HUB/onnx/model_int8.onnx --mode e2e   # JS runs the model (WASM) under Node
node parity/compare.mjs a/reference.jsonl b/reference.jsonl   # two Python runs, e.g. two ORT settings
```

With the posteriors of the Python run, the JS output is the same JSON, byte for byte, on all
300 utterances tested. With the fp32 model on both sides the scores are identical too. In
int8 they are not: each onnxruntime build has its own int8 kernels, and the total differs by
0.68 point on average (0.76 over the 2500 test utterances with the earlier int8 refit). Two optimization levels of the Python
runtime differ by as much. `parity/RESULTS.md` has the figures.

## What differs from Python

The decoding and resampling come from the browser (Web Audio) instead of librosa and soxr.
The YIN autocorrelation is computed in float64 instead of float32, which can move a pitch
frame now and then. The pYIN contour of the `prosody` key (display only, `prosody: true`) is a
port of librosa's and matches its frames to within one cent on 99.6 % of them.

## Other languages

The other languages need espeak-ng for the expected phones. espeak-ng has WASM builds
(GPL-3, so it would have to stay a separate, optional download); it is the natural next step,
together with the generic word model and scorer.

## Licence

AGPL-3.0-or-later, like the Python library. CMUdict is under a BSD-style licence, copied
next to the data as `models/cmudict.LICENSE`.
