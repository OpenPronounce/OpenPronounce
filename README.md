<h1 align="center">OpenPronounce</h1>

<p align="center">
  <b>Open-source, phoneme-level pronunciation assessment.</b><br>
  Give it a recording and the sentence it should contain. Get scores learned from human raters (overall, accuracy, fluency, prosody), the mispronounced words with the sounds actually heard (IPA), the transcription and the pitch and energy curves. Runs on your machine, on CPU.
</p>

<p align="center">
  <a href="https://pypi.org/project/openpronounce/"><img src="https://img.shields.io/pypi/v/openpronounce.svg" alt="PyPI"></a>
  <a href="https://github.com/Halleck45/OpenPronounce/actions/workflows/tests.yml"><img src="https://github.com/Halleck45/OpenPronounce/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
  <a href="https://colab.research.google.com/github/Halleck45/OpenPronounce/blob/main/OpenPronounce-demo.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://github.com/sponsors/Halleck45"><img src="https://img.shields.io/static/v1?label=Sponsor&message=%E2%9D%A4&logo=GitHub&color=%23fe8e86" alt="Sponsor"></a>
</p>

<p align="center">
  <img src="./docs/demo.gif" alt="OpenPronounce web demo: record a sentence, get a score, see which sounds went wrong" width="720">
</p>

```console
$ pip install openpronounce
$ openpronounce recording.wav "Hello, how are you?"
Score        : 59.0/100
Transcription: HELL NO WHO ARE YOU
Heard phones : /h ɛ l n oʊ h u ɑɹ j u/
Mispronounced:
  - hello: expected /həloʊ/, heard /hɛlnoʊ/ (confidence 89%)
  - how: expected /haʊ/, heard /hu/ (confidence 50%)
```

Azure Speech, SpeechAce or ELSA sell this behind an API key and a per-minute price. OpenPronounce is the MIT-licensed building block for language-learning apps, EdTech products and research: no key, no billing, and your learners' voices never leave your servers. English is the calibrated language; French, Spanish, German, Italian, Portuguese and Dutch work too, experimentally.

## Install

Python 3.10+, plus `ffmpeg` and `espeak-ng` on the system (`apt install ffmpeg espeak-ng`, `brew install ffmpeg espeak-ng`).

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU wheels, much smaller
pip install openpronounce
```

Two Wav2Vec2 checkpoints (~1.2 GB each) are downloaded from the Hugging Face Hub on first use: one for words, one for phones.

## Use it

**Command line**

```bash
openpronounce recording.wav "Hello, I am a developer"
openpronounce recording.mp3 "Hello, I am a developer" --json --no-prosody   # machine-readable
openpronounce bonjour.wav "Bonjour, je suis développeur" --lang fr
```

**Python**

```python
from openpronounce import load_audio, compare_audio_with_text

sound = load_audio("recording.wav")          # any format ffmpeg reads, resampled to 16 kHz mono
result = compare_audio_with_text(sound, "Hello, I am a developer")

print(result["score"], result["scores"])     # 77.6 {'total': 77.6, 'accuracy': 82.5, 'fluency': 79.4, 'prosodic': 81.1}
for err in result["differences"]["errors"]:
    print(err["word"], err["expected"], "->", err["actual"] or "(missing)", err["confidence"])
```

`compare_audio_with_text(..., acoustic=False)` skips the synthetic reference and the DTW (no network, about 1.5 s saved on CPU); the scores do not depend on it.

Every function takes `lang="en"`. Lower-level pieces are exposed too: `transcribe`, `transcribe_phones`, `get_phonemes`, `compare_phones`, `compare_transcriptions`.

**Web app**

```bash
docker run -p 8000:8000 $(docker build -q .)     # or: pip install -e ".[app]" && uvicorn server:app
```

Then open http://localhost:8000: record from the microphone or drop a file, pick the language, get the score and the words with their wrong sounds highlighted. Microphone access needs `https://` or `localhost`. GPU: `Dockerfile.gpu` and `OPENPRONOUNCE_DEVICE=cuda` (CUDA is picked automatically when available).

| Endpoint | Form fields | Returns |
|---|---|---|
| `POST /pronunciation` | `file`, `expected_text`, `lang` (default `en`) | the full analysis below |
| `POST /speech2text` | `file`, `lang` | `{"transcript": ...}` |
| `POST /phonemes` | `text`, `lang` | `{"phonemes": [...], "words": [...]}` |
| `POST /tts` | `text`, `lang` | reference pronunciation, 16 kHz wav |
| `GET /languages`, `GET /health`, `GET /docs` | | registry, liveness, Swagger UI |

**Notebook**: [open in Colab](https://colab.research.google.com/github/Halleck45/OpenPronounce/blob/main/OpenPronounce-demo.ipynb), no local setup.

## What you get

| Field | Meaning |
|---|---|
| `score` | 0-100, the `total` of `scores` (English); the weighted formula below for the other languages |
| `scores` | `total`, `accuracy`, `fluency`, `prosodic` on 0-100, learned from human ratings (ten times the raters' 0-10 scale: a good learner reading sits around 80) |
| `formula_score` | the pre-0.4 formula (acoustic distance, phone and word error rates), for comparison |
| `transcribe` | what the word recognizer heard |
| `differences.errors[]` | one entry per mispronounced word: `word`, `position`, `expected` and `actual` (IPA), `confidence` (probability that the raters would reject the word), `phones[]` with `expected`, `heard`, `confidence`, `start`, `end` for each sound |
| `differences.words[]` | the same for every word, flagged or not, with its time span and `mispronounced` |
| `differences.alignment` | does the recording match the sentence at all: `sentence_probability` and `reliable` (false for another sentence or a half-read one), `log_likelihood_per_phone`, `plausible_phone_ratio` |
| `differences.heard_phones`, `differences.expected_phones` | every sound recognized in the audio (with `heard_phones_confidence`), and the sounds expected for each word |
| `differences.phoneme_error_rate`, `differences.word_error_rate` | edited sounds / expected sounds, edited words / expected words |
| `acoustic_distance` | mean per-frame DTW distance between the learner's Wav2Vec2 embeddings and a synthetic native reading (`None` with `acoustic=False`) |
| `prosody.f0`, `prosody.energy` | pitch and loudness contours |
| `differences.expected_vector`, `differences.transcribed_vector` | aligned phoneme traces, ready to plot |

## How it works

1. A Wav2Vec2 model fine-tuned on espeak labels (`facebook/wav2vec2-lv-60-espeak-cv-ft`) recognizes the sounds actually said, straight from the audio. No language model gets a chance to "correct" the learner.
2. The sentence is phonemized with espeak-ng in the selected language and its sounds are force-aligned on the frame posteriors of that model (CTC Viterbi, in numpy). Every expected sound gets a time span and a set of goodness-of-pronunciation scores: how far its posterior is from the best token on its frames, how much the sentence likelihood drops when it is deleted or replaced by a close sound, its duration and context.
3. A small model trained on [speechocean762](https://github.com/jimbozhang/speechocean762) (5,000 utterances rated by five experts, phone by phone and word by word) turns these scores, together with the recognizer's own representation of the sound (its hidden states on those frames, reduced to 16 dimensions), into the probability that each sound, then each word, is mispronounced. A word is reported above 0.35, the point of best F1 against the raters. The same alignment says whether the recording is a reading of the sentence at all.
4. Gradient-boosted trees trained on the same corpus map 53 utterance-level features (aggregated goodness of pronunciation, speech rate, pauses, rhythm, pitch and energy statistics) to the raters' four scores. Wrong and half-read sentences, absent from the corpus, were added as synthetic examples so that they score low.
5. The audio is also transcribed with `facebook/wav2vec2-large-960h` (English) or a language-specific XLSR checkpoint, for the transcription and the word error rate.
6. Unless `acoustic=False`, the sentence is synthesized (gTTS by default, Piper or Kokoro offline), both recordings are encoded with Wav2Vec2 and aligned with DTW: that is the acoustic distance, which feeds `formula_score`.
7. Pitch (pYIN) and RMS energy give the prosody curves.

On the speechocean762 test split (2,500 utterances, speakers unseen in training), the scores reach Pearson 0.79 with the human total, 0.74 with accuracy, 0.82 with fluency and 0.81 with prosody; the formula of the previous versions reached 0.63. Of the words flagged, 53 % are rejected by the raters (20 % before), for 51 % of the words they reject (73 % before), with one word in twenty flagged instead of one in five. Details, scripts and the comparison of alternatives (WavLM, ZIPA, logistic variants) in [benchmarks/](benchmarks/README.md). Models are plain JSON files in `openpronounce/models/`, retrainable with `benchmarks/word_model.py` and `benchmarks/scorer.py`. For the other languages, `score` is the pre-0.4 formula `0.3 × acoustic + 0.4 × (1 − phoneme error rate) + 0.3 × (1 − word error rate)` and words are flagged by an edit-distance rule. The original idea is described in [this blog post](https://blog.lepine.pro/en/ai-wav2vec-pronunciation-vectorization/).

## Configuration

| Variable | Default | Effect |
|---|---|---|
| `OPENPRONOUNCE_TTS` | `gtts` | reference voice: `gtts` (network on first use of a sentence), `piper` or `kokoro` (offline, `pip install openpronounce[tts-piper]` / `[tts-kokoro]`). See [docs/reference-voice.md](docs/reference-voice.md). |
| `OPENPRONOUNCE_TTS_VOICE` | per engine | voice id (`en_GB-cori-medium`, `af_heart`, gTTS domain `co.uk`...) |
| `OPENPRONOUNCE_DEVICE` | auto | `cpu`, `cuda`, `cuda:1`, `mps` |
| `OPENPRONOUNCE_PHONEME_MODEL` | espeak model | `off` to skip the phone recognizer (word errors then come from the transcription, less precise) |
| `OPENPRONOUNCE_CACHE_DIR` | system temp | where synthesized references are cached |
| `HF_HOME` | `~/.cache/huggingface` | where the models live; `HF_HUB_OFFLINE=1` works once they are there |

## Limitations

- Only English has trained models. The other languages fall back on the edit-distance word rule and the weighted formula (only the acoustic baseline is per language) and rely on community XLSR models for the transcription; rated recordings would help.
- The models learned from Mandarin-speaking learners, half of them children, reading sentences of 2 to 12 words. The scores follow those raters: a clean adult reading gets about 80, not 100, and a strong accent of another origin is less well covered.
- Wav2Vec2 was trained on read speech by adults. Strong accents, children and noisy recordings degrade the recognition, and therefore the feedback.
- The phone recognizer has its own error rate (about one sound in ten on a clean native reading). The word model knows it and is lenient on function words and word-final consonants, as the raters are; it still misses accent traits they would not rate below 5 either.
- With gTTS, the first analysis of a sentence needs the network. Piper or Kokoro make it fully offline.

## Roadmap

- [ ] Hosted demo (the Docker image is ready, `scripts/sync_space.sh` pushes it to a Hugging Face Space)
- [ ] Show the four scores and the word probabilities in the web demo
- [ ] Trained models for French, Spanish, German, Italian, Portuguese, Dutch (needs rated recordings)
- [x] PyPI package, offline TTS, per-phone confidence, other languages, benchmark, GPU, forced alignment and models trained on human ratings

Contributions welcome, see the [open issues](https://github.com/Halleck45/OpenPronounce/issues).

## Contributing

```bash
git clone https://github.com/Halleck45/OpenPronounce.git && cd OpenPronounce
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[app,dev]"
pytest
```

Tests need espeak-ng and ffmpeg but neither the network nor the model weights.

## Support the project

If OpenPronounce saved you time, a star helps other developers and teachers find it. If it ends up in a product, [sponsoring](https://github.com/sponsors/Halleck45) helps me keep improving it.

## References

- [wav2vec 2.0](https://ai.meta.com/research/impact/wav2vec/), Baevski et al., 2020
- [speechocean762](https://github.com/jimbozhang/speechocean762), Zhang et al., 2021
- [Azure Speech visemes](https://learn.microsoft.com/azure/ai-services/speech-service/how-to-speech-synthesis-viseme) and the HumanBeanCMU39 mouth images used by the web UI

## License

MIT, see [LICENSE](LICENSE).
