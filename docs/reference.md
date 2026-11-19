# Reference

## Result of a read sentence

| Field | Meaning |
|---|---|
| `score` | 0-100, the `total` of `scores` |
| `scores` | `total`, `accuracy`, `fluency`, `prosodic` on 0-100, learned from human ratings (ten times the raters' 0-10 scale: a good learner reading sits around 80) |
| `formula_score` | the pre-0.4 formula (acoustic distance, phone and word error rates), for comparison |
| `transcribe` | what the word recognizer heard |
| `differences.errors[]` | one entry per mispronounced word: `word`, `position`, `expected` and `actual` (IPA), `confidence` (probability that the raters would reject the word; outside English, moved onto that scale by the threshold of the language), `phones[]` with `expected`, `heard`, `confidence`, `start`, `end` for each sound |
| `differences.words[]` | the same for every word, flagged or not, with its time span, `mispronounced`, and `missing` when the word was not said at all (it is then in `errors[]` too, with `missing: true`) |
| `differences.extra[]` | speech that is not in the sentence: `after` and `before` (positions of the words around it, `null` at either end), `start`, `end`, `heard` (the sounds) |
| `differences.alignment` | does the recording match the sentence at all: `sentence_probability` and `reliable` (false for another sentence or a half-read one), `log_likelihood_per_phone`, `plausible_phone_ratio` |
| `differences.heard_phones`, `differences.expected_phones` | every sound recognized in the audio (with `heard_phones_confidence`), and the sounds expected for each word |
| `differences.phoneme_error_rate`, `differences.word_error_rate` | edited sounds / expected sounds, edited words / expected words |
| `acoustic_distance` | mean per-frame DTW distance between the learner's Wav2Vec2 embeddings and a synthetic native reading (`None` with `acoustic=False`) |
| `prosody.f0`, `prosody.energy` | pitch and loudness contours |
| `differences.expected_vector`, `differences.transcribed_vector` | aligned phoneme traces, ready to plot |
| `audio_quality` | `duration`, `level_db`, `clipping_ratio`, `snr_db` and `issues` (`too_short`, `silent`, `too_quiet`, `clipped`, `noisy`) |
| `reliable`, `reasons` | `false` when the recording does not match the sentence, when words were not said, when much was said in excess or when the recording is not fit to be scored; the scores are still returned, do not show them to a learner as they are |

## Result of free speech

Without an expected text (`assess(waveform)`, `POST /free`), the result has its own shape:

| Field | Meaning |
|---|---|
| `mode` | `"free"` |
| `score`, `scores` | same four dimensions, weighted by duration over the segments; `null` only when no free-speech model is installed for the language |
| `transcribe` | what the word recognizer heard, the text everything else is measured against |
| `words[]` | every word with `start`, `end` (seconds in the whole recording), `confidence` that it was unclear, `unclear`, `expected` and `heard` (IPA) |
| `unclear_words` | the words above the threshold |
| `segments[]` | the parts the recording was cut into at its pauses, each with `start`, `end`, `transcribe`, `scores` |
| `duration`, `speech_duration` | seconds |
| `audio_quality`, `reliable`, `reasons` | as below |

## HTTP API

`uvicorn server:app` or the Docker image. Swagger UI on `/docs`.

| Endpoint | Form fields | Returns |
|---|---|---|
| `POST /pronunciation` | `file`, `expected_text`, `lang` (default `en`) | the full analysis below |
| `POST /free` | `file`, `lang` | free speech: scores, transcript, unclear words |
| `POST /speech2text` | `file`, `lang` | `{"transcript": ...}` |
| `POST /phonemes` | `text`, `lang` | `{"phonemes": [...], "words": [...]}` |
| `POST /tts` | `text`, `lang` | reference pronunciation, 16 kHz wav |
| `GET /languages`, `GET /health`, `GET /docs` | | registry, liveness, Swagger UI |

## Configuration

| Variable | Default | Effect |
|---|---|---|
| `OPENPRONOUNCE_TTS` | `gtts` | reference voice: `gtts` (network on first use of a sentence), `piper` or `kokoro` (offline, `pip install openpronounce[tts-piper]` / `[tts-kokoro]`). See [docs/reference-voice.md](docs/reference-voice.md). |
| `OPENPRONOUNCE_TTS_VOICE` | per engine | voice id (`en_GB-cori-medium`, `af_heart`, gTTS domain `co.uk`...) |
| `OPENPRONOUNCE_G2P` | per language | `lexicon` (English: CMUdict) or `espeak` |
| `OPENPRONOUNCE_ESPEAK` | `espeak-ng` on the PATH | path of the espeak-ng program, needed for every language other than English |
| `OPENPRONOUNCE_MODELS_DIR` | none | directories searched for model files before the shipped ones: add a language (its `native_<lang>.json`, or models of its own) or replace a model. `openpronounce --models` lists what is loaded, with licence and metrics |
| `OPENPRONOUNCE_DEVICE` | auto | `cpu`, `cuda`, `cuda:1`, `mps` |
| `OPENPRONOUNCE_PHONEME_MODEL` | espeak model | `off` to skip the phone recognizer (word errors then come from the transcription, less precise). A hub id or a folder selects another recognizer with the same phones, such as the distilled wav2vec2-base (`wav2vec2-base-phones-en`, English read speech, three times smaller): its word model and scorer (`*_en_student.json`) are picked with it. Transcription, free speech included, stays on `wav2vec2-large-960h` |
| `OPENPRONOUNCE_CACHE_DIR` | system temp | where synthesized references are cached |
| `HF_HOME` | `~/.cache/huggingface` | where the models live; `HF_HUB_OFFLINE=1` works once they are there |
