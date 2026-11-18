# Changelog

## Unreleased

- Changed: OpenPronounce is now published under the AGPL-3.0-or-later, with a commercial licence for closed products (`LICENSING.md`). Releases 0.1 to 0.4 stay MIT (`LICENSE-MIT-0.x`). `NOTICE` lists the third-party components, `CONTRIBUTING.md` the terms of contributions.
- Changed: English no longer needs espeak-ng. The expected sounds come from the CMU pronouncing dictionary shipped with the package (`openpronounce.g2p`); `Levenshtein` (GPL) leaves the dependencies, replaced by `rapidfuzz`. On the speechocean762 test split the results do not move (word AUC 0.928 against 0.924, total Pearson 0.791 against 0.790, same models). The other languages still use espeak-ng. `OPENPRONOUNCE_G2P=espeak` restores the former behaviour for English.
- Changed: espeak-ng is run as a separate program and read through its output (`g2p.espeak_binary`, `$OPENPRONOUNCE_ESPEAK`); the `phonemizer` package (GPL) and the `espeak` extra are gone. Same phones as before on 700 sentences in seven languages. Nothing OpenPronounce imports is under a copyleft licence any more, so the commercial licence covers every language.
- Added: words that were not said. A small model (`models/word_missing_generic.json`, `gop.missing_matrix`) reads what an absent word leaves in the alignment; such a word is reported with `missing: true` and named in `reasons`. On the test split, 83 % of the words added to a text are found, 0.7 % of the words really read are called missing (`benchmarks/missing_extra.py`).
- Added: speech that is not in the sentence. `differences.extra` lists the stretches where more was said than expected, with the words around them and the sounds heard (`gop.extra_speech`); the command line prints them as `Not expected`. 63 % of the words removed from a text are located, 2.8 % of the untouched sentences get a stretch by mistake.
- Changed: French, Spanish, German, Italian, Portuguese and Dutch get the trained pipeline instead of the edit-distance rule and the formula: `models/word_mdd_generic.json` (no phone identity, no hidden states; AUC 0.911 on the English test split against 0.924 for the English model) and `models/scorer_generic.json` (Pearson 0.771 against 0.790). `--fast` and `transcription=False` now work for them. `Language.variants` lists, per language, the recognizer tokens accepted for a phone of espeak. Checked on native speakers (FLEURS, 220 held-out recordings per language): natives score 72 to 77 with the native reference described below (English 74); 0.4 to 1.6 % of their words are reported, where the old rule reported 5 to 27 %. `Language.word_threshold` sets, per language, the probability from which the generic model reports a word, so that a word replaced by another is found as often as in English (74 to 85 %, against 34 to 68 % at the model's own threshold). `benchmarks/native_calibration.py` runs the same check on your own native recordings and proposes both settings.
- Added: a word missing from the dictionary is derived from a known word (plural, possessive, past tense, compound) or pronounced by rule; in that last case it is marked `guessed` in `differences.words[]` and never reported as mispronounced. Numbers written in digits are read as words.
- Added: free speech. `openpronounce.assess(waveform)` without a text, `openpronounce recording.wav` on the command line and `POST /free` score a recording of any length: it is cut into sentence-like parts at its pauses (`audio.split_speech`), transcribed, aligned on its own transcript and scored by a model trained for that (`models/free_scorer_en.json`, `benchmarks/free.py`). On the speechocean762 test split with the prompts hidden, Pearson with the human total 0.67 (0.78 for adults, 0.38 for children of 12 and under, whose speech the transcription gets wrong one word in two; 0.79 on sentences of 9 words and more, 0.45 under 5 words), against 0.54 for the read-speech scorer fed with the transcript. Results carry `reliable` and `reasons`; words are reported as `unclear`, never as mispronounced.
- Added: free speech in French, Spanish, German, Italian, Portuguese and Dutch (`models/free_scorer_generic.json`, Pearson 0.671 with the human total on the English test split against 0.672 for the English model). A language without a model of a kind now falls back on the generic one, whatever the kind.
- Added: a native reference per language (`models/native_<lang>.json`, `scorer.native_shift`). The features the scorer takes from the alignment are moved by the gap between native readers of the language and native readers of English (FLEURS, 120 `dev` recordings each), because the recognizer and espeak agree less well in some languages and the scorer read that as an accent. On 220 held-out native readings: German 73.4 to 76.3, Portuguese 69.6 to 74.6, Dutch 68.2 to 75.7, French 74.1 to 76.0, Spanish and Italian within half a point. Without a text: German 66.3 to 72.5, Portuguese 61.1 to 72.1, Dutch 58.3 to 70.8, English 70.7. Reported words do not change. `benchmarks/native_calibration.py --shift` writes the file for a new language.
- Added: natives against learners outside English, on the two public corpora that hold both (L2FOC for French and Italian, Heroico for Spanish): a native scores above a learner 93, 74 and 78 times in 100 in short answers and read sentences scored as free speech, on groups of 8 to 18 speakers. On a text read aloud the order holds in French (71 in 100) and not in Italian (36), where natives read faster than the learners. Unscripted native speech (VoxPopuli, CIEMPIESS, CORAA-MUPE) scores 66 to 73, hesitations being marked down. See `benchmarks/README.md`.
- Changed: the English free-speech scorer blends its trees with a linear head on the hidden states of the phone recognizer (`scorer.EmbeddingHead`, no extra model). Pearson with the human total on the test split 0.672 to 0.708, adults 0.785 to 0.820, children of 12 and under 0.376 to 0.417.
- Added: `benchmarks/telephone.py`, the test split through a simulated phone line (8 kHz, 300-3,400 Hz, G.711). Pearson with the raters 0.813 to 0.778 with the sentence, 0.684 to 0.636 without; a score moves by 3.6 and 3.5 points on average.
- Changed: the command line no longer prints the model-loading messages of transformers.
- Changed: the web demo scores free speech when the sentence is left empty, shows the three sub-scores and says when a score should not be trusted.
- Added: `audio_quality` in every result (`audio.quality`): duration, level, clipping, signal-to-noise estimate and `issues` (`too_short`, `silent`, `too_quiet`, `clipped`, `noisy`). `too_quiet` starts at -60 dBFS, where a 16-bit recording begins to lose points (1.5 there, 9 at -70).
- Added: `compare_audio_with_text(..., transcription=False)` and `openpronounce --fast`: only the phone recognizer runs, one checkpoint instead of two, same scores.
- Added: `openpronounce-evaluate manifest.csv` measures the agreement between the scores and your own human ratings (Pearson, Spearman, 95 % interval, error on your scale, per-group breakdown, resumable).
- Added: `OPENPRONOUNCE_MODELS_DIR` loads model files from other directories (a new language, a model trained on other data), `openpronounce --models` lists the models in use with their licence and metrics.
- Added: `benchmarks/groups.py`, the scorer by age group and gender on the test split (adults 0.85, children of 12 and under 0.67, teenagers 0.46 on 12 speakers with little spread in their ratings).
- Changed: mispronounced words are found by forced alignment of the expected phones on the recognizer's posteriors (`openpronounce.alignment`, CTC Viterbi in numpy) and a model trained on the speechocean762 ratings (`openpronounce.gop`, `models/word_mdd_en.json`: MLP on per-phone goodness-of-pronunciation features and on the recognizer's hidden states pooled over the phone's frames, trees on their per-word aggregates). On the 2,500-utterance test split, AUC 0.80 → 0.92; at the default threshold precision 0.20 → 0.53 for recall 0.75 → 0.51, 5 % of the words flagged instead of 20 %. `errors[].confidence` is now a probability, `errors[].phones[]` carry `start`/`end`. The edit-distance rule stays for plain phone lists, for languages without a model and with `compare_phones(..., use_model=False)`.
- Added: `differences.words[]` (every word with its probability, time span and per-phone details) and `differences.alignment` (`sentence_probability`, `reliable`: another sentence or a half-read one is detected, AUC 0.996 on synthetic cases).
- Changed: the English score comes from gradient-boosted trees trained on the speechocean762 ratings (`openpronounce.scorer`, `models/scorer_en.json`): 53 features from the alignment (goodness of pronunciation, speech rate, pauses, rhythm), the word model and a cheap prosody pass (YIN pitch, RMS energy). `scores` gives `total`, `accuracy`, `fluency`, `prosodic` on 0-100; `score` is the total. Pearson with the human total on the test split 0.63 → 0.79 (accuracy 0.74, fluency 0.82, prosody 0.81). The former formula is returned as `formula_score` and stays the score of the other languages.
- Added: `compare_audio_with_text(..., acoustic=False)` skips the synthetic reference and the DTW (no gTTS call, about 1.5 s saved on CPU); `distance` and `acoustic_distance` are then `None`.
- Added: `benchmarks/word_model.py` and `benchmarks/scorer.py` rebuild the models from the corpus; `benchmarks/word_detection.py --report` and `benchmarks/speechocean762.py` report the new system next to the old rule.
- Changed: the CLI summary prints `accuracy`, `fluency` and `prosodic` under the score and warns when the recording does not sound like a reading of the sentence.
- Added: `compare_audio_with_text(..., reference_audio=..., reference_embeddings=...)` to compare against a human recording of the sentence instead of the synthesized voice, and `prosody=False` to skip the pitch/energy contours.

## 0.3.0 (2026-08-15)

- Added: multi-language support (fr, es, de, it, pt, nl), `lang` parameter everywhere (`--lang` on the CLI, `lang` form field on the API), `GET /languages`. English stays the default and behaves as before.
- Changed: web demo redesigned (single column, language selector, word chips with per-phone highlighting and confidence, collapsed charts, clear error and loading states).
- Fixed: browser recordings (webm/opus) could not be decoded since librosa 1.0 dropped audioread; ffmpeg is now used directly as a fallback decoder.
- Removed: the Streamlit app (`streamlit_app.py`), whose bridge between the embedded UI and Streamlit never worked; use the FastAPI app or the Docker image.
- Fixed: in the transcription-based fallback, extra words heard inside a word ("hell no" for "hello") are now shown in `actual_word` (#6).
- Added: CUDA support (`OPENPRONOUNCE_DEVICE`, automatic when available) and `Dockerfile.gpu`.
- Fixed: sentences that espeak merges into one token ("would have to") lost all expected phones, so no word could be flagged and the phone error rate exploded (17 % of speechocean762 utterances).
- Added: `benchmarks/speechocean762.py`, benchmark of the score against human ratings.
- Changed: the acoustic term of the score has a per-language baseline (`Language.acoustic_good`): a native French/Spanish/... reading no longer loses 15-25 points because the English embeddings put two native voices of that language further apart.
- Changed: score recalibrated on speechocean762 (weights acoustic/phonemes/words 0.2/0.5/0.3 → 0.3/0.4/0.3, acoustic bounds 5/15 → 6/15), Spearman with the human total 0.638 → 0.652 on the same 500 utterances (0.583 with the 0.2.1 code, before the phonemization fix).
- Changed: word-level mispronunciation detection uses the CTC posteriors of the phone recognizer. `errors[]` entries gain `confidence` (0-1) and `phones[]` (`expected`, `heard`, `confidence` per phone), `differences.heard_phones_confidence` is added; `transcribe_phones(..., return_confidence=True)` and `recognize_phones()` expose the confidences. Close substitutions and word-final slips count less, and an expected phone the recognizer found plausible is not an error. On speechocean762 the precision of the flagged words goes from 0.12 to 0.20 (F1 0.21 to 0.31, one word in five flagged instead of two in five). Also fixed a few normalization gaps (Mandarin tone numbers, aspirated stops, `ɔɹ`/`oɹ` and `ɜ`/`ɚ` merges).
- Added: offline TTS backends (Piper, Kokoro), `OPENPRONOUNCE_TTS` env var, `OPENPRONOUNCE_TTS_VOICE` / `voice=` to pick the voice. `pip install openpronounce[tts-piper]` or `[tts-kokoro]`.

## 0.2.1 (2026-08-15)

- Install from PyPI in the docs and the notebook.
- Release workflow (`Release` action, `vX.Y.Z` input) that bumps, tags, publishes to PyPI and creates the GitHub Release.
- Fix test discovery on CI (`pythonpath`).

## 0.2.0 (2026-08-15)

### Breaking

- The code now lives in the `openpronounce` package: `from openpronounce import audio, speech`
  (was `import audio`, `import speech`). Installable with `pip install .` and shipped with an
  `openpronounce` console script.
- The score is now computed from length-independent measures (mean per-frame DTW distance,
  phoneme error rate, word error rate). Scores are not comparable with 0.1.x: previously a
  perfect reading of a long sentence could score 30/100 because raw DTW distances grow with
  the audio length.
- `compute_pronunciation_score(acoustic_distance, phoneme_error_rate, word_error_rate)` takes
  the normalized measures.
- `audio.webp2wav` is renamed `audio.webm2wav` (old name kept as an alias) and writes `<name>.16k.wav`.

### Fixed

- Upper-case reference text was phonemized letter by letter by espeak (`IT` -> /aɪtiː/), so
  every word of an upper-case sentence was flagged as mispronounced.
- The reference audio was written to a fixed `reference.mp3` in the working directory:
  concurrent web requests overwrote each other's file. References are now cached per
  sentence under `$OPENPRONOUNCE_CACHE_DIR` (default: system temp dir).
- Uploading a `.wav` to the API deleted the converted file before it was read.
- The test suite referenced functions that did not exist and CI had been red since December 2025.

### Added

- Phone-level assessment: `wav2vec2-lv-60-espeak-cv-ft` recognizes the phones actually
  said; `differences.errors` now shows, per word, the expected phones and the phones heard
  (`openpronounce/phones.py`, `transcribe_phones`, `compare_phones`). Set
  `OPENPRONOUNCE_PHONEME_MODEL=off` to fall back to the transcription-based detection.

### Changed

- Models are loaded lazily on first use, and the CTC checkpoint's encoder is reused for
  embeddings: half the memory and load time, and `import openpronounce` is instantaneous.
- `torchaudio` (whose `load`/`save` now require torchcodec) and unused dependencies
  (`coqui-tts`, `dtw-python`, `pydub`, `spacy` download step) are dropped.
- New fields: `acoustic_distance`, `differences.phoneme_error_rate`, `differences.word_error_rate`,
  `errors[].actual_word` is always present.
- Dockerfile, `/health` endpoint, Swagger metadata, CI matrix (3.10 / 3.12).
- Demo notebook rewritten to install from GitHub and run end to end on Colab.
