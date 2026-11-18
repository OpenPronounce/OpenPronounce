# Benchmarks

## speechocean762

[speechocean762](https://www.openslr.org/101/) (Zhang et al., 2021, CC BY 4.0) is a
free corpus of 5,000 English utterances read by 250 Mandarin-speaking learners (half of
them children), each rated by five experts. Every utterance carries sentence-level
`accuracy`, `fluency`, `prosodic` and `total` scores (0-10), and every word carries an
`accuracy` score (0-10) plus phone-level scores. Sentences are short (2 to 12 words).

`benchmarks/speechocean762.py` runs `speech.compare_audio_with_text` on a fixed random
sample of the test split and compares the output with the human ratings.

### How to run

```bash
# inference (resumable: utterances already in the CSV are skipped)
HF_HOME=~/.cache/huggingface python benchmarks/speechocean762.py --sample 500 \
    --out benchmarks/results/speechocean762-v0.3.csv

# analysis of the CSV (correlations, word-level precision/recall, grid search + CV)
python benchmarks/speechocean762.py --report --out benchmarks/results/speechocean762-v0.3.csv
```

`--report` also recomputes the score from the stored components with the constants of
the installed `openpronounce` when they differ from the ones used at run time, runs the
grid search (weights on a 0.1 step, `ACOUSTIC_DISTANCE_GOOD` in 3..8, `BAD` in 12..22)
and a 2-fold x 3-split cross-validation of the grid.

The parquet file (`mispeech/speechocean762` on the Hugging Face hub, audio included) is
downloaded to `~/.cache/openpronounce/speechocean762/` on first run (~300 MB for the
test split). The sample is stratified on the human `total` score with seed 0, so
`--sample 500` always picks the same 500 utterances. `--threads` sets the number of
torch threads (default 6). gTTS is called once per distinct sentence (network); the
script retries with backoff on failure.

CSV columns: `utt`, `speaker`, `text`, human `total`/`accuracy`/`fluency`/`prosodic`,
our `score`, the three components (`acoustic_distance`, `phoneme_error_rate`,
`word_error_rate`), word counts (`n_words`, `n_flagged`, `n_human_bad`, `n_hits`),
per-utterance word `recall`/`precision` (a word is "bad" for the raters when its
accuracy is below 5) and `wall_time` in seconds.

### Results, 0.3.0 (2026-08-15)

Same 500 utterances (seed 0), OpenPronounce 0.3.0 (worktree of commit 3759d4f: fixed
phonemizer fallback, confidence-weighted word flagging), same models, gTTS references,
CPU, 6 threads: 3.0 s per utterance, 25 min. Raw data: `results/speechocean762-v0.3.csv`.
`score` in the CSV is the run-time score with the 0.2.1 constants (0.2/0.5/0.3, bounds
5/15); the 0.3.0 constants (0.3/0.4/0.3, bounds 6/15) are evaluated by recomputing the
score from the stored components (`--report` does it).

Correlation with the human ratings (Pearson / Spearman):

| pair                                    | 0.2.1 run | 0.2.1 constants, 0.3.0 code | 0.3.0 constants |
|-----------------------------------------|----------:|----------------------------:|----------------:|
| score vs human total                    | 0.571 / 0.583 | 0.621 / 0.638 | **0.633 / 0.652** |
| score vs human accuracy                 | 0.535 / 0.553 | 0.591 / 0.618 | 0.603 / 0.631 |
| score vs human fluency                  | 0.532 / 0.515 | 0.578 / 0.571 | 0.595 / 0.590 |
| score vs human prosodic                 | 0.552 / 0.541 | 0.607 / 0.606 | 0.618 / 0.622 |
| per-speaker mean score vs mean total (106 speakers, Spearman) | 0.78 | 0.820 | 0.825 |
| acoustic_distance vs human total        | -0.639 / -0.666 | -0.638 / -0.666 | same |
| acoustic_distance vs human accuracy     | -0.608 / -0.642 | -0.607 / -0.641 | same |
| phoneme_error_rate vs human total       | -0.041 / -0.409 | -0.554 / -0.568 | same |
| phoneme_error_rate vs human accuracy    | -0.018 / -0.372 | -0.519 / -0.540 | same |
| word_error_rate vs human total          | -0.519 / -0.551 | -0.519 / -0.551 | same |
| word_error_rate vs human accuracy       | -0.503 / -0.548 | -0.503 / -0.548 | same |

The phonemization fix is what moved the phone error rate from noise (Pearson -0.04)
to a real signal (-0.55): only 2 utterances keep a PER above 1 (noisy recordings with
many insertions, not empty expected phones). Score mean 54.1, sd 26.8 with the new
constants (55.6 / 26.2 with the old ones); mean score per human total 3 -> 19,
4 -> 26, 5 -> 28, 6 -> 40, 7 -> 46, 8 -> 66, 9 -> 76 (old constants, monotonic, sd
11 to 23 within a level).

Word level (same rule as `word_detection.py`, "bad" = human accuracy < 5): micro
recall 0.732, precision 0.196, F1 0.309, 628 of 3,146 words flagged (20 %).

Grid search (Spearman of the recomputed score with the human total, best per weight
triple) and 2-fold x 3-split cross-validation (best combination of one half scored on
the other half, mean over the 6 held-out halves):

| weights (ac/ph/wo) | bounds | full-sample Spearman | CV held-out mean (6 folds) |
|---|---|--:|--:|
| best of each training half (0.7/0.3/0 or 0.7/0.2/0.1, bounds 3-7 / 16-21) | | | 0.676 |
| 0.7 / 0.2 / 0.1 | 5 / 18 | 0.680 | 0.678 |
| 0.8 / 0.2 / 0.0 | 5 / 22 | 0.682 (grid maximum) | |
| 0.8 / 0.1 / 0.1 | 5 / 18 | 0.677 | 0.676 |
| 0.6 / 0.2 / 0.2 | 5 / 18 | 0.672 | 0.669 |
| 1.0 / 0 / 0 (acoustic alone) | 5 / 18 | 0.666 | 0.665 |
| **0.3 / 0.4 / 0.3 (0.3.0 defaults)** | **6 / 15** | **0.652** | **0.649** |
| 0.3 / 0.3 / 0.4 | 8 / 15 | 0.648 | 0.646 |
| 0.4 / 0.4 / 0.2 | 5 / 15 | 0.662 | 0.660 |
| 0.2 / 0.5 / 0.3 (0.2.1 defaults) | 5 / 15 | 0.638 | 0.635 |

The 0.3.0 defaults beat the 0.2.1 ones on all 6 held-out halves (+0.006 to +0.016) and
the acoustic-heavy combinations beat both on every half.

Bundled samples (`assets/`), score with the 0.2.1 -> 0.3.0 constants, same components:

| file | text | acoustic distance | PER | WER | 0.2.1 | 0.3.0 |
|---|---|--:|--:|--:|--:|--:|
| `developer.wav` (good) | hello I am a developer | 6.32 | 0.00 | 0.00 | 97.4 | 98.9 |
| `developer_error_an.wav` (one wrong word) | same | 6.70 | 0.12 | 0.20 | 84.4 | 86.7 |
| `developer1.wav` (bad) | same | 11.43 | 0.56 | 0.40 | 47.0 | 47.4 |
| `example.mp3` (bad) | Hello, how are you? | 7.55 | 0.33 | 0.75 | 55.7 | 59.0 |
| `harvard.wav` (good, long) | `harvard_text.txt` | 10.06 | 0.12 | 0.00 | 83.7 | 81.5 |
| `harvard_2_errors.wav` | `harvard_text.txt` | 8.14 | 0.09 | 0.19 | 83.8 | 83.8 |
| `reference.wav` (wrong sentence) | hello I am a developer | 11.65 | 0.81 | 1.00 | 16.1 | 18.7 |

With the grid maximum (0.8/0.2/0, bounds 5/22) the wrong sentence would score 43 and
`example.mp3` 86; with 0.7/0.2/0.1 and 5/18, 38 and 72. The acoustic distance of a
wrong sentence (11.7) is the same as that of a bad reading of the right one (11.4), so
only the phone and word terms separate them.

### Why 0.3 / 0.4 / 0.3, bounds 6 / 15

- The corpus asks for a heavy acoustic weight (0.7 to 0.8): the human "total" is
  mostly an accuracy judgment on short sentences by Mandarin learners, half of them
  children, against one adult gTTS voice, and the DTW distance is the best single
  proxy for that native-vs-non-native distance (0.666 alone).
- The product asks for the opposite: the score has to react to a specific mistake (a
  wrong word, a missing syllable) and to collapse on a wrong sentence, which the
  acoustic distance does not see. Among the combinations that keep a wrong sentence at
  or under 20, a good reading at or above 85 (`developer.wav`) and the ordering of the
  bundled samples, 0.3/0.4/0.3 with 6/15 is the best on the full sample and the one
  selected on 5 of the 6 training halves (0.3/0.3/0.4 with 8/15 on the sixth); it
  keeps 70 % of the weight on the phone and word terms.
- The gain over the 0.2.1 defaults is small (Spearman 0.638 -> 0.652, +0.014 held out
  on every fold) and comes from moving 0.1 of weight from the phones (0.5 -> 0.4, PER
  is now a working but weaker signal than the acoustic distance) to the acoustic term
  and from `ACOUSTIC_DISTANCE_GOOD` at 6, where a clean native-like reading sits. The
  headroom (0.68) stays with the acoustic term; a non-linear combination (acoustic
  term gated by the word error rate) is the next thing to try, not more grid search.
- Caveats unchanged: single population and single reference voice, sentences of 2 to
  12 words. The reference cache key changed with the TTS backends, so gTTS
  re-synthesized all 500 references for this run; 180 acoustic distances differ
  slightly from the 0.2.1 run (gTTS is not bit-for-bit reproducible), without any
  effect on the acoustic correlations (identical to three decimals).

### Results, 0.2.1 (2026-08-15, history)

Kept as it was written; the phonemization bug described in the interpretation is
fixed in 0.3.0 and the recommendation was superseded by the section above.

- N = 500 utterances of the test split (123 speakers), OpenPronounce 0.2.1 (worktree of
  commit ea7f4df), `facebook/wav2vec2-large-960h` + `facebook/wav2vec2-lv-60-espeak-cv-ft`,
  default constants (`SCORE_WEIGHTS = {acoustic: 0.2, phonemes: 0.5, words: 0.3}`,
  `ACOUSTIC_DISTANCE_GOOD/BAD = 5/15`).
- CPU only, 6 torch threads on a 16-core desktop (other jobs running in parallel),
  torch 2.13, transformers 5.15: 3.5 s per utterance on average, 29 min for the run.
- No gTTS failure, no skipped utterance. Raw data: `results/speechocean762.csv`.

Human total: mean 7.26, sd 1.54 (the corpus is skewed toward good readings). Our
score: mean 49.8, sd 27.7.

Correlation with the human ratings:

| pair                                   | Pearson | Spearman |
|----------------------------------------|--------:|---------:|
| score vs human total                   |   0.571 |    0.583 |
| score vs human accuracy                |   0.535 |    0.553 |
| score vs human fluency                 |   0.532 |    0.515 |
| score vs human prosodic                |   0.552 |    0.541 |
| acoustic_distance vs human accuracy    |  -0.608 |   -0.642 |
| acoustic_distance vs human total       |  -0.639 |   -0.666 |
| phoneme_error_rate vs human accuracy   |  -0.018 |   -0.372 |
| phoneme_error_rate vs human total      |  -0.041 |   -0.409 |
| word_error_rate vs human accuracy      |  -0.503 |   -0.548 |
| word_error_rate vs human total         |  -0.519 |   -0.551 |

Mean score per human total: 3 -> 14, 4 -> 21, 5 -> 23, 6 -> 34, 7 -> 40, 8 -> 61,
9 -> 68 (monotonic, but the within-level standard deviation is 20-24 points).
Aggregated per speaker (106 speakers with at least 3 utterances), the Spearman
correlation between mean score and mean human total is 0.78.

Word level, "mispronounced" = human word accuracy < 5 (168 of 3,146 words, 5.3%):

| metric                              | value |
|-------------------------------------|------:|
| micro recall (bad words we flagged) | 0.696 |
| micro precision (flagged words that are bad) | 0.108 |
| micro F1                            | 0.188 |
| words flagged                       | 1,080 (34% of all words) |

Grid search on the score formula (recomputed from the stored components, Spearman with
human total; weights on a 0.1 grid, GOOD in {0..8}, BAD in {8..30}):

| Spearman | w_acoustic | w_phonemes | w_words | AD_GOOD | AD_BAD |
|---------:|-----------:|-----------:|--------:|--------:|-------:|
| 0.674 | 0.8 | 0.1 | 0.1 | 7 | 18 |
| 0.674 | 0.8 | 0.1 | 0.1 | 5 | 18 |
| 0.674 | 0.8 | 0.1 | 0.1 | 6 | 18 |
| 0.674 | 0.8 | 0.1 | 0.1 | 4 | 20 |
| 0.672 | 0.8 | 0.1 | 0.1 | 5 | 15 |
| 0.666 | 1.0 | 0 | 0 | any | any (acoustic distance alone) |
| 0.583 | 0.2 | 0.5 | 0.3 | 5 | 15 (current defaults) |

Two-fold cross-validation (3 random splits, best combination chosen on one half,
evaluated on the other): test Spearman 0.61 to 0.74 for the tuned weights against
0.51 to 0.66 for the defaults on the same halves, i.e. a gain of +0.07 to +0.10 on
every split. The winning weights are the same on every half (0.8/0.1/0.1); the acoustic
bounds are not identified (anything from 4-7 / 15-20 gives the same result).

### Interpretation

- The score is a usable but coarse proxy for human judgment: Spearman 0.58 with the
  human total at the utterance level, 0.78 once averaged per speaker. It ranks a
  learner correctly, it does not rank a single sentence reliably (sd of 20+ points at
  a given human level). For reference, the GOP + SVR baseline of the speechocean762 paper
  is around Pearson 0.64 on the total score and later supervised systems (GOPT and
  followers) reach 0.74 to 0.77 (from memory of the literature, to be checked before
  quoting).
- The signal is carried by the acoustic distance (DTW on Wav2Vec2 embeddings vs the TTS
  reference): alone it reaches Spearman 0.67, more than the combined score. The word
  error rate is second (0.55). The phoneme error rate is the weakest component
  (Spearman 0.41, Pearson close to 0), even though it has the largest weight (0.5),
  which is why the default score under-performs its own best component.
- Part of the PER weakness is a bug in `openpronounce/phones.py`, not a modelling
  limit: when espeak merges neighbouring words ("would have to" comes back as one
  group), `_expected_phones_by_word` falls back to per-word phonemization with
  `Separator(phone=" ", word=" ")`, which phonemizer rejects (`ValueError`, identical
  separators). Every word then gets an empty phone list, PER becomes
  `len(heard) / 1` (values up to 49 in the CSV) and no word can be flagged. This hits
  84 of the 500 utterances (17%). On the 416 unaffected utterances the PER reaches
  Spearman 0.53 with the human total, and word-level recall is 0.88 (precision 0.11).
- Word-level flagging is far too eager: one word in three is flagged while the raters
  reject one in twenty, so precision is 0.11 with recall 0.70. This is the phone
  recognizer plus a 0.5 edit-ratio threshold on very short words on child speech; the
  feedback list is not usable as is on this population.
- Recommendation: `SCORE_WEIGHTS = {acoustic: 0.8, phonemes: 0.1, words: 0.1}` with
  `ACOUSTIC_DISTANCE_GOOD/BAD = 5/18` (Spearman 0.674 vs 0.583). The gain is material
  and stable across splits, so it is not an artefact of the 500-utterance sample.
  Two caveats before adopting it: the corpus is a single population (Mandarin L2,
  many children, very short sentences, one TTS voice as reference), and a heavier
  acoustic weight also makes the score more sensitive to voice/recording mismatch
  with the gTTS reference. Re-running the grid search after fixing the phonemizer
  fallback is advisable, since the PER weight would probably not stay at 0.1 with a
  working PER. The mean score also drops from 50 to 40 with the tuned weights and
  bounds 5/15 (51 with 5/18): the bounds shift the level, the weights the ranking.

## Word model and scorer (0.4)

`benchmarks/word_model.py` and `benchmarks/scorer.py` train and evaluate the models shipped
in `openpronounce/models/` (plain JSON, read with numpy). Both use the whole corpus: the
2,500 utterances of the train split to fit, the 2,500 of the test split (disjoint speakers)
to report. Nothing below was tuned on the test split.

```bash
python benchmarks/word_model.py --extract train && python benchmarks/word_model.py --extract test   # posteriors + layer-20 hidden states, ~30 min and 1.4 GB each on CPU
python benchmarks/word_model.py --features train && python benchmarks/word_model.py --features test  # forced alignment + GOP features, ~1 min each
python benchmarks/word_model.py --fit --out openpronounce/models/word_mdd_en.json
python benchmarks/scorer.py --features train && python benchmarks/scorer.py --features test          # utterance features + synthetic negatives
python benchmarks/scorer.py --fit --out openpronounce/models/scorer_en.json
```

### Word model

The expected phones (espeak's own spelling, one recognizer token each when it exists) are
force-aligned on the frame posteriors of `wav2vec2-lv-60-espeak-cv-ft` (CTC Viterbi). Each
phone gets 21 features: GOP margin (mean and min over its frames of the target log posterior
minus the best one), target log posterior, "DNN" GOP, context max, likelihood drop when the
phone is deleted or replaced by a near phone (computed on a window of two phones on each
side), duration, position in the word, word length, function word, neighbours' margins,
utterance statistics (mean margin, alignment likelihood per phone, heard/expected ratio),
plus an embedding: the recognizer's hidden states of layer 20 (of 24) averaged over the
phone's frames widened by one on each side, reduced to 16 dimensions by a PCA fitted on the
train split. An MLP (64-32, one-hot phone identity appended) predicts `phone accuracy < 2`;
its out-of-fold probabilities are aggregated per word (min/max/mean, "any") with the GOP
aggregates and gradient-boosted trees predict `word accuracy < 5`. The threshold (0.35) is
the best F1 on the train split.

Test split, 43,781 phones with a human score and 15,967 words (2026-10-26):

| level | system | AUC | AP | P@R.5 | P@R.7 | at the default threshold |
|---|---|--:|--:|--:|--:|---|
| phone, accuracy < 2 | raw GOP margin | 0.712 | 0.405 | 0.423 | 0.185 | |
| phone, accuracy < 2 | MLP, GOP features | 0.851 | 0.590 | 0.597 | 0.482 | |
| phone, accuracy < 2 | MLP, GOP features + embedding (shipped) | 0.865 | 0.642 | 0.662 | 0.523 | |
| word, accuracy < 5 (5.4 %) | 0.3.0 rule | 0.804 | 0.208 | | | P 0.204, R 0.750, 19.9 % flagged |
| word, accuracy < 5 | min GOP margin alone | 0.883 | 0.324 | 0.319 | 0.236 | |
| word, accuracy < 5 | logistic phone + logistic word | 0.900 | 0.399 | 0.387 | 0.268 | P 0.381, R 0.507 |
| word, accuracy < 5 | MLP phone + trees word, GOP features | 0.913 | 0.456 | 0.503 | 0.322 | P 0.425, R 0.582, 7.4 % flagged |
| word, accuracy < 5 | **same + embedding (shipped)** | **0.924** | **0.525** | **0.537** | **0.376** | **P 0.533, R 0.506, 5.1 % flagged** |
| word, accuracy < 7 (8.1 %) | 0.3.0 rule | 0.760 | 0.268 | | | P 0.264, R 0.648 |
| word, accuracy < 7 | MLP phone + trees word, GOP features | 0.872 | 0.462 | 0.448 | 0.300 | P 0.494, R 0.451 |
| word, accuracy < 7 | same + embedding (shipped) | 0.886 | 0.518 | 0.498 | 0.343 | P 0.610, R 0.386 |

Pearson between the phone probability and the human phone accuracy: 0.54 (0.48 for the raw
GOP margin, in line with the 0.44 to 0.50 reported for GOP variants on this corpus). On the
500-utterance sample of the sections below (`word_detection.py --report`): P 0.509, R 0.512,
F1 0.510, 5.4 % flagged, against 0.198 / 0.726 / 0.311 / 19.6 % for the 0.3.0 rule.

The embedding buys precision: at equal recall (0.5) one flagged word in two is right instead
of one in two and a half, and the words flagged drop from 7.4 % to 5.1 %. Alone, the
embedding reaches a phone AUC of 0.80: the top layers of a CTC recognizer mostly repeat
what its posteriors say. Layer 20 beats 8, 12, 16 and 24 (0.853 to 0.863 at 64
dimensions); averaging over the span is better than over the segment between neighbouring
phones (0.858 at best); 16 dimensions beat 32, 64, 128 and 256; gradient boosting on the
same inputs matches the MLP (0.866) and a wider MLP does not help. The hidden states cost
17 ms per utterance (the forward pass already computes them).

What did not help: ZIPA small (a 64M-parameter multilingual IPA recognizer, 70 MB int8, 0.1 s
per utterance) as phone model gives a raw word AUC of 0.67 and 0.83 with the classifier, so
the espeak wav2vec2 model stays; normalizing the expected phones before alignment (merging
ɔ/ɑ, ɾ/t...) changes nothing measurable; training the word stage on in-sample rather than
out-of-fold phone probabilities neither.

### Scorer

53 utterance features: aggregates of the per-phone GOP (mean, std, min, 10th percentile,
share of phones under -1 and -3 nats, share with another token heard), alignment likelihood
per phone, share of plausible phones, speech rate, phone and word duration variability,
inter-word gaps, pauses (gaps over 250 ms), leading and trailing silence, articulation rate,
aggregates of the word model's probabilities, and a cheap prosody pass (YIN pitch mean,
spread and range in semitones, slope, frame-to-frame variation, RMS mean, variability and
dynamic range, voiced ratio, energy dips, duration). One gradient-boosted regressor per
target, 400 trees, trained on the 2,500 real readings plus 742 synthetic negatives (the
posteriors of one utterance against the text of another, rated 1; the first 60 % of a
reading against its full text, rated 0.4 times the original) because the corpus has no wrong
or incomplete reading and trees cannot extrapolate. A linear recalibration fitted on
out-of-fold predictions undoes the shrinkage of the trees toward the mean. A logistic
regression on the alignment features tells a real reading from a synthetic negative
(`sentence_probability`).

Test split, Pearson / Spearman with the human scores (2026-10-26):

| system | total | accuracy | fluency | prosodic |
|---|--:|--:|--:|--:|
| 0.3.0 formula (500-utterance sample, see below) | 0.633 / 0.652 | 0.603 / 0.631 | 0.595 / 0.590 | 0.618 / 0.622 |
| ridge, 35 alignment features | 0.736 / 0.717 | 0.721 / 0.681 | 0.760 / 0.752 | 0.745 / 0.739 |
| trees, 35 alignment features | 0.745 / 0.718 | 0.722 / 0.678 | 0.781 / 0.768 | 0.761 / 0.752 |
| trees, 53 features + negatives, word model without embedding | 0.774 / 0.727 | 0.721 / 0.685 | 0.807 / 0.765 | 0.792 / 0.748 |
| trees, 53 features, real readings only | 0.786 / 0.742 | 0.754 / 0.699 | 0.805 / 0.777 | 0.796 / 0.760 |
| **trees, 53 features + negatives (shipped)** | **0.790 / 0.733** | 0.742 / 0.693 | **0.824 / 0.775** | **0.807 / 0.755** |
| ridge, WavLM-base-plus mean+std of all 13 layers | 0.715 / 0.709 | 0.693 / 0.665 | 0.791 / 0.797 | 0.785 / 0.789 |
| ridge, 53 features (word model without embedding) + WavLM all layers | 0.748 / 0.732 | 0.730 / 0.693 | 0.804 / 0.804 | 0.798 / 0.795 |

The seven word-model aggregates carry the embedding's gain to the scorer: 0.774 → 0.790 on
the total, 0.721 → 0.742 on accuracy, with the other 46 features unchanged. Mean absolute
error of the shipped scorer: 0.71 (total), 0.77, 0.62, 0.64 on the 0-10 scale. Synthetic
negatives of the test split: wrong sentence 1.18 on average (90th percentile 1.63),
truncated reading 3.0; `sentence_probability` AUC 0.996 (1.4 % of the real readings fall
under 0.5, all poor ones). The negatives cost 0.01 on accuracy and buy the behaviour on
wrong sentences.

WavLM-base-plus (95M parameters, 0.3 s per utterance on CPU) added 0.01 on the total and
0.02 on fluency and prosody over the alignment features when measured against the 0.774
scorer (not re-run since): not worth a third model for now; the caches and the comparison
are in `scorer.py --wavlm` / `--fit --wavlm-layers`. For
reference, the published supervised systems on this corpus reach 0.74 to 0.80 on the total
(GOPT and followers, from the literature, not re-run here).

Bundled samples (`assets/`), `score` / `formula_score`, English, CPU:

| file | text | score | accuracy | fluency | prosodic | match | flagged | formula | time (acoustic on / off) |
|---|---|--:|--:|--:|--:|--:|---|--:|---|
| `developer.wav` (good) | hello I am a developer | 76.6 | 85.0 | 78.7 | 78.6 | 1.00 | | 98.9 | 1.06 s / 0.58 s |
| `developer_error_an.wav` | same | 81.3 | 86.8 | 85.0 | 79.2 | 0.99 | | 86.5 | 1.16 s / 0.60 s |
| `developer1.wav` (other voice) | same | 73.9 | 81.5 | 79.6 | 76.9 | 0.99 | | 47.2 | 1.39 s / 0.81 s |
| `reference.wav` (wrong sentence) | hello I am a developer | 14.6 | 15.6 | 18.7 | 15.2 | 0.00 | am, developer | 18.7 | 1.09 s / 0.61 s |
| `example.mp3` | Hello, how are you? | 63.8 | 69.1 | 75.5 | 69.9 | 0.97 | how (0.49) | 58.8 | 1.21 s / 0.68 s |
| `harvard.wav` (18 s) | `harvard_text.txt` | 75.9 | 82.7 | 78.5 | 71.5 | 1.00 | | 81.9 | 8.2 s / 4.6 s |
| `harvard.wav`, first half only | `harvard_text.txt` | 22.1 | | | | 0.00 | 29 of 43 | | |

The scale is the raters': 8/10 is a typical good learner reading, so a clean adult reading
scores around 80 rather than 99. `developer1.wav` was "bad" for the formula because of its
acoustic distance to the gTTS voice (11.4); the model hears nothing the raters would reject,
which may or may not be right, nobody rated it.

## Who it works for (`groups.py`)

```bash
python benchmarks/groups.py        # shipped scorer on the test split, by age group and gender
```

| group | utterances | Pearson with the human total | MAE (0-10) |
|---|---:|---:|---:|
| adults | 1,220 | 0.851 | 0.7 |
| children, 12 and under | 1,040 | 0.668 | 0.7 |
| teenagers, 13 to 17 | 240 (12 speakers) | 0.456 | 0.7 |
| women | | 0.827 | 0.7 |
| men | | 0.756 | 0.7 |

The error is the same everywhere; the correlation drops where the ratings have little
spread (the teenagers) and where Wav2Vec2, trained on adults, recognizes less well (the
children).

## English without espeak

The expected sounds of English come from the CMU pronouncing dictionary
(`openpronounce.g2p`), the models are the ones trained with espeak's phones, unchanged.
On the test split 12 % of the words get a different phone string, 0.12 % are missing
from the dictionary. Word AUC 0.928 (0.924 with espeak), precision of the flagged words
0.549 (0.53), total 0.791 (0.790), fluency 0.827.

## Free speech (`free.py`)

The prompt is hidden: the recording is transcribed (`facebook/wav2vec2-large-960h`,
the word model of the library) and the transcript replaces the text. Each utterance is
trimmed by `audio.split_speech`, as `free.assess` does.

```bash
python benchmarks/free.py --transcribe train && python benchmarks/free.py --transcribe test   # 50 min each on CPU
python benchmarks/free.py --features train && python benchmarks/free.py --features test
python benchmarks/free.py --fit --out openpronounce/models/free_scorer_en.json
python benchmarks/free.py --report
```

Pearson with the human ratings, test split, 2,500 utterances:

| system | total | accuracy | fluency | prosodic |
|---|---:|---:|---:|---:|
| read-speech scorer with the true prompt (reference) | 0.790 | 0.74 | 0.82 | 0.81 |
| read-speech scorer fed with the transcript | 0.541 | 0.478 | 0.620 | 0.617 |
| trained on the transcript, read-speech features | 0.674 | 0.643 | 0.753 | 0.740 |
| trained, features that need no text at all | 0.671 | 0.631 | 0.745 | 0.732 |
| trained, all free-speech features | 0.672 | 0.640 | 0.752 | 0.739 |
| the same, blended with the recognizer's hidden states (shipped) | 0.708 | 0.681 | 0.787 | 0.777 |

| subset | utterances | total | MAE (0-10) | word error rate against the prompt |
|---|---:|---:|---:|---:|
| adults | 1,220 | 0.820 | 0.78 | 0.45 |
| children, 12 and under | 1,040 | 0.417 | 0.82 | 0.53 |
| teenagers | 240 | 0.310 | 0.66 | 0.30 |
| 1 to 4 words | 387 | 0.469 | 0.82 | 0.61 |
| 5 to 8 words | 1,701 | 0.710 | 0.79 | 0.45 |
| 9 words and more | 412 | 0.812 | 0.74 | 0.42 |

The shipped English scorer averages two answers: the trees on the measures above, and
a linear model on the hidden states of the phone recognizer (layer 20, mean and standard
deviation over the frames), which keep what the measures drop. No model is added, the
recognizer is already loaded. The weight of each half was chosen on the train split,
speakers kept apart. The other languages keep the trees alone.

The transcript is wrong on almost one word in two (these are learners, and the
recognizer has no language model), yet the three trained systems are equivalent: what
carries the score is how sure the phone recognizer is of what it hears, the rhythm and
the pitch, not the words. Short utterances give too little of that. For children the
free-speech score should not be used.

### Longer recordings

No rated corpus of long learner speech can be used here, so the test is built: six
utterances of one speaker of the test split glued into one recording of about 22 s,
scored by `free.assess` and compared with the mean of the six human totals. 30 speakers
(17 adults, 13 under 18), so the intervals are wide: read these as orders of magnitude.

| segmentation (`audio.split_speech`) | adults, Pearson | adults, MAE (0-100) | under 18, Pearson |
|---|---:|---:|---:|
| cut at pauses of 0.5 s, short parts joined (shipped) | 0.74 to 0.77 | 8 to 9 | 0.1 |
| cut at pauses of 0.7 s | 0.67 to 0.72 | 9 to 11 | 0.1 |
| cut at pauses of 0.5 s, short parts kept apart | 0.64 to 0.71 | 9 to 10 | 0.1 |
| parts of up to 20 s (first version) | scores 20 points too low | | |

The two figures are for the original silences between the utterances (1.2 s and more)
and for silences shortened to 0.6 s. The scorer learned from single sentences: fed
with 20 s at once it reads the pauses between sentences as hesitations, hence the cut
at every pause of half a second. The mean score stays within 3 points of the raters'.

Time on CPU (16 threads, models loaded, best of three):

| recording | `--fast` | both recognizers, no synthetic reference | free speech |
|---|---:|---:|---:|
| 1.9 s | 0.45 s | 0.77 s | 0.77 s |
| 18 s | 3.3 s | 6.2 s | 5.3 s |
| 3 min | | | 52 s |

Peak memory 3.6 GB with both recognizers loaded.

## Missing words, extra speech (`missing_extra.py`)

The learners of the corpus read what they are given, so the cases are built from the
text: a word added to the sentence is a word the speaker did not say, a word removed
from it is a word said in excess. Test split, 2,500 sentences.

```bash
python benchmarks/missing_extra.py --fit --out openpronounce/models/word_missing_generic.json
python benchmarks/missing_extra.py --report
```

| | found | reported by mistake |
|---|---:|---:|
| word not said (2,500 words added to the texts) | 82.9 % | 0.71 % of the 15,967 words really read |
| speech in excess (2,490 words removed from the texts) | 62.9 %, 69.8 % from 3 phones up | 2.8 % of the untouched sentences |

Of the 114 read words called missing, 62 were rated below 5 by the raters. With them,
the words reported go from precision 0.548 / recall 0.490 to 0.537 / 0.501. The missing
word model is a logistic regression on nine alignment features (likelihood lost when the
word is deleted, plausibility of its phones nearby, frames and heard phones per expected
phone); its threshold lets 0.5 % of the read words of the train split through. Extra
speech is a count: confident heard phones between two words, plus half of what each
neighbour holds beyond its own phones, two or more. A learned detector on the same counts
did no better (62 % found at the same false-alarm rate).

## Other languages (`native_calibration.py`)

No rated learner corpus exists for them, so three things are checked: what the generic
models cost on English, how native speakers are treated in each language, and whether
natives come out above learners where recordings of both can be had
([further down](#free-speech-outside-english)).

| on the English test split | English models | generic models |
|---|---:|---:|
| word AUC (accuracy below 5) | 0.924 | 0.911 |
| precision / recall of the reported words | 0.533 / 0.506 | 0.463 / 0.491 |
| Pearson with the human total | 0.790 | 0.771 |

The generic word model drops the phone identity and the recognizer's hidden states; the
variant that kept the hidden states did better on English (0.917) and worse elsewhere.

```bash
python benchmarks/word_model.py --fit --generic --no-embeddings --out openpronounce/models/word_mdd_generic.json
python benchmarks/scorer.py --generic --features train && python benchmarks/scorer.py --generic --features test
python benchmarks/scorer.py --generic --fit --out openpronounce/models/scorer_generic.json
python benchmarks/free.py --generic --features train && python benchmarks/free.py --generic --features test
python benchmarks/free.py --generic --fit --out openpronounce/models/free_scorer_generic.json
python benchmarks/native_calibration.py english.csv --lang en --free --reference   # native readers of English
```

`--generic` reads English through the generic word model, as any other language is
read. The last line writes into both scorers what native readers of English look like
to them (here the 120 `dev` recordings of FLEURS), the reference of every
`native_<lang>.json`.

Native speakers come from [FLEURS](https://huggingface.co/datasets/google/fleurs)
(Conneau et al., 2022, CC BY 4.0): people reading Wikipedia sentences on their own
devices, 19 to 22 words and 9 to 13 seconds on average, far longer than the 2 to 10
words the models learned from. Sentences with digits are left out. The `dev` split
(120 recordings per language) served to choose the variants and the thresholds below;
every figure of this table comes from the `test` split (220 recordings per language,
26,600 words outside English), which played no part in those choices. `--fast` path.

| | score | one in ten under | words reported | old rule | a replaced word is found | another sentence is caught | a missing word is found | speech in excess is found |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| English, its own models | 73.7 | 68 | 0.3 % | 3.0 % | 66 % | 100 % | 91 % | 87 % |
| English, generic models | 73.9 | 69 | 0.4 % | 3.0 % | 78 % | 100 % | 91 % | 87 % |
| French | 76.0 | 72 | 1.6 % | 6.2 % | 85 % | 95 % | 83 % | 76 % |
| Spanish | 76.6 | 74 | 0.4 % | 5.4 % | 79 % | 96 % | 63 % | 85 % |
| German | 76.3 | 72 | 0.9 % | 6.6 % | 83 % | 87 % | 74 % | 80 % |
| Italian | 72.5 | 68 | 1.0 % | 12.8 % | 77 % | 95 % | 88 % | 82 % |
| Portuguese | 74.6 | 70 | 0.9 % | 26.8 % | 74 % | 87 % | 83 % | 70 % |
| Dutch | 75.7 | 72 | 0.8 % | 23.3 % | 75 % | 93 % | 55 % | 60 % |

How to read it. A native speaker is not at 100: the scale is the one of the raters of
speechocean762, and English natives themselves get 74 on these sentences (77 on those of
12 words at most, 73 from 20 words up). Natives of the six other languages now sit
between 72 and 77, which is the work of the native reference described below: without
it Portuguese and Dutch natives were at 70 and 68. "Words reported" are false alarms,
the speakers being native. The other columns are built from
the text: one word replaced by another word of the corpus (three draws per recording),
the text of another recording, one word added to the text, one word of four letters or
more removed from it. A missing word is called so by mistake for 0.2 to 0.95 % of the
words read, and speech in excess is found by mistake in 1.4 to 8.6 % of the untouched
recordings (English: 8.6 %).

Three settings per language came out of the `dev` split:

- `Language.variants`, the recognizer tokens accepted for a phone espeak writes. Those
  first found on a synthetic voice hold on real speakers (Portuguese natives go from 63
  to 70, Dutch ones from 63 to 68, and their words are told from a replaced word better:
  AUC 0.989 to 0.994 and 0.982 to 0.984). Dutch gets three more (ɪː heard eː, ɔː heard oː,
  ɵ heard ə, y or ʊ: AUC 0.988). Wider sets, down to every phone natives are heard to say
  otherwise, raise the native scores by up to 2 points and bring nothing to the detection:
  left out.
- `Language.word_threshold`. The generic model separates a native's own words from a
  replaced word as well as in English (AUC 0.986 to 0.997, English 0.996), but its
  probabilities are lower: at its own threshold (0.31, tuned on learners of English) a
  replaced word is found 34 to 68 times in 100, against 78 in English. The threshold of
  each language (0.06 to 0.22) is the one that finds it as often as in English on `dev`;
  the reports show the probability moved onto the model's scale, so 0.31 means the same
  everywhere. Assumption: what holds for a replaced word holds for a mispronounced one.
  Only rated learners could tell.
- `models/native_<lang>.json`, the native reference. The scorer learned on learners of
  English what a good reading looks like to the recognizer, and the recognizer does not
  agree with espeak equally well in every language: on Dutch and Portuguese natives the
  share of sounds heard as another sound sits 4 standard deviations above that of
  English natives, and the scorer reads that as an accent. So the 19 features that come
  from the alignment are moved, before scoring, by the gap between the natives of the
  language and those of English (the mean of 120 `dev` recordings on each side). Speech
  rate, pauses, rhythm and pitch are left alone. On `test`:

  | natives, score | French | Spanish | German | Italian | Portuguese | Dutch |
  |---|---:|---:|---:|---:|---:|---:|
  | without the reference | 74.1 | 76.6 | 73.4 | 73.0 | 69.6 | 68.2 |
  | with it (shipped) | 76.0 | 76.6 | 76.3 | 72.5 | 74.6 | 75.7 |

  The words reported and the three detections built on the word models do not move,
  the reference touching the scorer only. What moves is the catch of another sentence
  (German 82 to 87 %, Italian 98 to 95 %, French 96 to 95 %) and the score such a
  recording gets, which rises from 18 to 31 in Portuguese and from 22 to 35 in Dutch:
  it is flagged as unreliable, and that flag is what to read. Assumption: a learner
  of Dutch strays from Dutch natives the way a learner of English strays from English
  ones. The corpora below can only test it where the reference is small.

What this does not say: how the scores follow human raters for learners of these
languages. German, Portuguese and Dutch recordings are all tagged as male voices in the
corpus, Italian ones almost all as female. A foreign name is pronounced by espeak with
its foreign sounds and reported when read the local way ("Aerosmith" in a French
sentence).

Two more things these recordings showed:

- **Long sentences.** The scorer never saw more than 10 words. Scoring a long reading by
  windows of 10 words was tried on 300 readings made of 2 to 4 test utterances of one
  speaker (20 words on average): Pearson with the mean human rating 0.88 as a whole and
  0.90 by windows for the total, 0.73 and 0.85 for fluency, 0.81 and 0.86 for prosody.
  Either way the score sits 9 to 10 points under the raters, and natives do not gain.
  Not shipped: the readings are glued, which favours windows by construction.
- **Quiet recordings.** The level does not change the score (the models normalize it):
  20 dB less move it by a point. A 16-bit file loses 1.5 points at -60 dBFS and 9 at -70,
  hence `too_quiet` below -60 (it was -40, which most of the Dutch recordings tripped).
  FLEURS files are 32-bit floats, some at -65 dBFS: 16 % of the English ones are called
  too quiet and are scored like the others.

```bash
python benchmarks/native_calibration.py manifest.csv --lang nl    # columns: audio, text
```

prints, for your own native recordings, the scores, the words reported, the threshold
that matches English and the phones espeak and the recognizer disagree on. With
`--shift native_nl.json` it writes the native reference of the language, and with
`--free` it measures speech without a text as well.

## Free speech outside English

`models/free_scorer_generic.json` is the free-speech scorer trained with the generic
word model. On the speechocean762 test split with the prompts hidden it follows the
raters as the English one does: Pearson 0.671 with the total against 0.672, 0.787 for
adults, 0.372 for children of 12 and under. The other languages use it with their
native reference, which in free speech also covers the confidence of the recognizer.

**Natives reading, text hidden.** FLEURS `test`, 120 recordings per language, the
reference coming from `dev`.

| | without the reference | with it (shipped) | the same recordings with their text | words mistranscribed |
|---|---:|---:|---:|---:|
| English, generic models | 70.7 | | 73.8 | 14 % |
| French | 70.0 | 73.0 | 76.2 | 18 % |
| Spanish | 72.8 | 72.3 | 76.6 | 7 % |
| German | 66.3 | 72.5 | 76.1 | 12 % |
| Italian | 69.6 | 70.4 | 72.8 | 8 % |
| Portuguese | 61.1 | 72.1 | 74.5 | 15 % |
| Dutch | 58.3 | 70.8 | 75.8 | 21 % |

Without its text a native reading loses 2 to 5 points, in English too. Between two
natives the two scores hardly agree (Pearson 0.1 to 0.5 across recordings): natives
stand within 3 to 6 points of each other, and that is the noise of the measure.

**Natives speaking without a script.** 100 stretches of 8 to 25 seconds per corpus,
4 to 13 per speaker.

| corpus | score | fluency | prosody | words mistranscribed |
|---|---:|---:|---:|---:|
| VoxPopuli, English (speakers of every country) | 72.4 | 65.6 | 66.1 | 19 % |
| VoxPopuli, French | 73.1 | 65.9 | 65.7 | 21 % |
| VoxPopuli, Spanish | 72.6 | 65.1 | 64.1 | 13 % |
| VoxPopuli, German | 71.6 | 65.5 | 65.1 | 18 % |
| VoxPopuli, Italian | 70.8 | 63.8 | 62.9 | 31 % |
| VoxPopuli, Dutch | 71.7 | 65.1 | 64.0 | 26 % |
| CIEMPIESS, Mexican radio talk | 67.2 | 56.7 | 56.1 | 21 % |
| CORAA-MUPE, life stories told by eight elderly Brazilians | 65.7 | 58.6 | 58.9 | 45 % |

Speeches in parliament score like read sentences. Talk that is really unprepared loses
5 to 7 points, all of them in fluency and prosody: the scorer learned from read
sentences, where a pause or a restart is a fault. A fluent native who hesitates is
marked down for it, and so will a fluent learner be.

**Natives against learners.** Two public corpora hold both, without ratings:
[L2FOC](https://hdl.handle.net/11403/bianca-maria-de-paolis) (French spoken by Italians,
Italian spoken by French speakers, natives of both, short answers to questions about
pictures) and [Heroico](https://www.openslr.org/39/) (Spanish read by Mexican cadets
and by American ones). The figure is the chance that a native picked at random scores
above a learner picked at random, speakers being compared on the mean of 12 to 15
answers of about 3 seconds; in brackets the 95 % interval.

| | natives | learners | free speech | with the transcript as the text |
|---|---:|---:|---:|---:|
| French, answers | 82.8 (18 speakers) | 78.2 (14) | 0.93 (0.83 to 1.00) | 0.84 (0.68 to 0.96) |
| Italian, answers | 79.6 (16) | 76.6 (15) | 0.74 (0.54 to 0.90) | 0.73 (0.53 to 0.90) |
| Spanish, read sentences | 82.9 (8) | 81.4 (10) | 0.78 (0.53 to 0.97) | 0.72 (0.45 to 0.96) |

The order is the right one in the three languages and the gap is small: 1.5 to 4.6
points between the two groups, on recordings mostly too short to be called reliable
(under 3 seconds of speech). The native reference changes nothing to that order
(0.93, 0.77 and 0.76 without it). The level of the learners is not in the public
files, so nothing says how far from the natives a rater would have put them.

**The same people reading a text aloud: the order does not hold.** Every L2FOC speaker
also reads a text, about a minute of speech, scored here whole as free speech (13 to
16 parts per recording).

| | natives | learners | a native above a learner |
|---|---:|---:|---:|
| French | 74.8 (18 speakers) | 72.8 (18) | 0.71 (0.52 to 0.87) |
| Italian | 69.5 (16) | 71.5 (18) | 0.36 (0.18 to 0.57) |

In Italian the learners come out on top. The Italians read faster than the French
speakers reading Italian (12.8 sounds per second against 11.0), the recognizer is less
sure of what it hears in quick speech, and the learners read slowly and clearly a
language whose spelling tells how to say it. The scorer cannot tell quick native
speech from unclear speech. The natives of one language being the learners of the
other, the two effects can be told apart: averaged over both languages, being native
is worth 3.8 points in the answers and nothing in the reading, where the French
speakers sit 2 points above the Italians whatever the language.

What this does not say: how the scores follow a teacher's judgement, in any of these
languages. German, Portuguese and Dutch, the three languages the reference moves most,
have no learner recordings here at all.

## Telephone band (`telephone.py`)

The models were trained on 16 kHz recordings. A phone line carries 8 kHz at best. 500
utterances of the test split were sent through two channels and scored again: resampled
to 8 kHz and back, then the same with the band of a land line (300 to 3,400 Hz) and
G.711 coding.

```bash
python benchmarks/telephone.py --sample 500 && python benchmarks/telephone.py --report
```

| | Pearson with the raters | adults only | mean score | a score moves by | one in ten by more than |
|---|---:|---:|---:|---:|---:|
| sentence known, 16 kHz | 0.813 | 0.854 | 72.5 | | |
| sentence known, 8 kHz | 0.789 | 0.852 | 72.0 | 2.8 | 6.9 |
| sentence known, phone line | 0.778 | 0.844 | 71.6 | 3.6 | 7.5 |
| free speech, 16 kHz | 0.684 | 0.784 | 73.6 | | |
| free speech, 8 kHz | 0.655 | 0.788 | 72.5 | 2.5 | 6.0 |
| free speech, phone line | 0.636 | 0.775 | 72.3 | 3.5 | 7.8 |

The agreement with the raters holds, for adults almost untouched; the mean drops by 1 to
3 points and 5 % of the words are reported in every condition. What a line does change
is one person's score: 3.5 points on average between the same recording at 16 kHz
and on the phone, enough to cross a pass mark. The channel is simulated on clean
recordings, without line noise, mobile codec or lost packets, so a real call will do
worse than this.

## Word-level detection (0.3.0 rule)

`benchmarks/word_detection.py` calibrates the rule that flags a word as mispronounced,
on the same 500-utterance sample. It runs the phone recognizer once (`--extract`, the
frame posteriors are cached in `~/.cache/openpronounce/speechocean762/logits`), then
tunes and reports without the model (`--tune`, `--report`), and checks the bundled
samples (`--assets`). The 500 utterances are split by parity of their index: **250 for
tuning (even), 250 held out (odd)**; the constants below were chosen on the tuning half only.

### Rule (0.3.0)

Every wrong phone of a word gets an error confidence in [0, 1]:

- substitution: 1, or `NEAR_PHONE_COST = 0.5` when the two phones are close (voicing
  pairs, tense/lax vowels, ð/z/d, θ/s/t, n/ŋ, h/x...);
- deletion: 1, or `FINAL_DELETION_COST = 0.5` for the last phone of the word;
- extra phone next to a correct one: 1, or `FINAL_EXTRA_COST = 0.25` after the last
  phone of the word (epenthetic vowel, onset of the next word caught by the alignment);
- the value is then scaled by `1 - min(1, p / PHONE_PLAUSIBLE_POSTERIOR)` where `p` is
  the highest posterior of the expected phone in the frames it aligned to (its
  neighbours' frames for a deletion) and `PHONE_PLAUSIBLE_POSTERIOR = 0.05`: an
  expected phone the recognizer found plausible is not a confident error.

A word is flagged when the confidences add up to `PHONE_ERROR_THRESHOLD = 0.4` of its
phones or to `PHONE_ERROR_MIN_EDITS = 2` phones. Its `confidence` is
`min(1, max(sum / phones, sum / 2))`. The 0.2.1 rule was: plain edit distance of at
least 50 % of the phones or 3.

The F1 surface on the tuning half is flat (0.305 to 0.321 for thresholds between 0.4
and 0.7 and 2 to 3 minimum edits): 0.4 / 2 keeps recall (0.73 against 0.48 for the
F1-optimal 0.65 / 2.5) and keeps the bundled examples flagged. The recognizer's own
confidence in the heard phone (peak posterior) is not discriminative on this corpus
(same distribution for confirmed and refuted alarms) and does not enter the rule; the
posterior of the *expected* phone and the phone-pair costs carry the gain.

### Results (2026-08-15, N = 500, 3,146 words)

"bad" = human word accuracy < 5 (168 words, 5.3 %); "lenient" = accuracy < 7 (265
words, 8.4 %). P@R.5 / P@R.7 = best precision reachable at recall >= 0.5 / 0.7 by
moving the two thresholds of that system.

| system | split | P | R | F1 | flagged | P@R.5 | P@R.7 |
|---|---|--:|--:|--:|--:|--:|--:|
| 0.2.1 (edit distance, exact old code) | tune | 0.119 | 0.939 | 0.211 | 41.9 % | | |
| 0.2.1 (edit distance, exact old code) | held-out | 0.119 | 0.872 | 0.210 | 39.2 % | | |
| 0.2.1 rule, new normalization | tune | 0.127 | 0.939 | 0.223 | 39.4 % | 0.161 | 0.161 |
| 0.2.1 rule, new normalization | held-out | 0.126 | 0.860 | 0.221 | 36.5 % | 0.171 | 0.171 |
| confidence rule, no posteriors | tune | 0.152 | 0.902 | 0.261 | 31.5 % | 0.206 | 0.173 |
| confidence rule, no posteriors | held-out | 0.148 | 0.837 | 0.251 | 30.4 % | 0.198 | 0.168 |
| **confidence rule + posteriors (default)** | tune | 0.194 | 0.732 | 0.307 | 20.0 % | 0.227 | 0.198 |
| **confidence rule + posteriors (default)** | held-out | 0.201 | 0.721 | 0.314 | 19.3 % | 0.241 | 0.201 |

Same, lenient label (accuracy < 7):

| system | split | P | R | F1 | flagged | P@R.5 | P@R.7 |
|---|---|--:|--:|--:|--:|--:|--:|
| 0.2.1 (edit distance, exact old code) | tune | 0.179 | 0.866 | 0.297 | 41.9 % | | |
| 0.2.1 (edit distance, exact old code) | held-out | 0.170 | 0.817 | 0.282 | 39.2 % | | |
| 0.2.1 rule, new normalization | tune | 0.188 | 0.851 | 0.307 | 39.4 % | 0.232 | 0.232 |
| 0.2.1 rule, new normalization | held-out | 0.181 | 0.809 | 0.296 | 36.5 % | 0.241 | 0.231 |
| confidence rule, no posteriors | tune | 0.224 | 0.813 | 0.352 | 31.5 % | 0.285 | 0.262 |
| confidence rule, no posteriors | held-out | 0.207 | 0.771 | 0.327 | 30.4 % | 0.301 | 0.262 |
| **confidence rule + posteriors (default)** | tune | 0.269 | 0.619 | 0.375 | 20.0 % | 0.303 | 0.261 |
| **confidence rule + posteriors (default)** | held-out | 0.269 | 0.634 | 0.377 | 19.3 % | 0.328 | 0.262 |

"New normalization" = 0.2.1 thresholds on the plain edit distance, with the 0.3.0
phone normalization (Mandarin tone numbers and aspiration marks dropped, `ɔɹ`/`oɹ`,
`ɜ`/`ɚ` merged, `ai`/`ei`/`au`/`ou` spelled as diphthongs). "No posteriors" = the
0.3.0 costs without the plausibility scaling (what a plain list of phones gives).

Bundled samples (`--assets`), before -> after: `developer.wav` none -> none;
`example.mp3` hello, how -> hello (0.89), how (0.50); `harvard.wav` beer, al -> none;
`developer1.wav` i, developer -> developer (1.00).

### Interpretation

- Half of the false alarms went away (39 % of the words flagged -> 19 %) for a quarter
  of the hits (recall 0.87 -> 0.72 on the held-out half); precision 0.12 -> 0.20 (0.17
  -> 0.27 with the lenient label). Tuning and held-out halves agree within 0.01.
- Precision stays low in absolute terms because the raters are lenient: 90 % of the
  words are rated 10/10, accent traits such as ð -> z, θ -> s, ɪ -> i or a dropped
  final consonant are almost never rated below 5, and 2-phone function words ("the",
  "to", "it", "you") make up most of the remaining alarms. Among words with a given edit
  count, no more than 25 % are rated bad, so an edit-count rule cannot go much further.
- The recognizer's confidence in what it heard does not separate true from false
  alarms; the posterior of the expected phone (GOP-like) and the phone-pair costs do.
  Next steps: a phonetically weighted alignment (the costs inside the DP, not only after
  it), and a small classifier on the per-word features against the human labels.
