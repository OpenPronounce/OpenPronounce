# Contributing

Bug reports, benchmark results on new data and pull requests are welcome.

```bash
git clone https://github.com/OpenPronounce/OpenPronounce.git && cd OpenPronounce
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[app,dev]"
pytest
```

The tests need espeak-ng and ffmpeg on the system (for the languages other than English), but neither the network nor the model weights.

## Adding a language

A language is one line in `openpronounce/languages.py`, with its espeak-ng voice and a Wav2Vec2 CTC checkpoint that transcribes it, and one small file, its native reference. The scoring models are shared, so nothing has to be trained.

What has to be checked is how native speakers are treated. Collect a hundred native readings or more (the dev split of [FLEURS](https://huggingface.co/datasets/google/fleurs) does it for 100 languages), list them in a CSV with the columns `audio` and `text`, then run:

```bash
python benchmarks/native_calibration.py manifest.csv --lang xx --free --shift openpronounce/models/native_xx.json
```

It prints the scores natives get (English ones get 74 on FLEURS), the share of their words reported, the `word_threshold` that finds a wrong word as often as in English, and the sounds espeak-ng writes one way and the recognizer hears another, candidates for the `variants` of the language. `--shift` writes the native reference, which brings the natives of the language to the score of English natives; set the variants first and run it again, since they change what it measures. `--free` transcribes every recording to cover speech without a text. Keep the figures of a second set of recordings, which you did not tune on, for your pull request.

## Licence of your contribution

OpenPronounce is published under the AGPL and also sold under a commercial licence, which pays for its maintenance. For that to work, a contribution has to come in under a licence that allows both.

By opening a pull request you agree that your contribution is licensed to the project under the MIT licence, and you confirm that you wrote it or have the right to submit it. Add a `Signed-off-by` line to your commits (`git commit -s`) to say so.

You keep the copyright on what you write.
