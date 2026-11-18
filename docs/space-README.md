---
title: OpenPronounce
emoji: 🎤
colorFrom: purple
colorTo: pink
sdk: docker
app_port: 8000
pinned: false
license: agpl-3.0
short_description: Phoneme-level English pronunciation feedback, open source
---

# OpenPronounce

Record a sentence, get a score, the mispronounced words with the phones you actually said (IPA), and your pitch and energy curves.

Open source (AGPL-3.0), runs on CPU: [github.com/OpenPronounce/OpenPronounce](https://github.com/OpenPronounce/OpenPronounce), `pip install openpronounce`.

This Space runs the FastAPI app from the repository. The first analysis after a cold start takes a while (two Wav2Vec2 models are loaded); the next ones take about 15 s on the free CPU.
