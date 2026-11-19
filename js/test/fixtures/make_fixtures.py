"""Regenerate the fixtures of the JS tests from the Python library (run from the repo root):

    PYTHONPATH=. .venv-exp/bin/python js/test/fixtures/make_fixtures.py
"""
import json
import os
import random

import numpy as np
from rapidfuzz.distance import Levenshtein

os.environ["OPENPRONOUNCE_G2P"] = "lexicon"
from openpronounce import g2p, gop  # noqa: E402

here = os.path.dirname(os.path.abspath(__file__))
words = ["hello", "world", "butter", "water", "little", "bird", "nurse", "cats", "dogs", "watches", "jumped", "wanted",
         "played", "carried", "stopped", "jumpping", "quickly", "running", "birdbath", "zyxwv", "blorptastic", "42", "007",
         "1999", "123456789", "don't", "o'clock", "rock'n'roll", "the", "a", "elephant", "schmoozle", "happily", "teacher's"]
texts = ["Hello, world!", "'Tis the season.", "It's 5 o'clock; DON'T be late", "rock'n'roll -- forever", "MARK IS GOING TO SEE ELEPHANT"]
fixture = {
    "words": {w: {"phones": list(g2p._lexicon_word(w, "en")[0]), "guessed": g2p._lexicon_word(w, "en")[1]} for w in words},
    "texts": {t: list(g2p.words_of(t)) for t in texts},
}
rng = random.Random(0)
alphabet = ["a", "b", "c", "d", "e"]
pairs = []
for _ in range(200):
    a = [rng.choice(alphabet) for _ in range(rng.randint(0, 12))]
    b = [rng.choice(alphabet) for _ in range(rng.randint(0, 12))]
    pairs.append({"a": a, "b": b, "distance": Levenshtein.distance(a, b), "opcodes": [list(o) for o in Levenshtein.opcodes(a, b)]})
fixture["levenshtein"] = pairs
# CTC alignment on a random posterior matrix
from openpronounce import alignment  # noqa: E402
r = np.random.default_rng(0)
logits = r.normal(size=(30, 6)).astype(np.float32)
lp = (logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))).astype(np.float32)
labels = [1, 2, 2, 3, 5]
spans, best = alignment.forced_align(lp, labels)
fixture["alignment"] = {"lp": lp.tolist(), "labels": labels, "spans": [list(s) for s in spans], "log_prob": best,
                        "log_likelihood": alignment.log_likelihood(lp, labels)}
# models: random logistic / mlp / trees evaluated by the Python classes
x = r.normal(size=(5, 4))
logistic = {"type": "logistic", "features": list("abcd"), "mean": r.normal(size=4).tolist(), "scale": (r.random(4) + 0.5).tolist(),
            "coef": r.normal(size=4).tolist(), "intercept": 0.3, "phone_bias": {"p": 0.5}}
mlp = {"type": "mlp", "features": list("abcd"), "mean": r.normal(size=4).tolist(), "scale": (r.random(4) + 0.5).tolist(),
       "phones": ["p", "q"], "weights": [r.normal(size=(6, 3)).tolist(), r.normal(size=(3, 1)).tolist()],
       "biases": [r.normal(size=3).tolist(), r.normal(size=1).tolist()]}
trees = {"type": "trees", "features": list("abcd"), "baseline": -0.2, "link": "sigmoid",
         "trees": [[[0, 0.1, 1, 2, 0], [-1, 0, 0, 0, 0.4], [2, -0.3, 3, 4, 0], [-1, 0, 0, 0, -0.7], [-1, 0, 0, 0, 0.2]],
                   [[3, 0.0, 1, 2, 0], [-1, 0, 0, 0, 0.1], [-1, 0, 0, 0, -0.3]]]}
phones = ["p", "q", "z", "p", "q"]
fixture["models"] = {
    "x": x.tolist(), "phones": phones,
    "logistic": {"model": logistic, "out": gop.LogisticModel(logistic).predict(x, phones).tolist()},
    "mlp": {"model": mlp, "out": gop.MLPModel(mlp).predict(x, phones).tolist()},
    "trees": {"model": trees, "out": gop.TreeEnsemble(trees).predict(x).tolist()},
}
with open(os.path.join(here, "python.json"), "w", encoding="utf-8") as f:
    json.dump(fixture, f, ensure_ascii=False)
print("ok")
