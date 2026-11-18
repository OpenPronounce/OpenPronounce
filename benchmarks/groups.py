"""Shipped scorer on the speechocean762 test split, broken down by age group and gender.

    python benchmarks/groups.py            # needs utt_test.npz, see scorer.py --features test

Pearson falls when a group's human scores are bunched together, whatever the model does,
so each line also gives the spread of the human totals and the mean absolute error. The
interval is a 95 % bootstrap over speakers (utterances of one speaker are not independent).
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from speechocean762 import DATASET_DIR  # noqa: E402

TARGETS = ("total", "accuracy", "fluency", "prosodic")
AGE_GROUPS = (("children, 12 and under", 0, 12), ("teenagers, 13 to 17", 13, 17), ("adults", 18, 200))


def _pearson(y, p):
    return float(np.corrcoef(y, p)[0, 1]) if len(y) > 2 and y.std() > 0 and p.std() > 0 else float("nan")


def _interval(y, p, speakers, draws, seed=0):
    rng = np.random.default_rng(seed)
    rows = {s: np.flatnonzero(speakers == s) for s in np.unique(speakers)}
    names = list(rows)
    values = []
    for _ in range(draws):
        index = np.concatenate([rows[s] for s in rng.choice(names, len(names))])
        values.append(_pearson(y[index], p[index]))
    return np.nanquantile(values, [0.025, 0.975])


def speaker_metadata(split="test"):
    """``{speaker: (gender, age)}`` from the parquet file of ``split``."""
    import pyarrow.parquet as pq

    path = os.path.join(DATASET_DIR, "data", f"{split}-00000-of-00001.parquet")
    rows = pq.read_table(path, columns=["speaker", "gender", "age"]).to_pylist()
    return {str(row["speaker"]): (row["gender"], int(row["age"])) for row in rows}


def groups(ages, genders):
    """``[(name, boolean mask)]``: everyone, each age group, each gender, each age group x gender."""
    out = [("all", np.ones(len(ages), dtype=bool))]
    by_age = [(name, (ages >= low) & (ages <= high)) for name, low, high in AGE_GROUPS]
    by_gender = [({"f": "female", "m": "male"}.get(g, g), genders == g) for g in sorted(set(genders.tolist()))]
    out += by_age + by_gender
    out += [(f"{age.split(',')[0]}, {gender}", a & g) for age, a in by_age for gender, g in by_gender]
    return [(name, mask) for name, mask in out if mask.sum() > 2]


def report(lang="en", draws=1000):
    from openpronounce import scorer

    table = dict(np.load(os.path.join(DATASET_DIR, "utt_test.npz"), allow_pickle=True))
    model = scorer.load_scorer(lang)
    predictions = model.predict_matrix(np.column_stack([table[name].astype(float) for name in model.features]))
    speakers = np.array([str(s) for s in table["speaker"].tolist()])
    metadata = speaker_metadata()
    genders = np.array([metadata[s][0] for s in speakers])
    ages = np.array([metadata[s][1] for s in speakers])
    print(f"{'group':26s} {'utts':>5s} {'spk':>4s} {'human sd':>8s} {'total r':>8s} {'95 % interval':>15s} {'MAE':>5s} {'bias':>6s}"
          f" {'acc r':>6s} {'flu r':>6s} {'pro r':>6s}")
    rows = []
    for name, mask in groups(ages, genders):
        y, p = table["total"].astype(float)[mask], predictions["total"][mask]
        low, high = _interval(y, p, speakers[mask], draws)
        others = [_pearson(table[t].astype(float)[mask], predictions[t][mask]) for t in TARGETS[1:]]
        row = {"group": name, "utterances": int(mask.sum()), "speakers": len(set(speakers[mask].tolist())),
               "human_sd": float(y.std()), "pearson": _pearson(y, p), "low": float(low), "high": float(high),
               "mae": float(np.abs(y - p).mean()), "bias": float((p - y).mean()),
               **dict(zip(TARGETS[1:], others))}
        rows.append(row)
        print(f"{name:26s} {row['utterances']:5d} {row['speakers']:4d} {row['human_sd']:8.2f} {row['pearson']:8.3f}"
              f"   [{low:.3f}, {high:.3f}] {row['mae']:5.2f} {row['bias']:+6.2f} " + " ".join(f"{v:6.3f}" for v in others))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lang", default="en")
    parser.add_argument("--draws", type=int, default=1000, help="bootstrap draws (default: %(default)s)")
    args = parser.parse_args(argv)
    report(args.lang, args.draws)
    return 0


if __name__ == "__main__":
    sys.exit(main())
