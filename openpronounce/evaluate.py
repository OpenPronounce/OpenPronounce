"""Check OpenPronounce against your own human ratings.

    openpronounce-evaluate manifest.csv --out results.csv --group-by first_language

``manifest.csv`` has one recording per line: an ``audio`` column (path, relative to the
manifest), a ``score`` column (the rating of your own assessors, on any scale), an
optional ``text`` column (what the speaker had to say; empty for free speech) and any
other column you want a breakdown on (first language, age group, level, microphone).

The recordings are scored one by one into ``results.csv`` (a run that stops resumes where
it was), then the report says how well the scores follow your ratings: Pearson and
Spearman correlations with a 95 % interval, the error once mapped on your scale, the same
per group, and how many recordings were flagged unreliable or of poor quality.
"""

import argparse
import csv
import os
import sys

import numpy as np

RESULT_COLUMNS = ["audio", "mode", "score", "accuracy", "fluency", "prosodic", "reliable", "issues", "error"]


def read_manifest(path, audio_column="audio", text_column="text", score_column="score"):
    """Rows of the manifest as dicts, ``audio`` made absolute and ``score`` parsed (``None`` when empty)."""
    base = os.path.dirname(os.path.abspath(path))
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = [c for c in (audio_column, score_column) if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing column(s) {', '.join(missing)} (found: {', '.join(reader.fieldnames or [])})")
        rows = []
        for line in reader:
            raw = (line.get(score_column) or "").strip().replace(",", ".")
            rows.append({**line, "audio": os.path.normpath(os.path.join(base, line[audio_column].strip())),
                         "text": (line.get(text_column) or "").strip(), "score": float(raw) if raw else None})
    return rows


def assess_file(path, text, lang):
    """One result line for a recording: read speech when ``text`` is given, free speech otherwise."""
    from . import audio, free, speech

    sound = audio.load(path)
    if text:
        from . import scorer

        result = speech.compare_audio_with_text(sound, text, lang=lang, acoustic=False, prosody=False,
                                                transcription=scorer.load_scorer(lang) is None)
        scores = result["scores"] or {"total": result["score"]}
        reliable = result["differences"].get("alignment", {}).get("reliable", True)
        mode = "read"
    else:
        result = free.assess(sound, lang=lang)
        scores, reliable, mode = result["scores"] or {}, result["reliable"], "free"
    return {"mode": mode, "score": scores.get("total"), "accuracy": scores.get("accuracy"), "fluency": scores.get("fluency"),
            "prosodic": scores.get("prosodic"), "reliable": reliable, "issues": " ".join(result["audio_quality"]["issues"])}


def run(rows, out, lang="en", assess=assess_file, log=None):
    """Score every row not yet in ``out`` and return all result lines keyed by audio path."""
    done = {}
    if os.path.exists(out):
        with open(out, newline="", encoding="utf-8") as f:
            done = {line["audio"]: line for line in csv.DictReader(f)}
    with open(out, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        if not done:
            writer.writeheader()
        for k, row in enumerate(rows, 1):
            if row["audio"] in done:
                continue
            try:
                line = {"audio": row["audio"], "error": "", **assess(row["audio"], row["text"], lang)}
            except Exception as e:  # noqa: BLE001 - one unreadable file must not stop the run
                line = {"audio": row["audio"], "error": f"{type(e).__name__}: {e}"}
            writer.writerow({c: line.get(c, "") for c in RESULT_COLUMNS})
            f.flush()
            done[row["audio"]] = {c: "" if line.get(c) is None else str(line.get(c, "")) for c in RESULT_COLUMNS}
            if log:
                log(f"[{k}/{len(rows)}] {os.path.basename(row['audio'])}")
    return done


def _pearson(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and np.std(a) > 0 and np.std(b) > 0 else float("nan")


def _spearman(a, b):
    rank = lambda v: np.argsort(np.argsort(v, kind="stable"), kind="stable").astype(float)  # noqa: E731
    return _pearson(rank(np.asarray(a)), rank(np.asarray(b)))


def agreement(human, predicted, draws=1000, seed=0):
    """How ``predicted`` follows ``human``: correlations, 95 % interval, and the error once mapped on the human scale.

    ``slope`` and ``intercept`` map a score of ours onto your scale (least squares); ``mae``
    is the mean absolute error after that mapping, in the units of your ratings.
    """
    human, predicted = np.asarray(human, dtype=float), np.asarray(predicted, dtype=float)
    out = {"n": len(human), "pearson": _pearson(human, predicted), "spearman": _spearman(human, predicted),
           "low": float("nan"), "high": float("nan"), "slope": float("nan"), "intercept": float("nan"), "mae": float("nan")}
    if len(human) > 2 and np.std(predicted) > 0:
        slope, intercept = np.polyfit(predicted, human, 1)
        out.update(slope=float(slope), intercept=float(intercept),
                   mae=float(np.abs(slope * predicted + intercept - human).mean()))
    if len(human) >= 10:
        rng = np.random.default_rng(seed)
        values = [_pearson(human[i], predicted[i]) for i in (rng.integers(0, len(human), len(human)) for _ in range(draws))]
        out["low"], out["high"] = (float(v) for v in np.nanquantile(values, [0.025, 0.975]))
    return out


def report(rows, results, group_by=()):
    """Text report (list of lines) and the figures behind it."""
    scored, unreliable, poor, failed, unrated = [], 0, 0, 0, 0
    for row in rows:
        line = results.get(row["audio"])
        if line is None or line.get("error") or not line.get("score"):
            failed += 1
            continue
        if row["score"] is None:
            unrated += 1
            continue
        unreliable += line.get("reliable") == "False"
        poor += bool(line.get("issues"))
        scored.append((row, float(line["score"])))
    lines = [f"{len(rows)} recordings: {len(scored)} compared with a rating, {failed} could not be scored, {unrated} without a rating",
             f"flagged unreliable: {unreliable}, poor recording quality: {poor}"]
    figures = {"all": None, "groups": {}}
    if len(scored) < 3:
        lines.append("not enough rated recordings to measure an agreement (3 at least, 100 for a figure you can trust)")
        return lines, figures
    human = [row["score"] for row, _ in scored]
    predicted = [score for _, score in scored]

    def line_of(name, a):
        interval = f"[{a['low']:.2f}, {a['high']:.2f}]" if not np.isnan(a["low"]) else "(too few for an interval)"
        return f"{name:28s} n={a['n']:5d}  Pearson {a['pearson']:.3f} {interval}  Spearman {a['spearman']:.3f}  error {a['mae']:.2f}"

    figures["all"] = agreement(human, predicted)
    lines.append(line_of("all", figures["all"]))
    a = figures["all"]
    lines.append(f"to read our score on your scale: yours = {a['slope']:.4f} x ours + {a['intercept']:.3f}")
    for column in group_by:
        values = sorted({row.get(column, "") for row, _ in scored})
        for value in values:
            pairs = [(row["score"], score) for row, score in scored if row.get(column, "") == value]
            figures["groups"][(column, value)] = agreement([h for h, _ in pairs], [p for _, p in pairs])
            lines.append(line_of(f"{column}={value or '(empty)'}", figures["groups"][(column, value)]))
    if len(scored) < 100:
        lines.append("fewer than 100 rated recordings: read the intervals, not the point values")
    return lines, figures


def main(argv=None):
    parser = argparse.ArgumentParser(prog="openpronounce-evaluate", description=__doc__.split("\n\n")[0],
                                     epilog="Manifest columns: audio, score, optional text, plus any column to group by.")
    parser.add_argument("manifest", help="CSV file, one recording per line")
    parser.add_argument("--out", help="where the scores are written (default: <manifest>.results.csv)")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--audio-column", default="audio")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--score-column", default="score")
    parser.add_argument("--group-by", action="append", default=[], metavar="COLUMN",
                        help="break the report down by this column (repeatable)")
    parser.add_argument("--report-only", action="store_true", help="do not score anything, report on the existing results")
    args = parser.parse_args(argv)
    out = args.out or os.path.splitext(args.manifest)[0] + ".results.csv"
    try:
        rows = read_manifest(args.manifest, args.audio_column, args.text_column, args.score_column)
    except (OSError, ValueError) as e:
        parser.error(str(e))
    unknown = [c for c in args.group_by if rows and c not in rows[0]]
    if unknown:
        parser.error(f"unknown column(s) for --group-by: {', '.join(unknown)}")
    if args.report_only:
        results = {}
        if os.path.exists(out):
            with open(out, newline="", encoding="utf-8") as f:
                results = {line["audio"]: line for line in csv.DictReader(f)}
    else:
        results = run(rows, out, args.lang, log=lambda message: print(message, file=sys.stderr))
    lines, _ = report(rows, results, args.group_by)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
