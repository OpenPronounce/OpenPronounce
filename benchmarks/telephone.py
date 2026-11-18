"""speechocean762 through a telephone channel: what 8 kHz costs.

    python benchmarks/telephone.py --sample 500      # resumable, about an hour on CPU
    python benchmarks/telephone.py --report

Three conditions per utterance: ``orig`` (16 kHz), ``8k`` (resampled to 8 kHz and back) and
``phone`` (8 kHz, 300-3400 Hz band, G.711 mu-law). Each is scored with the sentence (fast
path) and as free speech. The channel is simulated on clean recordings: no line noise, no
mobile codec, no packet loss.
"""

import argparse
import io
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from speechocean762 import DATASET_DIR  # noqa: E402

OUT = os.path.join(DATASET_DIR, "telephone.jsonl")
CONDITIONS = ("orig", "8k", "phone")


def degrade(wave, condition):
    import librosa
    from scipy.signal import butter, sosfiltfilt

    if condition == "orig":
        return wave
    low = librosa.resample(wave, orig_sr=16000, target_sr=8000, res_type="soxr_hq")
    if condition == "phone":
        low = sosfiltfilt(butter(4, [300, 3400], btype="band", fs=8000, output="sos"), low)
        peak = max(1e-9, float(np.abs(low).max()))
        x = np.clip(low / peak, -1, 1)
        coded = np.round((np.sign(x) * np.log1p(255 * np.abs(x)) / np.log1p(255) + 1) / 2 * 255) / 255 * 2 - 1
        low = np.sign(coded) * ((1 + 255) ** np.abs(coded) - 1) / 255 * peak
    return librosa.resample(low.astype(np.float32), orig_sr=8000, target_sr=16000, res_type="soxr_hq").astype(np.float32)


def run(n, threads):
    import torch

    torch.set_num_threads(threads)
    import speechocean762 as so
    from groups import speaker_metadata
    from openpronounce import assess, audio

    meta = speaker_metadata()
    done = {json.loads(line)["utt"] for line in open(OUT)} if os.path.exists(OUT) else set()
    rows = [r for r in so.sample_rows(so.load_rows("test"), n) if r["utt"] not in done]
    print(len(rows), "to run", flush=True)
    with open(OUT, "a") as f:
        for k, row in enumerate(rows, 1):
            wave = audio.load(io.BytesIO(row["audio"]["bytes"]))
            text = so.normalize_text(row["text"])
            out = {"utt": row["utt"], "speaker": str(row["speaker"]), "age": meta[str(row["speaker"])][1],
                   "human": {t: row[t] for t in ("total", "accuracy", "fluency", "prosodic")}, "words": len(text.split())}
            for condition in CONDITIONS:
                piece = degrade(wave, condition)
                read = assess(piece, text, lang="en", transcription=False, acoustic=False, prosody=False)
                free = assess(piece, lang="en")
                out[condition] = {"read": read["scores"], "flagged": len(read["differences"]["errors"]),
                                  "reliable": read["reliable"], "free": free["scores"], "transcribe": free["transcribe"]}
            f.write(json.dumps(out) + "\n")
            f.flush()
            if k % 25 == 0:
                print(k, flush=True)


def report():
    rows = [json.loads(line) for line in open(OUT)]
    human = np.array([r["human"]["total"] for r in rows], float)
    adult = np.array([r["age"] >= 18 for r in rows])
    print(len(rows), "utterances,", int(adult.sum()), "from adults")
    for mode in ("read", "free"):
        base = np.array([(r["orig"][mode] or {}).get("total", np.nan) for r in rows], float)
        for condition in CONDITIONS:
            s = np.array([(r[condition][mode] or {}).get("total", np.nan) for r in rows], float)
            ok = ~np.isnan(s) & ~np.isnan(base)
            line = (f"{mode:5s} {condition:6s} scored {int((~np.isnan(s)).sum()):4d}  mean {np.nanmean(s):5.1f}  Pearson with raters {np.corrcoef(human[ok], s[ok])[0, 1]:.3f} "
                    f"(adults {np.corrcoef(human[ok & adult], s[ok & adult])[0, 1]:.3f})  with the 16 kHz score {np.corrcoef(base[ok], s[ok])[0, 1]:.3f}  "
                    f"mean change {np.mean(s[ok] - base[ok]):+5.1f}  |change| {np.mean(np.abs(s[ok] - base[ok])):4.1f} (p90 {np.percentile(np.abs(s[ok] - base[ok]), 90):4.1f})")
            if mode == "read":
                line += f"  words reported {sum(r[condition]['flagged'] for r in rows) / sum(r['words'] for r in rows):.3f}"
            print(line)


def main(argv=None):
    global OUT
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", type=int, default=500, help="utterances of the test split (default: %(default)s)")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--out", default=OUT, help="where the scores are kept (default: %(default)s)")
    args = parser.parse_args(argv)
    OUT = args.out
    report() if args.report else run(args.sample, args.threads)


if __name__ == "__main__":
    main()
