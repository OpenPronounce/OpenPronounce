"""Decode the distillation audio once to 16 kHz int16, one flat file plus an index.

Sources: the Common Voice 27 en clips present on /mnt/t7 (selection.tsv, ~30k clips, native and
non-native accents) and the speechocean762 *train* split (the test split is never used).

    .venv-exp/bin/python experiments/browser/predecode.py
Writes /mnt/t7/openpronounce-browser/distill_audio/{audio.int16, index.json}.
"""

import csv
import io
import json
import os
from multiprocessing import Pool

import numpy as np

CV = "/mnt/t7/datasets/common-voice/cv-corpus-27.0-2026-09-11/en"
OUT = "/mnt/t7/openpronounce-browser/distill_audio"
MAX_SECONDS = 15.0


def decode(path_or_bytes):
    import librosa
    import soundfile as sf

    try:
        data, sr = sf.read(path_or_bytes if isinstance(path_or_bytes, str) else io.BytesIO(path_or_bytes),
                           dtype="float32", always_2d=True)
        data = data.mean(axis=1)
        if sr != 16000:
            data = librosa.resample(data, orig_sr=sr, target_sr=16000)
        data, _ = librosa.effects.trim(data, top_db=40)
        data = data[: int(MAX_SECONDS * 16000)]
        if len(data) < 16000:
            return None
        return (np.clip(data, -1, 1) * 32767).astype(np.int16)
    except Exception:
        return None


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(CV, "selection.tsv")) as f:
        cv = [os.path.join(CV, "clips", r["path"]) for r in csv.DictReader(f, delimiter="\t")]
    import pyarrow.parquet as pq
    so = pq.read_table(os.path.expanduser("~/.cache/openpronounce/speechocean762/data/train-00000-of-00001.parquet")).to_pylist()
    items = [("so762:" + os.path.splitext(r["audio"]["path"])[0], r["audio"]["bytes"]) for r in so]
    items += [("cv:" + os.path.basename(p), p) for p in cv]
    index, offset = [], 0
    with open(os.path.join(OUT, "audio.int16"), "wb") as fout, Pool(6) as pool:
        for (name, _), wave in zip(items, pool.imap(decode, [src for _, src in items], chunksize=16)):
            if wave is None:
                continue
            fout.write(wave.tobytes())
            index.append({"name": name, "offset": offset, "length": len(wave)})
            offset += len(wave)
            if len(index) % 2000 == 0:
                print(len(index), offset / 16000 / 3600, "h", flush=True)
    with open(os.path.join(OUT, "index.json"), "w") as f:
        json.dump(index, f)
    print(f"{len(index)} clips, {offset / 16000 / 3600:.1f} h")


if __name__ == "__main__":
    main()
