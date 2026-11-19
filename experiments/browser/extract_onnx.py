"""Cache the phone posteriors and layer-20 hidden states of an ONNX phone model, in the
format of benchmarks/word_model.py --extract, under /mnt/t7/openpronounce-browser/so762/<variant>/.

    .venv-exp/bin/python experiments/browser/extract_onnx.py --model phone.int8.onnx --variant int8 --split test
"""

import argparse
import logging
import os
import time

import numpy as np

import common

logger = logging.getLogger("extract_onnx")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="file name in the onnx dir, or a path")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    path = args.model if os.path.exists(args.model) else os.path.join(common.ONNX_DIR, args.model)
    sess = common.session(path, args.threads)
    out = common.prepare_logits_dir(args.variant, args.split)
    rows = common.rows(args.split)
    t0, done = time.time(), 0
    for k, row in enumerate(rows, 1):
        p, h = os.path.join(out, row["utt"] + ".npy"), os.path.join(out, row["utt"] + ".hidden.npy")
        if os.path.exists(p) and os.path.exists(h):
            continue
        logits, hidden = common.run(sess, common.sound(row))[:2]
        np.save(p, common.log_softmax(logits).astype(np.float16))
        np.save(h, hidden.astype(np.float16))
        done += 1
        if k % 100 == 0:
            logger.info("[%d/%d] %.0fs", k, len(rows), time.time() - t0)
    logger.info("%d utterances in %.0fs", done, time.time() - t0)


if __name__ == "__main__":
    main()
