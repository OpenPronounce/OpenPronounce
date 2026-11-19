"""Time ONNX models with onnxruntime (python, CPU) on 5 s and 18 s of noise, 1 and 4 threads.

    .venv-exp/bin/python experiments/browser/bench_cpu.py phone.int8.onnx phone.fp32.onnx ...
Median of 3 runs after one warm-up run.
"""

import json
import os
import sys
import time

import numpy as np

import common

for name in sys.argv[1:]:
    path = name if os.path.exists(name) else os.path.join(common.ONNX_DIR, name)
    for threads in (1, 4):
        sess = common.session(path, threads)
        for seconds in (5, 18):
            wave = np.random.default_rng(0).uniform(-1, 1, seconds * 16000).astype(np.float32)
            common.run(sess, wave)
            times = []
            for _ in range(3):
                t = time.perf_counter()
                common.run(sess, wave)
                times.append(time.perf_counter() - t)
            print(json.dumps({"model": name, "threads": threads, "seconds": seconds,
                              "median_s": round(float(np.median(times)), 3), "min_s": round(min(times), 3)}), flush=True)
        del sess
