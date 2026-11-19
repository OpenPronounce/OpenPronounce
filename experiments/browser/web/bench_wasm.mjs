// Time an ONNX model with onnxruntime-web (WASM backend) under Node, the CPU path of a browser.
//   node bench_wasm.mjs /mnt/t7/openpronounce-browser/onnx/phone.int8.onnx [threads=1] [runs=3]
// Input: noise of 5 s and 18 s at 16 kHz (inference time does not depend on the content).
import { readFileSync } from "node:fs";
import * as ort from "onnxruntime-web";

const [path, threads = "1", runs = "3"] = process.argv.slice(2);
ort.env.wasm.numThreads = Number(threads);
ort.env.wasm.simd = true;
const t0 = performance.now();
const bytes = readFileSync(path);
const session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
console.log(JSON.stringify({ model: path, threads: Number(threads), load_ms: Math.round(performance.now() - t0),
  rss_mb: Math.round(process.memoryUsage().rss / 1e6) }));
for (const seconds of [5, 18]) {
  const n = seconds * 16000;
  const data = new Float32Array(n);
  for (let i = 0; i < n; i++) data[i] = (Math.random() * 2 - 1);
  const feeds = { input_values: new ort.Tensor("float32", data, [1, n]) };
  const times = [];
  for (let r = 0; r < Number(runs) + 1; r++) {
    const t = performance.now();
    await session.run(feeds);
    if (r > 0) times.push(performance.now() - t); // first run is warm-up
  }
  times.sort((a, b) => a - b);
  console.log(JSON.stringify({ seconds, threads: Number(threads), median_ms: Math.round(times[Math.floor(times.length / 2)]),
    min_ms: Math.round(times[0]), rss_mb: Math.round(process.memoryUsage().rss / 1e6) }));
}
