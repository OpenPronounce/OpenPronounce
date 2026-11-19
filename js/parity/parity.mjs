// Parity harness: run the JavaScript pipeline on the speechocean762 utterances written by
// parity/reference.py and compare with the Python library's results.
//
//   node parity/parity.mjs --ref /mnt/t7/openpronounce-browser/js/so762-shipped \
//        --model /mnt/t7/openpronounce-browser/hub/wav2vec2-base-phones-en/onnx/model_int8.onnx [--mode e2e|tensors] [--limit N] [--threads 4]
//
// e2e: the JS side runs the ONNX student itself (onnxruntime-web, WASM backend, under Node).
// tensors: the JS side reads the posteriors and hidden states the Python side computed
// (written for the first utterances with reference.py --dump-tensors): differences then
// come from the port alone.
import { appendFileSync, existsSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

import * as ort from "onnxruntime-web";

import { assessFromPosteriors, parseWav, runPhoneModel } from "../src/index.js";
import { prosodyContours } from "../src/prosody.js";
import { loadModelsFromDir } from "../src/node.js";

const here = dirname(fileURLToPath(import.meta.url));
const { values: args } = parseArgs({ options: {
  ref: { type: "string", default: "/mnt/t7/openpronounce-browser/js/so762-shipped" },
  model: { type: "string", default: "/mnt/t7/openpronounce-browser/hub/wav2vec2-base-phones-en/onnx/model_int8.onnx" },
  data: { type: "string", default: join(here, "..", "models") },
  mode: { type: "string", default: "e2e" },
  limit: { type: "string", default: "0" },
  threads: { type: "string", default: "4" },
  out: { type: "string" },
} });

const lines = readFileSync(join(args.ref, "reference.jsonl"), "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l));
const limit = Number(args.limit);
const refs = limit ? lines.slice(0, limit) : lines;
const { models, g2p } = loadModelsFromDir(args.data);
let session = null;
if (args.mode === "e2e") {
  ort.env.wasm.numThreads = Number(args.threads);
  session = await ort.InferenceSession.create(new Uint8Array(readFileSync(args.model)),
    { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
}
const outPath = args.out ?? join(args.ref, `parity_${args.mode}.jsonl`);
writeFileSync(outPath, "");

const strip = (r) => JSON.stringify({ ...r, prosody: undefined });
const pearson = (a, b) => {
  const n = a.length;
  const ma = a.reduce((s, x) => s + x, 0) / n;
  const mb = b.reduce((s, x) => s + x, 0) / n;
  let sab = 0, saa = 0, sbb = 0;
  for (let i = 0; i < n; i++) {
    sab += (a[i] - ma) * (b[i] - mb);
    saa += (a[i] - ma) ** 2;
    sbb += (b[i] - mb) ** 2;
  }
  return sab / Math.sqrt(saa * sbb);
};

const stats = { n: 0, identical: 0, js: [], py: [], human: [], dims: { total: [], accuracy: [], fluency: [], prosodic: [] },
  words: 0, wordAgree: 0, pyFlags: 0, jsFlags: 0, bothFlags: 0, reliableAgree: 0, nullScores: 0, modelMs: [], scoringMs: [],
  seconds: [], prosody: [] };
let skipped = 0;
for (const ref of refs) {
  const { samples } = parseWav(readFileSync(join(args.ref, "wav", `${ref.utt}.wav`)));
  let outputs;
  const t0 = performance.now();
  if (args.mode === "e2e") {
    outputs = await runPhoneModel(ort, session, samples);
  } else {
    const lpPath = join(args.ref, "tensors", `${ref.utt}.logp.f32`);
    if (!existsSync(lpPath)) {
      skipped++;
      continue;
    }
    const lp = new Float32Array(readFileSync(lpPath).buffer.slice(0));
    const hd = new Float32Array(readFileSync(join(args.ref, "tensors", `${ref.utt}.hidden.f32`)).buffer.slice(0));
    const classes = models.vocab.length;
    const frames = lp.length / classes;
    outputs = { logp: { data: lp, frames, classes }, hidden: { data: hd, frames, dim: hd.length / frames } };
  }
  const t1 = performance.now();
  const js = assessFromPosteriors(samples, ref.text, outputs, { models, g2p });
  const t2 = performance.now();
  const py = ref.result;
  stats.n++;
  stats.modelMs.push(t1 - t0);
  stats.scoringMs.push(t2 - t1);
  stats.seconds.push(samples.length / 16000);
  if (strip(js) === strip(py)) stats.identical++;
  if (js.scores && py.scores) {
    stats.js.push(js.scores.total);
    stats.py.push(py.scores.total);
    stats.human.push(ref.human.total);
    for (const k of Object.keys(stats.dims)) stats.dims[k].push(Math.abs(js.scores[k] - py.scores[k]));
  } else if (js.scores || py.scores) {
    stats.nullScores++;
  }
  const pw = py.differences.words ?? [];
  const jw = new Map((js.differences.words ?? []).map((w) => [w.position, w]));
  for (const w of pw) {
    const other = jw.get(w.position);
    stats.words++;
    const a = Boolean(w.mispronounced);
    const b = Boolean(other?.mispronounced);
    if (a === b) stats.wordAgree++;
    if (a) stats.pyFlags++;
    if (b) stats.jsFlags++;
    if (a && b) stats.bothFlags++;
  }
  if (js.reliable === py.reliable) stats.reliableAgree++;
  if (py.prosody) {
    const c = prosodyContours(samples);
    const f0p = py.prosody.f0;
    const n = Math.min(f0p.length, c.f0.length);
    const st = [];
    let close = 0;
    for (let i = 0; i < n; i++) {
      if (f0p[i] > 0 && c.f0[i] > 0) st.push(Math.abs(12 * Math.log2(c.f0[i] / f0p[i])));
      if (Math.abs(f0p[i] - c.f0[i]) <= 0.01 * f0p[i] / 17) close++;
    }
    st.sort((x, y) => x - y);
    const en = py.prosody.energy;
    let emax = 0;
    for (let i = 0; i < Math.min(en.length, c.energy.length); i++) emax = Math.max(emax, Math.abs(en[i] - c.energy[i]));
    stats.prosody.push({ frames: [f0p.length, c.f0.length], f0_pearson: pearson(f0p.slice(0, n), c.f0.slice(0, n)),
      f0_median_semitones: st[Math.floor(st.length / 2)], energy_max_abs: emax, f0_close: close / n });
  }
  appendFileSync(outPath, JSON.stringify({ utt: ref.utt, js_total: js.scores?.total ?? null, py_total: py.scores?.total ?? null,
    identical: strip(js) === strip(py), model_ms: Math.round(t1 - t0), scoring_ms: Math.round(t2 - t1),
    js_flags: (js.differences.words ?? []).filter((w) => w.mispronounced).map((w) => w.position),
    py_flags: pw.filter((w) => w.mispronounced).map((w) => w.position) }) + "\n");
  if (stats.n % 100 === 0) console.error(`[${stats.n}/${refs.length}]`);
}

const median = (xs) => {
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
};
const diffs = stats.js.map((x, i) => Math.abs(x - stats.py[i]));
const summary = {
  mode: args.mode,
  utterances: stats.n,
  skipped,
  identical_json: stats.identical,
  scored_both: stats.js.length,
  scored_one_side_only: stats.nullScores,
  total_mean_abs_diff: diffs.reduce((a, b) => a + b, 0) / diffs.length,
  total_within_0_5: diffs.filter((d) => d <= 0.5).length / diffs.length,
  total_max_abs_diff: Math.max(...diffs),
  total_pearson_js_python: pearson(stats.js, stats.py),
  total_pearson_human: { js: pearson(stats.js, stats.human), python: pearson(stats.py, stats.human) },
  dims_mean_abs_diff: Object.fromEntries(Object.entries(stats.dims).map(([k, v]) => [k, v.reduce((a, b) => a + b, 0) / v.length])),
  word_flag_agreement: stats.wordAgree / stats.words,
  words: stats.words,
  flags: { python: stats.pyFlags, js: stats.jsFlags, both: stats.bothFlags },
  reliable_agreement: stats.reliableAgree / stats.n,
  median_ms: { model: median(stats.modelMs), scoring: median(stats.scoringMs) },
  median_clip_seconds: median(stats.seconds),
  model_ms_per_audio_second: stats.modelMs.reduce((a, b) => a + b, 0) / stats.seconds.reduce((a, b) => a + b, 0),
};
if (stats.prosody.length) {
  summary.prosody_contour_vs_pyin = {
    utterances: stats.prosody.length,
    f0_pearson_median: median(stats.prosody.map((p) => p.f0_pearson)),
    f0_pearson_min: Math.min(...stats.prosody.map((p) => p.f0_pearson)),
    f0_frames_within_1_cent: stats.prosody.reduce((a, p) => a + p.f0_close, 0) / stats.prosody.length,
    f0_abs_semitones_median: median(stats.prosody.map((p) => p.f0_median_semitones)),
    same_frame_count: stats.prosody.filter((p) => p.frames[0] === p.frames[1]).length,
    energy_max_abs_diff: Math.max(...stats.prosody.map((p) => p.energy_max_abs)),
  };
}
console.log(JSON.stringify(summary, null, 1));
writeFileSync(outPath.replace(/\.jsonl$/, ".summary.json"), JSON.stringify(summary, null, 1));
