// Compare two reference.jsonl files of parity/reference.py on their common utterances,
// e.g. the same int8 model run by onnxruntime with two graph optimization levels: the
// noise floor of int8 inference, against which the JS/Python gap can be read.
//   node parity/compare.mjs <a/reference.jsonl> <b/reference.jsonl>
import { readFileSync } from "node:fs";

const load = (p) => new Map(readFileSync(p, "utf8").split("\n").filter(Boolean).map((l) => {
  const r = JSON.parse(l);
  return [r.utt, r.result];
}));
const [a, b] = process.argv.slice(2).map(load);
const xs = [];
const ys = [];
let words = 0;
let agree = 0;
for (const [utt, ra] of a) {
  const rb = b.get(utt);
  if (!rb || !ra.scores || !rb.scores) continue;
  xs.push(ra.scores.total);
  ys.push(rb.scores.total);
  const wb = new Map((rb.differences.words ?? []).map((w) => [w.position, w.mispronounced]));
  for (const w of ra.differences.words ?? []) {
    words++;
    if (Boolean(w.mispronounced) === Boolean(wb.get(w.position))) agree++;
  }
}
const n = xs.length;
const d = xs.map((x, i) => Math.abs(x - ys[i]));
const m = (v) => v.reduce((s, x) => s + x, 0) / v.length;
const mx = m(xs);
const my = m(ys);
const r = xs.reduce((s, x, i) => s + (x - mx) * (ys[i] - my), 0)
  / Math.sqrt(xs.reduce((s, x) => s + (x - mx) ** 2, 0) * ys.reduce((s, y) => s + (y - my) ** 2, 0));
console.log(JSON.stringify({ utterances: n, total_mean_abs_diff: m(d), total_within_0_5: d.filter((x) => x <= 0.5).length / n,
  total_max_abs_diff: Math.max(...d), pearson: r, word_flag_agreement: agree / words, words }, null, 1));
