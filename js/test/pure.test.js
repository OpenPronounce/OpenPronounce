// Pure functions against the Python library: fixtures written by test/fixtures/make_fixtures.py.
import assert from "node:assert/strict";
import { readFileSync, existsSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { test } from "node:test";

import { fits, forcedAlign, logLikelihood } from "../src/alignment.js";
import { arpabetToPhones, LexiconG2P, numberWords, parseLexicon, wordsOf } from "../src/g2p.js";
import { distance, opcodes } from "../src/levenshtein.js";
import { LogisticModel, MLPModel, TreeEnsemble } from "../src/models.js";
import { npMean, npQuantile, npStd, npSum, pyRound } from "../src/numeric.js";
import { decodeCtc, normalizePhone, normalizePhones } from "../src/phones.js";

const fixture = JSON.parse(readFileSync(new URL("./fixtures/python.json", import.meta.url), "utf8"));
const close = (a, b, tol = 1e-9) => assert.ok(Math.abs(a - b) <= tol * Math.max(1, Math.abs(b)), `${a} != ${b}`);
const matrix = (rows) => ({ data: Float32Array.from(rows.flat()), frames: rows.length, classes: rows[0].length });

test("forced alignment and CTC likelihood match the Python implementation", () => {
  const { lp, labels, spans, log_prob: logProb, log_likelihood: ll } = fixture.alignment;
  const result = forcedAlign(matrix(lp), labels);
  assert.deepEqual(result.spans, spans);
  close(result.logProb, logProb);
  close(logLikelihood(matrix(lp), labels), ll);
});

test("CTC likelihood is the sum over every path (brute force on a tiny case)", () => {
  const lp = [[-0.2, -1.9, -2.5], [-1.1, -0.7, -1.6], [-0.9, -1.4, -1.0]];
  const labels = [1, 2];
  // every path of length 3 that collapses to [1, 2]
  let total = 0;
  for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) for (let c = 0; c < 3; c++) {
    const path = [a, b, c];
    const collapsed = path.filter((x, i) => x !== 0 && (i === 0 || x !== path[i - 1]));
    if (collapsed.join() === labels.join()) total += Math.exp(lp[0][a] + lp[1][b] + lp[2][c]);
  }
  close(logLikelihood(matrix(lp), labels), Math.log(total), 1e-6);
});

test("fits needs a blank between repeated labels", () => {
  assert.equal(fits(3, [1, 1]), true);
  assert.equal(fits(2, [1, 1]), false);
  assert.equal(fits(2, [1, 2]), true);
});

test("Levenshtein distance and opcodes break ties like rapidfuzz", () => {
  for (const { a, b, distance: d, opcodes: ops } of fixture.levenshtein) {
    assert.equal(distance(a, b), d);
    assert.deepEqual(opcodes(a, b), ops);
  }
  const long = Array.from({ length: 70 }, (_, i) => String.fromCharCode(97 + (i % 7)));
  assert.equal(distance(long, long.slice(3)), 3);
});

test("logistic, MLP and tree models reproduce the Python predictions", () => {
  const { x, phones } = fixture.models;
  const cases = [[LogisticModel, "logistic", true], [MLPModel, "mlp", true], [TreeEnsemble, "trees", false]];
  for (const [Model, name, withPhones] of cases) {
    const { model, out } = fixture.models[name];
    const got = new Model(model).predict(x, withPhones ? phones : null);
    got.forEach((v, i) => close(v, out[i], 1e-12));
  }
});

test("ARPAbet mapping: r-coloured vowels, syllabic l, flaps, final unstressed IY", () => {
  assert.deepEqual(arpabetToPhones(["B", "AH1", "T", "ER0"]), ["b", "ʌ", "ɾ", "ɚ"]);
  assert.deepEqual(arpabetToPhones(["L", "IH1", "T", "AH0", "L"]), ["l", "ɪ", "ɾ", "əl"]);
  assert.deepEqual(arpabetToPhones(["K", "AA1", "R"]), ["k", "ɑːɹ"]);
  assert.deepEqual(arpabetToPhones(["HH", "AE1", "P", "IY0"]), ["h", "æ", "p", "i"]);
  assert.deepEqual(numberWords(1999), ["one", "thousand", "nine", "hundred", "ninety", "nine"]);
});

test("words of a text follow Python's \\b[\\w']+\\b", () => {
  for (const [text, words] of Object.entries(fixture.texts)) assert.deepEqual(wordsOf(text), words);
});

const lexiconPath = new URL("../models/cmudict.txt.gz", import.meta.url);
test("CMUdict G2P gives the Python phones, derived forms and guesses included", { skip: !existsSync(lexiconPath) && "run npm run build:data" }, () => {
  const g2p = new LexiconG2P(parseLexicon(gunzipSync(readFileSync(lexiconPath)).toString("utf8")));
  for (const [word, { phones, guessed }] of Object.entries(fixture.words)) {
    assert.deepEqual(g2p.word(word), { phones, guessed }, word);
  }
});

test("phone normalization and greedy CTC decoding", () => {
  assert.equal(normalizePhone("ɔː"), "ɑ");
  assert.equal(normalizePhone("tʰ"), "t");
  assert.deepEqual(normalizePhones(["ɚ", "ɹ", "d", "d", "ɾ"]), ["ɚ", "d", "t"]);
  const vocab = ["<pad>", "a", "b"];
  const lp = matrix([[0, -9, -9], [-9, 0, -9], [-9, -0.1, -9], [0, -9, -9], [-9, -9, 0]]);
  const rec = decodeCtc(lp, vocab);
  assert.deepEqual(rec.phones, ["a", "b"]);
  assert.deepEqual(rec.spans, [[1, 3], [4, 5]]);
});

test("numpy-compatible reductions and Python rounding", () => {
  const xs = Array.from({ length: 300 }, (_, i) => Math.sin(i) * 0.02);
  // pairwise sum differs from a plain loop in the last bits, never by more
  close(npSum(xs), xs.reduce((a, b) => a + b, 0), 1e-12);
  assert.equal(npStd([0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04]), 0);
  close(npMean([1, 2, 3, 4]), 2.5);
  assert.equal(npQuantile([1, 2, 3, 4, 5], 0.1), 1.4);
  assert.equal(pyRound(0.125, 2), 0.12);
  assert.equal(pyRound(0.375, 2), 0.38);
  assert.equal(pyRound(2.675, 2), 2.67);
  assert.equal(pyRound(86.45000000000002, 1), 86.5);
});
