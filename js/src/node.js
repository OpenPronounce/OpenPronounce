// Node helpers: load the models from disk and build an assessor with onnxruntime-web (WASM).
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { gunzipSync } from "node:zlib";

import { LexiconG2P, parseLexicon } from "./g2p.js";
import { createAssessor, loadModels } from "./index.js";

export function loadModelsFromDir(dir) {
  const json = (name) => JSON.parse(readFileSync(join(dir, name), "utf8"));
  const models = loadModels({ vocab: json("vocab.json"), wordModel: json("word_mdd_en.json"),
    missingModel: json("word_missing_generic.json"), scorer: json("scorer_en.json") });
  const g2p = new LexiconG2P(parseLexicon(gunzipSync(readFileSync(join(dir, "cmudict.txt.gz"))).toString("utf8")));
  return { models, g2p };
}

/** ``createAssessor`` reading every file from disk. */
export async function createNodeAssessor({ ort, modelPath, dataDir, executionProviders = ["wasm"] }) {
  const fetchFn = async (path) => {
    const bytes = readFileSync(path);
    return { url: path, arrayBuffer: async () => bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
      json: async () => JSON.parse(bytes.toString("utf8")) };
  };
  const files = Object.fromEntries(["vocab.json", "word_mdd_en.json", "word_missing_generic.json", "scorer_en.json",
    "cmudict.txt.gz"].map((n) => [n, join(dataDir, n)]));
  return createAssessor({ ort, modelBytes: new Uint8Array(readFileSync(modelPath)), files, executionProviders, fetchFn });
}
