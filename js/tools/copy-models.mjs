// Copy the JSON models the browser reads into models/:
//   node tools/copy-models.mjs [scorer dir] [vocab.json]
// The word model and the scorer are the ones the Python package ships for the student
// (openpronounce/models/word_mdd_en_student.json, scorer_en_student.json), written here as
// word_mdd_en.json and scorer_en.json. A scorer dir given as first argument replaces them with
// its own word_mdd_en.json / scorer_en.json. The missing-word model does not depend on the
// acoustic model; the vocabulary is the phone model's (a list or a token -> id map), by
// default the copy cached next to the speechocean762 logits. JSON is re-serialized without
// indentation.
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const repo = join(here, "..", "..");
const shipped = join(repo, "openpronounce", "models");
const scorerDir = process.argv[2] || undefined;
const vocabPath = process.argv[3] ?? join(process.env.HOME, ".cache/openpronounce/speechocean762/logits_test/vocab.json");
const out = join(here, "..", "models");
mkdirSync(out, { recursive: true });
const sources = {
  "word_mdd_en.json": scorerDir ? join(scorerDir, "word_mdd_en.json") : join(shipped, "word_mdd_en_student.json"),
  "scorer_en.json": scorerDir ? join(scorerDir, "scorer_en.json") : join(shipped, "scorer_en_student.json"),
  "word_missing_generic.json": join(shipped, "word_missing_generic.json"),
  "vocab.json": vocabPath,
};
for (const [name, path] of Object.entries(sources)) {
  if (!existsSync(path)) throw new Error(`missing ${path}`);
  let data = JSON.parse(readFileSync(path, "utf8"));
  if (name === "vocab.json" && !Array.isArray(data)) {
    data = Object.entries(data).sort((a, b) => a[1] - b[1]).map(([token]) => token);
  }
  writeFileSync(join(out, name), JSON.stringify(data));
  console.log(`${name} <- ${path}`);
}
